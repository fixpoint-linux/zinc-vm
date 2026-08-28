//! src/vm/marshal.zig — C Value <-> Shen tagged-form marshalling (milestone M2).
//!
//! C origin: zincvm.c:742-838 — marshal_to_tagged (:753-788) and
//! demarshal_from_tagged (:792-838), ported VERBATIM including every root
//! count: the cons case of marshal roots 5 slots (C:768-778), the [cons X Y]
//! case of demarshal roots 7 slots (C:812-834).
//!
//! Tagged forms (from interp.shen extract-kl):
//!   [number X]  = cons(symbol("number"), cons(X, nil))
//!   [symbol X]  = cons(symbol("symbol"), cons(X, nil))
//!   [string X]  = cons(symbol("string"), cons(X, nil))
//!   [boolean X] = cons(symbol("boolean"), cons(X, nil))
//!   [cons X Y]  = cons(symbol("cons"), cons(X', cons(Y', nil))) — a 3-element
//!                 list whose car/cdr ride RAW (NOT recursively marshaled):
//!                 extract-kl handles its own recursion on [cons X Y] by
//!                 calling itself on X and Y directly (C:763-767).  Recursive
//!                 marshalling would create impossibly deep nesting the
//!                 compiled interp patterns cannot match.
//!   [cons]      = cons(symbol("cons"), nil) — the empty list
//!   mark        = symbol("mark")
//! Unmarshallable types (lambdas, prims, errors, vectors, streams) pass
//! through unchanged.
//!
//! demarshal is the inverse: [tag X] unwraps the cadr (C:807-810), [cons]
//! (nil cdr) demarshals to nil, [cons X Y] rebuilds cons(demarshal X,
//! demarshal Y) with the actual cdr riding in the singleton wrapper
//! ((Y) nil) that marshal built for it, and the SYMBOL 'mark' demarshals to
//! nil (C:796).  A cons whose car is an unknown symbol tag passes through
//! unchanged (C:837) — so a tree is a fixed point of demarshal∘marshal
//! exactly when no cons car along it is one of the five reserved tag symbols
//! and 'mark' appears nowhere (the eval-kl chain relies on this: forms the
//! Shen compiler builds never use those heads raw).
//!
//! The `vm` parameter replaces C's globals: val_symbol interns through
//! vm.symbols, allocations through vm.gc.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const state = @import("state.zig");
const values = @import("values.zig");
const symbols = @import("symbols.zig");

const Gc = gc.Gc;
const Value = types.Value;
const Vm = state.Vm;

/// C: zincvm.c:753-788 marshal_to_tagged — C Value → Shen tagged form.
pub fn marshalToTagged(vm: *Vm, v: Value) Value {
    const g = vm.gc;
    switch (v.tag) {
        .number => return values.valCons(
            g,
            symbols.valSymbol(&vm.symbols, "number"),
            values.valCons(g, v, values.valNil()),
        ),
        .symbol => return values.valCons(
            g,
            symbols.valSymbol(&vm.symbols, "symbol"),
            values.valCons(g, v, values.valNil()),
        ),
        .string => return values.valCons(
            g,
            symbols.valSymbol(&vm.symbols, "string"),
            values.valCons(g, v, values.valNil()),
        ),
        .boolean => return values.valCons(
            g,
            symbols.valSymbol(&vm.symbols, "boolean"),
            values.valCons(g, v, values.valNil()),
        ),
        // C:763-780 — DON'T recursively marshal car/cdr (module doc).  The
        // 5-root dance is VERBATIM: v / car_val / cdr_val / inner / middle
        // (C:768-778); `result` is consumed before any further allocation, so
        // it needs no root of its own.
        .cons => {
            var vv = v;
            g.rootPushValue(&vv); // root v across nested val_cons allocs — C:768
            var car_val = vv.payload.cons.car.?.*;
            var cdr_val = vv.payload.cons.cdr.?.*;
            g.rootPushValue(&car_val); // C:771
            g.rootPushValue(&cdr_val); // C:772
            var inner = values.valCons(g, cdr_val, values.valNil());
            g.rootPushValue(&inner); // C:774
            var middle = values.valCons(g, car_val, inner);
            g.rootPushValue(&middle); // C:776
            const result = values.valCons(g, symbols.valSymbol(&vm.symbols, "cons"), middle);
            g.rootPop(); // middle — C:778 (x5)
            g.rootPop(); // inner
            g.rootPop(); // cdr_val
            g.rootPop(); // car_val
            g.rootPop(); // v
            return result;
        },
        .nil => return values.valCons(g, symbols.valSymbol(&vm.symbols, "cons"), values.valNil()),
        .mark => return symbols.valSymbol(&vm.symbols, "mark"),
        // Lambdas, prims, errors, vectors, streams — C:785-786.
        else => return v,
    }
}

/// C: zincvm.c:792-838 demarshal_from_tagged — Shen tagged form → C Value.
/// Non-tagged atoms pass through.
pub fn demarshalFromTagged(vm: *Vm, tagged_in: Value) Value {
    const g = vm.gc;
    var tagged = tagged_in;
    switch (tagged.tag) {
        .number, .string, .boolean => return tagged, // C:793-794
        .symbol => {
            // C:795-798 — 'mark' demarshals to nil.
            if (std.mem.eql(u8, values.symSlice(tagged), "mark")) return values.valNil();
            return tagged;
        },
        .cons => {}, // fall through to the tag checks below
        else => return tagged, // C:799 (nil, lambda, prim, error, vector, stream, mark)
    }

    // Check for tagged form: car is a symbol tag — C:801-803.
    const car = tagged.payload.cons.car.?.*;
    if (car.tag != .symbol) return tagged;
    const tag = values.symSlice(car);

    if (std.mem.eql(u8, tag, "number") or std.mem.eql(u8, tag, "symbol") or
        std.mem.eql(u8, tag, "string") or std.mem.eql(u8, tag, "boolean"))
    {
        // [tag X] — extract the value: cadr of the tagged form — C:807-810.
        const cdr = tagged.payload.cons.cdr.?.*;
        return cdr.payload.cons.car.?.*;
    }
    if (std.mem.eql(u8, tag, "cons")) {
        // C:811-835 — the 7-root dance, VERBATIM: tagged / cdr / tagged_car /
        // tagged_cdr / actual_cdr / r1 / r2 (C:812-834).
        g.rootPushValue(&tagged); // root tagged across recursive call allocs — C:812
        var cdr = tagged.payload.cons.cdr.?.*;
        g.rootPushValue(&cdr); // C:814
        if (cdr.tag == .nil) {
            g.rootPop(); // cdr — C:815 (x2)
            g.rootPop(); // tagged
            return values.valNil(); // [cons] — empty list
        }
        // [cons X Y] — recursively demarshal car and cdr.  The actual cdr
        // rides in the singleton wrapper marshal built: the tagged form is
        // (cons X (Y nil)), so tagged_cdr is (Y nil) and actual_cdr its car.
        var tagged_car = cdr.payload.cons.car.?.*;
        var tagged_cdr = cdr.payload.cons.cdr.?.*;
        var actual_cdr = tagged_cdr.payload.cons.car.?.*;
        g.rootPushValue(&tagged_car); // C:820
        g.rootPushValue(&tagged_cdr); // C:821
        g.rootPushValue(&actual_cdr); // C:822
        var r1 = demarshalFromTagged(vm, tagged_car);
        g.rootPushValue(&r1); // root r1 across demarshal of actual_cdr — C:824
        var r2 = demarshalFromTagged(vm, actual_cdr);
        g.rootPushValue(&r2); // root r2 across val_cons alloc — C:826
        const out = values.valCons(g, r1, r2);
        g.rootPop(); // r2 — C:828 (x7)
        g.rootPop(); // r1
        g.rootPop(); // actual_cdr
        g.rootPop(); // tagged_cdr
        g.rootPop(); // tagged_car
        g.rootPop(); // cdr
        g.rootPop(); // tagged
        return out;
    }
    return tagged; // unknown tag — C:837
}
