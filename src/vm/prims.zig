//! src/vm/prims.zig — the C primitives: table + dispatch + the pure subset
//! handlers (milestone M5).
//!
//! C origin: zincvm.c:1780-2794 (exec_primitive, PURE cases only),
//! zincvm.c:957-974 (prim_names[] / exec_primitive_valid — the prims.def
//! X-macro becomes the comptime `prim_table` below), zincvm.c:3752-3758
//! (init_globals, driven from state.zig), and the trap-error case
//! (zincvm.c:2600-2651) with the DECISION-A CatchSite chain.
//!
//! SCOPE (plan PRIMS deliverable — the pure subset of the ~70 prims.def
//! entries): the INCLUDE list is the plan's exact list (hot list ops,
//! arithmetic/comparison, predicates, strings+chars, vectors, control,
//! tuples/symbols/misc), plus the M6 stream I/O prims (write-byte/read-byte/
//! read-file-as-string/open/close).  DEFERRED to later milestones: eval-kl
//! (bundle/metacircular milestone).  OMITTED: the process subsystem
//! (exec-plan/wait/kill/cd/getcwd/getpid/getenv/setenv/glob) and the dead
//! dispatch cases (length/nth/fail/stinput/stoutput — namespace-2 OS defuns,
//! never reachable via a prims.def name).
//!
//! ROOTING (plan exec_primitive ROOTING observation, ported VERBATIM — the C
//! audit note at zincvm.c:1772-1779 applies unchanged): every popped Value
//! whose interior pointers are read across an allocating call must be rooted
//! (rootPushValue) or copied through valStringFrom's slot-rooting; the
//! per-prim discipline is annotated at each handler.  Alloc-free prims take
//! no roots.  Write barriers fire on every store of a possibly-nursery Value
//! into a possibly-oldgen Value array (address-> via gc.writeBarrierVectorStore).
//!
//! PORT-FIX (plan-mandated deviation from a latent C bug): error-to-string
//! copies the message through the slot-rooted valStringFromErr pattern
//! (values.zig) instead of C's raw `a.error.message` pointer pass into
//! val_string, whose GC_STR can move the message and leave the memcpy stale
//! (zincvm.c:1973).  See primErrorToString.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const state = @import("state.zig");
const values = @import("values.zig");
const symbols = @import("symbols.zig");
const interp = @import("interp.zig");
const streams = @import("streams.zig");
const execplan = @import("execplan.zig");
const marshal = @import("marshal.zig");
const hostcall = @import("hostcall.zig");

const Gc = gc.Gc;
const Value = types.Value;
const ValueArray = types.ValueArray;
const Vm = state.Vm;
const VmError = state.VmError;

/// wait/kill libc externs (the Shen OS wait/kill prims, ported here because
/// execplan.zig's copies are file-private and out of scope this phase).
/// Native-only: the vm module links libc.
extern "c" fn waitpid(pid: c_int, status: ?*c_int, options: c_int) c_int;
extern "c" fn kill(pid: c_int, sig: c_int) c_int;

/// The handler shape shared by the table and the dispatcher.
pub const PrimFn = *const fn (vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void;

/// One table entry — the Zig translation of a prims.def `PRIM(n, a)` line.
/// `arity` is informational (handlers pop what they need, exactly like C);
/// it documents the Shen-side arity that zinc-c compiles calls against.
pub const PrimDef = struct {
    name: [:0]const u8,
    arity: u8,
    func: PrimFn,
};

/// C: zincvm.c:957-965 prim_names[] (the prims.def X-macro), single source of
/// truth driving: dispatch (prim_map), initGlobals (state.zig), isValid (the
/// defun_get prim fallback), and the bundle loader's primitive?-names list.
/// Order mirrors prims.def (hot-first) minus the deferred/omitted entries.
pub const prim_table = [_]PrimDef{
    // ---- hot list ops + interp-internal hot prims (prims.def order) ----
    .{ .name = "assoc", .arity = 2, .func = primAssoc },
    .{ .name = "cons", .arity = 2, .func = primCons },
    .{ .name = "hd", .arity = 1, .func = primHd },
    .{ .name = "tl", .arity = 1, .func = primTl },
    .{ .name = "=", .arity = 2, .func = primEq },
    .{ .name = "empty?", .arity = 1, .func = primEmptyP },
    .{ .name = "reverse", .arity = 1, .func = primReverse },
    .{ .name = "append", .arity = 2, .func = primAppend },
    // ---- arithmetic + comparison ----
    .{ .name = "+", .arity = 2, .func = primAdd },
    .{ .name = "/", .arity = 2, .func = primDiv },
    .{ .name = "f/", .arity = 2, .func = primFdiv },
    .{ .name = "*", .arity = 2, .func = primMul },
    .{ .name = "-", .arity = 2, .func = primSub },
    .{ .name = ">", .arity = 2, .func = primGt },
    .{ .name = "<", .arity = 2, .func = primLt },
    .{ .name = ">=", .arity = 2, .func = primGe },
    .{ .name = "<=", .arity = 2, .func = primLe },
    // ---- predicates ----
    .{ .name = "number?", .arity = 1, .func = primNumberP },
    .{ .name = "string?", .arity = 1, .func = primStringP },
    .{ .name = "symbol?", .arity = 1, .func = primSymbolP },
    .{ .name = "boolean?", .arity = 1, .func = primBooleanP },
    .{ .name = "cons?", .arity = 1, .func = primConsP },
    .{ .name = "absvector?", .arity = 1, .func = primAbsvectorP },
    .{ .name = "function?", .arity = 1, .func = primFunctionP },
    .{ .name = "error?", .arity = 1, .func = primErrorP },
    .{ .name = "stream?", .arity = 1, .func = primStreamP },
    .{ .name = "variable?", .arity = 1, .func = primVariableP },
    // ---- strings + chars ----
    .{ .name = "pos", .arity = 2, .func = primPos },
    .{ .name = "tlstr", .arity = 1, .func = primTlstr },
    .{ .name = "hdstr", .arity = 1, .func = primHdstr },
    .{ .name = "cn", .arity = 2, .func = primCn },
    .{ .name = "str", .arity = 1, .func = primStr },
    .{ .name = "string->n", .arity = 1, .func = primStringToN },
    .{ .name = "n->string", .arity = 1, .func = primNToString },
    .{ .name = "c-strlen", .arity = 1, .func = primCStrlen },
    .{ .name = "char-code", .arity = 2, .func = primCharCode },
    .{ .name = "substring", .arity = 3, .func = primSubstring },
    .{ .name = "shen.str->bytes", .arity = 1, .func = primStrToBytes },
    .{ .name = "shen.bytes->string", .arity = 1, .func = primBytesToStr },
    // ---- vectors / addresses ----
    .{ .name = "absvector", .arity = 1, .func = primAbsvector },
    .{ .name = "address->", .arity = 3, .func = primAddressSet },
    .{ .name = "<-address", .arity = 2, .func = primAddressGet },
    .{ .name = "emptylist", .arity = 1, .func = primEmptylist },
    // ---- stream I/O (M6) ----
    .{ .name = "write-byte", .arity = 2, .func = streams.primWriteByte },
    .{ .name = "read-byte", .arity = 1, .func = streams.primReadByte },
    .{ .name = "read-file-as-string", .arity = 1, .func = streams.primReadFileAsString },
    .{ .name = "open", .arity = 2, .func = streams.primOpen },
    .{ .name = "close", .arity = 1, .func = streams.primClose },
    // ---- process execution (M8, execplan.zig; wait/kill are Shen OS-only,
    //      ported from shen into prims.zig since execplan.zig is frozen) ----
    .{ .name = "exec-plan", .arity = 1, .func = execplan.primExecPlan },
    .{ .name = "wait", .arity = 1, .func = primWait },
    .{ .name = "kill", .arity = 2, .func = primKill },
    .{ .name = "cd", .arity = 1, .func = execplan.primCd },
    .{ .name = "getcwd", .arity = 0, .func = execplan.primGetcwd },
    .{ .name = "getpid", .arity = 0, .func = execplan.primGetpid },
    .{ .name = "getenv", .arity = 1, .func = execplan.primGetenv },
    .{ .name = "setenv", .arity = 2, .func = execplan.primSetenv },
    .{ .name = "glob", .arity = 1, .func = execplan.primGlob },
    // ---- control / eval ----
    .{ .name = "trap-error", .arity = 2, .func = primTrapError },
    .{ .name = "simple-error", .arity = 1, .func = primSimpleError },
    .{ .name = "error-to-string", .arity = 1, .func = primErrorToString },
    // ---- eval-kl (Shen OS M2, marshal.zig + hostcall.zig): the
    //      bundle-driven compile+run chain.
    .{ .name = "eval-kl", .arity = 1, .func = primEvalKl },
    .{ .name = "get-time", .arity = 1, .func = primGetTime },
    .{ .name = "intern", .arity = 1, .func = primIntern },
    .{ .name = "set", .arity = 2, .func = primSet },
    .{ .name = "value", .arity = 1, .func = primValue },
    // ---- tuples / symbols / misc ----
    .{ .name = "@p", .arity = 2, .func = primAtP },
    .{ .name = "fst", .arity = 1, .func = primFst },
    .{ .name = "snd", .arity = 1, .func = primSnd },
    .{ .name = "gensym", .arity = 1, .func = primGensym },
    .{ .name = "newvar", .arity = 0, .func = primNewvar },
    .{ .name = "element?", .arity = 2, .func = primElementP },
    .{ .name = "shen.fail!", .arity = 1, .func = primShenFail },
};

/// Comptime name -> handler map (the dispatch half of the table).
const prim_map = std.StaticStringMap(PrimFn).initComptime(blk: {
    @setEvalBranchQuota(100000);
    var kvs: [prim_table.len]struct { []const u8, PrimFn } = undefined;
    for (&kvs, 0..) |*kv, i| kv.* = .{ prim_table[i].name, prim_table[i].func };
    break :blk kvs;
});

/// The table itself (initGlobals + bundle primitive?-names iterate this).
pub fn primNames() []const PrimDef {
    return &prim_table;
}

/// C: zincvm.c:967-974 exec_primitive_valid — true iff `name` is a known C
/// primitive (the defun_get prim fallback).
pub fn isValid(name: []const u8) bool {
    return name.len != 0 and prim_map.get(name) != null;
}

/// Table lookup returning the entry (state.defunGet uses the CANONICAL
/// [:0] name literal for valPrim, independent of the caller's buffer).
pub fn lookupDef(name: []const u8) ?*const PrimDef {
    for (&prim_table) |*def| {
        if (std.mem.eql(u8, std.mem.sliceTo(def.name, 0), name)) return def;
    }
    return null;
}

/// C: zincvm.c:2790-2793 unknown tail — print + return -1.  Mapped to
/// error.Halt: the eval-loop call sites catch it and break to done with acc
/// preserved (the C `exec_primitive() < 0 -> goto done`).
fn unknownPrim(name: []const u8) VmError {
    std.debug.print("runtime: unknown primitive '{s}'\n", .{name});
    return error.Halt;
}

/// C: zincvm.c:1780-1783 exec_primitive head.  Every name dispatched through
/// the comptime map; unknown (or empty) names hard-stop.
pub fn execPrimitive(vm: *Vm, name: []const u8, acc: *Value, stack: *ValueArray) VmError!void {
    if (name.len == 0) return unknownPrim(name);
    const f = prim_map.get(name) orelse return unknownPrim(name);
    return f(vm, acc, stack);
}

// =====================================================================
//  'a': absvector, absvector?, address->, assoc, append
// =====================================================================

/// C: zincvm.c:1785-1788 absvector.  val_vector allocates the element array;
/// the popped arg is a number (no interior pointers) — no root needed.
fn primAbsvector(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const a = interp.vaPop(stack);
    // C casts `(int)a.number` — truncating mod 2^32 (no range panic).
    const size: i32 = @truncate(a.payload.number);
    acc.* = values.valVector(vm.gc, size);
}

/// C: zincvm.c:1789-1791 absvector?.
fn primAbsvectorP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .vector);
}

/// C: zincvm.c:1792-1802 address->.  NO GC allocation — the element store
/// goes through the write barrier (heap.zig:982 ports C's gc_dirty_vectors_add
/// site exactly; the null `data` guard is the .? unwrap parity with C's UB).
fn primAddressSet(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    const vec = interp.vaPop(stack);
    const idx = interp.vaPop(stack);
    const val = interp.vaPop(stack);
    const i: usize = @intCast(idx.payload.number);
    g.writeBarrierVectorStore(vec.payload.vector.data.?, i, val);
    acc.* = vec;
}

/// C: zincvm.c:1810-1830 assoc.  Alloc-free (deep_equal); key/l rooted for
/// parity with C:1811-1812.
fn primAssoc(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var key = interp.vaPop(stack);
    var l = interp.vaPop(stack);
    g.rootPushValue(&key);
    g.rootPushValue(&l);
    var found = false;
    var result = values.valNil();
    while (l.tag == .cons) {
        const car = l.payload.cons.car.?;
        if (car.tag == .cons and values.deepEqual(key, car.payload.cons.car.?.*, 0)) {
            result = car.*;
            found = true;
            break;
        }
        l = l.payload.cons.cdr.?.*;
    }
    if (!found and l.tag != .nil) {
        g.rootPop();
        g.rootPop();
        return vm.throwShen("attempt to search a non-list with assoc");
    }
    acc.* = if (found) result else values.valNil();
    g.rootPop();
    g.rootPop();
}

/// C: zincvm.c:1836-1856 append.  cons-copy a1's prefix onto tail a2; both
/// args + rev + out rooted across the valCons loops (C parity: 4 roots).
fn primAppend(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a1 = interp.vaPop(stack);
    var a2 = interp.vaPop(stack);
    if (a1.tag != .nil and a1.tag != .cons)
        return vm.throwShen("attempt to append a non-list");
    g.rootPushValue(&a1);
    g.rootPushValue(&a2);
    if (a1.tag == .nil) {
        g.rootPop();
        g.rootPop();
        acc.* = a2;
        return;
    }
    var rev = values.valNil();
    g.rootPushValue(&rev);
    while (a1.tag == .cons) {
        rev = values.valCons(g, a1.payload.cons.car.?.*, rev);
        a1 = a1.payload.cons.cdr.?.*;
    }
    var out = a2;
    g.rootPushValue(&out);
    while (rev.tag == .cons) {
        out = values.valCons(g, rev.payload.cons.car.?.*, out);
        rev = rev.payload.cons.cdr.?.*;
    }
    acc.* = out;
    g.rootPop();
    g.rootPop();
    g.rootPop();
    g.rootPop();
}

// =====================================================================
//  'b': boolean?
// =====================================================================

/// C: zincvm.c:1859-1861 boolean?.
fn primBooleanP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .boolean);
}

// =====================================================================
//  'c': cons, cons?, cn, c-strlen, char-code
// =====================================================================

/// C: zincvm.c:1867-1870 cons.  valCons roots its by-value params internally
/// (values.zig, C:274-294) — safe by construction.
fn primCons(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    acc.* = values.valCons(vm.gc, a1, a2);
}

/// C: zincvm.c:1871-1873 cons?.
fn primConsP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .cons);
}

/// Helper: rendered length of one cn operand; numbers are pre-formatted into
/// `t` (a non-GC stack buffer) so the second pass needs no re-format.
fn cnMeasure(v: Value, t: *[32]u8) usize {
    return switch (v.tag) {
        .string => @intCast(@max(v.payload.str.len, 0)),
        .number => blk: {
            const s = std.fmt.bufPrint(t, "{d}", .{v.payload.number}) catch unreachable;
            break :blk s.len;
        },
        .symbol => values.symSlice(v).len,
        .boolean => if (v.payload.boolean != 0) @as(usize, 4) else 5,
        .nil => 2,
        else => 3,
    };
}

/// Helper: write one cn operand into `buf` (l = cnMeasure length).  Interior
/// reads (str.data / sym.name) go through the caller's ROOTED locals.
fn cnWrite(buf: []u8, v: Value, l: usize, t: *const [32]u8) void {
    switch (v.tag) {
        .string => @memcpy(buf[0..l], values.strSlice(v)[0..l]),
        .number => @memcpy(buf[0..l], t[0..l]),
        .symbol => @memcpy(buf[0..l], values.symSlice(v)[0..l]),
        .boolean => @memcpy(buf[0..l], if (v.payload.boolean != 0) "true" else "false"),
        .nil => @memcpy(buf[0..2], "[]"),
        else => @memcpy(buf[0..3], "[?]"),
    }
}

/// C: zincvm.c:1879-1916 cn.  Two-pass: measure (into non-GC stack buffers),
/// then ONE GC_STR alloc with a1/a2 ROOTED across it; interior reads after
/// the alloc go through the rooted locals (C:1900-1921).
fn primCn(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a1 = interp.vaPop(stack);
    var a2 = interp.vaPop(stack);
    var t1: [32]u8 = undefined;
    var t2: [32]u8 = undefined;
    const l1 = cnMeasure(a1, &t1);
    const l2 = cnMeasure(a2, &t2);
    const total = l1 + l2;
    g.rootPushValue(&a1);
    g.rootPushValue(&a2);
    const buf = g.allocRaw(total + 1);
    cnWrite(buf[0..l1], a1, l1, &t1);
    cnWrite(buf[l1..][0..l2], a2, l2, &t2);
    buf[total] = 0;
    g.rootPop();
    g.rootPop();
    acc.* = .{ .tag = .string, .payload = .{ .str = .{
        .data = buf,
        .len = @intCast(total),
    } } };
}

/// C: zincvm.c:1937-1939 c-strlen (O(1) string length).
fn primCStrlen(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valNumber(a.payload.str.len);
}

/// C: zincvm.c:1943-1952 char-code (byte at index; -1 out of bounds).
fn primCharCode(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack); // string
    const n = interp.vaPop(stack); // index
    const i = n.payload.number;
    const len: i64 = a.payload.str.len;
    if (i >= 0 and i < len) {
        const data = a.payload.str.data.?;
        acc.* = values.valNumber(@intCast(data[@intCast(i)]));
    } else {
        acc.* = values.valNumber(-1);
    }
}

// =====================================================================
//  'e': error?, error-to-string, element?, emptylist, empty?
// =====================================================================

/// C: zincvm.c:1962-1964 error?.
fn primErrorP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .error_);
}

/// C: zincvm.c:1967-1975 error-to-string — PORT-FIX: C passes the raw
/// `a.error.message` pointer into val_string, whose GC alloc can move the
/// message and leave the memcpy stale.  The port copies through the
/// slot-rooted valStringFromErr (values.zig) instead.  `a` is rooted across
/// the copy (C parity: gc_root_push_value(&a)).
fn primErrorToString(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a = interp.vaPop(stack);
    var guard = g.rootValue(&a);
    defer guard.end();
    if (a.tag == .error_) {
        acc.* = values.valStringFromErr(g, &a);
    } else if (a.tag == .string) {
        acc.* = a;
    } else {
        acc.* = values.valString(g, "unknown error");
    }
}

// =====================================================================
//  'e': eval-kl — the bundle-driven compile+run chain (Shen OS M2)
// =====================================================================

/// One eval-kl chain stage (C:2012-2017 extract-kl / :2028-2033 kl->zinc /
/// :2044-2049 toplevel-interp): resolve the bundled closure; a non-lambda
/// resolution warns on stderr and reports missing (null), else the stage
/// runs through the shared hostcall env-extend pattern.  `arg` must be a
/// rooted slot (the caller's chain roots).
fn evalKlStage(vm: *Vm, name: []const u8, arg: *Value) VmError!?Value {
    if (vm.defunGet(name).tag != .lambda) {
        std.debug.print("runtime: eval-kl: {s} not found in bundle\n", .{name});
        return null;
    }
    return hostcall.applyBundledN(vm, name, &.{arg.*});
}

/// The compile+run chain (C:2009-2060): marshal_to_tagged → extract-kl →
/// kl->zinc → toplevel-interp → demarshal_from_tagged — the first
/// bundle-scale execution path.  `a` (the input form) must ALREADY be rooted
/// by the caller.  Every intermediate is rooted exactly as C roots it
/// (tagged C:2010, the three closures ride applyBundledN's internal fn
/// root, klambda C:2026, zinc_code C:2042, tagged_result C:2058); the roots
/// are deliberately NOT popped here — the caller's single defer
/// rootPopTo(entry_wm) is the pop site (C's one gc_root_pop_to(eval_kl_wm)
/// at eval_kl_done, :2063/:2068 — a Zig defer replaces C's two-path pop_to
/// since error unwinding runs intermediate defers).  Returns null when a
/// bundle closure is missing (warned in evalKlStage; C jumps to
/// eval_kl_done with result = a).
fn evalKlChain(vm: *Vm, a: *Value) VmError!?Value {
    const g = vm.gc;
    var tagged = marshal.marshalToTagged(vm, a.*);
    g.rootPushValue(&tagged); // C:2010

    var klambda = (try evalKlStage(vm, "extract-kl", &tagged)) orelse return null;
    g.rootPushValue(&klambda); // C:2026

    var zinc_code = (try evalKlStage(vm, "kl->zinc", &klambda)) orelse return null;
    g.rootPushValue(&zinc_code); // C:2042

    var tagged_result = (try evalKlStage(vm, "toplevel-interp", &zinc_code)) orelse return null;
    g.rootPushValue(&tagged_result); // C:2058

    return marshal.demarshalFromTagged(vm, tagged_result); // C:2060
}

/// C: zincvm.c:1999-2082 eval-kl — THE bundle-scale execution path.  C's
/// volatile-locals + setjmp/longjmp dance collapses to plain locals + roots
/// + `catch` (plan DECISION A):
///   - CatchSite with in_trap_error = 0 wraps the WHOLE chain (C:2001-2004)
///     so a throw anywhere lands here, not in an outer handler;
///   - the popped form is rooted and the entry watermark + defer rootPopTo
///     replace C's eval_kl_wm + two-path pop_to (:2006/:2063/:2068);
///   - CRITICAL SEMANTIC (C:2070-2081): eval-kl RETURNS error values — on a
///     thrown ShenError, acc = vm.err_slot (the once-rooted DECISION-A
///     replacement for C's cf.error_val + S3 root dance) and the prim
///     SUCCEEDS; the error is never re-propagated as a VmError.  shensh's
///     eval_kl_form relies on this (exec_primitive's return is ignored).
/// PORT-FIX (deliberate deviation): C leaves the input form `a` unrooted —
/// its own :2070-2076 comment documents how the old `*acc = result = a`
/// echo read garbage after a collection moved the form's cells.  The port
/// roots `a`, so the missing-closure fallback (result = a, C goto
/// eval_kl_done) is collector-safe.
fn primEvalKl(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a = interp.vaPop(stack);
    const entry_wm = g.rootWatermark(); // C:2006 eval_kl_wm
    var site = state.CatchSite{ .in_trap_error = false, .parent = vm.catch_chain }; // C:2001-2003
    vm.catch_chain = &site;
    g.rootPushValue(&a); // PORT-FIX root (see doc)
    defer g.rootPopTo(entry_wm); // the ONE pop site — runs LAST (C:2063/:2068)
    defer vm.catch_chain = site.parent; // C:2064/:2069

    // C:2005 `volatile Value result = a` — the missing-closure fallback,
    // now read through the rooted `a`.
    var result: Value = a;
    if (evalKlChain(vm, &a)) |maybe| {
        if (maybe) |v| result = v;
        // null: a bundle closure missing — result stays the input form
        // (warned in evalKlStage; C goto eval_kl_done).
    } else |e| switch (e) {
        // C:2077-2081: propagate the real error VALUE, not the input form.
        error.ShenError => result = vm.err_slot,
        // vmExecEnv never propagates Halt (the eval loop contains it at its
        // own call sites); C has no equivalent arm.  Kept for exhaustiveness.
        error.Halt => return error.Halt,
    }
    acc.* = result; // C:2065/:2079 `*acc = result/errv; return 0`
}

/// C: zincvm.c:1984-1998 element? — deep_equal list membership.  Alloc-free;
/// x/l rooted for parity with C:1987-1988.
fn primElementP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var x = interp.vaPop(stack); // needle
    var l = interp.vaPop(stack); // haystack
    g.rootPushValue(&x);
    g.rootPushValue(&l);
    var found = false;
    var cur = l;
    while (cur.tag == .cons) {
        if (values.deepEqual(x, cur.payload.cons.car.?.*, 0)) {
            found = true;
            break;
        }
        cur = cur.payload.cons.cdr.?.*;
    }
    g.rootPop(); // l
    g.rootPop(); // x
    acc.* = values.valBoolean(found);
}

/// C: zincvm.c:2087-2091 emptylist: (number 0) -> nil; anything else falls
/// through the C dispatch to `unknown` -> return -1 (error.Halt here).
fn primEmptylist(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    if (a.tag == .number and a.payload.number == 0) {
        acc.* = values.valNil();
        return;
    }
    return error.Halt; // C falls through to unknown
}

/// C: zincvm.c:2094-2096 empty?.
fn primEmptyP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .nil);
}

// =====================================================================
//  'f': fst, function?
// =====================================================================

/// C: zincvm.c:2152-2155 fst — no nil guard in C (NULL deref); the .? unwrap
/// is the parity crash.
fn primFst(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = a.payload.cons.car.?.*;
}

/// C: zincvm.c:2156-2158 function?.
fn primFunctionP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .lambda or a.tag == .prim);
}

// =====================================================================
//  'g': gensym, get-time
// =====================================================================

/// C: zincvm.c:2164-2171 gensym (counter is a C static; the port keeps it on
/// Vm).  Pops a spurious arg if present (nullary-call convention).
fn primGensym(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    if (stack.len > 0) _ = interp.vaPop(stack);
    var buf: [64]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "shen.gensym_{d}", .{vm.gensym_counter}) catch unreachable;
    vm.gensym_counter += 1;
    acc.* = symbols.valSymbol(&vm.symbols, s);
}

/// Wall-clock seconds since epoch (C time(NULL)).  Zig 0.16's std.time
/// exposes no clock functions (constants only), so this reads the raw
/// clock_gettime syscall — the VM targets Linux (the GC's mmap heap).
/// Non-Linux targets (cross-checks only) get a deterministic 0.
fn wallSeconds() i64 {
    if (comptime @import("builtin").os.tag != .linux) return 0;
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.REALTIME), @intFromPtr(&ts));
    return @intCast(ts.sec);
}

/// Wall-clock milliseconds — the "run" mode source (C clock() has no Zig
/// counterpart; the value only feeds numeric comparisons).
fn wallMillis() i64 {
    if (comptime @import("builtin").os.tag != .linux) return 0;
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.REALTIME), @intFromPtr(&ts));
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

/// C: zincvm.c:2172-2178 get-time.  unix/real -> seconds since epoch;
/// run -> C's clock() mapped to wall-clock milliseconds (see wallMillis).
/// An unknown mode falls through the C dispatch to unknown.
fn primGetTime(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const mode = interp.vaPop(stack);
    if (mode.tag == .symbol) {
        const nm = values.symSlice(mode);
        if (std.mem.eql(u8, nm, "unix") or std.mem.eql(u8, nm, "real")) {
            acc.* = values.valNumber(wallSeconds());
            return;
        }
        if (std.mem.eql(u8, nm, "run")) {
            acc.* = values.valNumber(wallMillis());
            return;
        }
    }
    return error.Halt; // C falls through to unknown
}

// =====================================================================
//  'h': hd, hdstr
// =====================================================================

/// C: zincvm.c:2264-2269 hd (nil -> nil; else car).
fn primHd(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    if (a.tag == .nil) {
        acc.* = values.valNil();
        return;
    }
    acc.* = a.payload.cons.car.?.*;
}

/// C: zincvm.c:2270-2273 hdstr.  valStringFrom roots the popped slot across
/// its allocRaw internally — pass the popped local by pointer.
fn primHdstr(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    var a = interp.vaPop(stack);
    acc.* = values.valStringFrom(vm.gc, &a, 0, 1);
}

// =====================================================================
//  'i': intern
// =====================================================================

/// C: zincvm.c:2279-2286 intern (string -> symbol; 255-byte cap).
fn primIntern(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const a = interp.vaPop(stack);
    var buf: [256]u8 = undefined;
    const raw_len: usize = @intCast(@max(a.payload.str.len, 0));
    const n = @min(raw_len, 255);
    @memcpy(buf[0..n], values.strSlice(a)[0..n]);
    acc.* = symbols.valSymbol(&vm.symbols, buf[0..n]);
}

// =====================================================================
//  'n': n->string, number?, newvar
// =====================================================================

/// C: zincvm.c:2334-2337 n->string — single byte from the number.  The
/// `(char)` cast truncates (C parity via bitCast+truncate).
fn primNToString(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const a = interp.vaPop(stack);
    var buf: [1]u8 = .{@truncate(@as(u64, @bitCast(a.payload.number)))};
    acc.* = values.valString(vm.gc, buf[0..1]);
}

/// C: zincvm.c:2339-2341 number?.
/// M4: floats are numbers too (Elm `number?`/`isNumber` parity).
fn primNumberP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .number or a.tag == .float);
}

/// C: zincvm.c:2343-2350 newvar (counter on Vm; pops a spurious arg).
fn primNewvar(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    if (stack.len > 0) _ = interp.vaPop(stack);
    var buf: [64]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "V_{d}", .{vm.newvar_counter}) catch unreachable;
    vm.newvar_counter += 1;
    acc.* = symbols.valSymbol(&vm.symbols, s);
}

// =====================================================================
//  'p': pos
// =====================================================================

/// C: zincvm.c:2435-2446 pos — bounds-checked 1-char slice; out of bounds
/// throws inside trap-error, else an empty string.  valStringFrom roots a1
/// internally.
fn primPos(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a1 = interp.vaPop(stack); // string
    const a2 = interp.vaPop(stack); // index
    const pl = a2.payload.number;
    const slen: i64 = a1.payload.str.len;
    if (pl < 0 or pl >= slen) {
        if (vm.catch_chain != null and vm.catch_chain.?.in_trap_error)
            return vm.throwShen("pos out of bounds");
        acc.* = values.valString(g, "");
    } else {
        acc.* = values.valStringFrom(g, &a1, @intCast(pl), 1);
    }
}

// =====================================================================
//  'r': reverse
// =====================================================================

/// C: zincvm.c:2508-2526 reverse — acc-built reversal with a/out rooted
/// across the valCons loop (C:2418-2426 rooting, 2 roots).
fn primReverse(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a = interp.vaPop(stack);
    if (a.tag != .nil and a.tag != .cons)
        return vm.throwShen("attempt to reverse a non-list");
    g.rootPushValue(&a);
    var out = values.valNil();
    g.rootPushValue(&out);
    while (a.tag == .cons) {
        out = values.valCons(g, a.payload.cons.car.?.*, out);
        a = a.payload.cons.cdr.?.*;
    }
    acc.* = out;
    g.rootPop();
    g.rootPop();
}

// =====================================================================
//  's': symbol?, string?, simple-error, str, stream?, set, string->n,
//       shen.fail!, snd, substring, shen.str->bytes, shen.bytes->string
// =====================================================================

/// C: zincvm.c:2287-2289 symbol?.
fn primSymbolP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .symbol);
}

/// C: zincvm.c:2290-2292 string?.
fn primStringP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .string);
}

/// C: zincvm.c:2293-2304 simple-error.  The message is copied into a STACK
/// buffer first (C's msg[256]) — a GC-interior str.data pointer must not
/// survive into throwShen's valError alloc (the popped `a` is unrooted).
/// repl_mode's longjmp exit is omitted with the meta REPL.
fn primSimpleError(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = acc; // C overwrites acc's slot only via the error path
    const a = interp.vaPop(stack);
    var buf: [256]u8 = undefined;
    var msg: []const u8 = "simple-error called";
    if (a.tag == .string) {
        const s = values.strSlice(a);
        const n = @min(s.len, 255); // C snprintf("%.*s", a.str.len <= 255)
        @memcpy(buf[0..n], s[0..n]);
        msg = buf[0..n];
    }
    return vm.throwShen(msg); // throwShen copies msg before buf dies
}

/// C: zincvm.c:2305-2341 str.  Scalars take a non-GC stack buffer; the
/// composite case grows a C-heap buffer (malloc/realloc parity) until
/// strValue fits, then valString copies from the non-GC buffer (the
/// valString CONTRACT holds).
fn primStr(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    const a = interp.vaPop(stack);
    switch (a.tag) {
        .symbol => acc.* = values.valString(g, values.symSlice(a)),
        .string => acc.* = a,
        .number => {
            var buf: [64]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{a.payload.number}) catch unreachable;
            acc.* = values.valString(g, s);
        },
        .float => {
            var buf: [64]u8 = undefined;
            const s = values.floatText(&buf, a.payload.float);
            acc.* = values.valString(g, s);
        },
        .boolean => acc.* = values.valString(
            g,
            if (a.payload.boolean != 0) "true" else "false",
        ),
        else => {
            const a_alloc = std.heap.page_allocator;
            var cap: usize = 4096;
            while (true) {
                const buf = a_alloc.alloc(u8, cap) catch
                    return vm.throwShen("str: out of memory");
                var w: std.Io.Writer = .fixed(buf);
                if (values.strValue(&w, a, 0)) |_| {
                    const s = w.buffered();
                    acc.* = values.valString(g, s);
                    a_alloc.free(buf);
                    return;
                } else |e| {
                    a_alloc.free(buf);
                    if (e != error.NoSpaceLeft) return vm.throwShen("str: write error");
                    cap *= 2; // grow-until-fits (C realloc loop)
                    if (cap > (1 << 30)) return vm.throwShen("str: out of memory");
                }
            }
        },
    }
}

/// C: zincvm.c:2481-2483 stream?.
fn primStreamP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valBoolean(a.tag == .stream);
}

/// C: zincvm.c:2495-2497 set.  value_set stores into the GC-registered
/// values table; no alloc between pop and store (no root needed).
fn primSet(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const sym = interp.vaPop(stack);
    const v = interp.vaPop(stack);
    vm.valueSet(values.symSlice(sym), v);
    acc.* = v;
}

/// C: zincvm.c:2498-2500 string->n (first byte as number; 0 on empty).
fn primStringToN(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = values.valNumber(if (a.payload.str.len > 0)
        @intCast(a.payload.str.data.?[0])
    else
        0);
}

/// C: zincvm.c:2502-2509 shen.fail! — with an arg: (fail Arg); without: throw.
/// valCons roots its params internally (safe by construction).
fn primShenFail(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    if (stack.len > 0) {
        const arg = interp.vaPop(stack);
        const inner = values.valCons(g, arg, values.valNil());
        acc.* = values.valCons(
            g,
            symbols.valSymbol(&vm.symbols, "fail"),
            inner,
        );
        return;
    }
    return vm.throwShen("fail");
}

/// C: zincvm.c:2511-2513 snd — no nil guard (parity crash via .?).
fn primSnd(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    acc.* = a.payload.cons.cdr.?.*;
}

/// C: zincvm.c:2514-2533 substring — clamped Str[Start..Start+Len); `s`
/// rooted across valStringFrom (C:2627 comment: keep the source alive — the
/// copy may alias s's buffer and s may be in the nursery).
fn primSubstring(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var s = interp.vaPop(stack); // string
    const st = interp.vaPop(stack); // start
    const ln = interp.vaPop(stack); // len
    var start = st.payload.number;
    var len = ln.payload.number;
    const slen: i64 = s.payload.str.len;
    if (start < 0) start = 0;
    if (start > slen) start = slen;
    if (len < 0) len = 0;
    if (start + len > slen) len = slen - start;
    g.rootPushValue(&s);
    const r = values.valStringFrom(g, &s, @intCast(start), @intCast(len));
    g.rootPop();
    acc.* = r;
}

/// C: zincvm.c:2542-2549 shen.str->bytes — string -> list of byte codes,
/// built back-to-front so byte[0] lands at the head; `a` and `out` rooted
/// across the valCons loop, interior reads through the rooted `a`.
fn primStrToBytes(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a = interp.vaPop(stack);
    if (a.tag != .string)
        return vm.throwShen("attempt to convert a non-string with str->bytes");
    g.rootPushValue(&a);
    var out = values.valNil();
    g.rootPushValue(&out);
    var i: i64 = a.payload.str.len;
    while (i > 0) {
        i -= 1;
        const data = a.payload.str.data.?; // fresh read via the rooted slot
        out = values.valCons(g, values.valNumber(@intCast(data[@intCast(i)])), out);
    }
    acc.* = out;
    g.rootPop();
    g.rootPop();
}

/// C: zincvm.c:2562-2573 shen.bytes->string — count, ONE GC_STR alloc with
/// `a` rooted, then re-walk the list THROUGH the rooted `a` (cells may have
/// moved during the alloc).
fn primBytesToStr(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var a = interp.vaPop(stack);
    if (a.tag != .nil and a.tag != .cons)
        return vm.throwShen("attempt to convert a non-list with bytes->string");
    g.rootPushValue(&a);
    var n: usize = 0;
    var cur = a;
    while (cur.tag == .cons) {
        n += 1;
        cur = cur.payload.cons.cdr.?.*;
    }
    const buf = g.allocRaw(n + 1);
    var i: usize = 0;
    cur = a; // restart through the rooted slot
    while (cur.tag == .cons) {
        buf[i] = @truncate(@as(u64, @bitCast(cur.payload.cons.car.?.payload.number)));
        i += 1;
        cur = cur.payload.cons.cdr.?.*;
    }
    buf[n] = 0;
    g.rootPop();
    acc.* = .{ .tag = .string, .payload = .{ .str = .{
        .data = buf,
        .len = @intCast(n),
    } } };
}

// =====================================================================
//  't': tl, trap-error, tlstr
// =====================================================================

/// C: zincvm.c:2625-2630 tl (nil -> nil; else cdr).
fn primTl(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    if (a.tag == .nil) {
        acc.* = values.valNil();
        return;
    }
    acc.* = a.payload.cons.cdr.?.*;
}

/// C: zincvm.c:2632-2682 trap-error — the DECISION-A port of the setjmp/
/// longjmp CatchFrame dance:
///   - body/handler pushed as VALUE roots and kept rooted through BOTH the
///     body run and the error path (the C `volatile` + watermark pair);
///   - body_wm is taken AFTER the two pushes, so rootPopTo(body_wm) on the
///     error path drops only the garbage the body run left above them;
///   - the CatchSite is pushed on vm.catch_chain with in_trap_error armed
///     only around the body run (conditional throw sites consult it);
///   - the error value lives in the once-rooted vm.err_slot (C cf.error_val).
///
/// Zig error returns unwind intermediate vmExecEnv frames WITH their defers
/// (rootPopTo(entry_wm)) running — every intermediate root is balanced
/// before the error reaches this handler, exactly what C's longjmp sites
/// popped manually.
fn primTrapError(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var body = interp.vaPop(stack);
    var handler = interp.vaPop(stack);
    g.rootPushValue(&body);
    g.rootPushValue(&handler);
    var site = state.CatchSite{ .in_trap_error = false, .parent = vm.catch_chain };
    vm.catch_chain = &site;
    const body_wm = g.rootWatermark(); // ABOVE the body/handler roots

    var body_val: Value = undefined;
    var threw = false;
    site.in_trap_error = true;
    if (body.tag == .lambda) {
        const lambda_env_len = body.payload.lambda.env_len;
        const new_len = lambda_env_len + 1;
        // body is rooted, so its env read after the alloc is fresh.
        const new_env = g.allocArray(Value, @intCast(new_len));
        if (lambda_env_len > 0) {
            const lel: usize = @intCast(lambda_env_len);
            @memcpy(new_env[0..lel], body.payload.lambda.env.?[0..lel]);
            if (g.inOldgen(@intFromPtr(new_env))) {
                var j: usize = 0;
                while (j < lel) : (j += 1) {
                    if (gc.scan.valueReferencesNursery(g, &body.payload.lambda.env.?[j])) {
                        g.dirtyVectorsAdd(new_env);
                        break;
                    }
                }
            }
        }
        new_env[@intCast(lambda_env_len)] = values.valNil();
        if (interp.vmExecEnv(vm, body.payload.lambda.code, body.payload.lambda.code_len, new_env, new_len)) |v| {
            body_val = v;
        } else |_| {
            threw = true;
        }
    } else {
        body_val = body;
    }
    site.in_trap_error = false;

    if (!threw) {
        vm.catch_chain = site.parent;
        g.rootPop(); // handler
        g.rootPop(); // body
        acc.* = body_val;
        return;
    }

    // Error path (C's setjmp != 0 arm): restore the chain, drop everything
    // the body run pushed, then build the handler env with body/handler/err
    // all still rooted across the henv alloc.
    vm.catch_chain = site.parent;
    g.rootPopTo(body_wm);
    var err = vm.err_slot;
    const env_len = handler.payload.lambda.env_len;
    const new_env_len = env_len + 1;
    g.rootPushValue(&err);
    const henv = g.allocArray(Value, @intCast(new_env_len));
    if (env_len > 0) {
        const hel: usize = @intCast(env_len);
        @memcpy(henv[0..hel], handler.payload.lambda.env.?[0..hel]);
        if (g.inOldgen(@intFromPtr(henv))) {
            var j: usize = 0;
            while (j < hel) : (j += 1) {
                if (gc.scan.valueReferencesNursery(g, &handler.payload.lambda.env.?[j])) {
                    g.dirtyVectorsAdd(henv);
                    break;
                }
            }
        }
    }
    henv[@intCast(env_len)] = err;
    const hc = handler.payload.lambda.code;
    const hl = handler.payload.lambda.code_len;
    g.rootPop(); // err
    g.rootPop(); // handler
    g.rootPop(); // body
    acc.* = try interp.vmExecEnv(vm, hc, hl, henv, new_env_len);
}

/// C: zincvm.c:2684-2687 tlstr.  valStringFrom roots the popped slot; a
/// len<=1 string yields "" (C would underflow len-1 to a huge size_t and
/// crash on the empty string — deliberate safe deviation).
fn primTlstr(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    var a = interp.vaPop(stack);
    const n: usize = if (a.payload.str.len > 1) @intCast(a.payload.str.len - 1) else 0;
    acc.* = values.valStringFrom(vm.gc, &a, 1, n);
}

// =====================================================================
//  'v': value, variable?
// =====================================================================

/// C: zincvm.c:2692-2694 value (value_get carries the symbol fallback).
fn primValue(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const a = interp.vaPop(stack);
    acc.* = vm.valueGet(values.symSlice(a));
}

/// C: zincvm.c:2695-2716 variable? — uppercase-initial symbol whose
/// continuation chars are alphanumeric or symbol punctuation.
fn primVariableP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a = interp.vaPop(stack);
    if (a.tag != .symbol) {
        acc.* = values.valBoolean(false);
        return;
    }
    const s = values.symSlice(a);
    if (s.len == 0 or s[0] < 'A' or s[0] > 'Z') {
        acc.* = values.valBoolean(false);
        return;
    }
    for (s[1..]) |c| {
        const ok = (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z') or
            (c >= '0' and c <= '9') or
            (c == '`' or c == '=' or c == '*' or c == '/' or c == '+' or
                c == '_' or c == '?' or c == '$' or c == '!' or c == '@' or
                c == '~' or c == '.' or c == '>' or c == '<' or c == '&' or
                c == '%' or c == '\'' or c == '#');
        if (!ok) {
            acc.* = values.valBoolean(false);
            return;
        }
    }
    acc.* = values.valBoolean(true);
}

// =====================================================================
//  Arithmetic: +, -, *, /
// =====================================================================

/// C: zincvm.c:2743-2746 + (wrapping — two's-complement, the behavior of
/// the compiled C on every target we care about).
///
/// M4 runtime tag dispatch: (Int,Int) stays wrapping Int; any Float operand
/// promotes Int->Float and computes in f64 (Elm numeric-literal polymorphism:
/// 1 + 2.5 = 3.5).  NO type guard — bare arithmetic reads the number bits
/// directly (shen semantics, AGENTS.md; the metacircular interpreter relies
/// on it), while Float operands promote as fx-ui's M4 requires.
fn primAdd(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    // Bare arithmetic (shen semantics, AGENTS.md): NO type guard — the
    // metacircular interpreter passes Shen-level values the safe-wrapper
    // layer has validated, and shen's VM reads the number bits directly.
    // Float support (fx-ui M4): promote when either operand is a float.
    if (a1.tag == .float or a2.tag == .float) {
        acc.* = values.valFloat(asFloat(a1) + asFloat(a2));
    } else {
        acc.* = values.valNumber(a1.payload.number +% a2.payload.number);
    }
}

/// C: zincvm.c:2748-2751 -.
fn primSub(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    // Bare arithmetic (shen semantics, AGENTS.md): no type guard.
    if (a1.tag == .float or a2.tag == .float) {
        acc.* = values.valFloat(asFloat(a1) - asFloat(a2));
    } else {
        acc.* = values.valNumber(a1.payload.number -% a2.payload.number);
    }
}

/// C: zincvm.c:2753-2756 *.
fn primMul(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    // Bare arithmetic (shen semantics, AGENTS.md): no type guard.
    if (a1.tag == .float or a2.tag == .float) {
        acc.* = values.valFloat(asFloat(a1) * asFloat(a2));
    } else {
        acc.* = values.valNumber(a1.payload.number *% a2.payload.number);
    }
}

/// C: zincvm.c:2758-2761 /.  Elm `//` — INT-only division (Elm splits integer
/// `//` from float `/`; the latter is the separate `f/` prim).  Division by
/// zero traps (C: SIGFPE; Zig: panic).
fn primDiv(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    // Bare integer division (shen semantics, AGENTS.md): no type guard.
    acc.* = values.valNumber(@divTrunc(a1.payload.number, a2.payload.number));
}

/// M4 `f/` — Elm `/`: ALWAYS f64 division, promoting Int->Float so 2 / 3 =
/// 0.666... (and x / 0.0 = Infinity, matching Elm's float division).
fn primFdiv(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    // ALWAYS float division (fx-ui M4); no int type guard — f/ is the
    // dedicated float-division prim (shen has no f/; this is fx-ui-only).
    acc.* = values.valFloat(asFloat(a1) / asFloat(a2));
}

/// M4 numeric promote: a .number/.float Value as f64.  Callers guarantee the
/// tag is numeric (see the dispatch guards in the arithmetic/comparison prims).
fn asFloat(v: Value) f64 {
    return switch (v.tag) {
        .float => v.payload.float,
        .number => @floatFromInt(v.payload.number),
        else => unreachable,
    };
}

// =====================================================================
//  Comparison: =, <, <=, <-address, >, >=
// =====================================================================

/// C: zincvm.c:2765-2787 = — tag-pairwise, deep_equal for cons/vector,
/// SYMBOL-vs-PRIM name comparison in BOTH directions.
fn primEq(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    if (a1.tag == .number and a2.tag == .number) {
        acc.* = values.valBoolean(a1.payload.number == a2.payload.number);
    } else if (a1.tag == .string and a2.tag == .string) {
        acc.* = values.valBoolean(a1.payload.str.len == a2.payload.str.len and
            std.mem.eql(u8, values.strSlice(a1), values.strSlice(a2)));
    } else if (a1.tag == .symbol and a2.tag == .symbol) {
        acc.* = values.valBoolean(std.mem.eql(u8, values.symSlice(a1), values.symSlice(a2)));
    } else if (a1.tag == .boolean and a2.tag == .boolean) {
        acc.* = values.valBoolean(a1.payload.boolean == a2.payload.boolean);
    } else if ((a1.tag == .float or a1.tag == .number) and
        (a2.tag == .float or a2.tag == .number))
    {
        // M4: promote Int/Float so = and /= treat 2 == 2.0 as true (Elm
        // parity + consistency with the promoted < <= > >=).  The Int/Int
        // branch above already handled both-number, so asFloat sees at least
        // one .float here; both sides are numeric so else=>unreachable is
        // unreached.  IEEE NaN!=NaN preserved.  NOTE: the review fix's
        // literal `a1.tag == .float or a2.tag == .float` guard was widened
        // to both-numeric so a float-vs-nonnumber mismatch (2.0 == "x")
        // still falls through to false instead of panicking in asFloat.
        acc.* = values.valBoolean(asFloat(a1) == asFloat(a2));
    } else if ((a1.tag == .cons and a2.tag == .symbol) or
        (a1.tag == .symbol and a2.tag == .cons))
    {
        acc.* = values.valBoolean(false);
    } else if (a1.tag == .symbol and a2.tag == .prim) {
        acc.* = values.valBoolean(std.mem.eql(u8, values.symSlice(a1), values.primSlice(a2)));
    } else if (a1.tag == .prim and a2.tag == .symbol) {
        acc.* = values.valBoolean(std.mem.eql(u8, values.primSlice(a1), values.symSlice(a2)));
    } else if (a1.tag == .cons and a2.tag == .cons) {
        acc.* = values.valBoolean(values.deepEqual(a1, a2, 0));
    } else if (a1.tag == .vector and a2.tag == .vector) {
        acc.* = values.valBoolean(values.deepEqual(a1, a2, 0));
    } else {
        acc.* = values.valBoolean(a1.tag == .nil and a2.tag == .nil);
    }
}

/// C: zincvm.c:2789-2792 <.
/// M4 runtime tag dispatch: promote Int/Float across the comparison (2 < 2.5
/// is true); non-numeric operands compare False (unchanged).
fn primLt(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    if ((a1.tag == .float or a1.tag == .number) and
        (a2.tag == .float or a2.tag == .number))
    {
        if (a1.tag == .float or a2.tag == .float) {
            acc.* = values.valBoolean(asFloat(a1) < asFloat(a2));
        } else {
            acc.* = values.valBoolean(a1.payload.number < a2.payload.number);
        }
    } else {
        acc.* = values.valBoolean(false);
    }
}

/// C: zincvm.c:2793-2796 <=.
fn primLe(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    if ((a1.tag == .float or a1.tag == .number) and
        (a2.tag == .float or a2.tag == .number))
    {
        if (a1.tag == .float or a2.tag == .float) {
            acc.* = values.valBoolean(asFloat(a1) <= asFloat(a2));
        } else {
            acc.* = values.valBoolean(a1.payload.number <= a2.payload.number);
        }
    } else {
        acc.* = values.valBoolean(false);
    }
}

/// C: zincvm.c:2797-2801 <-address (no bounds guard — parity crash).
fn primAddressGet(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const vec = interp.vaPop(stack);
    const idx = interp.vaPop(stack);
    const i: usize = @intCast(idx.payload.number);
    acc.* = vec.payload.vector.data.?[i];
}

/// C: zincvm.c:2803-2806 >.
fn primGt(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    if ((a1.tag == .float or a1.tag == .number) and
        (a2.tag == .float or a2.tag == .number))
    {
        if (a1.tag == .float or a2.tag == .float) {
            acc.* = values.valBoolean(asFloat(a1) > asFloat(a2));
        } else {
            acc.* = values.valBoolean(a1.payload.number > a2.payload.number);
        }
    } else {
        acc.* = values.valBoolean(false);
    }
}

/// C: zincvm.c:2807-2810 >=.
fn primGe(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    if ((a1.tag == .float or a1.tag == .number) and
        (a2.tag == .float or a2.tag == .number))
    {
        if (a1.tag == .float or a2.tag == .float) {
            acc.* = values.valBoolean(asFloat(a1) >= asFloat(a2));
        } else {
            acc.* = values.valBoolean(a1.payload.number >= a2.payload.number);
        }
    } else {
        acc.* = values.valBoolean(false);
    }
}

// =====================================================================
//  '@': @p
// =====================================================================

/// C: zincvm.c:2814-2817 @p — valCons roots its params internally.
fn primAtP(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const a1 = interp.vaPop(stack);
    const a2 = interp.vaPop(stack);
    acc.* = values.valCons(vm.gc, a1, a2);
}

// =====================================================================
//  wait / kill — the Shen OS process prims (C: zincvm.c:2693-2700 /
//  :2294-2299).  Ported into prims.zig (not execplan.zig) because
//  execplan.zig is frozen this phase; the libc externs are declared at the
//  top of this file and waitStatusCode is shared via execplan.
// =====================================================================

/// C: zincvm.c:2294-2299 kill.  ZINC RTL: a1 = leftmost = Pid (popped
/// FIRST), a2 = Sig.
fn primKill(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const pidv = interp.vaPop(stack);
    const sigv = interp.vaPop(stack);
    _ = kill(@truncate(pidv.payload.number), @truncate(sigv.payload.number));
    acc.* = values.valBoolean(true);
}

/// C: zincvm.c:2693-2700 wait: Pid -> exit status (raw number).
fn primWait(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const pidv = interp.vaPop(stack);
    if (pidv.tag != .number) return vm.throwShen("wait: pid must be a number");
    var st: c_int = 0;
    _ = waitpid(@truncate(pidv.payload.number), &st, 0);
    acc.* = values.valNumber(execplan.waitStatusCode(@bitCast(st)));
}
