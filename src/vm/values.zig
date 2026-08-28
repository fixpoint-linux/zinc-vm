//! src/vm/values.zig — the ZINC value model (milestone M0).
//!
//! C origin: zincvm.c:157-404 (val_number..val_vector, val_mark/val_prim/
//! val_error, val_stream_in/out, print_value) and zincvm.c:847-955
//! (deep_equal, sv_append / str_value).
//!
//! Port notes:
//!   - Every allocating constructor takes `g: *gc.Gc`.  The C globals become
//!     the passed-in collector (no hidden globals).
//!   - valString(data) CONTRACT (C:161-168): `data` must NOT point into the GC
//!     heap — C copies after GC_STR (gc_alloc_raw).  If data were a GC interior
//!     pointer it would go stale across the alloc; the caller must either use
//!     valStringFrom (slot-rooting) or supply a non-GC buffer.
//!   - valStringFrom (C:169-181) pins src_slot via a ROOT_VALUE so its
//!     str.data survives the allocRaw, then copies from the rooted slot.
//!   - valCons ports C:274-294 verbatim (rootPushValue car, rootPushValue cdr,
//!     alloc car cell, ROOT_PTR the car cell slot across the second alloc,
//!     alloc cdr cell, store, 3 pops).
//!   - valLambda roots &code and &env ptr slots across allocArray, copies,
//!     and sets code AFTER the env alloc (C:299-321 — v is unrooted, a pre-GC
//!     assignment would go stale).
//!   - deep_equal: the C static depth counter becomes an explicit `depth`
//!     parameter with cap 1000.
//!   - print_value / str_value take an `anytype` writer (testable without
//!     stdout).  str_value keeps the depth-100 cap and the [a b c] list form.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;

const Gc = gc.Gc;
const Value = types.Value;

// ---------------------------------------------------------------------
//  Scalar constructors (no GC allocation)
// ---------------------------------------------------------------------

/// C: zincvm.c:157-160 val_number.
pub fn valNumber(n: i64) Value {
    return .{ .tag = .number, .payload = .{ .number = n } };
}

/// M4 val_float — an f64 literal (no GC allocation).
pub fn valFloat(f: f64) Value {
    return .{ .tag = .float, .payload = .{ .float = f } };
}

/// C: zincvm.c:270-273 val_boolean.
pub fn valBoolean(b: bool) Value {
    return .{ .tag = .boolean, .payload = .{ .boolean = @intFromBool(b) } };
}

/// C: zincvm.c:295-298 val_nil.
pub fn valNil() Value {
    return .{ .tag = .nil, .payload = .{ .number = 0 } };
}

/// C: zincvm.c:324-327 val_mark.
pub fn valMark() Value {
    return .{ .tag = .mark, .payload = .{ .number = 0 } };
}

/// C: zincvm.c:328-331 val_prim.  `name` must be a stable, null-terminated
/// literal (C stores a literal string pointer; never GC-allocated).
pub fn valPrim(name: [:0]const u8) Value {
    return .{ .tag = .prim, .payload = .{ .prim = .{ .name = @ptrCast(name.ptr) } } };
}

// ---------------------------------------------------------------------
//  Allocating constructors
// ---------------------------------------------------------------------

/// C: zincvm.c:161-168 val_string.  CONTRACT: `data` must NOT point into the
/// GC heap (see module doc).
pub fn valString(g: *Gc, data: []const u8) Value {
    const dst = g.allocRaw(data.len + 1);
    @memcpy(dst[0..data.len], data);
    dst[data.len] = 0;
    return .{ .tag = .string, .payload = .{ .str = .{ .data = @ptrCast(dst), .len = @intCast(data.len) } } };
}

/// C: zincvm.c:169-181 val_string_from.  Pins `src_slot` so its str.data
/// survives the allocRaw, then copies [off, off+len) from the rooted slot.
pub fn valStringFrom(g: *Gc, src_slot: *Value, off: usize, len: usize) Value {
    var guard = g.rootValue(src_slot);
    defer guard.end();
    const dst = g.allocRaw(len + 1);
    const src = src_slot.payload.str.data.?;
    @memcpy(dst[0..len], src[off .. off + len]);
    dst[len] = 0;
    return .{ .tag = .string, .payload = .{ .str = .{ .data = @ptrCast(dst), .len = @intCast(len) } } };
}

/// M5 PORT-FIX (plan ROOTING observation): slot-rooted copy of a VAL_ERROR's
/// message.  C's error-to-string (zincvm.c:1973) passes the raw error.message
/// pointer into val_string whose GC alloc can move it (stale memcpy) — the Zig
/// port roots the containing value so the message pointer is re-read after the
/// allocation.  The message lives in an allocAtomic block, so only the slot
/// needs pinning.
pub fn valStringFromErr(g: *Gc, src_slot: *Value) Value {
    var guard = g.rootValue(src_slot);
    defer guard.end();
    const len = std.mem.sliceTo(src_slot.payload.error_.message.?, 0).len; // only the scalar len crosses the alloc
    const dst = g.allocRaw(len + 1);
    const src = std.mem.sliceTo(src_slot.payload.error_.message.?, 0); // fresh post-GC
    @memcpy(dst[0..len], src[0..len]);
    dst[len] = 0;
    return .{ .tag = .string, .payload = .{ .str = .{ .data = @ptrCast(dst), .len = @intCast(len) } } };
}

/// C: zincvm.c:274-294 val_cons — ported verbatim (see module doc).
pub fn valCons(g: *Gc, car: Value, cdr: Value) Value {
    var carv = car;
    var cdrv = cdr;
    var g1 = g.rootValue(&carv);
    defer g1.end();
    var g2 = g.rootValue(&cdrv);
    defer g2.end();

    const car_cell = g.alloc(Value);
    var car_root: *Value = car_cell;
    g.rootPushPtr(@ptrCast(&car_root));
    defer g.rootPop(); // car_root

    const cdr_cell = g.alloc(Value);
    car_root.* = carv;
    cdr_cell.* = cdrv;

    return .{ .tag = .cons, .payload = .{ .cons = .{ .car = car_root, .cdr = cdr_cell } } };
}

/// C: zincvm.c:299-321 val_lambda — roots &code and &env across the
/// allocArray, copies, and sets code AFTER the env alloc (module doc).
pub fn valLambda(
    g: *Gc,
    code: ?*types.Instr,
    code_len: i32,
    env: ?[*]Value,
    env_len: i32,
) Value {
    var v: Value = .{
        .tag = .lambda,
        .payload = .{ .lambda = .{ .code = null, .code_len = code_len, .env = null, .env_len = 0 } },
    };
    if (env_len > 0) {
        var code_root: ?*types.Instr = code;
        var env_root: ?[*]Value = env;
        g.rootPushPtr(@ptrCast(&code_root));
        g.rootPushPtr(@ptrCast(&env_root));
        defer g.rootPop(); // env_root
        defer g.rootPop(); // code_root

        const len: usize = @intCast(env_len);
        const env_copy = g.allocArray(Value, len);
        @memcpy(env_copy[0..len], env_root.?[0..len]);

        // Write barrier: if env_copy landed in old-gen and references
        // the nursery, record it in the remembered set so the full
        // collector scans it (same pattern as interp.zig env build).
        if (g.inOldgen(@intFromPtr(env_copy))) {
            var j: usize = 0;
            while (j < len) : (j += 1) {
                if (gc.scan.valueReferencesNursery(g, &env_copy[j])) {
                    g.dirtyVectorsAdd(env_copy);
                    break;
                }
            }
        }

        v.payload.lambda.env = env_copy;
        v.payload.lambda.env_len = env_len;
        v.payload.lambda.code = code_root;
    } else {
        v.payload.lambda.env = null;
        v.payload.lambda.env_len = 0;
        v.payload.lambda.code = code;
    }
    return v;
}

/// C: zincvm.c:332-342 val_error.  GC-allocate the message so it is reclaimed
/// with the collector instead of leaking on every raised error.
pub fn valError(g: *Gc, msg: []const u8) Value {
    const buf = g.allocAtomic(msg.len + 1);
    @memcpy(buf[0..msg.len], msg);
    buf[msg.len] = 0;
    return .{ .tag = .error_, .payload = .{ .error_ = .{ .message = @ptrCast(buf) } } };
}

/// C: zincvm.c:343-348 val_vector.
pub fn valVector(g: *Gc, size: i32) Value {
    var v: Value = .{ .tag = .vector, .payload = .{ .vector = .{ .data = null, .len = size } } };
    if (size > 0) v.payload.vector.data = g.allocArray(Value, @intCast(size));
    return v;
}

/// C: zincvm.c:349-352 val_stream_in (plan DECISION D: value model only; the
/// stream I/O prims + val_string_stream_in are deferred to the I/O milestone).
pub fn valStreamIn(file: ?*anyopaque) Value {
    return .{ .tag = .stream, .payload = .{ .stream = .{ .file = file, .is_input = 1, .is_string = 0 } } };
}

/// C: zincvm.c:353-356 val_stream_out.
pub fn valStreamOut(file: ?*anyopaque) Value {
    return .{ .tag = .stream, .payload = .{ .stream = .{ .file = file, .is_input = 0, .is_string = 0 } } };
}

// ---------------------------------------------------------------------
//  Interior accessors
// ---------------------------------------------------------------------

/// Slice view of a VAL_STRING's data.
pub fn strSlice(v: Value) []const u8 {
    return v.payload.str.data.?[0..@intCast(v.payload.str.len)];
}

/// Slice view of a VAL_SYMBOL's (null-terminated) name.
pub fn symSlice(v: Value) []const u8 {
    return std.mem.sliceTo(v.payload.sym.name.?, 0);
}

/// Slice view of a VAL_PRIM's (null-terminated) name.
pub fn primSlice(v: Value) []const u8 {
    return std.mem.sliceTo(v.payload.prim.name.?, 0);
}

/// Slice view of a VAL_ERROR's (null-terminated) message.
pub fn errSlice(v: Value) []const u8 {
    return std.mem.sliceTo(v.payload.error_.message.?, 0);
}

// ---------------------------------------------------------------------
//  Printing — C: zincvm.c:383-404 print_value, 901-955 str_value
// ---------------------------------------------------------------------

/// M4 float renderer — Elm String.fromFloat parity: NaN/±Infinity as the
/// canonical tokens, finite as Zig `{d}` shortest-round-trip decimal, with a
/// trailing ".0" appended when the text has no '.'/'e'/'E' (so 2.0 prints
/// "2.0", DISTINGUISHABLE from Int "2").  Returns either a static literal or a
/// slice of `buf`; shared by printValue/strValue (and primStr via pub).
pub fn floatText(buf: []u8, v: f64) []const u8 {
    if (std.math.isNan(v)) return "NaN";
    if (std.math.isPositiveInf(v)) return "Infinity";
    if (std.math.isNegativeInf(v)) return "-Infinity";
    const s = std.fmt.bufPrint(buf, "{d}", .{v}) catch unreachable;
    if (std.mem.indexOfAny(u8, s, ".eE") == null) {
        const n = s.len;
        if (n + 2 <= buf.len) {
            buf[n] = '.';
            buf[n + 1] = '0';
            return buf[0 .. n + 2];
        }
    }
    return s;
}

/// C: zincvm.c:383-404 print_value — full printed form (cons as [cons X . Y]).
/// `writer` is any `std.io` writer (anytype); testable without stdout.
pub fn printValue(writer: anytype, v: Value) !void {
    switch (v.tag) {
        .number => try writer.print("{d}", .{v.payload.number}),
        .string => try writer.print("\"{s}\"", .{strSlice(v)}),
        .symbol => try writer.print("{s}", .{symSlice(v)}),
        .boolean => try writer.writeAll(if (v.payload.boolean != 0) "true" else "false"),
        .cons => {
            try writer.writeAll("[cons ");
            try printValue(writer, v.payload.cons.car.?.*);
            try writer.writeAll(" . ");
            try printValue(writer, v.payload.cons.cdr.?.*);
            try writer.writeAll("]");
        },
        .nil => try writer.writeAll("[]"),
        .lambda => try writer.print("[lambda {any} {d} env={any} {d}]", .{
            v.payload.lambda.code, v.payload.lambda.code_len,
            v.payload.lambda.env,  v.payload.lambda.env_len,
        }),
        .mark => try writer.writeAll("mark"),
        .prim => try writer.print("[prim {s}]", .{primSlice(v)}),
        .error_ => try writer.print("[error \"{s}\"]", .{errSlice(v)}),
        .vector => try writer.print("[vector {d}]", .{v.payload.vector.len}),
        .stream => try writer.print("[stream {s}]", .{if (v.payload.stream.is_input != 0) "in" else "out"}),
        .float => {
            var buf: [64]u8 = undefined;
            try writer.writeAll(floatText(&buf, v.payload.float));
        },
    }
}

/// C: zincvm.c:901-955 str_value — the `str` primitive's representation:
/// [a b c] list form, depth-100 cap, `<...>` forms for the opaque types.
/// The C bounds-safe sv_append (C:890-899) becomes an unbounded writer; the
/// depth cap is the sole recursion guard (the C also relied on the buffer).
pub fn strValue(writer: anytype, v: Value, depth: u32) !void {
    if (depth > 100) {
        try writer.writeAll("...");
        return;
    }
    switch (v.tag) {
        .symbol => try writer.print("{s}", .{symSlice(v)}),
        .string => try writer.print("\"{s}\"", .{strSlice(v)}),
        .number => try writer.print("{d}", .{v.payload.number}),
        .boolean => try writer.writeAll(if (v.payload.boolean != 0) "true" else "false"),
        .nil => try writer.writeAll("[]"),
        .cons => {
            var cur: *const Value = &v;
            var first = true;
            try writer.writeAll("[");
            while (cur.tag == .cons) : (cur = cur.payload.cons.cdr.?) {
                if (!first) try writer.writeAll(" ");
                first = false;
                try strValue(writer, cur.payload.cons.car.?.*, depth + 1);
            }
            if (cur.tag != .nil) {
                try writer.writeAll(" . ");
                try strValue(writer, cur.*, depth + 1);
            }
            try writer.writeAll("]");
        },
        .error_ => try writer.print("<error {s}>", .{errSlice(v)}),
        .lambda => try writer.writeAll("<lambda>"),
        .prim => try writer.print("<prim {s}>", .{primSlice(v)}),
        .vector => try writer.print("<vector {d}>", .{v.payload.vector.len}),
        .stream => try writer.writeAll("<stream>"),
        .float => {
            var buf: [64]u8 = undefined;
            try writer.writeAll(floatText(&buf, v.payload.float));
        },
        else => try writer.writeAll("<unknown>"),
    }
}

// ---------------------------------------------------------------------
//  Structural equality — C: zincvm.c:847-881 deep_equal
// ---------------------------------------------------------------------

/// C: zincvm.c:847-881 deep_equal.  The C static depth counter becomes an
/// explicit `depth` parameter (cap 1000) guarding against infinite recursion
/// on cyclic structures.
pub fn deepEqual(a: Value, b: Value, depth: u32) bool {
    if (depth > 1000) return false;
    if (a.tag != b.tag) return false;
    switch (a.tag) {
        .number => return a.payload.number == b.payload.number,
        .string => return a.payload.str.len == b.payload.str.len and
            std.mem.eql(u8, strSlice(a), strSlice(b)),
        .symbol => return std.mem.eql(u8, symSlice(a), symSlice(b)),
        .boolean => return a.payload.boolean == b.payload.boolean,
        .float => return a.payload.float == b.payload.float,
        .nil => return true,
        .cons => return deepEqual(a.payload.cons.car.?.*, b.payload.cons.car.?.*, depth + 1) and
            deepEqual(a.payload.cons.cdr.?.*, b.payload.cons.cdr.?.*, depth + 1),
        .vector => {
            if (a.payload.vector.len != b.payload.vector.len) return false;
            var i: usize = 0;
            const n: usize = @intCast(a.payload.vector.len);
            while (i < n) : (i += 1) {
                if (!deepEqual(a.payload.vector.data.?[i], b.payload.vector.data.?[i], depth + 1))
                    return false;
            }
            return true;
        },
        else => return false,
    }
}
