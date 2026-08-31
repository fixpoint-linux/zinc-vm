//! tests/vm_test.zig — M0 (value model) + M1 (symbol interner) for the ZINC VM.
//!
//! M0 covers the ported value constructors, deep_equal, print_value / str_value,
//! val_string_from, the Vm skeleton rooting, and a forced-scavenge cons-chain
//! survival test (uses only the GC API).  M1 covers the symbol interner
//! (identity, distinctness, resize survival, val_symbol).

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const heap = gc.heap;
const vm = @import("vm");
const values = vm.values;
const symbols = vm.symbols;
const state = vm.state;
const tables = vm.tables;
const parser = vm.parser;
const prims = vm.prims;

/// 16 MB heap (the C minimum) with a 64 MB reservation (avoids the 4 GB
/// default VAS), mirroring tests/gc_test.zig's testInit.
fn testInit() !heap.Gc {
    return heap.Gc.init(.{
        .heap_bytes = 16 * 1024 * 1024,
        .reserve_bytes = 64 * 1024 * 1024,
    });
}

/// Build a cons list [n0 n1 ... nk] of numbers (nil-terminated).
fn consNums(g: *heap.Gc, nums: []const i64) types.Value {
    var head = values.valNil();
    var i = nums.len;
    while (i > 0) {
        i -= 1;
        head = values.valCons(g, values.valNumber(nums[i]), head);
    }
    return head;
}

/// str_value / print_value a Value into a fixed module-level buffer and return
/// the written slice.  A single shared buffer is safe here because the tests
/// are sequential and each result is consumed immediately (within the same
/// expression or enclosing scope) before the next call overwrites it.
var fmtBuf: [16384]u8 = undefined;

fn strValueOf(v: types.Value) []const u8 {
    var w: std.Io.Writer = .fixed(&fmtBuf);
    values.strValue(&w, v, 0) catch unreachable;
    return w.buffered();
}

fn printValueOf(v: types.Value) []const u8 {
    var w: std.Io.Writer = .fixed(&fmtBuf);
    values.printValue(&w, v) catch unreachable;
    return w.buffered();
}

// =====================================================================
//  M0 — value round-trips
// =====================================================================

test "M0 scalar value round-trips" {
    var g = try testInit();
    defer g.deinit();

    const n = values.valNumber(42);
    try std.testing.expectEqual(types.ValTag.number, n.tag);
    try std.testing.expectEqual(@as(i64, 42), n.payload.number);

    const t = values.valBoolean(true);
    const f = values.valBoolean(false);
    try std.testing.expectEqual(types.ValTag.boolean, t.tag);
    try std.testing.expectEqual(@as(i32, 1), t.payload.boolean);
    try std.testing.expectEqual(@as(i32, 0), f.payload.boolean);

    try std.testing.expectEqual(types.ValTag.nil, values.valNil().tag);
    try std.testing.expectEqual(types.ValTag.mark, values.valMark().tag);

    const p = values.valPrim("reverse");
    try std.testing.expectEqual(types.ValTag.prim, p.tag);
    try std.testing.expect(std.mem.eql(u8, "reverse", values.primSlice(p)));

    const s = values.valString(&g, "hello");
    try std.testing.expectEqual(types.ValTag.string, s.tag);
    try std.testing.expectEqual(@as(i32, 5), s.payload.str.len);
    try std.testing.expect(std.mem.eql(u8, "hello", values.strSlice(s)));
}

test "M0 valStringFrom slices a rooted string, survives a scavenge" {
    var g = try testInit();
    defer g.deinit();

    var s = values.valString(&g, "hello world");
    var guard = g.rootValue(&s);
    defer guard.end();

    // Promote s to old-gen, then read through the still-rooted slot.
    g.collectNursery(.@"test");

    const sub = values.valStringFrom(&g, &s, 6, 5);
    try std.testing.expectEqual(types.ValTag.string, sub.tag);
    try std.testing.expectEqual(@as(i32, 5), sub.payload.str.len);
    try std.testing.expect(std.mem.eql(u8, "world", values.strSlice(sub)));
}

test "M0 cons list round-trip + deep equality" {
    var g = try testInit();
    defer g.deinit();

    const a = consNums(&g, &.{ 1, 2, 3 });
    const b = consNums(&g, &.{ 1, 2, 3 });
    const c = consNums(&g, &.{ 1, 2, 4 });

    try std.testing.expect(values.deepEqual(a, b, 0));
    try std.testing.expect(!values.deepEqual(a, c, 0));

    // Cons head cells are distinct objects even for equal lists.
    try std.testing.expect(a.payload.cons.car.? != b.payload.cons.car.?);
}

test "M0 deep_equal over vectors / strings / symbols / booleans" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    // Vectors.
    var va = values.valVector(&g, 3);
    va.payload.vector.data.?[0] = values.valNumber(1);
    va.payload.vector.data.?[1] = values.valNumber(2);
    va.payload.vector.data.?[2] = values.valNumber(3);
    var vb = values.valVector(&g, 3);
    vb.payload.vector.data.?[0] = values.valNumber(1);
    vb.payload.vector.data.?[1] = values.valNumber(2);
    vb.payload.vector.data.?[2] = values.valNumber(3);
    var vc = values.valVector(&g, 3);
    vc.payload.vector.data.?[0] = values.valNumber(1);
    vc.payload.vector.data.?[1] = values.valNumber(2);
    vc.payload.vector.data.?[2] = values.valNumber(9);
    try std.testing.expect(values.deepEqual(va, vb, 0));
    try std.testing.expect(!values.deepEqual(va, vc, 0));

    // Strings.
    try std.testing.expect(values.deepEqual(
        values.valString(&g, "same"),
        values.valString(&g, "same"),
        0,
    ));
    try std.testing.expect(!values.deepEqual(
        values.valString(&g, "same"),
        values.valString(&g, "diff"),
        0,
    ));

    // Symbols (interner-backed canonical names).
    const sy_a = symbols.valSymbol(&sym, "alpha");
    const sy_b = symbols.valSymbol(&sym, "alpha");
    const sy_c = symbols.valSymbol(&sym, "beta");
    try std.testing.expect(values.deepEqual(sy_a, sy_b, 0));
    try std.testing.expect(!values.deepEqual(sy_a, sy_c, 0));

    // Booleans.
    try std.testing.expect(values.deepEqual(values.valBoolean(true), values.valBoolean(true), 0));
    try std.testing.expect(!values.deepEqual(values.valBoolean(true), values.valBoolean(false), 0));
    try std.testing.expect(values.deepEqual(values.valNil(), values.valNil(), 0));

    // Mixed tags never compare equal.
    try std.testing.expect(!values.deepEqual(values.valNumber(1), values.valString(&g, "1"), 0));
}

test "M0 valLambda copies env and sets code after the env alloc" {
    var g = try testInit();
    defer g.deinit();

    // A small GC-allocated env array to copy.
    var env = g.allocArray(types.Value, 2);
    env[0] = values.valNumber(7);
    env[1] = values.valNumber(8);

    const lam = values.valLambda(&g, null, 0, env, 2);
    try std.testing.expectEqual(types.ValTag.lambda, lam.tag);
    try std.testing.expectEqual(@as(i32, 2), lam.payload.lambda.env_len);
    try std.testing.expect(lam.payload.lambda.env != null);
    try std.testing.expect(values.deepEqual(values.valNumber(7), lam.payload.lambda.env.?[0], 0));
    try std.testing.expect(values.deepEqual(values.valNumber(8), lam.payload.lambda.env.?[1], 0));

    // env_len == 0 -> null env, code still set.
    const lam0 = values.valLambda(&g, null, 0, null, 0);
    try std.testing.expect(lam0.payload.lambda.env == null);
    try std.testing.expectEqual(@as(i32, 0), lam0.payload.lambda.env_len);
}

test "M0 valError / valVector / valStream constructors" {
    var g = try testInit();
    defer g.deinit();

    const e = values.valError(&g, "boom");
    try std.testing.expectEqual(types.ValTag.error_, e.tag);
    try std.testing.expect(std.mem.eql(u8, "boom", values.errSlice(e)));

    const v = values.valVector(&g, 4);
    try std.testing.expectEqual(types.ValTag.vector, v.tag);
    try std.testing.expectEqual(@as(i32, 4), v.payload.vector.len);
    try std.testing.expect(v.payload.vector.data != null);
    try std.testing.expect(values.valVector(&g, 0).payload.vector.data == null);

    const si = values.valStreamIn(null);
    try std.testing.expectEqual(types.ValTag.stream, si.tag);
    try std.testing.expectEqual(@as(i32, 1), si.payload.stream.is_input);
    const so = values.valStreamOut(null);
    try std.testing.expectEqual(@as(i32, 0), so.payload.stream.is_input);
}

// =====================================================================
//  M0 — printing
// =====================================================================

test "M0 print_value golden strings" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    try std.testing.expectEqualStrings("42", printValueOf(values.valNumber(42)));
    try std.testing.expectEqualStrings("true", printValueOf(values.valBoolean(true)));
    try std.testing.expectEqualStrings("false", printValueOf(values.valBoolean(false)));
    try std.testing.expectEqualStrings("\"hi\"", printValueOf(values.valString(&g, "hi")));
    try std.testing.expectEqualStrings("abc", printValueOf(symbols.valSymbol(&sym, "abc")));
    try std.testing.expectEqualStrings("[]", printValueOf(values.valNil()));
    try std.testing.expectEqualStrings("mark", printValueOf(values.valMark()));
    try std.testing.expectEqualStrings("[prim reverse]", printValueOf(values.valPrim("reverse")));
    try std.testing.expectEqualStrings("[error \"boom\"]", printValueOf(values.valError(&g, "boom")));
    try std.testing.expectEqualStrings("[vector 3]", printValueOf(values.valVector(&g, 3)));
    try std.testing.expectEqualStrings("[stream in]", printValueOf(values.valStreamIn(null)));
    try std.testing.expectEqualStrings(
        "[cons 1 . [cons 2 . []]]",
        printValueOf(consNums(&g, &.{ 1, 2 })),
    );
}

test "M0 str_value list form and opaque forms" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    try std.testing.expectEqualStrings("42", strValueOf(values.valNumber(42)));
    try std.testing.expectEqualStrings("[1 2 3]", strValueOf(consNums(&g, &.{ 1, 2, 3 })));
    try std.testing.expectEqualStrings("[]", strValueOf(values.valNil()));
    try std.testing.expectEqualStrings("\"x\"", strValueOf(values.valString(&g, "x")));
    try std.testing.expectEqualStrings("<vector 3>", strValueOf(values.valVector(&g, 3)));
    try std.testing.expectEqualStrings("<prim reverse>", strValueOf(values.valPrim("reverse")));
    try std.testing.expectEqualStrings("<lambda>", strValueOf(values.valLambda(&g, null, 0, null, 0)));
    try std.testing.expectEqualStrings("<stream>", strValueOf(values.valStreamIn(null)));
    try std.testing.expectEqualStrings("<error boom>", strValueOf(values.valError(&g, "boom")));
    try std.testing.expectEqualStrings("sym1", strValueOf(symbols.valSymbol(&sym, "sym1")));

    // A dotted tail renders with " . ".
    const dotted = values.valCons(&g, values.valNumber(1), values.valNumber(2));
    try std.testing.expectEqualStrings("[1 . 2]", strValueOf(dotted));
}

test "M0 str_value of a long list exceeds 4096 chars (Test 14c shape)" {
    var g = try testInit();
    defer g.deinit();

    // 2000 cons cells -> >4096 chars of "[1 2 3 ... 2000]".
    var nums: [2000]i64 = undefined;
    var i: usize = 0;
    while (i < nums.len) : (i += 1) nums[i] = @intCast(i + 1);
    const list = consNums(&g, &nums);
    const s = strValueOf(list);
    try std.testing.expect(s.len > 4096);
    // Bracket-balanced and correctly framed.
    try std.testing.expectEqual(@as(u8, '['), s[0]);
    try std.testing.expectEqual(@as(u8, ']'), s[s.len - 1]);
}

// =====================================================================
//  M4 — floats: val_float tag, print/str goldens, deep_equal semantics
// =====================================================================

test "M4 valFloat tag + print_value/str_value goldens" {
    var g = try testInit();
    defer g.deinit();

    const f = values.valFloat(2.0);
    try std.testing.expectEqual(types.ValTag.float, f.tag);
    try std.testing.expectEqual(@as(f64, 2.0), f.payload.float);

    try std.testing.expectEqualStrings("2.0", printValueOf(values.valFloat(2.0)));
    try std.testing.expectEqualStrings("1.5", printValueOf(values.valFloat(1.5)));
    try std.testing.expectEqualStrings("-0.5", printValueOf(values.valFloat(-0.5)));
    try std.testing.expectEqualStrings("NaN", printValueOf(values.valFloat(std.math.nan(f64))));
    try std.testing.expectEqualStrings("Infinity", printValueOf(values.valFloat(std.math.inf(f64))));
    try std.testing.expectEqualStrings("-Infinity", printValueOf(values.valFloat(-std.math.inf(f64))));
    try std.testing.expectEqualStrings("2.0", strValueOf(values.valFloat(2.0)));
}

test "M4 deep_equal float semantics" {
    try std.testing.expect(values.deepEqual(values.valFloat(1.5), values.valFloat(1.5), 0));
    try std.testing.expect(!values.deepEqual(values.valFloat(1.5), values.valFloat(2.5), 0));
    // IEEE: NaN != NaN.
    try std.testing.expect(!values.deepEqual(values.valFloat(std.math.nan(f64)), values.valFloat(std.math.nan(f64)), 0));
    // Int 2 vs Float 2.0 are UNEQUAL under deepEqual (tag-distinct early
    // return) — structural equality does NOT promote (decision 3, unchanged;
    // only scalar =/</= via primEq promotes per the M4 review fix-1).
    try std.testing.expect(!values.deepEqual(values.valNumber(2), values.valFloat(2.0), 0));
}

// =====================================================================
//  M0 — forced-scavenge survival (uses only the GC API)
// =====================================================================

test "M0 cons chain survives a forced nursery scavenge" {
    var g = try testInit();
    defer g.deinit();

    var head = consNums(&g, &.{ 1, 2, 3 });
    const wm0 = g.rootWatermark();
    g.rootPushValue(&head); // root head across the scavenge

    const head_old = @intFromPtr(head.payload.cons.car.?);
    try std.testing.expect(g.inNursery(head_old));

    g.collectNursery(.@"test");

    // Interior pointers rewritten; contents intact.
    try std.testing.expect(!g.inNursery(@intFromPtr(head.payload.cons.car.?)));

    const expected = consNums(&g, &.{ 1, 2, 3 });
    const equal = values.deepEqual(head, expected, 0);

    g.rootPop(); // unroot head
    try std.testing.expect(equal);

    // Root stack balanced across the whole block.
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

// =====================================================================
//  M0 — Vm skeleton
// =====================================================================

test "M0 Vm skeleton roots err_slot once" {
    var g = try testInit();
    defer g.deinit();

    const wm = g.rootWatermark();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // err_slot (1) + the M10 frame-pool slot roots (FRAME_POOL_MAX) + the M12
    // stack-pool slot roots (STACK_POOL_MAX).
    try std.testing.expectEqual(wm + 1 + state.FRAME_POOL_MAX + state.STACK_POOL_MAX, g.rootWatermark());
    try std.testing.expectEqual(@as(*heap.Gc, &g), v.gc);
    try std.testing.expectEqual(types.ValTag.nil, v.err_slot.tag);
    // The pools start empty with no instrumentation yet.
    try std.testing.expectEqual(@as(usize, 0), v.frame_pool_live);
    try std.testing.expectEqual(@as(u64, 0), v.frame_pool_hits);
    try std.testing.expectEqual(@as(u64, 0), v.frame_pool_misses);
    try std.testing.expectEqual(@as(usize, 0), v.stack_pool_live);
    try std.testing.expectEqual(@as(u64, 0), v.stack_pool_hits);
    try std.testing.expectEqual(@as(u64, 0), v.stack_pool_misses);
}

// =====================================================================
//  M1 — symbol interner
// =====================================================================

test "M1 intern identity and distinctness" {
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const a1 = sym.intern("alpha");
    const a2 = sym.intern("alpha");
    const b = sym.intern("beta");

    try std.testing.expect(a1 == a2); // canonical pointer identity
    try std.testing.expect(a1 != b);
    try std.testing.expect(std.mem.eql(u8, "alpha", std.mem.sliceTo(a1, 0)));
}

test "M1 interning past 70% load grows and keeps old pointers canonical" {
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    var names: [180][24]u8 = undefined;
    var ptrs: [180][*:0]const u8 = undefined;
    var i: usize = 0;
    while (i < 180) : (i += 1) {
        const used = std.fmt.bufPrint(&names[i], "sym{d}", .{i}) catch unreachable;
        // bufPrint does not NUL-terminate; write one so the re-intern below can
        // hand back the whole `names[i]` buffer as a C-style name (mirrors C's
        // const char* contract and the sliceTo normalization in intern).
        names[i][used.len] = 0;
        ptrs[i] = sym.intern(used);
    }
    try std.testing.expectEqual(@as(usize, 180), sym.count);
    try std.testing.expectEqual(@as(usize, 256), sym.cap);

    // Intern well past the 70% threshold to force at least one x2 grow.
    var j: usize = 180;
    while (j < 400) : (j += 1) {
        var buf: [24]u8 = undefined;
        const nm = std.fmt.bufPrint(&buf, "sym{d}", .{j}) catch unreachable;
        _ = sym.intern(nm);
    }
    try std.testing.expect(sym.cap >= 512);
    try std.testing.expectEqual(@as(usize, 400), sym.count);

    // Every pre-resize pointer is still the canonical answer after the rehash.
    i = 0;
    while (i < 180) : (i += 1) {
        const again = sym.intern(&names[i]);
        try std.testing.expect(again == ptrs[i]);
    }
}

test "M1 valSymbol tag and name" {
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const v1 = symbols.valSymbol(&sym, "hello");
    const v2 = symbols.valSymbol(&sym, "hello");
    try std.testing.expectEqual(types.ValTag.symbol, v1.tag);
    try std.testing.expectEqual(types.ValTag.symbol, v2.tag);
    try std.testing.expect(std.mem.eql(u8, "hello", values.symSlice(v1)));

    // Canonical pointer equality: same name -> same interned name pointer.
    try std.testing.expect(v1.payload.sym.name == v2.payload.sym.name);
    try std.testing.expect(values.deepEqual(v1, v2, 0));
}

// =====================================================================
//  M2 — defun/values tables + GC registration
// =====================================================================

test "M2 defun set/get/overwrite/has" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    v.defunSet("f", values.valNumber(1));
    try std.testing.expectEqual(@as(i64, 1), v.defunGet("f").payload.number);
    try std.testing.expect(v.defunHas("f"));

    // Later store wins on overwrite (C: "later store wins").
    v.defunSet("f", values.valNumber(2));
    try std.testing.expectEqual(@as(i64, 2), v.defunGet("f").payload.number);

    try std.testing.expect(!v.defunHas("nope"));
    try std.testing.expect(!v.defunHas("absent"));
}

test "M2 values set/get/overwrite" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    v.valueSet("x", values.valNumber(10));
    try std.testing.expectEqual(@as(i64, 10), v.valueGet("x").payload.number);
    v.valueSet("x", values.valNumber(99)); // overwrite
    try std.testing.expectEqual(@as(i64, 99), v.valueGet("x").payload.number);

    // Different keys don't collide into each other's values.
    v.valueSet("y", values.valNumber(20));
    try std.testing.expectEqual(@as(i64, 99), v.valueGet("x").payload.number);
    try std.testing.expectEqual(@as(i64, 20), v.valueGet("y").payload.number);
}

test "M2 defun/value fallback: unknown names stay symbols" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // Unknown defun name -> valSymbol (macros/*stinput* must stay symbols).
    const gv = v.defunGet("*macros*");
    try std.testing.expectEqual(types.ValTag.symbol, gv.tag);
    try std.testing.expect(std.mem.eql(u8, "*macros*", values.symSlice(gv)));

    // value_get fallback is always val_symbol, no prim fallback (C:667).
    const vg = v.valueGet("+");
    try std.testing.expectEqual(types.ValTag.symbol, vg.tag);
    try std.testing.expectEqual(types.ValTag.symbol, v.valueGet("missing").tag);

    // initGlobals registers the stub prim list as VAL_PRIM globals.
    const plus = v.defunGet("+");
    try std.testing.expectEqual(types.ValTag.prim, plus.tag);
    try std.testing.expect(std.mem.eql(u8, "+", values.primSlice(plus)));
}

test "M2 defunSet marks the slot dirty (nursery value)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    const slot = tables.hashName("dirtytest", tables.DEFUN_TABLE_CAP);
    try std.testing.expect(!g.dirtyDefunsTest(@intCast(slot)));

    const nv = values.valCons(&g, values.valNumber(5), values.valNil());
    v.defunSet("dirtytest", nv);
    try std.testing.expect(g.dirtyDefunsTest(@intCast(slot)));

    // Overwrite also re-marks dirty (may now reference a nursery value).
    v.defunSet("dirtytest", values.valNumber(9));
    try std.testing.expect(g.dirtyDefunsTest(@intCast(slot)));
}

test "M2 defun table cons list survives scavenge + full collect (interior rewrite)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    const list = consNums(&g, &.{ 1, 2, 3 });
    // ALSO hold it via a root so root and table can be compared independently.
    var root_list = list;
    g.rootPushValue(&root_list);
    defer g.rootPop();

    v.defunSet("mylist", list);

    try std.testing.expect(values.deepEqual(v.defunGet("mylist"), root_list, 0));

    g.collectNursery(.@"test");
    g.collect(.@"test"); // full collect

    // Table value and root agree, with interior pointers rewritten in place.
    const got = v.defunGet("mylist");
    try std.testing.expect(values.deepEqual(got, root_list, 0));
    try std.testing.expectEqual(@as(i64, 1), got.payload.cons.car.?.payload.number);
    try std.testing.expectEqual(@as(i64, 2), got.payload.cons.cdr.?.payload.cons.car.?.payload.number);
}

test "M2 values table survives a full collect" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    v.valueSet("num", values.valNumber(42));
    v.valueSet("lst", consNums(&g, &.{ 7, 8 }));

    g.collect(.@"test"); // values table is always full-scanned

    try std.testing.expectEqual(@as(i64, 42), v.valueGet("num").payload.number);
    const lst = v.valueGet("lst");
    try std.testing.expect(values.deepEqual(consNums(&g, &.{ 7, 8 }), lst, 0));
    try std.testing.expectEqual(@as(i64, 7), lst.payload.cons.car.?.payload.number);
}

// =====================================================================
//  M3 — csexp parser + resolve_jumps + print_instr
// =====================================================================

/// print_instr a code array into the shared fixed buffer (testable without
/// stdout; single buffer safe because tests are sequential).
fn printInstrOf(code: [*]types.Instr, len: i32) []const u8 {
    var w: std.Io.Writer = .fixed(&fmtBuf);
    parser.printInstr(&w, code, len, 0) catch unreachable;
    return w.buffered();
}

test "M3 parse test-2 [lambda X X] structure" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(c(a[1:n]0v))", &code);
    try std.testing.expectEqual(@as(i32, 1), len);
    try std.testing.expectEqual(types.Opcode.cur, code.?[0].op);
    try std.testing.expectEqual(@as(i32, 2), code.?[0].closure_len);
    const child: [*]types.Instr = @ptrCast(code.?[0].closure_code.?);
    try std.testing.expectEqual(types.Opcode.access, child[0].op);
    try std.testing.expectEqual(@as(i64, 0), child[0].operand.payload.number);
    try std.testing.expectEqual(types.Opcode.ret, child[1].op);
}

test "M3 parse+print round-trip of zinctest built-in literals" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    // Test 1: [+ 1 2]
    var code1: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(mn[1:n]2n[1:n]1g[1:s]+p)", &code1);
    try std.testing.expectEqualStrings("pushmark\nnumber 2\nnumber 1\nglobal +\napply\n", printInstrOf(code1.?, 5));

    // Test 2: [lambda X X]
    var code2: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(c(a[1:n]0v))", &code2);
    try std.testing.expectEqualStrings("cur (code=2):\n  access 0\n  return\nendcur\n", printInstrOf(code2.?, 1));

    // Test 15: [string? "hi"]
    var code3: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(mS[2:S]hig[7:s]string?p)", &code3);
    try std.testing.expectEqualStrings("pushmark\nstring \"hi\"\nglobal string?\napply\n", printInstrOf(code3.?, 4));
}

test "M3 nested cur parse (test 36 appterm-in-apply)" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    // (mn[2:n]42c(ma[1:n]0c(a[1:n]0v)t)p)
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(mn[2:n]42c(ma[1:n]0c(a[1:n]0v)t)p)", &code);
    try std.testing.expectEqual(@as(i32, 4), len);

    const inner = code.?[2]; // the cur instr
    try std.testing.expectEqual(types.Opcode.cur, inner.op);
    try std.testing.expectEqual(@as(i32, 4), inner.closure_len);
    // inner body: m (pushmark), a 0 (access), c (...) cur, t (appterm)
    const inner_code: [*]types.Instr = @ptrCast(inner.closure_code.?);
    try std.testing.expectEqual(types.Opcode.pushmark, inner_code[0].op);
    try std.testing.expectEqual(types.Opcode.access, inner_code[1].op);
    try std.testing.expectEqual(types.Opcode.cur, inner_code[2].op);
    try std.testing.expectEqual(types.Opcode.appterm, inner_code[3].op);
    // innermost cur body: a 0, v
    const innermost: [*]types.Instr = @ptrCast(inner_code[2].closure_code.?);
    try std.testing.expectEqual(@as(i32, 2), inner_code[2].closure_len);
    try std.testing.expectEqual(types.Opcode.access, innermost[0].op);
    try std.testing.expectEqual(types.Opcode.ret, innermost[1].op);
}

test "M3 resolve_jumps fills jmp_target from numeric operands" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    // f[1:n]0j[1:n]2a[1:n]1  → jmpf 0, jmp 2, access 1
    var code: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(f[1:n]0j[1:n]2a[1:n]1)", &code);
    parser.resolveJumps(code.?, 3);
    try std.testing.expectEqual(@as(i32, 0), code.?[0].jmp_target);
    try std.testing.expectEqual(@as(i32, 2), code.?[1].jmp_target);
    try std.testing.expectEqual(@as(i32, 1), code.?[2].jmp_target);

    // Non-numeric operand -> jmp_target 0.
    var code2: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(f[2:s]xx)", &code2);
    parser.resolveJumps(code2.?, 1);
    try std.testing.expectEqual(@as(i32, 0), code2.?[0].jmp_target);
}

test "M3 successful parse leaves shadow stack balanced" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const wm0 = g.rootWatermark();
    var code: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(c(a[1:n]0c(a[1:n]0v)t)v)", &code);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M3 error cases return ParseError and leave shadow stack balanced" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const bad = [_][:0]const u8{
        "(x)", // unknown opcode
        "(n[1:n]1", // missing closing bracket
        "((n[1:n]1))", // unexpected nested list in body
        "(n[abc])", // bad atom (no digits / bad format)
        "nope", // not a list
    };
    for (bad) |src| {
        const wm0 = g.rootWatermark();
        var code: ?[*]types.Instr = null;
        try std.testing.expectError(error.ParseError, parser.parseBytecode(&g, &sym, src, &code));
        try std.testing.expect(code == null);
        // Shadow stack restored to entry watermark (C root_pop_to parity).
        try std.testing.expectEqual(wm0, g.rootWatermark());
    }
}

test "M3 parsed closure survives scavenge + full collect (children reachable)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    var code: ?[*]types.Instr = null;
    _ = try parser.parseBytecode(&g, &sym, "(c(a[1:n]0c(a[1:n]0v)t)v)", &code);
    // Root the outer code array so the closure children stay reachable.
    g.rootPushPtr(@ptrCast(&code));
    defer g.rootPop();

    g.collectNursery(.@"test");
    g.collect(.@"test");

    // Interior pointers rewritten; both levels still valid and callable.
    try std.testing.expectEqual(types.Opcode.cur, code.?[0].op);
    const inner: [*]types.Instr = @ptrCast(code.?[0].closure_code.?);
    try std.testing.expectEqual(types.Opcode.cur, inner[1].op);
    const innermost: [*]types.Instr = @ptrCast(inner[1].closure_code.?);
    try std.testing.expectEqual(types.Opcode.access, innermost[0].op);
    try std.testing.expectEqual(types.Opcode.ret, innermost[1].op);
}

// =====================================================================
//  M4 — the eval loop (interp.vmExecEnv / vmExec)
// =====================================================================

const interp = vm.interp;

/// Parse `src` with a fresh interner, resolve jumps, root the code for the
/// duration of the run (the C host's vm_root_code discipline), vmExec it,
/// and check the numeric result plus shadow-stack balance.  vmExecEnv pops
/// to its entry watermark on EVERY exit path (break, error.ShenError), so
/// the watermark must come back unchanged after parse + run.
fn expectRunNum(g: *heap.Gc, v: *state.Vm, src: [:0]const u8, want: i64) !void {
    const wm0 = g.rootWatermark();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(g, &sym, src, &code);
    parser.resolveJumps(code.?, len);
    // Root the program across vmExec, popping BEFORE the watermark check
    // (the code root is our own +1; the VM itself must balance to wm0).
    g.rootPushPtr(@ptrCast(&code));
    const got = interp.vmExec(v, @ptrCast(code.?), len) catch |e| {
        g.rootPop();
        return e;
    };
    g.rootPop();
    try std.testing.expectEqual(types.ValTag.number, got.tag);
    try std.testing.expectEqual(want, got.payload.number);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

/// Mirror of expectRunNum asserting a FLOAT result (tag == .float).  The
/// arith/dispatch fixtures use exactly-representable values (halves/quarters),
/// so exact f64 equality is sound.
fn expectRunFloat(g: *heap.Gc, v: *state.Vm, src: [:0]const u8, want: f64) !void {
    const wm0 = g.rootWatermark();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(g, &sym, src, &code);
    parser.resolveJumps(code.?, len);
    g.rootPushPtr(@ptrCast(&code));
    const got = interp.vmExec(v, @ptrCast(code.?), len) catch |e| {
        g.rootPop();
        return e;
    };
    g.rootPop();
    try std.testing.expectEqual(types.ValTag.float, got.tag);
    try std.testing.expectEqual(want, got.payload.float);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 float arith dispatch (+, f/, mixed < promote)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // (m F1.5 F2.5 g+ p): a1=top=2.5, a2=1.5 -> 4.0.
    try expectRunFloat(&g, &v, "(mF[3:F]1.5F[3:F]2.5g[1:s]+p)", 4.0);
    // (m F2.0 F7.0 gf/ p): RTL — push rhs=2.0 first, lhs=7.0 last (top),
    // so a1/a2 = 7.0/2.0 = 3.5 (matches the compiler's binop emission).
    try expectRunFloat(&g, &v, "(mF[3:F]2.0F[3:F]7.0g[2:s]f/p)", 3.5);
}

test "M4 mixed Int/Float comparison promotes" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // (m F2.5 n2 g< p): a1=top=2 (Int), a2=2.5 (Float) -> 2 < 2.5 = true.
    const wm0 = g.rootWatermark();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(mF[3:F]2.5n[1:n]2g[1:s]<p)", &code);
    parser.resolveJumps(code.?, len);
    g.rootPushPtr(@ptrCast(&code));
    const got = interp.vmExec(&v, @ptrCast(code.?), len) catch |e| {
        g.rootPop();
        return e;
    };
    g.rootPop();
    try std.testing.expectEqual(types.ValTag.boolean, got.tag);
    try std.testing.expectEqual(@as(i32, 1), got.payload.boolean);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 zinctest 2: [lambda X X] returns a closure" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const wm0 = g.rootWatermark();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(c(a[1:n]0v))", &code);
    parser.resolveJumps(code.?, len);
    const r = r: {
        g.rootPushPtr(@ptrCast(&code));
        defer g.rootPop();
        break :r try interp.vmExec(&v, @ptrCast(code.?), len);
    };
    // C expects "[lambda ...]": a lambda with an empty captured env whose
    // body is the 2-instr [access 0, return] child parsed from the cur.
    try std.testing.expectEqual(types.ValTag.lambda, r.tag);
    try std.testing.expectEqual(@as(i32, 0), r.payload.lambda.env_len);
    try std.testing.expectEqual(@as(i32, 2), r.payload.lambda.code_len);
    try std.testing.expect(r.payload.lambda.code != null);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 zinctest 3: [let X 1 X] binds and reads back" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    try expectRunNum(&g, &v, "(n[1:n]1ea[1:n]0d)", 1);
}

test "M4 zinctest 34: appterm tail-calls the closure (id 42)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    try expectRunNum(&g, &v, "(mn[2:n]42c(a[1:n]0v)t)", 42);
}

test "M4 zinctest 35: appterm 2-arg RTL env indexes rightmost" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // REWRITTEN for the currying model (the pre-currying program fed 2 args
    // to a 1-PARAM closure — under-application-as-dump then, OVER-application
    // now).  A legitimate 2-param closure (one leading grab => arity 2) keeps
    // the test's intent — RTL arg order + env indexing with access 0 = the
    // innermost param:
    // RTL: 99 pushed first, 42 last; env=[42,99]; access 0 -> env[1] = 99.
    try expectRunNum(&g, &v, "(mn[2:n]99n[2:n]42c(ra[1:n]0v)t)", 99);
}

test "M4 over-applying a 1-param closure throws (old zinctest-35 shape)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // The OLD zinctest-35 program: 2 args to a 1-param closure is now
    // Elm-style over-application — apply 42 to id -> 42, then 42 applied to
    // 99 is non-callable -> catchable ShenError (inside a trap-error site).
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const wm0 = g.rootWatermark();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(mn[2:n]99n[2:n]42c(a[1:n]0v)t)", &code);
    parser.resolveJumps(code.?, len);

    var site = state.CatchSite{ .in_trap_error = true };
    site.parent = v.catch_chain;
    v.catch_chain = &site;
    defer v.catch_chain = site.parent;

    g.rootPushPtr(@ptrCast(&code));
    err: {
        defer g.rootPop();
        try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
        break :err;
    }
    try std.testing.expectEqual(types.ValTag.error_, v.err_slot.tag);
    try std.testing.expectEqualStrings(
        "appterm: over-application to non-callable",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );
    // The error unwind also popped to the entry watermark.
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 zinctest 36: appterm-in-apply frame reuse" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    try expectRunNum(&g, &v, "(mn[2:n]42c(ma[1:n]0c(a[1:n]0v)t)p)", 42);
}

test "M4 grab pops an argument from the stack into the env" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // Body [number 1, grab, access 0, return]: grab binds the pushed 1 as
    // the newest env slot; access 0 reads it back.
    try expectRunNum(&g, &v, "(mn[2:n]42c(n[1:n]1ra[1:n]0v)p)", 1);
}

test "M4 pc past end without frames returns acc" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    try expectRunNum(&g, &v, "(n[1:n]5)", 5);
}

test "M4 pc past end with a live frame pops the CallFrame" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // Body [access 0] has no return: running off the end pops the apply's
    // CallFrame and resumes at cf.pc — itself past the top-level end, so
    // both pc-out-of-range unwind paths run in one program.
    try expectRunNum(&g, &v, "(mn[2:n]42c(a[1:n]0)p)", 42);
}

test "M4 env grows past cap 64 under 100 nested lets" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // (n[1:n]7 e×100 a[1:n]99 d×100) built by hand: 100 lets grow the env
    // 4->8->...->128 (well past STACK_INIT_CAP-style 64), access 99 reads
    // the OLDEST binding env[0]=7, then 100 endlets pop back to empty.
    var buf: [512]u8 = undefined;
    var n: usize = 0;
    const head = "(n[1:n]7";
    @memcpy(buf[n..][0..head.len], head);
    n += head.len;
    for (0..100) |_| {
        buf[n] = 'e';
        n += 1;
    }
    const mid = "a[2:n]99";
    @memcpy(buf[n..][0..mid.len], mid);
    n += mid.len;
    for (0..100) |_| {
        buf[n] = 'd';
        n += 1;
    }
    buf[n] = ')';
    n += 1;
    buf[n] = 0;
    const src: [:0]const u8 = buf[0..n :0];

    try expectRunNum(&g, &v, src, 7);
}

test "M4 apply with >64 args throws ShenError (DECISION A catchable)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // (m n[1:n]1 ×65 c(a[1:n]0v) p): collecting the 65th arg overflows the
    // 64-slot argbuf -> vm_throw, i.e. error.ShenError with the message in
    // the once-rooted err_slot.
    var buf: [1024]u8 = undefined;
    var n: usize = 0;
    const head = "(m";
    @memcpy(buf[n..][0..head.len], head);
    n += head.len;
    for (0..65) |_| {
        const piece = "n[1:n]1";
        @memcpy(buf[n..][0..piece.len], piece);
        n += piece.len;
    }
    const tail = "c(a[1:n]0v)p)";
    @memcpy(buf[n..][0..tail.len], tail);
    n += tail.len;
    buf[n] = 0;
    const src: [:0]const u8 = buf[0..n :0];

    const wm0 = g.rootWatermark();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, src, &code);
    parser.resolveJumps(code.?, len);
    g.rootPushPtr(@ptrCast(&code));
    err: {
        defer g.rootPop();
        try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
        break :err;
    }
    try std.testing.expectEqual(types.ValTag.error_, v.err_slot.tag);
    try std.testing.expectEqualStrings(
        "runtime: too many args (>64)",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );
    // The error unwind also popped to the entry watermark.
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 global lookup of an unknown name throws ShenError" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const wm0 = g.rootWatermark();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(g[3:s]foo)", &code);
    parser.resolveJumps(code.?, len);
    g.rootPushPtr(@ptrCast(&code));
    err: {
        defer g.rootPop();
        try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
        break :err;
    }
    try std.testing.expectEqual(types.ValTag.error_, v.err_slot.tag);
    try std.testing.expectEqualStrings(
        "global not found: foo",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M11 .global single-probe: bundled defun resolves, missing throws" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // Hit path: one defunLookup (no defunHas pre-probe, no fallback).
    v.defunSet("m11glob", values.valNumber(42));
    const wm0 = g.rootWatermark();
    try expectRunNum(&g, &v, "(g[7:s]m11glob)", 42);

    // Miss path (non-empty name): the same ShenError the M4 test asserts.
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(g[9:s]nomissing)", &code);
    parser.resolveJumps(code.?, len);
    g.rootPushPtr(@ptrCast(&code));
    {
        defer g.rootPop();
        try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
    }
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 apply non-callable: hard stop outside trap, throw inside trap" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // Outside any catch site (C:3332-3345): hard stop with acc preserved —
    // the number 42 comes back normally (stderr noise is expected).
    try expectRunNum(&g, &v, "(n[2:n]42p)", 42);

    // Inside a trap-error catch site: the same program throws (catchable).
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(n[2:n]42p)", &code);
    parser.resolveJumps(code.?, len);

    var site = state.CatchSite{ .in_trap_error = true };
    site.parent = v.catch_chain;
    v.catch_chain = &site;
    defer v.catch_chain = site.parent;

    g.rootPushPtr(@ptrCast(&code));
    err: {
        defer g.rootPop();
        try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
        break :err;
    }
    try std.testing.expectEqualStrings(
        "apply non-callable",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );
}

test "M4 endlet on an empty env is a guarded no-op (C:3387)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    // C guards OP_ENDLET with `if (env_len > 0) env_pop(...)` — so (d) on
    // an empty env neither throws nor dies; it is a no-op and the program
    // simply runs off the end.  envPop's trap-site throw branch is purely
    // defensive (unreachable from the eval loop).
    var site = state.CatchSite{ .in_trap_error = true };
    site.parent = v.catch_chain;
    v.catch_chain = &site;
    defer v.catch_chain = site.parent;

    const wm0 = g.rootWatermark();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(d)", &code);
    parser.resolveJumps(code.?, len);
    const r = r: {
        g.rootPushPtr(@ptrCast(&code));
        defer g.rootPop();
        break :r try interp.vmExec(&v, @ptrCast(code.?), len);
    };
    try std.testing.expectEqual(types.ValTag.nil, r.tag);
    try std.testing.expectEqual(types.ValTag.nil, v.err_slot.tag); // nothing thrown
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 apply missing pushmark hard-stops with the function in acc" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    // (n[1:n]1 c(a[1:n]0v) p): the arg is collected but no mark follows —
    // C prints "apply missing pushmark" and stops with acc = the closure.
    const wm0 = g.rootWatermark();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(&g, &sym, "(n[1:n]1c(a[1:n]0v)p)", &code);
    parser.resolveJumps(code.?, len);
    const r = r: {
        g.rootPushPtr(@ptrCast(&code));
        defer g.rootPop();
        break :r try interp.vmExec(&v, @ptrCast(code.?), len);
    };
    try std.testing.expectEqual(types.ValTag.lambda, r.tag);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M4 deep 2000-level cur/apply chain churns collections (verbose probe)" {
    // OOM INVESTIGATION (will shrink once diagnosed): at depth 10000 a
    // 64 MB/256 MB-reserved heap OOM'd; at depth 5000 even 128 MB/1 GB
    // reserved OOM'd — while the expected live set is only ~20 MB
    // (2000..5000 frames x ~2.7 KB).  This run is instrumented: verbose
    // GC banners print live_pages at each collect so the growth curve is
    // visible; verify_collects is OFF to discriminate a verifier side
    // effect from genuine retention.
    var g = try heap.Gc.init(.{
        .heap_bytes = 128 * 1024 * 1024,
        .reserve_bytes = 1024 * 1024 * 1024,
        .verbose = true,
    });
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    const depth = 2000;
    const wm0 = g.rootWatermark(); // the persistent init roots only (err_slot + pools)

    // Programmatic apply chain (no parser recursion):
    //   top        = [pushmark, number 42, cur(body_0), apply]
    //   body_i     = [pushmark, access 0, cur(body_{i+1}), apply, ret]
    //   body_depth = [access 0, ret]
    // Each level applies the next closure with one arg (42), driving
    // frames_sp 10000 deep; the per-level stack/env/closure allocs churn
    // collections while verify_collects re-verifies the heap each time.
    const a = std.heap.page_allocator;
    const slots = try a.alloc(?[*]types.Instr, depth + 1);
    defer a.free(slots);

    const nilv: types.Value = .{ .tag = .nil, .payload = .{ .number = 0 } };

    // Deepest body first; every child array is rooted through its native
    // slot across the next level's alloc (the parser's cc_slot discipline).
    const deepest = g.allocArray(types.Instr, 2);
    deepest[0] = .{ .op = .access, .operand = values.valNumber(0), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    deepest[1] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    slots[depth] = deepest;
    g.rootPushPtr(@ptrCast(&slots[depth]));

    var i: usize = depth;
    while (i > 0) {
        i -= 1;
        const arr = g.allocArray(types.Instr, 5);
        arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[1] = .{ .op = .access, .operand = values.valNumber(0), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        // closure_code read from the ROOTED child slot AFTER this level's
        // alloc — the slot value is post-GC fresh even if it collected.
        arr[2] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(slots[i + 1].?), .closure_len = 5, .jmp_target = 0 };
        arr[3] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[4] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        slots[i] = arr;
        g.rootPushPtr(@ptrCast(&slots[i]));
    }

    const top_arr = g.allocArray(types.Instr, 4);
    top_arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[1] = .{ .op = .number, .operand = values.valNumber(42), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[2] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(slots[0].?), .closure_len = 5, .jmp_target = 0 };
    top_arr[3] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };

    // Drop the 10001 build roots and keep ONLY the program root, so the
    // chain is reachable solely via top -> closure_code links — exactly the
    // reachability the eval loop itself must preserve.  No alloc happens
    // between the popTo and the re-push, so nothing can move.
    var top: ?[*]types.Instr = top_arr;
    g.rootPopTo(wm0);
    g.rootPushPtr(@ptrCast(&top));
    defer g.rootPop();

    const r = try interp.vmExec(&v, @ptrCast(top.?), 4);
    try std.testing.expectEqual(types.ValTag.number, r.tag);
    try std.testing.expectEqual(@as(i64, 42), r.payload.number);
    try std.testing.expectEqual(wm0 + 1, g.rootWatermark());
}

// =====================================================================
//  M4-C — currying / partial application (N<A) + Elm-style
//  over-application (N>A)   [plan T1-T8; T7 lives at zinctest 35,
//  T8 is the whole-suite gate]
// =====================================================================

test "T1 zincArity: leading grabs + 1, non-grab stops" {
    var g = try testInit();
    defer g.deinit();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();

    const Case = struct { src: [:0]const u8, want: i32 };
    const cases = [_]Case{
        .{ .src = "(c(a[1:n]0v))", .want = 1 }, // [access, ret]
        .{ .src = "(c(ra[1:n]0v))", .want = 2 }, // [grab, access, ret]
        .{ .src = "(c(rra[1:n]0v))", .want = 3 }, // [grab, grab, access, ret]
        .{ .src = "(c(ma[1:n]0v)t)", .want = 1 }, // leading non-grab stops
    };
    for (cases) |c| {
        const wm0 = g.rootWatermark();
        var code: ?[*]types.Instr = null;
        _ = try parser.parseBytecode(&g, &sym, c.src, &code);
        try std.testing.expectEqual(types.Opcode.cur, code.?[0].op);
        const body = code.?[0].closure_code;
        const body_len = code.?[0].closure_len;
        try std.testing.expectEqual(c.want, interp.zincArity(body, body_len));
        try std.testing.expectEqual(wm0, g.rootWatermark());
    }
}

test "T2 N<A apply: partial closure, arity chain 3->2->1, full-app value" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    const wm0 = g.rootWatermark();

    // 3-param fn (two leading grabs) returning its FIRST param (access 2).
    // Apply to 1 arg -> a partial closure: drop-grabs 1 (code_len 4 -> 3,
    // the visible drop) and env = [] ++ [7] (len 1).
    const p1 = try expectRunVal(&g, &v, "(mn[1:n]7c(rra[1:n]2v)p)");
    try std.testing.expectEqual(types.ValTag.lambda, p1.tag);
    try std.testing.expectEqual(@as(i32, 3), p1.payload.lambda.code_len);
    try std.testing.expectEqual(@as(i32, 1), p1.payload.lambda.env_len);
    try std.testing.expectEqual(wm0, g.rootWatermark());

    // defunSet is C-heap only (no GC alloc between the run and the store);
    // the dirty-marked table entry keeps the partial reachable afterwards.
    v.defunSet("p1", p1);

    // The partial applied to the remaining 2 args yields the full-app
    // value: env = [7] ++ [8,9]; access 2 -> env[0] = 7 (== (f 7 8 9)).
    try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8g[2:s]p1p)", 7);

    // The partial applied to 1 MORE arg yields another partial (arity
    // chain 3 -> 2 -> 1): code_len 3 -> 2, env = [7,8].
    const p2 = try expectRunVal(&g, &v, "(mn[1:n]8g[2:s]p1p)");
    try std.testing.expectEqual(types.ValTag.lambda, p2.tag);
    try std.testing.expectEqual(@as(i32, 2), p2.payload.lambda.code_len);
    try std.testing.expectEqual(@as(i32, 2), p2.payload.lambda.env_len);
    v.defunSet("p2", p2);
    try expectRunNum(&g, &v, "(mn[1:n]9g[2:s]p2p)", 7);
}

test "T3 map-style: partial flows as a value through env/argbuf" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // add = 2-param cur'd closure: [grab, pushmark, access 1, access 0,
    // global +, apply, ret] — the zinc-c shape for (lambda X Y (+ X Y)).
    const add = try expectRunVal(&g, &v, "(c(rma[1:n]1a[1:n]0g[1:s]+pv))");
    try std.testing.expectEqual(types.ValTag.lambda, add.tag);
    try std.testing.expectEqual(@as(i32, 2), interp.zincArity(add.payload.lambda.code, add.payload.lambda.code_len));
    v.defunSet("add", add);

    // Sanity: the exact-arity call still works (byte-compat path).
    try expectRunNum(&g, &v, "(mn[1:n]4n[1:n]3g[3:s]addp)", 7);

    // Partial application: add(5) -> a 1-param closure.
    const p5 = try expectRunVal(&g, &v, "(mn[1:n]5g[3:s]addp)");
    try std.testing.expectEqual(types.ValTag.lambda, p5.tag);
    v.defunSet("p5", p5);

    // Pass the PARTIAL as an ARG into another closure that applies it —
    // the function-as-value flows through env/argbuf like any value.
    try expectRunNum(&g, &v, "(mg[2:s]p5c(mn[1:n]3a[1:n]0pv)p)", 8);
}

test "T4 N>A apply: peel continuation, non-callable/prim throw" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // (1) 2-param fn returning a 1-param fn (cur inside the body), called
    // with 3 args at once: the peel applies [7,8], gets the closure back,
    // and the dispatch applies the remaining 9 -> access 2 = env[0] = 7.
    try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8n[1:n]7c(rc(a[1:n]2v)v)p)", 7);
    // Curried equivalent of the same call: ((f 7 8) 9) == 7 — the env
    // layout of the curried chain matches the single-shot call.  (9 sits
    // BELOW the inner mark so the inner apply takes exactly [7,8]; its
    // result closure is then the outer apply's fn with 9 its only arg.)
    try expectRunNum(&g, &v, "(mn[1:n]9mn[1:n]8n[1:n]7c(rc(a[1:n]2v)v)pp)", 7);

    // (2) Over-application to a fn returning a NON-callable, inside
    // trap-error: 2-param fn returning a1; the peel leaves the number 0
    // with arg 2 remaining -> catchable ShenError -> handler returns 77.
    try expectRunNum(
        &g,
        &v,
        "(mc(n[2:n]77v)c(mn[1:n]2n[1:n]1n[1:n]0c(ra[1:n]1v)pv)g[10:s]trap-errorp)",
        77,
    );
    // ... and the error message outside the trap path:
    {
        var sym = symbols.SymbolInterner.init();
        defer sym.deinit();
        const wm0 = g.rootWatermark();
        var code: ?[*]types.Instr = null;
        const len = try parser.parseBytecode(&g, &sym, "(mn[1:n]2n[1:n]1n[1:n]0c(ra[1:n]1v)p)", &code);
        parser.resolveJumps(code.?, len);
        var site = state.CatchSite{ .in_trap_error = true };
        site.parent = v.catch_chain;
        v.catch_chain = &site;
        defer v.catch_chain = site.parent;
        g.rootPushPtr(@ptrCast(&code));
        err: {
            defer g.rootPop();
            try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
            break :err;
        }
        try std.testing.expectEqualStrings(
            "apply: over-application to non-callable",
            std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
        );
        try std.testing.expectEqual(wm0, g.rootWatermark());
    }

    // (3) Over-application to a fn returning a PRIM: prims are
    // fixed-arity, NOT curried — the peel throws instead.
    {
        var sym = symbols.SymbolInterner.init();
        defer sym.deinit();
        var code: ?[*]types.Instr = null;
        const len = try parser.parseBytecode(&g, &sym, "(mn[1:n]2n[1:n]1n[1:n]0c(rg[1:s]+v)p)", &code);
        parser.resolveJumps(code.?, len);
        var site = state.CatchSite{ .in_trap_error = true };
        site.parent = v.catch_chain;
        v.catch_chain = &site;
        defer v.catch_chain = site.parent;
        g.rootPushPtr(@ptrCast(&code));
        err: {
            defer g.rootPop();
            try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(code.?), len));
            break :err;
        }
        try std.testing.expectEqualStrings(
            "apply: cannot over-apply primitive",
            std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
        );
    }
}

test "T5 appterm N<A returns the partial to the grand-caller; N>A returns the value" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // N<A at tail position: the outer apply frames; the inner body's
    // appterm under-applies a 3-param fn, so the PARTIAL (not the body
    // value) is returned through the .ret frame-restore flow to the top
    // level.  The 3-param closure captured env [7] at cur time, so the
    // partial's env = [7, 5] and its code is the body minus 1 grab.
    const r = try expectRunVal(&g, &v, "(mn[1:n]7c(mn[1:n]5c(rra[1:n]2v)t)p)");
    try std.testing.expectEqual(types.ValTag.lambda, r.tag);
    try std.testing.expectEqual(@as(i32, 3), r.payload.lambda.code_len);
    try std.testing.expectEqual(@as(i32, 2), r.payload.lambda.env_len);

    // The tail-returned partial still applies to the remaining 2 args:
    // env = [7,5,8,9]; access 2 -> env[1] = 5 (a1 of the appterm call).
    v.defunSet("tp", r);
    try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8g[2:s]tpp)", 5);

    // N>A at tail position: the peel bottoms out and the ret-flow returns
    // the final value through the outer frame.
    try expectRunNum(&g, &v, "(mn[2:n]99c(mn[1:n]9n[1:n]8n[1:n]7c(rc(a[1:n]2v)v)t)p)", 7);
}

test "T6 GC churn: partial closures + peels survive scavenge and full collect" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    const wm0 = g.rootWatermark();

    // Build a partial, then FORCE a nursery scavenge (promotes the partial,
    // its fresh drop-grabs code array and its env) and a FULL collect
    // (evacuates old-gen).  A stale interior pointer in the partial would
    // read garbage or crash here — demand the exact structure through the
    // moved pointers, then apply it.
    var p = try expectRunVal(&g, &v, "(mn[1:n]7c(rra[1:n]2v)p)");
    var guard = g.rootValue(&p);
    g.collectNursery(.@"test");
    g.collect(.@"test");
    const code_many: [*]types.Instr = @ptrCast(p.payload.lambda.code.?);
    try std.testing.expectEqual(types.Opcode.grab, code_many[0].op);
    try std.testing.expectEqual(types.Opcode.access, code_many[1].op);
    try std.testing.expectEqual(types.Opcode.ret, code_many[2].op);
    try std.testing.expectEqual(@as(i32, 3), p.payload.lambda.code_len);
    try std.testing.expectEqual(types.ValTag.number, p.payload.lambda.env.?[0].tag);
    try std.testing.expectEqual(@as(i64, 7), p.payload.lambda.env.?[0].payload.number);
    try std.testing.expect(g.nursery_scavenge_count > 0);
    try std.testing.expect(g.full_collect_count > 0);
    v.defunSet("pc", p);
    guard.end();
    try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8g[2:s]pcp)", 7);

    // Churn loop: unrooted cons garbage + repeated peels/partial-builds
    // drive natural scavenges and threshold full collects between the
    // runs; expectRunNum/expectRunVal assert the watermark balance after
    // every vmExec (the wm0 pattern).
    var i: usize = 0;
    while (i < 60) : (i += 1) {
        _ = consNums(&g, &.{ 1, 2, 3, 4, 5 }); // unrooted garbage
        try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8n[1:n]7c(rc(a[1:n]2v)v)p)", 7); // peel
        const q = try expectRunVal(&g, &v, "(mn[1:n]7c(rra[1:n]2v)p)"); // fresh partial
        v.defunSet("qc", q);
        try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8g[2:s]qcp)", 7); // partial applies
    }
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

// =====================================================================
//  M5 — exec_primitive, the pure subset (prims.zig)
// =====================================================================

/// expectRunNum's generalization: run `src`, return the raw result Value
/// (consumed before any further allocation), and check shadow balance.
fn expectRunVal(g: *heap.Gc, v: *state.Vm, src: [:0]const u8) !types.Value {
    const wm0 = g.rootWatermark();
    var sym = symbols.SymbolInterner.init();
    defer sym.deinit();
    var code: ?[*]types.Instr = null;
    const len = try parser.parseBytecode(g, &sym, src, &code);
    parser.resolveJumps(code.?, len);
    g.rootPushPtr(@ptrCast(&code));
    const got = interp.vmExec(v, @ptrCast(code.?), len) catch |e| {
        g.rootPop();
        return e;
    };
    g.rootPop();
    try std.testing.expectEqual(wm0, g.rootWatermark());
    return got;
}

fn expectRunBool(g: *heap.Gc, v: *state.Vm, src: [:0]const u8, want: bool) !void {
    const r = try expectRunVal(g, v, src);
    try std.testing.expectEqual(types.ValTag.boolean, r.tag);
    try std.testing.expectEqual(@as(i64, if (want) 1 else 0), r.payload.boolean);
}

fn expectRunStr(g: *heap.Gc, v: *state.Vm, src: [:0]const u8, want: []const u8) !void {
    const r = try expectRunVal(g, v, src);
    try std.testing.expectEqual(types.ValTag.string, r.tag);
    try std.testing.expectEqualStrings(want, values.strSlice(r));
}

/// Call a primitive directly with a synthetic stack (the unit-test half of
/// M5): `args` are in POP order — args[0] is popped first (a1, the FIRST
/// Shen arg), matching the RTL push convention (rightmost pushed first).
/// The stack's data slot is rooted exactly like the eval loop's prologue
/// root (4), and the watermark must balance.
fn primExec(
    g: *heap.Gc,
    v: *state.Vm,
    name: []const u8,
    args: []const types.Value,
    acc: *types.Value,
) !void {
    const wm0 = g.rootWatermark();
    var stack: types.ValueArray = .{ .data = null, .len = 0, .cap = 0 };
    g.rootPushPtr(@ptrCast(&stack.data));
    interp.vaInit(g, &stack);
    // Push in REVERSE so args[0] lands on top (popped first, = a1).
    var i: usize = args.len;
    while (i > 0) {
        i -= 1;
        interp.vaPush(g, &stack, args[i]);
    }
    try prims.execPrimitive(v, name, acc, &stack);
    g.rootPop(); // stack.data
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

/// Build a GC string Value (used by the M6 stream-prim tests).
fn strVal(g: *heap.Gc, s: []const u8) types.Value {
    return values.valString(g, s);
}

/// Build a number Value (used by the M6 stream-prim tests).
fn numVal(n: i64) types.Value {
    return values.valNumber(n);
}

test "M5 zinctest 1,4,5,6: arithmetic +,-,*,/" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]+p)", 3);
    try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]-p)", -1);
    try expectRunNum(&g, &v, "(mn[1:n]4n[1:n]3g[1:s]*p)", 12);
    try expectRunNum(&g, &v, "(mn[1:n]2n[2:n]10g[1:s]/p)", 5);
}

test "M5 zinctest 7-11,24,25: comparisons =,<,>,<=,>=" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    try expectRunBool(&g, &v, "(mn[1:n]1n[1:n]1g[1:s]=p)", true);
    try expectRunBool(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]<p)", true);
    try expectRunBool(&g, &v, "(mn[1:n]3n[1:n]5g[1:s]>p)", true);
    try expectRunBool(&g, &v, "(mn[1:n]2n[1:n]2g[2:s]<=p)", true);
    try expectRunBool(&g, &v, "(mn[1:n]3n[1:n]5g[2:s]>=p)", true);
    try expectRunBool(&g, &v, "(mS[2:S]abS[2:S]abg[1:s]=p)", true);
    try expectRunBool(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]=p)", false);
    // cross-type = never crashes: number vs string is false.
    try expectRunBool(&g, &v, "(mS[1:S]1n[1:n]1g[1:s]=p)", false);
    // M4: mixed Int/Float = promotes (2 == 2.0 is true, Elm parity).
    try expectRunBool(&g, &v, "(mF[3:F]2.0n[1:n]2g[1:s]=p)", true);
    // M4: float-vs-nonnumber still falls through to false (no asFloat panic).
    try expectRunBool(&g, &v, "(mS[1:S]xF[3:F]2.0g[1:s]=p)", false);
}

test "M5 zinctest 12-16: type predicates" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    try expectRunBool(&g, &v, "(mn[2:n]42g[7:s]number?p)", true);
    try expectRunBool(&g, &v, "(ms[5:s]hellog[7:s]symbol?p)", true);
    // M4: number? recognizes floats (review fix-2).
    try expectRunBool(&g, &v, "(mF[3:F]2.0g[7:s]number?p)", true);
    try expectRunBool(&g, &v, "(mS[3:S]2.0g[7:s]number?p)", false);
    try expectRunBool(&g, &v, "(mb[4:b]trueg[8:s]boolean?p)", true);
    try expectRunBool(&g, &v, "(mS[2:S]hig[7:s]string?p)", true);
    try expectRunBool(&g, &v, "(mn[2:n]42g[7:s]string?p)", false);
}

test "M5 zinctest 17: cons" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const r = try expectRunVal(&g, &v, "(mn[1:n]2n[1:n]1g[4:s]consp)");
    try std.testing.expectEqual(types.ValTag.cons, r.tag);
    const want = values.valCons(&g, values.valNumber(1), values.valNumber(2));
    try std.testing.expect(values.deepEqual(r, want, 0));
}

test "M5 zinctest 18-23: string prims cn/n->string/string->n/str/tlstr/intern" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    try expectRunStr(&g, &v, "(mS[5:S]worldS[5:S]hellog[2:s]cnp)", "helloworld");
    try expectRunStr(&g, &v, "(mn[2:n]42g[9:s]n->stringp)", "*"); // ASCII 42
    try expectRunNum(&g, &v, "(mS[2:S]42g[9:s]string->np)", 52); // ASCII '4'
    try expectRunStr(&g, &v, "(ms[5:s]hellog[3:s]strp)", "hello");
    try expectRunStr(&g, &v, "(mS[3:S]abcg[5:s]tlstrp)", "bc");
    const r = try expectRunVal(&g, &v, "(mS[3:S]foog[6:s]internp)");
    try std.testing.expectEqual(types.ValTag.symbol, r.tag);
    try std.testing.expectEqualStrings("foo", values.symSlice(r));
}

test "M5 zinctest 26: simple-error throws ShenError with the message" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    try std.testing.expectError(
        error.ShenError,
        expectRunVal(&g, &v, "(mS[4:S]boomg[12:s]simple-errorp)"),
    );
    try std.testing.expectEqualStrings(
        "boom",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );
}

test "M5 zinctest 27: trap-error runs the handler on the error" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    // (trap-error (simple-error "oops") (lambda E "caught")) — handler pushed
    // FIRST (bottom), body LAST (top), exactly the zinctest RTL comment.
    try expectRunStr(
        &g,
        &v,
        "(mc(S[6:S]caughtv)c(mS[4:S]oopsg[12:s]simple-errorpv)g[10:s]trap-errorp)",
        "caught",
    );
    // Non-throwing body: trap-error returns the body value untouched.
    try expectRunNum(
        &g,
        &v,
        "(mc(S[6:S]caughtv)c(mn[1:n]7v)g[10:s]trap-errorp)",
        7,
    );
}

test "M5 trap-error handler sees the error: error-to-string through the trap path" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    // Handler = (lambda E (error-to-string E)): access 0 reads the appended
    // error from the handler env, primErrorToString copies it through the
    // slot-rooted valStringFromErr (the C:1973 latent-bug port-fix).
    try expectRunStr(
        &g,
        &v,
        "(mc(ma[1:n]0g[15:s]error-to-stringpv)c(mS[4:S]boomg[12:s]simple-errorpv)g[10:s]trap-errorp)",
        "boom",
    );
}

test "M5 zinctest 28: get-time unix returns a sane epoch-seconds number" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const r = try expectRunVal(&g, &v, "(ms[4:s]unixg[8:s]get-timep)");
    try std.testing.expectEqual(types.ValTag.number, r.tag);
    try std.testing.expect(r.payload.number > 1_600_000_000); // > 2020-09
    try std.testing.expect(r.payload.number < 4_000_000_000); // < 2096
}

test "M5 zinctest 33: appterm to a primitive" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]+t)", 3);
}

test "M5 zinctest 37/38: appterm hard stops preserve acc (the closure)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    // 37: zero args — the eval loop prints and breaks to done, acc = closure.
    const r37 = try expectRunVal(&g, &v, "(c(a[1:n]0v)t)");
    try std.testing.expectEqual(types.ValTag.lambda, r37.tag);
    // 38: missing pushmark — same hard-stop shape.
    const r38 = try expectRunVal(&g, &v, "(n[2:n]42c(a[1:n]0v)t)");
    try std.testing.expectEqual(types.ValTag.lambda, r38.tag);
}

test "M5 unknown prim hard-stops (C: print + return -1)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = values.valNumber(99);
    // "frobnicate" is not in prim_table — error.Halt (the C `return -1`),
    // acc preserved.
    try std.testing.expectError(error.Halt, primExec(&g, &v, "frobnicate", &.{}, &acc));
    try std.testing.expectEqual(@as(i64, 99), acc.payload.number);
    // isValid/lookupDef agree with the table.
    try std.testing.expect(prims.isValid("+"));
    try std.testing.expect(!prims.isValid("frobnicate"));
    try std.testing.expect(prims.lookupDef("trap-error") != null);
}

test "M5 initGlobals registers every prim: defunGet falls back to valPrim" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    // [global absvector] resolves to the VAL_PRIM (M5 defunGet fallback);
    // a non-prim unknown name still falls back to a bare symbol.
    const p = v.defunGet("absvector");
    try std.testing.expectEqual(types.ValTag.prim, p.tag);
    try std.testing.expectEqualStrings("absvector", values.primSlice(p));
    const s = v.defunGet("*shen-macro*");
    try std.testing.expectEqual(types.ValTag.symbol, s.tag);
    // Every table name is registered (defunHas sees it).
    for (prims.primNames()) |def| {
        try std.testing.expect(v.defunHas(def.name));
    }
}

test "M5 hd/tl/empty? and list ops reverse/append/assoc/element?" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    const list = consNums(&g, &.{ 1, 2, 3 }); // [1 2 3]
    const pair = values.valCons(&g, values.valNumber(1), values.valNumber(2));

    try primExec(&g, &v, "hd", &.{list}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.number);
    try primExec(&g, &v, "tl", &.{list}, &acc);
    try std.testing.expect(values.deepEqual(acc, consNums(&g, &.{ 2, 3 }), 0));
    try primExec(&g, &v, "hd", &.{values.valNil()}, &acc); // nil -> nil
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);
    try primExec(&g, &v, "tl", &.{values.valNil()}, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);
    try primExec(&g, &v, "empty?", &.{values.valNil()}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.boolean);
    try primExec(&g, &v, "empty?", &.{list}, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.boolean);

    // reverse (and the non-list throw).
    try primExec(&g, &v, "reverse", &.{list}, &acc);
    try std.testing.expect(values.deepEqual(acc, consNums(&g, &.{ 3, 2, 1 }), 0));
    try std.testing.expectError(
        error.ShenError,
        primExec(&g, &v, "reverse", &.{values.valNumber(1)}, &acc),
    );
    try std.testing.expectEqualStrings(
        "attempt to reverse a non-list",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );

    // append [1 2] [3] = [1 2 3]; append nil x = x.
    try primExec(&g, &v, "append", &.{ consNums(&g, &.{ 1, 2 }), consNums(&g, &.{3}) }, &acc);
    try std.testing.expect(values.deepEqual(acc, list, 0));
    try primExec(&g, &v, "append", &.{ values.valNil(), pair }, &acc);
    try std.testing.expect(values.deepEqual(acc, pair, 0));

    // assoc: hit returns the pair, miss nil, non-list throws.
    const p12 = values.valCons(&g, values.valNumber(1), values.valNumber(2));
    const p34 = values.valCons(&g, values.valNumber(3), values.valNumber(4));
    const pairs = values.valCons(&g, p12, values.valCons(&g, p34, values.valNil()));
    try primExec(&g, &v, "assoc", &.{ values.valNumber(3), pairs }, &acc);
    try std.testing.expect(values.deepEqual(acc, p34, 0));
    try primExec(&g, &v, "assoc", &.{ values.valNumber(9), pairs }, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);
    try std.testing.expectError(
        error.ShenError,
        primExec(&g, &v, "assoc", &.{ values.valNumber(1), values.valNumber(7) }, &acc),
    );

    // element?
    try primExec(&g, &v, "element?", &.{ values.valNumber(2), list }, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.boolean);
    try primExec(&g, &v, "element?", &.{ values.valNumber(9), list }, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.boolean);
}

test "M5 vectors: absvector/address->/<-address with a forced scavenge (write barrier)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    try primExec(&g, &v, "absvector", &.{values.valNumber(100)}, &acc);
    try std.testing.expectEqual(types.ValTag.vector, acc.tag);
    try std.testing.expectEqual(@as(i32, 100), acc.payload.vector.len);

    // Store a NURSERY cons at slot 7 (address-> args in pop order:
    // vector, index, value).  acc is the only reference to the vector, so
    // root it across the scavenge below.
    g.rootPushValue(&acc);
    defer g.rootPop();
    const nursery_val = values.valCons(&g, values.valNumber(1), values.valNumber(2));
    try primExec(&g, &v, "address->", &.{ acc, values.valNumber(7), nursery_val }, &acc);
    try std.testing.expectEqual(types.ValTag.vector, acc.tag); // address-> returns the vector

    // Force a nursery scavenge: the vector element array must be re-scanned
    // via the remembered set (writeBarrierVectorStore) so the moved cons
    // stays reachable and the read-back is intact.
    g.collectNursery(.@"test");

    var got: types.Value = undefined;
    try primExec(&g, &v, "<-address", &.{ acc, values.valNumber(7) }, &got);
    try std.testing.expect(values.deepEqual(got, nursery_val, 0));
    // Fresh vector elements are zeroed Values: VAL_NUMBER(0) is tag 0
    // (C calloc parity — NOT nil).
    try primExec(&g, &v, "<-address", &.{ acc, values.valNumber(0) }, &got);
    try std.testing.expectEqual(types.ValTag.number, got.tag);
    try std.testing.expectEqual(@as(i64, 0), got.payload.number);

    try primExec(&g, &v, "absvector?", &.{acc}, &got);
    try std.testing.expectEqual(@as(i64, 1), got.payload.boolean);
    try primExec(&g, &v, "emptylist", &.{values.valNumber(0)}, &got);
    try std.testing.expectEqual(types.ValTag.nil, got.tag);
}

test "M5 error-to-string port-fix: message copy survives churn (valStringFromErr)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    // An error whose message sits in the NURSERY; churn the nursery, then
    // copy the message through the slot-rooted helper and check content.
    var err = values.valError(&g, "stale-pointer probe message");
    var guard = g.rootValue(&err);
    defer guard.end();
    var junk: types.Value = values.valNil();
    var jguard = g.rootValue(&junk);
    defer jguard.end();
    var i: usize = 0;
    while (i < 4096) : (i += 1) {
        junk = values.valCons(&g, values.valNumber(@intCast(i)), junk);
        if (i % 512 == 511) g.collectNursery(.@"test");
    }
    const s = values.valStringFromErr(&g, &err);
    try std.testing.expectEqual(types.ValTag.string, s.tag);
    try std.testing.expectEqualStrings("stale-pointer probe message", values.strSlice(s));

    // And through the prim: error->string, string passes through, anything
    // else becomes "unknown error".
    try primExec(&g, &v, "error-to-string", &.{err}, &acc);
    try std.testing.expectEqualStrings("stale-pointer probe message", values.strSlice(acc));
    try primExec(&g, &v, "error-to-string", &.{values.valNumber(1)}, &acc);
    try std.testing.expectEqualStrings("unknown error", values.strSlice(acc));
    try primExec(&g, &v, "error?", &.{err}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.boolean);
}

test "M5 string prims unit: pos/hdstr/substring/char-code/c-strlen/cn/str/bytes" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    const abc = values.valString(&g, "abc");
    try primExec(&g, &v, "pos", &.{ abc, values.valNumber(1) }, &acc);
    try std.testing.expectEqualStrings("b", values.strSlice(acc));
    try primExec(&g, &v, "pos", &.{ abc, values.valNumber(9) }, &acc);
    try std.testing.expectEqualStrings("", values.strSlice(acc)); // OOB outside trap
    try primExec(&g, &v, "hdstr", &.{abc}, &acc);
    try std.testing.expectEqualStrings("a", values.strSlice(acc));
    try primExec(&g, &v, "tlstr", &.{values.valString(&g, "a")}, &acc);
    try std.testing.expectEqualStrings("", values.strSlice(acc)); // safe len<=1 deviation
    try primExec(&g, &v, "substring", &.{ abc, values.valNumber(1), values.valNumber(5) }, &acc);
    try std.testing.expectEqualStrings("bc", values.strSlice(acc)); // clamped
    try primExec(&g, &v, "substring", &.{ abc, values.valNumber(-3), values.valNumber(2) }, &acc);
    try std.testing.expectEqualStrings("ab", values.strSlice(acc));
    try primExec(&g, &v, "char-code", &.{ abc, values.valNumber(0) }, &acc);
    try std.testing.expectEqual(@as(i64, 97), acc.payload.number);
    try primExec(&g, &v, "char-code", &.{ abc, values.valNumber(9) }, &acc);
    try std.testing.expectEqual(@as(i64, -1), acc.payload.number); // OOB
    try primExec(&g, &v, "c-strlen", &.{abc}, &acc);
    try std.testing.expectEqual(@as(i64, 3), acc.payload.number);

    // cn over numbers (pre-formatted into stack buffers) + symbols + nil.
    try primExec(&g, &v, "cn", &.{ values.valNumber(42), values.valString(&g, "x") }, &acc);
    try std.testing.expectEqualStrings("42x", values.strSlice(acc));
    try primExec(&g, &v, "cn", &.{ values.valBoolean(true), values.valNil() }, &acc);
    try std.testing.expectEqualStrings("true[]", values.strSlice(acc));

    // str of a composite renders the [a b c] list form (grow-loop path).
    const list = consNums(&g, &.{ 1, 2, 3 });
    try primExec(&g, &v, "str", &.{list}, &acc);
    try std.testing.expectEqualStrings("[1 2 3]", values.strSlice(acc));
    try primExec(&g, &v, "str", &.{values.valBoolean(false)}, &acc);
    try std.testing.expectEqualStrings("false", values.strSlice(acc));

    // str->bytes / bytes->string round-trip.
    const hi = values.valString(&g, "hi");
    try primExec(&g, &v, "shen.str->bytes", &.{hi}, &acc);
    try primExec(&g, &v, "shen.bytes->string", &.{acc}, &acc);
    try std.testing.expectEqualStrings("hi", values.strSlice(acc));
}

test "M5 set/value, gensym/newvar, @p/fst/snd, shen.fail!, variable?" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    // (set "counter" 42) returns the value; (value "counter") reads it back.
    const sym = symbols.valSymbol(&v.symbols, "counter");
    try primExec(&g, &v, "set", &.{ sym, values.valNumber(42) }, &acc);
    try std.testing.expectEqual(@as(i64, 42), acc.payload.number);
    try primExec(&g, &v, "value", &.{sym}, &acc);
    try std.testing.expectEqual(@as(i64, 42), acc.payload.number);
    // Unset (value X): symbol fallback (value_get has no prim fallback).
    try primExec(&g, &v, "value", &.{symbols.valSymbol(&v.symbols, "nope")}, &acc);
    try std.testing.expectEqual(types.ValTag.symbol, acc.tag);

    // gensym / newvar counters.
    try primExec(&g, &v, "gensym", &.{}, &acc);
    try std.testing.expectEqualStrings("shen.gensym_0", values.symSlice(acc));
    try primExec(&g, &v, "gensym", &.{}, &acc);
    try std.testing.expectEqualStrings("shen.gensym_1", values.symSlice(acc));
    try primExec(&g, &v, "newvar", &.{}, &acc);
    try std.testing.expectEqualStrings("V_0", values.symSlice(acc));

    // @p / fst / snd.
    try primExec(&g, &v, "@p", &.{ values.valNumber(1), values.valNumber(2) }, &acc);
    try std.testing.expectEqual(types.ValTag.cons, acc.tag);
    try primExec(&g, &v, "fst", &.{acc}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.number);
    try primExec(&g, &v, "@p", &.{ values.valNumber(3), values.valNumber(4) }, &acc);
    try primExec(&g, &v, "snd", &.{acc}, &acc);
    try std.testing.expectEqual(@as(i64, 4), acc.payload.number);

    // shen.fail! with an arg builds (fail Arg); without one it throws.
    try primExec(&g, &v, "shen.fail!", &.{values.valNumber(7)}, &acc);
    try std.testing.expectEqual(types.ValTag.cons, acc.tag);
    try std.testing.expectEqualStrings("fail", values.symSlice(acc.payload.cons.car.?.*));
    try std.testing.expectError(error.ShenError, primExec(&g, &v, "shen.fail!", &.{}, &acc));

    // variable?: uppercase-initial alnum/punct continuation.
    try primExec(&g, &v, "variable?", &.{symbols.valSymbol(&v.symbols, "X2?")}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.boolean);
    try primExec(&g, &v, "variable?", &.{symbols.valSymbol(&v.symbols, "x")}, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.boolean);
    try primExec(&g, &v, "variable?", &.{values.valNumber(1)}, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.boolean);

    // function? sees both lambdas and prims; stream? sees streams.
    try primExec(&g, &v, "function?", &.{values.valPrim("+")}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.boolean);
    try primExec(&g, &v, "function?", &.{values.valNumber(1)}, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.boolean);
    try primExec(&g, &v, "stream?", &.{values.valStreamIn(null)}, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.boolean);
    try primExec(&g, &v, "cons?", &.{values.valNil()}, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.boolean);
}

test "M5 simple-error caps the message at 255 bytes (C snprintf parity)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    var big: [300]u8 = undefined;
    @memset(&big, 'x');
    const bigs = values.valString(&g, big[0..]);
    try std.testing.expectError(
        error.ShenError,
        primExec(&g, &v, "simple-error", &.{bigs}, &acc),
    );
    const msg = std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0);
    try std.testing.expectEqual(@as(usize, 255), msg.len);
    try std.testing.expectError(
        error.ShenError,
        primExec(&g, &v, "simple-error", &.{values.valNumber(3)}, &acc),
    );
    try std.testing.expectEqualStrings(
        "simple-error called",
        std.mem.sliceTo(v.err_slot.payload.error_.message.?, 0),
    );
}

// =====================================================================
//  M6 — bundle loader (parse_bundle + vm_load_bundle)
// =====================================================================

test "M6 loadBundle: entry parsed, defun-registered, callable" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // One entry: (plus2 (cur body)) with body = the toplevel shape
    // [pushmark 2 1 global + apply ret] — a 0-arg closure computing 2+1.
    // The ret lives INSIDE the cur parens (the bundle cur convention); the
    // name atom is csexp form [5:s]plus2.
    const n = v.loadBundle("(([5:s]plus2 (c(mn[1:n]2n[1:n]1g[1:s]+pv))))");
    try std.testing.expectEqual(@as(i32, 1), n);
    try std.testing.expectEqual(wm0, g.rootWatermark());

    // defunGet resolves to the bundled lambda (explicit entry beats the
    // val_symbol fallback).
    const f = v.defunGet("plus2");
    try std.testing.expectEqual(types.ValTag.lambda, f.tag);

    // Call it: pushmark + global + apply -> nargs 0 -> run the body -> 3.
    try expectRunNum(&g, &v, "(mmg[5:s]plus2p)", 3);
}

// =====================================================================
//  M6 — stream I/O prims (streams.zig): string streams + file round-trip
// =====================================================================

/// The tmpDir is created under .zig-cache/tmp relative to the test's CWD;
/// resolve its ABSOLUTE path so the prims (plain openat/read/write on the
/// process cwd) can reach the fixture files.
fn tmpAbsPath(
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
    sub: []const u8,
) ![]u8 {
    const io = std.testing.io;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.parent_dir.realPathFile(io, tmp.sub_path[0..], &buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ buf[0..n], sub });
}

test "M6 streams: string-stream read-byte round-trip then EOF then close" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    // valStringStreamIn copies 'abc' into the registry; read-byte walks it.
    const s = v.streams.valStringStreamIn(&g, "abc");
    try std.testing.expectEqual(@as(i64, 1), s.payload.stream.is_string);
    try primExec(&g, &v, "read-byte", &.{s}, &acc);
    try std.testing.expectEqual(@as(i64, 'a'), acc.payload.number);
    try primExec(&g, &v, "read-byte", &.{s}, &acc);
    try std.testing.expectEqual(@as(i64, 'b'), acc.payload.number);
    try primExec(&g, &v, "read-byte", &.{s}, &acc);
    try std.testing.expectEqual(@as(i64, 'c'), acc.payload.number);
    // Exhausted: -1 (C EOF parity), repeatedly.
    try primExec(&g, &v, "read-byte", &.{s}, &acc);
    try std.testing.expectEqual(@as(i64, -1), acc.payload.number);
    try primExec(&g, &v, "read-byte", &.{s}, &acc);
    try std.testing.expectEqual(@as(i64, -1), acc.payload.number);
    // close frees the slot; a subsequent close of the SAME stale idx is a
    // no-op (freeStringStream zeroes the slot but keeps n_string_streams, so
    // the stale file ptr re-resolves to a freed-but-in-range slot -> nil).
    try primExec(&g, &v, "close", &.{s}, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);
    try primExec(&g, &v, "close", &.{s}, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);
}

test "M6 streams: read-file-as-string on a temp file" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "probe.txt", .data = "hello\n" });
    const path = try tmpAbsPath(std.testing.allocator, &tmp, "probe.txt");
    defer std.testing.allocator.free(path);

    try primExec(&g, &v, "read-file-as-string", &.{strVal(&g, path)}, &acc);
    try std.testing.expectEqualStrings("hello\n", values.strSlice(acc));
}

test "M6 streams: open 'in' existing file -> file stream read-byte; ENOENT -> string stream of the PATH" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "in.txt", .data = "Z!" });
    const path = try tmpAbsPath(std.testing.allocator, &tmp, "in.txt");
    defer std.testing.allocator.free(path);

    // Existing path: a real FILE stream; read-byte pulls the first byte.
    try primExec(&g, &v, "open", &.{ strVal(&g, path), strVal(&g, "in") }, &acc);
    try std.testing.expectEqual(types.ValTag.stream, acc.tag);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.stream.is_input);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.stream.is_string);
    try primExec(&g, &v, "read-byte", &.{acc}, &acc);
    try std.testing.expectEqual(@as(i64, 'Z'), acc.payload.number);

    // C quirk, ported: open 'in' on a MISSING path yields a STRING stream of
    // the path bytes, and read-byte over it yields the path's first byte.
    const missing = try std.fmt.allocPrint(std.testing.allocator, "{s}/nope.bin", .{path[0 .. path.len - "in.txt".len]});
    defer std.testing.allocator.free(missing);
    try primExec(&g, &v, "open", &.{ strVal(&g, missing), strVal(&g, "in") }, &acc);
    try std.testing.expectEqual(@as(i64, 1), acc.payload.stream.is_string);
    try primExec(&g, &v, "read-byte", &.{acc}, &acc);
    try std.testing.expectEqual(@as(i64, missing[0]), acc.payload.number);
}

test "M6 streams: open 'out' write + close + read-file-as-string round-trip" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(std.testing.allocator, &tmp, "out.txt");
    defer std.testing.allocator.free(path);

    try primExec(&g, &v, "open", &.{ strVal(&g, path), strVal(&g, "out") }, &acc);
    try std.testing.expectEqual(types.ValTag.stream, acc.tag);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.stream.is_input);
    const s = acc;
    // write-byte pops (byte, stream); returns the byte written.
    try primExec(&g, &v, "write-byte", &.{ numVal('o'), s }, &acc);
    try std.testing.expectEqual(@as(i64, 'o'), acc.payload.number);
    try primExec(&g, &v, "write-byte", &.{ numVal('k'), s }, &acc);
    try primExec(&g, &v, "write-byte", &.{ numVal('\n'), s }, &acc);
    try primExec(&g, &v, "close", &.{s}, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);

    try primExec(&g, &v, "read-file-as-string", &.{strVal(&g, path)}, &acc);
    try std.testing.expectEqualStrings("ok\n", values.strSlice(acc));
}

test "M6 streams: close returns nil for string and file streams" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    var acc: types.Value = undefined;

    // String stream: close frees the slot and returns nil.
    const ss = v.streams.valStringStreamIn(&g, "abc");
    try primExec(&g, &v, "close", &.{ss}, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);

    // File stream (temp file, 'out'): close returns nil.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(std.testing.allocator, &tmp, "c.txt");
    defer std.testing.allocator.free(path);
    try primExec(&g, &v, "open", &.{ strVal(&g, path), strVal(&g, "out") }, &acc);
    try std.testing.expectEqual(@as(i64, 0), acc.payload.stream.is_input);
    try primExec(&g, &v, "close", &.{acc}, &acc);
    try std.testing.expectEqual(types.ValTag.nil, acc.tag);
}

test "M6 loadBundle: keywords, streams, tables, primitive?-names" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    _ = v.loadBundle("(([5:s]plus2 (c(mn[1:n]2n[1:n]1g[1:s]+pv))))");

    // Pattern keywords resolve to bare symbols (structural matching);
    // bundled entries keep their closures.
    try std.testing.expectEqual(types.ValTag.symbol, v.defunGet("number").tag);
    try std.testing.expectEqual(types.ValTag.symbol, v.defunGet("lookup").tag);
    try std.testing.expectEqual(types.ValTag.lambda, v.defunGet("plus2").tag);

    // A bundled `lookup` must NOT be clobbered by keyword registration
    // (C:4042-4051 — the metacircular interp needs [global lookup] to stay
    // its closure).
    var g2 = try testInit();
    defer g2.deinit();
    var v2: state.Vm = undefined;
    v2.init(&g2);
    defer v2.deinit();
    try std.testing.expectEqual(@as(i32, 2), v2.loadBundle(
        "(([6:s]lookup (c(mn[1:n]2n[1:n]1g[1:s]+pv))) ([5:s]plus2 (c(mn[1:n]2n[1:n]1g[1:s]+pv))))",
    ));
    try std.testing.expectEqual(types.ValTag.lambda, v2.defunGet("lookup").tag);
    try expectRunNum(&g2, &v2, "(mmg[6:s]lookupp)", 3);

    // Streams: value variables with in/out flags (files are the I/O
    // milestone's; null for now).
    const stin = v.valueGet("*stinput*");
    try std.testing.expectEqual(types.ValTag.stream, stin.tag);
    try std.testing.expectEqual(@as(i64, 1), stin.payload.stream.is_input);
    const stout = v.valueGet("*stoutput*");
    try std.testing.expectEqual(types.ValTag.stream, stout.tag);
    try std.testing.expectEqual(@as(i64, 0), stout.payload.stream.is_input);
    const sterr = v.valueGet("*sterror*");
    try std.testing.expectEqual(types.ValTag.stream, sterr.tag);

    // global-table / value-table start as empty alists, not bare symbols.
    try std.testing.expectEqual(types.ValTag.nil, v.valueGet("global-table").tag);
    try std.testing.expectEqual(types.ValTag.nil, v.valueGet("value-table").tag);

    // primitive?-names: exactly one entry per prim, head = LAST table name
    // (C forward-build parity).
    const names = prims.primNames();
    var pn = v.valueGet("primitive?-names");
    var count: usize = 0;
    while (pn.tag == .cons) : (count += 1) pn = pn.payload.cons.cdr.?.*;
    try std.testing.expectEqual(names.len, count);
    const head = v.valueGet("primitive?-names").payload.cons.car.?.*;
    try std.testing.expectEqualStrings(names[names.len - 1].name, values.symSlice(head));
}

test "M6 loadBundle: nested curs, scavenge survival, error paths" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // Three entries: a plain body, a nested cur (inner closure applied at
    // runtime — the parser's cc-slot rooting across bundle entries), and a
    // string body.
    try std.testing.expectEqual(@as(i32, 3), v.loadBundle(
        "(([5:s]plus2 (c(mn[1:n]2n[1:n]1g[1:s]+pv))) ([5:s]seven (c(mc(mn[1:n]7v)pv))) ([2:s]hi (c(mS[2:S]hiv))))",
    ));
    try std.testing.expectEqual(wm0, g.rootWatermark());
    try expectRunNum(&g, &v, "(mmg[5:s]plus2p)", 3);
    try expectRunNum(&g, &v, "(mmg[5:s]sevenp)", 7);
    try expectRunStr(&g, &v, "(mmg[2:s]hip)", "hi");

    // The bundled closures live in the defun table: force a scavenge and
    // call again — the body code arrays must stay reachable and their
    // defun-table entries re-scanned (the dirty-scan path).
    g.collectNursery(.@"test");
    try expectRunNum(&g, &v, "(mmg[5:s]sevenp)", 7);
    try expectRunNum(&g, &v, "(mmg[5:s]plus2p)", 3);

    // Error paths (C semantics: print + partial count, never throw), each
    // leaving the shadow stack balanced.
    // Not a bundle (no '((*').
    try std.testing.expectEqual(@as(i32, 0), v.loadBundle("(mn[1:n]1g[1:s]+p)"));
    // Name atom is not a symbol.
    try std.testing.expectEqual(@as(i32, 0), v.loadBundle("(([1:n]5 (c(mn[1:n]1v))))"));
    // Code list has no cur wrapper.
    try std.testing.expectEqual(@as(i32, 0), v.loadBundle("(([1:s]f (mn[1:n]1v)))"));
    // Mid-bundle failure after one good entry: partial count.
    try std.testing.expectEqual(@as(i32, 1), v.loadBundle(
        "(([5:s]plus2 (c(mn[1:n]2n[1:n]1g[1:s]+pv))) (bad",
    ));
    try std.testing.expectEqual(wm0, g.rootWatermark());
    // The good entry is still callable after all that.
    try expectRunNum(&g, &v, "(mmg[5:s]plus2p)", 3);
}

// =====================================================================
//  M7 — stress hardening (allocation churn under verified collects)
// =====================================================================

/// primExec's rooting-safe variant for stress churn: roots the ARG SLOT
/// itself (not a caller-copied slice) so a scavenge during vaInit/vaPush —
/// certain under stress — can never push a stale interior pointer.  The
/// arg value is re-read through the root at push time (post-GC fresh).
fn primExecRooted(
    g: *heap.Gc,
    v: *state.Vm,
    name: []const u8,
    arg: *types.Value,
    acc: *types.Value,
) !void {
    const wm0 = g.rootWatermark();
    g.rootPushValue(arg);
    g.rootPushValue(acc);
    var stack: types.ValueArray = .{ .data = null, .len = 0, .cap = 0 };
    g.rootPushPtr(@ptrCast(&stack.data));
    interp.vaInit(g, &stack);
    interp.vaPush(g, &stack, arg.*); // fresh read through the root
    try prims.execPrimitive(v, name, acc, &stack);
    g.rootPop(); // stack.data
    g.rootPop(); // acc
    g.rootPop(); // arg
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

/// Two-arg variant of primExecRooted (append-shaped prims).  a1/a2 must be
/// distinct rooted slots (assign a copy with no intervening allocation).
fn primExecRooted2(
    g: *heap.Gc,
    v: *state.Vm,
    name: []const u8,
    a1: *types.Value,
    a2: *types.Value,
    acc: *types.Value,
) !void {
    const wm0 = g.rootWatermark();
    g.rootPushValue(a1);
    g.rootPushValue(a2);
    g.rootPushValue(acc);
    var stack: types.ValueArray = .{ .data = null, .len = 0, .cap = 0 };
    g.rootPushPtr(@ptrCast(&stack.data));
    interp.vaInit(g, &stack);
    // Push in REVERSE so a1 lands on top (popped first = first Shen arg).
    interp.vaPush(g, &stack, a2.*);
    interp.vaPush(g, &stack, a1.*);
    try prims.execPrimitive(v, name, acc, &stack);
    g.rootPop(); // stack.data
    g.rootPop(); // acc
    g.rootPop(); // a2
    g.rootPop(); // a1
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M7 stress: 50k-cons build/reverse/append churn under verify_collects" {
    var g = try heap.Gc.init(.{
        .heap_bytes = 128 * 1024 * 1024,
        .reserve_bytes = 1024 * 1024 * 1024,
        .verify_collects = true,
    });
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    var acc: types.Value = values.valNil();
    var acc2: types.Value = values.valNil();
    var hd: types.Value = values.valNil();

    var iter: usize = 0;
    while (iter < 50) : (iter += 1) {
        // Rooted 50k-cons build of [0..49999] (consNums is unsafe at this
        // scale — it never roots its accumulator).
        var list = values.valNil();
        var lguard = g.rootValue(&list);
        defer lguard.end();
        var i: usize = 50000;
        while (i > 0) {
            i -= 1;
            list = values.valCons(&g, values.valNumber(@intCast(i)), list);
        }

        // reverse: 50k fresh conses allocated inside execPrimitive.
        try primExecRooted(&g, &v, "reverse", &list, &acc);
        try primExecRooted(&g, &v, "hd", &acc, &hd);
        try std.testing.expectEqual(@as(i64, 49999), hd.payload.number);

        // append the reversed list to itself: 100k fresh conses.
        acc2 = acc; // no allocation between the copy and the rooted call
        try primExecRooted2(&g, &v, "append", &acc, &acc2, &hd);
        try primExecRooted(&g, &v, "hd", &hd, &acc2);
        try std.testing.expectEqual(@as(i64, 49999), acc2.payload.number);

        // Occasional full collect — verified end to end.
        if (iter % 25 == 24) g.collect(.@"test");
    }
}

test "M7 stress: 600-level cur/apply chain with prim calls interleaved" {
    var g = try heap.Gc.init(.{
        .heap_bytes = 128 * 1024 * 1024,
        .reserve_bytes = 1024 * 1024 * 1024,
    });
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // Chain shape (M4's deep-recursion test with a + prim call per level):
    //   top     = [pushmark, number 41, number 1, global +, apply,
    //              cur(body_0), apply]
    //   body_i  = [pushmark, access 0, number 1, global +, apply,
    //              cur(body_{i+1}), apply, ret]
    //   deepest = [access 0, ret]
    // The prim apply leaves N+1 on the stack under the still-open mark,
    // which then becomes the recursive call's single argument; each frame's
    // cur captures its own env, so envs grow linearly with depth (deeper
    // write-barrier/nursery-reference churn than the M4 chain).
    const depth: usize = 600;
    const wm0 = g.rootWatermark();
    const a = std.heap.page_allocator;
    const slots = try a.alloc(?[*]types.Instr, depth + 1);
    defer a.free(slots);
    const nilv: types.Value = .{ .tag = .nil, .payload = .{ .number = 0 } };
    const plus = symbols.valSymbol(&v.symbols, "+");

    const deepest = g.allocArray(types.Instr, 2);
    deepest[0] = .{ .op = .access, .operand = values.valNumber(0), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    deepest[1] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    slots[depth] = deepest;
    g.rootPushPtr(@ptrCast(&slots[depth]));

    var i: usize = depth;
    while (i > 0) {
        i -= 1;
        const arr = g.allocArray(types.Instr, 8);
        arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[1] = .{ .op = .access, .operand = values.valNumber(0), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[2] = .{ .op = .number, .operand = values.valNumber(1), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[3] = .{ .op = .global, .operand = plus, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[4] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        // closure_code read from the ROOTED child slot AFTER this level's
        // alloc — post-GC fresh even if it collected.
        arr[5] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(slots[i + 1].?), .closure_len = 8, .jmp_target = 0 };
        arr[6] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[7] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        slots[i] = arr;
        g.rootPushPtr(@ptrCast(&slots[i]));
    }

    const top_arr = g.allocArray(types.Instr, 7);
    top_arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[1] = .{ .op = .number, .operand = values.valNumber(41), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[2] = .{ .op = .number, .operand = values.valNumber(1), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[3] = .{ .op = .global, .operand = plus, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[4] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[5] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(slots[0].?), .closure_len = 8, .jmp_target = 0 };
    top_arr[6] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };

    // Keep ONLY the program root, exactly like the M4 chain test.
    var top: ?[*]types.Instr = top_arr;
    g.rootPopTo(wm0);
    g.rootPushPtr(@ptrCast(&top));
    defer g.rootPop();

    const r = try interp.vmExec(&v, @ptrCast(top.?), 7);
    try std.testing.expectEqual(types.ValTag.number, r.tag);
    try std.testing.expectEqual(@as(i64, 42 + @as(i64, @intCast(depth))), r.payload.number);
    try std.testing.expectEqual(wm0 + 1, g.rootWatermark());
}

test "M7 stress: defun mutation under forced scavenges keeps table integrity" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    var junk: types.Value = values.valNil();
    var jguard = g.rootValue(&junk);
    defer jguard.end();

    var i: usize = 0;
    while (i < 512) : (i += 1) {
        // A fresh NURSERY cons stored straight into the registered global
        // table (dirty-marked by defunSet) — no alloc between valCons and
        // defunSet, so the hand-off is safe.
        v.defunSet("stress-fn", values.valCons(&g, values.valNumber(@intCast(i)), values.valNil()));
        // Churn + a forced scavenge: the dirty slot must be re-scanned so
        // the moved cons (and its car cell) stays reachable and updated.
        junk = values.valCons(&g, values.valNumber(0), junk);
        g.collectNursery(.@"test");
        const got = v.defunGet("stress-fn");
        try std.testing.expectEqual(types.ValTag.cons, got.tag);
        try std.testing.expectEqual(@as(i64, @intCast(i)), got.payload.cons.car.?.*.payload.number);
    }
    // wm0 + 1: the junk guard root is still held (its defer ends later).
    try std.testing.expectEqual(wm0 + 1, g.rootWatermark());
}

// =====================================================================
//  M8 — process execution (execplan.zig): primExecPlan + env/cwd + glob
// =====================================================================

/// The tagged-list demarshal accessors (mirror execplan's tdl probes without
/// the page_allocator decode structs).  [cons H T] = cons(sym "cons",
/// cons(H, cons(T, nil))), so tdlFirst == payload of [tag X] == head of a
/// tagged list, and tdlSecond == the tail T.
fn tdlFirst(v: types.Value) types.Value {
    return v.payload.cons.cdr.?.*.payload.cons.car.?.*;
}

fn tdlSecond(v: types.Value) types.Value {
    return v.payload.cons.cdr.?.*.payload.cons.cdr.?.*.payload.cons.car.?.*;
}

/// [symbol X] = cons(sym "symbol", cons(sym X, nil)).  valSymbol interns into
/// the symbol table's own allocator (never the GC heap), so the inner cons's
/// interior pointers cannot move before the outer valCons roots it.
fn taggedSym(g: *heap.Gc, v: *state.Vm, name: []const u8) types.Value {
    const inner = values.valCons(g, symbols.valSymbol(&v.symbols, name), values.valNil());
    return values.valCons(g, symbols.valSymbol(&v.symbols, "symbol"), inner);
}

test "M8 process: primExecPlan decodes + runs a two-command pipeline" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const ep = vm.execplan;

    // argv1 = [string echo] [string -n] [string hi]
    var e = ep.makeTaggedString(&v, "echo");
    var e_g = g.rootValue(&e); defer e_g.end();
    var n = ep.makeTaggedString(&v, "-n");
    var n_g = g.rootValue(&n); defer n_g.end();
    var hi = ep.makeTaggedString(&v, "hi");
    var hi_g = g.rootValue(&hi); defer hi_g.end();
    var argv1 = ep.makeTaggedNil(&v);
    var argv1_g = g.rootValue(&argv1); defer argv1_g.end();
    // Prepend in reverse so the head is argv[0]: [echo -n hi].
    argv1 = ep.makeTaggedCons(&v, hi, argv1);
    argv1 = ep.makeTaggedCons(&v, n, argv1);
    argv1 = ep.makeTaggedCons(&v, e, argv1);

    // argv2 = [string tr] [string i] [string o]
    var tr = ep.makeTaggedString(&v, "tr");
    var tr_g = g.rootValue(&tr); defer tr_g.end();
    var ich = ep.makeTaggedString(&v, "i");
    var ich_g = g.rootValue(&ich); defer ich_g.end();
    var o1 = ep.makeTaggedString(&v, "o");
    var o1_g = g.rootValue(&o1); defer o1_g.end();
    var argv2 = ep.makeTaggedNil(&v);
    var argv2_g = g.rootValue(&argv2); defer argv2_g.end();
    // Prepend in reverse so the head is argv[0]: [tr i o].
    argv2 = ep.makeTaggedCons(&v, o1, argv2);
    argv2 = ep.makeTaggedCons(&v, ich, argv2);
    argv2 = ep.makeTaggedCons(&v, tr, argv2);

    // cmd = [Argv Redirs Sub]: empty redirs + plain sub are both [cons].
    var nil = ep.makeTaggedNil(&v);
    var nil_g = g.rootValue(&nil); defer nil_g.end();
    var cmd1 = ep.makeTaggedCons(&v, argv1, ep.makeTaggedCons(&v, nil, ep.makeTaggedCons(&v, nil, ep.makeTaggedNil(&v))));
    var cmd1_g = g.rootValue(&cmd1); defer cmd1_g.end();
    var cmd2 = ep.makeTaggedCons(&v, argv2, ep.makeTaggedCons(&v, nil, ep.makeTaggedCons(&v, nil, ep.makeTaggedNil(&v))));
    var cmd2_g = g.rootValue(&cmd2); defer cmd2_g.end();

    // pipeline = [cmd1 cmd2]; chain = [[symbol seq] pipeline]; program = [chain].
    var pipeline = ep.makeTaggedCons(&v, cmd1, ep.makeTaggedCons(&v, cmd2, ep.makeTaggedNil(&v)));
    var pipeline_g = g.rootValue(&pipeline); defer pipeline_g.end();
    var seq = taggedSym(&g, &v, "seq");
    var seq_g = g.rootValue(&seq); defer seq_g.end();
    var chain = ep.makeTaggedCons(&v, seq, ep.makeTaggedCons(&v, pipeline, ep.makeTaggedNil(&v)));
    var chain_g = g.rootValue(&chain); defer chain_g.end();
    var program = ep.makeTaggedCons(&v, chain, ep.makeTaggedNil(&v));
    var program_g = g.rootValue(&program); defer program_g.end();

    var acc: types.Value = undefined;
    try primExec(&g, &v, "exec-plan", &.{program}, &acc);

    // Result = [cons [number code] [cons [string out] [cons [string err] [cons]]]].
    const codeTag = tdlFirst(acc);
    const outList = tdlSecond(acc);
    const outTag = tdlFirst(outList);
    const errList = tdlSecond(outList);
    const errTag = tdlFirst(errList);
    try std.testing.expectEqual(@as(i64, 0), tdlFirst(codeTag).payload.number);
    try std.testing.expectEqualStrings("ho", values.strSlice(tdlFirst(outTag)));
    try std.testing.expectEqualStrings("", values.strSlice(tdlFirst(errTag)));
}

test "M8 process: setenv/getenv round-trip and unset fallback" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    var acc: types.Value = undefined;
    try primExec(&g, &v, "setenv", &.{ strVal(&g, "FX_M8_TEST"), strVal(&g, "hello") }, &acc);
    try std.testing.expectEqual(types.ValTag.boolean, acc.tag);
    try std.testing.expectEqual(@as(i32, 1), acc.payload.boolean);

    try primExec(&g, &v, "getenv", &.{strVal(&g, "FX_M8_TEST")}, &acc);
    try std.testing.expectEqual(types.ValTag.string, acc.tag);
    try std.testing.expectEqualStrings("hello", values.strSlice(acc));

    try primExec(&g, &v, "getenv", &.{strVal(&g, "FX_M8_TEST_UNSET")}, &acc);
    try std.testing.expectEqualStrings("", values.strSlice(acc));
}

test "M8 process: glob returns a sorted tagged list of names" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.parent_dir.realPathFile(std.testing.io, tmp.sub_path[0..], &buf);
    const dir = try std.fmt.allocPrint(std.testing.allocator, "{s}", .{buf[0..n]});
    defer std.testing.allocator.free(dir);
    const pattern = try std.fmt.allocPrint(std.testing.allocator, "{s}/*.txt", .{dir});
    defer std.testing.allocator.free(pattern);

    for ([_][]const u8{ "c.txt", "a.txt", "b.txt" }) |name| {
        var f = try tmp.dir.createFile(std.testing.io, name, .{});
        f.close(std.testing.io);
    }

    var acc: types.Value = undefined;
    try primExec(&g, &v, "glob", &.{strVal(&g, pattern)}, &acc);

    // Sorted ascending: [a.txt b.txt c.txt], then the empty tagged list [cons].
    const s1 = tdlFirst(acc);
    const rest1 = tdlSecond(acc);
    const s2 = tdlFirst(rest1);
    const rest2 = tdlSecond(rest1);
    const s3 = tdlFirst(rest2);
    const rest3 = tdlSecond(rest2);
    try std.testing.expectEqualStrings("a.txt", values.strSlice(tdlFirst(s1)));
    try std.testing.expectEqualStrings("b.txt", values.strSlice(tdlFirst(s2)));
    try std.testing.expectEqualStrings("c.txt", values.strSlice(tdlFirst(s3)));
    try std.testing.expect(std.mem.eql(u8, "cons", values.symSlice(rest3.payload.cons.car.?.*)));
    try std.testing.expectEqual(types.ValTag.nil, rest3.payload.cons.cdr.?.*.tag);
}

// =====================================================================
//  M10 — frame-stack pool (interp.frameStackAcquire/Release)
// =====================================================================

/// Build a `depth`-level arity-1 cur+apply chain whose deepest body applies
/// the simple-error primitive with every frame still live (the cd9be11
/// retention shape: a stale cf.env / cf.stack.data in the pooled array would
/// pin the per-level env arrays through the drain scan).  Returns the top
/// program code; the chain stays reachable only via closure_code, so the
/// caller must root the returned pointer before the next allocation.  Pops
/// its own build roots back to the entry watermark.
fn buildThrowingChain(g: *heap.Gc, v: *state.Vm, depth: usize) ![*]types.Instr {
    const wm0 = g.rootWatermark();
    const a = std.heap.page_allocator;
    const nilv: types.Value = .{ .tag = .nil, .payload = .{ .number = 0 } };
    const slots = try a.alloc(?[*]types.Instr, depth + 1);
    defer a.free(slots);
    const boom = symbols.valSymbol(&v.symbols, "simple-error");

    const deepest = g.allocArray(types.Instr, 4);
    deepest[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    deepest[1] = .{ .op = .string, .operand = values.valString(g, "boom"), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    deepest[2] = .{ .op = .global, .operand = boom, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    deepest[3] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    slots[depth] = deepest;
    g.rootPushPtr(@ptrCast(&slots[depth]));

    var i: usize = depth;
    while (i > 0) {
        i -= 1;
        const arr = g.allocArray(types.Instr, 5);
        arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[1] = .{ .op = .access, .operand = values.valNumber(0), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[2] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(slots[i + 1].?), .closure_len = 4, .jmp_target = 0 };
        arr[3] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[4] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        slots[i] = arr;
        g.rootPushPtr(@ptrCast(&slots[i]));
    }

    const top_arr = g.allocArray(types.Instr, 4);
    top_arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[1] = .{ .op = .number, .operand = values.valNumber(42), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[2] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(slots[0].?), .closure_len = 5, .jmp_target = 0 };
    top_arr[3] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };

    g.rootPopTo(wm0);
    return top_arr;
}

/// Build a k-level Elm-style over-application peel chain: each level is an
/// arity-1 closure whose body over-applies the next level (two args into a
/// 1-param closure), nesting one vmExecEnv per level; each level then returns
/// a fixed 1-arg closure (KONST, which ignores its arg and returns 7) so the
/// unwind applies the leftover arg to KONST instead of throwing.  The top
/// program returns 7 with a total vmExecEnv nesting depth of k + 1 (the
/// top thunk + k nested peels).  Pops its own build roots; the caller roots
/// the returned top code before the next allocation.
fn buildPeelChain(g: *heap.Gc, k: usize) ![*]types.Instr {
    std.debug.assert(k >= 2);
    const wm0 = g.rootWatermark();
    const a = std.heap.page_allocator;
    const nilv: types.Value = .{ .tag = .nil, .payload = .{ .number = 0 } };

    // KONST: the fixed arity-1 closure every level returns on unwind.
    const konstant = g.allocArray(types.Instr, 2);
    konstant[0] = .{ .op = .number, .operand = values.valNumber(7), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    konstant[1] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    var konstant_slot: ?[*]types.Instr = konstant;
    g.rootPushPtr(@ptrCast(&konstant_slot));

    const levels = try a.alloc(?[*]types.Instr, k);
    defer a.free(levels);

    // Base level L_k: body = cur(KONST) ret  (returns KONST, arity 1).
    const base = g.allocArray(types.Instr, 2);
    base[0] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(konstant), .closure_len = 2, .jmp_target = 0 };
    base[1] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    levels[k - 1] = base;
    g.rootPushPtr(@ptrCast(&levels[k - 1]));

    // Levels L_{k-1} .. L_1: pushmark number access0 cur(next) apply cur(KONST) ret.
    var i: usize = k - 1;
    while (i > 0) {
        i -= 1;
        const arr = g.allocArray(types.Instr, 7);
        arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[1] = .{ .op = .number, .operand = values.valNumber(1), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[2] = .{ .op = .access, .operand = values.valNumber(0), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[3] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(levels[i + 1].?), .closure_len = if (i + 1 == k - 1) 2 else 7, .jmp_target = 0 };
        arr[4] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        arr[5] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(konstant), .closure_len = 2, .jmp_target = 0 };
        arr[6] = .{ .op = .ret, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
        levels[i] = arr;
        g.rootPushPtr(@ptrCast(&levels[i]));
    }

    // Top thunk: over-apply L_1 with two args (42, 43).
    const top_arr = g.allocArray(types.Instr, 5);
    top_arr[0] = .{ .op = .pushmark, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[1] = .{ .op = .number, .operand = values.valNumber(42), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[2] = .{ .op = .number, .operand = values.valNumber(43), .closure_code = null, .closure_len = 0, .jmp_target = 0 };
    top_arr[3] = .{ .op = .cur, .operand = nilv, .closure_code = @ptrCast(levels[0].?), .closure_len = 7, .jmp_target = 0 };
    top_arr[4] = .{ .op = .apply, .operand = nilv, .closure_code = null, .closure_len = 0, .jmp_target = 0 };

    g.rootPopTo(wm0);
    return top_arr;
}

test "M10 frame-stack pool: sequential vmExec calls reuse one array" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // First call misses (fresh ~3 MB old-gen array), then releases to pool.
    try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]+t)", 3);
    try std.testing.expectEqual(@as(u64, 1), v.frame_pool_misses);
    try std.testing.expectEqual(@as(u64, 0), v.frame_pool_hits);
    try std.testing.expectEqual(@as(usize, 1), v.frame_pool_live);

    // Every later call hits the pooled array: misses stay 1, hits climb.
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]+t)", 3);
    }
    try std.testing.expectEqual(@as(u64, 1), v.frame_pool_misses);
    try std.testing.expectEqual(@as(u64, 50), v.frame_pool_hits);
    try std.testing.expectEqual(@as(usize, 1), v.frame_pool_live);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M10 frame-stack pool: error unwind clears the used range (retention regression)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // Clean floor: a trivial call pools one clean array; a full collect then
    // leaves only that array live (the result scalar has no GC payload).
    try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]+t)", 3);
    g.collect(.@"test");
    const clean = g.allocatedPages();

    // Deep apply chain whose deepest body throws (simple-error) with all
    // `depth` frames still live — the cd9be11 retention shape.  A stale
    // cf.env/cf.stack.data pointer left in the pooled array would pin the
    // per-level env arrays through every later full collect; release's
    // clear-only-live-range invariant must prevent that.
    const depth: usize = 300;
    // Keep ONLY the program root (the chain stays reachable via closure_code).
    var top: ?[*]types.Instr = try buildThrowingChain(&g, &v, depth);
    g.rootPushPtr(@ptrCast(&top));

    // The throw unwinds all `depth` frames; release clears [0..depth).
    try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(top.?), 4));

    // The released array is back in the pool, all-null across the used range
    // — the make-or-break invariant, checked white-box.
    try std.testing.expectEqual(@as(usize, 1), v.frame_pool_live);
    const pooled = v.frame_pool[0].?;
    var j: usize = 0;
    while (j <= depth) : (j += 1) {
        try std.testing.expect(pooled[j].code == null);
        try std.testing.expect(pooled[j].env == null);
        try std.testing.expect(pooled[j].stack.data == null);
    }

    // Drop the program root and force a full collect: the only extra live
    // object over the clean floor is the retained error value (a few pages).
    // A stale env pointer would pin ~depth env arrays and blow past +32.
    g.rootPop(); // top
    g.collect(.@"test");
    const after = g.allocatedPages();
    try std.testing.expect(after <= clean + 32);

    // Idle pooled array + repeated forced collects -> pages stay stable.
    g.collect(.@"test");
    try std.testing.expectEqual(after, g.allocatedPages());
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M10 frame-stack pool: reentrancy (trap-error + peel) keeps the pool bounded" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // Plain call establishes the single-array baseline.
    try expectRunNum(&g, &v, "(mn[1:n]2n[1:n]1g[1:s]+t)", 3);
    try std.testing.expectEqual(@as(u64, 1), v.frame_pool_misses);
    try std.testing.expectEqual(@as(usize, 1), v.frame_pool_live);

    // trap-error nests vmExecEnv twice (throwing body, then handler),
    // SEQUENTIALLY — the handler reuses the body's released array.  Misses
    // cap at 2 (nesting depth), never growing with the number of calls, and
    // the LIFO free-list means a nested call can never alias an outer frame.
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try expectRunStr(
            &g,
            &v,
            "(mc(S[6:S]caughtv)c(mS[4:S]oopsg[12:s]simple-errorpv)g[10:s]trap-errorp)",
            "caught",
        );
    }
    try std.testing.expectEqual(@as(u64, 2), v.frame_pool_misses);
    try std.testing.expectEqual(@as(usize, 2), v.frame_pool_live);

    // N>A peel nests one extra vmExecEnv per over-application step; a single
    // step (3 args into a 2-param fn returning a 1-param fn) keeps the pool
    // at depth 2, and the result (7 == env[0]) proves no aliasing.
    var k: usize = 0;
    while (k < 20) : (k += 1) {
        try expectRunNum(&g, &v, "(mn[1:n]9n[1:n]8n[1:n]7c(rc(a[1:n]2v)v)p)", 7);
    }
    try std.testing.expectEqual(@as(u64, 2), v.frame_pool_misses);
    try std.testing.expectEqual(@as(usize, 2), v.frame_pool_live);
    try std.testing.expect(v.frame_pool_hits > 0);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M10 frame-stack pool: pool-full drop branch (deep peel overflows the cap)" {
    // Larger than testInit: depth-3 nesting needs 3 simultaneously-live
    // ~3 MB arrays (~9 MB), which would sit just above the 16 MB heap's
    // 8 MB old-gen threshold and churn grow_heap (the SF-1 failure mode).
    // The test only verifies the drop branch, so give it headroom.
    var g = try heap.Gc.init(.{
        .heap_bytes = 64 * 1024 * 1024,
        .reserve_bytes = 256 * 1024 * 1024,
    });
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // k nested peel levels + the top thunk = k + 1 simultaneous vmExecEnv
    // entries, one past FRAME_POOL_MAX.  Every level over-applies the next,
    // so each acquisition during the descent is a fresh alloc (all entries
    // are live at once); on unwind the pool fills to FRAME_POOL_MAX and the
    // last release hits the drop branch.  (Depth is kept one above the cap
    // so the ~3 MB/array peak stays well within the 16 MB semi-space.)
    const k: usize = state.FRAME_POOL_MAX; // depth 4 > cap 3
    const depth: u64 = @intCast(k + 1); // + the top thunk
    const cap: u64 = @intCast(state.FRAME_POOL_MAX);

    var run: usize = 0;
    while (run < 2) : (run += 1) {
        var top: ?[*]types.Instr = try buildPeelChain(&g, k);
        g.rootPushPtr(@ptrCast(&top));
        const got = try interp.vmExec(&v, @ptrCast(top.?), 5);
        g.rootPop();
        try std.testing.expectEqual(types.ValTag.number, got.tag);
        try std.testing.expectEqual(@as(i64, 7), got.payload.number);
    }

    // Run 1 allocated `depth` fresh arrays (all misses) and dropped the
    // excess beyond the cap; run 2 reused the `cap` retained arrays and
    // re-allocated the other `depth - cap` (the drop branch).  live stays
    // capped at `cap` throughout — the drop path must re-allocate correctly
    // with no corruption.
    try std.testing.expectEqual(depth + (depth - cap), v.frame_pool_misses);
    try std.testing.expectEqual(cap, v.frame_pool_hits);
    try std.testing.expectEqual(state.FRAME_POOL_MAX, v.frame_pool_live);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M10 frame-stack pool: reuse clears only the live range (deep then shallow)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    const deep: usize = 300;
    const shallow: usize = 3;

    // Deep throw: dirties slots [0..deep); the error unwind leaves every
    // frame live, so release clears the whole [0..deep) range.
    var top: ?[*]types.Instr = try buildThrowingChain(&g, &v, deep);
    g.rootPushPtr(@ptrCast(&top));
    try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(top.?), 4));
    g.rootPop();
    try std.testing.expectEqual(@as(usize, 1), v.frame_pool_live);

    // Shallow throw REUSES the same pooled array: dirties [0..shallow),
    // release clears only that range — the induction step that makes
    // "clear only [0..sp)" sufficient across reuse.
    var top2: ?[*]types.Instr = try buildThrowingChain(&g, &v, shallow);
    g.rootPushPtr(@ptrCast(&top2));
    try std.testing.expectError(error.ShenError, interp.vmExec(&v, @ptrCast(top2.?), 4));
    g.rootPop();

    // The pooled array must be all-null across BOTH ranges: [0..shallow)
    // cleared by this release, [shallow..deep) still null from the deep
    // release (and never re-dirtied).  Any stale cf.env / cf.stack.data
    // would pin a dead env array through the full-capacity drain scan.
    try std.testing.expectEqual(@as(usize, 1), v.frame_pool_live);
    const pooled = v.frame_pool[0].?;
    var j: usize = 0;
    while (j < deep) : (j += 1) {
        try std.testing.expect(pooled[j].code == null);
        try std.testing.expect(pooled[j].env == null);
        try std.testing.expect(pooled[j].stack.data == null);
    }
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

// =====================================================================
//  M11 — appterm tail-env reuse (interp.zig appterm N==A)
// =====================================================================

/// Number of decimal digits of `v` (for csexp `a[<len>:n]<v>` access operands).
fn ndigits(v: usize) usize {
    var d: usize = 1;
    var x = v;
    while (x >= 10) : (x /= 10) d += 1;
    return d;
}

/// Build the M11 retention-test branchy self-loop BODY (no `(c ...)` wrapper)
/// into `buf`, returning the plain slice.  Arity 2: loop(n, a) binds K
/// let-values — each a cons chain of L literal conses — on ODD n, and
/// tail-jumps with env_len == 2 on EVEN n.  The odd path leaves the K lets in
/// the reused env array's [2..cap) tail, exactly the shape a missing tail-clear
/// would retain through the drain's full-capacity VALUE_ARRAY scan.
fn branchyBody(buf: []u8, K: usize, L: usize) []const u8 {
    const BIND: usize = 24; // odd-path start (after the even tail-call path)
    const odd_tail: usize = 9; // m n1 a<K> P+ n1 a<K+1> P- g branch t
    const RET: usize = BIND + K * (2 * L + 2) + odd_tail;

    var n: usize = 0;
    const put = struct {
        fn f(b: []u8, i: *usize, s: []const u8) void {
            @memcpy(b[i.*..][0..s.len], s);
            i.* += s.len;
        }
    }.f;
    const putNum = struct {
        fn f(b: []u8, i: *usize, v: usize) void {
            const s = std.fmt.bufPrint(b[i.*..], "{d}", .{v}) catch unreachable;
            i.* += s.len;
        }
    }.f;

    // 0..6: base-case check; the j[<len>:n]<RET> at index 6 skips to the ret.
    put(buf, &n, "rn[1:n]0a[1:n]1P[1:s]=f[1:n]7a[1:n]0j[");
    putNum(buf, &n, ndigits(RET));
    put(buf, &n, ":n]");
    putNum(buf, &n, RET);
    // 7..14: parity check (n == (n/2)*2); jmpf to BIND when odd.
    put(buf, &n, "n[1:n]2a[1:n]1P[1:s]/n[1:n]2P[1:s]*a[1:n]1P[1:s]=f[");
    putNum(buf, &n, ndigits(BIND));
    put(buf, &n, ":n]");
    putNum(buf, &n, BIND);
    // 15..23: EVEN path — no lets, tail-call branch(n-1, a+1).
    put(buf, &n, "mn[1:n]1a[1:n]0P[1:s]+n[1:n]1a[1:n]1P[1:s]-g[6:s]brancht");
    // ODD path: K lets, each a chain of L literal conses.
    var k: usize = 0;
    while (k < K) : (k += 1) {
        put(buf, &n, "n[1:n]1n[1:n]1P[4:s]cons");
        var l: usize = 1;
        while (l < L) : (l += 1) put(buf, &n, "n[1:n]1P[4:s]cons");
        put(buf, &n, "e");
    }
    // Odd tail call reads n/a at the fixed post-let offsets (access K / K+1).
    put(buf, &n, "mn[1:n]1a[");
    putNum(buf, &n, ndigits(K));
    put(buf, &n, ":n]");
    putNum(buf, &n, K);
    put(buf, &n, "P[1:s]+n[1:n]1a[");
    putNum(buf, &n, ndigits(K + 1));
    put(buf, &n, ":n]");
    putNum(buf, &n, K + 1);
    put(buf, &n, "P[1:s]-g[6:s]branchtv");
    return buf[0..n];
}

test "M11 tail-env reuse: deep self-recursion reuses one env array" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // loop(n, acc) = if n==0 acc else loop(n-1, acc+1) — a defun-registered
    // closure whose body tail-calls itself via g[loop] (appterm N==A).
    // Body [r n0 a1 P= f7 a0 j16 m n1 a0 P+ n1 a1 P- g loop t v].
    const loop = try expectRunVal(&g, &v,
        "(c(rn[1:n]0a[1:n]1P[1:s]=f[1:n]7a[1:n]0j[2:n]16mn[1:n]1a[1:n]0P[1:s]+n[1:n]1a[1:n]1P[1:s]-g[4:s]looptv))");
    try std.testing.expectEqual(types.ValTag.lambda, loop.tag);
    v.defunSet("loop", loop);

    // loop(2000, 0) == 2000; all 2000 tail calls reuse the apply-built env
    // array (misses stay 0 — the apply path's initial alloc is not counted).
    try expectRunNum(&g, &v, "(mn[1:n]0n[4:n]2000g[4:s]loopp)", 2000);
    try std.testing.expectEqual(@as(u64, 2000), v.env_reuse_hits);
    try std.testing.expectEqual(@as(u64, 0), v.env_reuse_misses);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M11 tail-env reuse: closure args survive scavenges (aliasing probe)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    const id = try expectRunVal(&g, &v, "(c(a[1:n]0v))");
    v.defunSet("id", id);

    // loop(n, f): if n==0 apply f to 0; else tail-call loop(n-1, cur(identity)).
    // The argbuf carries a FRESH closure each iteration whose env is a valLambda
    // COPY of the loop env, so the reused array's write barrier must keep the
    // closure chain alive across scavenges.
    const loop = try expectRunVal(&g, &v,
        "(c(rn[1:n]0a[1:n]1P[1:s]=f[2:n]10mn[1:n]0a[1:n]0pvmc(a[1:n]0v)n[1:n]1a[1:n]1P[1:s]-g[4:s]looptv))");
    v.defunSet("loop", loop);

    var i: usize = 0;
    while (i < 3) : (i += 1) {
        try expectRunNum(&g, &v, "(mg[2:s]idn[4:n]3000g[4:s]loopp)", 0);
        g.collectNursery(.@"test");
    }
    try std.testing.expectEqual(@as(u64, 9000), v.env_reuse_hits);
    try std.testing.expectEqual(@as(u64, 0), v.env_reuse_misses);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M11 tail-env reuse: stale tail cleared (retention regression)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // branch(n, a): ODD n binds 30 let-values (each a 30-cons chain — big
    // garbage) then tail-jumps; EVEN n tail-jumps with env_len == 2.  The odd
    // path grows the env array to cap 64 and the even path reuses it with
    // env_len == 2, leaving [2..cap) stale unless the tail-clear nils it.
    var body_buf: [16384]u8 = undefined;
    const body = branchyBody(&body_buf, 30, 30);
    var prog: [16384]u8 = undefined;
    const src = std.fmt.bufPrintZ(&prog, "(c({s}))", .{body}) catch unreachable;
    const branch = try expectRunVal(&g, &v, src);
    v.defunSet("branch", branch);

    // branch(200, 0) == 200.  The last odd iteration (n==1) binds the 30
    // cons-chains and tail-jumps; the base case then leaves the env array dead
    // but still in the remembered set (old-gen + dirty from those stores).
    try expectRunNum(&g, &v, "(mn[1:n]0n[3:n]200g[6:s]branchp)", 200);
    const before = g.allocatedPages();
    g.collectNursery(.@"test");
    // The post-run scavenge scans the dead-but-dirty env array by FULL capacity:
    // a missing tail-clear would promote the 30 stale cons-chains (~84 pages),
    // while the cleared tail promotes nothing.
    const after = g.allocatedPages();
    try std.testing.expect(after - before <= 16);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M11 tail-env reuse: mutual recursion with differing arities" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // f(n, a) arity 2: if n==0 a else g(n-1, a+1, 0); g(n, a, b) arity 3:
    // if n==0 a+b else f(n-1, a+1).  Arities alternate 2,3,2,3… so the retained
    // env array (cap 3) is reused both when the callee env shrinks (3->2, tail
    // cleared) and grows back (2->3) — one miss on the first 2->3 transition.
    const f = try expectRunVal(&g, &v,
        "(c(rn[1:n]0a[1:n]1P[1:s]=f[1:n]7a[1:n]0j[2:n]17mn[1:n]0n[1:n]1a[1:n]0P[1:s]+n[1:n]1a[1:n]1P[1:s]-g[1:s]gtv))");
    v.defunSet("f", f);
    const gg = try expectRunVal(&g, &v,
        "(c(rrn[1:n]0a[1:n]2P[1:s]=f[2:n]10a[1:n]0a[1:n]1P[1:s]+j[2:n]19mn[1:n]1a[1:n]1P[1:s]+n[1:n]1a[1:n]2P[1:s]-g[1:s]ftv))");
    v.defunSet("g", gg);

    // f(1000, 0) == 1000.
    try expectRunNum(&g, &v, "(mn[1:n]0n[4:n]1000g[1:s]fp)", 1000);
    // One miss (first f->g grows the env 2->3); every later tail call reuses.
    try std.testing.expectEqual(@as(u64, 999), v.env_reuse_hits);
    try std.testing.expectEqual(@as(u64, 1), v.env_reuse_misses);
    try std.testing.expectEqual(wm0, g.rootWatermark());
}

test "M11 tail-env reuse: interactions (peel, trap-error, partial)" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();
    const wm0 = g.rootWatermark();

    // (1) N>A peel inside a tail loop: each iteration over-applies a 2-param
    // fn to 3 args (result 7) and adds it to the accumulator — loop(100, 0)
    // == 700.  The peel runs a NESTED vmExecEnv whose init_env is a fresh
    // array, so the loop's reused env array is never aliased.
    const loop = try expectRunVal(&g, &v,
        "(c(rn[1:n]0a[1:n]1P[1:s]=f[1:n]7a[1:n]0j[2:n]21ma[1:n]0mn[1:n]9n[1:n]8n[1:n]7c(rc(a[1:n]2v)v)pP[1:s]+n[1:n]1a[1:n]1P[1:s]-g[4:s]looptv))");
    v.defunSet("loop", loop);
    try expectRunNum(&g, &v, "(mn[1:n]0n[3:n]100g[4:s]loopp)", 700);

    // (2) trap-error inside a tail-loop body: each iteration catches a
    // simple-error (handler returns 1) and adds it — loop2(50, 0) == 50.
    const loop2 = try expectRunVal(&g, &v,
        "(c(rn[1:n]0a[1:n]1P[1:s]=f[1:n]7a[1:n]0j[2:n]19ma[1:n]0c(n[1:n]1v)c(mS[4:S]oopsg[12:s]simple-errorpv)g[10:s]trap-errorpP[1:s]+n[1:n]1a[1:n]1P[1:s]-g[5:s]loop2tv))");
    v.defunSet("loop2", loop2);
    try expectRunNum(&g, &v, "(mn[1:n]0n[2:n]50g[5:s]loop2p)", 50);

    // (3) N<A partial built in TAIL position then applied later: mk(x) tail-
    // calls add with 1 arg (buildPartialClosure copies the env), returning
    // add(x); applying it to 6 yields 11.
    const add = try expectRunVal(&g, &v, "(c(rma[1:n]1a[1:n]0g[1:s]+pv))");
    v.defunSet("add", add);
    const mk = try expectRunVal(&g, &v, "(c(ma[1:n]0g[3:s]addt))");
    v.defunSet("mk", mk);
    const p5 = try expectRunVal(&g, &v, "(mn[1:n]5g[2:s]mkp)");
    try std.testing.expectEqual(types.ValTag.lambda, p5.tag);
    v.defunSet("p5", p5);
    try expectRunNum(&g, &v, "(mn[1:n]6g[2:s]p5p)", 11);

    try std.testing.expectEqual(wm0, g.rootWatermark());
}

// =====================================================================
//  Ported from shen — M2 marshal/demarshal + eval-kl (no-bundle) and the
//  M3 wait/kill process gate.  Self-contained (no bundle artifact); the
//  bundle-driven M2 zinctest parity cases stay shen-side.
// =====================================================================

// ---- raw libc for the M3 wait/kill test ----
// fork/_exit/nanosleep: the test binary links libc (via the vm module).  The
// fork/_exit externs are file-private in execplan.zig, so re-declare here.
const Timespec = extern struct { sec: isize, nsec: isize };
extern "c" fn nanosleep(req: *const Timespec, rem: ?*Timespec) c_int;
extern "c" fn fork() c_int;
extern "c" fn _exit(code: c_int) noreturn;

fn sleepSec(sec: isize) void {
    const ts = Timespec{ .sec = sec, .nsec = 0 };
    _ = nanosleep(&ts, null);
}

fn sleepMs(ms: u64) void {
    const ts = Timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    _ = nanosleep(&ts, null);
}

/// Deterministic LCG for the property test (std.rand is thread-scoped and
/// heavyweight for a test; a plain PCG-style step is enough here).
fn m2NextRand(rng: *u64) u64 {
    rng.* = rng.* *% 6364136223846793005 +% 1442695040888963407;
    return rng.* >> 33;
}

/// Build a random tree with leaves from {number, string, boolean, nil,
/// symbol foo|bar|quux} and cons nodes.  SYMBOLS ARE SAFE IN CAR POSITIONS:
/// the demarshal protocol only special-cases the five reserved tag symbols
/// (number/symbol/string/boolean/cons) and 'mark' — none of which this
/// generator can emit — so every generated tree is a fixed point of
/// demarshal∘marshal (the round-trip property under test).  The car is
/// rooted across the cdr build (valCons roots its own params internally).
fn m2BuildTree(v: *state.Vm, rng: *u64, depth: u32, counter: *u32) types.Value {
    const g = v.gc;
    if (depth == 0 or m2NextRand(rng) % 5 == 0) {
        switch (m2NextRand(rng) % 5) {
            0 => return values.valNumber(@intCast(m2NextRand(rng) % 1000)),
            1 => {
                var buf: [24]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "s{d}", .{counter.*}) catch unreachable;
                counter.* += 1;
                return values.valString(g, s);
            },
            2 => return values.valBoolean(m2NextRand(rng) % 2 == 0),
            3 => return values.valNil(),
            else => return switch (m2NextRand(rng) % 3) {
                0 => symbols.valSymbol(&v.symbols, "foo"),
                1 => symbols.valSymbol(&v.symbols, "bar"),
                else => symbols.valSymbol(&v.symbols, "quux"),
            },
        }
    }
    var car = m2BuildTree(v, rng, depth - 1, counter);
    var car_guard = g.rootValue(&car);
    defer car_guard.end();
    const cdr = m2BuildTree(v, rng, depth - 1, counter);
    return values.valCons(g, car, cdr);
}

/// Build a proper list [i0 i1 ... ik] from values (the eval-kl form
/// builder).  The items are copied into a rooted buffer so string items
/// survive the valCons churn (numbers/symbols carry no GC interiors, but
/// strings do).
fn m2List(g: *heap.Gc, items: []const types.Value) types.Value {
    var buf: [16]types.Value = undefined;
    var n: i32 = @intCast(items.len);
    for (items, 0..) |it, idx| buf[idx] = it;
    g.rootPushValueArray(&buf, &n);
    defer g.rootPop();
    var head = values.valNil();
    var guard = g.rootValue(&head);
    defer guard.end();
    var i: usize = items.len;
    while (i > 0) {
        i -= 1;
        head = values.valCons(g, buf[i], head);
    }
    return head;
}

test "M2 marshal/demarshal: scalar tags, [cons] empty, mark, passthroughs" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // [number 5] shape: cons(symbol number, cons(5, nil)); round-trips.
    const m5 = vm.marshal.marshalToTagged(&v, values.valNumber(5));
    try std.testing.expectEqual(types.ValTag.cons, m5.tag);
    try std.testing.expectEqualStrings("number", values.symSlice(m5.payload.cons.car.?.*));
    const cdr5 = m5.payload.cons.cdr.?.*;
    try std.testing.expectEqual(@as(i64, 5), cdr5.payload.cons.car.?.payload.number);
    try std.testing.expectEqual(types.ValTag.nil, cdr5.payload.cons.cdr.?.tag);
    try std.testing.expectEqual(@as(i64, 5), vm.marshal.demarshalFromTagged(&v, m5).payload.number);

    // Strings / symbols / booleans round-trip through their tags.
    const ms = vm.marshal.marshalToTagged(&v, values.valString(&g, "ab"));
    try std.testing.expectEqualStrings("ab", values.strSlice(vm.marshal.demarshalFromTagged(&v, ms)));
    const msym = vm.marshal.marshalToTagged(&v, symbols.valSymbol(&v.symbols, "foo"));
    try std.testing.expectEqualStrings("foo", values.symSlice(vm.marshal.demarshalFromTagged(&v, msym)));
    const mb = vm.marshal.marshalToTagged(&v, values.valBoolean(true));
    try std.testing.expectEqual(@as(i64, 1), vm.marshal.demarshalFromTagged(&v, mb).payload.boolean);

    // nil marshals to [cons] (symbol cons + nil cdr) and demarshals back.
    const mnil = vm.marshal.marshalToTagged(&v, values.valNil());
    try std.testing.expectEqualStrings("cons", values.symSlice(mnil.payload.cons.car.?.*));
    try std.testing.expectEqual(types.ValTag.nil, mnil.payload.cons.cdr.?.tag);
    try std.testing.expectEqual(types.ValTag.nil, vm.marshal.demarshalFromTagged(&v, mnil).tag);

    // mark marshals to the SYMBOL 'mark; the symbol 'mark demarshals to nil.
    const mmark = vm.marshal.marshalToTagged(&v, values.valMark());
    try std.testing.expectEqual(types.ValTag.symbol, mmark.tag);
    try std.testing.expectEqualStrings("mark", values.symSlice(mmark));
    try std.testing.expectEqual(types.ValTag.nil, vm.marshal.demarshalFromTagged(&v, symbols.valSymbol(&v.symbols, "mark")).tag);

    // Lambdas / vectors / errors pass through BOTH directions unchanged.
    const lam = values.valLambda(&g, null, 0, null, 0);
    try std.testing.expectEqual(types.ValTag.lambda, vm.marshal.marshalToTagged(&v, lam).tag);
    try std.testing.expectEqual(types.ValTag.lambda, vm.marshal.demarshalFromTagged(&v, lam).tag);
    const vec = values.valVector(&g, 2);
    try std.testing.expectEqual(types.ValTag.vector, vm.marshal.marshalToTagged(&v, vec).tag);
    const errv = values.valError(&g, "boom");
    try std.testing.expectEqual(types.ValTag.error_, vm.marshal.demarshalFromTagged(&v, errv).tag);

    // marshal of a cons is the 3-ELEMENT LIST [cons X Y] with RAW car/cdr
    // (the no-recursion rule, C:763-767): cadr is the raw number 1, and the
    // actual cdr rides in a SINGLETON wrapper (the 3rd element is (2 nil)).
    var pair = values.valCons(&g, values.valNumber(1), values.valNumber(2));
    var pair_guard = g.rootValue(&pair);
    defer pair_guard.end();
    const mp = vm.marshal.marshalToTagged(&v, pair);
    try std.testing.expectEqualStrings("cons", values.symSlice(mp.payload.cons.car.?.*));
    const mp_cdr = mp.payload.cons.cdr.?.*; // (1 (2 nil))
    try std.testing.expectEqual(@as(i64, 1), mp_cdr.payload.cons.car.?.payload.number); // RAW 1
    const wrapper = mp_cdr.payload.cons.cdr.?.*; // ((2 nil))
    try std.testing.expectEqual(types.ValTag.cons, wrapper.tag);
    try std.testing.expectEqual(@as(i64, 2), wrapper.payload.cons.car.?.payload.number); // RAW 2
    try std.testing.expectEqual(types.ValTag.nil, wrapper.payload.cons.cdr.?.tag);
    // ...and demarshal rebuilds the DOTTED pair cons(1 . 2).
    const back = vm.marshal.demarshalFromTagged(&v, mp);
    try std.testing.expect(values.deepEqual(pair, back, 0));
    try std.testing.expectEqual(types.ValTag.number, back.payload.cons.cdr.?.tag);
}

test "M2 marshal/demarshal round-trip property: random nested cons trees" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    var rng: u64 = 0x5eed_cafe_f00d;
    var counter: u32 = 0;
    var iter: usize = 0;
    while (iter < 64) : (iter += 1) {
        var tree = m2BuildTree(&v, &rng, 4, &counter);
        var tree_guard = g.rootValue(&tree);
        defer tree_guard.end();

        var tagged = vm.marshal.marshalToTagged(&v, tree);
        var tagged_guard = g.rootValue(&tagged);
        defer tagged_guard.end();

        // Churn + a forced scavenge BETWEEN marshal and demarshal: the
        // tagged form survives only via its root (its cons car/cdr interior
        // pointers must be re-read fresh through the rooted slot).
        var junk = values.valNil();
        var junk_guard = g.rootValue(&junk);
        defer junk_guard.end();
        var k: usize = 0;
        while (k < 200) : (k += 1)
            junk = values.valCons(&g, values.valNumber(@intCast(k)), junk);
        g.collectNursery(.@"test");

        var back = vm.marshal.demarshalFromTagged(&v, tagged);
        var back_guard = g.rootValue(&back);
        defer back_guard.end();
        try std.testing.expect(values.deepEqual(tree, back, 0));
    }
}

test "M2 eval-kl without a bundle: missing closure returns the input form" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // No loadBundle: extract-kl resolves to a bare symbol (defunGet's
    // fallback), the stage warns on stderr, and eval-kl's acc is the INPUT
    // FORM (C goto eval_kl_done with result = a).
    var form = m2List(&g, &.{ symbols.valSymbol(&v.symbols, "+"), values.valNumber(1), values.valNumber(2) });
    var acc: types.Value = values.valNil();
    try primExecRooted(&g, &v, "eval-kl", &form, &acc);
    try std.testing.expect(values.deepEqual(form, acc, 0));

    // Scalar forms too (marshal still runs first, then the fallback).
    var form2 = values.valNumber(42);
    try primExecRooted(&g, &v, "eval-kl", &form2, &acc);
    try std.testing.expectEqual(@as(i64, 42), acc.payload.number);
}

test "M3 gate: wait returns the child's exit code; kill -> 128+sig" {
    var g = try testInit();
    defer g.deinit();
    var v: state.Vm = undefined;
    v.init(&g);
    defer v.deinit();

    // wait: fork -> child _exit(42) -> primWait reports 42.
    {
        const pid = fork();
        try std.testing.expect(pid != -1);
        if (pid == 0) _exit(42);
        var acc: types.Value = values.valNil();
        try primExec(&g, &v, "wait", &.{values.valNumber(pid)}, &acc);
        try std.testing.expectEqual(types.ValTag.number, acc.tag);
        try std.testing.expectEqual(@as(i64, 42), acc.payload.number);
    }

    // kill: fork -> child sleeps 30s -> SIGKILL (9) -> wait -> 128+9 = 137.
    {
        const pid = fork();
        try std.testing.expect(pid != -1);
        if (pid == 0) {
            // The child never touches the GC heap — it only sleeps (the
            // same discipline the runner's children follow).
            sleepSec(30);
            _exit(0);
        }
        sleepMs(100); // let the child reach nanosleep
        var kacc: types.Value = values.valNil();
        try primExec(&g, &v, "kill", &.{ values.valNumber(pid), values.valNumber(9) }, &kacc);
        try std.testing.expectEqual(types.ValTag.boolean, kacc.tag);
        var wacc: types.Value = values.valNil();
        try primExec(&g, &v, "wait", &.{values.valNumber(pid)}, &wacc);
        try std.testing.expectEqual(@as(i64, 137), wacc.payload.number);
    }
}
