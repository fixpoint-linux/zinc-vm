//! src/vm/parser.zig — csexp bytecode parser + resolve_jumps + print_instr
//! (milestone M3).
//!
//! C origin: zincvm.c:2800-3004 (skip_ws, parse_int, parse_csexp_atom with
//! scratch mode, parse_body with cc_slots side-slot rooting, parse_csexp_list,
//! parse_bytecode) + 3089-3101 (resolve_jumps) + 3010-3040 (print_instr debug
//! aid).
//!
//! THE SUBTLETY (port faithfully, C comments 2882-2965): parse_body builds a
//! page_allocator scratch Instr buffer whose operand strings are C-heap
//! malloc'd in scratch mode (non-moving, GC-invisible); each OP_CUR allocates a
//! stable one-word malloc'd slot for the child code array and pushes it on the
//! shadow stack (rootPushPtr) for the WHOLE parse_body call so nested recursive
//! parses keep every child closure_code reachable; after the loop we re-sync
//! scratch[i].closure_code from the rooted slots (a scavenge during a nested
//! parse may have moved children — *slot updated, scratch copy stale), THEN
//! gc_alloc the final instr_array, bulk-copy, rootPushPtr(&code), re-wrap
//! operand VAL_STRINGs malloc->gc.allocRaw one by one, pop code root, pop+free
//! the cc slots.
//!
//! Error handling: C uses setjmp/longjmp (PARSE_ERROR); Zig uses error.ParseError
//! with defer/errdefer-driven cleanup.  On error, parse_bytecode restores the GC
//! shadow stack to parse_wm (C `gc_root_pop_to(parse_wm)`) via errdefer, and each
//! parse_body frame frees its own scratch buffer + operand strings + cc slots via
//! a completion-flag defer.  This is strictly leak-free versus C's innermost-only
//! cleanup.  The shadow stack is left balanced: rootWatermark unchanged after a
//! failed parse.
//!
//! parse_bytecode takes the symbol interner (val_symbol) and the collector
//! (val_string / gc_alloc) explicitly rather than reading C globals.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const values = @import("values.zig");
const symbols = @import("symbols.zig");
const state = @import("state.zig");

const Gc = gc.Gc;
const Instr = types.Instr;
const Value = types.Value;
const SymbolInterner = symbols.SymbolInterner;

pub const ParseError = error{ ParseError };

const ParseState = struct {
    input: [:0]const u8,
    p: usize = 0,
    start: usize = 0,
    scratch: i32 = 0, // 1 = produce C-heap operand strings for scratch buffer
    scratch_buf: ?[*]Instr = null,
    scratch_len: i32 = 0,
    cc_slots: ?[]*?[*]Instr = null, // growable array of malloc'd Instr** slots
    cc_len: usize = 0,
    cc_cap: usize = 0,
};

/// Current byte (C `*ps->p`; the `[:0]` sentinel gives the C NUL at end).
inline fn cur(ps: *const ParseState) u8 {
    return ps.input[ps.p];
}
inline fn advance(ps: *ParseState) void {
    ps.p += 1;
}

/// C: zincvm.c:2800-2802 skip_ws.
fn skipWs(ps: *ParseState) void {
    while (std.ascii.isWhitespace(cur(ps))) advance(ps);
}

/// C: zincvm.c:2803-2808 parse_int.
fn parseInt(ps: *ParseState) ParseError!i32 {
    var n: i32 = 0;
    if (!std.ascii.isDigit(cur(ps))) return error.ParseError; // expected digit
    while (std.ascii.isDigit(cur(ps))) {
        n = n * 10 + (cur(ps) - '0');
        advance(ps);
    }
    return n;
}

/// C: zincvm.c:2809-2843 parse_csexp_atom.  In scratch mode (inside a body) a
/// VAL_STRING's data is malloc'd (C-heap, non-moving, GC-invisible); outside
/// scratch mode it is GC-allocated via val_string.  pub for parse_bundle (M6).
pub fn parseCsexpAtom(ps: *ParseState, g: *Gc, sym: *SymbolInterner) ParseError!Value {
    const a = std.heap.page_allocator;
    skipWs(ps);
    if (cur(ps) != '[') return error.ParseError; // expected '[' for csexp atom
    advance(ps);
    const len = try parseInt(ps);
    if (cur(ps) != ':') return error.ParseError; // expected ':' after length
    advance(ps);
    const ty = cur(ps);
    advance(ps);
    if (cur(ps) != ']') return error.ParseError; // expected ']' after type
    advance(ps);
    if (len < 0) return error.ParseError; // negative length

    const ulen: usize = @intCast(len);
    const buf = a.alloc(u8, ulen + 1) catch unreachable;
    defer a.free(buf);
    @memcpy(buf[0..ulen], ps.input[ps.p .. ps.p + ulen]);
    buf[ulen] = 0;
    // C advances ps->p by len past the copied data.  len >= 0 checked above.
    ps.p += @as(usize, @intCast(len));

    var v: Value = undefined;
    switch (ty) {
        's' => v = symbols.valSymbol(sym, buf[0..ulen]),
        'n' => v = values.valNumber(std.fmt.parseInt(i64, buf[0..ulen], 10) catch 0),
        'S' => {
            if (ps.scratch != 0) {
                // scratch mode: C-heap, non-moving operand string.
                const data = a.alloc(u8, ulen + 1) catch unreachable;
                @memcpy(data[0..ulen], buf[0..ulen]);
                data[ulen] = 0;
                v = .{ .tag = .string, .payload = .{ .str = .{ .data = @ptrCast(data), .len = len } } };
            } else {
                v = values.valString(g, buf[0..ulen]);
            }
        },
        'b' => v = values.valBoolean(std.mem.eql(u8, buf[0..ulen], "true")),
        'F' => v = values.valFloat(std.fmt.parseFloat(f64, buf[0..ulen]) catch 0),
        else => return error.ParseError, // unknown csexp type
    }
    return v;
}

/// C: zincvm.c:2846-2974 parse_body.  See module doc for the cc_slots rooting
/// protocol.  Returns the code length and writes the GC-managed Instr array
/// head into `out`.
fn parseBody(ps: *ParseState, g: *Gc, sym: *SymbolInterner, out: *?[*]Instr) ParseError!i32 {
    const a = std.heap.page_allocator;
    var scratch = std.ArrayListUnmanaged(Instr).empty;
    var cc = std.ArrayListUnmanaged(*?[*]Instr).empty;
    var completed = false;
    // On the error path (not completed), free THIS frame's C-heap scratch
    // operand strings + scratch buffer + cc slots + cc array.  Root pops are
    // handled by parse_bytecode's rootPopTo(parse_wm), so none are done here.
    defer if (!completed) {
        for (scratch.items) |ins| {
            if (ins.operand.tag == .string)
                a.free(ins.operand.payload.str.data.?[0 .. @as(usize, @intCast(ins.operand.payload.str.len)) + 1]);
        }
        scratch.deinit(a);
        for (cc.items) |slot| a.destroy(slot);
        cc.deinit(a);
    };

    // Save caller's scratch state; set ours for nested parsing.
    const saved_scratch = ps.scratch;
    const saved_scratch_buf = ps.scratch_buf;
    const saved_scratch_len = ps.scratch_len;
    ps.scratch = 1;
    ps.scratch_buf = scratch.items.ptr;
    ps.scratch_len = 0;

    // Save caller's cc_slots state; this call builds its own.
    const saved_cc = ps.cc_slots;
    const saved_cc_len = ps.cc_len;
    const saved_cc_cap = ps.cc_cap;
    ps.cc_slots = null;
    ps.cc_len = 0;
    ps.cc_cap = 0;
    defer {
        ps.scratch = saved_scratch;
        ps.scratch_buf = saved_scratch_buf;
        ps.scratch_len = saved_scratch_len;
        ps.cc_slots = saved_cc;
        ps.cc_len = saved_cc_len;
        ps.cc_cap = saved_cc_cap;
    }

    while (true) {
        skipWs(ps);
        const c = cur(ps);
        if (c == ')' or c == 0) break;
        if (c == '(') return error.ParseError; // unexpected nested list in body

        var instr: Instr = std.mem.zeroes(Instr);
        instr.op = types.charToOpcode(c);
        advance(ps);
        switch (c) {
            'm', 'p', 'r', 'v', 'e', 'd', 't' => {}, // no operand
            'a', 'f', 'j', 'n', 'g', 's', 'P', 'S', 'b', 'F' => {
                instr.operand = try parseCsexpAtom(ps, g, sym);
            },
            'c' => {
                skipWs(ps);
                if (cur(ps) != '(') return error.ParseError; // expected '(' after 'c'
                advance(ps);
                // Stable one-word slot for the child closure_code, pushed on
                // the shadow stack for the whole parse_body (C:2882-2896).
                const slot = a.create(?[*]Instr) catch unreachable;
                cc.append(a, slot) catch unreachable;
                ps.cc_slots = cc.items;
                ps.cc_len = cc.items.len;
                ps.cc_cap = cc.capacity;
                g.rootPushPtr(@ptrCast(slot));
                instr.closure_len = try parseBody(ps, g, sym, slot);
                instr.closure_code = @ptrCast(slot.*);
                if (cur(ps) != ')') return error.ParseError; // expected ')' after cur body
                advance(ps);
            },
            else => return error.ParseError, // unknown opcode
        }
        scratch.append(a, instr) catch unreachable;
        ps.scratch_len = @intCast(scratch.items.len);
        ps.scratch_buf = scratch.items.ptr;
    }

    // Re-sync closure_code pointers from rooted side-slots into the scratch
    // buffer before the gc_alloc below (C:2915-2928).  A scavenge during a
    // nested parse may have moved a child, updating *slot but leaving the
    // scratch copy stale.
    var si: usize = 0;
    for (scratch.items) |*ins| {
        if (ins.op == .cur) {
            ins.closure_code = @ptrCast(cc.items[si].*);
            si += 1;
        }
    }

    // Allocate final GC-managed Instr array and bulk-copy (C:2930-2932).
    const code = g.allocArray(Instr, scratch.items.len);
    @memcpy(code[0..scratch.items.len], scratch.items);

    // Pin code across the GC_STR calls in the re-wrap loop (C:2934-2937).
    var code_slot: ?[*]Instr = code;
    g.rootPushPtr(@ptrCast(&code_slot));

    // Re-wrap VAL_STRING operand strings: malloc -> GC (C:2939-2954).
    for (code[0..scratch.items.len]) |*ins| {
        if (ins.operand.tag == .string) {
            const old_data = ins.operand.payload.str.data.?;
            const slen: usize = @intCast(ins.operand.payload.str.len);
            const new_data = g.allocRaw(slen + 1);
            @memcpy(new_data[0..slen], old_data[0..slen]);
            new_data[slen] = 0;
            ins.operand.payload.str.data = @ptrCast(new_data);
            a.free(old_data[0 .. slen + 1]);
        }
    }

    g.rootPop(); // code_slot

    // Pop and free the N cc slots (C:2956-2965).  Pushed order slot0..slotN
    // then &code; code popped above, now pop the N slots LIFO.
    const result_len: i32 = @intCast(scratch.items.len);
    for (cc.items) |_| g.rootPop();
    for (cc.items) |slot| a.destroy(slot);
    cc.deinit(a);
    scratch.deinit(a);

    completed = true;
    out.* = code;
    return result_len;
}

/// C: zincvm.c:2975-2983 parse_csexp_list.  pub for parse_bundle (M6).
pub fn parseCsexpList(ps: *ParseState, g: *Gc, sym: *SymbolInterner, out: *?[*]Instr) ParseError!i32 {
    skipWs(ps);
    if (cur(ps) != '(') return error.ParseError; // expected '(' for list
    advance(ps);
    const len = try parseBody(ps, g, sym, out);
    if (cur(ps) != ')') return error.ParseError; // expected ')' after list body
    advance(ps);
    return len;
}

/// C: zincvm.c:2984-3004 parse_bytecode.  Parses `str` (a csexp list) into a
/// GC-managed Instr array, writing its head into `out` and returning the length.
/// On error, returns error.ParseError, sets *out = null, and restores the GC
/// shadow stack to the entry watermark (leaving it balanced).
pub fn parseBytecode(g: *Gc, sym: *SymbolInterner, str: [:0]const u8, out: *?[*]Instr) ParseError!i32 {
    var ps = ParseState{ .input = str };
    const parse_wm = g.rootWatermark();
    errdefer g.rootPopTo(parse_wm);
    const len = parseCsexpList(&ps, g, sym, out) catch {
        out.* = null;
        return error.ParseError;
    };
    return len;
}

// ---------------------------------------------------------------------
//  parse_bundle — C: zincvm.c:3770-3857 (M6)
// ---------------------------------------------------------------------

/// C: zincvm.c:3770-3857 parse_bundle.  Parses a bundle
/// `((name1 code1) (name2 code2) ...)` — each entry is a
/// (name_csexp_atom, code_csexp_list) pair — and registers each entry as a
/// defun: the code list must be a single `cur` wrapper whose closure_code
/// becomes the lambda body (resolve_jumps applied, empty env).  Returns the
/// number of entries loaded: 0 on the outer error, the partial count on a
/// mid-bundle failure (C semantics — errors are printed, never thrown).
///
/// Error/cleanup parity: the C setjmp handler frees parse_body's scratch and
/// pops the shadow stack to parse_wm; here parse_body's own completion-flag
/// defer frees its scratch and the rootPopTo below rebalances any roots a
/// failed nested parse left pushed — every exit path leaves the watermark
/// where it entered.
///
/// Rooting: within an entry, parseCsexpList balances its own cc-slot roots.
/// Between the parse and defunSet no GC allocation happens (resolve_jumps is
/// in place; val_lambda with env_len == 0 takes the no-alloc path; defun_set
/// is C-heap only) — but the body array is pinned across the valLambda call
/// via a one-slot rootPushPtr anyway, so the closure's code operand is read
/// post-GC fresh even if val_lambda ever allocates.  Once defunSet registers
/// the closure, the body array stays reachable through the dirty-marked
/// defun table across later entries' parses (each of which allocates).
pub fn parseBundle(g: *Gc, sym: *SymbolInterner, v: *state.Vm, str: [:0]const u8) i32 {
    var ps = ParseState{ .input = str };
    const parse_wm = g.rootWatermark();
    defer g.rootPopTo(parse_wm); // C:3788 longjmp path; no-op when balanced

    skipWs(&ps);
    if (cur(&ps) != '(') {
        std.debug.print("bundle error: expected outer '('\n", .{});
        return 0;
    }
    advance(&ps);

    var count: i32 = 0;
    while (true) {
        skipWs(&ps);
        if (cur(&ps) == ')') {
            advance(&ps);
            break; // end of bundle
        }
        if (cur(&ps) != '(') {
            std.debug.print("bundle error: expected '(' for entry\n", .{});
            return count;
        }
        advance(&ps);

        // Parse the name atom (C:3811-3818).  Interner strings are immortal
        // C-heap, so `name` needs no rooting.  Keep the full safe.* name —
        // primitives stay under short names.
        const name_val = parseCsexpAtom(&ps, g, sym) catch {
            std.debug.print("bundle error: name atom parse failed\n", .{});
            return count;
        };
        if (name_val.tag != .symbol) {
            std.debug.print("bundle error: name must be a symbol\n", .{});
            return count;
        }
        const name = values.symSlice(name_val);

        // Parse the code list — a cur wrapping the closure body (C:3822-3827).
        var code: ?[*]Instr = null;
        const code_len = parseCsexpList(&ps, g, sym, &code) catch {
            std.debug.print("bundle error: failed to parse code for '{s}'\n", .{name});
            return count;
        };
        if (code_len <= 0 or code == null) {
            std.debug.print("bundle error: failed to parse code for '{s}'\n", .{name});
            return count;
        }

        // Unwrap the outer cur: its closure_code is the lambda body
        // (C:3829-3835).  All wrapper reads happen before any potential alloc.
        const wrapper = &code.?[0];
        if (wrapper.op != .cur or wrapper.closure_code == null) {
            std.debug.print("bundle error: expected cur wrapper for '{s}'\n", .{name});
            return count;
        }
        const body_code = wrapper.closure_code;
        const body_len = wrapper.closure_len;

        // Resolve jumps in the body (C:3837-3838) — in place, no alloc.
        resolveJumps(@ptrCast(body_code.?), body_len);

        // Create a closure from the body code (empty env) and store it in
        // the global table (C:3840-3842).
        var body_slot: ?*Instr = body_code;
        g.rootPushPtr(@ptrCast(&body_slot));
        defer g.rootPop(); // body_slot — per-iteration scope
        const closure = values.valLambda(g, body_slot, body_len, null, 0);
        v.defunSet(name, closure);

        // Consume the closing ')' of the entry (C:3845-3850).
        skipWs(&ps);
        if (cur(&ps) != ')') {
            std.debug.print("bundle error: expected ')' to close entry '{s}'\n", .{name});
            return count;
        }
        advance(&ps);

        count += 1;
    }

    return count;
}

// ---------------------------------------------------------------------
//  resolve_jumps — C: zincvm.c:3089-3101
// ---------------------------------------------------------------------

/// C: zincvm.c:3089-3101 resolve_jumps — copy each JMP/JMPF/ACCESS operand
/// number into jmp_target (non-number -> 0), recursing into OP_CUR bodies.
pub fn resolveJumps(code: [*]Instr, len: i32) void {
    var i: i32 = 0;
    while (i < len) : (i += 1) {
        const in = &code[@intCast(i)];
        switch (in.op) {
            .jmp, .jmpf, .access => {
                if (in.operand.tag == .number) {
                    in.jmp_target = @intCast(in.operand.payload.number);
                } else {
                    in.jmp_target = 0;
                }
            },
            .cur => resolveJumps(@ptrCast(in.closure_code.?), in.closure_len),
            else => {},
        }
    }
}

// ---------------------------------------------------------------------
//  print_instr — C: zincvm.c:3010-3040 (debug aid)
// ---------------------------------------------------------------------

/// C: zincvm.c:3010-3040 print_instr — disassemble `code[0..len]` to `writer`.
/// `writer` is any std.io writer (testable without stdout, like printValue).
pub fn printInstr(writer: anytype, code: [*]Instr, len: i32, indent: usize) !void {
    var i: i32 = 0;
    while (i < len) : (i += 1) {
        for (0..indent) |_| try writer.writeAll("  ");
        const in = &code[@intCast(i)];
        switch (in.op) {
            .pushmark => try writer.writeAll("pushmark\n"),
            .apply => try writer.writeAll("apply\n"),
            .grab => try writer.writeAll("grab\n"),
            .ret => try writer.writeAll("return\n"),
            .let => try writer.writeAll("let\n"),
            .endlet => try writer.writeAll("endlet\n"),
            .appterm => try writer.writeAll("appterm\n"),
            .access => {
                try writer.writeAll("access ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .global => {
                try writer.writeAll("global ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .jmpf => {
                try writer.writeAll("jmpf ");
                try values.printValue(writer, in.operand);
                try writer.print(" (tgt={d})\n", .{in.jmp_target});
            },
            .jmp => {
                try writer.writeAll("jmp ");
                try values.printValue(writer, in.operand);
                try writer.print(" (tgt={d})\n", .{in.jmp_target});
            },
            .number => {
                try writer.writeAll("number ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .string => {
                try writer.writeAll("string ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .symbol => {
                try writer.writeAll("symbol ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .boolean => {
                try writer.writeAll("boolean ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .float => {
                try writer.writeAll("float ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .prim => {
                try writer.writeAll("prim ");
                try values.printValue(writer, in.operand);
                try writer.writeAll("\n");
            },
            .cur => {
                try writer.print("cur (code={d}):\n", .{in.closure_len});
                try printInstr(writer, @ptrCast(in.closure_code.?), in.closure_len, indent + 1);
                for (0..indent) |_| try writer.writeAll("  ");
                try writer.writeAll("endcur\n");
            },
            else => try writer.print("??? (op={d})\n", .{@intFromEnum(in.op)}),
        }
    }
}
