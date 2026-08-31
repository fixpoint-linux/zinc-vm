//! src/gc/types.zig — Zig port of zinctypes.h (+ gc.h's GcTypeTag + header helpers).
//!
//! C origin: zinctypes.h (shared type definitions for the ZINC VM and GC).
//! This file is the load-bearing layout contract: the collector scans typed
//! objects (Value, Instr arrays, CallFrame arrays) through these exact sizes.
//! Do NOT reorder fields — the extern-struct field ordering below reproduces the
//! C ABI size classes exactly (Value=40, Instr=64, CallFrame=48, ValueArray=16,
//! align 8), verified on Zig 0.16.0 / LP64.
//!
//! C int -> i32, long -> i64, uintptr_t -> usize.  C VAL_ERROR renamed error_
//! (error is a Zig keyword).  Enum values are explicit so ORDER and numeric
//! values match the C ABI exactly for any future C interop.

const std = @import("std");

// ---------------------------------------------------------------------
//  Value types
// ---------------------------------------------------------------------

/// C: zinctypes.h ValTag — VAL_NUMBER..VAL_STREAM, in C enum order.
pub const ValTag = enum(u32) {
    number = 0, // VAL_NUMBER
    string = 1, // VAL_STRING
    symbol = 2, // VAL_SYMBOL
    boolean = 3, // VAL_BOOLEAN
    cons = 4, // VAL_CONS
    nil = 5, // VAL_NIL
    lambda = 6, // VAL_LAMBDA
    mark = 7, // VAL_MARK
    prim = 8, // VAL_PRIM
    error_ = 9, // VAL_ERROR (renamed; error is a Zig keyword)
    vector = 10, // VAL_VECTOR
    stream = 11, // VAL_STREAM
    float = 12, // VAL_FLOAT (M4)

    pub fn char(self: ValTag) u8 {
        return switch (self) {
            .number => '0',
            .string => 'S',
            .symbol => 's',
            .boolean => 'b',
            .cons => 'c',
            .nil => 'n',
            .lambda => 'l',
            .mark => 'm',
            .prim => 'P',
            .error_ => 'e',
            .vector => 'v',
            .stream => 's',
            .float => 'F',
        };
    }
};

/// A ZINC value.  C: typedef struct Value { ValTag tag; union {...}; } Value.
/// Ported as an extern struct with an extern union payload so the ABI layout
/// matches C exactly.  @sizeOf(Value) == 40 asserted below.
pub const Value = extern struct {
    tag: ValTag,
    payload: extern union {
        number: i64, // long
        str: extern struct { data: ?[*]u8, len: i32 }, // char *data; int len;
        sym: extern struct { name: ?[*:0]const u8 }, // const char *name;
        boolean: i32, // int
        cons: extern struct { car: ?*Value, cdr: ?*Value }, // struct Value *car, *cdr;
        lambda: extern struct { // struct Instr *code; int code_len; struct Value *env; int env_len;
            code: ?*Instr,
            code_len: i32,
            env: ?[*]Value,
            env_len: i32,
        },
        prim: extern struct { name: ?[*:0]const u8 }, // const char *name;
        error_: extern struct { message: ?[*]u8 }, // char *message;
        vector: extern struct { data: ?[*]Value, len: i32 }, // struct Value *data; int len;
        stream: extern struct { // FILE *file; int is_input; int is_string;
            file: ?*anyopaque, // FILE* (NULL for string streams)
            is_input: i32,
            is_string: i32,
        },
        float: f64, // double (M4)
    },
};

// ---------------------------------------------------------------------
//  Instruction types
// ---------------------------------------------------------------------

/// C: zinctypes.h Opcode — dense enum 0..16.  OP_COUNT=17 kept as `count`
/// for parity with char_to_opcode's default return.
///
/// P3 superinstructions (Zig-native extensions, opcodes >= 18): these are
/// strict fusions of two existing ops with byte-identical semantics.  Bundles
/// are TEXT (csexp), so the C text-exchange surface is unaffected; the
/// original 0..17 values are unchanged, so numeric parity for the C set holds.
///   access_prim   'A' = access + prim  (loads an env slot then runs a prim)
///   const_prim    'K' = literal + prim (loads a constant then runs a prim)
///   prim_return   'V' = prim + return  (runs a prim then returns its result)
///   global_apply  'Q' = global + apply (looks up a global then applies it)
///   global_appterm 'R'= global + appterm (looks up a global then tail-calls)
///   (Numeric order deviates from the artifact-2 spec listing, which puts
///   global_apply=20: here prim_return=20 comes first — internal-only and
///   self-consistent, no external numeric consumer.)
pub const Opcode = enum(u32) {
    access = 0, // OP_ACCESS   'a'
    global = 1, // OP_GLOBAL   'g'
    jmpf = 2, // OP_JMPF     'f'
    jmp = 3, // OP_JMP      'j'
    appterm = 4, // OP_APPTERM  't'
    apply = 5, // OP_APPLY    'p'
    pushmark = 6, // OP_PUSHMARK 'm'
    cur = 7, // OP_CUR      'c'
    grab = 8, // OP_GRAB     'r'
    ret = 9, // OP_RETURN   'v'
    let = 10, // OP_LET      'e'
    endlet = 11, // OP_ENDLET   'd'
    number = 12, // OP_NUMBER   'n'
    string = 13, // OP_STRING   'S'
    symbol = 14, // OP_SYMBOL   's'
    boolean = 15, // OP_BOOLEAN  'b'
    prim = 16, // OP_PRIM     'P'
    float = 17, // OP_FLOAT    'F' (M4)
    access_prim = 18, // OP_ACCESS_PRIM  'A' (P3 superinstruction)
    const_prim = 19, // OP_CONST_PRIM   'K' (P3 superinstruction)
    prim_return = 20, // OP_PRIM_RETURN  'V' (P3 superinstruction)
    global_apply = 21, // OP_GLOBAL_APPLY 'Q' (P3 superinstruction)
    global_appterm = 22, // OP_GLOBAL_APPTERM 'R' (P3 superinstruction)
    count = 23, // OP_COUNT (sentinel / char_to_opcode default)
};

/// C: zinctypes.h char_to_opcode — translate a csexp opcode character to the
/// dense enum.  Unknown chars map to Opcode.count (C: OP_COUNT).
pub fn charToOpcode(c: u8) Opcode {
    return switch (c) {
        'a' => .access,
        'g' => .global,
        'f' => .jmpf,
        'j' => .jmp,
        't' => .appterm,
        'p' => .apply,
        'm' => .pushmark,
        'c' => .cur,
        'r' => .grab,
        'v' => .ret,
        'e' => .let,
        'd' => .endlet,
        'n' => .number,
        'S' => .string,
        's' => .symbol,
        'b' => .boolean,
        'P' => .prim,
        'F' => .float,
        'A' => .access_prim,
        'K' => .const_prim,
        'V' => .prim_return,
        'Q' => .global_apply,
        'R' => .global_appterm,
        else => .count,
    };
}

/// C: zinctypes.h opcode_to_char — reverse mapping for decompilers.
/// out-of-range opcodes map to '?'.
pub fn opcodeToChar(op: Opcode) u8 {
    return switch (op) {
        .access => 'a',
        .global => 'g',
        .jmpf => 'f',
        .jmp => 'j',
        .appterm => 't',
        .apply => 'p',
        .pushmark => 'm',
        .cur => 'c',
        .grab => 'r',
        .ret => 'v',
        .let => 'e',
        .endlet => 'd',
        .number => 'n',
        .string => 'S',
        .symbol => 's',
        .boolean => 'b',
        .prim => 'P',
        .float => 'F',
        .access_prim => 'A',
        .const_prim => 'K',
        .prim_return => 'V',
        .global_apply => 'Q',
        .global_appterm => 'R',
        .count => '?',
    };
}

/// C: zinctypes.h Instr.
pub const Instr = extern struct {
    op: Opcode,
    operand: Value,
    closure_code: ?*Instr, // struct Instr *
    closure_len: i32,
    jmp_target: i32,
};

// ---------------------------------------------------------------------
//  Call frame (for GC scanning)
// ---------------------------------------------------------------------

pub const CALL_STACK_DEPTH = 65536; // C: #define CALL_STACK_DEPTH 65536

/// C: typedef struct { Value *data; int len; int cap; } ValueArray;
pub const ValueArray = extern struct {
    data: ?[*]Value,
    len: i32,
    cap: i32,
};

/// C: zinctypes.h CallFrame.
pub const CallFrame = extern struct {
    code: ?*Instr, // Instr *
    code_len: i32,
    pc: i32,
    env: ?[*]Value, // Value *
    env_len: i32,
    env_cap: i32,
    stack: ValueArray,
};

/// C: zincvm.h TableEntry { char *name; Value value; }.
pub const TableEntry = extern struct {
    name: ?[*:0]u8, // char *name
    value: Value,
};

// ---------------------------------------------------------------------
//  GC type tags + header helpers
// ---------------------------------------------------------------------

/// C: gc.h enum { GC_TYPE_RAW=0 ... GC_TYPE_CALLFRAME_ARRAY=4 }.
/// Stored in the header's repurposed high bits; tells the scavenger how to scan
/// an object's body.
pub const GcTypeTag = enum(u32) {
    raw = 0, // char[] string/error data — no scan
    value = 1, // single Value struct — scan by tag
    value_array = 2, // Value[] — scan each by tag
    instr_array = 3, // Instr[] — scan operand + closure_code
    callframe_array = 4, // CallFrame[] — scan env + stack.data
};

/// C: gc.h PAGEBYTES — GC heap page size in bytes (matches gc.c's private
/// PAGEBYTES).  Exposed so callers can page-align without duplicating 512.
pub const GC_PAGEBYTES = 512;

/// Words per heap page (C: PAGEWORDS).
pub const GC_PAGEWORDS = GC_PAGEBYTES / @sizeOf(usize);

/// C: gc.c MAKE_HEADER — (ty << 25) | (words << 1) | 1.
pub inline fn makeHeader(words: usize, ty: GcTypeTag) usize {
    return (@as(usize, @intFromEnum(ty)) << 25) | (words << 1) | 1;
}

/// C: gc.c FORWARDED — forwarding pointer iff header bit0 == 0.
pub inline fn forwarded(hdr: usize) bool {
    return (hdr & 1) == 0;
}

/// C: gc.c HEADER_TYPE — (hdr >> 25) & 0xFFFFF.
pub inline fn headerType(hdr: usize) u32 {
    return @intCast((hdr >> 25) & 0xFFFFF);
}

/// C: gc.c HEADER_WORDS — (hdr >> 1) & 0xFFFFFF.
pub inline fn headerWords(hdr: usize) usize {
    return (hdr >> 1) & 0xFFFFFF;
}

/// C: gc.c gcalloc_internal — words > 0xFFFFFF is a fatal allocation error.
/// Mirrors the C behavior (stderr message + exit(1)) via std.debug.panic.
pub fn assertWordsFits(words: usize, bytes: usize) void {
    if (words > 0xFFFFFF) {
        std.debug.panic("gcalloc: object too large ({d} bytes)\n", .{bytes});
    }
}

// ---------------------------------------------------------------------
//  load-bearing size classes (Phase 3/4 BiBOP) — MUST stay true.
//  C: zinctypes.h _Static_asserts.  Verified on Zig 0.16.0 / LP64.
// ---------------------------------------------------------------------
comptime {
    std.debug.assert(@sizeOf(Value) == 40);
    std.debug.assert(@sizeOf(Instr) == 64);
    std.debug.assert(@sizeOf(CallFrame) == 48);
    std.debug.assert(@sizeOf(ValueArray) == 16);
    std.debug.assert(@sizeOf(usize) == 8); // LP64
    std.debug.assert(@alignOf(Value) == 8);
    std.debug.assert(@alignOf(Instr) == 8);
    std.debug.assert(@alignOf(CallFrame) == 8);
    std.debug.assert(@alignOf(ValueArray) == 8);
    // zincvm.h TableEntry asserts: word-multiple + word-aligned for GC scan.
    std.debug.assert(@sizeOf(TableEntry) % @sizeOf(usize) == 0);
    std.debug.assert(@alignOf(TableEntry) >= @sizeOf(usize));
}
