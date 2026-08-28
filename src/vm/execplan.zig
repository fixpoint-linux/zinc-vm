//! src/vm/execplan.zig — the native process runner (exec-plan) + the process
//! prims (wait/kill/cd/getcwd/getpid/getenv/setenv/glob) for milestone M3.
//!
//! C origin: zincvm.c:1037-1780 (the exec-plan plan runner) + the prim cases:
//!   - tagged builders make_tagged_nil/string/number/cons  :993-1034
//!   - plan structs + limits                              :1071-1101
//!   - tagged-list probes tdl_* / td_*                    :1112-1217
//!   - decoders decode_* + plan_decode                    :1254-1412
//!   - child helpers + run_pipeline/run_program + slurp   :1414-1769
//!   - exec-plan :2100-2151   cd :1955-1962   getcwd :2191-2199
//!     getpid :2205-2209      getenv :2211-2219   glob :2224-2265
//!     kill :2294-2299        setenv :2580-2595    wait :2693-2700
//!
//! INVARIANTS (plan M3, preserved exactly):
//!   1. DECODE performs ZERO GC allocation — every structure and string lives
//!      on the page_allocator (C malloc/realloc/calloc parity), so the tagged
//!      plan Value's interior pointers cannot go stale mid-decode.  The plan
//!      Value is additionally rooted across decode+run (C parity).
//!   2. THE CHILD AFTER FORK touches ONLY the page_allocator decode structs,
//!      libc calls, and _exit — never the GC heap, never a Zig error-unwind
//!      path (all child-side functions are plain i32-returning, no `try`/
//!      `catch`/`errdefer`), never std buffered I/O (write(2) only).
//!   3. argv/path strings are dupeZ'd onto the page_allocator BEFORE any fork
//!      (they are decoded pre-fork by construction — decode precedes run).
//!
//! SYSCALL LAYER (artifact-1, verified): Zig 0.16 std.posix has NO fork/pipe/
//! exit/waitpid/dup/dup2/mkstemp, so the ENTIRE process layer uses libc
//! externs and the exe links libc (build.zig link_lib_c on the vm module).
//! std.posix.W DOES exist (= std.c.W = linux.W on Linux; IFEXITED/EXITSTATUS
//! match glibc's macros) — TERMSIG returns a SIG enum, so the 128+sig arm
//! derives the signal from the raw status bits (identical to glibc WTERMSIG).
//!
//! TMPFILES: mkstemp is extern'd directly (exact C parity: the probe's
//! hand-rolled alternative exists only because the probe never linked
//! mkstemp; the extern is strictly more faithful).  Every tmpfile is
//! unlinked immediately and shared by one open fd between parent and child.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const state = @import("state.zig");
const values = @import("values.zig");
const symbols = @import("symbols.zig");
const interp = @import("interp.zig");

const Gc = gc.Gc;
const Value = types.Value;
const ValueArray = types.ValueArray;
const Vm = state.Vm;
const VmError = state.VmError;


// =====================================================================
//  libc externs (the whole process/syscall layer — artifact-1)
// =====================================================================

extern "c" fn fork() c_int;
extern "c" fn execvp(file: [*:0]const u8, argv: [*:null]const ?[*:0]const u8) c_int;
extern "c" fn waitpid(pid: c_int, status: ?*c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn dup(oldfd: c_int) c_int;
extern "c" fn dup2(oldfd: c_int, newfd: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn mkstemp(template: [*:0]u8) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn chdir(path: [*:0]const u8) c_int;
extern "c" fn getcwd(buf: [*]u8, size: usize) ?[*:0]u8;
extern "c" fn getpid() c_int;
extern "c" fn kill(pid: c_int, sig: c_int) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn opendir(name: [*:0]const u8) ?*anyopaque;
extern "c" fn readdir(d: *anyopaque) ?*Dirent;
extern "c" fn closedir(d: *anyopaque) c_int;
extern "c" fn fnmatch(pattern: [*:0]const u8, string: [*:0]const u8, flags: c_int) c_int;

/// C: struct dirent (Linux x86-64 layout; d_name is NUL-terminated in
/// practice — readdir guarantees it even though the array is fixed-size).
const Dirent = extern struct {
    d_ino: u64,
    d_off: i64,
    d_reclen: c_ushort,
    d_type: u8,
    d_name: [256]u8,
};

// Flag constants (Linux x86-64 values — same as the artifact-1 probe).
const O_RDONLY: c_int = 0;
const O_WRONLY: c_int = 1;
const O_RDWR: c_int = 2;
const O_CREAT: c_int = 64;
const O_EXCL: c_int = 128;
const O_TRUNC: c_int = 512;
const O_APPEND: c_int = 1024;
const SEEK_SET: c_int = 0;
const SIGKILL: c_int = 9;

pub const PATH_MAX = 4096;

/// Decode failures all collapse to one throw, exactly like C's single
/// `return -1` (malloc failure included).
const DecodeError = error{ Malformed, OutOfMemory };

/// The page_allocator is the decode runner's only allocator (INVARIANT 1).
const ca = std.heap.page_allocator;

// =====================================================================
//  Plan structs + limits — C: zincvm.c:1071-1101
// =====================================================================

/// C: ROP_SEQ/ROP_AND/ROP_OR.  `and`/`or` are Zig keywords -> and_/or_.
/// (pub for the M9 async exec: the host classifies a decoded plan.)
pub const ChainOp = enum { seq, and_, or_ };

/// C: RR_IN/RR_OUT/RR_APPEND/RR_DUP/RR_HDOC/RR_HSTR.
/// (pub for the M9 async exec: single-command classification reads redirs.)
pub const RedirKind = enum { in, out, append, dup, hdoc, hstr };

/// C: RRedir.  path/body own page_allocator [:0]u8 copies (decode-time).
pub const RRedir = struct {
    kind: RedirKind = .in,
    fd: i32 = 0, // 0 / 1 / 2
    path: ?[:0]u8 = null, // in/out/append target path (owned)
    dup_fd: i32 = 0, // RR_DUP: target fd (1 or 2)
    body: ?[:0]u8 = null, // RR_HDOC/RR_HSTR body (owned; hstr carries trailing \n)
    tmpfd: i32 = -1, // runtime: pre-opened body tmpfile; -1 until wired
};

/// C: RCmd.  argv is a page_allocator sentinel(null)-terminated pointer
/// array (C's char** NULL-terminated); argv.len == argc.
pub const RCmd = struct {
    argv: [:null]const ?[*:0]const u8 = &[_:null]?[*:0]const u8{}, // empty (subshell) default
    redirs: []RRedir = &[_]RRedir{},
    sub: ?*RProg = null, // null = plain command; else nested program (argv.len == 0)
};

pub const RPipe = struct { cmds: []RCmd = &[_]RCmd{} };
pub const RChain = struct { op: ChainOp = .seq, pipe: RPipe = .{} };
pub const RProg = struct { chains: []RChain = &[_]RChain{} };

const PLAN_MAX_CHAINS = 256;
const PLAN_MAX_CMDS = 64;
const PLAN_MAX_ARGS = 256;
const PLAN_MAX_REDIRS = 64;
const PLAN_MAX_DEPTH = 32;

// =====================================================================
//  Tagged-value builders — C: zincvm.c:993-1034
//  ([cons] / [string S] / [number N] / [cons H T] with the 3-ELEMENT list
//  shape the metacircular interp and demarshal_from_tagged expect.)
// =====================================================================

/// C: zincvm.c:993-995 make_tagged_nil — [cons] = empty tagged list.
pub fn makeTaggedNil(vm: *Vm) Value {
    return values.valCons(vm.gc, symbols.valSymbol(&vm.symbols, "cons"), values.valNil());
}

/// C: zincvm.c:998-1006 make_tagged_string — [string S]; roots s then inner
/// across the valCons allocs.  `data` must NOT point into the GC heap (the
/// valString CONTRACT — all callers pass C-heap/stack buffers).
pub fn makeTaggedString(vm: *Vm, data: []const u8) Value {
    const g = vm.gc;
    var s = values.valString(g, data);
    g.rootPushValue(&s);
    var inner = values.valCons(g, s, values.valNil());
    g.rootPushValue(&inner);
    const result = values.valCons(g, symbols.valSymbol(&vm.symbols, "string"), inner);
    g.rootPop();
    g.rootPop();
    return result;
}

/// C: zincvm.c:1009-1015 make_tagged_number — [number N].
pub fn makeTaggedNumber(vm: *Vm, n: i64) Value {
    const g = vm.gc;
    var inner = values.valCons(g, values.valNumber(n), values.valNil());
    g.rootPushValue(&inner);
    const result = values.valCons(g, symbols.valSymbol(&vm.symbols, "number"), inner);
    g.rootPop();
    return result;
}

/// C: zincvm.c:1022-1034 make_tagged_cons — [cons H T] as the 3-element
/// list (cons . (Car . (Cdr . nil))), NOT a dotted pair.  The caller's
/// originals become stale (may move) — do not use them after this call.
pub fn makeTaggedCons(vm: *Vm, car: Value, cdr: Value) Value {
    const g = vm.gc;
    var carv = car;
    var cdrv = cdr;
    g.rootPushValue(&carv);
    g.rootPushValue(&cdrv);
    var inner = values.valCons(g, carv, values.valCons(g, cdrv, values.valNil()));
    g.rootPushValue(&inner);
    const result = values.valCons(g, symbols.valSymbol(&vm.symbols, "cons"), inner);
    g.rootPop();
    g.rootPop();
    g.rootPop();
    return result;
}

// =====================================================================
//  Tagged-list probes — C: zincvm.c:1112-1217
//  (NO GC allocation: they only read the plan Value's cells; decode runs
//  between the root push and pop so the interior pointers stay valid.)
// =====================================================================

/// C: zincvm.c:1112-1121 tdl_is_empty: 1 empty / 0 node / -1 malformed.
const TdlState = enum { empty, node, malformed };

fn tdlIsEmpty(v: Value) TdlState {
    if (v.tag != .cons) return .malformed;
    const car = v.payload.cons.car orelse return .malformed;
    if (car.tag != .symbol) return .malformed;
    if (!std.mem.eql(u8, values.symSlice(car.*), "cons")) return .malformed;
    const cdr = v.payload.cons.cdr.?.*;
    if (cdr.tag == .nil) return .empty;
    if (cdr.tag == .cons) return .node;
    return .malformed;
}

/// C: zincvm.c:1123-1131 tdl_pop: head + rest through the singleton wrapper.
fn tdlPop(v: Value, head: *Value, rest: *Value) bool {
    if (tdlIsEmpty(v) != .node) return false;
    const cdr = v.payload.cons.cdr.?.*; // (H . (T . nil))
    const t1 = cdr.payload.cons.cdr.?.*; // cons(T, nil) singleton wrapper
    if (t1.tag != .cons) return false;
    head.* = cdr.payload.cons.car.?.*;
    rest.* = t1.payload.cons.car.?.*;
    return true;
}

/// C: zincvm.c:1135-1155 tdl_collect — tagged list -> page_allocator Value
/// array (by-value copies; safe because decode never GC-allocates).
fn tdlCollect(v_in: Value, out: *[]Value, limit: usize) DecodeError!void {
    var cap: usize = 8;
    var n: usize = 0;
    var arr = ca.alloc(Value, cap) catch return error.OutOfMemory;
    var v = v_in;
    while (true) {
        switch (tdlIsEmpty(v)) {
            .malformed => {
                ca.free(arr);
                return error.Malformed;
            },
            .empty => break,
            .node => {},
        }
        if (n >= limit) {
            ca.free(arr);
            return error.Malformed;
        }
        if (n == cap) {
            cap *= 2;
            arr = ca.realloc(arr, cap) catch {
                ca.free(arr);
                return error.OutOfMemory;
            };
        }
        if (!tdlPop(v, &arr[n], &v)) {
            ca.free(arr);
            return error.Malformed;
        }
        n += 1;
    }
    out.* = arr[0..n];
}

/// C: zincvm.c:1157-1167 tdl_len.
fn tdlLen(v_in: Value, len: *usize) DecodeError!void {
    var n: usize = 0;
    var v = v_in;
    var h: Value = undefined;
    while (true) {
        switch (tdlIsEmpty(v)) {
            .malformed => return error.Malformed,
            .empty => {
                len.* = n;
                return;
            },
            .node => {},
        }
        if (!tdlPop(v, &h, &v)) return error.Malformed;
        n += 1;
    }
}

/// C: zincvm.c:1169-1177 tdl_elem (index or malformed -> error).
fn tdlElem(v_in: Value, idx: usize) DecodeError!Value {
    var v = v_in;
    var h: Value = undefined;
    var i: usize = 0;
    while (true) {
        if (tdlIsEmpty(v) != .node) return error.Malformed; // malformed or out of range
        if (i == idx) {
            if (!tdlPop(v, &h, &v)) return error.Malformed;
            return h;
        }
        if (!tdlPop(v, &h, &v)) return error.Malformed;
        i += 1;
    }
}

/// C: zincvm.c:1180-1191 td_string — [string S] -> page_allocator [:0]u8.
fn tdString(v: Value) DecodeError![:0]u8 {
    if (v.tag != .cons) return error.Malformed;
    const tag = v.payload.cons.car orelse return error.Malformed;
    if (tag.tag != .symbol or !std.mem.eql(u8, values.symSlice(tag.*), "string"))
        return error.Malformed;
    const cdr = v.payload.cons.cdr.?.*;
    if (cdr.tag != .cons) return error.Malformed;
    const s = cdr.payload.cons.car.?.*;
    if (s.tag != .string) return error.Malformed;
    return dupeZ(values.strSlice(s));
}

/// C: zincvm.c:1193-1204 td_number — [number N].
fn tdNumber(v: Value) DecodeError!i64 {
    if (v.tag != .cons) return error.Malformed;
    const tag = v.payload.cons.car orelse return error.Malformed;
    if (tag.tag != .symbol or !std.mem.eql(u8, values.symSlice(tag.*), "number"))
        return error.Malformed;
    const cdr = v.payload.cons.cdr.?.*;
    if (cdr.tag != .cons) return error.Malformed;
    const s = cdr.payload.cons.car.?.*;
    if (s.tag != .number) return error.Malformed;
    return s.payload.number;
}

/// C: zincvm.c:1206-1217 td_symbol — [symbol X] -> a borrowed name slice
/// (points into the GC symbol storage; used only for eql during decode,
/// which performs no GC allocation, so it cannot go stale).
fn tdSymbol(v: Value) DecodeError![]const u8 {
    if (v.tag != .cons) return error.Malformed;
    const tag = v.payload.cons.car orelse return error.Malformed;
    if (tag.tag != .symbol or !std.mem.eql(u8, values.symSlice(tag.*), "symbol"))
        return error.Malformed;
    const cdr = v.payload.cons.cdr.?.*;
    if (cdr.tag != .cons) return error.Malformed;
    const s = cdr.payload.cons.car.?.*;
    if (s.tag != .symbol) return error.Malformed;
    return values.symSlice(s);
}

/// NUL-terminated copy onto the page_allocator (C strndup parity).
fn dupeZ(s: []const u8) DecodeError![:0]u8 {
    const buf = ca.alloc(u8, s.len + 1) catch return error.OutOfMemory;
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

/// page_allocator free of a stored [*:0]const u8 (length via sliceTo;
/// Allocator.free adds the sentinel byte back for sentinel slices).
fn freeZ(p: [*:0]const u8) void {
    ca.free(std.mem.sliceTo(p, 0));
}

// =====================================================================
//  Freeing — C: zincvm.c:1219-1248 cmd_free/pipe_free/plan_free
//  (planFree is pub for the M9 async exec: the host frees the decoded plan
//  it holds across the fork+waitpid lifetime.)
// =====================================================================

fn cmdFree(c: *RCmd) void {
    for (c.argv) |a| {
        if (a) |p| freeZ(p);
    }
    if (c.argv.len != 0) ca.free(c.argv);
    for (c.redirs) |*r| {
        if (r.path) |p| ca.free(p);
        if (r.body) |b| ca.free(b);
    }
    if (c.redirs.len != 0) ca.free(c.redirs);
    if (c.sub) |sp| {
        planFree(sp);
        ca.destroy(sp);
    }
    c.* = .{};
}

fn pipeFree(pp: *RPipe) void {
    for (pp.cmds) |*c| cmdFree(c);
    if (pp.cmds.len != 0) ca.free(pp.cmds);
    pp.* = .{};
}

/// C: zincvm.c:1239-1248 plan_free.  (pub for the M9 async exec.)
pub fn planFree(p: *RProg) void {
    for (p.chains) |*ch| pipeFree(&ch.pipe);
    if (p.chains.len != 0) ca.free(p.chains);
    p.* = .{};
}

// =====================================================================
//  Decoders — C: zincvm.c:1254-1412
// =====================================================================

/// C: zincvm.c:1254-1307 decode_redir: [op fd target].  fd/shape validation
/// mirrors the reference parser: in/hdoc/hstr must target fd 0;
/// out/append/dup fd 1|2; dup target fd must be 1|2.
fn decodeRedir(v: Value, out: *RRedir) DecodeError!void {
    out.* = .{};
    var len: usize = 0;
    try tdlLen(v, &len);
    if (len != 3) return error.Malformed;
    const opv = try tdlElem(v, 0);
    const fdv = try tdlElem(v, 1);
    const tgv = try tdlElem(v, 2);
    const op = try tdSymbol(opv);
    const fd = try tdNumber(fdv);
    out.kind = if (std.mem.eql(u8, op, "in"))
        .in
    else if (std.mem.eql(u8, op, "out"))
        .out
    else if (std.mem.eql(u8, op, "append"))
        .append
    else if (std.mem.eql(u8, op, "dup"))
        .dup
    else if (std.mem.eql(u8, op, "hdoc"))
        .hdoc
    else if (std.mem.eql(u8, op, "hstr"))
        .hstr
    else
        return error.Malformed;
    switch (out.kind) {
        .in, .hdoc, .hstr => {
            if (fd != 0) return error.Malformed;
            out.fd = 0;
        },
        .out, .append, .dup => {
            if (fd != 1 and fd != 2) return error.Malformed;
            out.fd = @intCast(fd);
        },
    }
    if (out.kind == .dup) {
        const dfd = try tdNumber(tgv);
        if (dfd != 1 and dfd != 2) return error.Malformed;
        out.dup_fd = @intCast(dfd);
        return;
    }
    const s = try tdString(tgv);
    if (out.kind == .hdoc) {
        out.body = s;
    } else if (out.kind == .hstr) {
        // C:1295-1303 — append the trailing \n hstr semantics require.
        const b = ca.alloc(u8, s.len + 2) catch {
            ca.free(s);
            return error.OutOfMemory;
        };
        @memcpy(b[0..s.len], s);
        b[s.len] = '\n';
        b[s.len + 1] = 0;
        ca.free(s);
        out.body = b[0 .. s.len + 1 :0];
    } else {
        out.path = s;
    }
}

/// C: zincvm.c:1309-1357 decode_cmd: [Argv Redirs Sub].
fn decodeCmd(v: Value, out: *RCmd, depth: u32) DecodeError!void {
    out.* = .{};
    var len: usize = 0;
    try tdlLen(v, &len);
    if (len != 3) return error.Malformed;
    const argvv = try tdlElem(v, 0);
    const redirsv = try tdlElem(v, 1);
    const subv = try tdlElem(v, 2);

    // argv: (list string) -> page_allocator sentinel-terminated pointer array.
    var els: []Value = undefined;
    tdlCollect(argvv, &els, PLAN_MAX_ARGS) catch {
        return error.Malformed;
    };
    const raw = ca.alloc(?[*:0]const u8, els.len + 1) catch {
        ca.free(els);
        return error.OutOfMemory;
    };
    @memset(raw, null); // failure cleanup reads raw[0..els.len]; nulls are skipped
    var ok = true;
    for (els, 0..) |e, i| {
        if (tdString(e)) |s| {
            raw[i] = s.ptr;
        } else |_| {
            raw[i] = null;
            ok = false;
            break;
        }
    }
    ca.free(els);
    if (!ok) {
        for (raw[0 .. raw.len - 1]) |a| {
            if (a) |p| freeZ(p);
        }
        ca.free(raw);
        return error.Malformed;
    }
    raw[els.len] = null;
    out.argv = raw[0..els.len :null];

    // redirs: (list Redir), source order.
    tdlCollect(redirsv, &els, PLAN_MAX_REDIRS) catch {
        cmdFree(out); // frees argv + anything decoded so far
        return error.Malformed;
    };
    // Empty -> the const empty slice (cmdFree skips zero-length frees; no
    // allocation, no leak).  Non-empty: @memset RRedir{} (tmpfd = -1).
    if (els.len == 0) {
        out.redirs = &[_]RRedir{};
    } else {
        const rarr = ca.alloc(RRedir, els.len) catch {
            ca.free(els);
            cmdFree(out);
            return error.OutOfMemory;
        };
        @memset(rarr, RRedir{});
        out.redirs = rarr;
    }
    var nredir: usize = 0;
    var redir_ok = true;
    for (els, 0..) |e, i| {
        decodeRedir(e, &out.redirs[i]) catch {
            redir_ok = false;
            break;
        };
        nredir = i + 1;
    }
    ca.free(els);
    if (!redir_ok) {
        out.redirs.len = nredir; // free exactly the decoded prefix (C: out->nredir = i)
        cmdFree(out);
        return error.Malformed;
    }

    // sub: () plain | nested Program (requires empty argv).
    switch (tdlIsEmpty(subv)) {
        .malformed => {
            cmdFree(out);
            return error.Malformed;
        },
        .empty => {
            if (out.argv.len == 0) { // C:1343 empty command
                cmdFree(out);
                return error.Malformed;
            }
        },
        .node => {
            if (out.argv.len != 0) { // C:1345 subshell cmd carries no argv
                cmdFree(out);
                return error.Malformed;
            }
            if (depth >= PLAN_MAX_DEPTH) {
                cmdFree(out);
                return error.Malformed;
            }
            const sp = ca.create(RProg) catch {
                cmdFree(out);
                return error.OutOfMemory;
            };
            out.sub = sp;
            decodeProgram(subv, sp, depth + 1) catch {
                cmdFree(out); // cmdFree frees out.sub via planFree + destroy
                return error.Malformed;
            };
        },
    }

    // stdin exclusivity (reference parser: 'Multiple stdin redirects').
    var nstdin: usize = 0;
    for (out.redirs) |*r| {
        if (r.fd == 0) nstdin += 1;
    }
    if (nstdin > 1) {
        cmdFree(out);
        return error.Malformed;
    }
}

/// C: zincvm.c:1359-1374 decode_pipeline: (list Cmd), >= 1 command.
fn decodePipeline(v: Value, out: *RPipe, depth: u32) DecodeError!void {
    out.* = .{};
    var els: []Value = undefined;
    tdlCollect(v, &els, PLAN_MAX_CMDS) catch return error.Malformed;
    if (els.len == 0) { // a pipeline needs >= 1 command
        ca.free(els);
        return error.Malformed;
    }
    const cmds = ca.alloc(RCmd, els.len) catch {
        ca.free(els);
        return error.OutOfMemory;
    };
    @memset(cmds, RCmd{});
    out.cmds = cmds;
    var n: usize = 0;
    var ok = true;
    for (els, 0..) |e, i| {
        decodeCmd(e, &out.cmds[i], depth) catch {
            ok = false;
            break;
        };
        n = i + 1;
    }
    ca.free(els);
    if (!ok) {
        out.cmds = cmds[0..n];
        pipeFree(out);
        return error.Malformed;
    }
}

/// C: zincvm.c:1376-1390 decode_chain: [op Pipeline], op in {seq,and,or}.
fn decodeChain(v: Value, out: *RChain, depth: u32) DecodeError!void {
    out.* = .{};
    var len: usize = 0;
    try tdlLen(v, &len);
    if (len != 2) return error.Malformed;
    const opv = try tdlElem(v, 0);
    const pipev = try tdlElem(v, 1);
    const op = try tdSymbol(opv);
    out.op = if (std.mem.eql(u8, op, "seq"))
        .seq
    else if (std.mem.eql(u8, op, "and"))
        .and_
    else if (std.mem.eql(u8, op, "or"))
        .or_
    else
        return error.Malformed;
    try decodePipeline(pipev, &out.pipe, depth);
}

/// C: zincvm.c:1392-1406 decode_program: (list Chain).  An EMPTY program is
/// legal (C calloc(1) + zero iterations): runs nothing, exit 0.
fn decodeProgram(v: Value, out: *RProg, depth: u32) DecodeError!void {
    out.* = .{};
    var els: []Value = undefined;
    tdlCollect(v, &els, PLAN_MAX_CHAINS) catch return error.Malformed;
    if (els.len == 0) {
        ca.free(els);
        out.chains = &[_]RChain{};
        return;
    }
    const chains = ca.alloc(RChain, els.len) catch {
        ca.free(els);
        return error.OutOfMemory;
    };
    @memset(chains, RChain{});
    out.chains = chains;
    var n: usize = 0;
    var ok = true;
    for (els, 0..) |e, i| {
        decodeChain(e, &out.chains[i], depth) catch {
            ok = false;
            break;
        };
        n = i + 1;
    }
    ca.free(els);
    if (!ok) {
        out.chains.len = n; // free exactly the decoded prefix (C: out->n = i)
        planFree(out);
        return error.Malformed;
    }
}

/// C: zincvm.c:1410-1412 plan_decode: tagged plan tree -> C structs.
/// Malformed input from our own parser is an always-on vm_throw.
pub fn planDecode(v: Value, out: *RProg) bool {
    decodeProgram(v, out, 0) catch return false;
    return true;
}

// =====================================================================
//  Child-side helpers — C: zincvm.c:1417-1602
//  (write(2) only — NO stdio, NO GC; execvp's internal malloc is fine:
//  single-threaded child, C data only)
// =====================================================================

/// C: zincvm.c:1417-1421 child_w2.
fn childW2(a: ?[]const u8, b: ?[]const u8, c: ?[]const u8) void {
    if (a) |s| _ = write(2, s.ptr, s.len);
    if (b) |s| _ = write(2, s.ptr, s.len);
    if (c) |s| _ = write(2, s.ptr, s.len);
}

/// C: zincvm.c:1426-1442 open_body_tmpfile — write body to an unlinked
/// tmpfile, rewind.  Parent-side, pre-fork (re-entrant from a subshell
/// child, which may safely open files).  -1 on failure.
fn openBodyTmpfile(body: [:0]const u8) i32 {
    var tmpl = "/tmp/shensh-hdoc.XXXXXX".*;
    const fd = mkstemp(&tmpl);
    if (fd < 0) return -1;
    _ = unlink(&tmpl);
    var off: usize = 0;
    while (off < body.len) {
        const w = write(fd, body.ptr + off, body.len - off);
        if (w < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            _ = close(fd);
            return -1;
        }
        off += @intCast(w);
    }
    if (lseek(fd, 0, SEEK_SET) < 0) {
        _ = close(fd);
        return -1;
    }
    return fd;
}

/// C: zincvm.c:1448-1489 apply_redirects — sequential left-to-right dup2
/// application gives POSIX semantics directly (2>&1 before >file snapshots
/// the old stdout into fd 2; no lookahead needed).  Open failure:
/// write(2) + return -1 (caller _exit(1)).  (pub for the M9 async exec's
/// fork path — although the async fast path only runs REDIRECT-FREE single
/// commands, the helper is part of the shared child-side surface.)
pub fn applyRedirects(c: *RCmd) i32 {
    for (c.redirs) |*r| {
        switch (r.kind) {
            .in => {
                const fd = open(r.path.?.ptr, O_RDONLY);
                if (fd < 0) {
                    childW2("Input redirect file not found: ", r.path, "\n");
                    return -1;
                }
                if (dup2(fd, r.fd) < 0) {
                    _ = close(fd);
                    return -1;
                }
                _ = close(fd);
            },
            .out, .append => {
                const flags: c_int = O_WRONLY | O_CREAT |
                    (if (r.kind == .append) O_APPEND else O_TRUNC);
                const fd = open(r.path.?.ptr, flags, @as(c_uint, 0o666));
                if (fd < 0) {
                    childW2("Output redirect failed: ", r.path, "\n");
                    return -1;
                }
                if (dup2(fd, r.fd) < 0) {
                    _ = close(fd);
                    return -1;
                }
                _ = close(fd);
            },
            .dup => {
                if (dup2(r.dup_fd, r.fd) < 0) return -1;
            },
            .hdoc, .hstr => {
                if (dup2(r.tmpfd, r.fd) < 0) return -1;
                _ = close(r.tmpfd);
                r.tmpfd = -1; // consumed (an in-process caller skips it on close)
            },
        }
    }
    return 0;
}

/// C: zincvm.c:1494-1532 child_builtin — builtins runnable INSIDE pipeline/
/// subshell children.  Writes to fd 1/2 (already redirected).  Returns exit
/// code, or -1 when argv[0] is not a builtin.  (pub for the M9 async exec:
/// a single-command plan that names a builtin runs it in the forked child.)
pub fn childBuiltin(argc: usize, argv: [:null]const ?[*:0]const u8) i32 {
    if (argc == 0 or argv[0] == null) return -1;
    const a0 = std.mem.sliceTo(argv[0].?, 0);
    if (std.mem.eql(u8, a0, "echo")) {
        var i: usize = 1;
        var nl = true;
        if (i < argc and std.mem.eql(u8, std.mem.sliceTo(argv[i].?, 0), "-n")) {
            nl = false;
            i += 1;
        }
        const first = i;
        while (i < argc) : (i += 1) {
            if (i > first) _ = write(1, " ", 1);
            const arg = std.mem.sliceTo(argv[i].?, 0);
            _ = write(1, arg.ptr, arg.len);
        }
        if (nl) _ = write(1, "\n", 1);
        return 0;
    }
    if (std.mem.eql(u8, a0, "true")) return 0;
    if (std.mem.eql(u8, a0, "false")) return 1;
    if (std.mem.eql(u8, a0, ":")) return 0;
    if (std.mem.eql(u8, a0, "cd")) {
        var dir: ?[]const u8 = if (argc > 1) std.mem.sliceTo(argv[1].?, 0) else null;
        if (dir == null or dir.?.len == 0) dir = envHome();
        if (dir == null or dir.?.len == 0) dir = "/";
        var buf: [PATH_MAX:0]u8 = undefined;
        if (dir.?.len >= PATH_MAX) return 1;
        @memcpy(buf[0..dir.?.len], dir.?);
        buf[dir.?.len] = 0;
        if (chdir(&buf) != 0) {
            childW2("cd: ", buf[0..dir.?.len], "\n");
            return 1;
        }
        return 0;
    }
    if (std.mem.eql(u8, a0, "pwd")) {
        var buf: [PATH_MAX]u8 = undefined;
        if (getcwd(&buf, buf.len)) |p| {
            const l = std.mem.sliceTo(p, 0).len;
            _ = write(1, &buf, l);
            _ = write(1, "\n", 1);
            return 0;
        }
        _ = write(2, "pwd: cannot get cwd\n", 20);
        return 1;
    }
    return -1;
}

/// getenv("HOME") as a slice (null if unset / empty in the env).
fn envHome() ?[]const u8 {
    const h = getenv("HOME") orelse return null;
    const s = std.mem.sliceTo(h, 0);
    if (s.len == 0) return null;
    return s;
}

/// C: zincvm.c:1536-1541 is_child_builtin (exact match — no path parts).
fn isChildBuiltin(a0: []const u8) bool {
    const names = [_][]const u8{ "echo", "true", "false", ":", "cd", "pwd" };
    for (names) |n| {
        if (std.mem.eql(u8, a0, n)) return true;
    }
    return false;
}

/// C: zincvm.c:1551-1572 run_builtin_inprocess — execute a builtin IN THE
/// CURRENT process with fds 0/1/2 saved and restored around it (POSIX
/// simple-command semantics for builtins inside a subshell: (cd /; pwd)
/// must let the cd affect the subshell process).  Only ever called with
/// in_child==1; the parent exec-plan path forks instead.  (pub for the M9
/// async exec's shared child-side surface.)
pub fn runBuiltinInprocess(c: *RCmd, outfd: i32, errfd: i32) i32 {
    const sav0 = dup(0);
    const sav1 = dup(1);
    const sav2 = dup(2);
    if (sav0 < 0 or sav1 < 0 or sav2 < 0) {
        if (sav0 >= 0) _ = close(sav0);
        if (sav1 >= 0) _ = close(sav1);
        if (sav2 >= 0) _ = close(sav2);
        return -1;
    }
    var rc: i32 = undefined;
    _ = dup2(outfd, 1);
    _ = dup2(errfd, 2);
    if (applyRedirects(c) < 0) {
        rc = 1;
    } else {
        rc = childBuiltin(c.argv.len, c.argv);
        if (rc < 0) rc = 127; // caller pre-checked; defensive
    }
    _ = dup2(sav0, 0);
    _ = close(sav0);
    _ = dup2(sav1, 1);
    _ = close(sav1);
    _ = dup2(sav2, 2);
    _ = close(sav2);
    return rc;
}

/// C: zincvm.c:1574-1583 close_cmd_tmpfds / close_pipe_tmpfds.
fn closeCmdTmpfds(c: *RCmd) void {
    for (c.redirs) |*r| {
        if (r.tmpfd >= 0) {
            _ = close(r.tmpfd);
            r.tmpfd = -1;
        }
    }
}

fn closePipeTmpfds(pp: *RPipe) void {
    for (pp.cmds) |*c| closeCmdTmpfds(c);
}

/// C: zincvm.c:1588-1598 wire_pipe_tmpfds — pre-open heredoc/herestring
/// bodies parent-side.  0 ok / -1 tmpfile failure (already-opened fds stay
/// owned by the RRedir fields; the caller closes via closePipeTmpfds).
/// NOTE: C declares the RPipe const and mutates r->tmpfd anyway; the port
/// drops the lying const.
fn wirePipeTmpfds(pp: *RPipe) i32 {
    for (pp.cmds) |*cmd| {
        for (cmd.redirs) |*r| {
            if (r.kind == .hdoc or r.kind == .hstr) {
                r.tmpfd = openBodyTmpfile(r.body.?);
                if (r.tmpfd < 0) return -1;
            }
        }
    }
    return 0;
}

/// C: zincvm.c:1600-1602 wait_status_code.  std.posix.W EXISTS in Zig 0.16
/// (= std.c.W = linux.W); IFEXITED/EXITSTATUS match glibc.  TERMSIG returns
/// a SIG enum, so the signal arm re-derives from the raw status bits —
/// identical to glibc's WTERMSIG(s) = s & 0x7f (and C's 128+sig).
/// (pub for the M9 async exec: the host computes the exit code from waitpid.)
pub fn waitStatusCode(st: u32) i32 {
    if (std.posix.W.IFEXITED(st)) return std.posix.W.EXITSTATUS(st);
    return 128 + @as(i32, @intCast(st & 0x7f));
}

// =====================================================================
//  The runners — C: zincvm.c:1611-1769
// =====================================================================

/// C: zincvm.c:1732-1745 run_program — chains with &&/||/; short-circuit.
/// in_child marks execution inside a forked child (subshell body), where
/// single-builtin pipelines run in-process (POSIX semantics for builtins).
fn runProgram(prog: *RProg, outfd: i32, errfd: i32, in_child: bool) i32 {
    var last: i32 = 0;
    for (prog.chains, 0..) |*ch, i| {
        if (i > 0) {
            if (ch.op == .and_ and last != 0) continue;
            if (ch.op == .or_ and last == 0) continue;
        }
        const code = runPipeline(&ch.pipe, outfd, errfd, in_child);
        if (code < 0) return -1;
        last = code;
    }
    return last;
}

/// C: zincvm.c:1611-1725 run_pipeline — subshell (single Cmd with sub) ->
/// fork + recursive run_program; single builtin inside a CHILD process ->
/// in-process; else N children with N-1 pipes.  Exit = LAST stage (POSIX).
/// Returns exit code, or -1 on resource failure.
///
/// CHILD ORDER (C:1668-1673, critical):
///   1. dup2 defaults (stdout->outfd, stderr->errfd; stdin inherited)
///   2. pipe wiring (stdin <- prev read end; stdout -> this write end)
///   3. close ALL pipe fds + other stages' heredoc tmpfds
///   4. apply_redirects (left-to-right)
///   5. subshell re-entry | child builtin | execvp (THE single exec site)
fn runPipeline(pp: *RPipe, outfd: i32, errfd: i32, in_child: bool) i32 {
    const n = pp.cmds.len;

    // Subshell: one Cmd carrying a nested program (Argv empty).
    if (n == 1 and pp.cmds[0].sub != null) {
        const c = &pp.cmds[0];
        if (wirePipeTmpfds(pp) != 0) {
            closePipeTmpfds(pp);
            return -1;
        }
        const pid = fork();
        if (pid < 0) {
            closePipeTmpfds(pp);
            return -1;
        }
        if (pid == 0) {
            _ = dup2(outfd, 1);
            _ = dup2(errfd, 2);
            if (applyRedirects(c) < 0) _exit(1);
            // The subshell's own redirects already bound fd 1/2; drop the
            // capture originals (they'd leak into exec'd grandchildren and
            // restore the tmpfile onto fd 1, undoing a subshell redirect).
            if (outfd > 2) _ = close(outfd);
            if (errfd > 2 and errfd != outfd) _ = close(errfd);
            _exit(runProgram(c.sub.?, 1, 2, true));
        }
        closePipeTmpfds(pp);
        var st: c_int = 0;
        _ = waitpid(pid, &st, 0);
        return waitStatusCode(@bitCast(st));
    }

    // Single builtin inside a forked child (subshell body): run it
    // in-process so stateful builtins (cd) affect the child shell process,
    // with fds 0/1/2 saved and restored.  Never reached from the parent
    // exec-plan path (in_child == false there forks instead).
    if (in_child and n == 1 and pp.cmds[0].sub == null and
        pp.cmds[0].argv.len > 0 and isChildBuiltin(std.mem.sliceTo(pp.cmds[0].argv[0].?, 0)))
    {
        if (wirePipeTmpfds(pp) != 0) {
            closePipeTmpfds(pp);
            return -1;
        }
        const rc = runBuiltinInprocess(&pp.cmds[0], outfd, errfd);
        closeCmdTmpfds(&pp.cmds[0]);
        return rc;
    }

    // Pre-open heredoc tmpfiles (parent side, pre-fork).
    if (wirePipeTmpfds(pp) != 0) {
        closePipeTmpfds(pp);
        return -1;
    }

    var pfds: [2 * PLAN_MAX_CMDS]c_int = undefined;
    const npipes = n - 1;
    {
        var i: usize = 0;
        while (i < npipes) : (i += 1) {
            // C: pipe(&pfds[2*i]) — pipe() takes an int[2]; two adjacent
            // slots in pfds ARE an int[2] at that address.
            const pair: *[2]c_int = @ptrCast(pfds[2 * i ..].ptr);
            if (pipe(pair) < 0) {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    _ = close(pfds[2 * j]);
                    _ = close(pfds[2 * j + 1]);
                }
                closePipeTmpfds(pp);
                return -1;
            }
        }
    }

    var pids: [PLAN_MAX_CMDS]c_int = undefined;
    var forked: usize = 0;
    var fork_fail = false;
    {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const pid = fork();
            if (pid < 0) {
                fork_fail = true;
                break;
            }
            if (pid == 0) {
                const c = &pp.cmds[i];
                // 1. dup2 defaults
                _ = dup2(outfd, 1);
                _ = dup2(errfd, 2);
                // 2. pipe wiring
                if (i > 0) _ = dup2(pfds[2 * (i - 1)], 0);
                if (i < n - 1) _ = dup2(pfds[2 * i + 1], 1);
                // 3. close ALL pipe fds + other stages' heredoc tmpfds
                var j: usize = 0;
                while (j < 2 * npipes) : (j += 1) _ = close(pfds[j]);
                j = 0;
                while (j < n) : (j += 1) {
                    if (j != i) closeCmdTmpfds(&pp.cmds[j]);
                }
                // 4. apply_redirects (left-to-right)
                if (applyRedirects(c) < 0) _exit(1);
                // 5a. subshell re-entry
                if (c.sub != null) {
                    if (outfd > 2) _ = close(outfd);
                    if (errfd > 2 and errfd != outfd) _ = close(errfd);
                    _exit(runProgram(c.sub.?, 1, 2, true));
                }
                // exec path: the capture originals are dup'd onto 1/2 (or
                // about to be redirected); drop them so they don't leak
                // into execvp.
                if (outfd > 2) _ = close(outfd);
                if (errfd > 2 and errfd != outfd) _ = close(errfd);
                // 5b. child builtin
                const bcode = childBuiltin(c.argv.len, c.argv);
                if (bcode >= 0) _exit(bcode);
                // 5c. execvp — THE single exec site (C:1694)
                _ = execvp(c.argv[0].?, c.argv.ptr);
                if (std.c._errno().* == @intFromEnum(std.c.E.NOENT)) {
                    childW2("shensh: ", std.mem.sliceTo(c.argv[0].?, 0), ": not found\n");
                    _exit(127);
                }
                childW2("shensh: ", std.mem.sliceTo(c.argv[0].?, 0), ": cannot execute\n");
                _exit(126);
            }
            pids[i] = pid;
            forked += 1;
        }
    }

    // Parent: close every fd it opened.
    {
        var j: usize = 0;
        while (j < 2 * npipes) : (j += 1) _ = close(pfds[j]);
    }
    closePipeTmpfds(pp);

    if (fork_fail) {
        for (pids[0..forked]) |p| {
            _ = kill(p, SIGKILL);
            _ = waitpid(p, null, 0);
        }
        return -1;
    }

    var exit_code: i32 = 0;
    {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var st: c_int = 0;
            _ = waitpid(pids[i], &st, 0);
            if (i == n - 1) exit_code = waitStatusCode(@bitCast(st));
        }
    }
    return exit_code;
}

/// C: zincvm.c:1749-1769 slurp_fd — lseek(0) + read a captured tmpfile to
/// EOF.  Returns the FULL allocated slice (free with page_allocator) plus
/// the byte count; null on failure (mirrors C's NULL + *outlen = 0).
const Slurp = struct { buf: []u8, n: usize };

fn slurpFd(fd: i32) ?Slurp {
    var cap: usize = 8192;
    var buf = ca.alloc(u8, cap + 1) catch return null;
    if (lseek(fd, 0, SEEK_SET) < 0) {
        ca.free(buf);
        return null;
    }
    var n: usize = 0;
    while (true) {
        if (n + 4096 >= cap) {
            cap *= 2;
            buf = ca.realloc(buf, cap + 1) catch {
                ca.free(buf);
                return null;
            };
        }
        const r = read(fd, buf.ptr + n, 4096);
        if (r < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            break;
        }
        if (r == 0) break;
        n += @intCast(r);
    }
    buf[n] = 0;
    return .{ .buf = buf, .n = n };
}

// =====================================================================
//  exec-plan — C: zincvm.c:2100-2151
// =====================================================================

/// C: zincvm.c:2100-2151 exec-plan — run a decoded command PLAN natively
/// (fork/dup2/execvp, no /bin/sh).  Plan arrives as ONE tagged zinc-value
/// argument; stdout+stderr default to two unlinked tmpfiles; the result is
/// the TAGGED [exit stdout stderr] list built with the makeTagged* helpers
/// (rooted across each helper's allocations — the exact C:2133-2148 root
/// dance).  The plan Value is rooted once at entry; decode/run perform no
/// GC allocation, and every exit path pops (the single defer is C's
/// per-path gc_root_pop).
pub fn primExecPlan(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    var plan = interp.vaPop(stack);
    g.rootPushValue(&plan); // C:2102
    defer g.rootPop(); // every C exit path pops exactly once

    var prog: RProg = .{};
    if (!planDecode(plan, &prog)) {
        planFree(&prog);
        return vm.throwShen("exec-plan: malformed plan");
    }

    var otmpl = "/tmp/shensh-out.XXXXXX".*;
    var etmpl = "/tmp/shensh-err.XXXXXX".*;
    const outfd = mkstemp(&otmpl);
    const errfd = if (outfd >= 0) mkstemp(&etmpl) else -1;
    if (outfd < 0 or errfd < 0) {
        if (outfd >= 0) {
            _ = close(outfd);
            _ = unlink(&otmpl);
        }
        if (errfd >= 0) {
            _ = close(errfd);
            _ = unlink(&etmpl);
        }
        planFree(&prog);
        return vm.throwShen("exec-plan: tmpfile failed");
    }
    _ = unlink(&otmpl);
    _ = unlink(&etmpl);

    const code = runProgram(&prog, outfd, errfd, false);
    planFree(&prog);
    if (code < 0) {
        _ = close(outfd);
        _ = close(errfd);
        return vm.throwShen("exec-plan: fork/pipe failed");
    }

    const out_s = slurpFd(outfd);
    const err_s = slurpFd(errfd);
    _ = close(outfd);
    _ = close(errfd);
    defer if (out_s) |s| ca.free(s.buf);
    defer if (err_s) |s| ca.free(s.buf);

    // The C:2133-2148 root dance, verbatim: each intermediate is rooted
    // across the next helper's allocations, all popped before returning.
    var tag_out = makeTaggedString(vm, if (out_s) |s| s.buf[0..s.n] else "");
    g.rootPushValue(&tag_out);
    var tag_err = makeTaggedString(vm, if (err_s) |s| s.buf[0..s.n] else "");
    g.rootPushValue(&tag_err);
    var empty = makeTaggedNil(vm);
    g.rootPushValue(&empty);
    var err_list = makeTaggedCons(vm, tag_err, empty);
    g.rootPushValue(&err_list);
    var out_list = makeTaggedCons(vm, tag_out, err_list);
    g.rootPushValue(&out_list);
    var tag_code = makeTaggedNumber(vm, code);
    g.rootPushValue(&tag_code);
    const final = makeTaggedCons(vm, tag_code, out_list);
    g.rootPop(); // tag_code
    g.rootPop(); // out_list
    g.rootPop(); // err_list
    g.rootPop(); // empty
    g.rootPop(); // tag_err
    g.rootPop(); // tag_out
    // (the deferred rootPop for `plan` fires on return)

    acc.* = final;
}

// =====================================================================
//  cd / getcwd / getpid — C: zincvm.c:1955-1962 / 2191-2199 / 2205-2209
// =====================================================================

/// Copy a GC string into a NUL-terminated stack buffer (bounded by
/// buf.len-1); C's strndup-to-chdir/getenv pattern without any allocation.
fn strToBufZ(v: Value, buf: []u8) [:0]u8 {
    const s = values.strSlice(v);
    const n = @min(s.len, buf.len - 1);
    @memcpy(buf[0..n], s[0..n]);
    buf[n] = 0;
    return buf[0..n :0];
}

/// C: zincvm.c:1955-1962 cd: chdir to Path.  Returns raw boolean.
pub fn primCd(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const path = interp.vaPop(stack);
    if (path.tag != .string) return vm.throwShen("cd: path must be a string");
    var buf: [PATH_MAX]u8 = undefined;
    const p = strToBufZ(path, &buf);
    const rc = chdir(p.ptr);
    acc.* = values.valBoolean(rc == 0);
}

/// C: zincvm.c:2191-2199 getcwd.  Arity 0 — pop spurious arg if present
/// (newvar precedent).
pub fn primGetcwd(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    if (stack.len > 0) _ = interp.vaPop(stack);
    var buf: [PATH_MAX]u8 = undefined;
    if (getcwd(&buf, buf.len)) |p| {
        acc.* = values.valString(vm.gc, std.mem.sliceTo(p, 0));
    } else {
        acc.* = values.valString(vm.gc, "");
    }
}

/// C: zincvm.c:2205-2209 getpid.  Arity 0 with the same dummy-arg pop
/// (nullary prim calls cannot compile; used by $$ shell expansion).
pub fn primGetpid(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm; // no vm use once the is_wasm stub is stripped (native-only port)
    if (stack.len > 0) _ = interp.vaPop(stack);
    acc.* = values.valNumber(getpid());
}

// =====================================================================
//  getenv / setenv — C: 2211-2219 / 2580-2595
// =====================================================================

/// C: zincvm.c:2211-2219 getenv: value or "".
pub fn primGetenv(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const name_v = interp.vaPop(stack);
    if (name_v.tag != .string) return vm.throwShen("getenv: name must be a string");
    var buf: [PATH_MAX]u8 = undefined;
    const n = strToBufZ(name_v, &buf);
    const v = getenv(n.ptr);
    acc.* = if (v) |p|
        values.valString(vm.gc, std.mem.sliceTo(p, 0))
    else
        values.valString(vm.gc, "");
}

/// C: zincvm.c:2580-2595 setenv: Name Val -> true.  ZINC RTL: a1 = Name
/// (popped FIRST), a2 = Val.
pub fn primSetenv(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const name_v = interp.vaPop(stack);
    const val_v = interp.vaPop(stack);
    if (name_v.tag != .string) return vm.throwShen("setenv: name must be a string");
    if (val_v.tag != .string) return vm.throwShen("setenv: value must be a string");
    var nbuf: [PATH_MAX]u8 = undefined;
    const n = strToBufZ(name_v, &nbuf);
    var vbuf: [PATH_MAX]u8 = undefined;
    const v = strToBufZ(val_v, &vbuf);
    _ = setenv(n.ptr, v.ptr, 1);
    acc.* = values.valBoolean(true);
}

//  glob — C: zincvm.c:2224-2265
// =====================================================================

/// C: fnmatch(base, name, 0) — the REAL libc fnmatch (zincvm.c:2247 uses it
/// via <fnmatch.h>), so bracket expressions and escapes behave identically.
fn fm(pattern: [*:0]const u8, name: [*:0]const u8) bool {
    return fnmatch(pattern, name, 0) == 0;
}

/// strcmp order for the match sort (C qsort + cmp_str).
fn strLessThan(_: void, a: [:0]const u8, b: [:0]const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// C: zincvm.c:2224-2265 glob: Pattern -> sorted TAGGED list of matching
/// path strings.  Splits pattern into dirname + basename; opendir/readdir/
/// fnmatch; qsort; builds the tagged list inside-out (each makeTaggedCons
/// roots its args; `result` is rooted across iterations — C:2256-2262).
pub fn primGlob(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    const pat = interp.vaPop(stack);
    if (pat.tag != .string) return vm.throwShen("glob: pattern must be a string");

    // pattern copy (C truncates at PATH_MAX-1).
    var pattern: [PATH_MAX]u8 = undefined;
    const ps = values.strSlice(pat);
    const pl = @min(ps.len, PATH_MAX - 1);
    @memcpy(pattern[0..pl], ps[0..pl]);
    pattern[pl] = 0;

    // dir + base split at the LAST '/' (C strrchr).  base is NUL-terminated
    // (pattern[pl] == 0 above) so it can feed fnmatch directly.  NOTE: C does
    // NOT special-case a leading '/' — dlen==0 gives opendir("") which FAILS
    // and yields an empty list; that quirk is preserved verbatim.
    var dir: []const u8 = ".";
    var base: [:0]const u8 = pattern[0..pl :0];
    if (std.mem.lastIndexOfScalar(u8, pattern[0..pl], '/')) |si| {
        dir = pattern[0..si];
        base = pattern[si + 1 .. pl :0];
    }

    var dirbuf: [PATH_MAX]u8 = undefined;
    if (dir.len >= PATH_MAX) {
        acc.* = makeTaggedNil(vm);
        return;
    }
    @memcpy(dirbuf[0..dir.len], dir);
    dirbuf[dir.len] = 0;

    const d = opendir(dirbuf[0..dir.len :0].ptr) orelse {
        acc.* = makeTaggedNil(vm);
        return;
    };

    var matches: [1024][:0]const u8 = undefined;
    var nmatch: usize = 0;
    while (nmatch < 1024) {
        const ent = readdir(d) orelse break;
        // d_name is NUL-terminated by readdir even though the array is
        // fixed-size; re-slice with the sentinel so it can feed fnmatch.
        const name_raw = std.mem.sliceTo(&ent.d_name, 0);
        const name: [:0]const u8 = name_raw.ptr[0..name_raw.len :0];
        if (fm(base.ptr, name.ptr)) {
            // Dotfile guard (C:2248-2250): "." / ".." only when explicitly
            // matched by the pattern itself.
            if ((std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) and
                !fm(base.ptr, ".") and !fm(base.ptr, ".."))
                continue;
            matches[nmatch] = dupeZ(name) catch {
                _ = closedir(d);
                var i: usize = 0;
                while (i < nmatch) : (i += 1) ca.free(matches[i]);
                return vm.throwShen("glob: out of memory");
            };
            nmatch += 1;
        }
    }
    _ = closedir(d);

    std.sort.block([:0]const u8, matches[0..nmatch], {}, strLessThan);

    // Inside-out tagged list; `result` rooted across the loop (C:2256-2262).
    var result = makeTaggedNil(vm);
    g.rootPushValue(&result);
    var i: usize = nmatch;
    while (i > 0) {
        i -= 1;
        const s = makeTaggedString(vm, matches[i]);
        result = makeTaggedCons(vm, s, result);
    }
    g.rootPop();

    i = 0;
    while (i < nmatch) : (i += 1) ca.free(matches[i]);

    acc.* = result;
}
