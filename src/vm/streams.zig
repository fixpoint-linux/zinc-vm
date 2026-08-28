//! src/vm/streams.zig — string-stream registry + the stream I/O prims
//! (write-byte, read-byte, read-file-as-string, open, close) for milestone M6.
//!
//! C origin: zincvm.c stream storage + prim cases:
//!   - string_streams / n_string_streams / val_string_stream_in :358-381
//!   - close  :1926-1937   open :2346-2363   read-byte :2382-2395
//!   - read-file-as-string :2396-2412        write-byte :2685-2690
//!
//! VALUE MODEL (port of C's FILE* + is_string juggling):
//!   - A FILE stream carries its fd as an int-in-?*anyopaque: `stream.file =
//!     @ptrFromInt(fd)`.  fd 0 (stdin) therefore round-trips as a NULL
//!     optional pointer (C's real stdin FILE* is never null, but a null file
//!     pointer is exactly what valStreamIn(null) produced before M1, and the
//!     handler treats null == fd 0).
//!   - A STRING stream carries (idx+1) as an int-in-?*anyopaque so the value 0
//!     (= null) never appears, exactly like C's `+1 so 0 = no string stream`
//!     comment (:377).  The slot data is a page_allocator buffer (non-GC), so
//!     values built from it satisfy the valString CONTRACT.
//!
//! The registry is Vm-owned (a fixed array of 8 slots + a count), mirroring
//! C's file-scope statics without hidden globals.  None of these handlers
//! GC-allocate except valString/valError results copied from C-heap buffers
//! (safe by contract); string-stream data is page_allocator.
//!
//! NATIVE-ONLY: this is the M6 port trimmed to the native stream prims.  The
//! shen/zig source's wasm paths (is_wasm gate, wasm_out drain, diag freestanding
//! branch) are comptime-dead on native and are NOT carried over.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const state = @import("state.zig");
const values = @import("values.zig");
const interp = @import("interp.zig");

const Gc = gc.Gc;
const Value = types.Value;
const ValueArray = types.ValueArray;
const Vm = state.Vm;
const VmError = state.VmError;

/// C: zincvm.c:360 MAX_STRING_STREAMS.
pub const MAX_STRING_STREAMS = 8;

/// C: zincvm.c:361-362 string_streams[] + n_string_streams.  One slot per
/// live string stream; a closed slot stays consumed (C parity: close frees
/// and NULLs data but never decrements n_string_streams).
const StringStream = struct {
    data: ?[*]u8 = null,
    len: i32 = 0,
    pos: i32 = 0,
};

/// The Vm-owned registry: fixed array of MAX_STRING_STREAMS slots + count.
/// Zero-initialized (`.{ }` = all-null slots, count 0), so a fresh Vm needs
/// no setup.
pub const StreamRegistry = struct {
    slots: [MAX_STRING_STREAMS]StringStream = [_]StringStream{StringStream{}} ** MAX_STRING_STREAMS,
    n_string_streams: i32 = 0,

    /// C: zincvm.c:364-381 val_string_stream_in.  Copies `src` into a fresh
    /// page_allocator buffer and returns a VAL_STREAM whose `file` holds
    /// (idx+1) int-in-pointer with is_string=1.  Overflow returns a
    /// VAL_ERROR (C val_error) with a stderr note.
    pub fn valStringStreamIn(self: *StreamRegistry, g: *Gc, src: []const u8) Value {
        if (self.n_string_streams >= MAX_STRING_STREAMS) {
            std.debug.print("runtime: too many string streams\n", .{});
            return values.valError(g, "too many string streams");
        }
        const idx: usize = @intCast(self.n_string_streams);
        const a = std.heap.page_allocator;
        const data = a.alloc(u8, src.len + 1) catch {
            std.debug.print("runtime: string stream alloc failed\n", .{});
            return values.valError(g, "out of memory");
        };
        @memcpy(data[0..src.len], src);
        data[src.len] = 0;
        self.slots[idx] = .{ .data = data.ptr, .len = @intCast(src.len), .pos = 0 };
        self.n_string_streams += 1;
        return .{ .tag = .stream, .payload = .{ .stream = .{
            .file = @ptrFromInt(idx + 1), // +1 so null never appears
            .is_input = 1,
            .is_string = 1,
        } } };
    }

    /// C: close's string-stream arm (:1928-1933): free the data buffer and
    /// NULL the slot (the slot index stays consumed).  Idempotent-safe: a
    /// slot whose data is already null is a no-op.
    pub fn freeStringStream(self: *StreamRegistry, idx: usize) void {
        const ss = &self.slots[idx];
        if (ss.data) |d| {
            const len: usize = @intCast(ss.len);
            std.heap.page_allocator.free(d[0 .. len + 1]);
        }
        ss.* = .{};
    }
};

/// Read the fd back from a FILE stream's int-in-?*anyopaque payload.
/// A null optional == fd 0 (stdin); otherwise @intFromPtr.
fn streamFd(s: Value) i32 {
    const f = s.payload.stream.file;
    if (f == null) return 0;
    return @intCast(@intFromPtr(f.?));
}

/// Build a FILE stream value with the fd stored int-in-?*anyopaque.
fn fileStreamIn(fd: i32) Value {
    return values.valStreamIn(@ptrFromInt(@as(usize, @intCast(fd))));
}

/// Build a FILE stream value with the fd stored int-in-?*anyopaque.
fn fileStreamOut(fd: i32) Value {
    return values.valStreamOut(@ptrFromInt(@as(usize, @intCast(fd))));
}

/// Public helpers for the host (loadBundle wires *stinput*/*stoutput*/
/// /*sterror* to the real std fds 0/1/2).
pub fn valStreamInFd(fd: i32) Value {
    return fileStreamIn(fd);
}

pub fn valStreamOutFd(fd: i32) Value {
    return fileStreamOut(fd);
}

// =====================================================================
//  write-byte — C: zincvm.c:2685-2690
// =====================================================================

/// C: write-byte (2 args, RTL: pop byte THEN stream).  Write one byte to the
/// stream's fd; NO buffering (raw write), so no flush is needed.  Returns the
/// byte written.  C's stdout-fflush special-case is dead here: raw fd writes
/// are unbuffered by construction.
pub fn primWriteByte(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    _ = vm;
    const byte = interp.vaPop(stack);
    const s = interp.vaPop(stack);
    const fd = streamFd(s);
    const b: [1]u8 = .{@truncate(@as(u64, @bitCast(byte.payload.number)))};
    _ = std.posix.system.write(fd, &b, 1);
    acc.* = byte;
}

// =====================================================================
//  read-byte — C: zincvm.c:2382-2395
// =====================================================================

/// C: read-byte (1 arg).  String stream: decode (idx+1), pos<len else -1,
/// else the byte at pos (post-increment).  File stream: one-byte read with
/// EINTR retry (std.posix.read retries EINTR internally), EOF -> -1.  A hard
/// bad string-stream idx is error.Halt (the C `return -1`).
pub fn primReadByte(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const s = interp.vaPop(stack);
    if (s.payload.stream.is_string != 0) {
        const idx: i64 = @as(i64, @intCast(@intFromPtr(s.payload.stream.file.?))) - 1;
        if (idx < 0 or idx >= vm.streams.n_string_streams) return error.Halt;
        const ss = &vm.streams.slots[@intCast(idx)];
        if (ss.pos >= ss.len) {
            acc.* = values.valNumber(-1);
        } else {
            const b = ss.data.?[@intCast(ss.pos)];
            ss.pos += 1;
            acc.* = values.valNumber(b);
        }
        return;
    }
    const fd = streamFd(s);
    var buf: [1]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch {
        acc.* = values.valNumber(-1);
        return;
    };
    if (n == 0) {
        acc.* = values.valNumber(-1); // EOF
        return;
    }
    acc.* = values.valNumber(buf[0]);
}

// =====================================================================
//  read-file-as-string — C: zincvm.c:2396-2412
// =====================================================================

/// C: read-file-as-string (1 arg): read the whole file into a C-heap buffer
/// and val_string it.  The port reads into a page_allocator ArrayList (non-GC)
/// and valString copies from it (the valString CONTRACT holds).  On open
/// failure: stderr note + empty string (C parity).  valString is the only GC
/// allocation; buf.items is page_allocator so it never goes stale.
pub fn primReadFileAsString(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const g = vm.gc;
    const path = interp.vaPop(stack);
    const p = values.strSlice(path);
    const fd = std.posix.openat(std.posix.AT.FDCWD, p, .{}, 0) catch {
        std.debug.print("runtime: cannot open file for read-file-as-string\n", .{});
        acc.* = values.valString(g, "");
        return;
    };
    defer _ = std.posix.system.close(fd);
    const a = std.heap.page_allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(a);
    var tmp: [65536]u8 = undefined;
    while (true) {
        const n = std.posix.read(fd, &tmp) catch break;
        if (n == 0) break;
        buf.appendSlice(a, tmp[0..n]) catch break;
    }
    acc.* = values.valString(g, buf.items);
}

// =====================================================================
//  open — C: zincvm.c:2346-2363
// =====================================================================

/// C: open (2 args, RTL: pop path THEN dir).  'in': O_RDONLY; ENOENT
/// (error.FileNotFound) -> a STRING stream of the PATH bytes (the C quirk,
/// ported verbatim), any other open error -> false.  'out': O_WRONLY|O_CREAT|
/// O_TRUNC 0666, fail -> false.  A dir that is neither 'in' nor 'out' leaves
/// acc unchanged (C falls through to `break`).
///
/// DEVIATION (documented): the dir argument is a STRING ('in'/'out'), not a
/// symbol — Elm has no symbol literals, so the runtime passes "in"/"out" and
/// the port compares strSlice bytes (shen's symSlice).  C truncates the path
/// to 255 bytes into a fixed buffer before fopen; the port passes the full
/// path to openat (strictly more correct; no observable difference for paths
/// < 255, the only regime C supports).
pub fn primOpen(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const path = interp.vaPop(stack);
    const dir = interp.vaPop(stack);
    const p = values.strSlice(path);
    const d = values.strSlice(dir);
    if (std.mem.eql(u8, d, "in")) {
        const fd = std.posix.openat(std.posix.AT.FDCWD, p, .{}, 0) catch |e| {
            if (e == error.FileNotFound) {
                acc.* = vm.streams.valStringStreamIn(vm.gc, p);
                return;
            }
            acc.* = values.valBoolean(false);
            return;
        };
        acc.* = fileStreamIn(fd);
        return;
    }
    if (std.mem.eql(u8, d, "out")) {
        const flags: std.posix.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true };
        const fd = std.posix.openat(std.posix.AT.FDCWD, p, flags, 0o666) catch {
            acc.* = values.valBoolean(false);
            return;
        };
        acc.* = fileStreamOut(fd);
        return;
    }
    // neither 'in' nor 'out': fall through (C: acc unchanged).
}

// =====================================================================
//  close — C: zincvm.c:1926-1937
// =====================================================================

/// C: close (1 arg).  String stream: free data + NULL the slot (slot stays
/// consumed); a bad idx is a hard stop (error.Halt).  File stream: close the
/// fd only when the file pointer is non-null (C's `if (s.stream.file)`, which
/// skips null=fd0/stdin) — matching C's guard against closing an absent file.
pub fn primClose(vm: *Vm, acc: *Value, stack: *ValueArray) VmError!void {
    const s = interp.vaPop(stack);
    if (s.payload.stream.is_string != 0) {
        const idx: i64 = @as(i64, @intCast(@intFromPtr(s.payload.stream.file.?))) - 1;
        if (idx < 0 or idx >= vm.streams.n_string_streams) {
            std.debug.print("runtime: bad string stream idx\n", .{});
            return error.Halt;
        }
        vm.streams.freeStringStream(@intCast(idx));
        acc.* = values.valNil();
        return;
    }
    if (s.payload.stream.file != null) {
        _ = std.posix.system.close(streamFd(s));
    }
    acc.* = values.valNil();
}
