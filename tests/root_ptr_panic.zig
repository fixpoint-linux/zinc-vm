//! tests/root_ptr_panic.zig — T9: the ROOT_PTR interior-pointer defense
//! (plan DECISION 7).
//!
//! Zig 0.16 has no in-process panic assertion (std.testing has no
//! expectPanics), so T9 runs as a BUILD-LEVEL expected-failure executable
//! wired into the `gc-test` step: this root module overrides the panic
//! handler (`pub const panic = std.debug.FullPanic(...)` — std/builtin.zig
//! :1221-1240), matches the scanRoots defense message, and exits 42 on the
//! expected panic / 1 otherwise.  The Run step asserts exit code 42
//! (build.zig), proving both "it panicked" AND "with the C message".
//!
//! The defense under test — collect.zig scanRoots, C: gc.c:1527-1539: a
//! ROOT_PTR slot must point at an object HEAD.  A pointer into a multi-page
//! object's CONTINUED tail page would make gcMove read a garbage header at
//! *(ptr-1) and silently rewrite the root wrong; the cheap defense is that a
//! head page is never CONTINUED (only tail pages are), so any CONTINUED page
//! under a ROOT_PTR is a fatal interior pointer.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;

/// Root panic override (std/builtin.zig:1223-1233): FullPanic wraps a plain
/// `fn (msg, ret_addr) noreturn`.  std.debug.panic → std.builtin.panic.call
/// funnels EVERY panic (formatted or safety) through here.
pub const panic = std.debug.FullPanic(caughtPanic);

fn caughtPanic(msg: []const u8, ra: ?usize) noreturn {
    _ = ra;
    // Expected: the ROOT_PTR interior defense (collect.zig scanRoots,
    // C: gc.c:1527-1539).  The message carries the runtime page number, so
    // match the stable prefix.
    if (std.mem.startsWith(u8, msg, "gc: ROOT_PTR points into a multi-page object tail"))
        std.process.exit(42); // expected panic observed
    std.process.exit(1); // some OTHER panic — fail the run step
}

pub fn main() void {
    var g = gc.heap.Gc.init(.{
        .heap_bytes = 16 * 1024 * 1024,
        .reserve_bytes = 64 * 1024 * 1024,
    }) catch std.process.exit(2);

    // 100 Values = 4000 B body = 501 words = 8 pages (allocatepage tags page
    // 0 OBJECT + 7 CONTINUED — heap.zig allocatepage, C: gc.c:1917-1923).
    const big = g.allocArrayOldgen(types.Value, 100);

    // &big[90] sits at byte offset 3600 (word 451 of the object) — page 7 of
    // 8, a CONTINUED tail page: a textbook invalid interior root.
    var slot: ?*anyopaque = @ptrCast(&big[90]);
    g.rootPushPtr(@ptrCast(&slot));

    // Must panic inside scanRoots BEFORE evacuate would read the bogus
    // header.  Driven through collectNursery so the M4 entry point is the
    // one exercised (scanRoots is shared with full collect, C: gc.c:1666).
    g.collectNursery(.@"test");

    // Reached only if the defense failed to fire.
    std.process.exit(1);
}
