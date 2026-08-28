//! src/gc/roots.zig — precise-root shadow stack + typed-walker registrations
//! (milestone M3).
//!
//! C origin: gc.c:310-325 (GcRoot record + the shadow_stack / reg_* statics),
//! gc.c:2331-2402 (shadow_stack_grow, the gc_root_push_* family,
//! gc_root_pop / gc_root_pop_to / gc_root_watermark, gc_register_*).
//!
//! The shadow stack is C-heap memory in C (malloc/realloc) — page_allocator
//! memory in the port — and is NEVER scanned or evacuated by the collector.
//! It is the SOLE authoritative root source: collect.zig's scanRoots walks it
//! plus the registered typed walkers.  There is no conservative C-stack scan
//! (plan DECISION 4).
//!
//! Port contract (plan DECISION 5): the stack storage lives as fields on the
//! Gc struct (heap.zig) — `shadow_stack` / `shadow_len` / `shadow_cap` /
//! `reg_*` — while this file owns the API surface and the implementation.
//! heap.zig declares thin method wrappers on Gc delegating here, so callers
//! get the plan DECISION 1 ergonomics (`gc.rootPushValue(&v)`).

const std = @import("std");
const heap = @import("heap.zig");
const types = @import("types.zig");

const Gc = heap.Gc;

/// C: gc.h:153-154 RootKind — kinds of precise roots.  Numeric values match
/// the C enum order (ROOT_PTR..ROOT_CALLFRAME_ARRAY) for any future C-ABI
/// interop.
pub const RootKind = enum(u32) {
    ROOT_PTR = 0, // single GC pointer slot; MUST point at an object HEAD
    ROOT_VALUE = 1, // by-value Value; interior pointers rewritten in place
    ROOT_VALUE_ARRAY = 2, // N by-value Values rooted by base + live count
    ROOT_VALUE_VOLATILE = 3, // volatile-qualified Value (copy/scan/copy-back)
    ROOT_CALLFRAME_ARRAY = 4, // N CallFrames (scanned by the drain, not here)
};

/// C: gc.c:312 `typedef struct { RootKind kind; void *slot; int *np; } GcRoot;`
pub const GcRoot = struct {
    kind: RootKind,
    /// C: `void *slot` — the ADDRESS of the rooted slot (a pointer variable, a
    /// Value, or an array base).  Stored opaquely; scanRoots casts back per
    /// `kind`.
    slot: *anyopaque,
    /// C: `int *np` — element-count pointer for ROOT_VALUE_ARRAY /
    /// ROOT_CALLFRAME_ARRAY, null otherwise.
    np: ?*i32,
};

/// C: gc.c:2331 SHADOW_STACK_INIT_CAP.
const SHADOW_STACK_INIT_CAP: usize = 64;

// ---------------------------------------------------------------------
//  Shadow stack growth — C: gc.c:2333-2338 shadow_stack_grow
// ---------------------------------------------------------------------

/// C: gc.c:2333-2338 shadow_stack_grow — double the capacity (or initialise
/// to SHADOW_STACK_INIT_CAP on the first push).  OOM panics with the C
/// message (C: gc.c:2336 exit(1) → std.debug.panic).
fn shadowStackGrow(gc: *Gc) void {
    const pa = std.heap.page_allocator;
    const nc = if (gc.shadow_cap != 0) gc.shadow_cap * 2 else SHADOW_STACK_INIT_CAP;
    // shadow_cap == 0 means the slice was never allocated (it is still the
    // default empty &.{}), so realloc cannot be used — alloc fresh.
    const new_slice = if (gc.shadow_cap != 0)
        pa.realloc(gc.shadow_stack, nc) catch {
            std.debug.panic("gc_root_push: realloc failed", .{});
        }
    else
        pa.alloc(GcRoot, nc) catch {
            std.debug.panic("gc_root_push: realloc failed", .{});
        };
    gc.shadow_stack = new_slice;
    gc.shadow_cap = nc;
}

// ---------------------------------------------------------------------
//  Push / pop / watermark — C: gc.c:2340-2382
// ---------------------------------------------------------------------

/// C: gc.c:2340-2346 gc_root_push_ptr.  `slot` is the address of a one-word
/// pointer variable that MUST point at the HEAD of a GC object (gc.h:155-161):
/// gc_move reads the header at *(ptr-1), so an interior/tail pointer into a
/// multi-page object would read a garbage header.  scanRoots defends against
/// this with the fatal CONTINUED-page check (collect.zig, C: gc.c:1521-1539).
pub fn rootPushPtr(gc: *Gc, slot: *anyopaque) void {
    if (gc.shadow_len >= gc.shadow_cap) shadowStackGrow(gc);
    gc.shadow_stack[gc.shadow_len] = .{ .kind = .ROOT_PTR, .slot = slot, .np = null };
    gc.shadow_len += 1;
}

/// C: gc.c:2348-2354 gc_root_push_value — by-value Value root.  scanValue
/// rewrites the Value's interior pointers in place; the Value itself never
/// moves (C parity, plan DECISION 6 rule 4).
pub fn rootPushValue(gc: *Gc, vslot: *types.Value) void {
    if (gc.shadow_len >= gc.shadow_cap) shadowStackGrow(gc);
    gc.shadow_stack[gc.shadow_len] = .{
        .kind = .ROOT_VALUE,
        .slot = @ptrFromInt(@intFromPtr(vslot)),
        .np = null,
    };
    gc.shadow_len += 1;
}

/// C: gc.c:2356-2362 gc_root_push_value_volatile — volatile-qualified Value.
/// scanRoots handles it copy-to-tmp / scan / copy-back through the volatile
/// pointer (C: gc.c:1546-1552).
pub fn rootPushValueVolatile(gc: *Gc, vslot: *volatile types.Value) void {
    if (gc.shadow_len >= gc.shadow_cap) shadowStackGrow(gc);
    gc.shadow_stack[gc.shadow_len] = .{
        .kind = .ROOT_VALUE_VOLATILE,
        .slot = @ptrFromInt(@intFromPtr(vslot)),
        .np = null,
    };
    gc.shadow_len += 1;
}

/// C: gc.c:2364-2370 gc_root_push_value_array — N by-value Values rooted by
/// their base pointer plus a LIVE element-count pointer (the count is read at
/// scan time, not push time — C parity).
pub fn rootPushValueArray(gc: *Gc, base: [*]types.Value, np: *i32) void {
    if (gc.shadow_len >= gc.shadow_cap) shadowStackGrow(gc);
    gc.shadow_stack[gc.shadow_len] = .{
        .kind = .ROOT_VALUE_ARRAY,
        .slot = @ptrFromInt(@intFromPtr(base)),
        .np = np,
    };
    gc.shadow_len += 1;
}

/// C: gc.c:2399-2402 gc_root_push_callframe_array.  ROOT_CALLFRAME_ARRAY is
/// deliberately a no-op at scan time: the Cheney drain scans CallFrame arrays
/// via the GC_TYPE_CALLFRAME_ARRAY case once their page is queued (the
/// frame_stack is rooted separately as ROOT_PTR).  An explicit walker here
/// would be redundant AND could crash during Phase-0 promotion if a
/// stack.data pointer reads a zero/invalid header — see C: gc.c:1560-1569
/// (Bug #6, "gcalloc: object too large").
pub fn rootPushCallframeArray(gc: *Gc, arr: [*]types.CallFrame, np: *i32) void {
    if (gc.shadow_len >= gc.shadow_cap) shadowStackGrow(gc);
    gc.shadow_stack[gc.shadow_len] = .{
        .kind = .ROOT_CALLFRAME_ARRAY,
        .slot = @ptrFromInt(@intFromPtr(arr)),
        .np = np,
    };
    gc.shadow_len += 1;
}

/// C: gc.c:2372-2374 gc_root_pop — pop one entry (no-op when empty).
pub fn rootPop(gc: *Gc) void {
    // SAFETY-ENFORCEMENT (unit B): an imbalanced pop (shadow_len == 0) is a
    // root-balance bug — panic loudly in Debug/ReleaseSafe.  In ReleaseFast
    // the assert compiles out and the guard still prevents the underflow
    // (parity with the previous behaviour).
    std.debug.assert(gc.shadow_len != 0);
    if (gc.shadow_len != 0) gc.shadow_len -= 1;
}

/// C: gc.c:2376-2378 gc_root_pop_to — truncate to a watermark (longjmp unwind).
pub fn rootPopTo(gc: *Gc, watermark: usize) void {
    // SAFETY-ENFORCEMENT (unit B): truncating ABOVE the current depth would
    // GROW the stack (a lost-balance bug); panic on it in Debug/ReleaseSafe.
    std.debug.assert(watermark <= gc.shadow_len);
    gc.shadow_len = watermark;
}

/// C: gc.c:2380-2382 gc_root_watermark — snapshot the current depth.
pub fn rootWatermark(gc: *const Gc) usize {
    return gc.shadow_len;
}

/// SAFETY-ENFORCEMENT (unit D): RAII root guard.  Created by the Gc
/// convenience methods in heap.zig (`gc.rootValue(&v)` etc.); the idiom is
/// `var g = gc.rootValue(&v); defer g.end();` — end() pops the pushed root
/// when the enclosing block unwinds.  Multiple guards pop in reverse
/// declaration order = LIFO = the correct stack discipline, so root BALANCE
/// is automatic for the common case (forgot-to-root and leaked-pop become
/// impossible).  A double-ended (copied) guard fires Unit B's underflow assert
/// in Debug.
pub const RootGuard = struct {
    gc: *Gc,
    pub fn end(self: *const RootGuard) void {
        self.gc.rootPop();
    }
};

// ---------------------------------------------------------------------
//  Typed-walker registrations — C: gc.c:2384-2397
// ---------------------------------------------------------------------

/// C: gc.c:2384-2387 gc_register_global_table — the defun table.  Walked
/// dirty-gated during nursery scavenges (bitset skips stable old-gen pages),
/// fully during full collects, and — for Phase-0 promotion — its non-dirty
/// slots are scanned explicitly by collect() itself (gc.c:663-670).
pub fn registerGlobalTable(gc: *Gc, table: [*]types.TableEntry, len_p: *i32) void {
    gc.reg_global_table = table;
    gc.reg_global_table_len = len_p;
}

/// C: gc.c:2389-2392 gc_register_values_table — always full-scanned (no dirty
/// bitset; see gc.c:1594-1601).
pub fn registerValuesTable(gc: *Gc, table: [*]types.TableEntry, len_p: *i32) void {
    gc.reg_values_table = table;
    gc.reg_values_table_len = len_p;
}

/// C: gc.c:2394-2397 gc_register_traced_code — each slot is evacuated
/// directly by scanRoots (gc.c:1604-1610).
pub fn registerTracedCode(gc: *Gc, arr: [*]?*types.Instr, np: *i32) void {
    gc.reg_traced_code = arr;
    gc.reg_traced_code_len = np;
}
