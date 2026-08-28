//! src/gc/scan.zig — typed scanning of GC-managed objects (milestone M2).
//!
//! C origin: scan_fns_zincvm_excerpt.c (the staged excerpt of the gc_scan_value /
//! gc_evacuate mode-agnostic scan functions) plus evac_instr (gc.c:471-474) and
//! value_references_nursery (zincvm.c:112-121).
//!
//! These functions are mode-agnostic: they serve both full collect (evacuate to
//! next_space) and nursery scavenge (nursery->old-gen), dispatched by gc_move
//! (collect.zig) via in_scavenge.  They rewrite pointer slots in place through
//! single-machine-word views (plan DECISION 6 / artifact-2 pattern): a slot's
//! address is cast to `*usize`, the address is read, run through gc_move, and
//! written back.  All optional-pointer fields on Value/Instr/CallFrame are exactly
//! one word (null == 0), verified on Zig 0.16.0 / LP64.

const std = @import("std");
const types = @import("types.zig");
const heap = @import("heap.zig");
const collect = @import("collect.zig");

const Gc = heap.Gc;

/// C: scan_fns gc_evacuate — *slot = gc_move(*slot).  Update a single pointer
/// slot to point to the evacuated copy.  `slot` is the address of a one-word
/// pointer field viewed as `*usize` (all GC-managed pointer slots are one word).
pub fn evacuate(gc: *Gc, slot: *usize) void {
    const addr: usize = slot.*;
    slot.* = @intFromPtr(collect.gcMove(gc, @ptrFromInt(addr)));
}

/// C: scan_fns gc_scan_value — evacuate all GC-managed pointers within a Value.
/// The pointer-bearing tags are VAL_CONS (car/cdr), VAL_LAMBDA (code/env),
/// VAL_VECTOR (data), VAL_STRING (data), VAL_ERROR (message); the remaining tags
/// (number, symbol, boolean, nil, mark, prim, stream) contain no GC-managed
/// pointers and are a no-op.
pub fn scanValue(gc: *Gc, v: *types.Value) void {
    switch (v.tag) {
        .cons => {
            evacuate(gc, @ptrCast(&v.payload.cons.car));
            evacuate(gc, @ptrCast(&v.payload.cons.cdr));
        },
        .lambda => {
            evacuate(gc, @ptrCast(&v.payload.lambda.code));
            evacuate(gc, @ptrCast(&v.payload.lambda.env));
        },
        .vector => evacuate(gc, @ptrCast(&v.payload.vector.data)),
        .string => evacuate(gc, @ptrCast(&v.payload.str.data)),
        .error_ => evacuate(gc, @ptrCast(&v.payload.error_.message)),
        else => {},
    }
}

/// C: gc.c:471-474 evac_instr — scan a single Instr for GC pointers: the operand
/// Value (via scanValue) and the closure_code pointer (via evacuate).
pub fn evacInstr(gc: *Gc, in: *types.Instr) void {
    scanValue(gc, &in.operand);
    evacuate(gc, @ptrCast(&in.closure_code));
}

/// C: zincvm.c:112-121 value_references_nursery — true iff `v` references any GC
/// object in the nursery.  Must mirror EXACTLY the pointer fields gc_scan_value
/// evacuates.  NULL-safe: cons/lambda fields pass a null pointer through
/// inNursery (page of 0 < firstheappage, returns false); vector/string/error
/// check non-null first (C parity).
pub fn valueReferencesNursery(gc: *const Gc, v: *const types.Value) bool {
    return switch (v.tag) {
        .cons => gc.inNursery(@as(*const usize, @ptrCast(&v.payload.cons.car)).*) or
            gc.inNursery(@as(*const usize, @ptrCast(&v.payload.cons.cdr)).*),
        .lambda => gc.inNursery(@as(*const usize, @ptrCast(&v.payload.lambda.code)).*) or
            gc.inNursery(@as(*const usize, @ptrCast(&v.payload.lambda.env)).*),
        .vector => (v.payload.vector.data != null) and
            gc.inNursery(@as(*const usize, @ptrCast(&v.payload.vector.data)).*),
        .string => (v.payload.str.data != null) and
            gc.inNursery(@as(*const usize, @ptrCast(&v.payload.str.data)).*),
        .error_ => (v.payload.error_.message != null) and
            gc.inNursery(@as(*const usize, @ptrCast(&v.payload.error_.message)).*),
        else => false,
    };
}
