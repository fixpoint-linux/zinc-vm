//! src/gc.zig — module root for the Shen GC port.
//!
//! Layout (plan DECISION 3):
//!   gc/types.zig  — zinctypes.h + gc.h type layer (M0, done)
//!   gc/heap.zig   — Gc struct state + memory-management core (M1)
//!   gc/collect.zig— collector entry points (M2/M3/M4; M1 ships stubs)
//!   gc/scan.zig   — typed scanning (M2)
//!   gc/roots.zig  — precise-root shadow stack + registrations (M3)

pub const types = @import("gc/types.zig");
pub const heap = @import("gc/heap.zig");
pub const collect = @import("gc/collect.zig");
pub const scan = @import("gc/scan.zig");
pub const roots = @import("gc/roots.zig");

/// Convenience re-export: `gc.Gc` is the collector handle.
pub const Gc = heap.Gc;
