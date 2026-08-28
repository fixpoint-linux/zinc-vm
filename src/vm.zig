//! src/vm.zig — module root for the Shen ZINC VM port.
//!
//! Layout (plan DECISION / ARCHITECTURE):
//!   vm/state.zig    — Vm struct (owns *Gc, err slot, symbol interner, ...)
//!   vm/values.zig   — value model (val_* constructors, print_value, str_value,
//!                     deep_equal) — M0
//!   vm/symbols.zig  — symbol interner (sym_intern_hash / sym_dyn_get /
//!                     sym_dyn_resize) + val_symbol sans static store — M1
//!   vm/tables.zig   — defun/values global tables + GC registration glue — M2
//!   vm/parser.zig   — csexp parser + resolve_jumps + print_instr debug aid — M3
//!   vm/interp.zig   — ValueArray + env + the eval loop (vm_exec_env) — M4
//!   vm/prims.zig    — prim table + dispatch + the pure-subset exec_primitive — M5

pub const state = @import("vm/state.zig");
pub const values = @import("vm/values.zig");
pub const symbols = @import("vm/symbols.zig");
pub const tables = @import("vm/tables.zig");
pub const parser = @import("vm/parser.zig");
pub const interp = @import("vm/interp.zig");
pub const prims = @import("vm/prims.zig");
pub const streams = @import("vm/streams.zig");
pub const execplan = @import("vm/execplan.zig");
pub const hostcall = @import("vm/hostcall.zig");
pub const effectloop = @import("vm/effectloop.zig");
