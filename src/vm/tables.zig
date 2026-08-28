//! src/vm/tables.zig — defun/values global tables (milestone M2).
//!
//! C origin: zincvm.c:460-696 (hash_name, defun_set, defun_get, defun_has,
//! value_set, value_get).  Plan DECISION B: ONE open-addressed defun table
//! (FNV-1a `hash_name` mod cap, linear probing) over DEFUN_TABLE_CAP=4096 as a
//! zeroed page_allocator C-heap array field of Vm (NOT the GC heap — C BSS
//! parity), registered at Vm.init via gc.registerGlobalTable.  C's
//! bootstrap_keys + defun_freeze minimal-perfect-hash + overflow tail
//! (zincvm.c:482-497, 1869-4005, ~250 lines) are DELIBERATELY OMITTED: writing
//! straight into the REGISTERED + dirty-marked table is strictly safer (the C
//! bootstrap has a latent GC hazard where malloc'd bootstrap closures are
//! invisible to a mid-load scavenge).
//!
//! The values table is a direct port of value_set/value_get (open addressing,
//! VALUES_TABLE_CAP=256), registered via gc.registerValuesTable, always
//! full-scanned (no dirty bitset).  Table-full -> std.debug.panic (C exit(1)
//! mapping per docs/gc-zig.md).
//!
//! defun_get's primitive fallback (C:610-617 — known prim -> valPrim, else
//! valSymbol) is applied by the Vm wrapper (state.zig); here we expose the raw
//! lookup (defunLookup) returning an optional Value.  M5 completes the prim
//! half of the fallback when prims.zig exists.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;

/// C: zincvm.h DEFUN_TABLE_CAP — the GC dirty-defuns bitset is exactly this many
/// (heap.zig:64), so the table size and the bitset always agree.
pub const DEFUN_TABLE_CAP: usize = 4096;
/// C: zincvm.h VALUES_TABLE_CAP.
pub const VALUES_TABLE_CAP: usize = 256;

/// C: zincvm.c:503-510 hash_name — FNV-1a reduced mod cap.  Used by the values
/// table; the defun table shares the same open-addressed layout (DECISION B).
pub fn hashName(name: []const u8, cap: usize) usize {
    var h: u32 = 2166136261;
    for (name) |c| {
        h ^= @as(u32, c);
        h *%= 16777619;
    }
    return @as(usize, h) % cap;
}

/// Duplicate `name` into immortal C-heap memory (C strdup parity — the GC
/// treats a TableEntry.name as an opaque non-GC pointer and only null-checks it,
/// plan TABLES<->GC integration point 6).
fn dupName(name: []const u8) [*:0]u8 {
    const a = std.heap.page_allocator;
    const dup = a.dupeZ(u8, name) catch std.debug.panic("tables: strdup failed", .{});
    return @ptrCast(dup.ptr);
}

/// C: zincvm.c:537-590 defun_set (RUNTIME mode only; DECISION B removes
/// bootstrap).  Insert OR overwrite `name -> v`, always marking the slot dirty
/// via gc.dirtyDefunsMark (a slot may reference a nursery object).  Open
/// addressing, linear probing.  Table full -> std.debug.panic.
pub fn defunSet(g: *gc.Gc, table: [*]types.TableEntry, cap: usize, name: []const u8, v: types.Value) void {
    const n = std.mem.sliceTo(name, 0); // C string semantics
    const h = hashName(n, cap);
    var i: usize = 0;
    while (i < cap) : (i += 1) {
        const idx = (h + i) % cap;
        const e = &table[idx];
        if (e.name == null) {
            e.name = dupName(n);
            e.value = v;
            g.dirtyDefunsMark(@intCast(idx));
            return;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(e.name.?, 0), n)) {
            e.value = v; // later store wins
            g.dirtyDefunsMark(@intCast(idx));
            return;
        }
    }
    std.debug.panic("defun table full on '{s}'", .{n});
}

/// Raw defun-table lookup.  Returns null when `name` has no explicit entry.
/// The C primitive/symbol fallback (C:610-617) is applied by the Vm wrapper.
pub fn defunLookup(table: [*]types.TableEntry, cap: usize, name: []const u8) ?types.Value {
    const n = std.mem.sliceTo(name, 0);
    const h = hashName(n, cap);
    var i: usize = 0;
    while (i < cap) : (i += 1) {
        const idx = (h + i) % cap;
        const e = &table[idx];
        if (e.name == null) return null; // open addressing: end of probe run
        if (std.mem.eql(u8, std.mem.sliceTo(e.name.?, 0), n)) return e.value;
    }
    return null;
}

/// C: zincvm.c:624-642 defun_has — probe whether the defun table has an
/// explicit entry (NO val_prim/val_symbol fallback).  Used by bundle-load
/// keyword registration (M6) to avoid clobbering bundled closures.
pub fn defunHas(table: [*]types.TableEntry, cap: usize, name: []const u8) bool {
    const n = std.mem.sliceTo(name, 0);
    const h = hashName(n, cap);
    var i: usize = 0;
    while (i < cap) : (i += 1) {
        const idx = (h + i) % cap;
        const e = &table[idx];
        if (e.name == null) return false;
        if (std.mem.eql(u8, std.mem.sliceTo(e.name.?, 0), n)) return true;
    }
    return false;
}

/// C: zincvm.c:648-665 value_set — open-address insert with linear probing.
/// No dirty-bitset: the GC always full-scans the values table.  Table full ->
/// std.debug.panic (C exit(1)).
pub fn valueSet(table: [*]types.TableEntry, cap: usize, name: []const u8, v: types.Value) void {
    const n = std.mem.sliceTo(name, 0);
    var h = hashName(n, cap);
    // C reserves one slot (`values_table_used >= VALUES_TABLE_CAP - 1`).
    var i: usize = 0;
    while (i < cap) : (i += 1) {
        const e = &table[h];
        if (e.name == null) {
            e.name = dupName(n);
            e.value = v;
            return;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(e.name.?, 0), n)) {
            e.value = v; // overwrite
            return;
        }
        h = (h + 1) % cap;
    }
    std.debug.panic("values table full ({d} entries) on '{s}'", .{ cap, n });
}

/// C: zincvm.c:668-676 value_get — no primitive fallback: (value +) must return
/// the bare symbol `+`.  Returns null when absent; the Vm wrapper applies the
/// val_symbol fallback.
pub fn valueLookup(table: [*]types.TableEntry, cap: usize, name: []const u8) ?types.Value {
    const n = std.mem.sliceTo(name, 0);
    var h = hashName(n, cap);
    var i: usize = 0;
    while (i < cap) : (i += 1) {
        const e = &table[h];
        if (e.name == null) return null;
        if (std.mem.eql(u8, std.mem.sliceTo(e.name.?, 0), n)) return e.value;
        h = (h + 1) % cap;
    }
    return null;
}
