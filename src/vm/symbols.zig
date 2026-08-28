//! src/vm/symbols.zig — the dynamic symbol interner (milestone M1).
//!
//! C origin: zincvm.c:197-269 (sym_intern_hash, sym_dyn_resize, sym_dyn_get)
//! + val_symbol (zincvm.c:260-269) SANS the static store (plan DECISION C).
//!
//! DECISION C: only the DYNAMIC store is ported.  C's static store
//! (symbol_static_lookup, from generated symbol_static.c which is NOT in
//! reference/) is a closed-world QBE optimization and is omitted.  Canonical
//! pointer equality still holds: every name interns to exactly one immortal
//! C-heap string.
//!
//! Interning is C-heap only (page_allocator.dupeZ), immortal strings (never
//! freed — the C frees them only on resize re-bucket, never on the strings
//! themselves), and a resize only RE-BUCKETS the existing pointers, keeping
//! every outstanding VAL_SYMBOL.sym.name valid and canonical.  deinit frees
//! only the bucket array; the immortal strings are reclaimed by the OS.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;

const Value = types.Value;

/// C: zincvm.c:197 SYMBOL_DYN_INIT.
pub const SYMBOL_DYN_INIT: usize = 256;
/// C: zincvm.c:198 SYMBOL_DYN_MAXLOAD — grow at >70% load.
pub const SYMBOL_DYN_MAXLOAD: usize = 70;

/// C: zincvm.c:205-209 sym_intern_hash — DJB2.
fn internHash(s: []const u8) u32 {
    var h: u32 = 5381;
    for (s) |c| h = ((h << 5) +% h) +% c;
    return h;
}

/// The open-addressed, power-of-two, dynamically-grown symbol interner.
/// C: zincvm.c:201-258 (symbol_dyn / symbol_dyn_cap / symbol_dyn_count as
/// struct fields instead of C statics).
pub const SymbolInterner = struct {
    table: []?[*:0]const u8 = &.{},
    cap: usize = 0,
    count: usize = 0,
    allocator: std.mem.Allocator = std.heap.page_allocator,

    /// Fresh (empty) interner — the table is lazily allocated on the first
    /// intern (C: `if (!symbol_dyn) sym_dyn_resize(SYMBOL_DYN_INIT)`).
    pub fn init() SymbolInterner {
        return .{};
    }

    /// Free the bucket array only.  The interned strings are immortal by
    /// design (plan DECISION C) and reclaimed by the OS at process exit.
    pub fn deinit(self: *SymbolInterner) void {
        const a = self.allocator;
        if (self.cap != 0) a.free(self.table);
        self.* = undefined;
    }

    /// C: zincvm.c:216-230 sym_dyn_resize — (re)size to a power-of-two cap and
    /// rehash every existing (immortal, never-freed) string pointer.  Only the
    /// bucket array is realloc'd; the char* values are moved, not copied/freed.
    fn resize(self: *SymbolInterner, newcap: usize) void {
        const a = self.allocator;
        const newtab = a.alloc(?[*:0]const u8, newcap) catch {
            std.debug.panic("sym_dyn_resize: alloc failed", .{});
        };
        @memset(newtab, null); // calloc parity (C: calloc(newcap, sizeof(char*)))
        if (self.cap != 0) {
            for (self.table) |s| {
                const sp = s orelse continue;
                const h = internHash(std.mem.sliceTo(sp, 0)) & (newcap - 1);
                var j: usize = 0;
                while (j < newcap) : (j += 1) {
                    const idx = (h + j) & (newcap - 1);
                    if (newtab[idx] == null) {
                        newtab[idx] = sp;
                        break;
                    }
                }
            }
            a.free(self.table);
        }
        self.table = newtab;
        self.cap = newcap;
    }

    /// C: zincvm.c:235-258 sym_dyn_get — look up OR intern `name`, returning
    /// its canonical pointer.  Grows before insert when the load factor is
    /// high (shrink is dead code: symbols are immortal, so it is omitted).
    pub fn intern(self: *SymbolInterner, name: []const u8) [*:0]const u8 {
        const a = self.allocator;
        // C semantics: name is a null-terminated string.  Normalize a caller's
        // slice to its leading null-terminated prefix so hashing (sym_intern_hash
        // stops at *s) and the equality check (strcmp) agree whether the caller
        // passes a trimmed slice or a whole fixed-size buffer containing the
        // name plus trailing NUL/garbage.
        const n = std.mem.sliceTo(name, 0);
        if (self.cap == 0) {
            self.resize(SYMBOL_DYN_INIT);
        } else if (self.count * 100 / self.cap > SYMBOL_DYN_MAXLOAD) {
            self.resize(self.cap * 2);
        }

        const h = internHash(n) & (self.cap - 1);
        // Lookup — empty slot => not in table yet.
        var i: usize = 0;
        while (i < self.cap) : (i += 1) {
            const idx = (h + i) & (self.cap - 1);
            const existing = self.table[idx] orelse break;
            if (std.mem.eql(u8, std.mem.sliceTo(existing, 0), n)) return existing;
        }
        // Insert — growth above guarantees a free slot is reachable.
        i = 0;
        while (i < self.cap) : (i += 1) {
            const idx = (h + i) & (self.cap - 1);
            if (self.table[idx] == null) {
                const dup = a.dupeZ(u8, n) catch {
                    std.debug.panic("sym_dyn_get: strdup failed", .{});
                };
                const canon: [*:0]const u8 = @ptrCast(dup.ptr);
                self.table[idx] = canon;
                self.count += 1;
                return canon;
            }
        }
        unreachable; // unreachable; defensive (growth guarantees a free slot)
    }
};

/// C: zincvm.c:260-269 val_symbol, SANS the static store (plan DECISION C):
/// the canonical pointer is always `sym.intern(name)`.  No GC allocation.
pub fn valSymbol(sym: *SymbolInterner, name: []const u8) Value {
    const canon = sym.intern(name);
    return .{ .tag = .symbol, .payload = .{ .sym = .{ .name = canon } } };
}
