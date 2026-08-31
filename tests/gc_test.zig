//! tests/gc_test.zig — M0 smoke tests for the ported type layer.
//! Covers the load-bearing size-class asserts (already comptime-asserted in
//! types.zig) plus the header helper arithmetic and opcode char mapping.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;

test "M0 type layout size classes" {
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(types.Value));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(types.Instr));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(types.CallFrame));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(types.ValueArray));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(types.Value));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(types.Instr));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(types.CallFrame));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(types.ValueArray));
    // LP64 requirement (Phase 3/4).
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(usize));
}

test "M0 ValTag enum values match C order" {
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(types.ValTag.number));
    try std.testing.expectEqual(@as(u32, 4), @intFromEnum(types.ValTag.cons));
    try std.testing.expectEqual(@as(u32, 5), @intFromEnum(types.ValTag.nil));
    try std.testing.expectEqual(@as(u32, 9), @intFromEnum(types.ValTag.error_));
    try std.testing.expectEqual(@as(u32, 10), @intFromEnum(types.ValTag.vector));
    try std.testing.expectEqual(@as(u32, 11), @intFromEnum(types.ValTag.stream));
}

test "M0 header helpers arithmetic" {
    const words: usize = 3;
    const hdr = types.makeHeader(words, .value);
    try std.testing.expectEqual(words, types.headerWords(hdr));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.value)), types.headerType(hdr));
    try std.testing.expect(!types.forwarded(hdr)); // live object: bit0 == 1

    // Type tag occupies bits 25..; words 1..25; bit0 is the live marker.
    try std.testing.expectEqual(@as(usize, 1), hdr & 1);

    // A forwarding pointer clears bit0.
    const fwd = hdr & ~@as(usize, 1);
    try std.testing.expect(types.forwarded(fwd));

    // Full-width test: max words + a type tag still round-trips.
    const big_words: usize = 0x1FFFFF;
    const big = types.makeHeader(big_words, .callframe_array);
    try std.testing.expectEqual(big_words, types.headerWords(big));
    try std.testing.expectEqual(
        @as(u32, @intFromEnum(types.GcTypeTag.callframe_array)),
        types.headerType(big),
    );
}

test "M0 GcTypeTag values match gc.h" {
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(types.GcTypeTag.raw));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(types.GcTypeTag.value));
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(types.GcTypeTag.value_array));
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(types.GcTypeTag.instr_array));
    try std.testing.expectEqual(@as(u32, 4), @intFromEnum(types.GcTypeTag.callframe_array));
}

test "M0 opcode char mapping round-trips" {
    // Every dense opcode maps back to its own character.
    const ops = [_]types.Opcode{
        .access, .global, .jmpf,   .jmp,    .appterm, .apply,
        .pushmark, .cur, .grab, .ret,    .let,     .endlet,
        .number, .string, .symbol, .boolean, .prim,
    };
    for (ops) |op| {
        const c = types.opcodeToChar(op);
        try std.testing.expectEqual(op, types.charToOpcode(c));
    }
    // Unknown char -> sentinel, and sentinel -> '?'.
    try std.testing.expectEqual(types.Opcode.count, types.charToOpcode('Z'));
    try std.testing.expectEqual(@as(u8, '?'), types.opcodeToChar(.count));
    // Spot-check a couple of C values.
    try std.testing.expectEqual(types.Opcode.access, types.charToOpcode('a'));
    try std.testing.expectEqual(types.Opcode.string, types.charToOpcode('S'));
    try std.testing.expectEqual(types.Opcode.prim, types.charToOpcode('P'));
}

test "M0 TableEntry is word-multiple and word-aligned" {
    try std.testing.expect(@sizeOf(types.TableEntry) % @sizeOf(usize) == 0);
    try std.testing.expect(@alignOf(types.TableEntry) >= @sizeOf(usize));
}

test "M0 words>0xFFFFFF is fatal (documented panic)" {
    // assertWordsFits panics above the limit; the limit itself must not.
    types.assertWordsFits(0xFFFFFF, 0xFFFFFF * @sizeOf(usize));
}

// =====================================================================
//  M1 — heap core (plan DECISION 7: T1-ext + straddle / multi-page
//  asserts).  Collection triggers (M3/M4) are NOT exercised here: the
//  collect.zig stubs panic loudly if any test reaches one, so a green
//  suite also proves the trigger thresholds were not accidentally
//  crossed by small allocations.
// =====================================================================

const heap_mod = gc.heap;

/// 16 MB heap (the C minimum, MIN_HEAP_PAGES) with an explicit 64 MB
/// reservation (plan DECISION 4: avoids the 4 GB default VAS).
fn testInit() !heap_mod.Gc {
    return heap_mod.Gc.init(.{
        .heap_bytes = 16 * 1024 * 1024,
        .reserve_bytes = 64 * 1024 * 1024,
    });
}

/// Read the header word immediately before a body pointer.
fn headerOf(body: anytype) usize {
    return @as(*const usize, @ptrFromInt(@intFromPtr(body) - @sizeOf(usize))).*;
}

test "M1 init carves nursery, zeroes metadata, honours reserve" {
    var g = try testInit();
    defer g.deinit();

    // Extent from heap_bytes (C: gc.c:2098-2100).
    try std.testing.expectEqual(@as(usize, 32768), g.heappages);
    try std.testing.expectEqual(g.firstheappage + g.heappages - 1, g.lastheappage);

    // Nursery at the heap start (C: gc.c:2136-2137).
    try std.testing.expectEqual(g.firstheappage, g.nursery_first);
    try std.testing.expectEqual(g.nursery_first + heap_mod.NURSERY_PAGES - 1, g.nursery_last);

    // Bump cursor at region start; end one past the last byte (C: gc.c:2145-2146).
    try std.testing.expectEqual(@intFromPtr(heap_mod.pageToGcp(g.nursery_first)), g.nursery_cur);
    try std.testing.expectEqual(@intFromPtr(heap_mod.pageToGcp(g.nursery_last + 1)), g.nursery_end);

    // Every nursery page tagged NURSERY; the first old-gen page free.
    var pg = g.nursery_first;
    while (pg <= g.nursery_last) : (pg += 1)
        try std.testing.expectEqual(heap_mod.NURSERY, g.space[g.md(pg)]);
    try std.testing.expectEqual(@as(usize, 0), g.space[g.md(g.nursery_last + 1)]);

    // Semi-space state + scan start (C: gc.c:2148-2152).
    try std.testing.expectEqual(@as(usize, 1), g.current_space);
    try std.testing.expectEqual(@as(usize, 1), g.next_space);
    try std.testing.expectEqual(g.nursery_last + 1, g.freepage);
    try std.testing.expectEqual(@as(usize, 0), g.allocatedpages);

    // Predicates (C: gc.c:2306-2324).
    try std.testing.expect(g.nurseryIsEmpty());
    try std.testing.expect(g.nurseryNoOtherSpace());
    try std.testing.expectEqual(heap_mod.NURSERY_PAGES, g.nurseryCapacityPages());
    try std.testing.expectEqual(@as(usize, 0), g.allocatedPages());

    // Reservation honoured exactly (C: gc.c:2068-2070 formula bypassed by
    // the explicit opt).
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), g.heap_mmap_size);
}

test "M1 init rejects bad heap sizes" {
    // Not a multiple of PAGEBYTES (C doc contract, gc.h:32-34).
    try std.testing.expectError(error.InvalidHeapSize, heap_mod.Gc.init(.{ .heap_bytes = 1000 }));
    // Below the 16 MB minimum (C: gc.c:1799 MIN_HEAP_PAGES).
    try std.testing.expectError(error.InvalidHeapSize, heap_mod.Gc.init(.{ .heap_bytes = 1 * 1024 * 1024 }));
    try std.testing.expectError(error.InvalidHeapSize, heap_mod.Gc.init(.{ .heap_bytes = 0 }));
}

test "M1 typed allocs of every class: zeroed bodies, headers, metadata" {
    var g = try testInit();
    defer g.deinit();

    // -- single Value: 40 B body -> 6 words (48 B) -> nursery fast path.
    const v = g.alloc(types.Value);
    try std.testing.expect(g.inNursery(@intFromPtr(v)));
    try std.testing.expect(!g.inOldgen(@intFromPtr(v))); // space==NURSERY, not current
    // Body fully zeroed (C: gc.c:2217): tag 0 = number, payload 0.
    try std.testing.expectEqual(types.ValTag.number, v.tag);
    try std.testing.expectEqual(@as(i64, 0), v.payload.number);
    const vhdr = headerOf(v);
    try std.testing.expectEqual(@as(usize, 6), types.headerWords(vhdr));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.value)), types.headerType(vhdr));
    try std.testing.expect(!types.forwarded(vhdr));
    // Page/type metadata: nursery pages keep the NURSERY space tag (never
    // re-tagged by the bump allocator), type_page OBJECT (C: gc.c:2235-2240).
    const vpg = heap_mod.gcpToPage(@intFromPtr(v));
    try std.testing.expectEqual(heap_mod.NURSERY, g.space[g.md(vpg)]);
    try std.testing.expectEqual(heap_mod.OBJECT, g.type_page[g.md(vpg)]);

    // -- Value[10]: 400 B body -> 51 words (408 B) -> nursery.
    const arr = g.allocArray(types.Value, 10);
    try std.testing.expect(g.inNursery(@intFromPtr(arr)));
    const ahdr = headerOf(arr);
    try std.testing.expectEqual(@as(usize, 51), types.headerWords(ahdr));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.value_array)), types.headerType(ahdr));
    for (arr[0..10]) |*e| {
        try std.testing.expectEqual(types.ValTag.number, e.tag);
        try std.testing.expectEqual(@as(i64, 0), e.payload.number);
    }

    // -- Instr[3]: 192 B body -> 25 words (200 B) -> nursery.
    const ins = g.allocArray(types.Instr, 3);
    try std.testing.expect(g.inNursery(@intFromPtr(ins)));
    const ihdr = headerOf(ins);
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.instr_array)), types.headerType(ihdr));
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(ins[0].op));
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(ins[2].op));
    try std.testing.expectEqual(types.ValTag.number, ins[1].operand.tag);
    try std.testing.expect(ins[1].closure_code == null);

    // -- CallFrame[2]: 96 B body -> 13 words (104 B) -> nursery.
    const cf = g.allocArray(types.CallFrame, 2);
    try std.testing.expect(g.inNursery(@intFromPtr(cf)));
    const chdr = headerOf(cf);
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.callframe_array)), types.headerType(chdr));
    try std.testing.expect(cf[0].code == null);
    try std.testing.expect(cf[0].env == null);
    try std.testing.expect(cf[0].stack.data == null);
    try std.testing.expectEqual(@as(i32, 0), cf[1].stack.cap);

    // -- raw + atomic alias: zeroed bytes.
    const raw = g.allocRaw(100);
    for (raw[0..100]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    const atom = g.allocAtomic(64);
    try std.testing.expect(g.inNursery(@intFromPtr(atom)));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.raw)), types.headerType(headerOf(atom)));

    // -- explicit old-gen: single page + multi-page array.
    const og = g.allocOldgen(256, .raw);
    try std.testing.expect(!g.inNursery(@intFromPtr(og)));
    try std.testing.expect(g.inOldgen(@intFromPtr(og)));
    for (og[0..256]) |b| try std.testing.expectEqual(@as(u8, 0), b);

    const ogv = g.allocArrayOldgen(types.Value, 64); // 2560 B -> 321 words -> 6 pages
    try std.testing.expect(g.inOldgen(@intFromPtr(ogv)));
    try std.testing.expect(!g.inNursery(@intFromPtr(ogv)));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.value_array)), types.headerType(headerOf(ogv)));

    // -- per-class histogram (C: gc.c:2160/2286).
    try std.testing.expectEqualSlices(u64, &[_]u64{ 3, 1, 2, 1, 1 }, &g.alloc_class_count);

    // No collection trigger was reached by any of the above.
    try std.testing.expectEqual(@as(u64, 0), g.nursery_scavenge_count);
    try std.testing.expectEqual(@as(u64, 0), g.full_collect_count);
}

test "M1 nursery no-straddle guard" {
    var g = try testInit();
    defer g.deinit();

    // Values are 6 words (48 B); ~10 fit per 512 B page. Allocate enough
    // to cross several page boundaries and assert none straddles.
    var last_end: usize = 0;
    var n: usize = 0;
    while (n < 300) : (n += 1) {
        const v = g.alloc(types.Value);
        const start = @intFromPtr(v) - 8;
        const end = start + 48 - 1;

        // The whole object (header + body) stays within one page.
        try std.testing.expectEqual(start / 512, end / 512);
        try std.testing.expect(g.inNursery(@intFromPtr(v)));

        if (last_end != 0) {
            // Monotonic bump cursor...
            try std.testing.expect(start >= last_end);
            // ...and skips ONLY at page boundaries (the straddle guard).
            if (start != last_end) {
                try std.testing.expectEqual(@as(usize, 0), start % 512);
                try std.testing.expect(start / 512 > (last_end - 1) / 512);
            }
        }
        last_end = end + 1;
    }

    // 300 * 48 B = 14.4 KB << nursery/8 low-water: nothing triggered.
    try std.testing.expectEqual(@as(u64, 0), g.nursery_scavenge_count);
    try std.testing.expectEqual(@as(u64, 0), g.preemptive_scavenge_count);
    try std.testing.expectEqual(@as(u64, 0), g.reactive_scavenge_count);
}

test "M1 multi-page old-gen alloc: metadata, freep advance, fresh page next" {
    var g = try testInit();
    defer g.deinit();

    const pages_before = g.allocatedPages();
    const big = g.allocArray(types.Instr, 64); // 4096 B body -> 513 words -> 9 pages
    const start = @intFromPtr(big) - 8;

    try std.testing.expect(!g.inNursery(start));
    try std.testing.expect(g.inOldgen(start));

    // Multi-page objects start page-aligned (allocatepage gives fresh pages).
    try std.testing.expectEqual(@as(usize, 0), start % 512);
    const hdr = @as(*const usize, @ptrFromInt(start)).*;
    try std.testing.expectEqual(@as(usize, 513), types.headerWords(hdr));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.instr_array)), types.headerType(hdr));

    // All 9 pages tagged current_space; first OBJECT, rest CONTINUED.
    const first_pg = start / 512;
    const last_pg = (start + 513 * 8 - 1) / 512;
    try std.testing.expectEqual(@as(usize, 9), last_pg - first_pg + 1);
    var pg = first_pg;
    while (pg <= last_pg) : (pg += 1) {
        try std.testing.expectEqual(g.current_space, g.space[g.md(pg)]);
        if (pg == first_pg) {
            try std.testing.expectEqual(heap_mod.OBJECT, g.type_page[g.md(pg)]);
        } else {
            try std.testing.expectEqual(heap_mod.CONTINUED, g.type_page[g.md(pg)]);
        }
    }
    try std.testing.expect(g.allocatedPages() >= pages_before + 9);

    // The multi-page freep-advance subtlety (C: gc.c:1781-1791): freep is
    // past the object and freewords == 0, so the next alloc takes fresh
    // pages and the drain's cp != freep guard can still scan page 1.
    try std.testing.expectEqual(@as(usize, 0), g.freewords);
    try std.testing.expect(@intFromPtr(g.freep) >= start + 513 * 8);

    // Body zeroed.
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(big[63].op));
    try std.testing.expectEqual(types.ValTag.number, big[63].operand.tag);

    // A follow-on small old-gen alloc lands on a FRESH page.
    const og = g.allocOldgen(64, .raw);
    try std.testing.expect(g.inOldgen(@intFromPtr(og)));
    const og_pg = (@intFromPtr(og) - 8) / 512;
    try std.testing.expect(og_pg < first_pg or og_pg > last_pg);
}

test "M1 partial-page finalize writes filler header" {
    var g = try testInit();
    defer g.deinit();

    const og = g.allocOldgen(64, .raw); // 9 words; leaves a 55-word slack
    try std.testing.expectEqual(@as(usize, 55), g.freewords);

    // The bump cursor sits right after the 64-byte body.
    const cursor = @intFromPtr(og) + 64;
    _ = g.allocArray(types.Instr, 64); // too big for the slack: filler + 9 pages

    const filler = @as(*const usize, @ptrFromInt(cursor)).*;
    try std.testing.expectEqual(@as(usize, 55), types.headerWords(filler));
    try std.testing.expectEqual(@as(u32, @intFromEnum(types.GcTypeTag.raw)), types.headerType(filler));
    try std.testing.expect(!types.forwarded(filler));
}

test "M1 nursery eligibility boundary at exactly one page" {
    var g = try testInit();
    defer g.deinit();

    // 504 body bytes -> 64 words total == PAGEBYTES: single-page, nursery.
    const small = g.allocRaw(504);
    try std.testing.expect(g.inNursery(@intFromPtr(small)));
    try std.testing.expectEqual(@as(usize, 64), types.headerWords(headerOf(small)));

    // 505 body bytes -> 65 words == 520 B > PAGEBYTES: old-gen despite
    // bytes <= NURSERY_BYTES/8 (C: gc.c:2171-2172 — the second condition).
    const big = g.allocRaw(505);
    try std.testing.expect(!g.inNursery(@intFromPtr(big)));
    try std.testing.expect(g.inOldgen(@intFromPtr(big)));
    try std.testing.expectEqual(@as(usize, 65), types.headerWords(headerOf(big)));
}

test "M1 grow_heap doubles, honours min_needed, respects reservation" {
    var g = try testInit();
    defer g.deinit();

    try std.testing.expectEqual(@as(usize, 32768), g.heappages);

    // Double: 65536 pages = 32 MB <= 64 MB reservation.
    try std.testing.expect(g.grow_heap(1));
    try std.testing.expectEqual(@as(usize, 65536), g.heappages);
    try std.testing.expectEqual(g.firstheappage + g.heappages - 1, g.lastheappage);

    // New tail pages free and metadata zeroed; nursery tags preserved.
    var pg = g.lastheappage - 8;
    while (pg <= g.lastheappage) : (pg += 1) {
        try std.testing.expectEqual(@as(usize, 0), g.space[g.md(pg)]);
        try std.testing.expectEqual(@as(usize, 0), g.gc_link[g.md(pg)]);
        try std.testing.expectEqual(@as(usize, 0), g.type_page[g.md(pg)]);
        try std.testing.expectEqual(@as(u8, 0), g.page_queued[g.md(pg)]);
    }
    try std.testing.expect(g.nurseryIsEmpty());

    // Next doubling needs 64 MB + 511 > 64 MB reservation -> fails (C: -1).
    try std.testing.expect(!g.grow_heap(1));

    // min_needed dominates the doubling when the demand is large
    // (C: gc.c:1811-1815): (0 + 40000 + 512) * 2 = 81024 pages.
    var g2 = try testInit();
    defer g2.deinit();
    try std.testing.expect(g2.grow_heap(40000));
    try std.testing.expectEqual(@as(usize, 81024), g2.heappages);
}

test "M1 cheney queue dedup, links, reset" {
    var g = try testInit();
    defer g.deinit();

    const p1 = g.firstheappage + 5000;
    const p2 = g.firstheappage + 6000;
    g.queue(p1);
    g.queue(p2);
    g.queue(p1); // DOUBLE-QUEUE MUST BE A NO-OP (would clobber gc_link[p1]).

    try std.testing.expectEqual(p1, g.queue_head);
    try std.testing.expectEqual(p2, g.queue_tail);
    try std.testing.expectEqual(p2, g.gc_link[g.md(p1)]); // p1 -> p2 preserved
    try std.testing.expectEqual(@as(usize, 0), g.gc_link[g.md(p2)]); // tail terminates
    try std.testing.expectEqual(@as(u8, 1), g.page_queued[g.md(p1)]);
    try std.testing.expectEqual(@as(u8, 1), g.page_queued[g.md(p2)]);

    // next_page wraps at lastheappage (C: gc.c:435-437).
    try std.testing.expectEqual(g.firstheappage, g.next_page(g.lastheappage));
    try std.testing.expectEqual(p1 + 1, g.next_page(p1));

    // Reset clears head/tail and the dedup bits.
    g.queue_reset();
    try std.testing.expectEqual(@as(usize, 0), g.queue_head);
    try std.testing.expectEqual(@as(usize, 0), g.queue_tail);
    try std.testing.expectEqual(@as(u8, 0), g.page_queued[g.md(p1)]);
    try std.testing.expectEqual(@as(u8, 0), g.page_queued[g.md(p2)]);

    // After reset the page can legitimately be re-queued.
    g.queue(p1);
    try std.testing.expectEqual(p1, g.queue_head);
    try std.testing.expectEqual(p1, g.queue_tail);
}

test "M1 dirty vectors: dedup, cap + overflow valve, clear" {
    var g = try testInit();
    defer g.deinit();

    const a = g.allocArray(types.Value, 2);
    g.dirtyVectorsAdd(a);
    g.dirtyVectorsAdd(a); // dedup
    try std.testing.expectEqual(@as(usize, 1), g.dirty_vectors_count);
    try std.testing.expectEqual(@as(u64, 1), g.dirty_vectors_fired);
    try std.testing.expect(!g.dirty_vectors_overflow);

    // Fill to the cap with distinct arrays (88 B each; 720 KB total —
    // below the 1.75 MB preemptive low-water, so no scavenge fires).
    while (g.dirty_vectors_count < heap_mod.DIRTY_VECTORS_MAX) {
        const arr = g.allocArray(types.Value, 2);
        g.dirtyVectorsAdd(arr);
    }
    try std.testing.expectEqual(heap_mod.DIRTY_VECTORS_MAX, g.dirty_vectors_count);
    try std.testing.expectEqual(heap_mod.DIRTY_VECTORS_MAX, g.dirty_vectors_cap);
    try std.testing.expect(!g.dirty_vectors_overflow);

    // One more DISTINCT array trips the valve; fired stops counting.
    const extra = g.allocArray(types.Value, 2);
    g.dirtyVectorsAdd(extra);
    try std.testing.expect(g.dirty_vectors_overflow);
    try std.testing.expectEqual(heap_mod.DIRTY_VECTORS_MAX, g.dirty_vectors_count);
    try std.testing.expectEqual(@as(u64, heap_mod.DIRTY_VECTORS_MAX), g.dirty_vectors_fired);

    // While overflow is set, adds early-return.
    g.dirtyVectorsAdd(a);
    try std.testing.expectEqual(heap_mod.DIRTY_VECTORS_MAX, g.dirty_vectors_count);

    // Clear resets the count AND the valve (C: gc.c:376-379)...
    g.dirtyVectorsClear();
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);
    try std.testing.expect(!g.dirty_vectors_overflow);

    // ...so recording works again.
    g.dirtyVectorsAdd(a);
    try std.testing.expectEqual(@as(u64, heap_mod.DIRTY_VECTORS_MAX + 1), g.dirty_vectors_fired);
}

test "M1 dirty defuns bitset: mark/test/clear + counters" {
    var g = try testInit();
    defer g.deinit();

    g.dirtyDefunsMark(0);
    g.dirtyDefunsMark(64);
    g.dirtyDefunsMark(4095);
    g.dirtyDefunsMark(64); // re-mark must not double-count

    try std.testing.expectEqual(@as(u64, 3), g.dirty_defuns_fired);
    try std.testing.expect(g.dirtyDefunsTest(0));
    try std.testing.expect(!g.dirtyDefunsTest(1));
    try std.testing.expect(g.dirtyDefunsTest(64));
    try std.testing.expect(g.dirtyDefunsTest(4095));

    // Out-of-range indices ignored (C parity, gc.c:405/415).
    try std.testing.expect(!g.dirtyDefunsTest(-1));
    try std.testing.expect(!g.dirtyDefunsTest(4096));
    g.dirtyDefunsMark(-1);
    g.dirtyDefunsMark(4096);
    try std.testing.expectEqual(@as(u64, 3), g.dirty_defuns_fired);

    g.dirtyDefunsClear();
    try std.testing.expect(!g.dirtyDefunsTest(0));
    try std.testing.expect(!g.dirtyDefunsTest(64));
    try std.testing.expect(!g.dirtyDefunsTest(4095));
}

test "M1 stats snapshot" {
    var g = try testInit();
    defer g.deinit();

    _ = g.alloc(types.Value);
    _ = g.allocRaw(32);
    const s = g.stats();

    try std.testing.expectEqual(@as(u64, 0), s.nursery_scavenge_count);
    try std.testing.expectEqual(@as(u64, 0), s.full_collect_count);
    try std.testing.expectEqual(@as(usize, 0), s.allocated_pages);
    try std.testing.expect(s.nursery_is_empty);
    try std.testing.expect(s.nursery_no_other_space);
    try std.testing.expectEqual(heap_mod.NURSERY_PAGES, s.nursery_capacity_pages);
    try std.testing.expectEqual(@as(u64, 1), s.alloc_class_count[@intFromEnum(types.GcTypeTag.value)]);
    try std.testing.expectEqual(@as(u64, 1), s.alloc_class_count[@intFromEnum(types.GcTypeTag.raw)]);
    try std.testing.expectEqual(@as(u64, 0), s.dirty_vectors_fired);
}

// ---- M2 compile-reference (plan DECISION 8: functions unreachable until M3) ----
// Zig lazily analyzes; referencing the M2 functions forces their bodies to be
// type-checked.  The M3 collector (collect.zig) now drives them; T2-T5 below
// exercise the movement core at runtime.
test "M2 move+scan modules compile clean" {
    const collect_mod = gc.collect;
    const scan_mod = gc.scan;
    // Referencing each public M2 function value forces its body to be
    // type-checked (Zig lazy analysis).
    _ = &collect_mod.moveInternal;
    _ = &collect_mod.gcMove;
    _ = &collect_mod.drainScanObject;
    _ = &scan_mod.evacuate;
    _ = &scan_mod.scanValue;
    _ = &scan_mod.evacInstr;
    _ = &scan_mod.valueReferencesNursery;
}

// =====================================================================
//  M3 — roots + full collect (plan DECISION 7 / DECISION 8): root API
//  smoke, typed-walker registrations, T2 evacuate+move, T3 forwarding
//  aliasing, T4 full collect, T5 deep graph + old-gen cycle.
//
//  All collection is driven via gc.collect(.@"test") — the full collect.
//  Its Phase-0 (gc.c:654-694) promotes nursery survivors to old-gen, so
//  T2's "evacuate+move" assertions hold through the full-collect path;
//  the nursery-scavenge variant (scavenge counters, nursery cursor rewind)
//  is M4's collectNursery and is NOT exercised here (the stub panics).
// =====================================================================

test "M3 roots API: shadow stack push/pop/watermark + registrations" {
    var g = try testInit();
    defer g.deinit();

    try std.testing.expectEqual(@as(usize, 0), g.rootWatermark());

    // Push/pop discipline — C: gc.c:2340-2382.
    var v1: types.Value = .{ .tag = .nil, .payload = .{ .number = 0 } };
    g.rootPushValue(&v1);
    try std.testing.expectEqual(@as(usize, 1), g.rootWatermark());

    var arr = [_]types.Value{
        .{ .tag = .nil, .payload = .{ .number = 0 } },
        .{ .tag = .nil, .payload = .{ .number = 0 } },
    };
    var n: i32 = 2;
    g.rootPushValueArray(&arr, &n);
    try std.testing.expectEqual(@as(usize, 2), g.rootWatermark());

    g.rootPop();
    try std.testing.expectEqual(@as(usize, 1), g.rootWatermark());

    // Grow past the initial 64-entry capacity (shadowStackGrow, gc.c:2333).
    var i: usize = 0;
    while (i < 200) : (i += 1) g.rootPushValue(&v1);
    try std.testing.expectEqual(@as(usize, 201), g.rootWatermark());
    try std.testing.expect(g.shadow_cap >= 201);
    g.rootPopTo(1);
    try std.testing.expectEqual(@as(usize, 1), g.rootWatermark());
    g.rootPopTo(0);
    try std.testing.expectEqual(@as(usize, 0), g.rootWatermark());

    // Registrations store the C-heap pointers verbatim (gc.c:2384-2397).
    var defun_tbl = [_]types.TableEntry{
        .{ .name = @ptrCast(@constCast("f")), .value = .{ .tag = .nil, .payload = .{ .number = 0 } } },
    };
    var defun_len: i32 = 1;
    g.registerGlobalTable(&defun_tbl, &defun_len);
    try std.testing.expect(g.reg_global_table != null);
    try std.testing.expect(g.reg_global_table_len != null);

    var traced = [_]?*types.Instr{null};
    var traced_len: i32 = 1;
    g.registerTracedCode(&traced, &traced_len);
    try std.testing.expect(g.reg_traced_code != null);

    // No collection was triggered by root bookkeeping.
    try std.testing.expectEqual(@as(u64, 0), g.full_collect_count);
}

test "M3 typed walkers: defun/values tables + traced_code entries rewritten" {
    var g = try testInit();
    defer g.deinit();

    // The tables live OUTSIDE the GC heap (C-heap in C; test-local in the
    // port) — registration never moves them, only their interior pointers.
    var defun_tbl = [_]types.TableEntry{
        .{ .name = @ptrCast(@constCast("defn0")), .value = .{ .tag = .nil, .payload = .{ .number = 0 } } },
        .{ .name = null, .value = .{ .tag = .nil, .payload = .{ .number = 0 } } }, // skipped slot
    };
    var defun_len: i32 = 2;
    g.registerGlobalTable(&defun_tbl, &defun_len);

    var values_tbl = [_]types.TableEntry{
        .{ .name = @ptrCast(@constCast("val0")), .value = .{ .tag = .nil, .payload = .{ .number = 0 } } },
    };
    var values_len: i32 = 1;
    g.registerValuesTable(&values_tbl, &values_len);

    var traced = [_]?*types.Instr{null};
    var traced_len: i32 = 1;
    g.registerTracedCode(&traced, &traced_len);

    // Nursery closure stored in the (NON-dirty) defun slot: Phase-0's explicit
    // non-dirty defun scan (gc.c:659-670) must promote it even though the
    // dirty bitset is completely empty.
    const cl = g.alloc(types.Value);
    cl.* = .{ .tag = .number, .payload = .{ .number = 7 } };
    defun_tbl[0].value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = cl, .cdr = null } } };

    // Nursery value stored in the values table (always full-scanned).
    const nv = g.alloc(types.Value);
    nv.* = .{ .tag = .number, .payload = .{ .number = 9 } };
    values_tbl[0].value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = nv, .cdr = null } } };

    // Nursery Instr array stored in traced_code.
    const code = g.allocArray(types.Instr, 3);
    code[0].op = .cur;
    traced[0] = @ptrCast(code);
    const code_old = @intFromPtr(code);

    g.collect(.@"test");

    // Defun slot's interior pointer rewritten out of the nursery.
    const cl_new = defun_tbl[0].value.payload.cons.car.?;
    try std.testing.expect(!g.inNursery(@intFromPtr(cl_new)));
    try std.testing.expect(g.inOldgen(@intFromPtr(cl_new)));
    try std.testing.expectEqual(@as(i64, 7), cl_new.payload.number);

    // Values table entry rewritten (the nil slot in defun_tbl skipped).
    const nv_new = values_tbl[0].value.payload.cons.car.?;
    try std.testing.expect(!g.inNursery(@intFromPtr(nv_new)));
    try std.testing.expectEqual(@as(i64, 9), nv_new.payload.number);

    // traced_code slot rewritten IN PLACE to the evacuated array head
    // (evacuate writes through &tc[k], gc.c:1604-1610).
    const code_new = traced[0].?;
    try std.testing.expect(!g.inNursery(@intFromPtr(code_new)));
    try std.testing.expect(@intFromPtr(code_new) != code_old);
    try std.testing.expectEqual(types.Opcode.cur, code_new.op);

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
}

test "M3 T2 rooted nursery cons graph survives full collect" {
    var g = try testInit();
    defer g.deinit();

    // c2 -> 2, c1 -> c2, root -> c1 — all nursery except the root Value
    // itself (a plain local; ROOT_VALUE rewrites its interior pointers).
    const c2 = g.alloc(types.Value);
    c2.* = .{ .tag = .number, .payload = .{ .number = 2 } };
    const c1 = g.alloc(types.Value);
    c1.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = c2, .cdr = null } } };

    var root: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = c1, .cdr = null } } };
    const wm0 = g.rootWatermark();
    g.rootPushValue(&root);

    const c1_old = @intFromPtr(c1);
    const c2_old = @intFromPtr(c2);
    try std.testing.expect(g.inNursery(c1_old) and g.inNursery(c2_old));

    g.collect(.@"test"); // full collect: Phase-0 promotion + semi-space flip

    try std.testing.expectEqual(@as(u64, 1), g.full_collect_count);

    // Survivors intact AND moved out of the nursery (Phase-0 promoted them;
    // the main scavenge evacuated them into the new to-space).
    const c1_new = root.payload.cons.car.?;
    const c2_new = c1_new.payload.cons.car.?;
    try std.testing.expect(@intFromPtr(c1_new) != c1_old);
    try std.testing.expect(@intFromPtr(c2_new) != c2_old);
    try std.testing.expect(!g.inNursery(@intFromPtr(c1_new)));
    try std.testing.expect(!g.inNursery(@intFromPtr(c2_new)));
    try std.testing.expect(g.inOldgen(@intFromPtr(c1_new)));
    try std.testing.expect(g.inOldgen(@intFromPtr(c2_new)));

    try std.testing.expectEqual(types.ValTag.cons, c1_new.tag);
    try std.testing.expect(c1_new.payload.cons.cdr == null);
    try std.testing.expectEqual(types.ValTag.number, c2_new.tag);
    try std.testing.expectEqual(@as(i64, 2), c2_new.payload.number);

    // Nursery page TAGS still read NURSERY (a full collect does not reset
    // the nursery — that is collectNursery/M4); the tag-based predicate
    // therefore reports "empty" while the bump cursor stays put.
    try std.testing.expect(g.nurseryIsEmpty());

    // Garbage reclaimed: the live set is the two promoted Values (a page or
    // two), not the from-space + Phase-0 intermediates.
    try std.testing.expect(g.allocatedPages() <= 4);

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
    g.rootPopTo(wm0);
}

test "M3 T3 forwarding aliasing: two roots to one object converge" {
    var g = try testInit();
    defer g.deinit();

    const obj = g.alloc(types.Value);
    obj.* = .{ .tag = .number, .payload = .{ .number = 42 } };

    // Two rooted Values both referencing the SAME nursery object.
    var r1: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = obj, .cdr = null } } };
    var r2: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = obj, .cdr = null } } };
    const wm0 = g.rootWatermark();
    g.rootPushValue(&r1);
    g.rootPushValue(&r2);

    const obj_old = @intFromPtr(obj);
    g.collect(.@"test");

    // Both roots' car must hold the SAME new address: the first evacuation
    // moves the object and leaves a forwarding pointer in the old header; the
    // second gc_move short-circuits on FORWARDED (gc.c:2037/2042/2049).
    const new1 = r1.payload.cons.car.?;
    const new2 = r2.payload.cons.car.?;
    try std.testing.expectEqual(@intFromPtr(new1), @intFromPtr(new2));
    try std.testing.expect(@intFromPtr(new1) != obj_old);
    try std.testing.expectEqual(@as(i64, 42), new1.payload.number);
    try std.testing.expect(g.inOldgen(@intFromPtr(new1)));

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
    g.rootPopTo(wm0);
}

test "M3 T4 full collect: old-gen + nursery mix, dead space released" {
    var g = try testInit();
    defer g.deinit();

    // Old-gen multi-page Value array (bypasses the nursery).
    const og = g.allocArrayOldgen(types.Value, 64); // 2560 B body -> 6 pages
    og[0] = .{ .tag = .number, .payload = .{ .number = 100 } };
    og[63] = .{ .tag = .number, .payload = .{ .number = 163 } };

    // Nursery Value whose car points INTO the old-gen array (the
    // nursery-resident-references-old-gen case Phase-0 exists for).
    const root_v = g.alloc(types.Value);
    root_v.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = &og[0], .cdr = null } } };
    var root: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = root_v, .cdr = null } } };
    const wm0 = g.rootWatermark();
    g.rootPushValue(&root);

    const og_old = @intFromPtr(&og[0]);
    const root_v_old = @intFromPtr(root_v);

    g.collect(.@"test");

    try std.testing.expectEqual(@as(u64, 1), g.full_collect_count);

    // The nursery link was promoted out of the nursery.
    const root_v_new = root.payload.cons.car.?;
    try std.testing.expect(@intFromPtr(root_v_new) != root_v_old);
    try std.testing.expect(!g.inNursery(@intFromPtr(root_v_new)));

    // The old-gen array was evacuated to the new semi-space — the array
    // moved, contents intact (whole 321-word object copied by moveInternal).
    const og_new = root_v_new.payload.cons.car.?;
    try std.testing.expect(@intFromPtr(og_new) != og_old);
    try std.testing.expect(g.inOldgen(@intFromPtr(og_new)));
    try std.testing.expectEqual(@as(i64, 100), og_new.payload.number);
    const new_arr: [*]types.Value = @ptrCast(@alignCast(og_new));
    try std.testing.expectEqual(@as(i64, 163), new_arr[63].payload.number);

    // Dead from-space released (gc.c:722-727): every page is now live
    // (current_space), nursery, or free (0) — none carries the dead tag.
    var pg = g.firstheappage;
    while (pg <= g.lastheappage) : (pg += 1) {
        const sp = g.space[g.md(pg)];
        if (sp == g.current_space) continue;
        if (sp == heap_mod.NURSERY) continue;
        try std.testing.expectEqual(@as(usize, 0), sp);
    }

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
    g.rootPopTo(wm0);
}

test "M3 repeated collects flip semi-spaces back and forth" {
    var g = try testInit();
    defer g.deinit();

    var root: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = null, .cdr = null } } };
    const wm0 = g.rootWatermark();
    g.rootPushValue(&root);

    const obj = g.alloc(types.Value);
    obj.* = .{ .tag = .number, .payload = .{ .number = 99 } };
    root.payload.cons.car = obj;

    const start_space = g.current_space;
    g.collect(.@"test");
    try std.testing.expectEqual(@as(u64, 1), g.full_collect_count);
    try std.testing.expect(g.current_space != start_space); // flipped

    const after1 = root.payload.cons.car.?;
    try std.testing.expectEqual(@as(i64, 99), after1.payload.number);

    // The second collect must finalize the partial page left by the first
    // drain (gc.c:629-633), re-flip, and re-evacuate the survivor.
    g.collect(.@"test");
    try std.testing.expectEqual(@as(u64, 2), g.full_collect_count);
    try std.testing.expectEqual(start_space, g.current_space); // flipped back

    const after2 = root.payload.cons.car.?;
    try std.testing.expectEqual(@as(i64, 99), after2.payload.number);
    try std.testing.expect(g.inOldgen(@intFromPtr(after2)));

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
    g.rootPopTo(wm0);
}

test "M3 wiring: THRESHOLD + LASTRESORT trigger sites drive the real collect" {
    var g = try testInit();
    defer g.deinit();

    // THRESHOLD (gc_alloc_oldgen, C: gc.c:2287-2291): fill old-gen past
    // heappages/4 = 8192 pages with UNROOTED garbage, then the next old-gen
    // alloc fires collect(.alloc) and reclaims all of it.
    const big_bytes = 8300 * heap_mod.PAGEBYTES - 16; // ~8301 pages
    _ = g.allocOldgen(big_bytes, .raw);
    try std.testing.expect(g.allocatedPages() >= 8300);

    const small = g.allocOldgen(64, .raw); // fires the threshold check
    try std.testing.expect(g.full_collect_count >= 1);
    try std.testing.expect(g.allocatedPages() < 100);
    try std.testing.expect(g.inOldgen(@intFromPtr(small)));

    // LASTRESORT (allocatepage, C: gc.c:1867-1885): an allocation needing
    // more than half the heap forces collect(.lastresort) + grow + retry
    // (nothing is rooted, so the collect empties the heap and the retry
    // succeeds on the grown heap).
    const huge_bytes = 16500 * heap_mod.PAGEBYTES - 16;
    const huge = g.allocOldgen(huge_bytes, .raw);
    try std.testing.expect(g.full_collect_count >= 2);
    try std.testing.expect(g.heappages > 32768); // grow_heap fired
    try std.testing.expect(g.inOldgen(@intFromPtr(huge)));

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
}

test "M3 T5 deep graph + old-gen cycle terminates" {
    var g = try testInit();
    defer g.deinit();

    // Old-gen cycle: a -> b -> a (two cons cells in old-gen).  The drain must
    // terminate despite the cycle: forwarding pointers short-circuit
    // re-visits and page_queued dedup blocks re-queueing.
    const a = g.allocOldgen(@sizeOf(types.Value), .value);
    const a_v: *types.Value = @ptrCast(@alignCast(a));
    const b = g.allocOldgen(@sizeOf(types.Value), .value);
    const b_v: *types.Value = @ptrCast(@alignCast(b));
    a_v.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = null, .cdr = b_v } } };
    b_v.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = null, .cdr = a_v } } };

    // 500-node cons chain in the nursery, head's far end reaching into the
    // old-gen cycle (nursery-references-old-gen again).
    var head: *types.Value = g.alloc(types.Value);
    const wm0 = g.rootWatermark();
    g.rootPushPtr(@ptrCast(&head));
    head.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = a_v, .cdr = null } } };

    var i: usize = 1;
    while (i < 500) : (i += 1) {
        var node: *types.Value = g.alloc(types.Value);
        g.rootPushPtr(@ptrCast(&node));
        node.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = null, .cdr = head } } };
        head = node;
        g.rootPop();
    }

    // Completing the collect at all proves the drain terminates over the
    // 500-node graph + cycle (cycle safety).
    g.collect(.@"test");
    try std.testing.expectEqual(@as(u64, 1), g.full_collect_count);

    // The whole chain survived: walk it, count nodes, land on the cycle.
    try std.testing.expect(g.inOldgen(@intFromPtr(head)));
    var count: usize = 0;
    var cur: ?*types.Value = head;
    var tail_car: ?*types.Value = null;
    while (cur) |c| : (cur = c.payload.cons.cdr) {
        count += 1;
        if (c.payload.cons.car) |car| tail_car = car;
    }
    try std.testing.expectEqual(@as(usize, 500), count);

    // The tail's car is the (evacuated) cycle entry a'; a' -> b' -> a'.
    const cycle = tail_car.?;
    try std.testing.expect(g.inOldgen(@intFromPtr(cycle)));
    const b_evac = cycle.payload.cons.cdr.?;
    try std.testing.expect(g.inOldgen(@intFromPtr(b_evac)));
    try std.testing.expectEqual(
        @intFromPtr(cycle),
        @intFromPtr(b_evac.payload.cons.cdr.?),
    );

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
    g.rootPopTo(wm0);
}

// =====================================================================
//  M4 — nursery scavenge + write barrier (plan DECISION 7: T6/T7 in here;
//  T9 as the expected-panic executable tests/root_ptr_panic.zig wired into
//  the gc-test step — Zig 0.16 has no in-process panic assertion).
// =====================================================================

test "M4 T6 write barrier: dirty old-gen vector retains nursery value" {
    var g = try testInit();
    defer g.deinit();

    // Old-gen Value element array — write-barrier site 1's target
    // (gc.md site 1 / zincvm.c:912: the address-> store).
    var arr = g.allocArrayOldgen(types.Value, 4);
    const wm0 = g.rootWatermark();
    g.rootPushPtr(@ptrCast(&arr)); // survives the full-collect phase below

    // Nursery number + a by-value cons element referencing it.
    const num = g.alloc(types.Value);
    num.* = .{ .tag = .number, .payload = .{ .number = 777 } };
    const elem: types.Value = .{
        .tag = .cons,
        .payload = .{ .cons = .{ .car = num, .cdr = null } },
    };
    try std.testing.expect(g.inNursery(@intFromPtr(num)));
    try std.testing.expect(g.inOldgen(@intFromPtr(arr)));

    // Store through the barrier: array is old-gen (SPACE TAG test) and the
    // stored value references the nursery -> must dirty the array.
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);
    g.writeBarrierVectorStore(arr, 0, elem);
    try std.testing.expectEqual(@as(usize, 1), g.dirty_vectors_count);
    try std.testing.expect(g.dirty_vectors_fired >= 1);

    // NO root references num: the remembered set is its ONLY path to
    // survival.  Force the scavenge.
    g.collectNursery(.@"test");

    try std.testing.expectEqual(@as(u64, 1), g.nursery_scavenge_count);
    // The element now points at the PROMOTED copy — not into the nursery.
    const car = arr[0].payload.cons.car.?;
    try std.testing.expect(!g.inNursery(@intFromPtr(car)));
    try std.testing.expect(g.inOldgen(@intFromPtr(car)));
    try std.testing.expectEqual(types.ValTag.number, car.tag);
    try std.testing.expectEqual(@as(i64, 777), car.payload.number);

    // Nursery fully reclaimed: pages re-tagged + cursor rewound.
    try std.testing.expect(g.nurseryIsEmpty());
    try std.testing.expectEqual(
        @intFromPtr(heap_mod.pageToGcp(g.nursery_first)),
        g.nursery_cur,
    );
    try std.testing.expect(g.nursery_pages_reclaimed >= heap_mod.NURSERY_PAGES);

    // Remembered-set lifecycle: cleared at scavenge end (gc.c:1726).
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_scavenge));

    // SPACE TAG vs ADDRESS RANGE — the C Test 6 rationale (gc.md): a full
    // collect evacuates arr and releases its old page; the old address stays
    // past the nursery range forever, but its page tag flips to 0, so the
    // space-tag inOldgen must now report false.  An address-range test
    // (page > nursery_last) would still say true and wrongly keep dirtying
    // arrays in dead space.
    const arr_old_addr = @intFromPtr(arr);
    g.collect(.@"test");
    try std.testing.expect(@intFromPtr(arr) != arr_old_addr); // arr moved
    try std.testing.expect(!g.inOldgen(arr_old_addr)); // dead from-space page
    try std.testing.expect(g.inOldgen(@intFromPtr(arr))); // live copy

    // arr survived the full collect with the promoted contents intact.
    try std.testing.expectEqual(types.ValTag.cons, arr[0].tag);
    try std.testing.expectEqual(@as(i64, 777), arr[0].payload.cons.car.?.payload.number);
    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));
    g.rootPopTo(wm0);
}

test "M4 T6b write barrier skips non-old-gen arrays and non-nursery values" {
    var g = try testInit();
    defer g.deinit();

    const arr = g.allocArrayOldgen(types.Value, 2);
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);

    // Stored value with NO nursery reference -> no barrier (zincvm.c:912's
    // value_references_nursery conjunct).
    const plain: types.Value = .{ .tag = .number, .payload = .{ .number = 5 } };
    g.writeBarrierVectorStore(arr, 1, plain);
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);

    // Nursery-resident array -> no barrier (evacuated wholesale next scavenge;
    // the gc_in_oldgen conjunct).
    const narr = g.allocArray(types.Value, 2); // 80 B body -> nursery
    try std.testing.expect(g.inNursery(@intFromPtr(&narr[0])));
    const num = g.alloc(types.Value);
    num.* = .{ .tag = .number, .payload = .{ .number = 9 } };
    const hot: types.Value = .{
        .tag = .cons,
        .payload = .{ .cons = .{ .car = num, .cdr = null } },
    };
    g.writeBarrierVectorStore(narr, 0, hot);
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);

    // Contrast: same hot value into the OLD-GEN array does dirty it.
    g.writeBarrierVectorStore(arr, 0, hot);
    try std.testing.expectEqual(@as(usize, 1), g.dirty_vectors_count);
}

test "M4 T6c dirty-overflow fallback queues old-gen OBJECT pages" {
    var g = try testInit();
    defer g.deinit();

    // Two old-gen arrays: one holding a live nursery reference, one plain.
    var arr = g.allocArrayOldgen(types.Value, 2);
    const wm0 = g.rootWatermark();
    g.rootPushPtr(@ptrCast(&arr));
    const dead_arr = g.allocArrayOldgen(types.Value, 2);
    _ = dead_arr;

    const num = g.alloc(types.Value);
    num.* = .{ .tag = .number, .payload = .{ .number = 31337 } };
    const hot: types.Value = .{
        .tag = .cons,
        .payload = .{ .cons = .{ .car = num, .cdr = null } },
    };
    g.writeBarrierVectorStore(arr, 0, hot);
    try std.testing.expectEqual(@as(usize, 1), g.dirty_vectors_count);

    // Trip the overflow valve WITHOUT recording arr beyond its single entry
    // (the valve discards the pending set, C: gc.c:361-364) — simulates an
    // incomplete remembered set that the fallback must cover conservatively.
    g.dirty_vectors_overflow = true;

    g.collectNursery(.@"test");

    // The fallback queued every old-gen OBJECT page (gc.c:1669-1674), so the
    // drain scanned arr anyway and promoted num: retention still holds.
    const car = arr[0].payload.cons.car.?;
    try std.testing.expect(!g.inNursery(@intFromPtr(car)));
    try std.testing.expect(g.inOldgen(@intFromPtr(car)));
    try std.testing.expectEqual(@as(i64, 31337), car.payload.number);

    // The valve resets with the rest of the remembered set (gc.c:1726).
    try std.testing.expect(!g.dirty_vectors_overflow);
    try std.testing.expectEqual(@as(usize, 0), g.dirty_vectors_count);

    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_scavenge));
    g.rootPopTo(wm0);
}

test "M4 T7 trigger counters: burst allocation fires PREEMPTIVE, never REACTIVE" {
    var g = try testInit();
    defer g.deinit();

    // A burst of dead Values (48 B each = 40 B body + 8 B header word) that
    // crosses the pre-emptive low-water line mid-burst: PREEMPTIVE fires when
    // free nursery space drops to NURSERY_SCAVENGE_FREE_LOWATER (gc.c
    // :2177-2185), then the rewind restores the full region, so the remainder
    // never reaches REACTIVE.  The count derives from NURSERY_BYTES so the
    // test tracks a nursery retune (P0c: 2 MB -> 8 MB) instead of assuming the
    // C default — the invariant, not the literal count, is what is asserted.
    const bytes_per_value = 48;
    const count = (heap_mod.NURSERY_BYTES - heap_mod.NURSERY_SCAVENGE_FREE_LOWATER) /
        bytes_per_value + 1;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const v = g.alloc(types.Value);
        v.* = .{ .tag = .number, .payload = .{ .number = @intCast(i) } };
    }

    try std.testing.expect(g.preemptive_scavenge_count >= 1); // C Test 7
    try std.testing.expectEqual(@as(u64, 0), g.reactive_scavenge_count);
    try std.testing.expect(g.nursery_scavenge_count >= 1);
    // All Values were dead (no roots, no dirty arrays): nothing promoted, so
    // neither the old-gen THRESHOLD/LASTRESORT full collect nor any growth
    // fired.
    try std.testing.expectEqual(@as(u64, 0), g.full_collect_count);
    try std.testing.expectEqual(@as(usize, 0), g.allocatedPages());
    try std.testing.expect(g.nurseryIsEmpty());
}

test "M4 scavenge survivability across repeated scavenges (no flip)" {
    var g = try testInit();
    defer g.deinit();

    // A rooted cons pair survives three consecutive scavenges, staying out
    // of the nursery each time (after the first promotion the survivors are
    // old-gen, so gcMove's to-space branch returns them unchanged and merely
    // queues the page for scanning).
    const c2 = g.alloc(types.Value);
    c2.* = .{ .tag = .number, .payload = .{ .number = 2 } };
    const c1 = g.alloc(types.Value);
    c1.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = c2, .cdr = null } } };
    var root: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = c1, .cdr = null } } };
    const wm0 = g.rootWatermark();
    g.rootPushValue(&root);

    var round: usize = 0;
    while (round < 3) : (round += 1) {
        g.collectNursery(.@"test");
        try std.testing.expectEqual(@as(u64, round + 1), g.nursery_scavenge_count);
        const l1 = root.payload.cons.car.?;
        try std.testing.expect(!g.inNursery(@intFromPtr(l1)));
        try std.testing.expect(g.inOldgen(@intFromPtr(l1)));
        try std.testing.expectEqual(@as(i64, 2), l1.payload.cons.car.?.payload.number);
        try std.testing.expect(g.nurseryIsEmpty());
        try std.testing.expectEqual(
            @as(usize, 0),
            gc.collect.debugVerifyHeap(&g, .post_scavenge),
        );
        // NO semi-space flip during a scavenge (gc.c:1650).
        try std.testing.expectEqual(@as(usize, 1), g.current_space);
        try std.testing.expectEqual(@as(usize, 1), g.next_space);
    }

    g.rootPopTo(wm0);
}

test "M4 T9 ROOT_PTR interior defense (expected-panic exe)" {
    // The defense itself CANNOT run in-process: std.debug.panic aborts the
    // test binary (Zig 0.16 has no std.testing.expectPanics).  It is proven
    // by tests/root_ptr_panic.zig — a build-level expected-failure exe wired
    // into this gc-test step (build.zig, expectExitCode(42)) that registers
    // &big[90] (a CONTINUED tail page of an 8-page old-gen array) as a
    // ROOT_PTR, calls collectNursery, and asserts via its root panic
    // override that scanRoots panicked with the gc.c:1527-1539 message.
    // Here we only pin the structural precondition the defense relies on:
    // tail pages of a multi-page object are CONTINUED-tagged, head pages are
    // not.
    var g = try testInit();
    defer g.deinit();

    const big = g.allocArrayOldgen(types.Value, 100); // 501 words -> 8 pages
    const head_pg = heap_mod.gcpToPage(@intFromPtr(&big[0]));
    const tail_pg = heap_mod.gcpToPage(@intFromPtr(&big[90]));
    try std.testing.expect(tail_pg > head_pg);
    try std.testing.expectEqual(heap_mod.OBJECT, g.type_page[g.md(head_pg)]);
    try std.testing.expectEqual(heap_mod.CONTINUED, g.type_page[g.md(tail_pg)]);
}

// =====================================================================
//  M5 — gate hardening: T8 churn (plan DECISION 7).
//  Port of zinctest.c gc_root_churn_test: a 5000-node cons tree held ONLY
//  by a ROOT_VALUE precise root survives ~200K transient allocations with
//  forced scavenges + periodic full collects.  This is the missed-root
//  detector: under precise-authoritative collection the root is the tree's
//  ONLY survival path, so any root/scan slip reclaims a reachable node and
//  the final walk catches the corruption.
// =====================================================================

/// Deterministic LCG for the transient garbage stream (C: zinctest.c
/// churn_lcg, seed 0xDEADBEEF).  Exact values don't matter — determinism does.
var churn_lcg: u64 = 0xDEADBEEF;
fn churnLcgNext() u64 {
    churn_lcg = churn_lcg *% 1103515245 +% 12345;
    return churn_lcg & 0x7FFFFFFF;
}

/// Nil Value literal (C: zincvm.c val_nil()).
inline fn valNil() types.Value {
    return .{ .tag = .nil, .payload = .{ .number = 0 } };
}

/// Faithful port of C: zincvm.c val_cons — allocates a heap `car` cell and a
/// heap `cdr` cell (both GC_TYPE_VALUE) and returns a cons Value whose car/cdr
/// point at them.  Roots car, cdr, and car_cell across the two gc_alloc calls
/// (C's val_cons rooting): a natural preemptive/reactive scavenge CAN fire
/// between the two allocs, and the pinned slots keep the intermediates valid.
fn valCons(g: *gc.Gc, car: types.Value, cdr: types.Value) types.Value {
    var car_copy = car;
    var cdr_copy = cdr;
    g.rootPushValue(&car_copy);
    g.rootPushValue(&cdr_copy);
    const car_cell = g.alloc(types.Value);
    var car_root: ?*types.Value = car_cell;
    g.rootPushPtr(@ptrCast(&car_root));
    const cdr_cell = g.alloc(types.Value);
    car_root.?.* = car;
    cdr_cell.* = cdr;
    g.rootPop(); // car_root
    g.rootPop(); // cdr
    g.rootPop(); // car
    return .{ .tag = .cons, .payload = .{ .cons = .{ .car = car_root, .cdr = cdr_cell } } };
}

/// Walk the ROOT_VALUE-rooted cons tree and confirm every node's car is the
/// number `count` and the tail lands on nil with node_count nodes.  Faithful
/// to C's `cur = *cur.cdr` traversal (cdr points at a heap cell holding the
/// next cons Value; the tail cell holds a nil Value).
fn treeVerify(root: *const types.Value, node_count: usize) bool {
    var count: usize = 0;
    var cur = root.*;
    while (cur.tag == types.ValTag.cons) {
        if (count >= node_count) return false;
        const car = cur.payload.cons.car orelse return false;
        if (car.tag != types.ValTag.number) return false;
        if (car.payload.number != @as(i64, @intCast(count))) return false;
        const cdr = cur.payload.cons.cdr orelse return false;
        cur = cdr.*;
        count += 1;
    }
    return cur.tag == types.ValTag.nil and count == node_count;
}

test "M5 T8 churn: 5000-node ROOT_VALUE tree survives 200K alloc churn" {
    var g = try testInit();
    defer g.deinit();

    churn_lcg = 0xDEADBEEF;
    const node_count: usize = 5000;

    // Build the persistent tree bottom-up through the nursery, head held ONLY
    // on the precise root (ROOT_VALUE).  The build (~10k Value allocs = 480 KB)
    // stays below the 2 MB - 256 KB nursery low-water (gc.c:2177), so no
    // scavenge fires mid-build and the intermediate car/cdr pointers stay valid.
    var root: types.Value = valNil();
    const wm0 = g.rootWatermark();
    g.rootPushValue(&root);
    var i: i64 = @as(i64, @intCast(node_count)) - 1;
    while (i >= 0) : (i -= 1) {
        const num: types.Value = .{ .tag = .number, .payload = .{ .number = i } };
        root = valCons(&g, num, root);
    }
    try std.testing.expect(treeVerify(&root, node_count));

    const sv0 = g.nursery_scavenge_count;
    const fc0 = g.full_collect_count;

    var iter: usize = 0;
    while (iter < 200_000) : (iter += 1) {
        // Transient dead cons cells (C: g1/g2/g3) — unrooted, reclaimed next
        // scavenge.  Each val_cons allocs 2 Values, so 6 Values/iter.
        const n1: types.Value = .{ .tag = .number, .payload = .{ .number = @intCast(churnLcgNext()) } };
        const g1 = valCons(&g, n1, valNil());
        const n2: types.Value = .{ .tag = .number, .payload = .{ .number = @intCast(churnLcgNext()) } };
        const g2 = valCons(&g, n2, g1);
        const n3: types.Value = .{ .tag = .number, .payload = .{ .number = @intCast(churnLcgNext()) } };
        const g3 = valCons(&g, n3, g2);
        _ = g3;

        // Force a nursery scavenge every ~2000 iters (C: gc_alloc_atomic loop
        // forcing; here direct for determinism) + verify the heap is clean.
        if (iter % 2000 == 0) {
            g.collectNursery(.@"test");
            try std.testing.expectEqual(
                @as(usize, 0),
                gc.collect.debugVerifyHeap(&g, .post_scavenge),
            );
        }

        // Force a full collect (semi-space swap survival) + verify every 50K.
        if (iter % 50_000 == 0 and iter > 0) {
            g.collect(.@"test");
            try std.testing.expectEqual(
                @as(usize, 0),
                gc.collect.debugVerifyHeap(&g, .post_collect),
            );
        }

        // Walk + verify the whole tree every 20K iters (C: iter % 10000).
        if (iter % 20_000 == 0 and iter > 0) {
            try std.testing.expect(treeVerify(&root, node_count));
        }
    }

    // Final verification + one last heap check.
    try std.testing.expect(treeVerify(&root, node_count));
    try std.testing.expectEqual(
        @as(usize, 0),
        gc.collect.debugVerifyHeap(&g, .post_scavenge),
    );
    try std.testing.expect(g.nursery_scavenge_count > sv0); // scavenges fired
    try std.testing.expect(g.full_collect_count > fc0); // full collects fired

    std.debug.print(
        "  M5 T8: tree intact after 200K iters ({d} scavenges, {d} full collects)\n",
        .{ g.nursery_scavenge_count - sv0, g.full_collect_count - fc0 },
    );

    g.rootPopTo(wm0);
}

test "M5 T10 auto-verify hook: verify_collects runs debugVerifyHeap each collect" {
    // Unit A: with verify_collects=true the collector runs debugVerifyHeap and
    // asserts zero violations after EVERY collect / collectNursery.  Merely
    // running this test without panicking exercises the hook path; the final
    // explicit debugVerifyHeap confirms the post-collect heap is clean.
    var g = try heap_mod.Gc.init(.{
        .heap_bytes = 16 * 1024 * 1024,
        .reserve_bytes = 64 * 1024 * 1024,
        .verify_collects = true,
    });
    defer g.deinit();

    const c2 = g.alloc(types.Value);
    c2.* = .{ .tag = .number, .payload = .{ .number = 3 } };
    const c1 = g.alloc(types.Value);
    c1.* = .{ .tag = .cons, .payload = .{ .cons = .{ .car = c2, .cdr = null } } };
    var root: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = c1, .cdr = null } } };
    const wm0 = g.rootWatermark();
    g.rootPushValue(&root);

    // Nursery scavenge: hook fires with .post_scavenge.
    g.collectNursery(.@"test");
    try std.testing.expectEqual(@as(u64, 1), g.nursery_scavenge_count);

    // Full collect: hook fires with .post_collect.
    g.collect(.@"test");
    try std.testing.expectEqual(@as(u64, 1), g.full_collect_count);

    // Rooted graph survived both collections with contents intact.
    const car = root.payload.cons.car.?;
    try std.testing.expectEqual(types.ValTag.number, car.payload.cons.car.?.tag);
    try std.testing.expectEqual(@as(i64, 3), car.payload.cons.car.?.payload.number);
    try std.testing.expectEqual(@as(usize, 0), gc.collect.debugVerifyHeap(&g, .post_collect));

    g.rootPopTo(wm0);
}

test "M5 T11 RAII root guard: rootValue+defer keeps a Value alive across collection" {
    // Unit D: `var guard = gc.rootValue(&v); defer guard.end();` roots `v` for
    // the enclosing block and pops it automatically on unwind — root balance is
    // automatic and the rooted Value survives collection with contents intact.
    var g = try testInit();
    defer g.deinit();

    var root: types.Value = .{ .tag = .cons, .payload = .{ .cons = .{ .car = null, .cdr = null } } };
    {
        var guard = g.rootValue(&root);
        defer guard.end();

        const obj = g.alloc(types.Value);
        obj.* = .{ .tag = .number, .payload = .{ .number = 42 } };
        root.payload.cons.car = obj;

        // Force a full collect while the guard is live: the rooted Value's
        // interior pointer is rewritten in place and the object survives.
        g.collect(.@"test");
        try std.testing.expectEqual(@as(u64, 1), g.full_collect_count);

        const car = root.payload.cons.car.?;
        try std.testing.expect(!g.inNursery(@intFromPtr(car)));
        try std.testing.expect(g.inOldgen(@intFromPtr(car)));
        try std.testing.expectEqual(types.ValTag.number, car.tag);
        try std.testing.expectEqual(@as(i64, 42), car.payload.number);
    } // guard.end() pops the root here, on block unwind.

    // After the block the pushed root has been popped: balance restored.
    try std.testing.expectEqual(@as(usize, 0), g.rootWatermark());
}

