//! src/gc/collect.zig — collector driver + object movement (M2 + M3).
//!
//! The heap core (gc/heap.zig) calls these entry points from its allocation
//! triggers:
//!   - allocatepage  LASTRESORT full collect      (C: gc.c:1870)
//!   - gc_alloc      THRESHOLD full collect       (C: gc.c:2263)
//!   - gc_alloc      PREEMPTIVE nursery scavenge  (C: gc.c:2182)
//!   - gc_alloc      REACTIVE nursery scavenge    (C: gc.c:2248)
//!   - gc_alloc_oldgen ALLOC full collect         (C: gc.c:2291)
//!
//! Milestone ownership:
//!   M2 (done): moveInternal / gcMove / drainScanObject — the per-object
//!     pointer movement + typed scanning core.
//!   M3 (done): scanRoots (C: gc.c:1478-1611), cheneyDrain + drainWalkPage
//!     with the deferred-resume catch-up policy (C: gc.c:500-600), collect —
//!     Phase-0 promotion + semi-space flip + dead-page release (C: gc.c
//!     :604-746), and the test-only debugVerifyHeap.
//!   M4 (done): collectNursery — C: gc.c:1626-1735 (no-flip scavenge,
//!     dirty-vectors scan with overflow fallback, nursery reset + cursor
//!     rewind, dirty-set clears).

const std = @import("std");
const types = @import("types.zig");
const heap = @import("heap.zig");
const scan = @import("scan.zig");
const roots = @import("roots.zig");

const Gc = heap.Gc;
const Trigger = heap.Trigger;

// ---------------------------------------------------------------------
//  M2: object movement + typed drain (not yet driven by a collector)
// ---------------------------------------------------------------------

/// C: gc.c:1952-1991 move_internal — copy the object at `cp` (a body pointer,
/// so the header is at (cp-1)[0]) into to-space with the SAME type tag and write
/// a forwarding pointer into the old header.  Returns the new body pointer.
///
/// Faithful port notes (artifact-2 validated idiom):
///   - The header word is (cp-1)[0]; if already FORWARDED it IS the forwarding
///     pointer to the new body (header bit0 == 0).
///   - Allocation uses gcalloc_internal (the header word count minus the header
///     itself is the body byte count), so promoted objects land via the SAME
///     allocator the collector otherwise uses (respecting page/tag bookkeeping).
///   - The word copy copies header + body (HEADER_WORDS words total) from
///     cp-1 to np-1, then writes the new body address into the old header.
///   - The --gc-watch-alloc diagnostic is omitted (plan DECISION 4).
pub fn moveInternal(gc: *Gc, cp: [*]usize, type_tag: types.GcTypeTag) [*]usize {
    // C: gc.c:1959 `if (cp == NULL) return NULL;` — the only caller (gcMove)
    // already rejects null, so this branch is unreachable; a null body pointer
    // cannot be represented as [*]usize and is filtered in gcMove.
    const header = (cp - 1)[0];

    // C: gc.c:1962-1964.
    if (types.forwarded(header)) {
        return @ptrCast(@as(*usize, @ptrFromInt(header)));
    }

    // C: gc.c:1967 — allocate in to-space with the same type tag.  The body byte
    // count is (HEADER_WORDS - 1) * WORDBYTES.
    const np = gc.gcalloc_internal((types.headerWords(header) - 1) * heap.WORDBYTES, type_tag);

    // C: gc.c:1969-1975 — copy header + body (HEADER_WORDS words) from cp-1 to np-1.
    const to = np - 1;
    const from = cp - 1;
    const cnt = types.headerWords(header);
    var i: usize = 0;
    while (i < cnt) : (i += 1) {
        to[i] = from[i];
    }

    // C: gc.c:1988 — write the forwarding pointer (new body address) in the old
    // header.  bit0 == 0 makes FORWARDED true on re-visit.
    (cp - 1)[0] = @intFromPtr(np);

    return np;
}

/// C: gc.c:1995-2053 gc_move — public evacuation function.  Extracts the type
/// tag from the header and delegates to moveInternal, with the full case
/// analysis: not-in-heap / already-to-space / nursery (scavenge-only, space-tag
/// and forwarded handling) / old-gen forwarded / move.
///
/// Returns the address the caller should store in place of `p` (NULL passthrough).
pub fn gcMove(gc: *Gc, p: ?*anyopaque) ?*anyopaque {
    // C: gc.c:2000 `if (p == NULL) return NULL;`
    const p_addr = if (p) |ptr| @intFromPtr(ptr) else return null;
    const cp: [*]usize = @ptrCast(@as(*usize, @ptrFromInt(p_addr)));

    // C: gc.c:2003 — page of the body pointer.
    const page = heap.gcpToPage(p_addr);

    // C: gc.c:2017-2018 — not in heap at all?  (NULL already handled; non-GC
    // pointers such as C-heap strdup'd strings pass through unchanged.)
    if (page < gc.firstheappage or page > gc.lastheappage) return p;

    // C: gc.c:2021-2028 — already in to-space.  During a nursery scavenge,
    // old-gen objects (space == current == next) must be queued for scanning
    // since their bodies may contain nursery pointers.
    if (gc.space[gc.md(page)] == gc.next_space) {
        if (gc.in_scavenge) gc.queue(page);
        return p;
    }

    // C: gc.c:2032-2044 — nursery object.
    if (gc.inNursery(p_addr)) {
        // C: gc.c:2033 — full collect leaves the nursery untouched.
        if (!gc.in_scavenge) return p;
        // C: gc.c:2034-2039 — space != NURSERY: already promoted this scavenge
        // OR old-gen in nursery range.  Check forwarded / stay.
        if (gc.space[gc.md(page)] != heap.NURSERY) {
            const header = (cp - 1)[0];
            if (types.forwarded(header)) return @ptrCast(@as(*usize, @ptrFromInt(header)));
            return p;
        }
        // C: gc.c:2041-2043 — nursery survivor: copy to old-gen (current_space),
        // leaving a forwarding pointer.  Re-visits via two refs short-circuit.
        const header = (cp - 1)[0];
        if (types.forwarded(header)) return @ptrCast(@as(*usize, @ptrFromInt(header)));
        return @ptrCast(moveInternal(gc, cp, @enumFromInt(types.headerType(header))));
    }

    // C: gc.c:2046-2052 — old-gen (from-space) object: forwarded check, else move.
    const header = (cp - 1)[0];
    if (types.forwarded(header)) return @ptrCast(@as(*usize, @ptrFromInt(header)));
    return @ptrCast(moveInternal(gc, cp, @enumFromInt(types.headerType(header))));
}

/// C: gc.c:501-541 drain_scan_object — the shared per-object typed scan,
/// dispatched by type tag.  `body` points at the first body word (header is
/// (body-1)[0]); `hw` is the header word count.  RAW is a no-op; VALUE scans by
/// tag; VALUE_ARRAY / INSTR_ARRAY / CALLFRAME_ARRAY scan each element.
pub fn drainScanObject(gc: *Gc, body: [*]usize, ty: u32, hw: usize) void {
    switch (ty) {
        0 => {}, // GC_TYPE_RAW — no scan (string/error char data)

        1 => { // GC_TYPE_VALUE
            const v: *types.Value = @ptrCast(@alignCast(body));
            scan.scanValue(gc, v);
        },

        2 => { // GC_TYPE_VALUE_ARRAY
            const body_bytes = (hw - 1) * heap.WORDBYTES;
            const count: usize = @intCast(body_bytes / @sizeOf(types.Value));
            const arr: [*]types.Value = @ptrCast(@alignCast(body));
            var j: usize = 0;
            while (j < count) : (j += 1) scan.scanValue(gc, &arr[j]);
        },

        3 => { // GC_TYPE_INSTR_ARRAY
            const body_bytes = (hw - 1) * heap.WORDBYTES;
            const count: usize = @intCast(body_bytes / @sizeOf(types.Instr));
            const arr: [*]types.Instr = @ptrCast(@alignCast(body));
            var j: usize = 0;
            while (j < count) : (j += 1) scan.evacInstr(gc, &arr[j]);
        },

        4 => { // GC_TYPE_CALLFRAME_ARRAY
            const body_bytes = (hw - 1) * heap.WORDBYTES;
            const count: usize = @intCast(body_bytes / @sizeOf(types.CallFrame));
            const arr: [*]types.CallFrame = @ptrCast(@alignCast(body));
            var j: usize = 0;
            while (j < count) : (j += 1) {
                scan.evacuate(gc, @ptrCast(&arr[j].code));
                scan.evacuate(gc, @ptrCast(&arr[j].env));
                scan.evacuate(gc, @ptrCast(&arr[j].stack.data));
            }
        },

        else => {},
    }
}

// ---------------------------------------------------------------------
//  M3: shared Cheney drain with the deferred-resume catch-up policy
//  — C: gc.c:476-600 (the Bug 2 fix; port EXACTLY)
// ---------------------------------------------------------------------

/// C: gc.c:543-560 drain_walk_page — walk objects on page `qpg` starting at
/// `cp`, stopping at the page boundary, at freep (when freep is on this page),
/// or at an invalid header (false-positive guard — same break condition as the
/// C drain loops).  Returns the cursor where the walk stopped.  Scanning may
/// append pages to the queue and advance freep; both are picked up by the
/// caller (cheneyDrain).
fn drainWalkPage(gc: *Gc, qpg: usize, cp_in: [*]usize) [*]usize {
    var cp = cp_in;
    while (heap.gcpToPage(@intFromPtr(cp)) == qpg and
        @intFromPtr(cp) != @intFromPtr(gc.freep))
    {
        const hdr = cp[0];
        const hw = types.headerWords(hdr);
        const ty = types.headerType(hdr);

        // C: gc.c:553-554 — hw == 0 / ty out of range: false-positive guard.
        if (hw == 0) break;
        if (ty > @intFromEnum(types.GcTypeTag.callframe_array)) break;

        drainScanObject(gc, cp + 1, ty, hw);
        cp += hw;
    }
    return cp;
}

/// C: gc.c:562-600 cheney_drain — drain the Cheney queue with the
/// deferred-resume policy.  THE Bug-2 fix, ported exactly:
///
/// A page's object walk may end EARLY at cp == freep while the page still has
/// bump slack.  That early exit is only "done" if the queue is empty — the
/// queue can still hold OLD pages queued later via gcMove's queue(page)
/// branch, and scans of those pages promote/evacuate MORE objects into the
/// slack of the already-dequeued page (freep stays on it).  Those later
/// objects would never be visited (queue()'s page_queued dedup blocks
/// re-queueing), so their pointer fields would never be evacuated — stale
/// pointers into recycled memory.
///
/// Fix: only the freep page can receive new objects, so at most ONE page can
/// be mid-catch-up at any time.  When a walk ends at cp == freep with pages
/// still queued, record (page, cp) as the resume point instead of
/// re-queueing.  The deferred region is walked (a) immediately when a NEW
/// page catches up (the old page is filler-capped by then), or (b) after the
/// queue drains.  Each object is scanned at most twice, never quadratically.
fn cheneyDrain(gc: *Gc) void {
    var defer_pg: usize = 0;
    var defer_cp: ?[*]usize = null;

    outer: while (true) {
        while (gc.queue_head != 0) {
            const qpg = gc.queue_head;
            const cp = drainWalkPage(gc, qpg, heap.pageToGcp(qpg));
            gc.queue_head = gc.gc_link[gc.md(gc.queue_head)];

            // C: gc.c:574 — walk caught up with the bump frontier mid-page
            // while pages remain queued: defer this page's remaining slack.
            if (gc.queue_head != 0 and
                @intFromPtr(cp) == @intFromPtr(gc.freep) and
                heap.gcpToPage(@intFromPtr(gc.freep)) == qpg)
            {
                if (defer_pg != 0 and defer_pg != qpg) {
                    // C: gc.c:575-580 — a new page caught up, so freep left
                    // the old deferred page — its slack is filler-capped.
                    // Walk its tail now (its scanning may queue pages).
                    _ = drainWalkPage(gc, defer_pg, defer_cp.?);
                }
                defer_pg = qpg;
                defer_cp = cp;
            }
        }
        if (defer_pg == 0) break;

        // C: gc.c:587-593 — freep moved on: the deferred page's slack is
        // filler-capped; walk its tail once (stops at the filler / page
        // boundary).  The tail walk may have queued pages — loop back.
        if (heap.gcpToPage(@intFromPtr(gc.freep)) != defer_pg) {
            _ = drainWalkPage(gc, defer_pg, defer_cp.?);
            defer_pg = 0;
            continue :outer;
        }
        // C: gc.c:594 — no growth: genuinely done.
        if (@intFromPtr(gc.freep) == @intFromPtr(defer_cp.?)) break;
        // C: gc.c:595-598 — flush walk from the deferred cursor.
        const cp = drainWalkPage(gc, defer_pg, defer_cp.?);
        defer_cp = cp;
        if (heap.gcpToPage(@intFromPtr(cp)) != defer_pg) defer_pg = 0; // page crossed
        // loop back: the flush walk may have queued pages
    }
}

// ---------------------------------------------------------------------
//  M3: root scan — C: gc.c:1478-1611 gc_scan_roots
// ---------------------------------------------------------------------

/// C: gc.c:1478-1611 gc_scan_roots — walk the precise-root shadow stack plus
/// the registered typed walkers (defun table dirty-gated during scavenge /
/// full during full collect; values table always full; traced_code
/// evacuated).  This is the SOLE authoritative root source; roots are
/// EVACUATED in place (scanValue / evacuate rewrite the root slots).
///
/// The opt-in root-set dump (gc.c:1479-1515) is omitted (plan DECISION 4).
fn scanRoots(gc: *Gc) void {
    // 1. Shadow stack entries — C: gc.c:1517-1571.
    var i: usize = 0;
    while (i < gc.shadow_len) : (i += 1) {
        const r = &gc.shadow_stack[i];
        switch (r.kind) {
            .ROOT_PTR => {
                // C: gc.c:1521-1542 — ROOT_PTR must point at an object HEAD
                // (gc.h:155-161).  An interior pointer into a multi-page
                // object would make gcMove read a garbage header at *(ptr-1)
                // — UB / heap corruption.  Cheap defense: a head page is
                // never CONTINUED (only tail pages are).
                const slot: *usize = @ptrCast(@alignCast(r.slot));
                const addr = slot.*;
                if (addr != 0) {
                    const pg = heap.gcpToPage(addr);
                    if (pg >= gc.firstheappage and pg <= gc.lastheappage and
                        gc.type_page[gc.md(pg)] == heap.CONTINUED)
                    {
                        std.debug.panic(
                            "gc: ROOT_PTR points into a multi-page object " ++
                                "tail (page {d}) — interior pointer as root; " ++
                                "only object HEAD pointers are valid roots",
                            .{pg},
                        );
                    }
                }
                scan.evacuate(gc, slot);
            },
            .ROOT_VALUE => {
                // C: gc.c:1543-1545.
                const v: *types.Value = @ptrCast(@alignCast(r.slot));
                scan.scanValue(gc, v);
            },
            .ROOT_VALUE_VOLATILE => {
                // C: gc.c:1546-1552 — copy to tmp, scan, write back through
                // the volatile pointer.
                const vs: *volatile types.Value = @ptrCast(@alignCast(r.slot));
                var tmp: types.Value = vs.*;
                scan.scanValue(gc, &tmp);
                vs.* = tmp;
            },
            .ROOT_VALUE_ARRAY => {
                // C: gc.c:1553-1559 — count read LIVE from *np at scan time.
                const base: [*]types.Value = @ptrCast(@alignCast(r.slot));
                const n = r.np.?.*;
                var j: usize = 0;
                while (j < @as(usize, @intCast(n))) : (j += 1)
                    scan.scanValue(gc, &base[j]);
            },
            .ROOT_CALLFRAME_ARRAY => {
                // C: gc.c:1560-1569 — deliberate no-op: CallFrame headers are
                // evacuated by the Cheney drain's GC_TYPE_CALLFRAME_ARRAY
                // case once their page is queued (frame_stack is rooted as
                // ROOT_PTR).  An explicit walker here is redundant and can
                // crash during Phase-0 promotion if stack.data reads a
                // zero/invalid header (Bug #6).
            },
        }
    }

    // 2. Defun table — C: gc.c:1573-1592.  During a nursery scavenge, skip
    // non-dirty slots via the bitset to avoid re-enqueuing hundreds of stable
    // old-gen code/env pages.  Full collects scan every slot.
    if (gc.reg_global_table != null and gc.reg_global_table_len != null) {
        const gt = gc.reg_global_table.?;
        const n: usize = @intCast(gc.reg_global_table_len.?.*);
        if (gc.in_scavenge) {
            var k: usize = 0;
            while (k < n) : (k += 1) {
                if (gt[k].name != null and gc.dirtyDefunsTest(@intCast(k))) {
                    scan.scanValue(gc, &gt[k].value);
                    gc.dirty_defuns_scanned += 1;
                }
            }
        } else {
            var k: usize = 0;
            while (k < n) : (k += 1) {
                if (gt[k].name != null)
                    scan.scanValue(gc, &gt[k].value);
            }
        }
    }

    // 2b. Values table — C: gc.c:1594-1601 — always full-scanned (no bitset).
    if (gc.reg_values_table != null and gc.reg_values_table_len != null) {
        const vt = gc.reg_values_table.?;
        const n: usize = @intCast(gc.reg_values_table_len.?.*);
        var k: usize = 0;
        while (k < n) : (k += 1) {
            if (vt[k].name != null)
                scan.scanValue(gc, &vt[k].value);
        }
    }

    // 3. traced_code Instr arrays — C: gc.c:1603-1610.
    if (gc.reg_traced_code != null and gc.reg_traced_code_len != null) {
        const tc = gc.reg_traced_code.?;
        const n: usize = @intCast(gc.reg_traced_code_len.?.*);
        var k: usize = 0;
        while (k < n) : (k += 1) {
            if (tc[k] != null)
                scan.evacuate(gc, @ptrCast(&tc[k]));
        }
    }
}

// ---------------------------------------------------------------------
//  M3: full collection — C: gc.c:604-746 collect
// ---------------------------------------------------------------------

/// C: gc.c:604-746 collect — full semi-space collection:
///   1. finalize any partial page (gc.c:629-633),
///   2. Phase-0 promotion: promote all live nursery survivors to old-gen
///      (in_scavenge=1) incl. the explicit NON-dirty defun-table scan +
///      values-table scan, then drain + finalize again (gc.c:654-694),
///   3. swap semi-spaces + clear remembered sets (gc.c:696-701),
///   4. scanRoots + Cheney scavenge with deferred-resume (gc.c:705-711),
///   5. release dead from-space pages (space==current -> 0 / type_page=0)
///      and complete the flip (gc.c:722-728).
///
/// SIGALRM blocking (gc.c:607-615, 744-745) is omitted (plan DECISION 4 — no
/// alarm timeouts in the Zig runtime yet); the opt-in stale/verify diagnostics
/// (gc.c:730-742) are replaced by the test-only debugVerifyHeap below.
pub fn collect(gc: *Gc, trigger: Trigger) void {
    // C: gc.c:617-620 — guard against re-entrant collection mid-flip
    // (allocatepage's LASTRESORT never re-enters: during Phase-0
    // in_scavenge is set; during the main scavenge current != next).
    if (gc.next_space != gc.current_space) {
        std.debug.panic("gcalloc - Out of space during collect", .{});
    }

    // SAFETY-ENFORCEMENT (unit E): the semi-space tag is always 1 or 2 in
    // every legitimate collector state.
    std.debug.assert(gc.current_space == 1 or gc.current_space == 2);

    gc.full_collect_count += 1; // C: gc.c:622
    gc.collect_seq += 1; // C: gc.c:623
    if (gc.opts.verbose) {
        std.debug.print(
            "[GC FULL #{d}] trigger={s} shadow_depth={d} live_pages={d}\n",
            .{ gc.collect_seq, @tagName(trigger), gc.shadow_len, gc.allocatedpages },
        );
    }

    // C: gc.c:629-633 — finalize any partial page.
    if (gc.freewords != 0) {
        gc.freep[0] = types.makeHeader(gc.freewords, .raw);
        gc.freewords = 0;
    }

    // ---- Phase 0: promote nursery survivors to old-gen — C: gc.c:635-694 ----
    // During a full collect with in_scavenge=0, gcMove returns nursery objects
    // unchanged and nursery pages are never queued, so a nursery-resident
    // closure whose .code points into old-gen would keep a stale pointer after
    // the code array is evacuated.  Promoting every live nursery object FIRST
    // (in_scavenge=1) makes the main scavenge see no nursery pointers at all.
    // This is NOT collectNursery: no counter bumps, no dirty clears, no
    // nursery reset/cursor rewind — the full collect handles those.
    {
        gc.in_scavenge = true; // C: gc.c:655
        gc.queue_reset(); // C: gc.c:656
        scanRoots(gc); // C: gc.c:657 — dirty-gated defun scan (scavenge mode)

        // C: gc.c:659-670 — scan NON-dirty defun-table slots explicitly: every
        // entry must be walked so nursery closures reachable only through
        // non-dirty slots are also promoted (a non-dirty global may still
        // reference a nursery object stored between collections).
        if (gc.reg_global_table != null and gc.reg_global_table_len != null) {
            const gt = gc.reg_global_table.?;
            const n: usize = @intCast(gc.reg_global_table_len.?.*);
            var k: usize = 0;
            while (k < n) : (k += 1) {
                if (gt[k].name != null and !gc.dirtyDefunsTest(@intCast(k)))
                    scan.scanValue(gc, &gt[k].value);
            }
        }
        // C: gc.c:671-678 — values table: always full-scanned (scanRoots just
        // walked it; the C code scans it again — redundant but harmless since
        // forwarding pointers short-circuit — ported faithfully).
        if (gc.reg_values_table != null and gc.reg_values_table_len != null) {
            const vt = gc.reg_values_table.?;
            const n: usize = @intCast(gc.reg_values_table_len.?.*);
            var k: usize = 0;
            while (k < n) : (k += 1) {
                if (vt[k].name != null)
                    scan.scanValue(gc, &vt[k].value);
            }
        }

        // C: gc.c:680-682 — shared drain with the deferred-resume policy.
        cheneyDrain(gc);

        // C: gc.c:684-691 — promotion allocated in old-gen; finalize any
        // partial page so the full collect's gcalloc_internal starts from a
        // clean slate in to-space rather than consuming residual freewords on
        // a from-space page.
        if (gc.freewords != 0) {
            gc.freep[0] = types.makeHeader(gc.freewords, .raw);
            gc.freewords = 0;
        }

        gc.in_scavenge = false; // C: gc.c:693
    }

    // ---- Swap semi-spaces — C: gc.c:696-701 ----
    gc.next_space = if (gc.current_space == 1) 2 else 1;
    gc.allocatedpages = 0;
    gc.queue_reset();
    gc.dirtyVectorsClear();
    gc.dirtyDefunsClear();

    // ---- Root set — C: gc.c:705 ----
    scanRoots(gc);

    // ---- Cheney scavenge — C: gc.c:707-711 ----
    cheneyDrain(gc);

    // ---- finish — C: gc.c:713-728 ----
    // Every live object has been evacuated to next_space; any page still
    // tagged current_space is dead from-space and must be released to space=0
    // so the next collect's allocatepage can find it (without this, allocate-
    // page's free-page test refuses them, forcing grow_heap to mint new pages
    // — geometric heap growth and eventual OOM).
    {
        var pg = gc.firstheappage;
        while (pg <= gc.lastheappage) : (pg += 1) {
            if (gc.space[gc.md(pg)] == gc.current_space) {
                gc.space[gc.md(pg)] = 0;
                gc.type_page[gc.md(pg)] = 0;
            }
        }
    }
    gc.current_space = gc.next_space; // C: gc.c:728

    // SAFETY-ENFORCEMENT (unit A): auto-verify hook.  debugVerifyHeap is
    // test-only and its walk is gated on the current build mode so it is
    // compiled OUT of ReleaseFast/ReleaseSmall even if verify_collects is
    // set — it can never run in a production build.
    if (gc.opts.verify_collects and
        @import("builtin").mode != .ReleaseFast and
        @import("builtin").mode != .ReleaseSmall)
    {
        const e = debugVerifyHeap(gc, .post_collect);
        std.debug.assert(e == 0);
    }
}

// ---------------------------------------------------------------------
//  M4: nursery scavenge — C: gc.c:1626-1735 collect_nursery
// ---------------------------------------------------------------------

/// C: gc.c:1626-1735 collect_nursery — generational nursery scavenge:
///   1. re-entry guard (`in_scavenge` — gc.c:1630-1633),
///   2. finalize any partial old-gen page BEFORE scanning (gc.c:1644-1648)
///      so no promotion lands in an already-walked page's bump slack,
///   3. count the scavenge + verbose banner (gc.c:1652-1661),
///   4. queue_reset + scanRoots (gc.c:1663-1666; dirty-gated defun scan
///      because in_scavenge=1, values table + traced_code always full),
///   5. scan the dirty-vectors remembered set — or, on overflow, queue EVERY
///      old-gen OBJECT page as the fallback (gc.c:1668-1687),
///   6. cheneyDrain with the shared deferred-resume policy (gc.c:1698) —
///      nursery survivors are copied to old-gen by gcMove's nursery branch
///      and their destination pages queued on allocate (heap.zig
///      allocatepage), so the drain transitively scans them,
///   7. reset ALL nursery pages to NURSERY + rewind the bump cursor — the
///      full-reclaim trick that makes the nursery reusable every cycle
///      (gc.c:1711-1720),
///   8. clear the write-barrier remembered sets (gc.c:1726-1727) — their
///      entries were all just scanned (or the overflow valve reset),
///   9. in_scavenge = 0 (gc.c:1733).
///
/// NO semi-space flip, NO allocatedpages reset, NO promotion of the defun
/// table's non-dirty slots (that is full-collect Phase-0's job).
///
/// SIGALRM blocking (gc.c:1636-1641, 1735) is omitted (plan DECISION 4 — no
/// alarm timeouts in the Zig runtime yet); the opt-in verifiers
/// (gc.c:1700-1710, 1722-1732) are replaced by the test-only
/// debugVerifyHeap(.post_scavenge).
pub fn collectNursery(gc: *Gc, trigger: Trigger) void {
    // C: gc.c:1630-1633 — guard against recursive entry (exit(1) → panic,
    // port convention).
    if (gc.in_scavenge) {
        std.debug.panic("collect_nursery: re-entered during scavenge", .{});
    }

    gc.in_scavenge = true; // C: gc.c:1642

    // C: gc.c:1644-1648 — finalize any partial old-gen page before scanning.
    // Load-bearing: without this, a promotion could land in the bump slack of
    // a page whose walk already passed freep — exactly the stale-pointer
    // hazard the deferred-resume drain patch guards within ONE collection;
    // across the finalize boundary the filler cap is what terminates walks.
    if (gc.freewords != 0) {
        gc.freep[0] = types.makeHeader(gc.freewords, .raw);
        gc.freewords = 0;
    }

    // No semi-space swap; no reset of allocatedpages — C: gc.c:1650.

    // C: gc.c:1652-1661 — count this as a real scavenge.
    gc.nursery_scavenge_count += 1;
    gc.collect_seq += 1;
    if (gc.opts.verbose) {
        std.debug.print(
            "[GC NURSERY #{d}] trigger={s} shadow_depth={d} nursery_free={d}\n",
            .{ gc.collect_seq, @tagName(trigger), gc.shadow_len, gc.nursery_end - gc.nursery_cur },
        );
    }

    // C: gc.c:1663 — reset the Cheney queue.
    gc.queue_reset();

    // ---- root set — C: gc.c:1666 ----
    // in_scavenge=1 makes scanRoots' defun-table walk dirty-gated (only
    // entries marked by the site-2 global_set barrier are evacuated).
    scanRoots(gc);

    // ---- scan dirty old-gen vectors (write-barrier remembered set) —
    // C: gc.c:1668-1687 ----
    if (gc.dirty_vectors_overflow) {
        // C: gc.c:1669-1674 — overflow valve tripped (gc.c:361-364): the
        // remembered set is incomplete, so fall back to queuing EVERY
        // old-gen OBJECT page in current_space and letting the drain scan
        // them all.  Starts past the nursery region (which is never tagged
        // current_space anyway) — C iterates nursery_last+1..lastheappage.
        var pg = gc.nursery_last + 1;
        while (pg <= gc.lastheappage) : (pg += 1) {
            if (gc.space[gc.md(pg)] == gc.current_space and
                gc.type_page[gc.md(pg)] == heap.OBJECT)
            {
                gc.queue(pg);
            }
        }
    } else {
        // C: gc.c:1675-1686 — scan each remembered array inline.
        // NOTE: `data` is old-gen (inOldgen = SPACE TAG test, gc.c:338-342 —
        // NOT an address-range test: an address-range version silently
        // misses promoted-in-place arrays and wrongly includes dead
        // from-space pages), so it cannot move during this scavenge and the
        // cached pointer stays valid across the whole loop.
        var k: usize = 0;
        while (k < gc.dirty_vectors_count) : (k += 1) {
            const data = gc.dirty_vectors[k];
            if (!gc.inOldgen(@intFromPtr(data))) continue; // C: gc.c:1677
            const cp: [*]usize = @ptrCast(data);
            const header = (cp - 1)[0]; // C: `(uintptr_t *)data - 1`
            const ty = types.headerType(header);
            if (ty != @intFromEnum(types.GcTypeTag.value_array)) continue; // C: gc.c:1680
            const hw = types.headerWords(header);
            const body_bytes = (hw - 1) * heap.WORDBYTES;
            const count = body_bytes / @sizeOf(types.Value);
            var j: usize = 0;
            while (j < count) : (j += 1)
                scan.scanValue(gc, &data[j]); // C: gc.c:1685
        }
    }

    // ---- Cheney scavenge — C: gc.c:1698 ----
    // Nursery survivors are copied to old-gen (gcMove nursery branch →
    // moveInternal) and their destination pages queued via allocatepage's
    // queue-on-allocate-during-scavenge; nursery pages themselves are never
    // queued.  Shared drain with the deferred-resume catch-up policy (Bug 2
    // fix): nursery survivors promoted into a promotion page's bump slack
    // after that page's walk caught up with freep are still scanned.
    cheneyDrain(gc);

    // ---- reset nursery: full reclaim — C: gc.c:1711-1720 ----
    // Survivors have been copied to old-gen; reset ALL nursery pages to
    // NURSERY and rewind the bump cursor to the region start.  Nothing can
    // live in the nursery across this point: allocatepage never hands out
    // NURSERY-tagged pages to old-gen allocations, so promoted copies always
    // sit past nursery_last.
    {
        var pg = gc.nursery_first;
        while (pg <= gc.nursery_last) : (pg += 1)
            gc.space[gc.md(pg)] = heap.NURSERY;
        gc.nursery_pages_reclaimed += heap.NURSERY_PAGES;
        gc.nursery_cur = @intFromPtr(heap.pageToGcp(gc.nursery_first));
    }

    // C: gc.c:1726-1727 — remembered-set lifecycle: cleared at the end of
    // every scavenge (and on full-collect semi-space flip, see collect).
    gc.dirtyVectorsClear();
    gc.dirtyDefunsClear();

    // C: gc.c:1733 (opt-in stale/verify diagnostics omitted, plan DECISION 4).
    gc.in_scavenge = false;

    // SAFETY-ENFORCEMENT (unit A): auto-verify hook after a nursery scavenge.
    // Same mode gating as collect() — compiled out of ReleaseFast/ReleaseSmall.
    if (gc.opts.verify_collects and
        @import("builtin").mode != .ReleaseFast and
        @import("builtin").mode != .ReleaseSmall)
    {
        const e = debugVerifyHeap(gc, .post_scavenge);
        std.debug.assert(e == 0);
    }
}

// ---------------------------------------------------------------------
//  M3: test-only heap verification — plan DECISION 4 (replaces the ~1000
//  lines of opt-in C verifiers with ONE distilled checker called only from
//  tests; NOT a semantic part of the collector).
// ---------------------------------------------------------------------

/// Phase selector for debugVerifyHeap's live-pointer rule.
pub const VerifyPhase = enum {
    /// After a nursery scavenge: no scanned (live) object may hold a nursery
    /// pointer (distilled from vlive_check post_nursery=1 — M4 exercises it).
    post_scavenge,
    /// After a full collect: every live object sits in current_space, so no
    /// pointer field may point into the nursery or a dead/released page.
    post_collect,
};

/// Read a one-word pointer field through a `*const usize` view (same idiom as
/// scan.zig; all optional-pointer fields are exactly one word, null == 0).
inline fn ptrWord(field: anytype) usize {
    return @as(*const usize, @ptrCast(field)).*;
}

/// Check one pointer slot against the phase rule.  Returns 1 on violation.
fn verifyPtr(gc: *const Gc, addr: usize, phase: VerifyPhase) usize {
    if (addr == 0) return 0;
    const pg = heap.gcpToPage(addr);
    // Non-heap pointers (C-heap strings etc.) pass through gcMove unchanged
    // and are fine.
    if (pg < gc.firstheappage or pg > gc.lastheappage) return 0;
    const sp = gc.space[gc.md(pg)];
    return switch (phase) {
        // Live object's pointer must point into live old-gen.  Nursery,
        // dead from-space, and released pages are all stale post-collect.
        .post_collect => if (sp != gc.current_space) 1 else 0,
        // Scanned (live) objects must not hold nursery pointers.
        .post_scavenge => if (sp == heap.NURSERY) 1 else 0,
    };
}

/// Distilled vlive_check_value — check the pointer fields scanValue evacuates
/// (must mirror scan.zig exactly).
fn verifyValue(gc: *const Gc, v: *const types.Value, phase: VerifyPhase) usize {
    return switch (v.tag) {
        .cons => verifyPtr(gc, ptrWord(&v.payload.cons.car), phase) +
            verifyPtr(gc, ptrWord(&v.payload.cons.cdr), phase),
        .lambda => verifyPtr(gc, ptrWord(&v.payload.lambda.code), phase) +
            verifyPtr(gc, ptrWord(&v.payload.lambda.env), phase),
        .vector => verifyPtr(gc, ptrWord(&v.payload.vector.data), phase),
        .string => verifyPtr(gc, ptrWord(&v.payload.str.data), phase),
        .error_ => verifyPtr(gc, ptrWord(&v.payload.error_.message), phase),
        else => 0,
    };
}

/// Distilled vlive_object — dispatch by GC type tag (mirrors drainScanObject).
fn verifyObject(gc: *const Gc, body: [*]usize, ty: u32, hw: usize, phase: VerifyPhase) usize {
    var e: usize = 0;
    switch (ty) {
        0 => {}, // GC_TYPE_RAW — no pointers
        1 => e += verifyValue(gc, @ptrCast(@alignCast(body)), phase),
        2 => {
            const count = (hw - 1) * heap.WORDBYTES / @sizeOf(types.Value);
            const arr: [*]const types.Value = @ptrCast(@alignCast(body));
            var j: usize = 0;
            while (j < count) : (j += 1) e += verifyValue(gc, &arr[j], phase);
        },
        3 => {
            const count = (hw - 1) * heap.WORDBYTES / @sizeOf(types.Instr);
            const arr: [*]const types.Instr = @ptrCast(@alignCast(body));
            var j: usize = 0;
            while (j < count) : (j += 1) {
                e += verifyValue(gc, &arr[j].operand, phase);
                e += verifyPtr(gc, ptrWord(&arr[j].closure_code), phase);
            }
        },
        4 => {
            const count = (hw - 1) * heap.WORDBYTES / @sizeOf(types.CallFrame);
            const arr: [*]const types.CallFrame = @ptrCast(@alignCast(body));
            var j: usize = 0;
            while (j < count) : (j += 1) {
                e += verifyPtr(gc, ptrWord(&arr[j].code), phase);
                e += verifyPtr(gc, ptrWord(&arr[j].env), phase);
                e += verifyPtr(gc, ptrWord(&arr[j].stack.data), phase);
            }
        },
        else => {},
    }
    return e;
}

/// Test-only heap verification (plan DECISION 4): page invariants distilled
/// from gc_verify_heap (C: gc.c:886-935) plus the live-pointer check distilled
/// from vlive_check (C: gc.c:1425-1464).  Prints each violation; returns the
/// error count (tests assert 0).  Called ONLY from tests — never from the
/// collector.
pub fn debugVerifyHeap(gc: *const Gc, phase: VerifyPhase) usize {
    var errors: usize = 0;

    // ---- page invariants — C: gc.c:894-930 ----
    var pg = gc.firstheappage;
    while (pg <= gc.lastheappage) : (pg += 1) {
        const sp = gc.space[gc.md(pg)];
        const tp = gc.type_page[gc.md(pg)];

        // Invariant 1: a free page must be untagged (C: gc.c:899-905).
        if (sp == 0 and tp != 0) {
            std.debug.print("verify: free page {d} has type_page={d}\n", .{ pg, tp });
            errors += 1;
        }
        // Invariant 2: a CONTINUED page must not follow a free page (C: gc.c
        // :907-913).
        if (tp == heap.CONTINUED and pg > gc.firstheappage and
            gc.space[gc.md(pg - 1)] == 0)
        {
            std.debug.print("verify: CONTINUED page {d} follows a free page\n", .{pg});
            errors += 1;
        }
    }

    // ---- live-pointer check — walk live OBJECT pages in current_space with
    // the same cursor/break discipline as the drain (C: gc.c:1440-1457) ----
    var p2 = gc.firstheappage;
    while (p2 <= gc.lastheappage) : (p2 += 1) {
        if (gc.space[gc.md(p2)] != gc.current_space) continue;
        if (gc.type_page[gc.md(p2)] != heap.OBJECT) continue; // tail pages via head

        var cp = heap.pageToGcp(p2);
        // Stale retained from-space head (forwarded first word) — benign,
        // skip the page (C: gc.c:1447).
        if (types.forwarded(cp[0])) continue;
        while (heap.gcpToPage(@intFromPtr(cp)) == p2 and
            @intFromPtr(cp) != @intFromPtr(gc.freep))
        {
            const hdr = cp[0];
            const ty = types.headerType(hdr);
            // Same break as the drain (C: gc.c:1451-1453).
            if (types.forwarded(hdr) or types.headerWords(hdr) == 0 or
                ty > @intFromEnum(types.GcTypeTag.callframe_array)) break;
            errors += verifyObject(gc, cp + 1, ty, types.headerWords(hdr), phase);
            cp += types.headerWords(hdr);
        }
    }
    return errors;
}
