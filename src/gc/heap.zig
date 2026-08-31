//! src/gc/heap.zig — Gc struct state + memory-management core (milestone M1).
//!
//! C origin: gc.c statics (gc.c:56-96), gc_init (gc.c:2057-2153), the Cheney
//! queue (gc.c:435-462), gcalloc_internal (gc.c:1743-1794), grow_heap
//! (gc.c:1807-1846), allocatepage (gc.c:1854-1946), the public gc_alloc
//! nursery fast path (gc.c:2155-2277), gc_alloc_oldgen (gc.c:2283-2298),
//! nursery predicates (gc.c:331-342, 2300-2324), and the write-barrier
//! remembered sets (gc.c:357-421).
//!
//! Port contract (plan DECISION 4/5): every ported function carries a
//! `/// C: gc.c:NNNN name` doc comment — the reviewability anchor against
//! reference/shen-gc/gc.c.  All C statics become Gc struct fields with their
//! C names (snake_case, faithful).  Metadata arrays are NON-GC memory
//! (std.heap.page_allocator), indexed by (page - firstheappage) — the Zig
//! form of C's negative-base-pointer trick (gc.c:2114-2117).  C exit(1)
//! becomes std.debug.panic with the same message.
//!
//! The collector entry points (collect / collectNursery) live in
//! gc/collect.zig (milestones M3/M4); this file calls them through the
//! module import.  M1 ships loud panicking stubs for them so no allocation
//! path can silently skip a collection trigger.

const std = @import("std");
const types = @import("types.zig");
const collect_mod = @import("collect.zig");
const roots_mod = @import("roots.zig");
const scan_mod = @import("scan.zig");

// ---------------------------------------------------------------------
//  Constants — C: gc.c:28-54, 1798-1799; zincvm.h:38
// ---------------------------------------------------------------------

/// C: gc.c:28 PAGEBYTES.
pub const PAGEBYTES = 512;
/// C: gc.c:29 PAGEWORDS.
pub const PAGEWORDS = PAGEBYTES / @sizeOf(usize);
/// C: gc.c:30 WORDBYTES.
pub const WORDBYTES = @sizeOf(usize);

/// C: gc.c:40-41 page-kind tags for type_page[].
pub const OBJECT: usize = 0;
pub const CONTINUED: usize = 1;

/// C: gc.c:47 NURSERY — space[] tag for nursery pages (never 0/1/2, so
/// allocatepage's free-page scan skips them, gc.c:1893-1895).
pub const NURSERY: usize = 3;
/// C: gc.c:48 NURSERY_BYTES — fixed nursery at the start of the heap.  The C
/// 2 MB default makes ~3 KB of old-gen per-call churn (STACK_INIT_CAP=64) force
/// a nursery scavenge every few hundred calls; 8 MB amortises scavenges ~8x
/// (P0c).  The host heap must stay >= ~4x the nursery (old-gen semi-space
/// must fit the M10 frame pool + live set), so heaps below ~32 MB are too
/// tight for 8 MB.
pub const NURSERY_BYTES = 8 * 1024 * 1024;
/// C: gc.c:49 NURSERY_PAGES.
pub const NURSERY_PAGES = NURSERY_BYTES / PAGEBYTES;
/// C: gc.c:54 NURSERY_SCAVENGE_FREE_LOWATER — fire a pre-emptive nursery
/// scavenge when free nursery space drops to 1/8 of the region (87.5% full).
pub const NURSERY_SCAVENGE_FREE_LOWATER = NURSERY_BYTES / 8;

/// C: gc.c:1799 MIN_HEAP_PAGES — minimum heap size: 16 MB, never shrink below.
pub const MIN_HEAP_PAGES = 32768;
pub const MIN_HEAP_BYTES = MIN_HEAP_PAGES * PAGEBYTES;

/// C: gc.c:351 DIRTY_VECTORS_MAX — remembered-set capacity valve.
pub const DIRTY_VECTORS_MAX = 8192;

/// P2-10: the dirty-vectors dedup index capacity (open addressing wants
/// headroom, so 2x the array cap) and the linear-scan/hash handoff
/// threshold.  Below the threshold the linear scan is cache-friendlier (most
/// frames keep <10 distinct arrays post-P1 pooling); at/above it the
/// epoch-stamped open-addressing index makes dedup O(1).
pub const DIRTY_VECTORS_HASH_CAPACITY = 2 * DIRTY_VECTORS_MAX;
const DIRTY_VECTORS_HASH_THRESHOLD = 64;

/// C: zincvm.h:38 DEFUN_TABLE_CAP — dirty-defuns bitset is exactly this many
/// bits (4096 bits = 512 bytes = 64 x u64 words).
pub const DEFUN_TABLE_CAP = 4096;

// ---------------------------------------------------------------------
//  Public helper types
// ---------------------------------------------------------------------

/// Collection trigger labels (C passes const char *trigger strings; the Zig
/// port enumerates them — plan DECISION 5).  Values map to the C literals:
/// gc_alloc "THRESHOLD", allocatepage "LASTRESORT", gc_alloc "PREEMPTIVE"/
/// "REACTIVE", gc_alloc_oldgen "ALLOC", plus test-forced collections.
pub const Trigger = enum {
    @"test", // test-forced collection (keyword-escaped)
    threshold,
    lastresort,
    preemptive,
    reactive,
    alloc,
};

/// P2-10: one slot of the dirty-vectors dedup index.  `ptr` is the array
/// BASE address (0 = empty, no GC object lives at 0); `epoch` is the
/// generation stamp under which `ptr` was recorded — a clear bumps the
/// current epoch, so every slot from the previous epoch reads as stale/empty
/// and is overwritten lazily (O(1) clear, no rehash).
pub const HashSlot = extern struct {
    ptr: usize = 0,
    epoch: u64 = 0,
};

/// P2-10: index hash for an array base pointer.  Arrays are 8-byte-aligned
/// (page-aligned in practice), so the low 6 bits are constant noise — shift
/// them out, then avalanche the remaining bits (a splitmix64 finalizer) so
/// adjacent heap addresses spread uniformly across the table.
fn dirtyVectorHash(ptr: usize) u64 {
    var h: u64 = @intCast(ptr >> 6);
    h ^= h >> 33;
    h *%= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *%= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
}

/// C: gc_init(uintptr_t heap_size) + the gc_set_verbose flag.  heap_bytes
/// must be a multiple of PAGEBYTES (gc.h:32-34) and >= MIN_HEAP_BYTES;
/// reserve_bytes is the VAS reservation that grow_heap grows into (C: gc.c
/// :2064-2070 — max(heap*16, 4GB) by default; tests pass a small explicit
/// reserve to avoid 4GB VAS on constrained hosts).
pub const Options = struct {
    heap_bytes: usize = MIN_HEAP_BYTES,
    reserve_bytes: ?usize = null,
    verbose: bool = false,
    /// SAFETY-ENFORCEMENT: when true, collect() / collectNursery() run
    /// debugVerifyHeap after every collection and assert zero violations
    /// (see collect.zig).  The verify hook is compiled out in ReleaseFast /
    /// ReleaseSmall regardless of this flag, so it can never slow a
    /// production build.
    verify_collects: bool = false,
};

/// Snapshot of every instrumentation counter + derived predicate
/// (plan DECISION 1 stats()).  C origins noted per field.
pub const Stats = struct {
    nursery_scavenge_count: u64, // C: gc.c:91 gc_nursery_scavenge_count
    nursery_pages_reclaimed: u64, // C: gc.c:92 gc_nursery_pages_reclaimed
    preemptive_scavenge_count: u64, // C: gc.c:94
    reactive_scavenge_count: u64, // C: gc.c:95
    full_collect_count: u64, // C: gc.c:96
    allocated_pages: usize, // C: gc.c:2300 gc_allocatedpages
    nursery_is_empty: bool, // C: gc.c:2306 gc_nursery_is_empty
    nursery_capacity_pages: usize, // C: gc.c:2315
    nursery_no_other_space: bool, // C: gc.c:2319
    alloc_class_count: [5]u64, // C: gc.c:296
    dirty_vectors_fired: u64, // C: gc.c:385
    dirty_defuns_fired: u64, // C: gc.c:401
    dirty_defuns_scanned: u64, // C: gc.c:402
};

// ---------------------------------------------------------------------
//  Address <-> page conversions — C: gc.c:32-33
// ---------------------------------------------------------------------

/// C: gc.c:32 PAGE_to_GCP — page index -> word pointer at page start.
pub inline fn pageToGcp(pg: usize) [*]usize {
    return @ptrFromInt(pg * PAGEBYTES);
}

/// C: gc.c:33 GCP_to_PAGE — address -> page index (addr/512).
pub inline fn gcpToPage(addr: usize) usize {
    return addr / PAGEBYTES;
}

// ---------------------------------------------------------------------
//  The collector state — C: gc.c:56-96 statics as struct fields
// ---------------------------------------------------------------------

pub const Gc = struct {
    // ---- heap extent — C: gc.c:58-60 ----
    firstheappage: usize,
    lastheappage: usize,
    heappages: usize,

    // ---- old-gen bump allocator — C: gc.c:62-65 ----
    freewords: usize = 0,
    /// Bump cursor (C: gc.c:63 `uintptr_t *freep`).  Only meaningful while
    /// freewords != 0; allocatepage sets both together.  Initialised to the
    /// heap start so it is never a dangling value (C leaves it uninitialised;
    /// every dereference is guarded by freewords != 0).
    freep: [*]usize,
    allocatedpages: usize = 0,
    freepage: usize,

    // ---- nursery region + bump cursor — C: gc.c:69, 74-75 ----
    nursery_first: usize,
    nursery_last: usize,
    /// Address of the next free nursery byte (C: char *nursery_cur).
    nursery_cur: usize,
    /// One past the last nursery byte (C: char *nursery_end).
    nursery_end: usize,

    // ---- page metadata, NON-GC memory — C: gc.c:78-81 ----
    /// C: space[] — 0=free, 1=semi-space-1, 2=semi-space-2, NURSERY=3.
    space: []usize,
    /// C: gc_link[] — Cheney queue links.
    gc_link: []usize,
    /// C: type_page[] — OBJECT / CONTINUED.
    type_page: []usize,
    /// C: page_queued[] — 1 iff page currently in the Cheney queue (dedup).
    page_queued: []u8,

    // ---- Cheney queue — C: gc.c:83-87 ----
    queue_head: usize = 0,
    queue_tail: usize = 0,
    /// C: gc.c:85 in_scavenge — guard against recursive collection.
    in_scavenge: bool = false,
    current_space: usize,
    next_space: usize,

    // ---- mmap bookkeeping — C: gc.c:303, 308 (raw_heap_start/heap_mmap_size)
    raw_heap_start: usize,
    heap_mmap_size: usize,

    // ---- instrumentation counters — C: gc.c:91-96, 100, 296, 385, 401-402 ----
    collect_seq: u64 = 0, // C: gc.c:100 gc_collect_seq (banners, M3/M4)
    nursery_scavenge_count: u64 = 0,
    nursery_pages_reclaimed: u64 = 0,
    preemptive_scavenge_count: u64 = 0,
    reactive_scavenge_count: u64 = 0,
    full_collect_count: u64 = 0,
    /// C: gc.c:296 gc_alloc_class_count[5], indexed by GcTypeTag value.
    alloc_class_count: [5]u64 = [_]u64{0} ** 5,
    dirty_vectors_fired: u64 = 0,
    dirty_defuns_fired: u64 = 0,
    dirty_defuns_scanned: u64 = 0,

    // ---- write-barrier remembered set: dirty vectors — C: gc.c:352-355 ----
    dirty_vectors: [][*]types.Value = &.{},
    dirty_vectors_count: usize = 0,
    dirty_vectors_cap: usize = 0,
    dirty_vectors_overflow: bool = false,

    // ---- P2-10 dedup index (NEVER read by the collector: the ARRAY above
    // is the remembered-set invariant; this is a pure membership index kept
    // in lockstep with it).  Lazily allocated+filled on the first insert
    // that grows count past DIRTY_VECTORS_HASH_THRESHOLD; epoch stamping
    // makes dirtyVectorsClear an O(1) bump instead of a rehash.
    dirty_vectors_hash: ?[]HashSlot = null,
    dirty_vectors_epoch: u64 = 1,

    // ---- write-barrier remembered set: dirty defuns bitset — C: gc.c:400 ----
    dirty_defuns: [DEFUN_TABLE_CAP / 64]u64 = [_]u64{0} ** (DEFUN_TABLE_CAP / 64),

    // ---- precise-root shadow stack — C: gc.c:314-316 ----
    /// C: gc.c:314 shadow_stack (malloc'd, NON-GC memory; never scanned or
    /// evacuated by the collector).  The API surface + implementation live in
    /// gc/roots.zig (plan DECISION 5); Gc methods delegate there.
    shadow_stack: []roots_mod.GcRoot = &.{},
    /// C: gc.c:315 shadow_len.
    shadow_len: usize = 0,
    /// C: gc.c:316 shadow_cap (slice capacity — shadow_len entries in use).
    shadow_cap: usize = 0,

    // ---- typed-walker registrations — C: gc.c:320-325 ----
    /// C: gc.c:320 reg_global_table (the defun table).
    reg_global_table: ?[*]types.TableEntry = null,
    /// C: gc.c:321 reg_global_table_len.
    reg_global_table_len: ?*i32 = null,
    /// C: gc.c:322 reg_values_table.
    reg_values_table: ?[*]types.TableEntry = null,
    /// C: gc.c:323 reg_values_table_len.
    reg_values_table_len: ?*i32 = null,
    /// C: gc.c:324 reg_traced_code.
    reg_traced_code: ?[*]?*types.Instr = null,
    /// C: gc.c:325 reg_traced_code_len.
    reg_traced_code_len: ?*i32 = null,

    opts: Options = .{},

    // -----------------------------------------------------------------
    //  Metadata indexing — C: gc.c:2114-2117
    // -----------------------------------------------------------------

    /// C indexes the metadata arrays from negative base pointers
    /// (`space = space_ptr - firstheappage`); the Zig port subtracts the
    /// base per access instead.  `pg` is an absolute page index
    /// (addr/PAGEBYTES) in [firstheappage, lastheappage].
    pub inline fn md(self: *const Gc, pg: usize) usize {
        // SAFETY-ENFORCEMENT (unit E): page index must be within the heap
        // extent.  (pg < firstheappage would already panic on the usize
        // subtraction in safe modes; this also guards the upper bound.)
        std.debug.assert(pg >= self.firstheappage and pg <= self.lastheappage);
        return pg - self.firstheappage;
    }

    // -----------------------------------------------------------------
    //  init / deinit — C: gc.c:2057-2153 gc_init (no C counterpart for
    //  teardown; the raw_* pointers were kept "for eventual teardown",
    //  gc.c:302-308 — the port provides it).
    // -----------------------------------------------------------------

    /// C: gc.c:2057-2153 gc_init.
    /// mmap reservation (C: gc.c:2064-2089), metadata alloc (C: gc.c
    /// :2102-2129), nursery carve + bump-allocator init (C: gc.c:2131-2152).
    /// Differences from C (deliberate, per plan DECISION 4): init returns an
    /// error union instead of exit(1); heap_bytes < MIN_HEAP_BYTES or not a
    /// multiple of 512 is reported as error.InvalidHeapSize (C would slice
    /// metadata out of range and corrupt memory).
    pub fn init(opts: Options) !Gc {
        if (opts.heap_bytes == 0 or opts.heap_bytes % PAGEBYTES != 0)
            return error.InvalidHeapSize;
        if (opts.heap_bytes < MIN_HEAP_BYTES)
            return error.InvalidHeapSize;

        const page_count = opts.heap_bytes / PAGEBYTES;

        // C: gc.c:2068-2070 — reserve max(heap*16, 4GB) rounded up to a page.
        const four_gb: usize = 4096 * 1024 * 1024;
        const default_reserve = if (opts.heap_bytes * 16 > four_gb)
            opts.heap_bytes * 16 + PAGEBYTES - 1
        else
            four_gb + PAGEBYTES - 1;
        const reserve_raw = opts.reserve_bytes orelse default_reserve;
        const reserve = (reserve_raw + std.heap.page_size_min - 1) /
            std.heap.page_size_min * std.heap.page_size_min;

        // C: gc.c:2080-2089 mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON).
        const mapping = try std.posix.mmap(
            null,
            reserve,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        errdefer std.posix.munmap(mapping);

        // C: gc.c:2093-2096 — page-align the heap start to PAGEBYTES (mmap
        // already returns >= 4096-aligned, so this is normally a no-op).
        var heap_addr = @intFromPtr(mapping.ptr);
        if (heap_addr & (PAGEBYTES - 1) != 0)
            heap_addr += PAGEBYTES - (heap_addr & (PAGEBYTES - 1));

        const firstpage = gcpToPage(heap_addr);

        // C: gc.c:2102-2111 — calloc'd metadata arrays (NON-GC memory).
        const pa = std.heap.page_allocator;
        const space = try pa.alloc(usize, page_count);
        errdefer pa.free(space);
        const link = try pa.alloc(usize, page_count);
        errdefer pa.free(link);
        const typep = try pa.alloc(usize, page_count);
        errdefer pa.free(typep);
        const pq = try pa.alloc(u8, page_count);
        errdefer pa.free(pq);

        var g = Gc{
            .firstheappage = firstpage,
            .lastheappage = firstpage + page_count - 1,
            .heappages = page_count,
            .freep = pageToGcp(firstpage), // guarded by freewords==0 (C parity)
            .freepage = 0, // set below
            .nursery_first = firstpage,
            .nursery_last = firstpage + NURSERY_PAGES - 1,
            .nursery_cur = 0, // set below
            .nursery_end = 0, // set below
            .space = space,
            .gc_link = link,
            .type_page = typep,
            .page_queued = pq,
            .current_space = 1, // C: gc.c:2148
            .next_space = 1, // C: gc.c:2149
            .raw_heap_start = @intFromPtr(mapping.ptr),
            .heap_mmap_size = reserve,
            .opts = opts,
        };

        // C: gc.c:2124-2129 — zero all metadata (calloc parity).
        @memset(g.space, 0);
        @memset(g.gc_link, 0);
        @memset(g.type_page, 0);
        @memset(g.page_queued, 0);

        // C: gc.c:2136-2139 — carve the nursery at the heap start; pages
        // tagged NURSERY are never selected by allocatepage's free scan.
        var i = g.nursery_first;
        while (i <= g.nursery_last) : (i += 1)
            g.space[g.md(i)] = NURSERY;

        // C: gc.c:2145-2146 — nursery bump allocator: cur at region start,
        // end one past the last byte.
        g.nursery_cur = @intFromPtr(pageToGcp(g.nursery_first));
        g.nursery_end = @intFromPtr(pageToGcp(g.nursery_last + 1));

        // C: gc.c:2150-2152.
        g.freepage = g.firstheappage + NURSERY_PAGES; // start after nursery
        g.allocatedpages = 0;
        g.queue_head = 0;
        return g;
    }

    /// Release the mmap reservation and all NON-GC metadata (C: gc.c:302-308
    /// kept raw pointers "for eventual teardown"; the port provides it).
    pub fn deinit(self: *Gc) void {
        const mmap_slice = @as([*]align(std.heap.page_size_min) const u8, @ptrFromInt(self.raw_heap_start))[0..self.heap_mmap_size];
        std.posix.munmap(mmap_slice);
        const pa = std.heap.page_allocator;
        pa.free(self.space);
        pa.free(self.gc_link);
        pa.free(self.type_page);
        pa.free(self.page_queued);
        if (self.dirty_vectors.len > 0)
            pa.free(self.dirty_vectors);
        if (self.dirty_vectors_hash) |hs|
            pa.free(hs);
        // Shadow stack (page_allocator memory, grown on demand — roots.zig
        // shadowStackGrow; cap==0 means it was never allocated).
        if (self.shadow_cap != 0)
            pa.free(self.shadow_stack[0..self.shadow_cap]);
        self.* = undefined;
    }

    // -----------------------------------------------------------------
    //  Cheney queue — C: gc.c:435-462
    // -----------------------------------------------------------------

    /// C: gc.c:435-437 next_page.
    pub fn next_page(self: *const Gc, page: usize) usize {
        return if (page == self.lastheappage) self.firstheappage else page + 1;
    }

    /// C: gc.c:439-456 queue.  Dedup via page_queued is load-bearing: a page
    /// enqueued twice clobbers gc_link[P]=0, truncating the traversal and
    /// losing every page that follows P (duplicate enqueues happen when
    /// gc_move's "already to-space" branch re-queues a queued page).
    pub fn queue(self: *Gc, page: usize) void {
        if (self.page_queued[self.md(page)] != 0) return;
        self.page_queued[self.md(page)] = 1;
        if (self.queue_head != 0) {
            self.gc_link[self.md(self.queue_tail)] = page;
            self.gc_link[self.md(page)] = 0;
            self.queue_tail = page;
        } else {
            self.queue_head = page;
            self.gc_link[self.md(page)] = 0;
            self.queue_tail = page;
        }
    }

    /// C: gc.c:458-462 queue_reset.  The slice covers exactly
    /// [firstheappage, lastheappage], so a whole-slice memset is the C
    /// `memset(page_queued + firstheappage, 0, ...)` range.
    pub fn queue_reset(self: *Gc) void {
        self.queue_head = 0;
        self.queue_tail = 0;
        @memset(self.page_queued, 0);
    }

    // -----------------------------------------------------------------
    //  Old-gen collect trigger thresholds — C: gc.c:1849-1850
    // -----------------------------------------------------------------

    /// C: gc.c:1849 oldgen_collect_threshold — heappages/4, read live so
    /// grow_heap growth raises the threshold.
    pub fn oldgen_collect_threshold(self: *const Gc) usize {
        return self.heappages / 4;
    }

    /// C: gc.c:1850 oldgen_collect_lastresort — heappages/2.
    pub fn oldgen_collect_lastresort(self: *const Gc) usize {
        return self.heappages / 2;
    }

    // -----------------------------------------------------------------
    //  Heap growth — C: gc.c:1807-1846 grow_heap
    // -----------------------------------------------------------------

    /// C: gc.c:1807-1846 grow_heap — double the logical heap (or jump to
    /// min_needed), realloc the metadata, zero the new tail.  Pure
    /// bookkeeping within the VAS reservation.  Returns false where C
    /// returns -1 (OOM or reservation exhausted); prints the same message
    /// on reservation exhaustion (C: gc.c:1843-1844).
    /// (pub for M1 tests / future tuning; C keeps it static.)
    pub fn grow_heap(self: *Gc, pages_needed: usize) bool {
        var new_heappages = self.heappages * 2;
        var new_heap_size = new_heappages * PAGEBYTES;

        const min_needed = (self.allocatedpages + pages_needed + 512) * 2;
        if (new_heappages < min_needed) {
            new_heappages = min_needed;
            new_heap_size = new_heappages * PAGEBYTES;
        }

        // C: gc.c:1818 — fits within the mmap reservation: logical growth.
        if (new_heap_size + PAGEBYTES - 1 <= self.heap_mmap_size) {
            const pa = std.heap.page_allocator;
            // C: gc.c:1819-1823 realloc all four metadata arrays.  As in C,
            // a failure mid-way leaves the already-realloc'd arrays grown —
            // the caller only sees the failure and never uses the heap after
            // the ensuing panic/retry bookkeeping.
            const new_space = pa.realloc(self.space, new_heappages) catch return false;
            self.space = new_space;
            const new_link = pa.realloc(self.gc_link, new_heappages) catch return false;
            self.gc_link = new_link;
            const new_type = pa.realloc(self.type_page, new_heappages) catch return false;
            self.type_page = new_type;
            const new_pq = pa.realloc(self.page_queued, new_heappages) catch return false;
            self.page_queued = new_pq;

            // C: gc.c:1834-1839 — extend the extent and zero the new tail
            // (page_allocator.realloc leaves the new region uninitialised;
            // C's realloc has the same property and zeroes explicitly).
            const old_last = self.lastheappage;
            self.lastheappage = self.firstheappage + new_heappages - 1;
            self.heappages = new_heappages;
            var i = old_last + 1;
            while (i <= self.lastheappage) : (i += 1) {
                self.space[self.md(i)] = 0;
                self.gc_link[self.md(i)] = 0;
                self.type_page[self.md(i)] = 0;
                self.page_queued[self.md(i)] = 0;
            }
            return true;
        }

        // C: gc.c:1843-1845.
        std.debug.print(
            "[gc] grow_heap: need {d} MB but reservation is {d} MB\n",
            .{ new_heap_size / (1024 * 1024), self.heap_mmap_size / (1024 * 1024) },
        );
        return false;
    }

    // -----------------------------------------------------------------
    //  Page allocation — C: gc.c:1854-1946 allocatepage
    // -----------------------------------------------------------------

    /// C: gc.c:1854-1946 allocatepage — cyclic contiguous-free scan for
    /// `pages` consecutive pages not in current_space, next_space, or
    /// NURSERY.  Includes the LASTRESORT collect trigger (never re-entered
    /// mid-collection: guarded by current_space == next_space and
    /// !in_scavenge) and queue-on-allocate during a flip/scavenge so pages
    /// receiving promoted objects are drained.  (pub: called by
    /// gcalloc_internal; C keeps it static.)
    pub fn allocatepage(self: *Gc, pages: usize) void {
        var retried = false;

        // SAFETY-ENFORCEMENT (unit E): allocation size and running footprint
        // invariants that hold in every legitimate collector state.
        std.debug.assert(pages > 0);
        std.debug.assert(self.allocatedpages + pages <= self.heappages);

        // C: gc.c:1860 `retry:` — the label sits above the trigger block, so
        // a post-grow retry re-evaluates LASTRESORT against the grown heap.
        retry: while (true) {
            // C: gc.c:1867-1885 — LASTRESORT trigger.
            if (self.current_space == self.next_space and
                !self.in_scavenge and
                self.allocatedpages + pages >= self.oldgen_collect_lastresort())
            {
                collect_mod.collect(self, .lastresort);
                if (self.allocatedpages + pages >= self.oldgen_collect_lastresort()) {
                    if (!retried and self.grow_heap(pages)) {
                        retried = true;
                        continue :retry;
                    }
                    // C: gc.c:1876-1883.
                    std.debug.panic(
                        "gcalloc - Out of memory: need {d} pages, " ++
                            "live set is {d} pages " ++
                            "(semi-space capacity {d} pages)",
                        .{ pages, self.allocatedpages, self.heappages / 2 },
                    );
                }
            }

            // C: gc.c:1887-1932 — cyclic contiguous-free scan from freepage.
            var free_count: usize = 0;
            var firstpage: usize = 0;
            var allpages = self.heappages;

            while (allpages > 0) : (allpages -= 1) {
                const sp = self.space[self.md(self.freepage)];
                if (sp != self.current_space and sp != self.next_space and sp != NURSERY) {
                    // C: `if (free++ == 0) firstpage = freepage;`
                    if (free_count == 0) firstpage = self.freepage;
                    free_count += 1;

                    if (free_count == pages) {
                        // C: gc.c:1901.
                        self.freep = pageToGcp(firstpage);

                        // C: gc.c:1903-1904 — during a flip or nursery
                        // scavenge the new page receives promoted objects
                        // and must be drained by the Cheney queue.
                        if (self.current_space != self.next_space or self.in_scavenge)
                            self.queue(firstpage);

                        self.freewords = pages * PAGEWORDS;
                        self.allocatedpages += pages;
                        self.freepage = self.next_page(self.freepage);

                        self.space[self.md(firstpage)] = self.next_space;
                        self.type_page[self.md(firstpage)] = OBJECT;

                        // C: gc.c:1917-1923 `while (--pages)` — tag the
                        // remaining pages as CONTINUED.  `rest` never
                        // mutates `pages` (the C version consumes its
                        // parameter but returns immediately after).
                        var rest = pages;
                        while (rest > 1) : (rest -= 1) {
                            firstpage += 1;
                            self.space[self.md(firstpage)] = self.next_space;
                            self.type_page[self.md(firstpage)] = CONTINUED;
                        }
                        return;
                    }
                } else {
                    free_count = 0;
                }
                self.freepage = self.next_page(self.freepage);
                if (self.freepage == self.firstheappage)
                    free_count = 0; // C: gc.c:1930-1931 — wrapped; restart count
            }

            // C: gc.c:1937-1940 — scan exhausted: grow once and retry.
            if (!retried and self.grow_heap(pages)) {
                retried = true;
                continue :retry;
            }

            // C: gc.c:1942-1945.
            std.debug.panic(
                "gcalloc - Unable to allocate {d} pages in a {d} page heap",
                .{ pages, self.heappages },
            );
        }
    }

    // -----------------------------------------------------------------
    //  Internal bump allocator — C: gc.c:1743-1794 gcalloc_internal
    // -----------------------------------------------------------------

    /// C: gc.c:1743-1794 gcalloc_internal — allocate `bytes` with tag,
    /// writing a filler header over any partial page first.  May trigger
    /// collect() (via allocatepage's LASTRESORT) but never gc_alloc (no
    /// recursion).  Returns the body pointer past the header word.
    ///
    /// The multi-page branch is the classic silent-corruption site: an
    /// object of >= PAGEWORDS words must advance freep PAST the object with
    /// freewords = 0 (C: gc.c:1781-1791), so the Cheney drain's `cp != freep`
    /// guard can still scan the object's first page.
    /// (pub: move_internal in M2 calls it; C keeps it static.)
    pub fn gcalloc_internal(self: *Gc, bytes: usize, type_tag: types.GcTypeTag) [*]usize {
        // C: gc.c:1745 — 1 header word + ceil(bytes/WORDBYTES) body words.
        const words = (bytes + WORDBYTES - 1) / WORDBYTES + 1;

        // SAFETY-ENFORCEMENT (unit E): words is >= 1 for every input (0 bytes
        // yields a 1-word header-only object).
        std.debug.assert(words != 0);

        // C: gc.c:1747-1751 — words > 0xFFFFFF cannot fit the header.
        types.assertWordsFits(words, bytes);

        // C: gc.c:1753-1759 — finalize the partial page with a filler header
        // and take fresh contiguous pages.
        while (words > self.freewords) {
            if (self.freewords != 0) {
                self.freep[0] = types.makeHeader(self.freewords, .raw);
            }
            self.freewords = 0;
            self.allocatepage((words + PAGEWORDS - 1) / PAGEWORDS);
        }

        // C: gc.c:1762 — write the header.
        self.freep[0] = types.makeHeader(words, type_tag);

        // C: gc.c:1765 — zero the entire body.  P1: skip for RAW bodies —
        // every allocRaw/allocAtomic caller fully overwrites its body+NUL
        // before the bytes are ever read, and RAW is never scanned
        // (drainScanObject case 0 / verifyObject case 0).  Only the
        // <=7B word-padding tail stays dirty, which nothing reads.
        if (type_tag != .raw)
            @memset(self.freep[1..words], 0);

        const object = self.freep + 1;

        // C: gc.c:1778-1791.
        if (words < PAGEWORDS) {
            self.freewords -= words;
            self.freep += words;
        } else {
            // Multi-page object: advance freep past the object (see doc
            // comment above); freewords=0 forces the next alloc through
            // allocatepage, so allocation semantics are unchanged.
            self.freep += words;
            self.freewords = 0;
        }

        return object;
    }

    // -----------------------------------------------------------------
    //  Public allocation — C: gc.c:2155-2277 gc_alloc (+ gc_alloc_atomic
    //  gc.c:2279-2281, gc_alloc_oldgen gc.c:2283-2298)
    // -----------------------------------------------------------------

    /// C: gc.c:2155-2277 gc_alloc — public entry: per-class histogram,
    /// nursery fast path for single-page objects, old-gen THRESHOLD
    /// collect + anti-thrash grow, then gcalloc_internal.
    /// Marked noinline for C parity (spill caller registers — gc.c:2155).
    pub noinline fn gc_alloc(self: *Gc, bytes: usize, type_tag: types.GcTypeTag) [*]u8 {
        // C: gc.c:2160 — count at the public entry by class (the enum is
        // always in [0,4]; C guards against stray int tags).
        self.alloc_class_count[@intFromEnum(type_tag)] += 1;

        // C: gc.c:2171-2172 — nursery eligibility: bytes <= NURSERY_BYTES/8
        // AND the padded total (header + body, words) fits in ONE page.
        if (bytes <= NURSERY_BYTES / 8 and
            (((bytes + WORDBYTES - 1) / WORDBYTES + 1) * WORDBYTES) <= PAGEBYTES)
        {
            // SAFETY-ENFORCEMENT (unit E): the nursery bump cursor stays within
            // [nursery_first byte, nursery_end] in every legitimate state.
            std.debug.assert(self.nursery_cur >= @intFromPtr(pageToGcp(self.nursery_first)));
            std.debug.assert(self.nursery_cur <= self.nursery_end);

            // C: gc.c:2173-2175.
            const words = (bytes + WORDBYTES - 1) / WORDBYTES + 1;
            const total = words * WORDBYTES;
            var nursery_tried = false;

            // C: gc.c:2180-2185 — pre-emptive low-water scavenge BEFORE the
            // bump cursor exhausts the nursery (decoupled from reactive).
            if (!self.in_scavenge and
                (self.nursery_end - self.nursery_cur) <= NURSERY_SCAVENGE_FREE_LOWATER)
            {
                collect_mod.collectNursery(self, .preemptive);
                nursery_tried = true;
                self.preemptive_scavenge_count += 1;
            }

            // C: gc.c:2187 `nursery_retry:` — both the straddle skip and the
            // reactive retry re-enter here (above the fit check, below the
            // pre-emptive trigger).
            nursery_retry: while (true) {
                // C: gc.c:2203-2210 — no-straddle guard: a single-page
                // object must not cross a page boundary, else a later
                // object sharing its tail page aborts the drain's page walk
                // on the misread zero-padding "header" (hw == 0 break).
                if (total <= PAGEBYTES and self.nursery_cur < self.nursery_end) {
                    const s_addr = self.nursery_cur;
                    const e_addr = s_addr + total;
                    if (gcpToPage(e_addr - 1) != gcpToPage(s_addr)) {
                        self.nursery_cur = @intFromPtr(pageToGcp(gcpToPage(s_addr) + 1));
                        continue :nursery_retry;
                    }
                }

                // C: gc.c:2212-2244 — bump-allocate if it fits.
                if (self.nursery_end - self.nursery_cur >= total) {
                    const header: *usize = @ptrFromInt(self.nursery_cur);
                    header.* = types.makeHeader(words, type_tag);

                    // C: gc.c:2217 — zero the body.  P1: skip for RAW bodies
                    // (same rationale as the old-gen path above).
                    const body: [*]usize = @ptrFromInt(self.nursery_cur + WORDBYTES);
                    if (type_tag != .raw)
                        @memset(body[0 .. words - 1], 0);

                    self.nursery_cur += total;

                    // C: gc.c:2235-2241 — type_page markers.  Nursery
                    // objects are single-page, so the CONTINUED loop never
                    // runs; kept for metadata consistency.
                    const first_page = gcpToPage(@intFromPtr(header));
                    const last_page = gcpToPage(self.nursery_cur - 1);
                    self.type_page[self.md(first_page)] = OBJECT;
                    var pg = first_page + 1;
                    while (pg <= last_page) : (pg += 1)
                        self.type_page[self.md(pg)] = CONTINUED;

                    return @ptrCast(body);
                }

                // C: gc.c:2247-2252 — nursery full: collect reactively and
                // retry once (never twice — PREEMPTIVE may have consumed
                // the single try already).
                if (!nursery_tried) {
                    collect_mod.collectNursery(self, .reactive);
                    nursery_tried = true;
                    self.reactive_scavenge_count += 1;
                    continue :nursery_retry;
                }

                // C: gc.c:2254 — no nursery space: fall through to old-gen.
                break;
            }
        }

        // C: gc.c:2262-2274 — old-gen THRESHOLD collect + anti-thrash grow
        // (if the LIVE set still sits above the threshold after collecting,
        // grow so the threshold rises above the live set).
        if (self.allocatedpages > 0 and
            self.allocatedpages > self.oldgen_collect_threshold() and
            !self.in_scavenge)
        {
            collect_mod.collect(self, .threshold);
            if (self.allocatedpages > self.oldgen_collect_threshold())
                _ = self.grow_heap(1);
        }

        // C: gc.c:2276.
        return @ptrCast(self.gcalloc_internal(bytes, type_tag));
    }

    /// C: gc.c:2283-2298 gc_alloc_oldgen — bypass the nursery entirely; used
    /// for large objects that would fragment it.  Same THRESHOLD trigger +
    /// anti-thrash grow as gc_alloc's old-gen path (label "ALLOC").
    pub noinline fn gc_alloc_oldgen(self: *Gc, bytes: usize, type_tag: types.GcTypeTag) [*]u8 {
        // C: gc.c:2286.
        self.alloc_class_count[@intFromEnum(type_tag)] += 1;

        if (self.allocatedpages > 0 and
            self.allocatedpages > self.oldgen_collect_threshold() and
            !self.in_scavenge)
        {
            collect_mod.collect(self, .alloc);
            if (self.allocatedpages > self.oldgen_collect_threshold())
                _ = self.grow_heap(1);
        }

        return @ptrCast(self.gcalloc_internal(bytes, type_tag));
    }

    // -----------------------------------------------------------------
    //  Typed allocation wrappers (Zig-native ergonomics over the C entry
    //  points — plan DECISION 1).  All marked noinline (C parity).
    // -----------------------------------------------------------------

    /// Comptime tag mapping (plan DECISION 1): a single Value scans by tag;
    /// arrays scan per element class; everything else is raw.  Note C never
    /// allocates a single Instr — a lone non-Value object is raw by design.
    pub fn tagFor(comptime T: type) types.GcTypeTag {
        return comptime switch (@typeInfo(T)) {
            .pointer => |p| if (p.size == .slice) arrayTagFor(p.child) else singleTagFor(T),
            else => singleTagFor(T),
        };
    }

    fn singleTagFor(comptime T: type) types.GcTypeTag {
        return if (T == types.Value) .value else .raw;
    }

    fn arrayTagFor(comptime Elem: type) types.GcTypeTag {
        return switch (Elem) {
            types.Value => .value_array,
            types.Instr => .instr_array,
            types.CallFrame => .callframe_array,
            else => .raw,
        };
    }

    /// Allocate one zeroed T.  C: gc_alloc(sizeof(T), tagFor(T)).
    pub noinline fn alloc(self: *Gc, comptime T: type) *T {
        const body = self.gc_alloc(@sizeOf(T), comptime tagFor(T));
        return @ptrCast(@alignCast(body));
    }

    /// Allocate n zeroed Ts.  C: gc_alloc(n * sizeof(T), tag-array).
    pub noinline fn allocArray(self: *Gc, comptime T: type, n: usize) [*]T {
        const body = self.gc_alloc(n * @sizeOf(T), comptime tagFor([]T));
        return @ptrCast(@alignCast(body));
    }

    /// Allocate zeroed unscanned bytes.  C: gc_alloc(bytes, GC_TYPE_RAW).
    pub noinline fn allocRaw(self: *Gc, bytes: usize) [*]u8 {
        return self.gc_alloc(bytes, .raw);
    }

    /// C: gc.c:2279-2281 gc_alloc_atomic — alias for allocRaw.
    pub noinline fn allocAtomic(self: *Gc, bytes: usize) [*]u8 {
        return self.gc_alloc(bytes, .raw);
    }

    /// Typed old-gen allocation bypassing the nursery.  C: gc.c:2283-2298
    /// gc_alloc_oldgen.  (T is the ELEMENT type of the array.)
    pub noinline fn allocArrayOldgen(self: *Gc, comptime T: type, n: usize) [*]T {
        const body = self.gc_alloc_oldgen(n * @sizeOf(T), comptime arrayTagFor(T));
        return @ptrCast(@alignCast(body));
    }

    /// Bytes-only old-gen allocation with an explicit tag.
    /// C: gc.c:2283-2298 gc_alloc_oldgen.
    pub noinline fn allocOldgen(self: *Gc, bytes: usize, tag: types.GcTypeTag) [*]u8 {
        return self.gc_alloc_oldgen(bytes, tag);
    }

    // -----------------------------------------------------------------
    //  Nursery / old-gen predicates — C: gc.c:331-342, 2300-2324
    // -----------------------------------------------------------------

    /// C: gc.c:331-334 gc_in_nursery — page-index range test against the
    /// nursery region.
    pub fn inNursery(self: *const Gc, addr: usize) bool {
        const page = gcpToPage(addr);
        return page >= self.nursery_first and page <= self.nursery_last;
    }

    /// C: gc.c:338-342 gc_in_oldgen — SPACE TAG test (space == current_space),
    /// NOT an address-range test.  A nursery-page address fails even though
    /// it is inside the heap; a from-space page also fails.
    pub fn inOldgen(self: *const Gc, addr: usize) bool {
        const page = gcpToPage(addr);
        return page >= self.firstheappage and page <= self.lastheappage and
            self.space[self.md(page)] == self.current_space;
    }

    /// C: gc.c:2300-2302 gc_allocatedpages.
    pub fn allocatedPages(self: *const Gc) usize {
        return self.allocatedpages;
    }

    /// C: gc.c:2306-2313 gc_nursery_is_empty — true iff every nursery page
    /// is NURSERY-tagged (checks page tags only; nursery_cur may have
    /// advanced past region start from post-scavenge allocations).
    pub fn nurseryIsEmpty(self: *const Gc) bool {
        var pg = self.nursery_first;
        while (pg <= self.nursery_last) : (pg += 1) {
            if (self.space[self.md(pg)] != NURSERY) return false;
        }
        return true;
    }

    /// C: gc.c:2315-2317 gc_nursery_capacity_pages.
    pub fn nurseryCapacityPages(self: *const Gc) usize {
        _ = self; // C reads the global NURSERY_PAGES constant
        return NURSERY_PAGES;
    }

    /// C: gc.c:2319-2324 gc_nursery_no_other_space — no nursery page carries
    /// the OTHER semi-space tag.
    pub fn nurseryNoOtherSpace(self: *const Gc) bool {
        const other: usize = if (self.current_space == 1) 2 else 1;
        var pg = self.nursery_first;
        while (pg <= self.nursery_last) : (pg += 1) {
            if (self.space[self.md(pg)] == other) return false;
        }
        return true;
    }

    // -----------------------------------------------------------------
    //  Write-barrier remembered sets — C: gc.c:357-421
    // -----------------------------------------------------------------

    /// C: gc.c:357-374 gc_dirty_vectors_add — record an old-gen vector
    /// element array that may now hold nursery pointers.  Dedup by pointer,
    /// capacity-capped at DIRTY_VECTORS_MAX with an overflow valve (on
    /// overflow the nursery scavenge falls back to a full old-gen
    /// OBJECT-page queue — M4).
    ///
    /// P2-10: dedup uses an epoch-stamped open-addressing index once the
    /// distinct-array count passes DIRTY_VECTORS_HASH_THRESHOLD (the O(n)
    /// linear scan stays below it, where it is cache-friendlier).  The ARRAY
    /// is unchanged — its contents, order, growth and overflow valve are
    /// byte-identical to the C original and remain the remembered-set
    /// invariant collect.zig reads; the hash is a pure membership index kept
    /// in lockstep with it (set contents == dirty_vectors[0..count] at all
    /// times).
    pub fn dirtyVectorsAdd(self: *Gc, data: [*]types.Value) void {
        if (self.dirty_vectors_overflow) return; // C: gc.c:358

        if (self.dirty_vectors_hash) |hs| {
            if (self.dirtyVectorHashProbe(hs, @intFromPtr(data))) return;
        } else {
            var i: usize = 0; // C: gc.c:359-360 — dedup scan.
            while (i < self.dirty_vectors_count) : (i += 1) {
                if (self.dirty_vectors[i] == data) return;
            }
        }

        // C: gc.c:361-364 — overflow valve.
        if (self.dirty_vectors_count >= DIRTY_VECTORS_MAX) {
            self.dirty_vectors_overflow = true;
            return;
        }

        // C: gc.c:365-371 — geometric growth 256 -> ... -> DIRTY_VECTORS_MAX.
        if (self.dirty_vectors_count >= self.dirty_vectors_cap) {
            var nc: usize = if (self.dirty_vectors_cap != 0) self.dirty_vectors_cap * 2 else 256;
            if (nc > DIRTY_VECTORS_MAX) nc = DIRTY_VECTORS_MAX;
            const np = std.heap.page_allocator.realloc(self.dirty_vectors, nc) catch {
                self.dirty_vectors_overflow = true; // C: gc.c:369
                return;
            };
            self.dirty_vectors = np;
            self.dirty_vectors_cap = nc;
        }

        self.dirty_vectors[self.dirty_vectors_count] = data;
        self.dirty_vectors_count += 1;
        self.dirty_vectors_fired += 1; // C: gc.c:373

        // P2-10: keep the index in lockstep with the array.  The lazy
        // allocation fires on the first insert that grows count PAST the
        // threshold and populates the index from the whole array (so the
        // first THRESHOLD entries are present too, not just the new one).
        // On alloc failure the hash stays null and the linear scan keeps
        // serving — still correct, just O(n).
        if (self.dirty_vectors_hash) |hs| {
            self.dirtyVectorHashPut(hs, @intFromPtr(data));
        } else if (self.dirty_vectors_count > DIRTY_VECTORS_HASH_THRESHOLD) {
            _ = self.dirtyVectorHashAllocFill();
        }
    }

    /// C: gc.c:376-379 gc_dirty_vectors_clear — end of each nursery scavenge
    /// and on full-collect semi-space flip.  P2-10: the epoch bump is the
    /// O(1) index clear — every previous-epoch slot reads stale and is
    /// overwritten lazily, no rehash.
    pub fn dirtyVectorsClear(self: *Gc) void {
        self.dirty_vectors_count = 0;
        self.dirty_vectors_overflow = false;
        self.dirty_vectors_epoch += 1;
    }

    /// P2-10: true iff `ptr` is present under the current epoch.  A slot
    /// whose epoch does not match is stale/empty — it ends the probe cluster
    /// (open addressing inserts only into the current epoch, so a live run
    /// of current-epoch slots is never split by a stale one).
    fn dirtyVectorHashProbe(self: *const Gc, hs: []const HashSlot, ptr: usize) bool {
        const cap = hs.len;
        var idx: usize = @intCast(dirtyVectorHash(ptr) % cap);
        var n: usize = 0;
        while (n < cap) : (n += 1) {
            const slot = hs[idx];
            if (slot.epoch != self.dirty_vectors_epoch) return false;
            if (slot.ptr == ptr) return true;
            idx += 1;
            if (idx == cap) idx = 0;
        }
        return false; // unreachable: cap is 2x DIRTY_VECTORS_MAX, count <= cap/2
    }

    /// P2-10: record `ptr` under the current epoch (caller has already proven
    /// absence via dirtyVectorHashProbe).  The table can never fill — at most
    /// DIRTY_VECTORS_MAX entries are ever inserted into twice that capacity.
    fn dirtyVectorHashPut(self: *const Gc, hs: []HashSlot, ptr: usize) void {
        const cap = hs.len;
        var idx: usize = @intCast(dirtyVectorHash(ptr) % cap);
        while (true) {
            const slot = hs[idx];
            if (slot.epoch != self.dirty_vectors_epoch or slot.ptr == 0) {
                hs[idx].ptr = ptr;
                hs[idx].epoch = self.dirty_vectors_epoch;
                return;
            }
            if (slot.ptr == ptr) return; // already present (shouldn't happen)
            idx += 1;
            if (idx == cap) idx = 0;
        }
    }

    /// P2-10: allocate the 16384-slot index and populate it from the current
    /// array contents [0..count] (which already includes the just-appended
    /// entry at the transition).  Returns false on alloc failure; the caller
    /// then keeps using the linear scan.
    fn dirtyVectorHashAllocFill(self: *Gc) bool {
        if (self.dirty_vectors_hash != null) return true;
        const hs = std.heap.page_allocator.alloc(HashSlot, DIRTY_VECTORS_HASH_CAPACITY) catch
            return false;
        @memset(hs, std.mem.zeroes(HashSlot));
        for (self.dirty_vectors[0..self.dirty_vectors_count]) |dv| {
            self.dirtyVectorHashPut(hs, @intFromPtr(dv));
        }
        self.dirty_vectors_hash = hs;
        return true;
    }

    /// C: gc.c:404-412 gc_dirty_defuns_mark — bit i of the fixed bitset;
    /// out-of-range indices are ignored (C parity).
    pub fn dirtyDefunsMark(self: *Gc, idx: i32) void {
        if (idx < 0 or idx >= DEFUN_TABLE_CAP) return;
        const i: usize = @intCast(idx);
        const word = i / 64;
        const mask = @as(u64, 1) << @intCast(i % 64);
        if ((self.dirty_defuns[word] & mask) == 0) {
            self.dirty_defuns[word] |= mask;
            self.dirty_defuns_fired += 1;
        }
    }

    /// C: gc.c:414-417 gc_dirty_defuns_test.
    pub fn dirtyDefunsTest(self: *const Gc, idx: i32) bool {
        if (idx < 0 or idx >= DEFUN_TABLE_CAP) return false;
        const i: usize = @intCast(idx);
        return (self.dirty_defuns[i / 64] >> @intCast(i % 64)) & 1 != 0;
    }

    /// C: gc.c:419-421 gc_dirty_defuns_clear — same lifecycle as
    /// dirtyVectorsClear.
    pub fn dirtyDefunsClear(self: *Gc) void {
        @memset(&self.dirty_defuns, 0);
    }

    /// C: zincvm.c:912 — write-barrier site 1 (gc.md "Barrier site 1"): the
    /// `address->` vector-element store.  Stores `val` into `data[i]`, then
    /// records the element array in the remembered set iff BOTH:
    ///   - `data` lives in old-gen, by the SPACE TAG test (gc_in_oldgen,
    ///     gc.c:338-342 — NOT an address-range test: an address-range
    ///     version silently misses arrays promoted into nursery-range pages
    ///     and wrongly includes dead from-space pages — the C Test 6
    ///     rationale), and
    ///   - the stored Value references any nursery object
    ///     (value_references_nursery, zincvm.c:112-121 — mirrors exactly the
    ///     fields gc_scan_value evacuates).
    ///
    /// The barrier records the array BASE `data` (sufficient: collectNursery
    /// scans the whole element array, C: gc.c:1675-1686).  Nursery-resident
    /// arrays take no barrier — the next scavenge evacuates them wholesale.
    /// Callers holding a nullable `?[*]Value` must null-check before calling
    /// (the C site's `vec.vector.data &&` guard, gc.md site 1).
    pub fn writeBarrierVectorStore(
        self: *Gc,
        data: [*]types.Value,
        i: usize,
        val: types.Value,
    ) void {
        data[i] = val;
        // C checks `&val` (the stored copy); `&data[i]` is now that copy and
        // valueReferencesNursery is read-only — identical behaviour.
        if (self.inOldgen(@intFromPtr(data)) and
            scan_mod.valueReferencesNursery(self, &data[i]))
        {
            self.dirtyVectorsAdd(data);
        }
    }

    // -----------------------------------------------------------------
    //  Stats — plan DECISION 1 stats()
    // -----------------------------------------------------------------

    /// Snapshot of all counters + derived predicates (see Stats fields for
    /// C origins).
    pub fn stats(self: *const Gc) Stats {
        return .{
            .nursery_scavenge_count = self.nursery_scavenge_count,
            .nursery_pages_reclaimed = self.nursery_pages_reclaimed,
            .preemptive_scavenge_count = self.preemptive_scavenge_count,
            .reactive_scavenge_count = self.reactive_scavenge_count,
            .full_collect_count = self.full_collect_count,
            .allocated_pages = self.allocatedpages,
            .nursery_is_empty = self.nurseryIsEmpty(),
            .nursery_capacity_pages = self.nurseryCapacityPages(),
            .nursery_no_other_space = self.nurseryNoOtherSpace(),
            .alloc_class_count = self.alloc_class_count,
            .dirty_vectors_fired = self.dirty_vectors_fired,
            .dirty_defuns_fired = self.dirty_defuns_fired,
            .dirty_defuns_scanned = self.dirty_defuns_scanned,
        };
    }

    /// C: gc.c:604-746 collect — full semi-space collection (implementation
    /// in gc/collect.zig; pub so tests force it deterministically, plan
    /// DECISION 1).
    pub fn collect(self: *Gc, trigger: Trigger) void {
        collect_mod.collect(self, trigger);
    }

    /// C: gc.c:1626-1735 collect_nursery — nursery scavenge (gc/collect.zig;
    /// M4 implements it, the stub panics loudly until then).
    pub fn collectNursery(self: *Gc, trigger: Trigger) void {
        collect_mod.collectNursery(self, trigger);
    }

    // -----------------------------------------------------------------
    //  Precise-root API — thin wrappers over gc/roots.zig (plan DECISION 1
    //  ergonomics: `gc.rootPushValue(&v)`; the implementation + C-line docs
    //  live in roots.zig, C: gc.c:2340-2402).
    // -----------------------------------------------------------------

    /// C: gc.c:2340 gc_root_push_ptr — slot must point at an object HEAD.
    pub fn rootPushPtr(self: *Gc, slot: *anyopaque) void {
        roots_mod.rootPushPtr(self, slot);
    }

    /// C: gc.c:2348 gc_root_push_value.
    pub fn rootPushValue(self: *Gc, vslot: *types.Value) void {
        roots_mod.rootPushValue(self, vslot);
    }

    /// C: gc.c:2356 gc_root_push_value_volatile.
    pub fn rootPushValueVolatile(self: *Gc, vslot: *volatile types.Value) void {
        roots_mod.rootPushValueVolatile(self, vslot);
    }

    /// C: gc.c:2364 gc_root_push_value_array.
    pub fn rootPushValueArray(self: *Gc, base: [*]types.Value, np: *i32) void {
        roots_mod.rootPushValueArray(self, base, np);
    }

    /// C: gc.c:2399 gc_root_push_callframe_array.
    pub fn rootPushCallframeArray(self: *Gc, arr: [*]types.CallFrame, np: *i32) void {
        roots_mod.rootPushCallframeArray(self, arr, np);
    }

    /// C: gc.c:2372 gc_root_pop.
    pub fn rootPop(self: *Gc) void {
        roots_mod.rootPop(self);
    }

    /// C: gc.c:2376 gc_root_pop_to.
    pub fn rootPopTo(self: *Gc, watermark: usize) void {
        roots_mod.rootPopTo(self, watermark);
    }

    /// C: gc.c:2380 gc_root_watermark.
    pub fn rootWatermark(self: *const Gc) usize {
        return roots_mod.rootWatermark(self);
    }

    // ---- RAII root guards (unit D) ----
    // Each pushes the corresponding root and returns a RootGuard whose end()
    // pops it: `var g = gc.rootValue(&v); defer g.end();`.  Balance is
    // automatic and multiple guards unwind LIFO (correct).  Names intentionally
    // avoid the raw rootPush* / rootPop surface above.

    /// SAFETY-ENFORCEMENT (unit D): RAII guard for a by-value Value root.
    pub fn rootValue(self: *Gc, vslot: *types.Value) roots_mod.RootGuard {
        self.rootPushValue(vslot);
        return .{ .gc = self };
    }

    /// SAFETY-ENFORCEMENT (unit D): RAII guard for a ROOT_PTR slot.
    pub fn rootPtr(self: *Gc, slot: *anyopaque) roots_mod.RootGuard {
        self.rootPushPtr(slot);
        return .{ .gc = self };
    }

    /// SAFETY-ENFORCEMENT (unit D): RAII guard for a volatile Value root.
    pub fn rootValueVolatile(self: *Gc, vslot: *volatile types.Value) roots_mod.RootGuard {
        self.rootPushValueVolatile(vslot);
        return .{ .gc = self };
    }

    /// SAFETY-ENFORCEMENT (unit D): RAII guard for a Value array root.
    pub fn rootValueArray(self: *Gc, base: [*]types.Value, np: *i32) roots_mod.RootGuard {
        self.rootPushValueArray(base, np);
        return .{ .gc = self };
    }

    /// SAFETY-ENFORCEMENT (unit D): RAII guard for a CallFrame array root.
    pub fn rootCallframeArray(self: *Gc, arr: [*]types.CallFrame, np: *i32) roots_mod.RootGuard {
        self.rootPushCallframeArray(arr, np);
        return .{ .gc = self };
    }

    /// C: gc.c:2384 gc_register_global_table (defun table).
    pub fn registerGlobalTable(self: *Gc, table: [*]types.TableEntry, len_p: *i32) void {
        roots_mod.registerGlobalTable(self, table, len_p);
    }

    /// C: gc.c:2389 gc_register_values_table.
    pub fn registerValuesTable(self: *Gc, table: [*]types.TableEntry, len_p: *i32) void {
        roots_mod.registerValuesTable(self, table, len_p);
    }

    /// C: gc.c:2394 gc_register_traced_code.
    pub fn registerTracedCode(self: *Gc, arr: [*]?*types.Instr, np: *i32) void {
        roots_mod.registerTracedCode(self, arr, np);
    }
};
