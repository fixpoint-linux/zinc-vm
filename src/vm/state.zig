//! src/vm/state.zig — the Vm struct (M0 skeleton + M1 interner + M2 tables
//! + M4 DECISION-A error model).
//!
//! C origin: the VM's global interpreter state (zincvm.c statics) gathered
//! into one struct.  M0 adds the skeleton: owns `*Gc`, and roots `err_slot`
//! ONCE at init (this is the C "S3 cf.error_val rooting handled once" —
//! plan DECISION A).  M1 adds the symbol interner.  M2 adds the defun/values
//! global tables (tables.zig) plus their GC registration and the initGlobals
//! stub.  M4 adds the CatchSite chain + VmError + throwShen + instr_limit
//! (see interp.zig).
//!
//! err_slot is rooted via rootPushValue at init and popped at deinit, so it
//! never needs re-rooting: every ShenError message built by a future throw
//! writes into this permanently-rooted slot.  NOTE: the root holds the ADDRESS
//! of `vm.err_slot`, so the Vm must not be moved/copied after init (tests keep
//! it in one local; later milestones heap-allocate it).  The same address
//! stability requirement covers `&vm.defun_table_cap` / `&vm.values_table_cap`,
//! which the GC reads at scan time.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const symbols = @import("symbols.zig");
const tables = @import("tables.zig");
const values = @import("values.zig");
const prims = @import("prims.zig");
const streams = @import("streams.zig");
const parser = @import("parser.zig");

const Gc = gc.Gc;

/// Plan DECISION A: C's setjmp/longjmp CatchFrame chain (zincvm.c:706-720
/// vm_catch_chain / vm_throw) becomes Zig error unions plus this linked
/// chain of stack-allocated CatchSites.  error.Halt is the C
/// `exec_primitive() < 0` hard stop (non-catchable, acc preserved);
/// error.ShenError is the longjmp (catchable at a CatchSite).
pub const VmError = error{ ShenError, Halt };

/// M10 frame-stack pool: max idle old-gen CALLFRAME_ARRAYs held for reuse
/// across vmExecEnv entries (see interp.zig frameStackAcquire/Release).
/// Bounded to real vmExecEnv nesting depth — outer entry + trap-error body,
/// or outer entry + N>A peel = 2 (a trap-error body is sequential with its
/// handler, so trap-error alone never nests past 2; the depth-3 case of a
/// peel inside a trap-error body degrades gracefully by dropping one array).
/// Retaining more than the nesting depth never pays off (the LIFO free-list
/// can only hand them back at that depth) and pins ~3 MB per extra array:
/// at 3 idle arrays (9 MB) the base live set exceeds the grown 32 MB heap's
/// old-gen threshold (8 MB), forcing a failed grow_heap per old-gen alloc on
/// reserve-constrained heaps.  2 x 3 MB = 6 MB stays under the threshold.
pub const FRAME_POOL_MAX: usize = 2;

/// C: zincvm.h CatchFrame — DECISION A shape.  Stack-allocated at each
/// catch site (trap-error in M5, the host harness): push by setting
/// `.parent = vm.catch_chain; vm.catch_chain = &site;` and restore the
/// parent on exit.  `in_trap_error` gates the throw-vs-hard-stop routing at
/// the CONDITIONAL throw sites (env_pop, apply/appterm non-callable).  The
/// C per-site `error_val` field is replaced by the ONCE-rooted vm.err_slot.
pub const CatchSite = struct {
    in_trap_error: bool = false,
    parent: ?*CatchSite = null,
};

pub const Vm = struct {
    /// Owned *Gc — the collector this VM allocates from.
    gc: *Gc,
    /// Rooted ONCE at init (plan DECISION A): the ShenError value slot.
    err_slot: types.Value,
    /// Dynamic symbol interner (M1).
    symbols: symbols.SymbolInterner,
    /// Defun global table (M2): page_allocator zeroed array OUTSIDE the GC heap
    /// (C BSS parity), registered via gc.registerGlobalTable.  [ * ]Types.TableEntry
    /// pointing at DEFUN_TABLE_CAP entries; the cap field is the GC's length
    /// register (sizes the dirty-defuns bitset).
    defun_table: [*]types.TableEntry = undefined,
    /// C: zincvm.c:479 defun_table_cap — read by the GC at scan time.
    defun_table_cap: i32 = @intCast(tables.DEFUN_TABLE_CAP),
    /// Values global table (M2): page_allocator zeroed array, registered via
    /// gc.registerValuesTable (always full-scanned).
    values_table: [*]types.TableEntry = undefined,
    /// C: zincvm.c:480 values_table_cap.
    values_table_cap: i32 = @intCast(tables.VALUES_TABLE_CAP),
    /// C: zincvm.c:706 vm_catch_chain — head of the stack-allocated
    /// CatchSite chain (DECISION A; null outside catch regions).
    catch_chain: ?*CatchSite = null,
    /// C: zincvm.c:3146-3153 get_instr_limit — hard instruction budget,
    /// default 5e9.  C caches the $ZINCVM_INSTR_LIMIT env override per
    /// vm_exec_env entry; the port keeps the constant default and exposes
    /// the field for the host to set directly (init comment).
    instr_limit: u64 = 5_000_000_000,
    /// Cumulative instructions executed across all vmExec calls (harness
    /// instrumentation; no C counterpart).
    instr_exec: u64 = 0,
    /// C: zincvm.c:2250-2258 gensym — the static counter behind
    /// "shen.gensym_N" (never reset; symbols are interned so distinct N =>
    /// distinct symbols for the life of the VM).
    gensym_counter: u64 = 0,
    /// C: zincvm.c:2565-2575 newvar — the static counter behind "V_N"
    /// (0-based, same never-reset contract).
    newvar_counter: u32 = 0,
    /// M6 string-stream registry (streams.zig): fixed array of 8 slots + a
    /// count, zero-initialized (`.{ }`), so a fresh Vm needs no setup.
    streams: streams.StreamRegistry = .{},
    /// M10 frame-stack pool: a LIFO free-list of up to FRAME_POOL_MAX idle
    /// old-gen CALLFRAME_ARRAYs (65536 x 48 B ≈ 3 MB each), reused across
    /// vmExecEnv entries instead of bump-allocating a fresh array per call.
    /// Each slot is a PERSISTENT ROOT_PTR pushed at init (err_slot
    /// precedent), so an idle pooled array is pinned below every vmExecEnv
    /// entry watermark — its full-capacity drain scan (collect.zig) then
    /// sees an all-null body and pins nothing.
    frame_pool: [FRAME_POOL_MAX]?[*]types.CallFrame = .{null} ** FRAME_POOL_MAX,
    /// Number of non-null entries in frame_pool[0..frame_pool_live).
    frame_pool_live: usize = 0,
    /// Instrumentation (instr_exec precedent): pool hits/misses across runs.
    frame_pool_hits: u64 = 0,
    frame_pool_misses: u64 = 0,
    /// M11 tail-env reuse (interp.zig appterm N==A): hits = a tail call
    /// reused the current env array (the dead caller env fits the new
    /// arity); misses = a tail call had to allocate a fresh exact-size env
    /// array (first tail call after a frame restore, or the new env is
    /// larger than the retained physical capacity).
    env_reuse_hits: u64 = 0,
    env_reuse_misses: u64 = 0,

    /// Initialize a Vm into `vm` (caller-provided storage so `&vm.err_slot` /
    /// `&vm.defun_table_cap` / `&vm.values_table_cap` stay stable across the
    /// rooting and GC registration), rooting err_slot once.  Order (plan M2):
    /// alloc+zero tables -> gc.registerGlobalTable/registerValuesTable ->
    /// initGlobals.  The symbol interner is created empty (lazily allocates on
    /// first intern).
    pub fn init(vm: *Vm, g: *Gc) void {
        const a = std.heap.page_allocator;
        vm.* = .{
            .gc = g,
            .err_slot = .{ .tag = .nil, .payload = .{ .number = 0 } },
            .symbols = symbols.SymbolInterner.init(),
        };
        // alloc + zero the tables (C BSS calloc parity).
        const da = a.alloc(types.TableEntry, tables.DEFUN_TABLE_CAP) catch
            std.debug.panic("Vm.init: defun table alloc failed", .{});
        @memset(da, emptyEntry);
        vm.defun_table = da.ptr;
        const va = a.alloc(types.TableEntry, tables.VALUES_TABLE_CAP) catch
            std.debug.panic("Vm.init: values table alloc failed", .{});
        @memset(va, emptyEntry);
        vm.values_table = va.ptr;

        // Register with the GC BEFORE initGlobals stores any nursery value.
        g.registerGlobalTable(vm.defun_table, &vm.defun_table_cap);
        g.registerValuesTable(vm.values_table, &vm.values_table_cap);

        // C: zincvm.c:3146-3153 get_instr_limit reads $ZINCVM_INSTR_LIMIT per
        // vm_exec_env entry.  Zig 0.16 has no library-level env accessor
        // (std.posix.getenv / std.process.getEnvVarOwned are gone); the port
        // keeps the 5e9 DEFAULT here and lets the host set vm.instr_limit
        // directly before exec (strictly more flexible; M7's harness reads
        // the env once if the knob is needed).
        vm.instr_limit = 5_000_000_000;

        g.rootPushValue(&vm.err_slot);

        // M10: persistent pool-slot roots (err_slot precedent), pushed at
        // init and popped in reverse at deinit so idle pooled arrays stay
        // pinned below every vmExecEnv entry watermark.
        for (0..FRAME_POOL_MAX) |i| g.rootPushPtr(@ptrCast(&vm.frame_pool[i]));

        vm.initGlobals();
    }

    /// Pop the err_slot root, free the table arrays and tear down the interner.
    pub fn deinit(vm: *Vm) void {
        const a = std.heap.page_allocator;
        a.free(vm.defun_table[0..@as(usize, @intCast(vm.defun_table_cap))]);
        a.free(vm.values_table[0..@as(usize, @intCast(vm.values_table_cap))]);
        // Pop the pool-slot roots in reverse (LIFO) BEFORE err_slot to keep
        // the root stack balanced.
        var i = FRAME_POOL_MAX;
        while (i > 0) {
            i -= 1;
            vm.gc.rootPop(); // frame_pool[i]
        }
        vm.gc.rootPop(); // err_slot
        vm.symbols.deinit();
        vm.* = undefined;
    }

    // -----------------------------------------------------------------
    //  Defun / values table wrappers (fallback semantics live here so the
    //  prim half can be added in M5 when prims.zig exists).
    // -----------------------------------------------------------------

    /// C: zincvm.c:537-590 defun_set — insert/overwrite, always dirty-marked.
    pub fn defunSet(self: *Vm, name: []const u8, v: types.Value) void {
        tables.defunSet(self.gc, self.defun_table, tables.DEFUN_TABLE_CAP, name, v);
    }

    /// C: zincvm.c:593-618 defun_get — explicit entry, else the primitive /
    /// symbol fallback.  M5: known prim -> valPrim (the canonical table name
    /// literal), else valSymbol (macros/*stinput* must stay symbols).
    pub fn defunGet(self: *Vm, name: []const u8) types.Value {
        if (tables.defunLookup(self.defun_table, tables.DEFUN_TABLE_CAP, name)) |v| return v;
        if (prims.lookupDef(name)) |def| return values.valPrim(def.name);
        return symbols.valSymbol(&self.symbols, name);
    }

    /// Single-probe .global fetch: one defunLookup instead of the
    /// defunHas + defunGet double probe.  The prim/symbol fallback in
    /// defunGet is unreachable for a missing non-empty name (that throws),
    /// so only the empty-name edge (non-symbol operand) routes through
    /// defunGet, preserving its intern-"" behavior.
    pub fn defunGetChecked(self: *Vm, name: []const u8) VmError!types.Value {
        if (tables.defunLookup(self.defun_table, tables.DEFUN_TABLE_CAP, name)) |v| return v;
        if (name.len == 0) return self.defunGet(name);
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "global not found: {s}", .{name})
            catch "global not found";
        return self.throwShen(msg);
    }

    /// C: zincvm.c:624-642 defun_has — probe without the fallback.
    pub fn defunHas(self: *Vm, name: []const u8) bool {
        return tables.defunHas(self.defun_table, tables.DEFUN_TABLE_CAP, name);
    }

    /// C: zincvm.c:648-665 value_set.
    pub fn valueSet(self: *Vm, name: []const u8, v: types.Value) void {
        tables.valueSet(self.values_table, tables.VALUES_TABLE_CAP, name, v);
    }

    /// C: zincvm.c:668-676 value_get — no primitive fallback: (value +) must
    /// return the bare symbol `+`.
    pub fn valueGet(self: *Vm, name: []const u8) types.Value {
        if (tables.valueLookup(self.values_table, tables.VALUES_TABLE_CAP, name)) |v| return v;
        return symbols.valSymbol(&self.symbols, name);
    }

    // -----------------------------------------------------------------
    //  Error model — plan DECISION A
    // -----------------------------------------------------------------

    /// C: zincvm.c:711-720 vm_throw.  Builds the GC-allocated error value
    /// into vm.err_slot — rooted ONCE at init, which replaces the C S3
    /// per-catch-site error_val rooting — and unwinds as error.ShenError
    /// (the longjmp).  C aborts ("uncaught Shen error") when the catch
    /// chain is empty; the Zig port simply propagates the error to the
    /// host, which decides (harness reports failure).
    pub fn throwShen(vm: *Vm, msg: []const u8) VmError {
        vm.err_slot = values.valError(vm.gc, msg);
        return error.ShenError;
    }

    // -----------------------------------------------------------------
    //  initGlobals — C: zincvm.c:3752-3758
    // -----------------------------------------------------------------

    /// C: zincvm.c:3752-3758 init_globals — register every prim name as a
    /// VAL_PRIM global so [global X] falls back to it.  M5: the full
    /// prim_table (prims.zig), the single source (prims.def in C).
    pub fn initGlobals(self: *Vm) void {
        for (prims.primNames()) |def| self.defunSet(def.name, values.valPrim(def.name));
    }

    // -----------------------------------------------------------------
    //  vm_load_bundle — C: zincvm.c:4015-4102 (M6)
    // -----------------------------------------------------------------

    /// C: zincvm.c:4015-4102 vm_load_bundle.  Loads a bundle string into
    /// the global table (parser.parseBundle), then sets up the global-table
    /// environment: ZINC pattern keywords as bare symbols (never over a
    /// bundled closure), the standard I/O stream variables, the empty-alist
    /// global-table / value-table vars, and the rooted primitive?-names
    /// list built forward from the prim table (head = LAST name, C parity).
    ///
    /// Deviations (deliberate): C's trailing defun_freeze() perfect-hash
    /// build has no counterpart — DECISION B keeps the open-addressed table
    /// as the runtime structure.  The stream file handles are null: the I/O
    /// milestone owns them.  C's stdout printf goes to std.debug.print.
    /// Returns the number of closures loaded (0 on the outer shape error).
    pub fn loadBundle(self: *Vm, buf: [:0]const u8) i32 {
        // A bundle starts with '((' after optional whitespace (C:4016-4021).
        // The [:0] sentinel terminates the whitespace scan at end-of-input.
        var p: usize = 0;
        while (std.ascii.isWhitespace(buf[p])) p += 1;
        if (p + 1 >= buf.len or buf[p] != '(' or buf[p + 1] != '(') {
            std.debug.print("bundle error: not a bundle (expected ((...)))\n", .{});
            return 0;
        }

        const n = parser.parseBundle(self.gc, &self.symbols, self, buf);
        std.debug.print("Loaded {d} closures into global table\n", .{n});

        // Register ZINC pattern keywords as symbols (C:4033-4051) — ONLY
        // when the bundle did not itself provide the entry: the bundled
        // metacircular interpreter needs e.g. [global lookup] to resolve to
        // its closure, while structural matching needs the tag symbols.
        // Prim names (e.g. `cons`) already hold VAL_PRIM entries from
        // initGlobals and are skipped by the same defunHas guard — exact C
        // behavior, since C registers prim_names before loading too.
        const keywords = [_][]const u8{
            "number",     "symbol",     "string",     "boolean", "cons",
            "lambda",     "function",   "error",      "absvector",
            "stream in",  "stream out", "let",        "if",
            "lookup",     "freeze",     "type",       "defun",
            "define",     "cond",       "and",        "or",
            "do",         "fn",         "list",       "where",
        };
        for (keywords) |kw| {
            if (!self.defunHas(kw))
                self.defunSet(kw, symbols.valSymbol(&self.symbols, kw));
        }

        // Standard I/O stream variables (C:4057-4069): the bundled
        // stinput/stoutput closures read (value *stinput*) etc.;
        // shen.initialise-environment does not set them — the host must.
        // M6: wire the REAL std fds 0/1/2 (the I/O milestone owns these) —
        // stinput = fd 0 (stdin), stoutput = fd 1 (stdout), sterror = fd 2.
        self.valueSet("*stinput*", streams.valStreamInFd(0));
        self.valueSet("*stoutput*", streams.valStreamOutFd(1));
        self.valueSet("*sterror*", streams.valStreamOutFd(2));

        // The metacircular interpreter's global-table / value-table vars
        // must start as empty alists (C:4075, 4081), not the bare symbol
        // that an unset value_get would return.
        self.valueSet("global-table", values.valNil());
        self.valueSet("value-table", values.valNil());

        // primitive?-names (C:4088-4095), built forward from the single
        // source prim table so C's primitive set and Shen's primitive?
        // stay in sync.  pn is rooted across the valCons churn; the head
        // of the finished list is the LAST table name (C forward-loop
        // parity).
        {
            var pn = values.valNil();
            self.gc.rootPushValue(&pn);
            defer self.gc.rootPop();
            for (prims.primNames()) |def|
                pn = values.valCons(self.gc, symbols.valSymbol(&self.symbols, def.name), pn);
            self.valueSet("primitive?-names", pn);
        }

        return n;
    }
};

/// A zeroed TableEntry (C `memset(&e,0,sizeof e)` — name=NULL, value tag 0).
const emptyEntry = types.TableEntry{
    .name = null,
    .value = .{ .tag = .nil, .payload = .{ .number = 0 } },
};
