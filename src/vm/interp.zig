//! src/vm/interp.zig — the ZINC eval loop (milestone M4).
//!
//! C origin: zincvm.c:406-437 (ValueArray va_init/va_push/va_pop/va_peek/
//! va_free), zincvm.c:3107-3137 (lookup_env / env_push / env_pop), and
//! zincvm.c:3154-3474 (vm_exec_env, vm_exec).
//!
//! M4 SCOPE: the eval loop with full per-opcode rooting discipline, LAMBDA
//! paths only.  exec_primitive is NOT yet linked (plan M5 ports the pure
//! subset into vm/prims.zig): the OP_PRIM / apply-prim / appterm-prim call
//! sites go through a stub that always hard-stops, already shaped as the
//! final DECISION-A discipline (error.Halt caught HERE = C's
//! `exec_primitive() < 0 → goto done` with acc preserved; error.ShenError
//! propagates to the enclosing CatchSite chain).
//!
//! ERROR MODEL (plan DECISION A): C's setjmp/longjmp + CatchFrame chain
//! becomes VmError = error{ShenError, Halt} (state.zig) plus a linked chain
//! of stack-allocated CatchSites on Vm.  vm_throw → vm.throwShen(msg)
//! (builds valError into the permanently-rooted vm.err_slot, returns
//! error.ShenError).  Because Zig error returns unwind frames WITH defers
//! running (longjmp skips them), ONE `defer gc.rootPopTo(entry_wm)` per
//! vmExecEnv frame replaces C's 9 manual pops at done (C:3464-3466) and
//! every pop_to a longjmp landing site would have needed.
//!
//! ROOTING CONTRACT (the crux — ported verbatim from the C, C line refs on
//! each site):
//!   [PROLOGUE] code(1)/init_env(2)/env(3)/stack.data(4)/acc(5) are pushed
//!   BEFORE any allocation, while env/stack.data/acc are still NULL/nil —
//!   NULL slots pin nothing, but the SLOTS must be stable addresses because
//!   scanRoots reads their CURRENT value at collect time; then va_init, the
//!   init_env copy + barrier, the old-gen CallFrame array (65536 entries,
//!   ~3 MB), callframe_array(6)/cur_code(7)/frame_stack(8) roots.
//!   frames_sp is a plain native local (C needed a GC-heap int only because
//!   longjmp dangles stack locals; ROOT_CALLFRAME_ARRAY's np is never
//!   dereferenced at scan time — collect.zig scanRoots is a deliberate
//!   no-op for that kind).
//!   [PER-OPCODE] transient roots (the 64-slot argbuf in apply/appterm) are
//!   pushed before the first alloc in the opcode and popped right after the
//!   env fill; va_push / env_push root their argument internally across the
//!   grow alloc.  `Instr *in` is re-derived from the rooted cur_code at the
//!   top of EVERY iteration and never cached across an allocating call.
//!   [CURRYING] apply/appterm partial application (N<A) and Elm-style
//!   over-application (N>A; the C VM and the metacircular reference are
//!   full-arity-only here).  N==0 apply KEEPS the legacy full-arity jump
//!   (hand-bundle top-level thunks are called with nargs==0 via
//!   `mm <name> p`; zinc-c never emits a 0-arg apply): the partial's
//!   drop-grabs code is a FRESH
//!   allocArray(Instr) + copy of the suffix — interior pointers like
//!   code+N are FORBIDDEN roots in a moving collector (gcMove reads the
//!   header at *(p-1)) — and the fresh array is itself rooted across the
//!   later env/closure allocs (buildPartialClosure).  The N>A peel runs
//!   each callee body through a NESTED vmExecEnv (fresh frame stack, the
//!   prims.zig trap-error precedent) with the outer frame's roots —
//!   including the rooted &acc slot and the argbuf — still live below the
//!   callee's entry watermark (peelOverArgs).  Both paths re-read every
//!   lambda field through the rooted acc slot (5) after each allocation.
//!   [EPILOGUE] the defer rootPopTo(entry_wm) above covers every exit,
//!   including error returns.
//!   [ENV OWNERSHIP] the current env array (slot 3) has exactly ONE referee:
//!   closure Values hold only valLambda COPIES of an env (values.zig:154 —
//!   the running array is never stored into a closure), saved CallFrames hold
//!   OTHER arrays (ownership transfers &env<->cf.env atomically at apply push
//!   and popFramePushAcc), and nested vmExecEnv entries pass freshly-built
//!   arrays that the prologue COPIES (:569-584).  This invariant is what makes
//!   the appterm N==A tail-env REUSE safe (the old env is dead after the tail
//!   jump); the reused array's tail [new_env_len..env_cap) must be nil-cleared
//!   and env_cap must remain the TRUE physical capacity.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const state = @import("state.zig");
const values = @import("values.zig");
const prims = @import("prims.zig");

const Gc = gc.Gc;
const Value = types.Value;
const Vm = state.Vm;
const VmError = state.VmError;

// =====================================================================
//  Value stack — C: zincvm.c:406-437
// =====================================================================

/// C: zincvm.c:407 STACK_INIT_CAP.
pub const STACK_INIT_CAP: i32 = 12;

/// C: zincvm.c:410-413 va_init.  Must only be called once the caller's
/// stable slots for a->data (and anything read during the alloc) are rooted
/// — vmExecEnv's prologue does this before its va_init.
pub fn vaInit(g: *Gc, a: *types.ValueArray) void {
    a.data = g.allocArray(Value, @intCast(STACK_INIT_CAP));
    a.len = 0;
    a.cap = STACK_INIT_CAP;
}

/// C: zincvm.c:414-432 va_push.  On grow, v is rooted across the
/// GC_VALUE_ARRAY (v may carry interior pointers — lambda.code/env,
/// cons.car/cdr, str.data — that a collection fired during the grow would
/// otherwise leave stale in this local, C:416-421); after the store, the
/// write barrier records the element array in the remembered set iff it is
/// old-gen AND the stored Value references the nursery (C:429-431).
pub fn vaPush(g: *Gc, a: *types.ValueArray, v: Value) void {
    var vv = v;
    if (a.len >= a.cap) {
        const new_cap: i32 = a.cap * 2;
        var guard = g.rootValue(&vv); // root v across GC_VALUE_ARRAY — C:422
        defer guard.end();
        const new_data = g.allocArray(Value, @intCast(new_cap));
        const ln: usize = @intCast(a.len);
        @memcpy(new_data[0..ln], a.data.?[0..ln]);
        // M5 fix: the grow copies the OLD elements into a possibly-oldgen
        // array; barrier them (a copied nursery reference would otherwise go
        // stale at the next scavenge).  Mirrors applyBundledN / interp apply.
        if (g.inOldgen(@intFromPtr(new_data))) {
            var j: usize = 0;
            while (j < ln) : (j += 1) {
                if (gc.scan.valueReferencesNursery(g, &a.data.?[j])) {
                    g.dirtyVectorsAdd(new_data);
                    break;
                }
            }
        }
        a.data = new_data;
        a.cap = new_cap;
    }
    const idx: usize = @intCast(a.len);
    a.data.?[idx] = vv;
    a.len += 1;
    // C checks &v (the stored copy); &a->data[a->len-1] is now that copy and
    // valueReferencesNursery is read-only — identical behaviour.
    if (g.inOldgen(@intFromPtr(a.data.?)) and
        gc.scan.valueReferencesNursery(g, &a.data.?[idx]))
        g.dirtyVectorsAdd(a.data.?);
}

/// C: zincvm.c:433-436 va_pop — pop from an empty stack is fatal
/// (C fprintf + exit(1) → std.debug.panic).
pub fn vaPop(a: *types.ValueArray) Value {
    if (a.len <= 0) std.debug.panic("fatal: pop from empty stack", .{});
    a.len -= 1;
    // Clear the vacated slot: the GC scans value_arrays by full capacity,
    // so a stale ref here would retain popped Values (closure envs).
    const v = a.data.?[@intCast(a.len)];
    a.data.?[@intCast(a.len)] = values.valNil();
    return v;
}

/// C: zincvm.c:437 va_peek.
pub fn vaPeek(a: *types.ValueArray) Value {
    return a.data.?[@intCast(a.len - 1)];
}

/// C: zincvm.c:438 va_free — release the slots only (the array itself is
/// GC-managed); the rooted &stack.data slot now pins nothing.
pub fn vaFree(a: *types.ValueArray) void {
    a.data = null;
    a.len = 0;
    a.cap = 0;
}

// =====================================================================
//  Environment access — C: zincvm.c:3107-3137
// =====================================================================

/// C: zincvm.c:3107-3116 lookup_env.  Out-of-bounds access returns 0
/// silently — this occurs in nested closures with empty captured
/// environments during interp execution; downstream guards (cons?, =, ...)
/// reject the sentinel.
pub fn lookupEnv(n: i32, env: ?[*]Value, env_len: i32) Value {
    if (n < 0 or n >= env_len) return values.valNumber(0);
    return env.?[@intCast(env_len - 1 - n)];
}

/// C: zincvm.c:3117-3132 env_push.  env/env_len/env_cap are the caller's
/// frame locals — in vmExecEnv `&env` is a ROOT_PTR pushed in the prologue,
/// so writes through these pointers update the rooted slot.  The grow roots
/// v across the GC_VALUE_ARRAY (C:3120); the store takes the old-gen write
/// barrier (C:3128-3129).
pub fn envPush(g: *Gc, env: *?[*]Value, env_len: *i32, env_cap: *i32, v: Value) void {
    var vv = v;
    if (env_len.* >= env_cap.*) {
        const new_cap: i32 = if (env_cap.* != 0) env_cap.* * 2 else 4;
        var guard = g.rootValue(&vv); // root v across GC_VALUE_ARRAY — C:3120
        defer guard.end();
        const new_env = g.allocArray(Value, @intCast(new_cap));
        const ln: usize = @intCast(env_len.*);
        if (ln > 0) {
            @memcpy(new_env[0..ln], env.*.?[0..ln]);
            // M5 fix: barrier copied env elements on grow (same rationale as
            // the vaPush grow barrier).
            if (g.inOldgen(@intFromPtr(new_env))) {
                var j: usize = 0;
                while (j < ln) : (j += 1) {
                    if (gc.scan.valueReferencesNursery(g, &env.*.?[j])) {
                        g.dirtyVectorsAdd(new_env);
                        break;
                    }
                }
            }
        }
        env.* = new_env;
        env_cap.* = new_cap;
    }
    const idx: usize = @intCast(env_len.*);
    env.*.?[idx] = vv;
    env_len.* += 1;
    if (g.inOldgen(@intFromPtr(env.*.?)) and
        gc.scan.valueReferencesNursery(g, &env.*.?[idx]))
        g.dirtyVectorsAdd(env.*.?);
}

/// C: zincvm.c:3130-3137 env_pop.  Inside a trap-error catch site
/// (in_trap_error) a pop of an empty environment throws — catchable;
/// anywhere else it is fatal.
pub fn envPop(vm: *Vm, env: *?[*]Value, env_len: *i32) VmError!Value {
    if (env_len.* <= 0) {
        if (vm.catch_chain != null and vm.catch_chain.?.in_trap_error)
            return vm.throwShen("runtime: pop empty environment");
        std.debug.panic("runtime: pop empty environment", .{});
    }
    env_len.* -= 1;
    // Same capacity-scan concern as vaPop: clear the vacated env slot.
    const v = env.*.?[@intCast(env_len.*)];
    env.*.?[@intCast(env_len.*)] = values.valNil();
    return v;
}

// =====================================================================
//  exec_primitive — C: zincvm.c:1780-2794 (ported in prims.zig, M5)
// =====================================================================

/// C: zincvm.c:1780-2794 exec_primitive, pure subset — the implementation
/// lives in prims.zig (M5).  DECISION A contract:
///   error.Halt       → caught at the call site → break to done, acc
///                      preserved (C: `exec_primitive(...) < 0 → goto done`);
///   error.ShenError  → propagates up to the enclosing CatchSite chain.
fn execPrimitive(vm: *Vm, name: []const u8, acc: *Value, stack: *types.ValueArray) VmError!void {
    return prims.execPrimitive(vm, name, acc, stack);
}

// =====================================================================
//  Currying helpers — partial application (N<A) + Elm-style
//  over-application peel (N>A).
//
//  NOT in the C VM: zincvm.c:3274-3350 OP_APPLY / OP_APPTERM are
//  full-arity-only (under/over-application silently corrupts the env).
//  The N<A partial-closure semantics follow the metacircular reference
//  (interp.shen:193-245, zinc-arity + drop-grabs); the N>A continuation is
//  the user-approved Elm-style divergence where the reference errors
//  ("too many args").  NOTE for frontend compilers (supersedes the
//  elm-csexp plan's TARGET SEMANTICS): curried calls are now LEGAL at the
//  ZINC level, but primitives are NOT curried — a prim left with args
//  after a peel throws, so compilers must wrap prims in curried closures.
// =====================================================================

/// Arity of a closure body: leading grabs + 1, minimum 1 (a K-param lambda
/// compiles to [grab x (K-1), body..., ret]; the wrapper [cur] binds the
/// first param via APPLY, every further param via a leading grab).
/// Reference: interp.shen:78-85 zinc-arity.  Pure walk, NO allocation.
/// pub for the vm_test unit tests (T1); every runtime caller reads the
/// fields through a ROOTED slot.
pub fn zincArity(code: ?*types.Instr, code_len: i32) i32 {
    if (code == null) return 1;
    const arr: [*]types.Instr = @ptrCast(code.?);
    var i: i32 = 0;
    var grabs: i32 = 0;
    while (i < code_len and arr[@intCast(i)].op == .grab) {
        grabs += 1;
        i += 1;
    }
    return grabs + 1;
}

/// The .ret arm's frame-restore body (C:3347-3360), extracted so .ret and
/// appterm's under-application paths share ONE source of truth for the
/// rooting-sensitive frame-restore discipline.  Pops the current CallFrame,
/// restores the caller's code/env/stack (writing THROUGH the rooted frame
/// slots cur_code (7) / env (3) / stack.data (4)), releases the stale
/// pointers in the popped frame slot (the old-gen CALLFRAME_ARRAY drain
/// scan must not keep dead frame envs/stacks reachable — C:3217-3221), and
/// pushes `accv` as the return value on the restored caller stack
/// (vaPush roots accv across its own grow alloc internally).  Returns
/// false when the frame stack is exhausted — the caller breaks :run and
/// vmExecEnv's epilogue returns acc.
fn popFramePushAcc(
    g: *Gc,
    accv: Value,
    frame_stack: [*]types.CallFrame,
    frames_sp: *i32,
    cur_code: *?*types.Instr,
    cur_len: *i32,
    pc: *i32,
    env: *?[*]Value,
    env_len: *i32,
    env_cap: *i32,
    stack: *types.ValueArray,
) bool {
    if (frames_sp.* <= 0) return false;
    frames_sp.* -= 1;
    const cf = &frame_stack[@intCast(frames_sp.*)];
    cur_code.* = cf.code;
    cur_len.* = cf.code_len;
    pc.* = cf.pc;
    env.* = cf.env;
    env_len.* = cf.env_len;
    env_cap.* = cf.env_cap;
    vaFree(stack);
    stack.* = cf.stack;
    cf.env = null;
    cf.stack.data = null;
    cf.stack.len = 0;
    cf.code = null;
    cf.code_len = 0;
    cf.pc = 0;
    vaPush(g, stack, accv); // push return value to caller stack
    return true;
}

/// buildPartialClosure — drop-grabs + env capture for the N<A
/// under-application case of apply/appterm.  Returns a closure whose code
/// is the callee body with its `nargs` leading grabs dropped and whose env
/// is closure_env ++ argbuf[0..nargs].  The env layout is CONSISTENT with
/// a single-shot call: a later apply APPENDS the remaining args, and
/// access j then resolves exactly as if all args had been supplied at once
/// (proof in the plan's GROUND TRUTH 1).
///
/// DROP-GRABS IS A FRESH ALLOC, NEVER POINTER ARITHMETIC: closure bodies
/// are GC-allocated, MOVING instr_arrays — `code + nargs` would be an
/// interior pointer, forbidden as a root by the precise-root contract
/// (gcMove reads the header at *(p-1): garbage header = heap corruption).
/// The fresh array is scanned as instr_array, so the copied Instrs'
/// operand strings and closure_code children stay reachable even if the
/// original body array dies.
///
/// ROOTING WALKTHROUGH (the M4 hazard discipline):
///   - fnv_slot is the CALLER'S ROOTED &acc slot (prologue root (5)) and
///     argbuf is the caller's rootPushValueArray pair: both stay pinned
///     across every allocation below, and every lambda field is re-read
///     through fnv_slot AFTER each alloc (fresh post-GC).
///   - ALLOC#1 (fresh instr_array): fnv_slot pins the ORIGINAL code array;
///     the source head is re-read through fnv_slot after the alloc.  The
///     i32 bounds (code_len/nargs) are plain copies — GC-invariant.
///   - new_code is itself rooted across ALLOC#2/#3: the fresh array must
///     survive the env allocs even if the original body dies.
///   - ALLOC#2 (env concat): lambda_env is re-read through fnv_slot after
///     the alloc; the fill mirrors the full-arity apply env build's
///     inOldgen / valueReferencesNursery / dirtyVectorsAdd barrier dance.
///   - ALLOC#3 (the closure's env copy) happens INSIDE valLambda, which
///     roots its own code/env slot copies; no alloc occurs between the
///     fill above and its prologue pushes.
fn buildPartialClosure(g: *Gc, fnv_slot: *Value, argbuf: [*]Value, nargs: i32) Value {
    const code_len = fnv_slot.payload.lambda.code_len; // i32 — GC-invariant
    const new_len = code_len - nargs; // drop-grabs nargs
    // ALLOC#1 — fresh instr_array holding the suffix [nargs..code_len).
    var new_code = g.allocArray(types.Instr, @intCast(new_len));
    // Source head re-read through the rooted slot AFTER the alloc; the
    // i32 bounds are pre-GC copies and still exact.
    const src: [*]types.Instr = @ptrCast(fnv_slot.payload.lambda.code.?);
    @memcpy(
        new_code[0..@as(usize, @intCast(new_len))],
        src[@intCast(nargs)..@intCast(code_len)],
    );
    // Drop-grabs shifts instruction positions by -nargs, so the absolute
    // jmp/jmpf targets (resolved at Emit time against the grab-INCLUSIVE
    // layout) must be rebased to the grab-EXCLUSIVE suffix layout here.
    //   - Access operands are env indices; env = closure_env ++ argbuf
    //     preserves the layout — do NOT touch them.
    //   - Nested cur bodies have their own label spaces and never shift.
    //   - Compiler guarantee (compileOne/emitLambda/ctorEntry/wrappers all
    //     emit grabs++body with labels only inside the body): every target
    //     lives in [nargs, code_len); the @max(0, ...) clamp is
    //     belt-and-braces only.
    for (new_code[0..@as(usize, @intCast(new_len))]) |*ins| {
        switch (ins.op) {
            .jmp, .jmpf => ins.jmp_target = @max(0, ins.jmp_target - nargs),
            else => {},
        }
    }
    g.rootPushPtr(@ptrCast(&new_code)); // pin the fresh array across #2/#3
    defer g.rootPop(); // new_code

    const lambda_env_len = fnv_slot.payload.lambda.env_len; // i32 — invariant
    // ALLOC#2 — the env concat closure_env ++ argbuf[0..nargs].
    const ne = g.allocArray(Value, @intCast(lambda_env_len + nargs));
    const lambda_env = fnv_slot.payload.lambda.env; // fresh post-alloc read
    const ne_is_oldgen = g.inOldgen(@intFromPtr(ne));
    if (lambda_env_len > 0 and lambda_env != null) {
        const lel: usize = @intCast(lambda_env_len);
        @memcpy(ne[0..lel], lambda_env.?[0..lel]);
        if (ne_is_oldgen) {
            var j: usize = 0;
            while (j < lel) : (j += 1) {
                if (gc.scan.valueReferencesNursery(g, &lambda_env.?[j])) {
                    g.dirtyVectorsAdd(ne);
                    break;
                }
            }
        }
    }
    var i: usize = 0;
    while (i < @as(usize, @intCast(nargs))) : (i += 1) {
        const idx = @as(usize, @intCast(lambda_env_len)) + i;
        ne[idx] = argbuf[i];
        if (ne_is_oldgen and gc.scan.valueReferencesNursery(g, &argbuf[i]))
            g.dirtyVectorsAdd(ne);
    }
    // ALLOC#3 inside valLambda (the env copy); its internal roots cover the
    // interval.  new_code is read through the rooted slot — post-#3 fresh.
    return values.valLambda(
        g,
        @ptrCast(new_code),
        new_len,
        ne,
        lambda_env_len + nargs,
    );
}

/// peelOverArgs — the N>A Elm-style over-application loop shared by apply
/// and appterm: apply the callee's FIRST `arity` source args through a
/// NESTED vmExecEnv (fresh frame stack — the prims.zig trap-error
/// precedent), then continue applying the remaining args to the result
/// until the application is no longer over-applied.
///
/// Exits, returning to the caller's N==A / N<A dispatch: acc_slot is a
/// .lambda with nargs <= zincArity(acc), or acc_slot is the final value
/// with nargs == 0.  Throws (catchable ShenError, trap-error compatible)
/// when a peel level's result is a non-lambda with args remaining — for a
/// .prim specifically: primitives are fixed-arity, NOT curried; the
/// frontend compiler must wrap them in curried closures.
///
/// TERMINATION: zincArity >= 1 always, so nargs strictly decreases per
/// level (bounded by the 64-slot argbuf).  Each nested vmExecEnv gets a
/// fresh instr_limit budget and allocates its own frame stack (~3 MB
/// old-gen per level, bounded by over-application depth) — noted future
/// optimizations, accepted for this unit (same as trap-error).
///
/// Rooting: acc_slot is the caller's rooted &acc slot (5) — every lambda
/// field read goes through it AFTER the previous alloc; argbuf/nargs are
/// the caller's rootPushValueArray pair, still live BELOW the nested
/// vmExecEnv's entry watermark (its defer rootPopTo rebalances only its
/// own roots).  The env concat is filled and handed straight to vmExecEnv
/// with NO alloc in between — the callee's prologue roots init_env at its
/// slot (2) before its first allocation (primTrapError precedent).
fn peelOverArgs(
    vm: *Vm,
    acc_slot: *Value,
    argbuf: *[64]Value,
    nargs: *i32,
    op_name: []const u8,
) VmError!void {
    const g = vm.gc;
    while (true) {
        // Loop invariant: acc_slot.* is a .lambda and nargs.* > its arity.
        const arity = zincArity(acc_slot.payload.lambda.code, acc_slot.payload.lambda.code_len);
        const lambda_env_len = acc_slot.payload.lambda.env_len; // i32 copy
        // env_A = closure_env ++ argbuf[0..arity] — the FIRST arity SOURCE
        // args (argbuf holds source order: argbuf[i] = call arg i+1).
        const ne = g.allocArray(Value, @intCast(lambda_env_len + arity));
        // Callee read FRESH post-alloc through the rooted acc_slot.
        const code = acc_slot.payload.lambda.code;
        const code_len = acc_slot.payload.lambda.code_len;
        const lambda_env = acc_slot.payload.lambda.env;
        const ne_is_oldgen = g.inOldgen(@intFromPtr(ne));
        if (lambda_env_len > 0 and lambda_env != null) {
            const lel: usize = @intCast(lambda_env_len);
            @memcpy(ne[0..lel], lambda_env.?[0..lel]);
            if (ne_is_oldgen) {
                var j: usize = 0;
                while (j < lel) : (j += 1) {
                    if (gc.scan.valueReferencesNursery(g, &lambda_env.?[j])) {
                        g.dirtyVectorsAdd(ne);
                        break;
                    }
                }
            }
        }
        var i: usize = 0;
        while (i < @as(usize, @intCast(arity))) : (i += 1) {
            const idx = @as(usize, @intCast(lambda_env_len)) + i;
            ne[idx] = argbuf[i];
            if (ne_is_oldgen and gc.scan.valueReferencesNursery(g, &argbuf[i]))
                g.dirtyVectorsAdd(ne);
        }
        // Nested body run: fresh frame stack, returns the body's result.
        acc_slot.* = try vmExecEnv(vm, code, code_len, ne, lambda_env_len + arity);
        // Left-shift argbuf by arity — plain Value copies, no alloc
        // (copyForwards: dst < src but the ranges may overlap when
        // nargs > 2*arity); the ROOT_VALUE_ARRAY count is read live at
        // scan time, so the remaining prefix stays the scanned region.
        const rem = nargs.* - arity;
        if (rem > 0) {
            std.mem.copyForwards(
                Value,
                argbuf[0..@as(usize, @intCast(rem))],
                argbuf[@intCast(arity)..@intCast(nargs.*)],
            );
        }
        nargs.* = rem;

        if (nargs.* == 0) return; // acc is the final value
        if (acc_slot.tag != .lambda) {
            const what = if (acc_slot.tag == .prim)
                "cannot over-apply primitive"
            else
                "over-application to non-callable";
            var buf: [80]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "{s}: {s}", .{ op_name, what }) catch what;
            return vm.throwShen(msg);
        }
        if (nargs.* <= zincArity(acc_slot.payload.lambda.code, acc_slot.payload.lambda.code_len))
            return; // caller dispatches N==A (full) / N<A (partial)
        // Still over-applied — peel another level.
    }
}

// =====================================================================
//  The eval loop — C: zincvm.c:3154-3466 vm_exec_env
// =====================================================================

/// M10 frame-stack pool: acquire a CallFrame array for one vmExecEnv entry.
/// A hit reuses an idle pooled array (all-null at rest by the clean-only-
/// live-range invariant, so no @memset needed); a miss allocates a fresh
/// old-gen array whose body gcalloc_internal already zeroes.  No allocation
/// on the hit path — the array moves pool-slot -> local with no collect
/// possible mid-transfer (the pool slot is a persistent root).
fn frameStackAcquire(vm: *Vm) [*]types.CallFrame {
    if (vm.frame_pool_live > 0) {
        vm.frame_pool_live -= 1;
        const arr = vm.frame_pool[vm.frame_pool_live].?;
        vm.frame_pool[vm.frame_pool_live] = null;
        vm.frame_pool_hits += 1;
        return arr;
    }
    vm.frame_pool_misses += 1;
    return vm.gc.allocArrayOldgen(types.CallFrame, types.CALL_STACK_DEPTH);
}

/// M10 frame-stack pool: release `arr` back to the pool on vmExecEnv exit.
/// MUST contain no allocation — the only live reference during the transfer
/// is the (still-rooted) local slot on the way in and the persistent pool
/// slot on the way out, so no collect can move the array mid-copy.
/// Clears ONLY the used range [0..sp): every pop site nulls its slot and
/// pushes write slot[sp] then increment, so the dirtied slots are a subset
/// of [0..sp) — the array is all-null at rest, making the full-capacity
/// drain scan (collect.zig) cheap and retention-free.  When the pool is
/// full the array is dropped: its pages are never queued, so it is never
/// scanned and needs no clearing (graceful degradation to per-call alloc).
fn frameStackRelease(vm: *Vm, arr: [*]types.CallFrame, sp: i32) void {
    if (vm.frame_pool_live < state.FRAME_POOL_MAX) {
        @memset(arr[0..@intCast(sp)], std.mem.zeroes(types.CallFrame));
        vm.frame_pool[vm.frame_pool_live] = arr;
        vm.frame_pool_live += 1;
    }
}

/// C: zincvm.c:3154-3466 vm_exec_env.  THE ROOTING CRUX — see the module
/// doc for the full contract.  Every root push is annotated with its C line;
/// the single defer rootPopTo(entry_wm) covers ALL exits (break-to-done,
/// error.ShenError), replacing C's manual pops and its longjmp discipline.
pub fn vmExecEnv(
    vm: *Vm,
    code_in: ?*types.Instr,
    code_len: i32,
    init_env_in: ?[*]Value,
    init_env_len: i32,
) VmError!Value {
    const g = vm.gc;
    const entry_wm = g.rootWatermark();
    // ONE pop-to for every exit path (plan DECISION A; C: 9 pops at
    // C:3464-3466 + every longjmp site's pop_to).
    defer g.rootPopTo(entry_wm);

    // ---- PROLOGUE: push all root slots BEFORE any allocation (C:3159-3166).
    // &env / &stack.data / &acc are pushed early (while still NULL/nil) so
    // gc_scan_roots reads the CURRENT slot value at collect time — they are
    // reassigned across gc_alloc calls below.  NULL slots pin nothing, so
    // pushing early is safe.
    var code = code_in;
    var init_env = init_env_in;
    g.rootPushPtr(@ptrCast(&code)); // (1) root code across allocs — C:3159
    g.rootPushPtr(@ptrCast(&init_env)); // (2) root init_env across allocs — C:3160
    var stack: types.ValueArray = .{ .data = null, .len = 0, .cap = 0 };
    var env: ?[*]Value = null;
    var env_len: i32 = 0;
    var env_cap: i32 = 0;
    var acc: Value = values.valNil();
    g.rootPushPtr(@ptrCast(&env)); // (3) ROOT_PTR — stable slot for env — C:3167
    g.rootPushPtr(@ptrCast(&stack.data)); // (4) ROOT_PTR — stable slot for stack.data — C:3168
    g.rootPushValue(&acc); // (5) ROOT_VALUE — stable slot for acc — C:3169
    vaInit(g, &stack); // now safe: all slots above are rooted — C:3171

    // ---- initial environment copy (C:3172-3181).  init_env is rooted (2),
    // so post-alloc reads of its elements are fresh.
    if (init_env_len > 0 and init_env != null) {
        env_cap = init_env_len;
        const cap: usize = @intCast(env_cap);
        env = g.allocArray(Value, cap);
        @memcpy(env.?[0..cap], init_env.?[0..cap]);
        env_len = init_env_len;
        if (g.inOldgen(@intFromPtr(env.?))) {
            var j: usize = 0;
            while (j < cap) : (j += 1) {
                if (gc.scan.valueReferencesNursery(g, &init_env.?[j])) {
                    g.dirtyVectorsAdd(env.?);
                    break;
                }
            }
        }
    }

    // ---- call-frame stack (C:3182-3191): one old-gen CALLFRAME_ARRAY per
    // vmExecEnv call, REUSED from the Vm frame-stack pool (M10).  Fresh
    // arrays are zeroed by gcalloc_internal; pooled arrays are all-null at
    // rest (release clears the used range).  allocatepage's LASTRESORT may
    // run a full collect HERE with only roots (1)-(5) live — same as C.
    var frame_stack: [*]types.CallFrame = frameStackAcquire(vm);
    // C allocates a GC-heap int for frames_sp so it survives longjmp
    // (C:3186-3188).  Zig error unwinding runs defers before this frame
    // dies, so a plain native local is safe (plan DECISION A); the
    // ROOT_CALLFRAME_ARRAY np below is never dereferenced at scan time.
    var frames_sp: i32 = 0;
    g.rootPushCallframeArray(frame_stack, &frames_sp); // (6) — C:3195
    // Release the frame stack back to the pool on EVERY exit.  Registered
    // AFTER the rootPopTo(entry_wm) defer above, so defers run it FIRST
    // (reverse order) while root (8) below still pins frame_stack, and
    // BEFORE the entry_wm pop truncates the shadow stack.  Zig `defer`
    // evaluates its expression at scope exit, reading frames_sp's CURRENT
    // value — so a deep-call error unwind clears the true used range, never
    // the 0 frames_sp held here at registration time.
    defer frameStackRelease(vm, frame_stack, frames_sp);

    var pc: i32 = 0;
    var cur_code = code; // current body's Instr array head (?*Instr)
    var cur_len: i32 = code_len;
    var instr_count: u64 = 0;
    const instr_limit = vm.instr_limit; // cached once — C:3146-3153 get_instr_limit
    g.rootPushPtr(@ptrCast(&cur_code)); // (7) ROOT_PTR — Instr** — C:3201
    g.rootPushPtr(@ptrCast(&frame_stack)); // (8) ROOT_PTR — CallFrame** — C:3202

    run: while (true) {
        // C:3204-3208 — hard instruction limit.
        instr_count += 1;
        vm.instr_exec += 1;
        if (instr_count >= instr_limit) {
            std.debug.print(
                "[HARD LIMIT] {d} instructions, aborting at pc={d} frames={d}\n",
                .{ instr_limit, pc, frames_sp },
            );
            break :run; // goto done
        }

        // C:3209-3228 — pc out of range: pop a CallFrame and resume in the
        // caller, or finish when the frame stack is exhausted.
        if (pc < 0 or pc >= cur_len) {
            if (frames_sp > 0) {
                frames_sp -= 1;
                const cf = &frame_stack[@intCast(frames_sp)];
                cur_code = cf.code;
                cur_len = cf.code_len;
                pc = cf.pc;
                env = cf.env;
                env_len = cf.env_len;
                env_cap = cf.env_cap;
                vaFree(&stack);
                stack = cf.stack;
                // Release stale GC pointers in the popped slot so the full
                // CALLFRAME_ARRAY drain scan does not keep dead frame envs /
                // stacks reachable until the slot is reused (C:3217-3221).
                cf.env = null;
                cf.stack.data = null;
                cf.stack.len = 0;
                cf.code = null;
                cf.code_len = 0;
                cf.pc = 0;
                continue :run;
            }
            break :run; // frames exhausted — C:3227
        }

        // Re-derived EVERY iteration from the rooted cur_code — never cached
        // across an allocating call (an alloc may evacuate the code array).
        const cur_many: [*]types.Instr = @ptrCast(cur_code.?);
        const in: *types.Instr = &cur_many[@intCast(pc)];

        switch (in.op) {
            // C:3230-3232 — literal loads.
            .number, .string, .symbol, .boolean, .float => {
                acc = in.operand;
                vaPush(g, &stack, acc);
                pc += 1;
            },

            // C:3233-3244 — [prim X]: args already on stack (auto-pushed by
            // loads); execute the primitive, push the result.
            .prim => {
                const pn = if (in.operand.tag == .symbol) values.symSlice(in.operand) else "";
                execPrimitive(vm, pn, &acc, &stack) catch |e| switch (e) {
                    // C: exec_primitive() < 0 → goto done (acc preserved).
                    error.Halt => break :run,
                    // A prim-thrown Shen error unwinds to the catch site.
                    error.ShenError => return error.ShenError,
                };
                vaPush(g, &stack, acc);
                pc += 1;
            },

            // C:3245 — pushmark.
            .pushmark => {
                vaPush(g, &stack, values.valMark());
                pc += 1;
            },

            // C:3246-3263 — grab: env push, or (mark on top) partial
            // application return.
            .grab => {
                if (stack.len > 0 and vaPeek(&stack).tag == .mark) {
                    _ = vaPop(&stack);
                    if (frames_sp > 0) {
                        frames_sp -= 1;
                        const cf = &frame_stack[@intCast(frames_sp)];
                        cur_code = cf.code;
                        cur_len = cf.code_len;
                        pc = cf.pc;
                        env = cf.env;
                        env_len = cf.env_len;
                        env_cap = cf.env_cap;
                        // C:3257 — no va_free here; the current stack array
                        // is simply GC'd later.
                        stack = cf.stack;
                        cf.env = null;
                        cf.stack.data = null;
                        cf.stack.len = 0;
                        cf.code = null;
                        cf.code_len = 0;
                        cf.pc = 0;
                        vaPush(g, &stack, acc); // push return value to caller stack
                    } else break :run;
                } else if (stack.len > 0) {
                    const v = vaPop(&stack);
                    envPush(g, &env, &env_len, &env_cap, v);
                    pc += 1;
                } else pc += 1;
            },

            // C:3264-3346 — apply.  Currying extension: N<A returns a
            // partial closure (drop-grabs + env capture), N>A is the
            // Elm-style peel; the N==A full-arity path is byte-identical
            // to the C VM.
            .apply => {
                if (stack.len > 0) acc = vaPop(&stack); // pop function
                if (acc.tag == .lambda) {
                    // Collect all non-mark args (stop at the mark) —
                    // alloc-free (va_pop/va_peek never allocate).
                    var nargs: i32 = 0;
                    var argbuf: [64]Value = undefined;
                    while (stack.len > 0 and vaPeek(&stack).tag != .mark) {
                        if (nargs < 64) {
                            argbuf[@intCast(nargs)] = vaPop(&stack);
                            nargs += 1;
                        } else {
                            return vm.throwShen("runtime: too many args (>64)");
                        }
                    }
                    // Pop the required mark (zinc-c always emits pushmark).
                    if (stack.len == 0 or vaPeek(&stack).tag != .mark) {
                        std.debug.print("runtime: apply missing pushmark\n", .{});
                        break :run;
                    }
                    _ = vaPop(&stack);
                    g.rootPushValueArray(&argbuf, &nargs); // root argbuf BEFORE any alloc below — C:3294

                    // Arity dispatch (currying).  zincArity never allocates;
                    // acc is the rooted slot (5), so the code read is fresh.
                    var arity = zincArity(acc.payload.lambda.code, acc.payload.lambda.code_len);
                    if (nargs > arity) {
                        // N>A — Elm-style peel: apply the first A args, feed
                        // the rest to the result, repeat.  Returns with
                        // nargs <= arity(acc) (acc callable) or nargs == 0
                        // (acc is the final value — a NON-lambda with 0
                        // args left; never read its lambda fields then).
                        try peelOverArgs(vm, &acc, &argbuf, &nargs, "apply");
                        if (acc.tag == .lambda)
                            arity = zincArity(acc.payload.lambda.code, acc.payload.lambda.code_len);
                    }

                    if (acc.tag == .lambda and (nargs == arity or nargs == 0)) {
                        // N==A — the EXISTING full-arity path, byte-identical
                        // to the C VM (frame push + env build + pc=0); all
                        // exact-arity bytecode (the Shen bundle) keeps
                        // working.  N==0 keeps the SAME old path (jump into
                        // the body with env = closure_env): zinc-c never
                        // emits a 0-arg apply, but hand-bundle top-level
                        // thunks ([pushmark 2 1 global + apply ret] bodies)
                        // are CALLED with nargs==0 via `mm <name> p` — the
                        // identity rule would break them, and the old path
                        // is the byte-compat requirement of this unit.
                        if (frames_sp >= types.CALL_STACK_DEPTH) break :run; // C:3296
                        const cf = &frame_stack[@intCast(frames_sp)];
                        frames_sp += 1;
                        cf.code = cur_code;
                        cf.code_len = cur_len;
                        cf.pc = pc + 1;
                        cf.env = env;
                        cf.env_len = env_len;
                        cf.env_cap = env_cap;
                        cf.stack = stack;
                        vaInit(g, &stack); // ALLOC — cf must not be touched after this

                        env = null;
                        env_len = 0;
                        env_cap = 0;

                        const lambda_env_len = acc.payload.lambda.env_len;
                        const new_env_len = lambda_env_len + nargs;
                        const ne = g.allocArray(Value, @intCast(new_env_len));
                        // acc is rooted (5), so acc.lambda.code/env are read
                        // AFTER the alloc and are post-GC fresh (C:3303-3305);
                        // acc.lambda.env stays reachable via the shadow stack.
                        cur_code = acc.payload.lambda.code;
                        cur_len = acc.payload.lambda.code_len;
                        const lambda_env = acc.payload.lambda.env;
                        const ne_is_oldgen = g.inOldgen(@intFromPtr(ne));
                        if (lambda_env_len > 0 and lambda_env != null) {
                            const lel: usize = @intCast(lambda_env_len);
                            @memcpy(ne[0..lel], lambda_env.?[0..lel]);
                            if (ne_is_oldgen) {
                                var j: usize = 0;
                                while (j < lel) : (j += 1) {
                                    if (gc.scan.valueReferencesNursery(g, &lambda_env.?[j])) {
                                        g.dirtyVectorsAdd(ne);
                                        break;
                                    }
                                }
                            }
                        }
                        var i: usize = 0;
                        while (i < @as(usize, @intCast(nargs))) : (i += 1) {
                            const idx = @as(usize, @intCast(lambda_env_len)) + i;
                            ne[idx] = argbuf[i];
                            if (ne_is_oldgen and gc.scan.valueReferencesNursery(g, &argbuf[i]))
                                g.dirtyVectorsAdd(ne);
                        }
                        env = ne;
                        env_len = new_env_len;
                        env_cap = new_env_len;
                        g.rootPop(); // argbuf — C:3340
                        pc = 0;
                    } else {
                        // N<A with nargs>0 (a post-peel final value can no
                        // longer land here — the peel loop only exits to a
                        // lambda or throws): NO frame push — the caller frame
                        // stays current (args + mark are already popped).
                        // buildPartialClosure re-reads everything through
                        // the rooted acc slot; the vaPush grow is covered
                        // by acc's root (5) + vaPush's internal root.
                        // (The nargs>0 tag guard is belt-and-braces: the
                        // peel only ever exits to a .lambda or throws.)
                        if (acc.tag == .lambda and nargs > 0)
                            acc = buildPartialClosure(g, &acc, &argbuf, nargs);
                        vaPush(g, &stack, acc);
                        pc += 1;
                        g.rootPop(); // argbuf
                    }
                } else if (acc.tag == .prim) {
                    // Function already popped; pop mark before args if present.
                    if (stack.len > 0 and vaPeek(&stack).tag == .mark) _ = vaPop(&stack);
                    const pn = values.primSlice(acc);
                    execPrimitive(vm, pn, &acc, &stack) catch |e| switch (e) {
                        error.Halt => break :run,
                        error.ShenError => return error.ShenError,
                    };
                    vaPush(g, &stack, acc);
                    pc += 1;
                } else {
                    // C:3332-3345 — non-callable: catchable inside
                    // trap-error, hard stop otherwise.
                    if (vm.catch_chain != null and vm.catch_chain.?.in_trap_error)
                        return vm.throwShen("apply non-callable");
                    std.debug.print("runtime: apply non-callable tag={d}", .{@intFromEnum(acc.tag)});
                    if (acc.tag == .symbol)
                        std.debug.print(" sym='{s}'", .{values.symSlice(acc)});
                    std.debug.print(" at pc={d} depth={d}\n", .{ pc, frames_sp });
                    break :run;
                }
            },

            // C:3347-3360 — return: pop the frame, push acc to the caller.
            // (The frame-restore body lives in popFramePushAcc — shared with
            // appterm's under-application paths below.)
            .ret => {
                if (!popFramePushAcc(
                    g,
                    acc,
                    frame_stack,
                    &frames_sp,
                    &cur_code,
                    &cur_len,
                    &pc,
                    &env,
                    &env_len,
                    &env_cap,
                    &stack,
                )) break :run;
            },

            // C:3361-3364 — access.
            .access => {
                const n: i32 = if (in.operand.tag == .number)
                    @intCast(in.operand.payload.number)
                else
                    in.jmp_target;
                acc = lookupEnv(n, env, env_len);
                vaPush(g, &stack, acc);
                pc += 1;
            },

            // C:3365-3378 — global lookup (defun table; the fallback
            // val_symbol interns on the C heap only — no GC alloc).
            .global => {
                const nm = if (in.operand.tag == .symbol) values.symSlice(in.operand) else "";
                acc = try vm.defunGetChecked(nm);
                vaPush(g, &stack, acc);
                pc += 1;
            },

            // C:3379-3383 — let: bind the stack top (or acc).
            .let => {
                const v: Value = if (stack.len > 0) vaPop(&stack) else acc;
                envPush(g, &env, &env_len, &env_cap, v);
                pc += 1;
            },

            // C:3384 — endlet.
            .endlet => {
                if (env_len > 0) _ = try envPop(vm, &env, &env_len);
                pc += 1;
            },

            // C:3385 — jmp.
            .jmp => pc = in.jmp_target,

            // C:3386-3391 — jmpf: pop cond, jump only on boolean false.
            .jmpf => {
                const cond: Value = if (stack.len > 0) vaPop(&stack) else acc;
                if (!(cond.tag == .boolean and cond.payload.boolean == 0)) pc += 1 else pc = in.jmp_target;
            },

            // C:3392-3396 — cur: build the closure.  valLambda roots the
            // code/env ptr slots across its env-copy alloc internally
            // (values.zig, C:299-321).  in->closure_code is read BEFORE that
            // call and `in` is never touched after; the operand stays fresh
            // because cur_code is rooted and `in` is re-derived next
            // iteration.
            .cur => {
                acc = values.valLambda(g, in.closure_code, in.closure_len, env, env_len);
                vaPush(g, &stack, acc);
                pc += 1;
            },

            // C:3397-3461 — appterm: tail-call in the current frame
            // (pc = 0, no new CallFrame — frame reuse).  Currying
            // extension: N<A builds the partial and RETURNS it via the
            // .ret frame-restore flow (the tail-call result goes to the
            // caller's caller); N>A peels; N==A is the byte-identical
            // tail-jump.
            .appterm => {
                if (stack.len > 0) acc = vaPop(&stack); // pop function
                if (acc.tag == .lambda) {
                    if (stack.len <= 0) {
                        std.debug.print("runtime: appterm empty stack\n", .{});
                        break :run;
                    }
                    var nargs: i32 = 0;
                    var argbuf: [64]Value = undefined;
                    while (stack.len > 0 and vaPeek(&stack).tag != .mark) {
                        if (nargs < 64) {
                            argbuf[@intCast(nargs)] = vaPop(&stack);
                            nargs += 1;
                        } else {
                            return vm.throwShen("runtime: appterm too many args (>64)");
                        }
                    }
                    // zinc-t always emits pushmark — required.
                    if (stack.len == 0 or vaPeek(&stack).tag != .mark) {
                        std.debug.print("runtime: appterm missing pushmark\n", .{});
                        break :run;
                    }
                    _ = vaPop(&stack); // pop mark
                    if (nargs == 0) {
                        std.debug.print("runtime: appterm zero args\n", .{});
                        break :run;
                    }

                    g.rootPushValueArray(&argbuf, &nargs); // root argbuf before any alloc

                    // Same arity dispatch as apply (currying).
                    var arity = zincArity(acc.payload.lambda.code, acc.payload.lambda.code_len);
                    if (nargs > arity) {
                        try peelOverArgs(vm, &acc, &argbuf, &nargs, "appterm");
                        // peel may return a NON-lambda final value with
                        // nargs == 0 — never read its lambda fields then.
                        if (acc.tag == .lambda)
                            arity = zincArity(acc.payload.lambda.code, acc.payload.lambda.code_len);
                    }

                    if (acc.tag == .lambda and nargs == arity) {
                        // N==A — tail-jump path (pc = 0, no new CallFrame —
                        // frame reuse).  M11: REUSE the current env array when
                        // it fits.  The old env is DEAD here: a closure Value
                        // holds only a valLambda COPY of its env (values.zig),
                        // saved CallFrames hold OTHER arrays (ownership moves
                        // atomically at apply push / popFramePushAcc), and
                        // nested vmExecEnv entries COPied their init_env in the
                        // prologue — so the current array's ONLY referee is the
                        // rooted &env slot (3).  Reuse is therefore safe with
                        // no liveness analysis, provided the dead tail is
                        // cleared (the drain scans VALUE_ARRAYs by full
                        // capacity) and env_cap stays the TRUE physical
                        // capacity (envPush growth math reads it).
                        const lambda_env_len = acc.payload.lambda.env_len;
                        const new_env_len = lambda_env_len + nargs;
                        const reuse = env != null and env_cap >= new_env_len;
                        const ne: [*]Value = if (reuse) env.? else g.allocArray(Value, @intCast(new_env_len));
                        // new_cap is the array's TRUE physical capacity: on
                        // reuse that is the retained env_cap (NOT new_env_len —
                        // faking it would break envPush's grow bounds); on a
                        // miss it is the exact-size alloc.
                        const new_cap: i32 = if (reuse) env_cap else new_env_len;
                        if (reuse) {
                            if (env_cap > new_env_len) {
                                // Tail clear: [new_env_len..env_cap) still holds
                                // the dead caller's stale refs; nil them so the
                                // full-capacity drain scan pins nothing.
                                const tail: usize = @intCast(new_env_len);
                                const cap: usize = @intCast(env_cap);
                                @memset(ne[tail..cap], values.valNil());
                            }
                            vm.env_reuse_hits += 1;
                        } else {
                            vm.env_reuse_misses += 1;
                        }
                        // NO-ALLOC WINDOW (load-bearing): between `ne = env.?`
                        // above and `env = ne` below there must be NO GC
                        // allocation — `ne` is a raw copy of the rooted slot's
                        // pointer and would go stale if a collect moved the
                        // array.  Today every statement here is alloc-free
                        // (memcpy/stores are no-ops, dirtyVectorsAdd uses
                        // page_allocator), so `ne` stays fresh.
                        // cur_code set AFTER any alloc from the rooted acc — the
                        // rooted cur_code slot (7) makes any interim value safe,
                        // and reading acc fresh after the alloc matches C:3423-3425.
                        cur_code = acc.payload.lambda.code;
                        cur_len = acc.payload.lambda.code_len;
                        const lambda_env = acc.payload.lambda.env;
                        const ne_is_oldgen = g.inOldgen(@intFromPtr(ne));
                        if (lambda_env_len > 0 and lambda_env != null) {
                            const lel: usize = @intCast(lambda_env_len);
                            @memcpy(ne[0..lel], lambda_env.?[0..lel]);
                            if (ne_is_oldgen) {
                                var j: usize = 0;
                                while (j < lel) : (j += 1) {
                                    if (gc.scan.valueReferencesNursery(g, &lambda_env.?[j])) {
                                        g.dirtyVectorsAdd(ne);
                                        break;
                                    }
                                }
                            }
                        }
                        var i: usize = 0;
                        while (i < @as(usize, @intCast(nargs))) : (i += 1) {
                            const idx = @as(usize, @intCast(lambda_env_len)) + i;
                            ne[idx] = argbuf[i];
                            if (ne_is_oldgen and gc.scan.valueReferencesNursery(g, &argbuf[i]))
                                g.dirtyVectorsAdd(ne);
                        }
                        env = ne;
                        env_len = new_env_len;
                        env_cap = new_cap;
                        g.rootPop(); // argbuf — C:3451
                        pc = 0;
                    } else {
                        // N<A: the partial closure (or post-peel final value)
                        // IS the tail-call result — return it to the caller's
                        // caller through the .ret frame-restore flow
                        // (reference interp.shen:225/238; at top level
                        // frames_sp==0 breaks :run and acc becomes the
                        // program value).
                        if (acc.tag == .lambda and nargs > 0)
                            acc = buildPartialClosure(g, &acc, &argbuf, nargs);
                        g.rootPop(); // argbuf
                        if (!popFramePushAcc(
                            g,
                            acc,
                            frame_stack,
                            &frames_sp,
                            &cur_code,
                            &cur_len,
                            &pc,
                            &env,
                            &env_len,
                            &env_cap,
                            &stack,
                        )) break :run;
                    }
                } else if (acc.tag == .prim) {
                    // Function already popped; pop mark before args if present.
                    if (stack.len > 0 and vaPeek(&stack).tag == .mark) _ = vaPop(&stack);
                    const pn = values.primSlice(acc);
                    execPrimitive(vm, pn, &acc, &stack) catch |e| switch (e) {
                        error.Halt => break :run,
                        error.ShenError => return error.ShenError,
                    };
                    vaPush(g, &stack, acc);
                    pc += 1;
                } else {
                    if (vm.catch_chain != null and vm.catch_chain.?.in_trap_error)
                        return vm.throwShen("appterm non-lambda");
                    std.debug.print("runtime: appterm non-lambda\n", .{});
                    break :run;
                }
            },

            // C:3462 — unknown op (OP_COUNT never appears as a real opcode;
            // char_to_opcode maps unknown chars to it at parse time).
            .count => {
                std.debug.print("runtime: unknown op {d} at pc={d}\n", .{ @intFromEnum(in.op), pc });
                break :run;
            },
        }
    }

    // done: (C:3463-3468) — the 8 root pops are the defer rootPopTo above;
    // frame_stack is GC-allocated (no free needed); acc is returned.
    vaFree(&stack);
    return acc;
}

/// C: zincvm.c:3470-3472 vm_exec — top-level entry with an empty env.
pub fn vmExec(vm: *Vm, code: ?*types.Instr, code_len: i32) VmError!Value {
    return vmExecEnv(vm, code, code_len, null, 0);
}
