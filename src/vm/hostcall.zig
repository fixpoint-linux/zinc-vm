//! src/vm/hostcall.zig — host-side calls into BUNDLED closures (milestone M9).
//!
//! C origin (via shen/zig/src/vm/hostcall.zig): the env-extend call pattern
//! shared by zincvm.c:3483-3528 call_closure1/call_closure3 and the eval-kl
//! inline stages (defun_get → VAL_LAMBDA check → GC_VALUE_ARRAY(env_len+n) →
//! memcpy env → append args → vm_exec_env).  This is the exact interp-apply
//! rooting pattern, repurposed for the HOST (tools/elmvm.zig + the M9 async
//! effect loop): apply a bundled closure WITHOUT running it inside a bytecode
//! snippet — instead a FRESH vmExecEnv call with the closure's captured env
//! extended by the call args.
//!
//! CONVENTION (why the args ride AFTER the captured env): a bundled closure
//! reads its parameter via `access N`, which indexes the environment from the
//! END (reverse-index lookup) — appending the args after env_len makes the
//! last arg access 0 exactly like a normal call would have pushed it.
//!
//! TWO FLAVORS (plan M9):
//!   - applyBundledN — by-NAME: resolve the global defun and call it (the
//!     `main` entry the host drives).  Returns null when `name` does not
//!     resolve to a bundled lambda; the caller decides the fallback.
//!   - applyClosureN — by-VALUE: call a closure that arrives as a Value
//!     inside a data structure (the M9 Task/Program vectors carry the
//!     continuation/update closures as Values).  Same body minus the
//!     defunGet/resolve step; asserts the value is a .lambda.
//!
//! Both are NON-catching: error.ShenError propagates to the caller (which may
//! push its own CatchSite), error.Halt is contained inside vmExecEnv.

const std = @import("std");
const gc = @import("gc");
const types = gc.types;
const state = @import("state.zig");
const values = @import("values.zig");
const interp = @import("interp.zig");

const Gc = gc.Gc;
const Value = types.Value;
const Vm = state.Vm;
const VmError = state.VmError;

/// Host-call arg budget: the M9 call sites are 1 (continuation/handler) or 2
/// (update msg model) args; 8 leaves headroom without unbounded stack use.
const MAX_HOSTCALL_ARGS = 8;

/// C: zincvm.c:3483-3528 call_closure1/call_closure3 + the eval-kl stages —
/// the shared env-extend pattern, generalized to N args.  ROOTING (C parity):
/// fn + the arg array are rooted across the env allocArray (call_closure1
/// roots g/arg at :3489-3490, call_closure3 roots all four at :3510-3513);
/// ROOT_VALUE_ARRAY is the array form of that discipline (the eval loop's
/// apply/appterm argbuf uses it, C:3294/3421).  Reads of fn.lambda.env/code
/// after the alloc go through the rooted fnv (fresh even if it collected);
/// the two rootPops before the vmExecEnv call never allocate, so reading
/// fnv afterwards is safe (exact C:3497-3499 shape).  The write barrier
/// fires per stored arg that references the nursery into an oldgen env
/// (call_closure3's :3518-3522 per-arg checks).
pub fn applyBundledN(vm: *Vm, name: []const u8, args: []const Value) VmError!?Value {
    std.debug.assert(args.len <= MAX_HOSTCALL_ARGS);
    const fnv = vm.defunGet(name);
    if (fnv.tag != .lambda) return null;

    return applyClosureN(vm, fnv, args);
}

/// C: the call_closure1/call_closure3 body WITHOUT the defunGet/resolve step —
/// the fnv is already a .lambda VALUE (the continuation/update closures inside
/// the M9 Task/Program vectors).  Rooting identical to applyBundledN; the
/// caller must pass args whose interior pointers are valid at entry (they are
/// copied into the rooted argbuf before the first allocating call).
pub fn applyClosureN(vm: *Vm, fnv_in: Value, args: []const Value) VmError!Value {
    const g = vm.gc;
    std.debug.assert(args.len <= MAX_HOSTCALL_ARGS);
    var fnv = fnv_in;
    std.debug.assert(fnv.tag == .lambda);

    g.rootPushValue(&fnv);
    var argbuf: [MAX_HOSTCALL_ARGS]Value = undefined;
    var nargs: i32 = 0;
    for (args) |av| {
        argbuf[@intCast(nargs)] = av;
        nargs += 1;
    }
    g.rootPushValueArray(&argbuf, &nargs);

    const env_len = fnv.payload.lambda.env_len;
    const total: i32 = env_len + nargs;
    const env = g.allocArray(Value, @intCast(total));
    const env_is_oldgen = g.inOldgen(@intFromPtr(env));
    if (env_len > 0) {
        const lel: usize = @intCast(env_len);
        const lambda_env = fnv.payload.lambda.env.?;
        @memcpy(env[0..lel], lambda_env[0..lel]);
        // M5 fix: barrier the COPIED env elements too, not just the appended
        // args (mirrors interp.zig apply's copied-env barrier).
        if (env_is_oldgen) {
            var j: usize = 0;
            while (j < lel) : (j += 1) {
                if (gc.scan.valueReferencesNursery(g, &lambda_env[j])) {
                    g.dirtyVectorsAdd(env);
                    break;
                }
            }
        }
    }
    var i: usize = 0;
    while (i < @as(usize, @intCast(nargs))) : (i += 1) {
        const idx = @as(usize, @intCast(env_len)) + i;
        env[idx] = argbuf[i];
        if (env_is_oldgen and gc.scan.valueReferencesNursery(g, &argbuf[i]))
            g.dirtyVectorsAdd(env);
    }
    g.rootPop(); // argbuf
    g.rootPop(); // fnv (pops never allocate — the fnv read below is safe)
    return try interp.vmExecEnv(
        vm,
        fnv.payload.lambda.code,
        fnv.payload.lambda.code_len,
        env,
        total,
    );
}
