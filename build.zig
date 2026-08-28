const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // ---- Shen GC module (plan DECISION 3) ----
    // The collector is a standalone importable module rooted at src/gc.zig.
    // Named with b.addModule so consumers can `b.dependency(...).module("gc")`.
    const gc_mod = b.addModule("gc", .{
        .root_source_file = b.path("src/gc.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ---- Shen ZINC VM module (src/vm.zig re-exports parser/interp/state) ----
    // Also b.addModule (public package API); vm imports gc.
    _ = b.addModule("vm", .{
        .root_source_file = b.path("src/vm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "gc", .module = gc_mod } },
    });

    const test_step = b.step("test", "Run tests");

    // ---- Shen GC test step + permanent multi-mode gate (units A-C) ----
    // addGcTestSet creates a fully self-contained gc module + gc_test module +
    // addTest + run + T9 expected-panic exe for ONE hardcoded optimize mode,
    // and returns the run step.  std.debug.assert inside the gc module is gated
    // by THAT module's own optimize, so the same source is exercised under
    // every mode the gate cares about — that is what makes ReleaseSafe a real
    // safety gate rather than a Debug-only check.
    const gc_test_step = b.step("gc-test", "Run Shen GC tests (honours -Doptimize)");
    gc_test_step.dependOn(addGcTestSet(b, target, optimize));
    test_step.dependOn(gc_test_step);

    // ---- Shen VM test step (plan M0): same shape as gc-test. ----
    const vm_test_step = b.step("vm-test", "Run Shen VM tests (honours -Doptimize)");
    vm_test_step.dependOn(addVmTestSet(b, target, optimize));
    test_step.dependOn(vm_test_step);

    // ---- `gate`: the permanent ReleaseSafe build gate (unit C) ----
    // Runs the full Shen GC + VM suites in Debug + ReleaseSafe + ReleaseFast in
    // one command.  ReleaseSafe keeps std.debug.assert live, so every
    // safety-enforcement added in the GC units B/E is proven under the gate,
    // not just in Debug.
    const gate_step = b.step("gate", "Run Shen GC + VM tests in Debug + ReleaseSafe + ReleaseFast");
    gate_step.dependOn(addGcTestSet(b, target, .Debug));
    gate_step.dependOn(addGcTestSet(b, target, .ReleaseSafe));
    gate_step.dependOn(addGcTestSet(b, target, .ReleaseFast));
    gate_step.dependOn(addVmTestSet(b, target, .Debug));
    gate_step.dependOn(addVmTestSet(b, target, .ReleaseSafe));
    gate_step.dependOn(addVmTestSet(b, target, .ReleaseFast));
}

/// SAFETY-ENFORCEMENT (unit C): build one self-contained Shen GC test set
/// compiled at `opt` and return its run step.  Because each mode needs its own
/// gc module (std.debug.assert inside the collector is gated by that module's
/// optimize), every call builds an independent gc_mod + gc_test_mod + addTest +
/// run + T9 expected-panic exe.  Named top-level modules (b.addModule) are NOT
/// used here to avoid duplicate "gc" module names across the 3 gate instances;
/// the unnamed modules (b.createModule) carry their own .optimize.
fn addGcTestSet(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    opt: std.builtin.OptimizeMode,
) *std.Build.Step {
    const gc_mod = b.createModule(.{
        .root_source_file = b.path("src/gc.zig"),
        .target = target,
        .optimize = opt,
    });

    const gc_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/gc_test.zig"),
        .target = target,
        .optimize = opt,
        .imports = &.{ .{ .name = "gc", .module = gc_mod } },
    });
    const gc_tests = b.addTest(.{ .root_module = gc_test_mod });
    const run_gc_tests = b.addRunArtifact(gc_tests);

    // ---- T9: expected-panic executable (plan DECISION 7) ----
    // Zig 0.16 has no in-process panic assertion, so the ROOT_PTR
    // interior-pointer defense (gc.c:1527-1539) is proven by a tiny exe that
    // overrides its root panic handler, matches the defense message, and
    // exits 42; the Run step expects exactly that.  Panic exit paths through
    // abort() are signal-based (nondeterministic for expect_term), hence the
    // handler-normalized exit code.
    const t9_mod = b.createModule(.{
        .root_source_file = b.path("tests/root_ptr_panic.zig"),
        .target = target,
        .optimize = opt,
        .imports = &.{ .{ .name = "gc", .module = gc_mod } },
    });
    const t9_exe = b.addExecutable(.{
        .name = "gc_root_ptr_panic",
        .root_module = t9_mod,
    });
    const run_t9 = b.addRunArtifact(t9_exe);
    run_t9.expectExitCode(42);

    // The T9 exe runs as part of this mode's gc-test set (before the tests).
    run_gc_tests.step.dependOn(&run_t9.step);

    return &run_gc_tests.step;
}

/// Build one self-contained Shen VM test set compiled at `opt` and return its
/// run step (plan M0, mirroring addGcTestSet).  Each mode gets its OWN unnamed
/// gc module + vm module + vm_test module (b.createModule, so no top-level
/// "gc"/"vm" module-name clashes across the gate's three instances); the vm
/// module imports the mode's gc module and the vm_test module imports both.
fn addVmTestSet(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    opt: std.builtin.OptimizeMode,
) *std.Build.Step {
    const gc_mod = b.createModule(.{
        .root_source_file = b.path("src/gc.zig"),
        .target = target,
        .optimize = opt,
    });

    const vm_mod = b.createModule(.{
        .root_source_file = b.path("src/vm.zig"),
        .target = target,
        .optimize = opt,
        .link_libc = true,
        .imports = &.{ .{ .name = "gc", .module = gc_mod } },
    });

    const vm_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/vm_test.zig"),
        .target = target,
        .optimize = opt,
        .link_libc = true,
        .imports = &.{
            .{ .name = "gc", .module = gc_mod },
            .{ .name = "vm", .module = vm_mod },
        },
    });
    const vm_tests = b.addTest(.{ .root_module = vm_test_mod });
    const run_vm_tests = b.addRunArtifact(vm_tests);

    return &run_vm_tests.step;
}
