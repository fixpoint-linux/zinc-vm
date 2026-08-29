# zinc-vm

The shared ZINC VM executor for fixpoint-linux — a single, high-throughput
interpreter + moving generational GC consumed by the language stacks that run
on the system.

A Zig 0.16 native library package. Exposes two modules:

- `gc` — the moving generational collector (`src/gc.zig`: types/heap/collect/scan/roots)
- `vm` — the ZINC interpreter + prims (`src/vm.zig`: state/values/symbols/tables/parser/interp/prims/streams/execplan/hostcall/marshal)

## Consumers

- **fx-ui** — Elm → ZINC-csexp compiler + runtime (`src/effectloop.zig` is consumer-side)
- **shen** — the self-hosting Shen OS (shensh, zincdec)

Both consume this package via a `build.zig.zon` path dependency (later a git
submodule / `zig fetch`).

## The shared-executor consolidation

The org previously had two divergent copies of this VM (fx-ui's and shen's).
This package is the single canonical source: fx-ui's newer base (M6–M11 perf:
frame-stack pool, tail-env reuse, single-probe global lookup) reconciled with
shen's eval-kl chain (marshal, catching hostcalls, wait/kill). Arithmetic prims
deliberately carry **no type guards** (bare `+%`/`-%`/`*%`/`@divTrunc`
semantics) — per AGENTS.md the metacircular interpreter relies on it and type
errors belong to the safe-wrapper layer, not the VM.

## Build & test

```sh
zig build            # library
zig build gate       # gc+vm tests in Debug/ReleaseSafe/ReleaseFast (3 × 138)
zig build test       # Debug tests
```

## Repo conventions

- Native-only (no wasm target).
- `build.zig` exposes `gc` and `vm` via `b.addModule` for package consumers;
  per-mode gate instances use `b.createModule` (independent optimize).
- Commits follow conventional style (`feat:`/`fix:`/`chore:`/`refactor:`).
