# Repo layout

Two rules shape this tree:

1. **The runtime path is flat and readable.** No nesting below two levels, no per-feature
   subdirectories, no `impl/` or `detail/` folders. You should be able to list `include/soma/` and see
   the entire engine.
2. **Python is grouped and unmistakably non-runtime.** Everything under `tools/admission/` is
   admission-time only. The engine never spawns Python, never links it, and never requires it to be
   installed.

---

## Tree

```
include/soma/            invariant core — flat, no subdirectories
include/soma/arch/       per-architecture backends (gqa.hpp, mla.hpp)
src/soma/                mm_soma static lib + the header self-test
src/soma/arch/           per-architecture translation units
src/soma/arch_registry.cpp   the ONE core TU permitted to include soma/arch/
src/soma/main.cpp        `soma` executable: serve | plan --json | admit-verify

include/{common,node,control}/   Mantic-Mind, unchanged in shape
src/{common,node,control}/

tools/admission/         Python — offline only, never a runtime dependency
tools/ci/                check_seam.py and friends
tools/*.toml             config templates (existing convention)

schemas/arch-ir.md       the architecture IR spec
schemas/container.md     the on-disk container format
schemas/registry/*.sql   versioned migrations

docs/architecture.md            the seam, tiers, scheduler, before/after
docs/external-api.md            the full /v1/* surface and scopes
docs/mantic-mind-integration.md engine↔node, verdict routing, FTXUI
docs/roadmap.md                 G0–G8 validation gates
docs/repo-layout.md             this file

tests/soma/              mm_soma_tests + the conformance ladder
tests/fixtures/tiny/     tiny-random checkpoints, one per family, committed
```

---

## Why the seam is a directory boundary and not a naming convention

`include/soma/arch/` exists so the dependency rule can be checked by a script rather than remembered by
a person:

- **R1** — no core file includes `soma/arch/…`. Only the backends, `arch_registry.cpp`, and the tests may.
- **R2** — no architecture identifier appears in core *code*, with `arch_ir.hpp` allow-listed because
  the IR is a *description* of an architecture, not logic that executes one.

Both are enforced by [`tools/ci/check_seam.py`](../tools/ci/check_seam.py) on every commit. See
[architecture.md §11](architecture.md).

`arch_registry.cpp` is the single exception, and it is a small one by design: it holds
`resolve_f32_backend()` and `resolve_attention_backend()` and nothing else. Those are the only two
resolution functions that switch on `AttentionFamily`, and they run once per model load. A switch on
family anywhere in a loop is a seam violation regardless of which file it is in.

---

## `tools/admission/` — offline only

| Script | Purpose |
|---|---|
| `convert.py` | HF checkpoint → canonical container + sidecar index + conversion metadata |
| `make_oracle.py` | Tiny-random model with the **real** arch config + `transformers` teacher-forcing oracle |
| `compile_tokenizer.py` | `tokenizer.json` → the normalized C++-side format |
| `profile_streaming.py` | Streaming economics → the verdict |
| `profile_lookahead.py` | Per-layer router-lookahead recall → `pilot_profile` |
| `autotune.py` | Kernel selection over the model's shape set → `kernel_choice` |
| `bootstrap_heat.py` | Calibration-corpus routing histogram → `expert_heat` |

**The contract:** these write to the registry and the model directory. Clients read those through the
API. Python is never in a request path, never a link dependency, and never required at runtime — which
is what makes "admission is offline" a structural fact rather than a stated intention.

This follows the existing precedent (`tools/qwen_tts_service.py`): stdlib-first, `#!/usr/bin/env
python3`, `from __future__ import annotations`, `argparse`, fully annotated, invoked by an explicit
`python tools/…` command. Unlike the TTS sidecar, admission tooling **does** need `torch` and
`transformers` — so it gets a `tools/admission/requirements.txt`, the first in this repo, and it is
never installed on a serving host.

---

## Where new code goes

| Adding… | Goes in | Also touch |
|---|---|---|
| A new architecture | `include/soma/arch/<name>.hpp` + `src/soma/arch/<name>.cpp` | `arch_registry.cpp`, a tiny fixture, `SOMA_ARCH_HEADERS` in `src/soma/CMakeLists.txt` |
| A core capability | `include/soma/<name>.hpp` + `src/soma/<name>.cpp` | `SOMA_CORE_HEADERS` |
| A `/v1/*` route | `src/control/control_api_server.cpp` | **`route_scope_table()`** — startup fails without an entry |
| A registry column | `schemas/registry/NNN_*.sql` | Never edit an applied migration |
| An admission pass | `tools/admission/` | The registry table it writes |

The `route_scope_table()` requirement is deliberate friction. A route with no scope entry is a startup
failure rather than a default, because defaulting an unlisted route to `read` silently under-protects a
new mutation and defaulting to `operator` silently breaks a new GET.

---

## Conventions inherited from the existing repo

These are not re-litigated:

- **Explicit source lists.** There is no `file(GLOB)` anywhere in this project and Soma does not
  introduce one.
- **`/W4 /WX` and `-Wall -Wextra -Wpedantic -Werror` are global and unconditional.** They apply to
  `src/soma` automatically. There is precedent (commit `b5df2c4`) for Windows-only code breaking the
  Linux `-Werror` build, so the header self-test runs on GCC, Clang, and MSVC.
- **New dependencies** go in `vcpkg.json` with `version>=` plus a `find_package` in the **root**
  `CMakeLists.txt`, following the CONFIG → pkg-config → manual-find chain with an apt hint.
- **Config keys** live in `tools/*.toml` templates and are copied to the working directory; local copies
  are gitignored.

One convention **is** changed: `ConfigFile` currently parses `[section]` headers and discards them, so
all keys are flat. With two engines that forces a `soma_*` prefix explosion on top of the existing 13
`llama_*` fields. Sections become `section.key`; files without a header keep working byte-identically.
See [mantic-mind-integration.md §4](mantic-mind-integration.md).
