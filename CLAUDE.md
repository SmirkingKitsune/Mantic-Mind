# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository
**and in its two sibling projects**, which live beside it rather than in it.

It is the canonical copy and it is versioned here on purpose: guidance that is not in a repository is
guidance that drifts from the code without a diff to show for it. `../CLAUDE.md` is now only a
pointer to this file. **A session started in `../Mafia-Machine/` or `../Yappy-Agent/` does not load
this file automatically** — it sees that pointer — so open this one explicitly when working there.

## Repository Overview

Three related projects share the parent directory; only the first is this repository:

- **Mantic-Mind** — Distributed LLM inference cluster, plus **Soma**, its own streaming MoE inference engine (C++20, CMake + vcpkg). This repo.
- **Mafia-Machine** — LLM agent Mafia game that uses Mantic-Mind as its backend (Python 3.11+). Sibling directory, separate tree.
- **Yappy-Agent** — Responsive SvelteKit web UI for Mantic-Mind control chat, memory curation, read-only status, and a local app launcher (Node/TypeScript). Sibling directory, separate tree.

---

## Mantic-Mind (C++)

### Build

Requires `VCPKG_ROOT` environment variable set to your vcpkg installation.

```sh
# Configure (vcpkg installs dependencies automatically)
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build (Debug by default)
cmake --build build

# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

Binaries output to:
- `build/src/node/Debug/mantic-mind[.exe]`
- `build/src/control/Debug/mantic-mind-control[.exe]`
- `build/src/soma/Debug/soma[.exe]`

`soma` is also **copied beside** `mantic-mind` and `mantic-mind-control` at build time, and installs
into the same `bin`. Both resolve it by name — the node spawns it, control runs `soma plan` and
`soma conform` during admission — and neither directory is on the other's PATH, so the copy is what
makes "soma ships with the node" true rather than merely documented.

Tests are CTest targets; run `ctest --test-dir build --output-on-failure`.

### Dependencies (via vcpkg)

ftxui ≥5.0, SQLiteCpp ≥3.3.1, cpp-httplib ≥0.15.3, nlohmann-json ≥3.11.3, spdlog ≥1.14.1, OpenSSL ≥3.3.0

### Source Layout

```
src/
  common/   — shared library: agent, agent_db, config_file, conversation_manager,
              engine_capabilities, engine_client, engine_config, footprint,
              http_client, http_server, logger, memory_manager, node_discovery,
              pairing, process_exec, sse_emitter, util
  node/     — mantic-mind executable: node_state, runtime_process, engine_process,
              engine_supervisor, engine_descriptor, engine_provisioner,
              engine_manager, node_api_server, node_ui, llama_runtime,
              llama_cpp_provisioner, model_store, kv_checkpoint_backend, main
  control/  — mantic-mind-control executable: agent_manager, agent_queue,
              agent_scheduler, agent_config_validator, control_api_server,
              control_ui, admission_panels, engine_config_store, engine_panels,
              node_registry,
              model_registry, route_scope, soma_dashboard, soma_panels,
              performance_tracker, main
  soma/     — mm_soma static lib + soma executable: arch_ir, plan, conformance,
              container, expert_store, memory_hierarchy, kv_cache, scheduler,
              f32_model, f32_backend, quant, tokenizer, telemetry, autotune,
              arch_registry, server, main
  soma/arch/— per-architecture backends: gqa.cpp (MHA/GQA/GQA+BSA), mla.cpp (MLA/MLA+DSA),
              gdn.cpp, kda.cpp, compressed_sparse.cpp, deepseek_dspark.cpp
include/
  common/   — headers mirroring src/common
  node/     — node-specific headers (node_config.hpp, node_state.hpp, node_ui.hpp, …)
  control/  — control-specific headers (control_config.hpp, agent_scheduler.hpp, …)
  soma/     — the invariant engine core, flat (arch_ir.hpp, plan.hpp, serve.hpp, …)
  soma/arch/— the per-family seam (gqa.hpp, mla.hpp)
```

### Architecture

Three-tier system:
1. **engine subprocess** (`:8080`) — either `llama-server` (llama.cpp, external) or `soma` (in-tree, see below). Both are registered in the node's `EngineRegistry` at startup and a load request names which one it wants; both speak OpenAI-compatible HTTP, which is why the node needs only one supervision path
2. **mantic-mind** node (`:7070`) — spawns and supervises engine subprocesses via `EngineSupervisor`; exposes Node API; shows FTXUI status TUI
3. **mantic-mind-control** (`:9090`) — cluster head; routes requests to nodes; manages agents/conversations/memories via per-agent SQLite databases (`data/agents/{uuid}/agent.db`); exposes External Client API; shows FTXUI management TUI

Key control-side components:
- `AgentManager` — CRUD + per-agent SQLite via SQLiteCpp
- `NodeRegistry` — node list + health polling; also the **convergence loop** for engine config (see below)
- `AgentScheduler` — placement engine: existing/suspended placement → preferred → stored → VRAM → distribute → evict+retry; serializes scheduling on one mutex, tracks placements behind a separate state mutex. Owns backend selection; there is no separate router class. (`include/control/placement_engine.hpp` describes a replacement that was never implemented — roadmap D46.)
- `AgentQueue` — per-agent FIFO worker threads to serialize chat requests
- `EngineConfigStore` — the cluster's engine policy, persisted to `data/engine_config.json`
- `ControlApiServer` — REST endpoints + SSE chat proxy

### Master engine configuration

Control (the **master**) owns one engine policy for the whole cluster; nodes conform to it. Before
this, every engine setting lived in each node's `.toml` and control could observe the result but
never drive it — ten nodes could run ten different llama.cpp builds and all ten rendered healthy.

- **Configuration is intent, never per-machine fact.** `ClusterEngineConfig` names the primary
  engine, an *optional* backup (default `llama-cpp`, explicitly clearable), versions, install
  method, and update policy. It carries **no accelerator, no `cuda_arch`, no paths** — those are
  resolved per node by `LlamaEngineProvisioner`. Sending one is a 400 naming the field, not an
  ignored key: silently dropping it would leave the operator believing the cluster was told.
- **First run is forced.** No config → the TUI opens its Engines tab modally, `--mode cli` runs
  `engines setup`, and placement refuses with a message naming the fix. The API stays up either
  way, so an automated deploy can `PUT` the config instead.
- **Convergence rides the existing health poll.** Nodes report `engine_config_version`; a mismatch
  triggers a push. No new thread, and a node that was offline during a change converges on its next
  successful poll.
- **Non-conformance blocks placement, not registration.** A node reports
  `unconfigured | converging | conforming | drifted | failed`; only `conforming` receives agents.
  The gate lives in `NodeRegistry::available_nodes()` / `nodes_with_capacity()`, and applies only
  when an engine-config provider is set — an unmanaged registry is not silently placed nowhere.
- **Built engines are shared node-to-node, brokered by control.** Source packages and hashes,
  control relays the digest to the target and mints a scoped one-shot credential, source pushes,
  control revokes. A leaked credential cannot substitute different bytes; a *compromised source*
  still supplies both artifact and hash, which only signed builds would fix.

Node side: `NodeEngineManager` owns one `EngineProvisioner` per engine (`LlamaEngineProvisioner`
wraps the existing 2200-line `LlamaCppProvisioner` unchanged; `SomaEngineProvisioner` resolves the
in-tree binary and does not pretend to build). It provisions **exactly** `required_engines()` —
which is what makes the backup genuinely optional rather than merely unused.

### Soma (streaming MoE engine)

`soma` is a from-scratch inference engine for mixture-of-experts models too large to hold in
RAM: the dense half stays resident, routed experts stream from disk per token. It speaks
OpenAI-compatible HTTP (`/health`, `/v1/models`, `/v1/chat/completions`, default `:8080`), so
the node supervises Soma and llama.cpp behind the *same* boundary rather than growing a
parallel universe.

```sh
soma serve   --model-dir DIR [--host H] [--port N] [--ctx-size N] [--ram-budget BYTES]
                             [--pin BYTES] [--kv-dir DIR] [--quant-dense DTYPE] [--served-name NAME]
soma plan    --model-dir DIR [--json] [--quant|--expert-down|--quant-dense DTYPE] [--group N]
                             [--ram|--ram-free|--disk-bw SIZE] [--ctx N] [--min-tok-s RATE]
soma conform --model-dir DIR [--json]
```

Every `serve` field is settable by both a flag and an env var (`SOMA_PORT`, `SOMA_MODEL_DIR`, …)
— the node spawns it as a subprocess, and argv quoting across Windows and POSIX is a worse
place to find a mistake than an environment block.

**The verdict.** `plan` reads headers only and allocates nothing, so the scheduler can call it
on a node that could not host the model at all. Its answer is a property of
`(model, quantization, host budget)` — *never* of the model alone; the same weights at two
quantizations, or on two disks, get two verdicts. The `--quant*` and `--ram*`/`--disk-bw` flags
ask about a hypothetical and convert nothing. Token rate is ceilinged by
`disk_bandwidth / bytes_per_token`.

**The seam.** `include/soma/` is invariant core and may not name an architecture; only
`src/soma/arch/*.cpp` and `arch_registry.cpp` may. `tools/ci/check_seam.py` enforces it. Attention
families that ship: GQA (Qwen3-MoE, Qwen2-MoE, Mixtral, OLMoE), MLA (DeepSeek-V2/V3), MLA+DSA
(GLM-5.2 and GLM-5.3 — one family, one fixture: 5.3 is the same base model re-post-trained, and its
`config.json` differs from 5.2's only in `transformers_version`), compressed+sparse (DeepSeek-V4),
GQA+BSA (MiniMax-M3), and two HYBRIDS whose stacks mix
cached and recurrent layers — MLA+KDA (Kimi-K3) and GQA+GDN (Qwen3.5-MoE). A hybrid's per-sequence
cache is affine, not linear, in context: `kv_bytes_for_context` is the only honest figure for one.
BSA is block-sparse SELECTION over ordinary GQA: every token is still cached, its indexer key widens
the K plane, and a query attends `topk_blocks * block_size` keys — 2048 on MiniMax-M3, not 16.
`F32Backend` is *the* production execution path — "fp32" names its activations, not its weights,
which run quantized SIMD kernels.

### Adding a model NEVER removes one

**Not negotiable, and not a judgement call.** Every family listed above is one somebody proved: a
committed fixture, an oracle built from that model's own reference implementation, and a gate that
grades against it. Support that took a week to establish can be removed by one careless edit to a
shared table, and the removal is usually SILENT — the container still converts, the model still
loads, the answers still read fluently. This has already been paid for once: Kimi-K3's engine side was
complete and correct while `convert.py` lacked twenty of its tensor roles, so no Kimi container could
be produced by any route, and re-establishing that support was work nobody should have had to do
twice.

Three rules, in the order they come up:

1. **Never narrow an existing family to widen a new one.** If a shared function, table, dialect,
   refusal, or fixture has to change shape for a new model, the change must be ADDITIVE for every
   family already through the seam. A refusal added for a new checkpoint that also starts refusing an
   old one is a regression, not a stricter gate. When additive is genuinely impossible, stop and say
   so rather than trading one model for another.

2. **Reuse the component; rename it, do not fork it.** When the right implementation for a new family
   already exists under a model-specific name, make the NAME generic and keep the ONE implementation.
   Never copy it under a second name, and never leave the new family reaching into something that
   reads as another model's private property. Rename first, then comment which families require the
   behaviour and why. `ArchIr::routed_expert_width()` is the pattern: Kimi-K3's latent MoE needed it,
   the name says what it IS rather than who asked for it, and it returns `d_model` for every family
   with no latent space — so the ordinary path is untouched. `resolve_f32_backend` is the same move at
   the seam: `MlaDsa` and `GqaBsa` share `arch::mla` and `arch::gqa` rather than growing copies, each
   with a comment naming the hazard if that sharing were wrong. Two implementations of one behaviour
   is how two families come to disagree, and the one that drifts is always the one nobody is currently
   working on.

3. **Another model's fixture is not scratch space.** `tests/fixtures/tiny/*`,
   `tests/fixtures/tokenizers/*` and `tests/fixtures/containers/*` are the evidence, not the
   scaffolding. Do not regenerate, retune, or "fix" another family's committed fixture as a side
   effect of adding yours. If one genuinely must change — it was stale, or a format version moved —
   that is its own decision with its own justification: say it out loud, show that model's gate
   passing afterwards, and never fold it silently into a change about something else.

**The check is the whole suite, not the new test.** `ctest` is the only thing that knows every family
still works; a green run of the gate just written proves nothing about the six that were not touched.
Run it before claiming a model is supported, and read the skips — a family whose gate reports
`skipped` because its fixture stopped being found has already lost its coverage, which is one edit
away from losing its support.

**Admission.** Models are converted to a Soma container before serving, from `tools/admission/`
(Python; `convert`/`verify_payload` run in `tools/admission/.venv`, `make_oracle` in the separate
`.venv-oracle`, which carries transformers):

```bash
python tools/admission/convert.py HF_DIR --out CONTAINER --quant q4_g --expert-down q6_g --group 128
python tools/admission/verify_payload.py CONTAINER --source HF_DIR
python tools/admission/make_oracle.py HF_DIR --out FIXTURE_DIR   # reference logits for conformance
```

**Chat templates are MEASURED, not interpreted.** `tools/admission/chat_template.py` runs the
checkpoint's real Jinja template (from `chat_template.jinja`, or the `chat_template` key in
`tokenizer_config.json`) against probe conversations built from sentinels, reads the scaffolding off
the text around each sentinel, resolves it to TOKEN IDS, and appends it to `tokenizer.soma` (format
2; version 1 is still accepted and means "no template"). The engine only concatenates — there is no
Jinja interpreter in C++ and there must not be one. Two gates: `verify()` requires the piecewise
assembly to equal the whole rendered string BOTH as text and as ids (the id half proves no BPE merge
reaches across a seam, which is a property of the tokenizer, not the template), and
`soma_chat_template_g0` plus the `chat_template` conformance stage grade the engine against
`chat_oracle.bin`. **Recognize or refuse**: a template this shape cannot express compiles to nothing,
with the reason in `chat_template.unsupported`, and `soma serve` falls back to flattening messages.
Options the template does not have (`enable_thinking`, `clear_thinking`, `reasoning_effort`) are
REFUSED with a 422 rather than ignored; an unrecognized `reasoning_effort` VALUE is not, because GLM
renders `medium` as its default rather than erroring. `tools` and assistant `tool_calls` are refused —
the scaffold covers conversation framing, not tool-call encoding. **Adding a conformance stage needs a
control-side migration**: `conformance.stage` carries a `CHECK (stage IN ...)` allow-list, the ladder
is written in one transaction, and a rejected name rolls back every row — presenting as "conformance
never ran".

**Source formats.** `convert.py` reads f32, bf16, and **blockwise fp8** — `F8_E4M3` weights beside a
`<tensor>_scale_inv` of one f32 multiplier per `weight_block_size` tile, as `quantization_config`
`quant_method: fp8` declares. Everything else pre-quantized is still refused: compressed-tensors, AWQ
and GPTQ pack sub-byte levels in layouts of their own, and quantizing their packed bytes yields a
container that loads and generates noise. fp8 is an exception on the merits — it dequantizes EXACTLY,
one multiply per tile — and it exists because GLM-5.3's primary upload is fp8 (756 GB) while its bf16
twin is a separate 1.5 TB repo. **Which tensors get dequantized is decided by DTYPE, not by
`modules_to_not_convert`**: an fp8 upload publishes its norms, routers, embeddings and head
unquantized and the tensor headers already say so. An fp8 weight with no scale beside it REFUSES
rather than converting a matrix ~400x too small. `container_meta.json` records the source codec in
`source_quantization`, because nothing else in the file does; `tools/ci/check_fp8_source.py` (ctest
`mm_fp8_source`) holds it, by quantizing a fixture blockwise and requiring the fp8 and f32 sources to
convert byte-identically.

A multimodal wrapper is REFUSED by default — the language model nests under `text_config` and every
key the converter reads is absent at the top level, so left alone it reports "no routed experts"
about a model with hundreds. **Only a wrapper is refused**: the gate is `text_config` being present,
so an ordinary text-only checkpoint never consults `SOURCE_DIALECTS` at all and converts with
identity naming. The exception is a checkpoint listed in `SOURCE_DIALECTS`, which names the tower's
tensors and the spellings that differ from the ones the loader binds. Two are listed, and they
differ instructively:

- **MiniMax-M3** renames a great deal — `block_sparse_moe` for `mlp`, a selection bias off the block
  onto the gate, four `index_*` tensors into an `indexer.*` block.
- **Kimi-K3** renames *nothing*. `KimiK3ForConditionalGeneration` builds exactly `vision_tower`,
  `mm_projector`, and a `language_model` that is a `KimiLinearForCausalLM` — the same module
  `tests/fixtures/tiny/Kimi-Linear-Tiny` carries. So its whole dialect is a `language_model.` prefix
  and a drop list, with an EMPTY suffix map. `moe_block` is deliberately absent from it: that key
  does not describe a checkpoint, it *means* "rewrite this block to `mlp`", which is right for
  MiniMax and renames every Kimi expert to a name the loader does not bind.

Their text stacks convert while their vision towers do not. The container keeps `config.json`
verbatim either way, so the plan still reports `vision+text` and says which half it serves —
`ModalitySpec` recording that while nothing reported it is exactly how a model ends up answering
about an image it never received. `tools/ci/check_{minimax,kimi}_dialect.py` keep the maps honest by
rewriting the committed fixture into each production dialect and asserting both containers describe
the same weights.

**A dialect is necessary, not sufficient.** Adding one only decides whether the wrapper is unwrapped;
the tensor ROLES still have to be in `DENSE_SUFFIXES` (converter side), and the block names in
`TensorNaming` (engine side, `include/soma/arch_ir.hpp`). The
Kimi family needed twenty roles the converter lacked — KDA under `self_attn` (a different module and
different spellings from Qwen3.5's `linear_attn` GDN), the latent-MoE `routed_expert_*` projections,
and the block-residual gates — so no Kimi container could be produced by any route, wrapped or not,
while `arch_ir.cpp`, `kda.cpp` and `f32_model.cpp` all bound those tensors happily (roadmap D69).

`convert.py` compiles the tokenizer into the container and records the quantization in
`container_meta.json` — that file, not a hardcoded guess, is what `serve` reads back.

Specs live in `schemas/arch-ir.md` and `schemas/container.md`; current state, deferred work, and
the open-defect table are in `docs/roadmap.md`.

### Configuration

The node and control executables read a `.toml` file in the working directory; environment variables override any key. Copy template configs from `tools/` before first run:

```sh
cp tools/mantic-mind-control.toml .
cp tools/mantic-mind.toml .
```

Key env vars: `MM_CONTROL_URL`, `MM_SELF_URL`, `MM_LLAMA_PATH`, `MM_SOMA_PATH`, `MM_CONTROL_PORT`

Engine settings are NO LONGER node-local: `llama_*` keys in `mantic-mind.toml` are deployment
paths (where the binary lands), while which engine to run, at what version, and by what method
come from the master. See **Master engine configuration** above.

`soma` itself takes no `.toml` — it is configured entirely by flag or `SOMA_*` env var, because the
node passes its configuration down when it spawns the subprocess.

### Compiler Flags

MSVC: `/W4 /WX /utf-8` (warnings as errors). GCC/Clang: `-Wall -Wextra -Wpedantic -Werror`.

---

## Mafia-Machine (Python)

### Run

From `Mafia-Machine/`:

```bash
python3 run.py \
  --control-url http://127.0.0.1:9090 \
  --agent-ids agent-1,agent-2,agent-3,agent-4,agent-5,agent-6,agent-7 \
  --players 7 \
  --mafia 2 \
  --seed 42
```

Requires a running `mantic-mind-control` server with pre-created agent IDs (via `POST /v1/agents`).

### Module Layout

```
mafia_machine/
  models.py         — dataclasses: Player, GameConfig, ToolInvocation, ChatResult
  mantic_client.py  — HTTP client wrapping Mantic-Mind's REST/SSE API
  engine.py         — game loop: role assignment, night/day cycles, win detection
  cli.py            — argparse entry point
run.py              — thin launcher
```

### Key Behaviour

- Agents communicate via structured JSON tool calls in chat output
- Invalid model output triggers a deterministic fallback action (game never stalls)
- Agent IDs are validated at startup (count, uniqueness, existence in `/v1/agents`)
- Roles: `mafia`, `town`, `detective`, `doctor`

---

## REST API Quick Reference

**Admission (control, `:9090`)** — converting a model to a Soma container, hours long
```
POST  /v1/models/admit                     → SSE; the id is in the first frame
GET   /v1/models/admissions                every operation, running or finished
GET   /v1/models/admissions/{op}           rejoin one after disconnecting
POST  /v1/models/admissions/{op}/cancel
```
Reachable from the control TUI's **Admissions** tab (key `0`) and `models admit|admissions|cancel`
in `--mode cli`. Disconnecting does not cancel — the worker is detached so it outlives the request,
which is what makes rejoin work. Concurrency is capped (`admission_max_concurrent`, default 1) and
a duplicate admission of the same model JOINS the running one rather than starting a second into
the same container directory.

**Placement lifecycle (control, `:9090`)** — the P1 gap that had no route at all
```
GET   /v1/placements                       where every agent actually is
POST  /v1/agents/{id}/suspend              409 if it holds no placement
POST  /v1/agents/{id}/restore              = ensure_agent_running; restores from checkpoint
POST  /v1/agents/{id}/release              idempotent
```
`tools/ci/check_surface_parity.py` (ctest `mm_surface_parity`) asserts every CLI `/v1` call hits a
registered route and every in-process mutation in the control TUI has a `/v1` equivalent.

**External Client (control, `:9090`)**
```
GET/POST       /v1/agents
GET/PUT/DELETE /v1/agents/{id}
POST           /v1/agents/{id}/chat          → SSE
GET            /v1/agents/{id}/conversations
GET            /v1/agents/{id}/memories
GET            /v1/nodes
```

**Node API (`:7070`, requires `Authorization: Bearer <node-api-key>`)**
```
POST  /api/node/load-model
POST  /api/node/infer        → SSE
GET   /api/node/health
GET   /api/node/status
GET   /api/node/models/local

                             engines — these REPLACE /api/node/runtime/llama/*,
                             which named one engine in the URL:
GET   /api/node/engines
POST  /api/node/engine-config              master pushes the cluster policy
POST  /api/node/engines/:id/provision      :id must be llama-cpp for the
POST  /api/node/engines/:id/check-update   llama-shaped wizard actions;
POST  /api/node/engines/:id/switch         others get a 400 saying why
POST  /api/node/engines/:id/diagnose
POST  /api/node/engines/:id/recover
POST  /api/node/engines/prepare            package + hash, send nothing
POST  /api/node/engines/expect             control relays the expected digest
POST  /api/node/engines/share              push the prepared package to a peer
POST  /api/node/engines/receive            accept, verify digest, install
```

**Cluster engine configuration (control, `:9090`)**
```
GET   /v1/cluster/engines/config           200 + configured:false before setup
PUT   /v1/cluster/engines/config           operator; version assigned server-side
GET   /v1/cluster/engines/conformance      per node, incl. placement_eligible
POST  /v1/cluster/engines/resync           re-push to every stale node
POST  /v1/cluster/engines/share            broker one artifact between nodes

                                           per-node engine actions, brokered to
                                           /api/node/engines/:id/* — 202, the
                                           node works in the background:
POST  /v1/cluster/engines/nodes/:id/provision
POST  /v1/cluster/engines/nodes/:id/check-update
POST  /v1/cluster/engines/nodes/:id/switch        body: variant
POST  /v1/cluster/engines/nodes/:id/diagnose
POST  /v1/cluster/engines/nodes/:id/recover       body: action, variant
```

**Resync is not a repair lever, and these are.** `resync` re-pushes only to nodes whose config
*version* differs, and a node bumps its version on **accepting** a configuration, not on
**conforming** to one. A node that accepted v3 and then failed its build sits at v3 with state
`failed`: resync skips it and answers `{"resynced": true}`. Before these routes, the only cure was
the wizard on that node's own TUI, which was wired in-process and reached five node routes no
client ever called — control could configure a cluster it could not repair. In the control TUI they
are the Engines tab's second button row (they act on the **selected** node) and
`engines provision|check-update|switch|diagnose|recover <node_id>` in `--mode cli`.

All six node-side actions — including the scheduled auto-update — now serialise on one worker with
one busy flag, so a second request gets a `409` instead of racing the first through the provisioner.
`tools/ci/check_surface_parity.py` gained a node half that asserts every node-TUI mutation has an
`/api/node` route, which is the check that would have caught this gap.

**Soma engine (`:8080` by default, spawned and supervised by the node)**
```
GET   /health                      readiness; what the node's health poll hits
GET   /v1/models
POST  /v1/chat/completions         JSON and SSE; when the container carries a
                                   compiled chat template it also takes
                                   `reasoning_effort`, `enable_thinking`,
                                   `clear_thinking` and `add_generation_prompt`,
                                   and 422s any of them the template has no
                                   switch for rather than ignoring it

                                   the rest are node-only:
GET   /internal/plan               the plan document for the loaded model
GET   /internal/sessions
POST  /internal/kv/save
POST  /internal/kv/restore
GET   /internal/telemetry          → SSE
GET   /internal/telemetry/dump
GET   /internal/heat               expert-access heat map
```
