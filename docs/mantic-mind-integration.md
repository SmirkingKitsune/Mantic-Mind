# Soma ↔ Mantic-Mind integration

Two boundaries, kept distinct:

1. **Engine ↔ node** — internal, hot path, OpenAI-compatible HTTP over a supervised subprocess.
2. **Control's External Client API** — the single control plane. Covered in
   [external-api.md](external-api.md).

Conflating them is how a telemetry feed ends up in the chat stream.

---

## 1. Engine ↔ node

### What is kept, because it was right

The node supervises engines as **subprocesses** and talks to them over an **OpenAI-compatible HTTP
boundary**. Soma sits behind that same boundary. Extending the supervision machinery beats building a
parallel universe, and the existing machinery is closer to generic than it looks:

- `start_with_args(runtime_name, exe, argv, port, timeout)`
  ([runtime_process.hpp:54](../include/node/runtime_process.hpp:54)) is already engine-neutral.
  `start_llama_server` is a ~15-line adapter over it.
- Readiness is already an HTTP health poll with early abort on child exit
  ([runtime_process.cpp:676](../src/node/runtime_process.cpp:676)) — **there is no log sentinel anywhere
  in this codebase**, so the Windows fragility the design brief warns about is not being carried
  forward. `ReadinessProbe::StdoutJsonLine` is declared because it costs nothing, not because
  `HttpHealth` is suspect.
- Multi-agent engine sharing, RAII request leases, port allocation with a bind probe, and
  suspended-engines-don't-consume-capacity all survive intact.

### Soma's `serve` mode

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness. What the node's existing poll hits, unchanged. |
| `GET /v1/models` | Served model name |
| `POST /v1/chat/completions` | JSON + SSE. `enable_thinking`, `reasoning_effort`, `determinism`. |
| `GET /internal/plan` | The plan document for the loaded model |
| `GET /internal/telemetry` | SSE, terse frames — node-only |

Configuration is available through **both** CLI flags and env vars for every field, because the node
spawns this as a subprocess and argv quoting across Windows and POSIX is a worse place to discover a
mistake than an environment block:

```
--model-dir / SOMA_MODEL_DIR      --ram-budget / SOMA_RAM_BUDGET
--kv-dir    / SOMA_KV_DIR         --pin        / SOMA_PIN
--port      / SOMA_PORT           --ctx-size   / SOMA_CTX_SIZE
--kv-slots  / SOMA_KV_SLOTS       --max-batch  / SOMA_MAX_BATCH   (0 = the gate decides)
--determinism / SOMA_DETERMINISM  --telemetry-hz / SOMA_TELEMETRY_HZ
```

`EngineProcess` gains an **env block** — `CreateProcessA` and `execvp` both inherit the parent's
verbatim today (`nullptr` env), so a subprocess could only ever be configured through argv.

### `inference_backend = "soma"`

`model_path` is a **converted model directory** (the admission output), not a single GGUF. The fallback
keeps its single-file path. Both flow through the same `EngineLoadRequest`.

### The descriptor registry

`EngineDescriptor` is the abstraction that makes llama.cpp and Soma **two descriptors rather than two
code paths**. It supplies `build_launch`, `readiness`, `make_client`, `kv`, `verify_capabilities`,
`estimate_footprint`, and `launch_compatible`.

What that deletes:

| Before | After |
|---|---|
| `backend != "llama-cpp" → 400`, hardcoded at [node_api_server.cpp:449](../src/node/node_api_server.cpp:449) **and** `:911` | One registry lookup. The 400 body lists `EngineRegistry::ids()` — accurate by construction rather than by a hand-maintained `supported_backends: ["llama-cpp"]` literal in two places |
| `LlamaRuntimeStatus` single scalar on `NodeState`, 7 `Llama*Callback` types, 6 `/api/node/runtime/llama/*` routes | `RuntimeStatus` map keyed by engine id; `/api/node/runtime/{id}/*`; one callback set parameterized by id |
| `SlotInfo::backend` hardcoded `"llama-cpp"` in `make_slot_info` | From the descriptor |
| `Slot` holding `unique_ptr<RuntimeProcess>` + manager-wide `llama_server_path_` | `Engine` holding an `EngineDescriptor*` |

### Crash supervision — new

Today nothing polls the child once `state_ == Ready`. A dead engine advertises `SlotState::Ready` until
an inference request happens to fail at the HTTP layer. `EngineProcess::set_crash_callback` plus an
`EngineSupervisor` watchdog closes that.

This is a pre-existing gap, not one Soma introduces — but streaming makes engine crashes likelier (I/O
pressure, OOM under a mis-sized cache cap), so it gets fixed as part of the rebuild rather than left.

### KV persistence

`KvCheckpointBackend` (`LlamaKvBackend`, `SomaKvBackend`) replaces two hardcoded calls to
`POST /slots/0?action=save|restore`.

**Both hardcode sequence 0.** A slot launched with `--parallel > 1` therefore only ever checkpoints its
first sequence — a latent data-loss bug in the current system. `LlamaKvBackend::supports_multi_sequence()`
returns `false`, and the supervisor turns that into an **explicit refusal** instead of a silent partial
save. `SomaKvBackend` returns `true`.

One format serves warm reopen, scheduler preemption, and cluster slot suspend/restore
([architecture.md §6](architecture.md)).

---

## 2. Verdict-driven routing in `AgentScheduler`

> Written when this was going to be a new class called `PlacementEngine`. It never was: the header
> was declared, implemented by nothing, and deleted (roadmap D46). **Most** of what follows shipped
> into `AgentScheduler` itself — see [architecture.md §8.0](architecture.md) for the mapping — with
> two exceptions flagged inline below, because a note saying "everything below shipped" over a
> section where two things did not would be the same kind of claim this deletion exists to remove.

### The algorithm is preserved

`AgentScheduler`'s placement sequence was arrived at by fixing real problems and survives intact:

```
existing placement → suspended restore → preferred node → shared engine
                   → capacity fit → evict LRU-idle + retry
```

So does the **two-mutex split**, which is load-bearing: `schedule_mutex_` serializes whole scheduling
operations including multi-GB model transfers, while `state_mutex_` guards only the placement map — so
`GET /v1/placements` and `GET /v1/agents` never block behind an in-flight transfer.

### What changes around it

**Backend selection happens first**, from the registry verdict, and is recorded with a reason. What
it replaced was the opposite — an unconditional refusal of anything that was not llama.cpp:

```cpp
// then
if (!is_llama_backend(cfg.inference_backend)) { release_agent(cfg.id); return nullopt; }
```

```cpp
// now — the check survives, meaning something else entirely: node-local vs API-backed.
// Soma passes through it and `soma::select_backend` picks.
AgentScheduler::BackendRouting AgentScheduler::resolve_backend(
    const AgentConfig& cfg, const soma::AdmissionRecord& record);   // {engine_id, reason}
```

`resolve_backend(cfg, record)` is **pure** given `(config, record)` and is `static` — it deserves to
be tested without a node in the loop — with `resolve_backend_for(cfg)` doing the registry lookup for
callers that have one. The reason string travels with the decision, so a client can see it without
causing a placement.

**`engine_fingerprint` gains `engine_id`** — the same weights served by two different engines are not
the same engine, so a backend change forces a reload. Live: `engine_fingerprint(cfg, models_dir, engine_id)`.

> **`arch_hash` is not a named component**, and the reason it may not need to be is worth recording
> rather than asserting either way. The `model` component is `file_manifest_identity()`, which for a
> container directory walks it recursively and records every file's relative path, size and mtime.
> A requantization written into the SAME directory therefore changes the fingerprint already, without
> anyone naming `arch_hash`. What is NOT traced here is the cross-directory case — a re-admission at a
> different quantization lands in its own container (`<name>-q4_g-q6_g-g128`), and whether the agent's
> `model_path` then resolves to the new one is a question about ref resolution, not about this
> fingerprint. Stated as an open question because that is what it is; no defect is claimed.

### The placement estimate is re-shaped, not re-tuned

| | Before | After |
|---|---|---|
| Type | `int64_t vram_needed` | `ResourceFootprint{vram_mb, ram_mb, disk_mb}` |
| Source | `estimate_inference_vram_mb()` | Soma: **read from `plan --json`**. Fallback: recursive directory sizing. **Not shipped control-side** — every footprint control builds still sets `vram_mb` alone (D62); the three-axis figure is computed node-side, after placement has chosen. |
| Node query | `nodes_with_available_vram(int64_t)` | `nodes_with_capacity(ResourceFootprint)` |
| Disk | Never consulted | Hard constraint |

Two concrete defects this fixes:

1. **`estimate_inference_vram_mb` returns a flat 2048 MB for any directory.** `fs::file_size()` sets an
   `error_code` on a directory and the code falls back to `kFallbackModelMb`
   ([inference_sizing.cpp:87](../src/common/inference_sizing.cpp:87)). Every converted Soma directory —
   and every multi-shard HF model on the *fallback* path today — sizes identically.
2. **`NodeInfo::disk_free_mb` is collected by the health poll and never used for placement.** Soma's
   footprint is RAM + disk + optional VRAM; disk has no offload equivalent to trade against, which is
   precisely why a scalar could not express it.

Soma's footprint is **measured, not estimated** — `plan --json` reads headers only and allocates
nothing, so it is safe to compute for a node that could not host the model. `footprint_source` on
`/v1/placements` reports which of the two a placement used.

### Capacity pressure becomes a code

`response_indicates_capacity_pressure` ([agent_scheduler.cpp:904](../src/control/agent_scheduler.cpp:904))
substring-matches six English phrases against the node's error body. A new engine would have had to
reproduce those literals verbatim to earn an evict-and-retry. Both engines emit
`{"error":{"code":"capacity_pressure"}}` instead.

### Shared engines are multi-sequence concurrency

Multiple agents on one Soma process via `kv_slots` / `max_batch`, with per-slot state on
`GET /v1/engines/{id}/slots`. This is the concurrency design viewed from the cluster side, not a
separate feature.

**`AgentQueue` stays, and the layering is deliberate.** It looks like the FIFO-in-front-of-a-single-
sequence-engine that the step-major scheduler exists to replace. It is not:

- `AgentQueue` serializes **one agent's own turns**. An agent is a conversation with mutable state; two
  simultaneous turns on one agent is a semantic error regardless of engine capability.
- Soma's `max_batch` is **intra-engine concurrency across different agents**.

They compose exactly: N agents with in-flight turns produce N rows in one union forward. What is being
removed is the assumption that the engine below can only do one sequence — not the per-agent ordering
guarantee above.

---

## 3. FTXUI panels consume the API

The control TUI is **one client of `/v1/*`**, not a privileged inspector. No panel reaches into
in-process engine state.

This is a real change of habit. `ControlUI` today is a hybrid: writes and streaming go over loopback
HTTP (`cli.stream_post`, `/v1/performance`, `/compact`), but most reads reach directly into
`registry_.list_nodes()`, `agents_.list_agents()`, and `AgentDB` — the latter *on every frame*, with a
source comment already noting the cost. New Soma panels do not extend that pattern.

| Panel | Route | Notes |
|---|---|---|
| **Tier bar** | `GET /v1/engines/{id}/telemetry` | Stacked block characters: VRAM / RAM / disk occupancy. VRAM renders and reads zero in v1 — declared, honest, and ready. |
| **Brain cortex** | same, `include=heat` | One cell per expert (bucketed). Colour = tier, brightness = decayed heat. A coloured block-character grid; FTXUI carries it natively. |
| **Cache health** | same | Hit rate, evictions, `prefetch_wasted`, io-wait. |
| **Concurrency** | `GET /v1/engines/{id}/slots` | Batch size, `effective_max_batch`, `max_batch_limited_by`, per-sequence rows. |
| **Registry** | `GET /v1/models` | Verdict per model, with reason. |
| **Admission** | `POST /v1/models/admit` (SSE) | Progress stages; cancel. Reuses the existing `NodeActionProgress` shape. |
| **Placement** | `GET /v1/placements` | Engine + `backend_reason` per agent. |

**No second UI framework.** FTXUI is this project's only UI. Richer external visualization — anything
resembling an admission-pipeline inspector — belongs to the separate SvelteKit stack consuming these
same routes. That is exactly why P1 is non-negotiable: if a capability is not on the API, that stack
cannot see it.

### The brain panel's refresh model

The panel subscribes at the default **2 Hz**, bucketed. It does **not** request `resolution=full`, and it
does not raise `hz` — a TUI redrawing a 4096-cell grid faster than the eye resolves is spending CPU to
look identical.

Per-token flashes are deliberately **not** in the design. Heat is aggregated inside the engine and
sampled at the tick rate; a flash-on-routing effect would require per-token events, which is the exact
bandwidth the throttle exists to prevent. The grid animates through *decayed heat*, which conveys the
same thing at 2 Hz and costs nothing.

---

## 4. Configuration

`ConfigFile` parses `[section]` headers and **discards them** — "all keys are flat"
([config_file.hpp:17](../include/common/config_file.hpp:17)). With two engines that forces a `soma_*`
prefix explosion on top of the existing 13 `llama_*` fields.

The fix is ~20 lines: namespace section keys as `section.key`. Files without a section header keep
working byte-identically, so every existing config and env override is unaffected.

```toml
# mantic-mind.toml
max_slots = 4                    # unchanged, flat

[engine.llama-cpp]
server_path = "llama-server"
auto_provision = true

[engine.soma]
enabled = true
ram_budget_gb = 24
pin_gb = 4
kv_slots = 4
max_batch = 0                    # 0 = the cache-aware gate decides
telemetry_hz = 2
```

Env overrides follow the existing convention: `MM_ENGINE_SOMA_RAM_BUDGET_GB`.

`max_batch = 0` is the recommended default and is not laziness — a fixed constant would be a bug wearing
a config key's clothing. See [architecture.md §4.2](architecture.md).

---

## 5. Node API changes

| Route | Change |
|---|---|
| `POST /api/node/load-model` | `backend` is a registry lookup; unknown → 400 listing real ids. `runtime_settings` gains the Soma block. |
| `POST /api/node/restore-slot` | Same lookup (the duplicated gate at `:911` disappears). Rejects a cross-`arch_hash` checkpoint **before** spawning. |
| `POST /api/node/suspend-slot` | 409 `multi_sequence_unsupported` instead of a silent partial save. |
| `GET /api/node/status` | `llama_runtime` → `runtimes[]`, keyed by engine id. |
| `/api/node/runtime/llama/*` | → `/api/node/runtime/{id}/*` |
| `GET /api/node/engines/{slot}/telemetry` | **New.** Node-side SSE forwarded from Soma; control re-publishes it. |
| `POST /api/node/infer` | Chunk→event mapping becomes independent emission rather than an `if/else-if` priority chain, so a chunk with both `thinking_delta` and `delta_content` stops dropping one. `tool_result_json` is emitted. |

---

## See also

- [architecture.md](architecture.md) — the seam, the three tiers, before/after in full
- [external-api.md](external-api.md) — the `/v1/*` surface and scopes
- [roadmap.md](roadmap.md) — which gate lands which piece
