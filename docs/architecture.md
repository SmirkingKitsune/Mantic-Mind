# Soma — Architecture

> **Status:** active experimental implementation; GLM-5.2 serves end to end through the production path.
> **Scope:** experimental branch, destructive refactor. No cross-compatibility with `main`.

Soma is the MoE-primary inference engine in Mantic-Mind. It runs oversized Mixture-of-Experts models on
modest consumer hardware by treating VRAM, RAM, and disk as one managed memory hierarchy and streaming
routed experts from NVMe on demand. The dense part — attention, shared experts, embeddings — stays
resident; routed experts live on disk and page through an LRU cache with a pinned hot set.

llama.cpp remains in the system as the **fallback**: the engine for models Soma can't or shouldn't run.
Which engine gets a model is not a configuration choice. It is a **verdict**, computed offline at
admission and stored in the registry.

---

## 1. Two principles

**P1 — The API is the single control plane.** Every capability Soma introduces is reachable as a
`/v1/*` route on control's External Client API: admission, registry and verdict queries, placement
plans, backend-selection policy, tier/heat/brain telemetry, concurrency and slot state, determinism
controls. The FTXUI TUI is *one client*. If the TUI can do it, an API client can do it through the same
route. See [external-api.md](external-api.md).

**P2 — Structure over compatibility.** There is no back-compat constraint, so subsystems are rebuilt
where the clean design demands it rather than adapted. §7 and §8 give the before/after for every one.

---

## 2. The seam

The seam is the interface between the architecture-invariant core and per-architecture backends. It is
a **hard structural requirement**, not a style preference.

```
include/soma/*.hpp        invariant core   — never learns about architectures
include/soma/arch/*.hpp   backends         — everything model-specific
```

**Dependency rule, mechanically enforced:** `arch/` headers may include core headers. Core headers may
**not** include `arch/`, and may not mention an architecture-specific identifier. CI greps for this
(§11) — the seam is checked, not merely intended.

| Invariant core | Per-architecture backend |
|---|---|
| `MemoryHierarchy` — tiers, LRU, pin, heat | `F32Backend` — execution descriptor and per-layer operations |
| `ExpertStore` — sidecar index, aligned reads, readahead pool | `AttentionBackend` — planner sizing and KV persistence format |
| `Scheduler` — step-major loop, ragged batching, admission control | Router semantics — score fn, normalization, bias correction, grouping |
| `Kernels` — quant GEMM family, static dispatch from `kernel_choice` | Activation, norm placement, RoPE variant |
| `KvCheckpointStore` — versioned, `arch_hash`-gated | Expert layout descriptor, draft/MTP head |
| `Serve` — OpenAI-compatible HTTP/SSE | — |
| `Telemetry` — tier occupancy, routing heat | — |

### 2.1 What is deliberately *not* in the seam

**The tokenizer.** Admission compiles `tokenizer.json` into a normalized on-disk format: merge table,
pretokenizer as a compiled byte-class NFA (not a live regex engine), special/added tokens, byte-fallback
flag, chat template as a token-level struct. A Unigram/SentencePiece decoder path is a second compiled
form. The runtime loads it as **data**. Two architectures with the same tokenizer share zero code paths
in `arch/` and that is correct.

**Quantization.** Specified per *tensor role*, not per architecture and not per tensor. The role map
lives in the architecture IR; the kernels are core.

### 2.2 The attention families this seam is designed against

Shaping an interface to one architecture and discovering it at the second is the classic failure. The
seam was therefore designed against **GQA** (Qwen3-MoE, Mixtral, GPT-OSS, Llama-family) and **MLA**
(DeepSeek-V2-Lite, Moonlight) simultaneously. Both now run through it.

A **third** family has since gone through it, and it is the one to read this section against:
**MLA + DSA** (GLM-5.2) — DeepSeek Sparse Attention with IndexShare. It is MLA plus a learned sparse
key selector, and 57 of its 78 layers own no indexer weights at all; they reuse a selection computed
by a different layer. The seam carried it without changing shape: one opaque `ArchLayerPayload` on
the per-forward workspace. The concern recorded here earlier — that per-layer function pointers had
no channel for cross-layer state — was the wrong shape of problem. The index is per-(row, step) and
never persists between steps, so it needed a per-FORWARD slot, not a per-sequence one.

Their genuine differences are **KV cache shape** and **decode algebra**:

| | GQA | MLA |
|---|---|---|
| Cache contents | Full K and V, `n_kv_heads × head_dim` each | Compressed latent (`kv_lora_rank`) + a small RoPE-carrying slice |
| Bytes/token | Large — competes directly with the expert cache for RAM | Small — roughly two orders of magnitude less on comparable configs |
| Decode-time algebra | Standard projection | **Weight absorption** — move up-projections to the query side so cached decode never materializes full K/V. No GQA analogue. |
| Decode | `repeat_kv` then standard SDPA | Operates in latent space |

The interface consequences, and why each exists:

- **The core owns cache storage, not cache algebra.** `KvCache` allocates two planes from the
  backend's `KvGeometry` and hands each row to `F32Backend::attention_kv`; the core never interprets
  either plane. GQA uses them as full K and V, MLA uses only the latent K plane, and DSA uses the
  second plane for indexer keys.
- **Absorption happens per cached-decode step inside MLA.** Folding at load would keep a transposed
  fp32 copy of the up-projection resident (1.96 GB on GLM-5.2) to save arithmetic that was never the
  bottleneck. `MlaSpec::absorb_weights` selects the absorbed form or the expanded reference form;
  there is no load hook and no second execution path (roadmap D38, D39).
- **`kv_geometry()` reports BOTH cache planes**, because they are not the same size. A single width
  cannot express "this family stores no second plane", and for want of that MLA allocated a full V
  plane at the K plane's width for every layer, holding nothing — 2.94 GB on GLM-5.2 at 4k x 4 slots.
  MLA derives V from the latent, so its V plane is zero; DSA's holds the indexer key, which depends
  on a past token's hidden state at that layer and cannot be recomputed later (roadmap D35, D37).
- **`F32Backend::attention_kv` takes the whole ragged batch.** Each row carries its own cache slot,
  position and visible length. A future paged-KV implementation would extend this actual row/cache
  interface with a block table.
- **`persist_format_id`** tags KV checkpoints so an MLA checkpoint can never be replayed into a GQA
  engine. A cheap guard against a genuinely confusing class of bug report, and it has since earned
  its keep WITHIN a family: MLA's tag went to `.v2` when the V plane changed shape, so a checkpoint
  written under the old layout is refused rather than replayed into a differently-shaped cache.

**Mixtral is the third design input**, and it is a negative one. Eight experts at top-2, ~88 MB each,
activates a quarter of every layer — streaming buys almost nothing. The seam must represent it
faithfully so that admission can look at it and say `resident-only`. Soma is not required to be good at
everything; it is required to know what it is bad at.

### 2.3 Devirtualization

Backends are structs of function pointers resolved once at load. Hot calls devirtualize behind
`-DSOMA_ARCH=<name>`, which resolves the descriptor at compile time. **The generic pointer path stays
compiled and is what the conformance harness exercises**, so both paths are always live and a
divergence between them is a test failure rather than a production surprise.

---

## 3. Three state-ownership tiers

Concurrency correctness reduces to knowing who owns what. Three tiers, and the boundaries are the point:

| Tier | Contents | Lifetime | Locking |
|---|---|---|---|
| **`model`** | Dense tensors, compiled tokenizer, architecture IR, kernel dispatch table, expert index | Immutable after load | **None.** Read-only, shared freely across threads. |
| **`seq`** | KV slot, position, sampler + RNG, draft window, grammar state, stop condition | Per-sequence | Owned by one sequence; no cross-sequence access. |
| **`exec`** | Per-step scratch — ragged batch buffers, router logits, expert-union workspace | Per-step, sized for `max_batch` | Mutex-held for the step. |

**Only `exec` and the expert cache need locking.** That is the whole concurrency story, and it is why
the model tier being genuinely immutable matters more than it looks: the moment something mutates
`model` at runtime — a lazily-materialized tensor, a cached transpose — the lock-free read collapses.
Admission is therefore responsible for doing every transformation ahead of time, including transposing
fused 3D expert tensors. **Never at runtime.**

---

## 4. The step-major scheduler

Not `for each request { for each token }`. Instead:

```
for each step {
    collect ready sequences        →  ragged batch of (seq*, token) pairs
    one batched forward            →  dense GEMMs + union MoE
    scatter outputs                →  per-sequence sampling, stop checks
}
```

Dense projections, MLP, router, and embeddings batch trivially into one GEMM over concatenated rows.
Decode rows and prefill rows are **just rows** — chunked prefill (512 tokens/chunk default) interleaves
with decode in the same forward, with a per-step fairness cap so one long prompt cannot starve
interactive turns.

### 4.1 The MoE union is the payoff

Union the expert IDs across the **whole batch**. Read each unique expert once. Apply it to every row
that selected it, regardless of which sequence the row came from.

The prior art does this only within one prompt's prefill. Generalizing it across concurrent sequences is
the central claim of this design: **read cost is per-expert and independent of how many rows consume
it**, so aggregate throughput in the disk-bound regime scales better than linearly in concurrency. Two
sequences that happen to route to overlapping experts cost barely more than one.

### 4.2 Cache-aware admission control

This is the constraint most likely to be got wrong, and getting it wrong inverts the benefit above.

N sequences × top-k experts against a small per-layer LRU cap **thrashes**. Every step evicts what the
next step needs; the union degenerates into per-row reads plus eviction overhead, and throughput falls
below single-sequence. So:

```
max_batch  ≤  cap_per_layer / expected_unique_experts_per_step
```

`expected_unique_experts_per_step` comes from the planner (measured at admission, not assumed), and
`cap_per_layer` from the memory budget. **The gate drops only when the expert set is fully resident** —
at which point there is no read to amortize and concurrency is bounded by compute instead.

A fixed `max_batch` constant would be a bug wearing a config key's clothing.

### 4.3 Preemption

Under memory pressure, a sequence is evicted by persisting its KV checkpoint and re-admitting later.
This is nearly free once KV persistence exists, and it is **the same mechanism** as warm conversation
reopen and as cluster-level slot suspend/restore. One format, three callers — see §6.

### 4.4 Speculation

Disabled when `batch > 1` in v1: batching already amortizes the reads speculation was buying, and the
interaction between draft acceptance and batch composition is a second-order problem not worth paying
for at G3. Grammar-forced drafts compose more easily and stay on.

---

## 5. Memory hierarchy and expert streaming

### 5.1 The static partition

Everything else is bookkeeping around this split:

- **Dense / resident** — attention projections, shared experts, embeddings, norms, router weights.
  Loaded once, never evicted.
- **Routed / streamed** — routed expert weights. Live on disk; page through the cache.

### 5.2 Tiers

```
Vram  ─ hot pinned set                        (v1: declared, always empty)
Ram   ─ per-layer LRU + pinned hot store      (v1: the working tier)
Disk  ─ expert store, with OS page cache as a free L2
```

**v1 is CPU-only.** `MemoryTier::Vram` exists in the enum, is reported on the API and rendered in the
brain grid, and is always zero-occupancy until after G4. `plan --json` emits `vram_hot_gb: 0`. This
keeps the tier a real concept in every schema, format, and route from day one, so adding GPU residency
later is an implementation, not a migration.

### 5.3 Expert store layout

Streaming imposes hard requirements on the on-disk container, all satisfied at admission:

- **One expert = one contiguous byte range.** Gate/up/down interleaved so a single read fetches the
  whole SwiGLU triple.
- **4 KB-aligned offsets** for `O_DIRECT`.
- **Sidecar index** `expert_id → (shard, offset, len)`, so a cache miss never parses a safetensors
  header.
- **Pre-transposed** fused 3D expert tensors.

### 5.4 Acquisition is RAII

`MemoryHierarchy::acquire(layer, expert)` returns an `ExpertRef` that pins the expert for its lifetime.
An expert in use cannot be evicted. This is the one piece of the memory manager where a subtle bug is
both easy to write and catastrophic under concurrency, so it is expressed in the type system rather than
in a convention.

### 5.5 I/O overlap

- Async readahead on the expert store.
- A bounded background load pool: resident experts compute while cold ones load.
- **Router-lookahead prefetch** — apply layer L+1's router to layer L's hidden state to predict the next
  layer's expert set. The prior art measured ~72% recall on one checkpoint. **That is a measurement, not
  a law.** Soma measures it per model *per layer* at admission (`pilot_profile`), and enables prefetch
  only on layers that clear the threshold. A layer with poor recall gets no prefetch, because a wrong
  prefetch is worse than none — it evicts something useful.

### 5.6 Heat and the pinned hot set

A persisted routing histogram (`expert_heat`) pins the hottest experts at startup so the cache is not
cold on first run. Bootstrapped at admission over a calibration corpus; updated with exponential decay
during serving.

---

## 6. KV persistence — one format, three callers

| Caller | What it needs |
|---|---|
| Warm conversation reopen | Reopen with zero re-prefill |
| Scheduler preemption (§4.3) | Evict a sequence under memory pressure, re-admit later |
| Cluster slot suspend/restore | Stop the process, resume byte-identically elsewhere |

These are the same operation. They get one format: versioned, `arch_hash`-gated, tagged with the
attention backend's `persist_format_id`, and per-sequence.

Per-sequence is a real change from the fallback. llama.cpp's `POST /slots/0?action=save`
([slot_manager.cpp:322](../src/node/slot_manager.cpp:322)) hardcodes sequence 0, so a `--parallel > 1`
slot only ever checkpoints its first sequence — a latent data-loss bug in the current system that the
rebuild does not inherit.

---

## 7. Before / after — node side

The node supervises engine subprocesses and talks to them over an OpenAI-compatible HTTP boundary. That
instinct is right and is kept. What changes is that the machinery stops being llama-shaped.

| Before | After | Why |
|---|---|---|
| `RuntimeProcess` with llama-specific `start_llama_server` | `EngineProcess` taking `EngineLaunchSpec{exe, argv, env, port}` + a `ReadinessProbe` variant | `start_with_args(runtime_name, exe, argv, port, timeout)` ([runtime_process.hpp:54](../include/node/runtime_process.hpp:54)) is *already* engine-neutral and private. This promotes it rather than wrapping it again. |
| `SlotManager`; `Slot` holds `unique_ptr<RuntimeProcess>` + `llama_server_path_` | `EngineSupervisor` owning `Engine` records; `EngineDescriptor` supplies launch/probe/kv/capability hooks | One manager, N engine kinds. llama.cpp and Soma become two descriptors, not two code paths. |
| `RuntimeClient::stream_complete` is **non-virtual** | `EngineClient` with virtual `stream_complete`; `LlamaEngineClient`, `SomaEngineClient` | The non-virtual streaming path is why control bypasses `RuntimeClient` entirely and calls `stream_post("/api/node/infer", …)` inline ([control_api_server.cpp:1949](../src/control/control_api_server.cpp:1949)). Making it virtual removes the reason for the bypass. |
| KV save/restore hardcoded to `POST /slots/0?action=save\|restore`, seq 0 only | `KvCheckpointBackend` interface; `LlamaKvBackend` (current wire format) and `SomaKvBackend` (per-sequence, versioned) | §6. |
| `LlamaRuntimeStatus` as a single scalar on `NodeState`, 7 `Llama*Callback` types, 6 `/api/node/runtime/llama/*` routes | `RuntimeStatus` **map keyed by runtime id**; `/api/node/runtime/{id}/*`; one callback set parameterized by id | The largest single surface to generalize, and the one that silently assumes exactly one runtime exists. |
| `backend != "llama-cpp"` → 400, duplicated at [node_api_server.cpp:449](../src/node/node_api_server.cpp:449) and `:911` | Descriptor-registry lookup; unknown backend → 400 listing the registry's actual contents | Duplicated string literals are how the second one gets missed. |
| `NodeConfig` — 13 flat `llama_*` fields | Sectioned config, over a fixed `ConfigFile` | `ConfigFile` parses `[section]` headers and **discards** them ([config_file.hpp:17](../include/common/config_file.hpp:17)). Namespacing them as `section.key` avoids a `soma_*` prefix explosion; files without section headers keep working unchanged. |
| **No crash supervision.** Once `state_ == Ready`, nothing polls the child. A dead engine stays `SlotState::Ready` until an inference request fails at the HTTP layer. | `EngineSupervisor` watchdog: child-exit detection promotes the engine to `Error` and publishes it | Pre-existing gap, not introduced by Soma — but streaming makes engine crashes more likely (I/O pressure, OOM under a mis-sized cache cap), so it gets fixed as part of the rebuild. |

**Readiness detection needs no change.** `RuntimeProcess::poll_health()`
([runtime_process.cpp:676](../src/node/runtime_process.cpp:676)) already polls `GET /health` with early
abort on child exit and a 600 s budget. There is no log sentinel anywhere in this codebase, so the
Windows sentinel fragility the kickoff warns about cannot recur. `ReadinessProbe` keeps `HttpHealth` as
the default and adds `StdoutJsonLine` only because it costs nothing to declare.

---

## 8. Before / after — control side

| Before | After | Why |
|---|---|---|
| `AgentScheduler` — *"VRAM-aware scheduler for llama.cpp agents"*, hard `is_llama_backend` gate that released the agent and returned `nullopt` | **`AgentScheduler`, rewritten in place** — backend-agnostic, verdict-driven, `ResourceFootprint{vram_mb, ram_mb, disk_mb}`. `resolve_backend()` returns `BackendRouting{engine_id, reason}`; the surviving `is_llama_backend` check now only separates node-local agents from API-backed ones, and Soma passes straight through it. | The gate was the first thing a Soma agent hit. Placement must select a backend, not assume one. **This row shipped into the existing class, not a new one** — see the note below. |
| `estimate_inference_vram_mb` — single-file `fs::file_size`, **2048 MB flat fallback for any directory** ([inference_sizing.cpp:87](../src/common/inference_sizing.cpp:87)) | `FootprintEstimator` — recursive directory sizing for the fallback path; for Soma the footprint is **read from `plan --json`, never estimated** | Every converted Soma model dir would otherwise size identically. This is also a live bug on the fallback path, since multi-shard HF dirs hit it. |
| `nodes_with_available_vram(int64_t)` — ignores `disk_free_mb` despite the health poll collecting it | `nodes_with_capacity(ResourceFootprint)` — all three axes | Soma's footprint is RAM + disk + optional VRAM. A different shape, not a different number. |
| `response_indicates_capacity_pressure` — substring-matches six English phrases against the node's error body | Structured `{"error":{"code":"capacity_pressure"}}` from both engines; the same function now reads the code as authoritative and keeps the six phrases only as a rolling-upgrade fallback | A new engine would otherwise have to reproduce those literals verbatim to get evict-and-retry. |
| **No control-side database.** Only per-agent `agent.db`. | `control.db` — model registry, API tokens + scopes, placement history | The registry is the source of truth for the verdict, and the verdict drives everything. |
| Flat bearer token, one `SetPreRoutingHandler`, no scope mechanism at all | Scope-annotated route table + `RouteScope` middleware: `read` / `chat` / `operator` | Admission kicks off hours-long, resource-consuming work. It must not sit behind the same token that lets a client send a chat message. Telemetry gets read-only. |
| `SseEmitter` — dead code, zero call sites | **Deleted.** `SseInferCtx` for chat; new `TelemetryFeed` (coalescing, rate-limited) for tier/heat | |
| `ModelRouter` — documented in `CLAUDE.md`, **does not exist** | Removed from the docs; `AgentScheduler::resolve_backend()` owns backend selection | Zero hits repo-wide. Routing actually lives in `resolve_openai_agent_model()`, `ensure_agent_running()`, and an anonymous-namespace `NodeProxyRuntimeClient`. |
| `AgentQueue` — per-agent FIFO worker thread | **Retained, unchanged.** | See below. |

### 8.0 The "After" column landed in `AgentScheduler`, and `PlacementEngine` never existed

This table was written against a planned class called `PlacementEngine`, declared in
`include/control/placement_engine.hpp`. **That header was never implemented, never included, and
never named by any CMakeLists** — `grep -rn PlacementEngine --include=*.cpp` returned zero hits for
its whole life. Every idea in it shipped anyway, into `AgentScheduler`:

| The header proposed | Where it actually lives |
|---|---|
| backend selection first, recorded with a reason | `AgentScheduler::BackendRouting{engine_id, reason}`, `resolve_backend()` / `resolve_backend_for()`, delegating to `soma::select_backend()` and keeping its `explain()` |
| `ResourceFootprint{vram, ram, disk}`, disk a real constraint | `common/footprint.hpp`; `NodeRegistry::nodes_with_capacity(const ResourceFootprint&)`; disk headroom checked on every placement. **Supply side only so far** — every footprint control constructs still sets `vram_mb` alone (roadmap D62) |
| structured capacity pressure, not six English substrings | `{"error":{"code":"capacity_pressure"}}` from both engines, with the substring match kept only as a rolling-upgrade fallback for a pre-code node |
| the two-mutex split, kept | `schedule_mutex_` / `state_mutex_`, unchanged |
| suspend/restore promoted to `/v1` | `POST /v1/agents/{id}/{suspend,restore,release}` (roadmap D51) |

So the header was deleted (roadmap D46) rather than implemented: it described the current design in
the future tense, quoted a gate that no longer does what it said, and named a class that would have
been a second implementation of a scheduler that already works. **It is kept here as a caution.** A
design written only in a header that nothing compiles cannot be contradicted by a build, so it stays
plausible indefinitely — and the suspend/restore gap it correctly identified sat open for exactly
that reason, because being right in an uncompiled file fails nothing.

### 8.1 `AgentQueue` stays, and the layering is deliberate

It looks like the FIFO-in-front-of-a-single-sequence-engine that §4 exists to replace. It is not, and
the distinction is worth stating once so it is not re-litigated:

- `AgentQueue` serializes **one agent's own turns**. An agent is a conversation with mutable state;
  two simultaneous turns on one agent is a semantic error regardless of engine capability.
- Soma's `max_batch` is **intra-engine concurrency across different agents**.

They compose exactly: N agents with in-flight turns produce N rows in one union forward. The thing being
removed is the assumption that the engine below can only do one sequence — not the per-agent ordering
guarantee above.

### 8.2 Verdict-driven backend selection

The registry `verdict` column is the source of truth:

| Verdict | Backend | Rationale |
|---|---|---|
| `stream` | Soma | Streaming economics are favourable |
| `hybrid` | Soma | Partial residency wins |
| `resident-only` | **Fallback** | Fits, but streaming buys nothing (Mixtral-class) |
| `reject` | **Fallback** | Failed conformance stage 1 or 2 |
| *no record* | **Fallback** | Absence of a record is not evidence of admissibility |

Operator override is `backend_override: auto | soma | fallback`, **persistent on the agent**. A
per-placement override would evaporate on the next eviction cycle and read as a flapping bug. The
decision and its reason are visible on `GET /v1/placements`.

---

## 9. Admission — the offline pipeline

Admission maps `(arch config, weights)` → `(canonical container, specialization header, registry rows,
conformance report, verdict)`. Per-model recompilation is acceptable; the target is **model IR +
compile-time specialization**, not a runtime interpreter.

### 9.1 The conformance ladder — what makes this real

```
1. fp32 path,      tiny random model, token-exact teacher-forced   → fail: reject → fallback
2. quantized path, tiny random model, token-exact greedy           → fail: reject → fallback
3. real checkpoint, logit-KL vs fp16 reference, few hundred positions
4. accuracy floor on a held-out task
```

`make_oracle.py <hf_repo>` builds a tiny-random model with the **real** architecture config and runs a
`transformers` teacher-forcing + greedy oracle.

**Stage 3 failing while 1 and 2 pass is a quantization finding, not a correctness bug** — different
remediation entirely (re-quantize a role, raise a group-scale granularity), and conflating the two costs
days.

### 9.2 Profiling passes

All write to the registry, all re-measured per model rather than inherited as constants:

| Pass | Output | Consumed by |
|---|---|---|
| Streaming economics | `bytes/token = n_moe_layers × top_k × expert_bytes` against **measured** disk bandwidth at this model's expert size | the verdict |
| Router-lookahead recall, per layer | `pilot_profile.recall_at_k` | prefetch enable, per layer |
| Kernel autotune over the model's actual shape set | `kernel_choice` rows → static dispatch table in the specialization header | `Kernels` |
| Heat bootstrap over a calibration corpus | `expert_heat` | startup pin |

The four findings carried from the prior art are all **measurements, not laws**: lookahead recall,
expert size, which kernel wins at a given shape (int4 single-row measured *slower* than fp32 there), and
whether streaming pays at all. Each is re-derived here.

### 9.3 Weight normalization

A per-family adapter maps upstream tensor names to canonical roles, then writes our layout (§5.3).
**Routers stay fp32, unconditionally** — enforced by schema validation, not convention. Quantizing
router logits changes *which experts fire*: a semantic change, not a precision one.

---

## 10. Determinism

Quantized integer kernels are shape-dependent. Batched, GPU, and speculative forwards round differently
from the single-row path, and at int4 that can flip argmax ties. Concurrency makes this unavoidable:
**a greedy request's exact token stream can depend on who else is on the server.**

Every emitted token remains the argmax of a *valid* forward, so quality holds. Reproducibility across
batch sizes does not. There is no free middle ground.

The policy is both halves of the choice, because they answer different questions:

- **Default: accept and document.** Batching on, tokens vary with batch composition.
- **`determinism: "strict"`** pins a sequence to a serialized single-row path with the single-row kernel
  family. Available as a **per-request field** on chat *and* as a `runtime_settings` field so a specific
  agent can demand it permanently.

Per-request is what makes the tradeoff visible in the API surface rather than only in this document —
which was the actual requirement.

---

## 11. Module graph and enforcement

```
serve ──────> scheduler ──────> F32Backend
                 │                    │
                 ├──> memory_hierarchy ──> expert_store ──> disk
                 ├──> kv_checkpoint
                 ├──> kernels          (dispatch table from registry)
                 └──> telemetry ──────> node ──> control ──> /v1/*

plan ───────> AttentionBackend  (sizing + KV format only)
model  — the three tiers (§3); data every box above operates on
plan   — derived from registry + host probe; configures scheduler and reports to placement
```

Three checks, all mechanical, all in CI (§ [roadmap](roadmap.md)):

**1. Header self-test.** Every `include/soma/**.hpp` compiles standalone, included twice, under
`/W4 /WX /std:c++20 /permissive-` and `-Werror`. Catches missing includes, cycles, and broken guards
before any implementation exists.

**2. Seam falsification** — [`tools/ci/check_seam.py`](../tools/ci/check_seam.py). Two rules:

- **R1** — no core file includes `soma/arch/…`. Only the backends themselves, the single
  `src/soma/arch_registry.cpp` resolver TU, and the tests may.
- **R2** — architecture-specific identifiers (`mla`, `gqa`, `rope`, `swiglu`, `sigmoid`, `yarn`,
  `deepseek`, `qwen`, `mixtral`, …) appear in core **code** nowhere except `arch_ir.hpp`.

R2's allow-list for `arch_ir.hpp` is the load-bearing distinction, not an exemption. The IR is a
*description* — it has to be able to say "this model is MLA with `kv_lora_rank` 512". That is data
passing through the core to a backend. Everything else under `include/soma/` and `src/soma/` is core
*logic*, where an architecture name means a branch has leaked somewhere family-agnostic. Comments and
string literals are stripped before matching, so explaining *why* an interface is shaped a certain way
is never a violation — it is the opposite of one.

> This check earned its keep during the design pass. It caught `KvFormatId` enumerating
> `GqaFullKv_v1` / `MlaLatent_v1` in a core header — which would have meant **adding a third
> architecture required editing core**. The tag is now a backend-owned FNV hash of a format name. It
> also caught two core structs defaulting to `AttentionFamily::Gqa`, precisely the quiet bias toward
> the first architecture shipped that the two-family co-design exists to prevent. `AttentionFamily`
> now has an `Unknown` zero value and there is no default family anywhere.

**3. Two-family conformance ladder** on tiny-random fixtures, every commit. Without it the second
architecture silently breaks the first within a month; kernel-shape sensitivity makes
cross-architecture regressions unusually easy to introduce and hard to spot.

---

## 12. Deferred, and named as such

Explicitly out of scope for v1, listed so they are not mistaken for oversights:

- **GPU tier residency and GPU kernels** — tier declared and reported, empty until after G4 (§5.2).
- **Paged KV with a block table** — attention loops per-sequence inside the batched layer; the interface
  is shaped so this can land without changing it (§2.2).
- **Speculation under batching** — off when `batch > 1` (§4.4).
- **Multimodal** — text-only. Image content parts are rejected with **422**, never dropped silently.
- **Training, fine-tuning, novel quantization research** — inference only; we consume quantized weights
  and requantize at admission.
- **Beating llama.cpp on small dense models** — that is what the fallback is for.
- **Any second UI framework or bundled web UI** — FTXUI is this project's only UI. Richer external
  visualization belongs to a separate stack consuming the same `/v1/*` routes, which is precisely why P1
  is non-negotiable.

---

## See also

- [external-api.md](external-api.md) — the full `/v1/*` surface, auth scoping, streaming semantics
- [mantic-mind-integration.md](mantic-mind-integration.md) — engine↔node boundary, verdict routing, FTXUI
- [roadmap.md](roadmap.md) — G0–G8 validation gates
- [../schemas/arch-ir.md](../schemas/arch-ir.md) — the architecture IR
