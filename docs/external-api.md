# External Client API — `/v1/*`

> **P1 — the API is the single control plane.** Every capability Soma introduces is reachable here.
> The FTXUI TUI is *one client*; a separate SvelteKit debug stack is another. If the TUI can do it, an
> API client can do it through the same route. There are no TUI-only features and no internal-only
> capabilities.

Base: `http://<control>:9090`. A separate OpenAI-compatibility listener runs on `:9091` (§9).

---

## 1. Authentication and scopes

### What exists today

One flat bearer token (`ControlConfig::external_api_token`), compared with a plain `!=`, gating every
`/v1/*` path identically, via a single `SetPreRoutingHandler`. There is **no scope, role, or permission
concept anywhere in the codebase**. Auth is entirely opt-in: an empty token means the whole surface is
open.

### What replaces it

Three scopes, split by **blast radius** rather than by resource:

| Scope | Covers | Blast radius |
|---|---|---|
| `read` | Every GET; every telemetry SSE stream | No side effects |
| `chat` | Agent chat, conversations, memories, attachments, curation | Bounded, per-agent |
| `operator` | Admit, reprofile, delete, verdict override, backend override, suspend/restore, node pairing | Unbounded or destructive |

Tokens are rows in `control.db` (`api_token`), storing **only** `sha256(token)`. The plaintext is
returned once at creation and never persisted.

`POST /v1/models/admit` is the reason this exists. It starts hours of conversion and profiling and
consumes tens of GB. It must not sit behind the same credential that lets a client send a chat message.
Symmetrically, a telemetry dashboard that only ever reads should not hold a token that can delete an
agent.

```http
Authorization: Bearer <token>
```

| Failure | Status | Body |
|---|---|---|
| No/malformed header | 401 | `{"error":{"code":"missing_bearer_token"}}` |
| Unknown or revoked token | 403 | `{"error":{"code":"invalid_bearer_token"}}` |
| Valid token, insufficient scope | 403 | `{"error":{"code":"insufficient_scope","required":"operator","granted":["read","chat"]}}` |

**Backwards compatibility.** The existing `external_api_token` is grandfathered as an all-scopes
credential, matched *before* the database lookup and never inserted as a row — so rotating it in config
cannot leave a stale grant behind. If it is empty **and** `api_token` has no rows, auth is off exactly
as today, and control logs that loudly at startup: a system that is open because nobody configured it
should say so.

**Route coverage is exhaustive and checked at startup.** Every route has an entry in
`route_scope_table()`; a registered handler with no entry is a startup failure, not a default.
Defaulting an unlisted route to `read` would silently under-protect a new mutation; defaulting to
`operator` would silently break a new GET. Both are worse than refusing to start.

---

## 2. Admission and registry

### ⚠ `/v1/models` is reclaimed

Today `GET /v1/models` on `:9090` returns `mantic_model_list(agents)` — agents wearing model costumes —
plus an `openai_compat_note` pointing at port 9091. Under P2 the honest meaning wins: **`/v1/models` is
the admission registry.** The agents-as-models catalog remains on the `:9091` OpenAI-compat listener,
where clients expecting OpenAI semantics already look for it.

### `POST /v1/models/admit` — scope: `operator`

Convert, compile the tokenizer, run the conformance ladder, profile, write registry rows, compute a
verdict. **Hours long.** Progress over SSE.

```jsonc
// request
{ "source": "Qwen/Qwen3-30B-A3B",   // HF repo id or local path
  "quantization": { "expert_gate": "q4_k", "expert_down": "q6_k" },  // optional; per tensor role
  "calibration_corpus": "default" }
```

```
data: {"type":"progress","stage":"convert","step":3,"total_steps":9,"fraction":0.28,"detail":"shard 3/12"}
data: {"type":"conformance","stage":"quant_tiny_greedy","passed":true}
data: {"type":"done","model_id":42,"arch_hash":"…","verdict":"stream"}
data: {"type":"error","stage":"conformance","message":"…"}
data: [DONE]
```

Stages: `fetch → convert → tokenize → conformance → profile → finalize`. Cancel with
`POST /v1/models/admit/{operation_id}/cancel` (`operator`).

**Failing conformance stage 1 or 2 yields `verdict: "reject"` — it does not fail the request.** A
rejected model is a successfully-admitted *record* saying "run this on the fallback". That distinction
matters: the operator asked whether Soma can run it, and "no, here's why" is an answer.

### `GET /v1/models` — scope: `read`

```jsonc
{ "models": [{
    "id": 42, "arch_hash": "…", "name": "Qwen3-30B-A3B",
    "attention_family": "gqa",
    "n_layers": 48, "n_moe_layers": 48, "n_experts": 128, "top_k": 8,
    "economics": { "expert_bytes": 2359296, "bytes_per_token": 906362880,
                   "total_routed_bytes": 15569256448, "active_fraction": 0.0625 },
    "verdict": "resident-only",
    "verdict_basis": { "ram_budget_gb": 32, "ctx": 4096, "quant": "q4_k", "host": "n5" },
    "verdict_reason": "routed set (14.5 GB) fits the RAM budget; streaming has nothing to do",
    "admitted_at": 1753574400000 }] }
```

**The `verdict` field is the admission-host verdict, and it is not a property of the model.** It is a
property of `(model, quantization, host budget)`. Qwen3-30B-A3B is `resident-only` at q4 on a 32 GB host
and `stream` at bf16. Clients wanting the *effective* verdict for a particular node must read
`/v1/models/{id}/plan` or `/v1/placements`, never this field alone.

### `GET /v1/models/{id}` — `read`
Full record plus the complete `arch.json`.

### `GET /v1/models/{id}/plan?node_id=<id>` — `read`

The `plan --json` document, **byte-identical to what `soma plan --json` prints** — one serializer, so
the CLI and the API can never disagree about a footprint. Computed from headers only; allocates
nothing; safe to call for a node that could not host the model.

```jsonc
{ "arch_hash": "…", "model_name": "Qwen3-30B-A3B",
  "footprint": { "vram_mb": 0, "ram_mb": 19840, "disk_mb": 14848 },
  "dense_resident_bytes": 3221225472,
  "kv_bytes_at_ctx": 3435973836,
  "expert_cache_bytes": 12884901888,
  "vram_hot_bytes": 0,
  "cap_per_layer": 96,
  "expected_unique_experts_per_step": 7.4,
  "max_batch": 12,
  "expert_set_fully_resident": false,
  "bytes_per_token": 906362880,
  "projected_tok_s": 5.9,
  "prefetch_enabled_layers": 31,
  "verdict": "stream",
  "verdict_reason": "active_fraction 6.3% with 2.4 MB experts; 14.5 GB routed set exceeds the 12.9 GB cache" }
```

Note `max_batch: 12` is **derived, not configured** — `cap_per_layer / expected_unique_experts_per_step`.
See §5.

### `GET /v1/models/{id}/conformance` — `read`

```jsonc
{ "stages": [
  { "stage": "fp32_tiny_tf",      "passed": true,  "ran_at": …, "detail": {"positions": 512, "mismatches": 0} },
  { "stage": "quant_tiny_greedy", "passed": true,  "ran_at": …, "detail": {"tokens": 256, "mismatches": 0} },
  { "stage": "real_logit_kl",     "passed": false, "ran_at": …, "detail": {"mean_kl": 0.031, "threshold": 0.02} },
  { "stage": "accuracy_floor",    "passed": true,  "ran_at": …, "detail": {"score": 0.71, "floor": 0.68} } ] }
```

**Stage 3 failing while 1 and 2 pass is a quantization finding, not a correctness bug.** The remediation
is different — requantize a role, raise group-scale granularity — and conflating the two costs days.
The response therefore reports stages individually rather than a rolled-up boolean.

### `GET /v1/models/{id}/heat?resolution=bucketed|full` — `read`
Persisted routing histogram. `bucketed` (default) caps at 4096 cells.

### `POST /v1/models/{id}/reprofile` — `operator`
Re-runs profiling only (no conversion, no conformance). SSE, same frames as admit. Does **not** change
`arch_hash`: economics are outside the hash precisely so re-profiling on faster disks does not
invalidate KV checkpoints.

### `PUT /v1/models/{id}/verdict` — `operator`
```jsonc
{ "verdict": "stream", "reason": "forcing Soma for G4 seam validation" }
```
Recorded with the reason, so the override is visible rather than mysterious.

### `DELETE /v1/models/{id}` — `operator`
409 if any live placement references it.

---

## 3. Placement and backend selection

### `GET /v1/placements` — `read`

Extended with the decision and **its reason**. "Which backend" is far less useful than "which backend,
and why".

```jsonc
{ "placements": [{
    "agent_id": "agent-1", "node_id": "node-a", "slot_id": "slot-3",
    "engine_id": "soma",
    "backend_reason": "verdict",
    "backend_detail": "registry verdict 'stream' for arch_hash …",
    "footprint": { "vram_mb": 0, "ram_mb": 19840, "disk_mb": 14848 },
    "footprint_source": "plan",        // "plan" = measured | "estimate" = from file size
    "model_id": 42,
    "suspended": false, "is_active": true }] }
```

`backend_reason` ∈ `verdict` | `no_admission_record` | `operator_override` | `verdict_reject` |
`resident_only` | `remote_api`.

### Routing policy

| Verdict | Engine | Why |
|---|---|---|
| `stream` | Soma | Streaming economics favourable |
| `hybrid` | Soma | Partial residency wins |
| `resident-only` | **fallback** | It fits; streaming buys nothing |
| `reject` | **fallback** | Failed conformance stage 1 or 2 |
| *no record* | **fallback** | Absence of a record is not evidence of admissibility |

### `PUT /v1/agents/{id}/backend` — `operator`

```jsonc
{ "backend_override": "auto" | "soma" | "fallback" }
```

**Persistent, on the agent.** Placement is recomputed on every `ensure_agent_running`; a per-placement
override would evaporate on the next eviction cycle and read as a flapping bug.

This is also how G4 runs: DeepSeek-V2-Lite at q4 admits as `resident-only` (7.2 GB routed set fits in
RAM), so forcing `soma` is exactly what validates the seam against a second attention family.

---

## 4. Runtime telemetry

### `GET /v1/engines` — `read`
Live engines cluster-wide, with a tier-occupancy summary per engine.

### `GET /v1/engines/{engine_id}/telemetry` — `read` — **SSE**

| Query | Default | Range | Notes |
|---|---|---|---|
| `hz` | `2` | 1–10 | Frame rate, clamped server-side |
| `resolution` | `bucketed` | `bucketed` \| `full` | `full` is an explicit opt-in |
| `include` | `occupancy,cache,scheduler` | + `heat` | Heat is opt-in even bucketed |

```
data: {"type":"tick","tick_ms":…,
       "occupancy":{"vram_experts":0,"ram_experts":2914,"disk_experts":3230,
                    "pinned_experts":512,"ram_bytes":12884901888,"ram_capacity_bytes":12884901888},
       "cache":{"hits":184203,"misses":9117,"evictions":8605,
                "prefetch_hits":5512,"prefetch_wasted":388,"io_wait_ns":…},
       "scheduler":{"active_sequences":7,"current_batch":7,"effective_max_batch":12,
                    "unique_experts_last_step":41,"naive_expert_reads_last_step":56}}
data: {"type":"heat","layer_bucket":1,"expert_bucket":2,"n_layers":48,"n_experts":128,
       "cells":[[0,0,"ram",1842,0.71], …]}   // [layer, expert, tier, count, decayed]
```

**Throttling is not advisory.** Heat counters are aggregated **inside the engine** and sampled at the
tick rate — nothing is emitted per token. A per-token brain-grid feed over tens of thousands of experts
is orders of magnitude above the chat event stream, and a throttle applied at the HTTP layer would still
have paid for producing the data. Aggregating engine-side means a careless dashboard client cannot
saturate control even by asking for everything.

`bucketed` caps the grid at **4096 cells**. A 48×128 model (6144 cells) buckets 2 experts per cell; a
60×256 model (15360) buckets more. `full` requires the `read` scope and an explicit
`resolution=full` — it is available, just never accidental.

`unique_experts_last_step` vs `naive_expert_reads_last_step` is the batch-union payoff made observable.
A ratio near 1.0 means the union is buying nothing and something upstream is wrong.

### `GET /v1/engines/{engine_id}/heat` — `read`
Non-streaming snapshot, same shape as the `heat` frame.

---

## 5. Concurrency and slots

### `GET /v1/engines/{engine_id}/slots` — `read`

```jsonc
{ "slot_id": "slot-3", "engine_id": "soma",
  "queue_depth": 2, "current_batch": 7,
  "effective_max_batch": 12,
  "max_batch_limited_by": "expert_cache",   // "expert_cache" | "kv_slots" | "config"
  "sequences": [
    { "index": 0, "agent_id": "agent-1", "position": 1204, "kv_tokens": 1204,
      "prefilling": false, "suspended": false, "determinism": "batched" } ] }
```

`effective_max_batch` and `max_batch_limited_by` are reported so an operator can see **why**
concurrency is limited rather than inferring it from throughput. When the limiter is `expert_cache`, the
gate is `cap_per_layer / expected_unique_experts_per_step` — raising the RAM budget raises concurrency;
raising a config number does not.

Nothing like `sequences` exists today: the node keeps one `active_requests` counter per slot and
llama-server assigns its own internal slots invisibly. A stalled sequence and a saturated batch look
identical from a request counter.

### `POST /v1/engines/{id}/slots/{n}/suspend` | `/restore` — `operator`

**The first operator-visible suspend/restore.** Both mechanisms exist today but only as internal
scheduler decisions reachable through the node API; there is no `/v1/*` route for either. Promoted
because P1 says a capability the system has is a capability the API exposes.

`suspend` returns 409 with `{"error":{"code":"multi_sequence_unsupported"}}` when the engine has live
sequences beyond index 0 and its KV backend cannot checkpoint them. The fallback is in exactly that
position — `POST /slots/0?action=save` hardcodes sequence 0, so a `--parallel > 1` slot currently saves
its first sequence and **silently discards the rest**. The rebuild makes that an explicit refusal.

---

## 6. Determinism

Quantized integer kernels are shape-dependent: batched and single-row forwards round differently, and at
int4 that can flip argmax ties. **A greedy request's exact token stream can therefore depend on who else
is on the server.** Every emitted token remains the argmax of a valid forward, so quality holds;
reproducibility across batch sizes does not. There is no free middle ground.

Both halves of the tradeoff are exposed, because they answer different questions:

**Per-agent default** — `PUT /v1/agents/{id}`:
```jsonc
{ "runtime_settings": { "determinism": "batched" | "strict" } }
```

**Per-request override** — `POST /v1/agents/{id}/chat`:
```jsonc
{ "message": "…", "determinism": "strict" }
```

`strict` pins the sequence to a serialized single-row path **and** the single-row kernel family. Both
halves are required: a batched kernel invoked with `m == 1` may still round differently from the
dedicated single-row path.

A `strict` request under load queues rather than joining a batch. The response reports what it got:

```jsonc
{ "type": "done", "determinism": "strict", "batch_size_at_emit": 1 }
```

Putting this on the request rather than only in a startup flag is what makes the tradeoff **visible in
the API surface** rather than only in a document — which was the actual requirement.

---

## 7. Errors

Uniform envelope. Machine-readable `code` first, prose second.

```jsonc
{ "error": { "code": "capacity_pressure", "message": "…", "detail": { … } } }
```

| Code | Status | Meaning |
|---|---|---|
| `missing_bearer_token` | 401 | |
| `invalid_bearer_token` | 403 | |
| `insufficient_scope` | 403 | Includes `required` and `granted` |
| `validation_failed` | 422 | Existing agent-config shape, with `issues[]` |
| `unsupported_content` | 422 | Image content part; **text-only v1** |
| `model_not_admitted` | 409 | Soma requested, no record |
| `capacity_pressure` | 503 | Retryable; drives evict-and-retry |
| `multi_sequence_unsupported` | 409 | Suspend on a multi-seq slot the backend can't checkpoint |
| `engine_unavailable` | 503 | Runtime not provisioned on any node |

**`capacity_pressure` as a code replaces substring matching.** `AgentScheduler::response_indicates_capacity_pressure`
currently searches the node's error body for six English phrases (`"max slots reached"`,
`"no available ports"`, `"out of memory"`, …). A new engine would have had to reproduce those literals
verbatim to earn an evict-and-retry. Both engines emit codes now.

**Image parts are rejected with 422, never dropped silently.** Silent dropping is the failure mode worth
designing out — a multimodal request that returns a plausible text-only answer is worse than one that
fails.

---

## 8. Scope map for the existing surface

All 51 pre-existing routes, annotated. Unchanged in shape; this is the retrofit.

| Scope | Routes |
|---|---|
| `read` | `GET /v1/nodes`, `/v1/nodes/discovered`, `/v1/activity`, `/v1/performance`, `/v1/agents`, `/v1/agents/:id`, `/v1/placements`, `/v1/agents/:id/conversations`(+`/:cid`, `/local-memories`), `/v1/agents/:id/memories`, `/v1/agents/:id/voice`(+`/proposals`, `/proposals/:pid/sample`), `/v1/agents/:id/attachments/:aid`, `/v1/agents/:id/speech/cache/:cid` |
| `chat` | `POST /v1/agents/:id/chat`, `POST/PUT/DELETE` on conversations, local-memories, memories, `POST /v1/agents/:id/memories/extract`, `/curation/proposals`, `/curation/apply`, `…/:cid/compact`, `…/:cid/activate`, `POST/DELETE /v1/agents/:id/attachments`, `POST /v1/agents/:id/speech`, `POST /v1/audio/speech`, voice proposal create/sample |
| `operator` | `POST/DELETE /v1/agents`, `PUT /v1/agents/:id`, `POST /v1/nodes`, `DELETE /v1/nodes/:id`, `POST /v1/nodes/:id/forget`, all `/v1/nodes/pair/*`, `DELETE /v1/performance`, voice `approve`/`reject`, **and every new route in §2–§5 marked `operator`** |

Two judgement calls worth naming:

- **`DELETE /v1/performance` is `operator`, not `chat`.** It destroys shared observability state that
  another client may be mid-analysis on.
- **Voice `approve`/`reject` is `operator`, not `chat`.** Approving a proposal mutates the agent's
  persistent identity; creating and sampling one does not.

`POST /api/control/register-node` keeps its separate registered-node auth and is deliberately outside
the `/v1` scope gate, as today.

---

## 9. OpenAI-compatibility listener (`:9091`)

Unchanged: `GET /v1/models`, `GET /v1/models/:model`, `POST /v1/chat/completions`. Still gates every path
with a flat token — clients expecting OpenAI semantics do not expect scopes, and adding them would break
every off-the-shelf SDK for no security gain the `:9090` surface does not already provide.

This listener keeps the **agents-as-models catalog** displaced from `:9090` by §2.

Known limitation, unchanged: the compat stream drops `thinking` and `tool_call` deltas
(`if (chunk.delta_content.empty()) return;`). Documented rather than silently inherited — the Mantic SSE
surface on `:9090` carries all four event types.

---

## 10. SSE semantics

Chat (`POST /v1/agents/:id/chat`) is unchanged:

```
data: {"type":"thinking","content":"…"}
data: {"type":"delta","content":"…"}
data: {"type":"tool_call","id":"…","name":"…","arguments":"…"}
data: {"type":"done","conv_id":"…","success":true}
data: [DONE]
```

Two defects in the current node-side mapping are fixed rather than carried forward
([node_api_server.cpp:1223](../src/node/node_api_server.cpp:1223)):

1. The chunk→event mapping is an `if / else-if` **priority chain**, so a chunk carrying both
   `thinking_delta` and `delta_content` silently drops one. It becomes independent emission.
2. `tool_result_json` is never emitted by `/api/node/infer` at all, despite being carried on
   `InferenceChunk`.

Exactly one `done` is always delivered, on every path including error — that guarantee is preserved.

**Telemetry streams are separate from chat streams** and are rate-limited independently (§4). A client
subscribing to both gets chat at production rate and telemetry at `hz`.

---

## See also

- [architecture.md](architecture.md) — the seam, the scheduler, determinism
- [mantic-mind-integration.md](mantic-mind-integration.md) — engine↔node boundary, FTXUI consumption
- [../schemas/arch-ir.md](../schemas/arch-ir.md) — `arch.json` and the verdict function
