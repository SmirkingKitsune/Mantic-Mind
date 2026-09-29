# External Client API — `/v1/*`

## Reading this document

A route heading marked **(planned)** describes a capability that is designed and
argued for but **not yet registered** — calling it answers 404. Everything else
describes a route that exists; `tools/ci/check_api_docs.py` fails the build if an
unmarked heading has no entry in the scope table.

That check exists because `GET /v1/models/{id}/conformance` sat here unmarked and
unregistered from the day this document was written. `require_complete_coverage()`
walks registered handlers and asserts each has a scope; nothing walked the other
way, so a documented route that was never built raised nothing. A documented route
that does not exist is worse than an undocumented one — it is a promise, and a
client written against it fails with a 404 that reads as a missing resource
rather than a missing endpoint.

---

> **P1 — the API is the single control plane.** Every capability Soma introduces is reachable here.
> The FTXUI TUI is *one client*; a separate SvelteKit debug stack is another. If the TUI can do it, an
> API client can do it through the same route. There are no TUI-only features and no internal-only
> capabilities.

Base: `http://<control>:9090`. A separate OpenAI-compatibility listener runs on `:9091` (§11).

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
| `chat` | A user's chat, conversations, memories, attachments, curation | Bounded, per-user |
| `operator` | Admit, reprofile, delete, verdict override, backend override, suspend/restore, node pairing | Unbounded or destructive |

Tokens are rows in `control.db` (`api_token`), storing **only** `sha256(token)`. The plaintext is
returned once at creation and never persisted.

`POST /v1/models/admit` is the reason this exists. It starts hours of conversion and profiling and
consumes tens of GB. It must not sit behind the same credential that lets a client send a chat message.
Symmetrically, a telemetry dashboard that only ever reads should not hold a token that can delete a
user and everything they own.

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

### Binding a token to one user

A token may name a `user_id` (`POST /v1/tokens`). Binding is **optional and narrowing only**: it can
never grant what the scopes do not, and it is checked after the scope, so the two failures stay
distinguishable.

A bound token may do, within its scopes:

| Where | What it may do |
|---|---|
| `/v1/users/{its own}/…` | everything its scopes allow |
| `/v1/users/{anyone else}/…` | nothing — `403 user_binding` |
| `GET /v1/users` | allowed, and **filtered to its own user** |
| `POST /v1/users`, `/v1/tokens*`, `/v1/activity`, `/v1/performance` | nothing — they span users |
| everything else under `/v1/*` | **read-only** (plus `POST /v1/chat/completions` and `/v1/audio/speech`) |

`POST /v1/audio/speech` additionally requires `voice: "user:{its own}"`.

`DELETE /v1/users/{id}` revokes every token bound to that user, and the response says how many
(`tokens_revoked`). A later user created under the same id must not inherit the old one's
credentials — which is also why a binding must name a user that already exists (400 otherwise).

An unbound token, the legacy flat token, and auth-off behave exactly as before: binding is something
an operator opts into per token.

**Route coverage is exhaustive and checked at startup.** Every route has an entry in
`route_scope_table()`; a registered handler with no entry is a startup failure, not a default.
Defaulting an unlisted route to `read` would silently under-protect a new mutation; defaulting to
`operator` would silently break a new GET. Both are worse than refusing to start.

---

## 2. Admission and registry

### ⚠ `/v1/models` is reclaimed

`GET /v1/models` on `:9090` used to return `mantic_model_list(agents)` — agents wearing model
costumes — plus an `openai_compat_note` pointing at port 9091. Under P2 the honest meaning wins:
**`/v1/models` is the admission registry.** The `:9091` OpenAI-compat listener serves the catalog
OpenAI clients pick a model from — the served models of §2b (§11).

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

Stages: `fetch → convert → tokenize → profile → conformance → finalize`. Cancel with
`POST /v1/models/admit/{operation_id}/cancel` (`operator`).

**The architecture is checked before conversion starts.** A `source` whose config Soma cannot parse
fails the request in milliseconds rather than after hours of conversion. One whose config parses but
whose attention family has no backend is admitted as a `reject` record — conversion is skipped, because
no host will ever read that container.

**`step` and `total_steps` describe THIS run, not a fixed ladder.** A local directory skips `fetch`; an
already-converted container (`admit_container`, and what `reprofile` runs) skips `fetch`, `convert` and
`tokenize` and reports 3 total. Read `stage` for what is happening and `fraction` for how far along;
`step`/`total_steps` are for rendering "2 of 6".

**`bytes_done` / `bytes_total` are populated by `fetch` only** — it is the one stage whose remaining
time is estimable. Zero elsewhere means "not reported", not "nothing to transfer".

`source` is a local path if one exists at that name, otherwise a HuggingFace repo id, optionally
`repo@revision`. A repo id becomes a directory under `sources_dir`, so it is validated as one: at most
one `/`, no `..`, no backslash. Auth is whatever `huggingface_hub` already resolves (`HF_TOKEN` or a
cached login) — control never reads or stores a credential.

**`quantization` is honoured, and it changes the model's identity.** `expert_gate`, `expert_down` and
`group` each fall back to the deployment default when omitted. Admitting the same `source` at a
different quantization produces a **second record with a different `arch_hash`**, in its own container
directory — not an update of the first. KV checkpoints written against one refuse to load under the
other. `POST /v1/models/{id}/reprofile` deliberately does none of that: it re-derives a verdict from the
same bytes and leaves the hash alone.

**Which record a served model gets when several match one name**: a verdict that selects Soma wins, then the
most recently profiled. Pass an `arch_hash` as the model ref to name a specific variant — it is the only
identity that cannot be ambiguous.

**A repo with no safetensors is refused unless `admission_allow_pickle` is set.** Converting `.bin`
weights means unpickling them, which executes code from the repo; that is an operator's decision per
deployment rather than a default. Framework duplicates (`.h5`, `.msgpack`, `.onnx`, `.gguf`) are never
transferred, and neither are `.bin` files in a repo that also ships safetensors.

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
  "ctx_size": 4096, "max_context": 32768, "kv_slots": 4,
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
See §7.

### `GET /v1/models/{id}/conformance` — `read`

Each stage carries `status` ∈ `passed` | `failed` | `skipped`, and `detail` as JSON. **Read `status`,
not `passed`** — `passed` is a boolean and cannot express "did not run", which is what most of this
ladder says on a serving host. `fp32_tiny_tf` and `real_logit_kl` need a `transformers` oracle for the
specific model; they are recorded as `skipped` with what they would need, never as passing.

Stages that DO run at admission: `quant_codec` (the container's declared formats against its own dense
weights — measured bits/weight versus an independently written formula, plus round-trip relative RMS)
and `tokenizer_roundtrip` (the compiled tokenizer against HF's own ids, byte-for-byte).

**A failed stage yields `verdict: "reject"`; the admission still succeeds.** A rejected model is a
successfully admitted *record* meaning "route this to the fallback".

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

## 2b. Served models — the catalog users pick from

A **served model** is the operator's answer to "how is this model served": which weights, which engine
policy, which launch settings, which default sampling. It belongs to nobody. A client names one as
`model`, every user who picks it shares its placement, and one conversation may be answered by several.

Two sources, one list:

| Source | What | Stored? |
|---|---|---|
| `admitted` | Every admission whose verdict Soma can serve (`stream`, `hybrid`) | **No** — computed from the registry on every read |
| `explicit` | Rows an operator wrote: llama.cpp GGUF models, remote-API models, and overrides of admitted ones | `control.db` `served_model` |

Rules worth knowing before writing a client:

- **Ids** are lowercase `[a-z0-9._-]`, at most 128 characters, and **immutable** — conversations and
  placement history name them. An admitted entry's id is a slug of its container directory's name
  (admission's own collision key), e.g. `qwen3-30b-a3b-q4_g-q6_g-g128`.
- An admitted entry is addressed by **`arch_hash`** and pinned to **`backend_override: "soma"`**: a name is
  ambiguous across quantizations, and under an engine order of `[llama-cpp, soma]` the `auto` policy would
  hand llama.cpp a container directory, which it cannot load.
- **`PUT` on an admitted id writes the operator's override under the same id**, so conversations that
  named it keep working. Deleting that override brings the admitted entry back. An admitted entry itself
  cannot be deleted (409) — its source is the admission; disable it with `enabled: false` instead.
- An explicit row that resolves to an admission **shadows** it: the admitted entry disappears from the list.
- `api_settings.api_key` is accepted and **never returned or stored** — process memory only, as it
  always was. Responses carry `api_key_configured`. An update that omits the key keeps the one held.
- `inference_backend` is `node` (served on a cluster node, engine chosen by routing) or `api` (a remote
  OpenAI-compatible provider). The agent-era value `llama-cpp` is refused: since Soma it meant "node", not
  "llama.cpp".

### `GET /v1/served-models` — `read`

```jsonc
{ "object": "list",
  "data": [{
    "id": "qwen3-30b-a3b-q4_g-q6_g-g128", "display_name": "Qwen3-30B-A3B (q4_g-q6_g-g128)",
    "model_path": "8f3a…",            // the arch_hash for an admitted entry
    "inference_backend": "node", "backend_override": "soma", "engine_model_name": "",
    "runtime_settings": { "ctx_size": 4096, "temperature": 0.7, … },
    "api_settings": { "base_url": "…", "chat_completions_path": "/v1/chat/completions",
                      "api_key_env": "OPENAI_API_KEY", "api_key_configured": false },
    "vision_settings": { "enabled": false, "mmproj_path": "" },
    "tool_calling": false, "preferred_node_id": "", "enabled": true,
    "created_at_ms": 1755400000000, "updated_at_ms": 1755400000000,

    "source": "admitted",             // admitted | explicit
    "available": true,                // false: disabled, or admitted with no Soma in the engine policy
    "unavailable_reason": "…",        // present only when unavailable
    "admission_id": 3,                // 0 = resolves to no admission
    "arch_hash": "8f3a…",
    "backend": "soma", "backend_reason": "soma (operator_override, verdict=stream)",
    "status": "active",               // unplaced | idle | active | suspended | api
    "placement": { "served_model_id": "…", "node_id": "…", "slot_id": "…", "in_flight": 2, … } // when placed
  }] }
```

Sorted by id. `backend` is the same pure routing decision the scheduler acts on; asking cannot start an
engine.

### `GET /v1/served-models/{id}` — `read`

One entry, as above, plus `node_compatibility` and **where this model has run**, newest first, at most
20 rows. 404 for an unknown id.

```jsonc
{ /* …the entry… */
  "node_compatibility": { /* … */ },
  "placement_history": [{
    "node_id": "node-b", "slot_id": "slot-7",
    "backend": "soma", "backend_reason": "soma (verdict=stream)",
    "footprint": { "vram_mb": 0, "ram_mb": 19840, "disk_mb": 14848 },
    "placed_at_ms": 1755400000000,
    "released_at_ms": 0,          // 0 = still open, never null
    "open": true
  }] }
```

`placement_history` reads the `placement_history` table, whose writer and index shipped with no caller
and no query (roadmap D60) — a row is inserted when a model acquires a slot (fresh load or
restore-from-suspend, not on a refresh of an unchanged placement) and closed when the placement ends.
It lives on this route rather than behind a route of its own: the index is already
`(served_model_id, placed_at DESC)`, and reusing a route that CLI and TUI already reach keeps both
parity directions at zero without inventing a surface for one reader.

### `POST /v1/served-models` — `operator`

Body: a served model; every field optional except `model_path`. Without an `id`, one is derived from
`display_name` (else the model path's file name) and made unique with `-2`, `-3`, ….

`201` with the entry and any validation **warnings** under `issues`. `400 {"error":"validation_failed",
"issues":[…]}` on errors — the same issue shape agents used. `409` when the id is taken by any entry,
admitted ones included.

### `PUT /v1/served-models/{id}` — `operator`

Replaces the entry (on an admitted id: writes the override, see above). An `id` in the body that differs
from the path is 400 — ids do not change. A changed **launch** setting (context size, GPU layers,
parallelism, …) does not detach the running placement: the next request that needs the new settings gets
`draining` until the requests using the old placement finish, then replaces it.

### `DELETE /v1/served-models/{id}` — `operator`

Removes an explicit row and releases its placement once nothing is using it. 409 for an admitted entry,
404 for an unknown id.

### `PUT /v1/served-models/{id}/backend` — `operator`

```jsonc
{ "backend_override": "auto" | "soma" | "fallback" }
```

Returns the **decision**, not just the setting — `{served_model_id, backend_override, backend,
backend_reason}` — because asking for `soma` on weights nothing has admitted is refused, and an operator
told only "saved" would reasonably believe it took effect.

Suspending, restoring and releasing a model's placement are in §4.

---

## 2c. Users, conversations and chat

A **user** owns conversations, the notes taken in them, global memories and a voice. A user owns no
model: one is chosen per turn from the catalog (§2b), so a single conversation can be answered by
several models, and switching costs a field rather than a migration.

This replaces agents, which bundled three unrelated things into one record — how a model is served, a
persona, and a private store. Changing the model meant changing the agent, so the conversations and
memories were tied to whichever weights happened to answer them.

**Ids** are immutable, lowercase `[a-z0-9._-]`, at most 64 characters. An id names the user's
directory (`data/users/{id}`) *and* keys any token bound to them (§1), so a rename would have to move
both at once, and a binding that missed the move would reach nobody — or somebody new. `POST /v1/users`
derives an id from `display_name` when the body gives none.

Nothing reads `data/agents/`. It is left on disk, untouched and unimported; control says so once at
startup if it is there.

### Which model answers a turn

1. the `model` the request names — and it **sticks**: it is saved on the conversation, so switching
   models is one request rather than an update plus a turn;
2. the conversation's `model_id`;
3. the user's `default_model`;
4. otherwise `400 model_required`.

An id no catalog entry has is `404 model_not_found`; a disabled entry, or one whose engine the cluster
is not configured to run, is `409 model_unavailable`.

### What a turn sends

One system message — the user's instructions, the conversation's own prompt, their global memories,
this conversation's notes, then the rolling summary — followed by every message after the summary
watermark.

That history is then **rewritten for the model about to read it**: to a model without vision, image
parts become `[image omitted: <file>]`; to one without tools, each tool round is folded into text.
Without that, moving a conversation to a text-only model — or to Soma, which refuses a history
carrying `tool_calls` outright — would fail every later turn on history the model cannot represent.

**Tools** are offered only when the user has memories *and* tools enabled and the model and the engine
serving it can call them; and only offered tools are executed. A model that emits a call to a tool it
was never given does not get to run it.

### Ordering

Everything that touches one conversation's history — a turn, compaction, extraction, deletion, an
applied curation — runs on that conversation's own strand. Two turns of one conversation never
interleave; turns of different conversations run in parallel on a bounded pool
(`chat_worker_threads`, default 16). A conversation is resolved or created *before* its work is
queued, so two first turns cannot create two conversations.

### `GET /v1/users` — `read`

```jsonc
{ "object": "list",
  "data": [{ "id": "alice", "display_name": "Alice", "custom_instructions": "…",
             "default_model": "qwen3-30b-a3b-q4_g-q6_g-g128",
             "memories_enabled": true, "tools_enabled": false,
             "created_at_ms": 1755400000000, "updated_at_ms": 1755400000000,
             "active_conversation_id": "…" }] }   // null when they have none
```

Sorted by id. A token bound to a user sees that user and no other (§1).

### `POST /v1/users` — `operator`

Body: any profile field; `id` or `display_name` is required. `201` with the user, `400` for an invalid
id or an unknown field, `409 user_exists` for an id in use — including one whose deletion is still
waiting on a request that holds it.

A field that is not a profile field is **refused rather than ignored**: a misspelled `tools_enabeld`
that silently did nothing reads as "the server ignored my setting".

### `GET /v1/users/{id}` — `read`
### `PUT /v1/users/{id}` — `chat`

Patch semantics: only the fields present change. `id` may appear only if it equals the current one.
A `default_model` the catalog does not have is `400`.

### `DELETE /v1/users/{id}` — `operator`

Forgets the user now and removes their store when the last request holding it lets go; revokes every
token bound to them and reports the count (`tokens_revoked`). The id cannot be reused until the
directory is gone, so a new user can never open a store that is about to be removed underneath it.

### `GET /v1/users/{id}/conversations` — `read`
### `POST /v1/users/{id}/conversations` — `chat`

```jsonc
{ "title": "…", "model": "…", "system_prompt": "…", "settings": { "temperature": 0.3 },
  "parent_conv_id": "…", "set_active": true }
```

`201 {"conversation": {…}}`. Every field is optional: a conversation with no model defers to the
user's default. At most one conversation is active at a time — a partial unique index enforces it, not
a convention — and activating another queues a review of the one left behind (see extraction below).

### `GET /v1/users/{id}/conversations/{cid}` — `read`

The conversation and its whole stored history, including messages before the summary watermark: the
summary is what is *sent*, not what is *kept*. Each assistant message carries the `model_id` that
wrote it.

### `PUT /v1/users/{id}/conversations/{cid}` — `chat`

`title`, `model`, `system_prompt`, `settings`. `model: ""` clears it, so the user's default applies
again. Any other key is `400`.

### `POST /v1/users/{id}/conversations/{cid}/activate` — `chat`
### `DELETE /v1/users/{id}/conversations/{cid}` — `operator`

`409` for the active conversation — activate another first. The delete runs on the conversation's
strand, so a turn in flight finishes rather than writing into a conversation that no longer exists.

### `POST /v1/users/{id}/conversations/{cid}/compact` — `chat`

Folds the history into the conversation's rolling summary now, using the `model` named, else the
conversation's, else the user's default.

```jsonc
{ "conversation_id": "…", "status": "compacted",   // not_needed | compacted | failed | context_overflow
  "summary_through_seq": 7, "summary_model_id": "small-model" }
```

Compaction is **in place**: the conversation keeps its id, its notes and every stored message; only
what is sent shrinks, and the summary can be rebuilt from what is kept. A turn compacts on its own
when its prompt reaches 0.8 of the answering model's window, keeping the longest recent tail that
starts at a user turn and fits 0.4 of it; this route compacts regardless, keeping only the latest user
turn. Folding is chunked, each summarizer call sized to fit the model doing the folding, and the
watermark is saved after every chunk — so a failure part-way resumes where it stopped.

This replaces compaction by **forking**, which sent every old message to the summarizer in one prompt
— overflowing as soon as a conversation moved to a model with a smaller window — and then seeded a
NEW conversation with the result, leaving the id every client held pointing at a dead end.

### `GET /v1/users/{id}/conversations/{cid}/local-memories` — `read`
### `POST /v1/users/{id}/conversations/{cid}/local-memories` — `chat`
### `PUT /v1/users/{id}/conversations/{cid}/local-memories/{mid}` — `chat`
### `DELETE /v1/users/{id}/conversations/{cid}/local-memories/{mid}` — `operator`

Notes scoped to one conversation: `{"content": "…"}`. They are shown to the model in that
conversation only, and the memory tools write them when tools are offered.

### `GET /v1/users/{id}/memories` — `read`
### `POST /v1/users/{id}/memories` — `chat`
### `PUT /v1/users/{id}/memories/{mid}` — `chat`
### `DELETE /v1/users/{id}/memories/{mid}` — `operator`

The user's global memories: `{"content": "…", "importance": 0.5, "source_conv_id": "…"}`. They are
the user's, not a model's — every conversation of theirs can be shown them, whichever model answers.

### `POST /v1/users/{id}/memories/extract` — `chat`

```jsonc
{ "conversation_id": "…", "start_index": 0, "end_index": 12, "context_before": 2, "model": "…" }
```

Queues extraction on the conversation's strand and answers `{"status":"queued", "conversation_id",
"model_id", "selected_count"}` — the work runs a model, so the reply says what was queued rather than
holding the connection open for it. Indices are sequence numbers; `end_index` past the last message is
`400`.

A conversation that stops being active is reviewed the same way, automatically, with the model that
last answered it (else the conversation's, else the user's default). With a model that can call tools
the memory tools do the work; with one that cannot — Soma — the same review runs with a JSON reply
instead. Recall used to offer tools unconditionally, which is why it had never once worked on Soma.

### `POST /v1/users/{id}/curation/proposals` — `chat`
### `POST /v1/users/{id}/curation/apply` — `chat`

Proposals are computed, not stored; `apply` takes them back as `{"proposals": [...]}`. Every proposal
is validated before any is applied, so a batch with one bad entry changes nothing. (The
`/curation/proposals/apply` alias is gone; no client used it.)

### `POST /v1/users/{id}/chat` — `chat` — **SSE**

```jsonc
{ "message": "…", "conversation_id": "…", "model": "…",
  "attachment_ids": ["…"], "max_tokens": 0, "settings": { "temperature": 0.3 } }
```

Without a `conversation_id` the user's active conversation is used, or created. `message` may be empty
only when `attachment_ids` are supplied. Refused **before the stream opens**, where a status code can
still say so: no model to answer with (400/404/409, §2c), or images to a model or engine that cannot
see them (422 `images_unsupported`). Once the stream is open, every failure arrives as the `done`
event's `error` and `code` (§12).

### `POST /v1/users/{id}/attachments` — `chat`
### `GET /v1/users/{id}/attachments/{attachment_id}` — `read`
### `DELETE /v1/users/{id}/attachments/{attachment_id}` — `operator`

Upload streams to the user's `attachments/` directory with the filename in `X-Filename`; the body's
magic bytes must match the declared type. A `201` attachment is pending until a message references it,
after which it lives as long as that conversation does; `DELETE` of a referenced attachment is `409`.

### `GET /v1/users/{id}/voice` — `read`
### `GET /v1/users/{id}/voice/proposals` — `read`
### `POST /v1/users/{id}/voice/proposals` — `chat`
### `POST /v1/users/{id}/voice/proposals/{proposal_id}/sample` — `chat`
### `GET /v1/users/{id}/voice/proposals/{proposal_id}/sample` — `read`
### `POST /v1/users/{id}/voice/proposals/{proposal_id}/approve` — `operator`
### `POST /v1/users/{id}/voice/proposals/{proposal_id}/reject` — `operator`
### `POST /v1/users/{id}/speech` — `chat`
### `GET /v1/users/{id}/speech/cache/{cache_id}` — `read`
### `POST /v1/audio/speech` — `chat`

A voice belongs to the user, not to a model: `voice_description` is designed by a model (the one
named, else the user's default), sampled, and approved by an operator. `POST /v1/audio/speech` is the
OpenAI spelling of the same synthesis and takes `voice: "user:{id}"`.

---

## 3. Placement and backend selection

### `GET /v1/placements` — `read`

Extended with the decision and **its reason**. "Which backend" is far less useful than "which backend,
and why".

A **bare array**, not an object with a `placements` key:

```jsonc
[{
  "served_model_id": "qwen3-30b-a3b-q4_g-q6_g-g128", "node_id": "node-a", "slot_id": "slot-3",
  "suspended": false, "in_flight": 2,
  "kv_cache_node_path": "…", "engine_fingerprint": "…",
  "placed_at_ms": 1755400000000, "last_active_ms": 1755400090000,

  // added per row by the handler, from the same pure function the scheduler acted on
  "backend": "soma",
  "backend_reason": "soma (verdict=hybrid)"
}]
```

Placements are keyed by **served model**, and every user who picks one shares its placement.
`in_flight` is the number of requests using it right now — a count, because "busy" has to stay true
until the last of them finishes. A placement with `in_flight > 0` is never evicted, suspended or
released out from under those requests (release takes `?force=true` for an operator who means it).

`backend_reason` is a **composed sentence**, not an enum token: `BackendDecision::explain()` renders
`"<engine> (<reason>)"`, folding the verdict in — `soma (verdict=hybrid)`, or
`llama-cpp (override_refused_conformance, verdict=reject)` where the reason and the verdict are
different facts and both are worth having. The reason component is one of `verdict` |
`no_admission_record` | `stale_admission_record` | `operator_override` |
`override_refused_conformance` (`soma::BackendReason`, `include/soma/routing.hpp`).

> This block previously documented a `{"placements": […]}` wrapper, an `engine_id` field, and
> `backend_detail` / `footprint` / `footprint_source` / `model_id` fields — none of which this route
> emits — over an enum (`verdict_reject`, `resident_only`, `remote_api`) that came from
> `placement_engine.hpp`, a header that was never implemented and is now deleted. A published
> contract copied from an unbuilt design is worse than an absent one, because a client codes against
> it. Found by the D46 sweep and corrected here (roadmap D63).

### Routing policy

| Verdict | Engine | Why |
|---|---|---|
| `stream` | Soma | Streaming economics favourable |
| `hybrid` | Soma | Partial residency wins |
| `resident-only` | **fallback** | It fits; streaming buys nothing |
| `reject` | **fallback** | Failed conformance stage 1 or 2 |
| *no record* | **fallback** | Absence of a record is not evidence of admissibility |

### Placement failure codes

Any route that places a model — `POST /v1/served-models/{id}/restore`, and the chat paths that place
on demand — answers a failure with a machine-readable `code` beside the prose, and a `retryable` flag:

```jsonc
{ "error": "no eligible node: none is connected and conforming to the cluster engine configuration",
  "code": "no_eligible_node",
  "retryable": true,
  "served_model_id": "qwen3-30b-a3b-q4_g-q6_g-g128" }
```

| `code` | Means | `retryable` |
|---|---|---|
| `engine_config_missing` | The cluster has no engine policy yet | **false** — an operator must set one |
| `no_local_backend` | The model is API-backed and owns no node slot by design | **false** — retrying never helps |
| `no_eligible_node` | Nothing passed the connected + conforming filter | true — a node can rejoin or converge on its own |
| `no_capacity` | Eligible nodes are connected; none could take this model | true |
| `model_transfer_failed` | The model could not be placed on the target node | true |
| `node_rejected` | The node answered with an HTTP error | true |
| `node_unreachable` | The request to the node threw | true |
| `node_protocol_error` | The node answered OK with no slot id | true |
| `draining` | The model's launch settings changed while requests were still using its placement; it is replaced once they finish | true |

Chat waits out the *transient* ones — `no_capacity`, `no_eligible_node` and `draining` — a second
at a time for up to `MM_CHAT_QUEUE_WAIT_MS` (default three minutes) before reporting them; the rest
fail the turn at once.

`no_eligible_node` and `no_capacity` are the pair this exists for: they call for **opposite operator
actions** — fix the cluster, versus wait or add hardware — and before this both produced the sentence
"no capacity: no connected node could load this model", so telling them apart meant matching English
(roadmap D64). `retryable` is deliberately biased toward `true` where it is arguable: a wrong `true`
costs a client one wasted poll, a wrong `false` makes it abandon a placement that would have worked.

---

## 4. Placement lifecycle

Where a served model runs, and the three verbs that move it. `GET /v1/placements` reports the table;
these change it.

They exist because they were the clearest **P1 violation in the system**: the scheduler suspends
placements on its own under capacity pressure, the node API has always exposed
`/api/node/suspend-slot` and `/api/node/restore-slot`, and no `/v1/*` route could reach any of it.
An operator holding the entire control API could not do a thing the scheduler does routinely.
A design header argued for promoting exactly these — and was compiled by nothing, so the argument
never became a route. It has since been deleted (roadmap D46); these routes are what it wanted.

All three refuse a placement with requests in flight, because a placement is shared: every user who
picked that model is on it, and "busy" stays true until the last of them finishes.

### `POST /v1/served-models/{id}/suspend` — `operator`
Checkpoint the placement's KV and free its slot, remembering it so a restore does not reload from
scratch. `404` when no such model exists; **`409`** when it exists but holds no live placement —
different states, and a script needs to tell them apart — and `409 busy` while any request is using
it, including one on another model sharing its engine. `502` when the node refuses.

### `POST /v1/served-models/{id}/restore` — `operator`
Place it now. This *is* `ensure_running`: its first step is "existing/suspended placement", so a
suspended model restores from its checkpoint rather than reloading. A second code path would be a
second implementation of the placement ladder. Returns the `node_id` and `slot_id` it landed on;
`409` with the placement failure `code` (§3) and `retryable`, or `model_unavailable`.

### `POST /v1/served-models/{id}/release` — `operator`
Drop the placement entirely and detach it from its node. **Idempotent**: releasing a model that holds
nothing is a no-op, not an error — an operator clearing a stuck placement should not have to check
first. `409 busy` while requests are in flight, unless `?force=true`, which ends them.

---

## 5. Cluster engine configuration

What the cluster is configured to RUN, as opposed to what is running. Namespaced under
`/v1/cluster/` because `/v1/engines` is already taken and means a live engine *process* on some
node; these are about engine *kinds* and policy. Two resources one word apart is how a client
ends up reading slot state and believing it is policy.

The configuration states **intent** — which engines, at what version, acquired how, updated on
what policy — and never anything per-machine. `accelerator`, `cuda_arch`, `variant`, and any
executable path are **rejected** with a 400 naming the field: a Metal Mac and a CUDA box cannot
share those values, so a cluster config that carried them would leave every heterogeneous
cluster permanently non-conforming. Each node resolves them from hardware only it can see.

### `GET /v1/cluster/engines/config` — `read`
The current policy. Answers `200` with `{"configured": false, …}` before first-run setup rather
than `404`: "nobody has configured this yet" is a state of the cluster a client needs to read,
and it is what drives the setup surface.

### `PUT /v1/cluster/engines/config` — `operator`
Replace the policy. `version` is assigned server-side and the client's value ignored — a client
echoing a stale version back could otherwise make the cluster converge backwards. On success the
new config is pushed to every connected node whose reported version differs.

`operator` by blast radius: one request can start a source compile on every node at once.

| Field | Notes |
|---|---|
| `primary_engine` | Required. Must have a matching entry in `engines`. |
| `backup_engine` | `""` means **no backup**, which is a real configuration and not an unset one. Defaults to `llama-cpp` at setup. |
| `engines[]` | `engine_id`, `version`, `install_method`, `update_policy` (`prompt\|auto\|manual`), `update_check`, `update_check_interval_hours`, `cmake_args`, `build_jobs`. Soma/llama.cpp accept `auto\|release\|source\|path`; vLLM accepts `auto\|wheel\|source\|path`. |
| `engines[].vllm` | Present only for vLLM: model/sequence/batch-token limits, TP/PP, GPU-memory utilization, dtype, quantization, trust/prefix/tool/sleep controls, tool parser, additional arguments, and automatic Ray policy. |
| `share_builds` | Whether a node that built an engine may serve it to a node that needs the same one. |

### `GET /v1/cluster/engines/conformance` — `read`
Per node: its conformance state, the config version it last applied, every engine it can
provision, and whether placement may target it. `placement_eligible` is stated per node because
"why is nothing scheduling here" is the question this route exists to answer.

States are `unconfigured`, `converging`, `conforming`, `drifted`, `failed`. Only `conforming`
permits placement; `drifted` and `failed` carry a non-empty `detail`.

### `GET /v1/cluster/engines/ray` — `read`

Returns whether automatic Ray topology is required by the configured vLLM
profile, the eligible-node count, active group summaries, TP/PP, transport,
and diagnostics. The endpoint remains readable when vLLM is not selected and
then reports `state: "hidden"`.

### `POST /v1/cluster/engines/resync` — `operator`
Re-push the configuration to every node whose version differs, without waiting for the next
health poll. `409` when no configuration exists.

### `POST /v1/cluster/engines/share` — `operator`
Broker one engine artifact from a node that has it to a node that needs it. The bytes go
node-to-node; control never stores them. `source_node_id` is optional — omitted, control picks a
connected node advertising the exact fingerprint.

Five steps, in this order:

1. the **source** packages and hashes the artifact, sending nothing
2. control relays that digest to the **target** over its own authenticated channel
3. control mints a scoped transfer credential into the target
4. the source pushes the package it already hashed
5. control revokes the credential — on success *and* on failure

Steps 1–3 preceding 4 is the point. **What it guarantees:** the digest the target will accept is
fixed by an authenticated peer before the transfer credential exists, so a credential that leaks
cannot be used to push different bytes. **What it does not:** a compromised source supplies both
the artifact and the hash that validates it, and no ordering fixes that — signed builds would,
and this is not one.

Fingerprint equality is exact across `(engine_id, version, platform, arch, variant)`; a
near-match is a wrong binary, not a close one. The receiving node re-checks the artifact against
its own platform and arch, because control's view of a node is a report from that node.

### `POST /v1/cluster/engines/nodes/{node_id}/provision` — `operator`
### `POST /v1/cluster/engines/nodes/{node_id}/check-update` — `operator`
### `POST /v1/cluster/engines/nodes/{node_id}/switch` — `operator`
### `POST /v1/cluster/engines/nodes/{node_id}/diagnose` — `operator`
### `POST /v1/cluster/engines/nodes/{node_id}/recover` — `operator`

Run one engine action on one node. Control brokers to `/api/node/engines/{engine_id}/{action}`
and passes the node's status and body through unchanged.

**Why these exist, stated as the gap they close.** The node has had these actions since the
engine work landed and nothing on control ever called them: the wizard driving them lived only
in the node's own TUI, wired in-process. Recovering a node meant opening a session on that
machine — on a cluster head whose premise is that you do not have to.

**And why `resync` was not already enough.** `POST /v1/cluster/engines/resync` re-pushes only to
nodes whose config *version* differs, and a node bumps its version when it **accepts** a
configuration, not when it **conforms** to one. A node that accepted v3 and then failed its
build sits at v3 with state `failed`: resync skips it and answers `{"resynced": true}`. These
routes are the lever that actually moves that node.

Addressed by node, not by engine — a node with no working engine has no slot to name, and
provisioning is precisely what a node does when it has none.

| field | where | meaning |
|---|---|---|
| `engine_id` | body, default `llama-cpp` | every action is llama.cpp-only today; the node refuses others by name with a `400` |
| `variant` | body, **required** for `switch` | execution variant, e.g. `cuda-12`, `vulkan`, `cpu` |
| `action` | body, **required** for `recover` | `retry` \| `target` \| `compile-anyway` \| `release` |
| `variant` | body, required when `action: "release"` | a report variant id |
| `update`, `accelerator` | body, `provision` only | `update: true` runs an update instead; `accelerator` is valid only with it |

**`202`, not `200`.** The node starts a worker and returns; it does not run the action on the
request. A source build is minutes, and a synchronous handler would hold a connection across it
on both ends while control's health poll marked the node unreachable in the middle. The body
carries `started`, the action, and `llama_runtime` **as it was when the worker started** — a
starting point, not an outcome. Watch progress through `action_progress` on
`GET /v1/cluster/engines/conformance`, which the Engines tab renders as its Activity panel.

`409` when another llama.cpp operation is already running on that node, or when the node is not
connected. All six actions serialise on one worker inside the node — including its scheduled
auto-update — so an operator pressing Provision during a build is refused rather than queued
behind it. `404` for an unknown node, `502` when the node does not answer.

---

## 6. Runtime telemetry

### `GET /v1/engines` — `read`
Live engines cluster-wide, with a tier-occupancy summary per engine. **Engine processes**, not
engine kinds — see §5 for the configuration surface.

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

## 7. Concurrency and slots

### `GET /v1/engines/{engine_id}/slots` — `read`

```jsonc
{ "slot_id": "slot-3", "engine_id": "soma",
  "queue_depth": 2, "current_batch": 7,
  "effective_max_batch": 12,
  "max_batch_limited_by": "expert_cache",   // "expert_cache" | "kv_slots" | "config"
  "sequences": [
    { "index": 0, "conversation": "", "position": 1204, "kv_tokens": 1204,
      "prefilling": false, "suspended": false, "determinism": "batched" } ] }
```

`effective_max_batch` and `max_batch_limited_by` are reported so an operator can see **why**
concurrency is limited rather than inferring it from throughput. When the limiter is `expert_cache`, the
gate is `cap_per_layer / expected_unique_experts_per_step` — raising the RAM budget raises concurrency;
raising a config number does not.

Nothing like `sequences` exists today: the node keeps one `active_requests` counter per slot and
llama-server assigns its own internal slots invisibly. A stalled sequence and a saturated batch look
identical from a request counter.

### `POST /v1/engines/{id}/slots/{n}/suspend` | `/restore` — `operator` (planned)

**The first operator-visible suspend/restore.** Both mechanisms exist today but only as internal
scheduler decisions reachable through the node API; there is no `/v1/*` route for either. Promoted
because P1 says a capability the system has is a capability the API exposes.

`suspend` returns 409 with `{"error":{"code":"multi_sequence_unsupported"}}` when the engine has live
sequences beyond index 0 and its KV backend cannot checkpoint them. The fallback is in exactly that
position — `POST /slots/0?action=save` hardcodes sequence 0, so a `--parallel > 1` slot currently saves
its first sequence and **silently discards the rest**. The rebuild makes that an explicit refusal.

---

## 8. Determinism

Quantized integer kernels are shape-dependent: batched and single-row forwards round differently, and at
int4 that can flip argmax ties. **A greedy request's exact token stream can therefore depend on who else
is on the server.** Every emitted token remains the argmax of a valid forward, so quality holds;
reproducibility across batch sizes does not. There is no free middle ground.

Both halves of the tradeoff are exposed, because they answer different questions:

**Per-model default** — `PUT /v1/served-models/{id}`:
```jsonc
{ "runtime_settings": { "determinism": "batched" | "strict" } }
```

**Per-request override** — `POST /v1/users/{id}/chat`:
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

## 9. Errors

Uniform envelope. Machine-readable `code` first, prose second.

```jsonc
{ "error": { "code": "capacity_pressure", "message": "…", "detail": { … } } }
```

| Code | Status | Meaning |
|---|---|---|
| `missing_bearer_token` | 401 | |
| `invalid_bearer_token` | 403 | |
| `insufficient_scope` | 403 | Includes `required` and `granted` |
| `validation_failed` | 422 | Served-model validation, with `issues[]` |
| `bad_request` | 400 | Invalid model-protocol option, including unsupported V4 reasoning/tool forcing |
| `unsupported_content` | 422 | Image content part; **text-only v1** |
| `protocol_error` | 502 | The model emitted malformed DSML; no tool call is fabricated |
| `model_not_admitted` | 409 | Soma requested, no record |
| `capacity_pressure` | 503 | Retryable; drives evict-and-retry |
| `multi_sequence_unsupported` | 409 | Suspend on a multi-seq slot the backend can't checkpoint |
| `engine_unavailable` | 503 | Runtime not provisioned on any node |

**`capacity_pressure` as a code replaces substring matching.** `PlacementScheduler::response_indicates_capacity_pressure`
currently searches the node's error body for six English phrases (`"max slots reached"`,
`"no available ports"`, `"out of memory"`, …). A new engine would have had to reproduce those literals
verbatim to earn an evict-and-retry. Both engines emit codes now.

**Image parts are rejected with 422, never dropped silently.** Silent dropping is the failure mode worth
designing out — a multimodal request that returns a plausible text-only answer is worse than one that
fails.

---

## 10. Scope map

Every route, by scope. `route_scope.cpp` is the authority; this is a reader's index of it.

| Scope | Routes |
|---|---|
| `read` | `GET /v1/nodes`, `/v1/nodes/discovered`, `/v1/activity`, `/v1/performance`, `/v1/placements`, `/v1/engines`(+`/:id/slots`, `/heat`, `/telemetry`), `/v1/models`(+`/:id` and its plan, conformance and heat), `/v1/served-models`(+`/:id`), `/v1/users`, `/v1/users/:id`, `…/conversations`(+`/:cid`, `/local-memories`), `…/memories`, `…/voice`(+`/proposals`, `/proposals/:pid/sample`), `…/attachments/:aid`, `…/speech/cache/:cid`, `/v1/cluster/engines/*` reads |
| `chat` | `POST /v1/users/:id/chat`, `PUT /v1/users/:id`, `POST/PUT` on conversations, local-memories and memories, `POST …/memories/extract`, `…/curation/proposals`, `…/curation/apply`, `…/:cid/compact`, `…/:cid/activate`, `POST …/attachments`, `POST …/speech`, `POST /v1/audio/speech`, voice proposal create/sample |
| `operator` | `POST /v1/users`, `DELETE /v1/users/:id`, **every `DELETE`** on conversations, local-memories, memories and attachments, voice `approve`/`reject`, every served-model write (§2b) and placement verb (§4), every admission write (§2), `POST /v1/nodes`, `DELETE /v1/nodes/:id`, `POST /v1/nodes/:id/forget`, `/v1/nodes/pair/psk`, `DELETE /v1/performance`, `GET/POST/DELETE /v1/tokens` |

Four judgement calls worth naming:

- **Every `DELETE` is `operator`, even per-user and small.** Losing a conversation is not recoverable
  from the API, and "bounded" is about what a mistake costs, not how many rows it touches. (This table
  listed them under `chat` until it was checked against `route_scope.cpp`, which has always said
  `operator`; `check_api_docs.py` parses route headings, not this table, so nothing caught it.)

- **`PUT /v1/users/:id` is `chat`, `POST /v1/users` is `operator`.** Editing your own profile is
  bounded by the user you already reach; creating one is not bounded at all — and a bound token is
  refused it outright (§1).
- **`DELETE /v1/performance` is `operator`, not `chat`.** It destroys shared observability state that
  another client may be mid-analysis on.
- **Voice `approve`/`reject` is `operator`, not `chat`.** Approving a proposal mutates the user's
  persistent identity; creating and sampling one does not.

`POST /api/control/register-node` keeps its separate registered-node auth and is deliberately outside
the `/v1` scope gate, as today.

---

## 11. OpenAI-compatibility listener (`:9091`)

**Stateless.** A client that owns its conversation — history, tools, sampling — sends an ordinary chat
completion naming a served model (§2b), and the request reaches the engine placed for that model
**unchanged**: no history is added, no memories, no system prompt, and nothing is stored. The only
rewrite is the model name — in, from the served-model id to what the engine or remote provider calls it;
out, back to the served-model id.

| Route | Scope | |
|---|---|---|
| `GET /v1/models` | `read` | The **available** served models, as OpenAI model objects (`metadata` carries `source`, `vision`, `tool_calling`, `context_window`) |
| `GET /v1/models/:id` | `read` | One of them; 404 `model_not_found` |
| `POST /v1/chat/completions` | `chat` | Passed through; `stream` relays every frame, tool calls and reasoning included |

- **Authorization** is the same `ScopeAuthorizer` as `:9090`, answered in OpenAI's error envelope. A
  scoped token means the same thing on both ports. (This listener used to honour only the legacy flat
  token and was wide open whenever that was unset — even with scoped tokens issued.)
- **Refused before anything is sent:** `n > 1` (400); images to a model or engine that cannot see them
  (422); `tools`, or a history carrying `tool_calls`, to a model or engine that cannot render them (422 —
  Soma cannot); an unknown model (404 `model_not_found`); an unavailable one (409 `model_unavailable`).
- **Placement** is the same as chat's: a node-served model is placed on demand, waiting out *transient*
  failures for up to `MM_CHAT_QUEUE_WAIT_MS`; a failure answers 503 (`retryable`) or 409 with the
  placement `code` (§3) as the error `code`.
- **The engine's answer is the client's**, status included: an engine or provider that refuses a request
  outright is relayed with its own status and body rather than wrapped in a 200.
- Node-served models go through the node's `POST /api/node/engines/{slot}/chat/completions` passthrough;
  API-backed models go straight to their `base_url` + `chat_completions_path` with their key.

This listener used to list **agents** as models (`agent:{id}`) and flatten the client's whole message
array into one user turn in that agent's own conversation, with the agent's history and memories
prepended — and its stream dropped `thinking` and `tool_call` deltas. None of that survives.

---

## 12. SSE semantics

Chat (`POST /v1/users/:id/chat`):

```
data: {"type":"thinking","content":"…"}
data: {"type":"delta","content":"…"}
data: {"type":"tool_call","id":"…","name":"…","arguments":"…"}
data: {"type":"tool_result","tool_call_id":"…","function_name":"…","result":"…"}
data: {"type":"done","conv_id":"…","success":true,"model_id":"…"}
data: [DONE]
```

The `done` event carries **`model_id`** — which model actually answered, after the request, the
conversation and the user's default were resolved — and, when `success` is false, `error` and a
machine-readable `code` (`model_required`, `model_not_found`, `model_unavailable`,
`context_overflow`, `images_unsupported`, `inference_failed`, a placement `code` from §3, or
`shutting_down`). A client cannot otherwise tell "the model refused" from "no model was chosen".

Two defects in the current node-side mapping are fixed rather than carried forward
([node_api_server.cpp:1223](../src/node/node_api_server.cpp:1223)):

1. The chunk→event mapping is an `if / else-if` **priority chain**, so a chunk carrying both
   `thinking_delta` and `delta_content` silently drops one. It becomes independent emission.
2. `tool_result_json` is never emitted by `/api/node/infer` at all, despite being carried on
   `InferenceChunk`.

Exactly one `done` is always delivered, on every path including error — that guarantee is preserved.

**Telemetry streams are separate from chat streams** and are rate-limited independently (§6). A client
subscribing to both gets chat at production rate and telemetry at `hz`.

---

## See also

- [architecture.md](architecture.md) — the seam, the scheduler, determinism
- [mantic-mind-integration.md](mantic-mind-integration.md) — engine↔node boundary, FTXUI consumption
- [../schemas/arch-ir.md](../schemas/arch-ir.md) — `arch.json` and the verdict function
