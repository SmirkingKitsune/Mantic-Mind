# Architecture IR — canonical registry JSON

**Schema versions:** 1 (existing families), 2 (DeepSeek V4 compressed/sparse)
**Produced by:** admission, by adapting the source `config.json`
**Consumed by:** the Soma loader, the planner, `soma plan --json`, the registry

The architecture IR is the complete, normalized description of a model. Admission persists its JSON
form in the registry's `arch_json` column; a converted container does not carry a separate
`arch.json` file. At load, the same IR is adapted from the container's copied `config.json`, then the
conversion's quantization record is applied. It is hashed into `arch_hash`, and everything the engine
needs to decide *how to execute* a model is in the IR or the sidecar index; nothing is inferred from
tensor names, filenames, or heuristics.

Two rules govern the whole document:

1. **Every on-disk artifact carries `schema_version` and `arch_hash` and refuses to load across a
   mismatch.** A stale KV checkpoint replayed against the wrong architecture is otherwise a very
   confusing bug report.
2. **Quantization is specified per *tensor role*, never per tensor and never globally.** See §5.

---

## 1. Document shape

```jsonc
{
  "schema_version": 1,
  "arch_hash": "…",              // computed over §2–§6 canonicalized, excluding this field
  "source": {
    "repo": "Qwen/Qwen3-30B-A3B",
    "revision": "…",             // commit sha of the source checkpoint
    "model_type": "qwen3_moe",   // upstream config.json model_type, informational
    "adapter": "qwen3_moe"       // which per-family name adapter was used
  },
  "topology":      { … },        // §2
  "attention":     { … },        // §3
  "router":        { … },        // §4
  "ffn":           { … },        // §4
  "quantization":  { … },        // §5
  "tokenizer":     { … },        // §6
  "economics":     { … }         // §7 — measured, feeds the verdict
}
```

`arch_hash` covers §2–§6 (what the model *is*) but **not** §7 (what we measured about it). Re-profiling
a model on faster disks must not invalidate its KV checkpoints. Requantizing it must, and does, because
§5 is inside the hash.

Schema v2 is intentionally narrow. DeepSeek V4 is adapted to v2; every existing GQA, MLA, and
MLA+DSA model remains v1 and keeps its container and KV checkpoint formats. A v2 document may select
the “compressed+sparse” attention family and the V4 routing/hyper-connection fields below. No core
caller infers V4 from model_type: resolved backend descriptors carry the model/layer payload binders,
block hooks, opaque KV implementation, exact sizing functions, and optional prompt codec.

---

## 2. `topology`

| Field | Type | Notes |
|---|---|---|
| `n_layers` | int | |
| `d_model` | int | `hidden_size` |
| `vocab_size` | int | |
| `layer_kinds` | `[string]`, length `n_layers` | `"dense"` \| `"moe"` — explicit per layer, never derived from a stride at runtime |
| `first_k_dense` | int | Informational; `layer_kinds` is authoritative |
| `draft_layer` | int \| null | MTP / draft head layer index |
| `tie_word_embeddings` | bool | |
| `max_position_embeddings` | int | Hard request/admission ceiling; V4 is 1,048,576 |
| `eos_token_ids` | `[int]` | Generation stops before emitting one of these tokens |

`layer_kinds` is a materialized array rather than the upstream `decoder_sparse_step` /
`moe_layer_freq` / `mlp_only_layers` triangle. Three different families express "which layers are MoE"
three different ways; resolving that at admission means the core never has to.

---

## 3. `attention`

| Field | Type | Notes |
|---|---|---|
| `family` | enum | `mha` \| `gqa` \| `mla` \| `mla+dsa` \| `compressed+sparse` — **selects the backend** |
| `n_heads` | int | |
| `n_kv_heads` | int | `gqa`/`mha` |
| `head_dim` | int | |
| `qk_norm` | enum | `none` \| `per_head` \| `full_width` — NOT a bool; see below |
| `sliding_window` | int \| null | Span in tokens; null = full |
| `bias` | bool | |
| `rope` | object | `{ theta, partial_dim, interleaved, scaling }` |
| `rope.scaling` | object \| null | `{ type: "yarn"\|"linear"\|"ntk", factor, original_max_position, beta_fast, beta_slow, mscale, mscale_all_dim }` |
| `mla` | object \| null | Present iff `family` starts with `mla` |
| `dsa` | object \| null | Present iff `family` is `mla+dsa` |
| `compressed` | object \| null | Schema v2; present iff `family` is `compressed+sparse` |

### `compressed` sub-object — DeepSeek V4

The V4 backend combines a 128-token live BF16 window with per-layer BF16 compressed history. Ratio-4
layers additionally retain BF16 indexer history; ratio-128 layers attend the complete compressed
history. Compressor value/score carry remains FP32 because the official model performs compression
in FP32; it is backend-owned opaque working state rather than an attention-key storage plane.
Checkpoints contain only the live window, completed histories, and the carry needed to resume an
incomplete compression group under the `compressed-sparse-bf16-v1` persistence format.

| Field | Notes |
|---|---|
| `compress_ratios` | Per-layer array; the pinned base stack alternates 128 and 4 |
| `compress_rope_theta` | RoPE base used by compressed attention |
| `q_lora_rank` | Q down-projection width |
| `rope_head_dim` | RoPE-carrying suffix of each 512-wide attention head |
| `o_groups`, `o_lora_rank` | Grouped low-rank output projection |
| `index_n_heads`, `index_head_dim`, `index_topk` | Ratio-4 sparse indexer geometry |
| `semantic_fp8_quant_dequant` | FP8 Q/DQ simulation is model semantics, not a storage dtype |
| `semantic_fp4_quant_dequant` | FP4 indexer Q/DQ simulation is model semantics |

Both low-precision operations are implemented as software quantize/dequantize. The runtime does not
require native FP8 or FP4 kernels.

`rms_norm_eps` is a TOP-LEVEL field, not an attention one, and it does not apply to every norm. The
layer norms and the output norm take it; MLA's two LATENT norms (`q_a_layernorm`, `kv_a_layernorm`)
do NOT — both reference implementations construct those with the RMSNorm class default of `1e-6`,
whatever the config says. DeepSeek-V2-Lite hides the difference by setting `1e-6` itself; Moonlight
and GLM-5.2 set `1e-5`, and using it cost Moonlight a conformance error of 7.25e-05, seventy times
every other fixture, passing and unexplained until GLM-5.2 made it fail outright (roadmap D29).

**`qk_norm` is an enum because the two forms normalize over different things.** `per_head` applies
over `head_dim` independently per head (Qwen3-MoE: q_norm is [16] with head_dim 16); `full_width`
applies over `n_heads * head_dim` (OLMoE: q_norm is [64] with 4 heads x 16). Both report
`"qk_norm": true` upstream. Reading it as one bit produces a model that runs, converges to plausible
logits, and is wrong.

`mla` sub-object:

| Field | Notes |
|---|---|
| `kv_lora_rank` | Compressed latent width |
| `q_lora_rank` \| null | null = no Q down-projection (V2-Lite) |
| `qk_nope_head_dim` | Non-RoPE part of the QK head |
| `qk_rope_head_dim` | RoPE-carrying part |
| `v_head_dim` | |
| `absorb_weights` | bool — move the KV up-projection to the query side during decode |

**`absorb_weights` has no GQA analogue**, which is exactly why it lives in the `mla` sub-object.

It applies **per step, not at load**, which is a correction: this once said "load-time" while nothing
implemented absorption at all (roadmap D38, D39). Folding at load means keeping a transposed fp32 copy
of the up-projection resident — 1.96 GB on GLM-5.2 — to save arithmetic that was never the bottleneck.
The unused load hook and its parallel execution API were deleted; absorption lives only in the cached
MLA decode that actually serves.

The identity is:

    q_nope . (W_k c_j)   =  (W_k^T q_nope) . c_j
    sum_j a_j (W_v c_j)  =  W_v (sum_j a_j c_j)

Both move `kv_b` off the side that depends on `j`, so it is touched once per head per step however
many keys are attended. `false` selects the expanded form, kept as the reference the absorbed one is
checked against.

### `dsa` sub-object — DeepSeek Sparse Attention + IndexShare

Present iff `family` is `mla+dsa`. Part of `arch_hash`, so two quantizations or two indexer
configurations of the same weights are two models.

| Field | Notes |
|---|---|
| `index_topk` | Keys that survive selection. **The number that decides whether a test means anything**: with fewer tokens in context than this, top-k selects everything and the sparse path is bit-identical to dense |
| `n_index_heads` | Indexer heads. NOT a size knob — see below |
| `index_head_dim` | Indexer head width |
| `index_freq` | How often a `full` layer recurs. Informational; `layer_kinds` is authoritative |
| `layer_kinds` | Per layer: `full` (computes an index) \| `shared` (reuses the nearest preceding `full` layer's) |

**IndexShare is why `layer_kinds` cannot be a stride plus an offset.** On GLM-5.2, 57 of 78 layers
own no indexer weights at all and cannot compute attention without state produced by a different
layer. A re-derived stride is a second description waiting to disagree with the weights.

**`n_index_heads` is semantic, not dimensional.** An index score is `sum_h w[h] * relu(q[h].k)`, so
it is exactly 0.0 only when ReLU zeroes every head at once — probability ~2^-H. Shrinking it for a
test fixture manufactures ties at the top-k cut, which `torch.topk` then resolves by internals that
are neither ascending nor descending index order. Measured: 50.69% of scores exactly zero at 1 head,
27.12% at 2, 13.52% at 3, 6.99% at 4, extrapolating to ~2e-8% at GLM-5.2's real 32 (roadmap D32).

### KV cost — the number the planner actually cares about

`kv_bytes_per_token()` and `kv_geometry()` are the attention properties that cross the seam. Worked
from the real configs:

| Model | Family | Elements/token/layer | Layers | Bytes/token @fp16 | @32k ctx |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | gqa | `2 × 4 × 128` = **1024** | 48 | 98 KB | **3.2 GB** |
| DeepSeek-V2-Lite | mla | `512 + 64` = **576** | 27 | 31 KB | **1.0 GB** |
| Mixtral-8x7B | gqa | `2 × 8 × 128` = **2048** | 32 | 131 KB | **4.3 GB** |
| GLM-5.2 | mla+dsa | `(512 + 64) + 128` = **704** | 78 | 107 KB | **3.5 GB** |

**The cache has TWO PLANES and they are not the same size.** `kv_geometry()` reports both, because a
single width could not express "this family stores no second plane" — and for want of that, MLA
allocated a full second plane at the K plane's width for every layer, holding nothing. GQA stores
per-head K and V, so both planes are `n_kv_heads * head_dim`. MLA stores a compressed latent and
DERIVES V from it, so its V plane is **zero**. DSA is the exception: its indexer key must be cached,
because it depends on a past token's hidden state at that layer and cannot be recomputed at a later
step, so the otherwise-dead plane is exactly where it goes (roadmap D35, D37).

On GLM-5.2 at 4k context with 4 slots that correction is 5.89 GB down to 3.60 GB, and the reclaimed
2.29 GB goes to the expert cache. DeepSeek-V2-Lite and Moonlight lose their second plane outright.

MLA's compression against the same model's uncompressed form is ~8.9× on V2-Lite
(`576` vs `16 heads × (128+64) + 16 × 128 = 5120`), and considerably larger on full-size V2/V3 where the
head count is 128.

**This is why GQA's KV is a planner input and not a footnote.** At 32k context Qwen3's KV cache costs
3.2 GB of the same RAM the expert cache wants. The planner models that competition explicitly; a design
that sized the expert cache first and let KV take what's left would thrash on long contexts and look
like an unrelated bug.

---

## 4. `router` and `ffn`

### `router`

| Field | Type | Notes |
|---|---|---|
| `n_experts` | int | Routed experts only |
| `top_k` | int | |
| `score_fn` | enum | `softmax` \| `sigmoid` \| `sqrtsoftplus` (computes `sqrt(softplus(x))`) |
| `normalize_topk` | bool | `norm_topk_prob` |
| `routed_scaling_factor` | float | |
| `bias_correction` | bool | Per-expert bias added before top-k (V3-style) |
| `n_groups` | int | Group-limited routing; 1 = ungrouped |
| `topk_group` | int | |
| `n_shared_experts` | int | 0 = none |
| `n_hash_layers` | int | V4: first three layers use token-id hash routing |

### `ffn`

| Field | Type | Notes |
|---|---|---|
| `activation` | enum | `swiglu` \| `geglu` \| `relu2` |
| `has_gate` | bool | |
| `expert_intermediate` | int | `moe_intermediate_size` |
| `dense_intermediate` | int | For `layer_kinds == "dense"` layers |
| `shared_intermediate` | int | |
| `swiglu_limit` | float | V4 clamps gate above and up symmetrically before SwiGLU |
| `expert_layout` | enum | `interleaved_gud` — gate/up/down interleaved per expert (§5.3 of architecture.md) |

Schema v2 also carries `hyper_connections = { multiplier, sinkhorn_iters, eps }`. V4 uses four
streams, with learned block pre/post controls and a Sinkhorn-normalized stream mixing matrix at both
attention and FFN boundaries.

---

## 5. `quantization` — per tensor role

```jsonc
"quantization": {
  "embed":        { "dtype": "q8_0" },
  "attn_proj":    { "dtype": "q4_g", "group": 128 },
  "expert_gate":  { "dtype": "q4_g", "group": 128 },
  "expert_up":    { "dtype": "q4_g", "group": 128 },
  "expert_down":  { "dtype": "q6_g", "group": 128 },
  "shared_expert":{ "dtype": "q4_g", "group": 128 },
  "router":       { "dtype": "f32" },     // ← enforced, see below
  "draft_head":   { "dtype": "q8_0" },
  "norms":        { "dtype": "f32" }
}
```

**`router.dtype` MUST be `f32`.** Validation rejects anything else at admission — it is a schema
constraint, not a convention and not a default.

The reasoning is worth stating because it is easy to mistake for excessive caution: quantizing router
logits does not degrade output precision, it changes **which experts fire**. That is a semantic change.
A model whose router is quantized is a different model, and it will fail conformance stage 2 in a way
that looks like a kernel bug for as long as it takes someone to check.

`norms` stays f32 for the ordinary reason (numerical range), and that one *is* just caution.

---

## 6. `tokenizer`

Admission compiles `tokenizer.json` into a normalized sidecar; this section records what was compiled
and what it must round-trip against.

| Field | Notes |
|---|---|
| `kind` | `bpe` \| `unigram` |
| `compiled_path` | Relative to the model dir |
| `byte_fallback` | bool |
| `n_special_tokens` | |
| `pretokenizer` | `compiled_nfa` — a byte-class NFA, not a live regex engine |
| `chat_template` | `token_struct` — resolved to token IDs at admission |
| `roundtrip_sha` | SHA of the corpus round-trip result vs HF `tokenizers` |

**Admission is gated on the round-trip test.** A tokenizer that does not reproduce HF `tokenizers`
byte-for-byte over the calibration corpus fails admission outright — it is the cheapest possible bug to
catch here and one of the most expensive to catch at G2, where it presents as "the model is subtly
stupid."

---

## 7. `economics` — measured, not assumed

Everything here is re-measured per model at admission. **None of it is inherited as a constant from
prior art.**

```jsonc
"economics": {
  "expert_bytes":        2359296,     // one expert, quantized, incl. scales
  "n_moe_layers":        48,
  "bytes_per_token":     906M,        // n_moe_layers × top_k × expert_bytes
  "total_routed_bytes":  14.5G,       // n_moe_layers × n_experts × expert_bytes
  "dense_resident_bytes": …,
  "active_fraction":     0.0625,      // top_k / n_experts
  "measured_disk_bw":    …,           // at THIS model's expert size, not a spec sheet number
  "measured_at_host":    "…"
}
```

`measured_disk_bw` is measured with reads the size of *this model's* experts. A 2.4 MB read and an
88 MB read do not achieve the same bandwidth on the same drive, and using a single headline number is
how you get a verdict that is confidently wrong.

---

## 8. The verdict function

The verdict decides Soma vs. fallback. It is computed from §7 plus a host budget:

```
resident_ok  =  total_routed_bytes + dense_resident_bytes + kv_bytes_at_ctx  ≤  ram_budget

verdict:
  resident_ok                        → resident-only     // streaming has nothing to do
  active_fraction > 0.15             → resident-only if it fits, else reject
  projected_tok_s ≥ floor            → stream | hybrid
  otherwise                          → reject
```

Two discriminators, and a model must pass **both** to be worth streaming: a low **active fraction**
(little of each layer fires) and **small experts** (each miss is cheap). Coarse-grained MoE fails both.

### Worked verdicts, from the real configs

| | Qwen3-30B-A3B | DeepSeek-V2-Lite | Mixtral-8x7B |
|---|---|---|---|
| MoE layers | 48 | 26 (`first_k_dense=1`) | 32 |
| Experts × top-k | 128 × 8 | 64 × 6 | 8 × 2 |
| **Active fraction** | **6.3 %** | **9.4 %** | **25 %** |
| Expert params | 3 × 2048 × 768 = 4.72 M | 3 × 2048 × 1408 = 8.65 M | 3 × 4096 × 14336 = 176 M |
| **Expert bytes @q4** | **2.36 MB** | **4.33 MB** | **88.1 MB** |
| `bytes_per_token` @q4 | 906 MB | 675 MB | **5.6 GB** |
| `total_routed_bytes` @q4 | 14.5 GB | 7.2 GB | 22.6 GB |
| **Verdict @q4, 32 GB host** | `resident-only`¹ | `resident-only`¹ | **`resident-only`** |
| **Verdict @bf16** | `stream` (58 GB routed) | `hybrid` (28.8 GB routed) | `reject`² |

¹ At q4 on a large-RAM host the routed set simply fits — streaming has nothing to do. This is the
correct answer, and it is the single most important thing the table shows.

² Mixtral at bf16 needs 90 GB routed at 25 % active fraction: it neither fits nor streams. Fallback,
with a smaller quantization.

### The consequence: **the verdict is not a property of the model**

It is a property of `(model, quantization, host budget)`, and **the throughput floor is part of the
host budget** — `HostBudget::min_tok_s`, beside `ram_total_bytes` and `disk_bandwidth`, settable with
`soma plan --min-tok-s`. It was a constant of `1.0` in plan.cpp, which refused GLM-5.2 at every host
size: 0.087 tok/s on a 24 GiB workstation, 0.79 at 128 GiB with a 7 GB/s disk. Colibri served those
same 744B weights on 16–24 GB and the result was considered useful, so the constant and the proof
disagreed. The reasoning it was written with still holds — a model streaming at 0.2 tok/s is not
usefully served — but "usefully" depends on who is asking, and for a 744B model on a workstation 0.1
tok/s may be the entire point (roadmap D21).

`0` means UNSTATED and resolves to `1.0`, not to "no floor", so a default-constructed budget guards
exactly as before and lowering the bar takes a deliberate statement. The refusal names the figure and
whether it was chosen or inherited, because "raise your tolerance" and "this host is too small" are
different answers.

Mixtral is `resident-only` because of its
*shape* — 25 % active fraction is disqualifying at any size — but Qwen3 flips between `resident-only`
and `stream` purely on quantization and host RAM.

So:

- The registry stores the **measurements** (§7) and an admission-host `verdict`, plus `verdict_basis`
  recording the host assumptions that produced it.
- **`soma plan --json` re-derives the verdict against the actual target node's budget.**
- `GET /v1/placements` reports the *effective* verdict and its reason, not the stored one.

A stored scalar verdict alone would mis-route the same model between two nodes with different RAM,
silently.

### G4's test vehicle will likely be `resident-only`, and that is fine

DeepSeek-V2-Lite at q4 fits in RAM on any reasonable host. G4's job is proving the **seam** carries a
second attention family — architectural correctness, not streaming economics. The conformance ladder
runs regardless of verdict; production *placement* is what the verdict gates. Forcing Soma for the test
is exactly what `backend_override: soma` exists for.

---

## 9. Validation

Admission rejects a document that fails any of:

- `schema_version` unknown
- `len(layer_kinds) != n_layers`
- `family` requires a sub-object that is absent (`mla*` without `mla`)
- `quantization.router.dtype != "f32"`
- `top_k > n_experts`, `n_groups` does not divide `n_experts`, `topk_group > n_groups`
- `n_shared_experts > 0` with `shared_intermediate == 0`
- tokenizer round-trip mismatch
- any `economics` field absent (profiling did not complete)

---

## See also

- [registry/001_init.sql](registry/001_init.sql) — the registry DDL
- [../docs/architecture.md](../docs/architecture.md) — the seam this IR feeds
