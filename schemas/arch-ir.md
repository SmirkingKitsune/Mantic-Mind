# Architecture IR — `arch.json`

**Schema version:** 1
**Produced by:** admission (`tools/admission/convert.py`)
**Consumed by:** the Soma loader, the planner, `soma plan --json`, the registry

`arch.json` is the complete, normalized description of a model's architecture. It is written once at
admission, hashed into `arch_hash`, and never edited by the runtime. Everything the engine needs to
decide *how to execute* a model is in here or in the sidecar index; nothing is inferred from tensor
names, filenames, or heuristics at load time.

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

`layer_kinds` is a materialized array rather than the upstream `decoder_sparse_step` /
`moe_layer_freq` / `mlp_only_layers` triangle. Three different families express "which layers are MoE"
three different ways; resolving that at admission means the core never has to.

---

## 3. `attention`

| Field | Type | Notes |
|---|---|---|
| `family` | enum | `mha` \| `gqa` \| `mla` \| `mla+dsa` — **selects the backend** |
| `n_heads` | int | |
| `n_kv_heads` | int | `gqa`/`mha` |
| `head_dim` | int | |
| `qk_norm` | bool | Per-head q/k RMSNorm (Qwen3) |
| `sliding_window` | int \| null | Span in tokens; null = full |
| `bias` | bool | |
| `rope` | object | `{ theta, partial_dim, interleaved, scaling }` |
| `rope.scaling` | object \| null | `{ type: "yarn"\|"linear"\|"ntk", factor, original_max_position, beta_fast, beta_slow, mscale, mscale_all_dim }` |
| `mla` | object \| null | Present iff `family` starts with `mla` |

`mla` sub-object:

| Field | Notes |
|---|---|
| `kv_lora_rank` | Compressed latent width |
| `q_lora_rank` \| null | null = no Q down-projection (V2-Lite) |
| `qk_nope_head_dim` | Non-RoPE part of the QK head |
| `qk_rope_head_dim` | RoPE-carrying part |
| `v_head_dim` | |
| `absorb_weights` | bool — enable load-time weight absorption |

**`absorb_weights` has no GQA analogue**, which is exactly why it lives in the `mla` sub-object and
`AttentionBackend::prepare_weights()` exists as a hook rather than as a step in the loader.

### KV cost — the number the planner actually cares about

`kv_bytes_per_token()` is the only attention property that crosses the seam. Worked from the real
configs:

| Model | Family | Elements/token/layer | Layers | Bytes/token @fp16 | @32k ctx |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | gqa | `2 × 4 × 128` = **1024** | 48 | 98 KB | **3.2 GB** |
| DeepSeek-V2-Lite | mla | `512 + 64` = **576** | 27 | 31 KB | **1.0 GB** |
| Mixtral-8x7B | gqa | `2 × 8 × 128` = **2048** | 32 | 131 KB | **4.3 GB** |

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
| `score_fn` | enum | `softmax` \| `sigmoid` |
| `normalize_topk` | bool | `norm_topk_prob` |
| `routed_scaling_factor` | float | |
| `bias_correction` | bool | Per-expert bias added before top-k (V3-style) |
| `n_groups` | int | Group-limited routing; 1 = ungrouped |
| `topk_group` | int | |
| `n_shared_experts` | int | 0 = none |

### `ffn`

| Field | Type | Notes |
|---|---|---|
| `activation` | enum | `swiglu` \| `geglu` \| `relu2` |
| `has_gate` | bool | |
| `expert_intermediate` | int | `moe_intermediate_size` |
| `dense_intermediate` | int | For `layer_kinds == "dense"` layers |
| `shared_intermediate` | int | |
| `expert_layout` | enum | `interleaved_gud` — gate/up/down interleaved per expert (§5.3 of architecture.md) |

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

It is a property of `(model, quantization, host budget)`. Mixtral is `resident-only` because of its
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
