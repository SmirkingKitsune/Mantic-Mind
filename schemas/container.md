# Soma container — `soma.container` + `experts-*.bin`

**Format version:** 1
**Produced by:** `tools/admission/convert.py`
**Consumed by:** `soma::ExpertStore`

The on-disk form the streaming path reads. Everything about it exists to make a cache miss cheap.

---

## Why not just read the safetensors

Four requirements that safetensors does not meet, each of which would otherwise cost real time on every
expert miss:

1. **One expert = one contiguous byte range.** A routed expert is gate + up + down. Upstream stores those
   as three separate tensors, so fetching one expert means three seeks. Here they are concatenated, and a
   single read fetches the whole SwiGLU triple.
2. **4 KB-aligned offsets**, so `O_DIRECT` / unbuffered reads are legal. Misaligned ranges force a
   read-modify-write through the page cache, which is exactly the layer streaming is trying to bypass.
3. **A sidecar index**, so a miss never parses a JSON header. The safetensors header for a 61 GB
   checkpoint is megabytes of JSON; re-parsing it per miss is absurd, and caching it in memory means
   holding a structure whose only purpose is answering a question the index answers in one lookup.
4. **Pre-transposed** fused 3D expert tensors. Transposing at runtime would mutate state the model tier
   promises is immutable, and the lock-free read of that tier depends on the promise being literal.

---

## Layout

```
<model_dir>/
  config.json             the SOURCE config, copied verbatim
  container_meta.json     the record of this conversion — see below
  soma.container          header + sidecar index — small, read once, fully
  experts-00000.bin       shard: 4 KB-aligned expert ranges
  experts-00001.bin       …
  dense.safetensors       the resident half: attn projections, norms, embeddings,
                          router weights, shared experts
  tokenizer.soma          compiled tokenizer      ) all three, or
  tokenizer_oracle.bin    golden ids for the gate ) tokenizer.unsupported
  tokenizer_meta.json                             ) with the reason
```

**There is no `arch.json` FILE in a container.** This listing named one and no converter has ever
written it. The IR is ADAPTED from `config.json` at load, by the same `resolve_arch()` that
`soma plan` uses, so a container and a plain HF checkpoint go down one path — a second description
file that had to agree with the first is how the two drift.

The IR is still a real persisted artifact elsewhere: admission stores it in the registry's `arch_json`
column, and `GET /v1/models/{id}` returns it. What does not exist is a copy of it sitting beside the
weights.

`container_meta.json` is not a second description of the architecture. It records what the CONVERSION
did, and it is the only place the quantization exists at all: `dtype_gate_up`, `dtype_down`, `group`,
`effective_groups`, `expert_bytes`, `total_expert_bytes`, `n_shards`, `layer_kinds`, `dense_tensors`,
and `tokenizer` (`compiled` | `unsupported`). `arch_hash` covers the quant map precisely so that the
same weights at two quantizations are two models, with two verdicts and two sets of KV checkpoints.

The dense half is stored **F32 regardless of `--quant-dense`**, and that is deliberate: the loader
quantizes it into RAM per the role's spec, so the resident precision can be changed without
reconverting a byte — which is exactly what the expert half cannot do. `--quant-dense` is therefore a
flag on `plan` and `serve`, not on the converter.

The tokenizer is compiled INTO the container, before the expert loop, and the outcome is recorded in
`container_meta.json` and repeated in the converter's final summary. It is NON-FATAL: most families'
pretokenizers are not compiled yet, and aborting a multi-hour conversion over a tokenizer would be
disproportionate to a gap the container can be used without. A container without one still serves —
`soma serve` falls back to one token per byte, which produces real tokens from real weights and
meaningless text, and `conform` reports `tokenizer_roundtrip` as skipped rather than passed.

The **dense half stays in safetensors** deliberately. It is loaded once, in full, at startup — none of
the four requirements above apply to it, and keeping a standard format means it stays inspectable with
ordinary tools.

### `soma.container`

All integers little-endian.

| Offset | Type | Field |
|---|---|---|
| 0 | `char[8]` | magic `SOMACTNR` |
| 8 | `u32` | `format_version` |
| 12 | `u32` | `flags` |
| 16 | `u32` | `arch_hash_len` |
| 20 | `char[n]` | `arch_hash` |
| … | `u32` | `n_layers` |
| … | `u32` | `n_experts` |
| … | `u32` | `n_shards` |
| … | `u32` | `expert_dtype` |
| … | `u32` | `expert_group` |
| … | `u64` | `expert_bytes` — uniform stride, or 0 if variable |
| … | `u64` | `total_expert_bytes` |
| … | index | `n_layers × n_experts` entries |

Index entry, one per `(layer, expert)` in layer-major order:

| Type | Field |
|---|---|
| `u32` | `shard` |
| `u64` | `offset` — into that shard, 4 KB-aligned |
| `u32` | `length` |

`expert_bytes` is uniform for any single model and quantization, so the index is strictly redundant
today. It is written anyway: variable-length experts are plausible (mixed per-expert precision, pruned
experts), and a format that assumed uniformity would need a version bump to allow them.

**`arch_hash` is checked on open.** A container is refused against a model whose IR does not match —
requantization changes the hash, and reading q4 bytes as q6 produces finite, wrong numbers rather than
an error.

### `experts-*.bin`

Concatenated expert ranges, each padded to a 4 KB boundary. Within one expert:

```
[ gate rows | up rows | down rows ]
```

each already quantized per its tensor role. Gate and up share a dtype in every map seen so far; down is
commonly higher precision (`schemas/arch-ir.md` §5), so the three sections may differ in bytes-per-row.
The section sizes are derivable from the IR — and from `container_meta.json`, which records the
dtypes and the effective group this conversion actually used — so they are not repeated per expert.

Shards are capped (default 4 GiB) so the format works on filesystems without large-file support and so a
partial conversion can be resumed at shard granularity.

---

## Padding cost

Alignment wastes at most 4 KB − 1 per expert. Worked from the real configs at q4_g:

| Model | expert bytes | experts | padding waste |
|---|---|---|---|
| **Qwen3-30B-A3B** (measured) | **2,998,272 B** | **6144** | **0 B (0.000 %)** |
| DeepSeek-V2-Lite (computed) | 4.87 MB | 1664 | ≤ 6.5 MB (0.08 %) |
| Mixtral-8x7B (computed) | 99 MB | 256 | ≤ 1 MB (0.00 %) |

Qwen3's row is from a real conversion with the gate/up `q4_g` + down `q6_g` map: 884,736 + 884,736 +
1,228,800 = 2,998,272 B, which is 732 × 4096 exactly. Zero padding is luck rather than design, but the
bound holds regardless — the worst case is 4 KB − 1 per expert, and the alternative (unaligned reads)
costs a read-modify-write on every miss.

---

## What this does NOT store

- **KV checkpoints.** Separate format, separate lifetime, separate version gate
  (`include/soma/kv_checkpoint.hpp`).
- **The heat map.** Lives in the registry, because it is *measured* and mutates during serving while the
  container is immutable.
- **Kernel choices.** Registry too, for the same reason — and because they are host-specific while the
  container is portable.
