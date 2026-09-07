# Soma container — `soma.container` + `experts-*.bin`

**Format version:** 2
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

DeepSeek V4 uses the compatible indexed/sharded resident form because its resident half is itself too
large for a single dense sidecar:

```
  experts-00000.bin … experts-00060.bin  one atomically completed routed layer each
  dense.safetensors.index.json           lossless norms/routing/sinks/HC controls
  dense-00000.safetensors …              top-level then one lossless layer shard
  dense.qweights.index.json              mmap index for runtime-layout QTensor payloads
  dense-q-00000.bin …                    quantized embeddings/projections/shared experts/head
  conversion-manifest.json               pinned revision, config/index hashes, completed shards,
                                         and the committed DSpark/omission state
```

With `convert.py --include-dspark`, the pinned `mtp.0/1/2` source tensors are translated into stable
`model.dspark.*` names and a second compatible sidecar set:

```
  soma.dspark                             three-stage routed-expert index
  dspark-experts-00000.bin … -00002.bin   384 atomically completed experts per stage
  dspark.safetensors.index.json           lossless HC/norm/router controls
  dspark-dense-00000.safetensors …        one lossless resident shard per stage
  dspark.qweights.index.json              runtime-layout projection/Markov/head index
  dspark-dense-q-00000.bin …              quantized resident payloads
```

The manifest records `dspark_included: true` and an empty `omitted_namespaces` only after every
auxiliary shard and index is committed. A resumed base-only conversion therefore remains safely
autoregressive until the augmentation is complete. Without the flag, the three `mtp.*` namespaces are
the only permitted omissions and remain recorded explicitly. Single-file dense sidecars are
unchanged.

`SafeTensors::open_dir` unions the two resident indexes. Existing single-file
`dense.safetensors` remains valid and is not rewritten.

Legacy v1 indexes remain parseable only behind the explicit development escape
`soma serve --allow-unstamped`. Strict serving refuses them because v1 did not
require a per-role descriptor, so even a populated hash cannot rule out a
same-size role permutation. A descriptorless v1 index cannot be safely stamped;
reconvert it with the current converter. Transitional v1 indexes that already
carry a complete descriptor can be validated and upgraded by `soma stamp`.

**There is no `arch.json` FILE in a container.** This listing named one and no converter has ever
written it. The IR is ADAPTED from `config.json` at load, by the same `resolve_arch()` that
`soma plan` uses, so a container and a plain HF checkpoint go down one path — a second description
file that had to agree with the first is how the two drift.

The IR is still a real persisted artifact elsewhere: admission stores it in the registry's `arch_json`
column, and `GET /v1/models/{id}` returns it. What does not exist is a copy of it sitting beside the
weights.

`container_meta.json` is not a second description of the architecture. It records what the CONVERSION
did: `dtype_gate_up`, `dtype_down`, `group`,
`effective_groups`, `effective_groups_by_role`, `expert_bytes`, `total_expert_bytes`, `n_shards`,
`layer_kinds`, `dense_tensors`, and `tokenizer` (`compiled` | `unsupported`).
The v2 binary header independently repeats each routed role's dtype and effective
group; the runtime compares that descriptor with the IR resolved from this
record, while the stamp binds the resolved record to the canonical hash.

`effective_groups` is keyed by **dtype**, which is wrong whenever two roles share a dtype but not a
row width — gate/up quantize along `d_model` and down along `expert_intermediate`, so the two collide
in that dict and the last one written wins. `effective_groups_by_role` is the one to read; the
dtype-keyed dict stays for readers that already consume it. `arch_hash` covers the quant map precisely so that the
same weights at two quantizations are two models, with two verdicts and two sets of KV checkpoints.

For ordinary containers, the dense half is stored **F32 regardless of `--quant-dense`**, and that is
deliberate: the loader
quantizes it into RAM per the role's spec, so the resident precision can be changed without
reconverting a byte — which is exactly what the expert half cannot do. `--quant-dense` is therefore a
flag on `plan` and `serve`, not on the converter.

The tokenizer is compiled INTO the container, before the expert loop, and the outcome is recorded in
`container_meta.json` and repeated in the converter's final summary. It is NON-FATAL: most families'
pretokenizers are not compiled yet, and aborting a multi-hour conversion over a tokenizer would be
disproportionate to a gap the container can be used without. A container without one still serves —
`soma serve` falls back to one token per byte, which produces real tokens from real weights and
meaningless text, and `conform` reports `tokenizer_roundtrip` as skipped rather than passed.

The ordinary **dense half stays in safetensors** deliberately. It is loaded once, in full, at startup — none of
the four requirements above apply to it, and keeping a standard format means it stays inspectable with
ordinary tools.

V4 is the exception described above: large resident matrices are translated offline into the chosen
Soma QTensor layout and bound directly from `dense-q-*.bin`; lossless controls remain inspectable
SafeTensors. A serve-time resident dtype that disagrees with this prequantized index is refused rather
than silently requantized.

### `soma.container`

All integers little-endian.

| Offset | Type | Field |
|---|---|---|
| 0 | `char[8]` | magic `SOMACTNR` |
| 8 | `u32` | `format_version` |
| 12 | `u32` | `flags` — **must-understand**; see below |
| 16 | `u32` | `arch_hash_len` |
| 20 | `char[n]` | `arch_hash` |
| … | `u32` | `n_layers` |
| … | `u32` | `n_experts` |
| … | `u32` | `n_shards` |
| … | `u32` | `expert_dtype` — legacy; cannot express a split map, see below |
| … | `u32` | `expert_group` |
| … | *role descriptor* | present iff `flags & 0x1`; see below |
| … | `u64` | `expert_bytes` — exact uniform expert length; variable layouts are refused |
| … | `u64` | `total_expert_bytes` |
| … | index | `n_layers × n_experts` entries |

### `flags` — must-understand

A reader that meets a bit it does not know **refuses the container**. It does not
skip the bit and carry on: an unknown bit means a field the reader cannot see sits
somewhere in the header, so every offset after it is a guess — and the guess does
not fail, it yields a plausible index pointing at the wrong bytes.

V2 also bumps `format_version` because shipped v1 readers ignored this flags
word. A must-understand bit protects future v2-aware readers; it cannot retrofit
that behavior into an old executable. Old readers therefore fail on v2 at the
version gate instead of parsing the descriptor bytes as sizes and indexes.

| Bit | Name | Meaning |
|---|---|---|
| `0x1` | `per_role_quant` | the role descriptor below is present |

### Role descriptor

```
u32  n_roles
n_roles ×  { u32 role_id, u32 dtype_id, u32 group }
```

`role_id` is `soma::TensorRole` — 2 gate, 3 up, 4 down — pinned by a
`static_assert` so reordering that enum breaks the build rather than silently
redefining these bytes. `group` is the **effective** group, after the reduction
`quantize_tensor()` applies for the row width, not the group the operator asked
for. V2 requires exactly one descriptor for each of gate, up, and down. Unknown,
missing, or duplicate role ids are fatal, as is an unknown flag.

It exists because `expert_dtype` is a single value and **cannot describe the
default map** — gate/up at `q4_g` with down at `q6_g`. A reader had only
`expert_bytes` to check the IR against, and that is a size proxy rather than a
statement about which role holds which format: gate, up and down all hold
`expert_intermediate × d_model` elements, so any map permuting dtypes between roles
totals identically. No converter or overlay produces such a map today, because
`container_meta.json` carries one `dtype_gate_up` field and gate and up therefore
cannot differ — but that is a property of this JSON schema, not of the format, and
the engine's own reader discarded `expert_dtype` into `(void)` for the whole life
of the field.

Index entry, one per `(layer, expert)` in layer-major order:

| Type | Field |
|---|---|
| `u32` | `shard` |
| `u64` | `offset` — into that shard, 4 KB-aligned |
| `u32` | `length` |

`expert_bytes` is uniform for any single model and quantization, so the index is strictly redundant
today. It is written anyway: variable-length experts are plausible (mixed per-expert precision, pruned
experts), and a format that assumed uniformity would need a version bump to allow them.

Every MoE slot must have that exact length; dense-layer slots must be empty.
Within each shard, live ranges must follow the index order from offset zero,
with only alignment padding between them and after the last range. Aliases,
gaps, missing experts, and unexpected shard bytes are refused by both open and
stamp. Shard count is bounded by the index entry count and 100,000.

DSpark auxiliary indexes skip the architecture hash through an explicit internal
policy, but still require v2 role descriptors and the same range checks.

**`arch_hash` is checked on open**, and a container that carries none is refused
outright. Requantization changes the hash, and reading q4 bytes as q6 produces
finite, wrong numbers rather than an error.

The hash is written by `soma stamp DIR`, which resolves the
container's IR through the same `resolve_arch()` the planner and the server use,
verifies the mandatory role descriptor against it, and then stamps the hash into
the index. `convert.py` cannot do it: the canonical hash is defined
by the C++ IR canonicalization and a second implementation in Python would agree
until it did not. Only the small index file is rewritten — through a temporary plus
a rename — so stamping a 61 GB container costs a few hundred kilobytes of I/O.

Until this command existed, nothing stamped anything, so the mismatch branch could
never fire on any container ever written. Unstamped is therefore refused rather
than accepted. `soma serve --allow-unstamped` is the development-only escape;
conformance enforces the same strict boundary as serving.

**The stamp is the CONTAINER's identity, not the loaded model's.** These differ in
one direction: `--quant-dense` chooses the precision of the resident half at load,
and since that half is stored F32 on disk precisely so the choice costs no
reconversion, it moves `arch_hash` without moving one byte of the container. The
stamp covers the IR carrying the map `container_meta.json` declares — what the
shards actually are — which the engine tracks as `ArchIr::container_arch_hash`.

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
