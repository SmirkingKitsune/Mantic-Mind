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
2. **4 KB-aligned offsets**, so no expert range shares a page with its neighbour and an unbuffered read
   of one is legal. Reading an expert therefore touches the minimum number of pages and never pulls in
   part of another.

   **Ordinary expert reads are buffered**, and the OS page cache is a deliberate free L2 under
   `MemoryHierarchy` (`include/soma/memory_hierarchy.hpp`). Three comments used to claim reads were
   issued `O_DIRECT`; none ever were, and the claim contradicted the L2 design it sat beside. What the
   alignment actually buys is the paragraph above, plus keeping unbuffered reads *available* to a caller
   that can meet the other two conditions — an aligned destination and a length rounded up into the
   padding. The bandwidth probe does both (see below); the memory tier, which allocates an exact-length
   `std::vector`, does neither, and adopting unbuffered reads on the serving path would start there.
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
  chat_oracle.bin         conversations + the ids HF's own renderer produced,
                          or chat_template.unsupported with the reason
```

The compiled chat template lives INSIDE `tokenizer.soma` (format 2; version 1 is still accepted and
means the file carries none) rather than beside it, because a template is only meaningful against the
tokenizer that resolved it to ids — two separate files could be paired wrongly, and the symptom would
be a served model whose prompt framing is one vocabulary out. `chat_oracle.bin` is separate because it
is the GRADER, not the artifact: the `chat_template` conformance stage fails a container that carries a
template with nothing to check it against.

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

`source_quantization` records the codec the conversion read, which is a different fact from every other
one in the file: `none` for an ordinary bf16/f32 upload, `fp8-e4m3-block-<bx>x<by>` for a blockwise-fp8
one. Nothing else could tell those apart, because the container records the codec it WROTE and never the
one it was handed — and they are not the same artifact. A q4_g container built from
`zai-org/GLM-5.3` and one built from `zai-org/GLM-5.3-BF16` hold the same architecture and different
numbers, and only the second can be compared against bf16 weights at all.

The copied `config.json` is not that record and must not be read as one. It is verbatim, so a
blockwise-fp8 source leaves a `quantization_config` in it that describes the **upload**, not the
container — whose experts are `dtype_gate_up`/`dtype_down` and whose dense half is unquantized. Nothing in the
engine reads that key, and `verify_payload.py` asks the source directory's own `config.json` rather than
this copy, because `--source` may legitimately point at a different upload of the same weights.

For ordinary containers, the dense half is stored **unquantized regardless of `--quant-dense`**, and
that is deliberate: the loader quantizes it into RAM per the role's spec, so the resident precision can
be changed without reconverting a byte — which is exactly what the expert half cannot do.
`--quant-dense` is therefore a flag on `plan` and `serve`, not on the converter.

**Unquantized is not the same as f32.** `--dense-storage source` (the default) copies the resident
matrices in BF16 or F16 when that is the checkpoint's precision, halving their payload.
BF16 is the top 16 bits of an F32, so the engine's widening at load is exact, and
upcasting on the way to disk merely moved that widening earlier onto a file that then has to be stored,
transferred to every node and read at every startup. An f32 source stays f32 in full; storing it narrow
would lose bits the checkpoint actually had. `--dense-storage f32` restores F32 storage.

The 1-D **controls** — norms, router biases, sinks — stay f32 whatever the source was. They are
kilobytes beside the matrices and the engine binds them as zero-copy views, so full precision costs
nothing worth counting. `container_meta.json` records `dense_storage` and `dense_narrow_tensors`.

The trade this makes is worth naming: an f32 matrix is bound as a zero-copy view straight into the
mapped file, while a narrow one is widened into memory the model owns. Same resident bytes either way —
which is what the plan already counts — but anonymous pages rather than reclaimable file-backed ones,
paid once at load. `tools/ci/check_bf16_source.py` pins the part that must not move: the expert payload
is byte-identical from either source, and every dense tensor widens back exactly.

The tokenizer is compiled INTO the container, before the expert loop, and the outcome is recorded in
`container_meta.json` and repeated in the converter's final summary. It is NON-FATAL: most families'
pretokenizers are not compiled yet, and aborting a multi-hour conversion over a tokenizer would be
disproportionate to a gap the container can be used without. Without a compiled tokenizer, the fallback
is one token per byte, which produces real tokens from real weights and meaningless text.
`soma serve` refuses that fallback unless `--allow-byte-tokenizer` is supplied, and
`conform` reports `tokenizer_roundtrip` as skipped rather than passed.

That refusal is the point of recording `tokenizer` in the meta at all. Three arrivals, three sentences:
a family whose pretokenizer is not compiled yet is a known gap; a container whose meta says `compiled`
with no `tokenizer.soma` beside it has lost a file; one that will not open is broken. The fallback stays
reachable because it is how the engine, the scheduler and the KV path are exercised on the families that
have no compiled pretokenizer — which is most of them — but it is a decision now, and a loud one:
`GET /v1/models` reports `"tokenizer": "byte-fallback"` with the reason, and the line `soma serve`
prints once it is listening carries the same warning. A client connecting to a server already running
has no other way to learn that its text means nothing.

On a node, the flag travels in the agent's `extra_args`. The engine descriptor does not add it — a node
choosing that for an operator would put the decision back where it was.

The ordinary **dense half stays in safetensors** deliberately. It is loaded once, in full, at startup — none of
the four requirements above apply to it, and keeping a standard format means it stays inspectable with
ordinary tools, at whichever precision it was stored.

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
| `0x2` | `expert_digests` | a per-expert digest table follows the index |

`0x2` was added with **no version bump**, which is this mechanism working as
designed: every v2 reader already refuses a bit it does not know, so no v2 build
can walk past the digest table and misread what follows. The role descriptor
needed the bump only because v1 readers ignored the word entirely.

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

### Digest table

Present iff `flags & 0x2`, immediately after the entries:

```
n_layers × n_experts × u64
```

One per slot, in the same order, so the entry stride stays 16 bytes and every
offset computation that predates the table keeps working. Each value is the first
8 bytes of the SHA-256 of that expert's `length` bytes, little-endian. A
zero-length slot — a dense layer's — stores `0` rather than the digest of an empty
string, so a missing entry and a real one are never confusable.

**A corruption check, not a tamper check.** It answers whether these bytes survived
being written, streamed to a node, and left on a disk for a month, which is the
question a cluster that copies container directories actually has. Truncation to
64 bits keeps the table at 8 B per expert — 125 KB for DeepSeek V4's 15,616 — with
a per-expert false-accept rate of 2^-64 against accidental damage. It is not
collision-resistant and nothing should read it as evidence against a deliberate
substitution. SHA-256 because OpenSSL is already linked for `arch_hash` and
`hashlib` is in Python's standard library: one obvious implementation per side,
no new dependency on either.

Three places check it, and they answer different questions:

| | reads | catches |
|---|---|---|
| the converter | the tensors, once | nothing — it *writes* the table |
| `soma stamp` | every expert when writing a stamp | damage between conversion and initial admission |
| `soma verify DIR` | every shard, on demand | damage in transit or at rest; needs no source checkpoint |
| the read path | each expert, on first touch | a bad byte at the moment it would enter the model |

The read path hashes bytes already in the destination buffer once per expert per
store opening. This assumes immutable files during that opening; later disk
changes require `soma verify` or reopening the store to detect. Hashing cost
depends on the CPU and storage. The bandwidth probe reads directly without
hashing, changing the model's verification policy, or marking experts verified.

An unchanged stamp is a no-op and explicitly reports that payload was not
re-read. Use `soma verify` after transfer. Containers without digest tables remain
readable; verification reports unsupported until a table is recorded. Recording
digests for an old container establishes a baseline for its current bytes and
does not prove those bytes survived the original conversion unchanged.

Everything else in this format checks *shape*: that the ranges pack canonically,
that each shard file is exactly the size those ranges imply, that the roles carry
the dtypes the IR names. **A correctly sized file full of wrong bytes satisfies all
of it.** Until the table existed, nothing read a payload byte at all.

### Measuring bandwidth

`ExpertStore::measure_bandwidth()` reads at this model's expert size in random order,
attempting to bypass the local OS page cache. It reports the method used:

| method | how | what the number means |
|---|---|---|
| `unbuffered` | `O_DIRECT` / `FILE_FLAG_NO_BUFFERING` on the probe's own handles | local page cache bypassed; device/server caches may remain |
| `cache-eviction-advised` | writes flushed, eviction advised for each padded range | cache bypass is not guaranteed |
| `buffered` | neither was available | **an upper bound; may be page-cache speed** |

The probe can do what the serving path cannot because it owns both ends: its own handles, its own
4 KB-aligned buffer, and a read length rounded up into the inter-expert padding — safe because every
shard is itself padded to the boundary and `validate_ranges()` has already required the file to be
exactly that size.

This matters when supplying the measured rate as `HostBudget::disk_bandwidth` for a verdict.
The planner currently uses a default or caller-supplied rate, not an automatic probe. Measured warm,
the fixture container reports **5793 MB/s**; measured unbuffered on the same file on the same host, **26
MB/s**. A verdict built on the first number says streaming is affordable on a host where it is not.
`SOMA_PROBE_NO_DIRECT=1` forces the fallback path, so the two can be compared on real hardware.

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
a rename. Initial stamping also reads every expert to check or establish its
digest; only the small index is rewritten.

Until this command existed, nothing stamped anything, so the mismatch branch could
never fire on any container ever written. Unstamped is therefore refused rather
than accepted. `soma serve --allow-unstamped` is the development-only escape;
conformance enforces the same strict boundary as serving.

**The stamp is the CONTAINER's identity, not the loaded model's.** These differ in
one direction: `--quant-dense` chooses the precision of the resident half at load,
and since that half is stored unquantized on disk precisely so the choice costs no
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
- **A digest of the dense half.** Dense tensors and alignment padding are outside
  this digest table. Same-size corruption of dense tensor values is not detected
  by this change.
