# Soma — roadmap

**Gates, not milestones.** A gate is a falsifiable condition. "Streaming works" is a milestone; "measured
`bytes/token` matches the planner's prediction within 10% and logit-KL against the fp16 reference stays
under threshold over 500 prefill positions" is a gate. Only the second one can fail.

## Hard sequencing rule

> **The conformance harness and the seam must both exist before the second architecture is admitted.**

Otherwise you debug the seam and the model simultaneously with no ground truth, and every failure is
ambiguous between the two. Concretely: the harness lands at **G0/G1**, the seam is exercised in anger at
**G2**, and the second architecture arrives at **G4** — never earlier, even if it looks easy.

## Gate summary

| Gate | Deliverable | Passes when |
|---|---|---|
| **G0** | fp32, tiny model, single sequence | Token-exact teacher-forced match vs. the `transformers` oracle |
| **G1** | Quantized path, tiny model | Token-exact greedy match |
| **G2** | Streaming from disk, GQA | Logit-KL under threshold on the real checkpoint; measured economics match the plan |
| **G3** | Concurrency — batch-union across sequences | Aggregate throughput scales super-linearly in the disk-bound regime **+ tier/heat text dump exists** |
| **G4** | Second architecture (MLA) through the seam | Full ladder passes on both families with **zero core diffs** |
| **G5** | Subprocess integration + verdict routing | Mixtral admits `resident-only` and routes to the fallback, unprompted |
| **G6** | API surface complete | Every capability on `/v1/*`, scope-gated, coverage check green |
| **G7** | FTXUI dashboards on that API | Panels consume only `/v1/*` |
| **G8** | Admission self-service | A new HF repo goes end-to-end with no C++ change |

---

## G0 — fp32, tiny model, token-exact, single sequence

**Build:** `ArchIr` parse/validate, model loader, GQA attention (fp32), fp32 router, SwiGLU, RMSNorm,
RoPE, compiled tokenizer, single-sequence forward. `tools/admission/make_oracle.py`.

**Gate:**
- `make_oracle.py Qwen/Qwen3-30B-A3B` produces a tiny-random model with the **real** architecture config
  (48 layers → 4, `d_model` 2048 → 64, 128 experts → 8; `top_k`, `active_fraction`, and every semantic
  field preserved).
- Teacher-forced logits match the `transformers` oracle to fp32 tolerance over ≥512 positions.
- Greedy generation is **token-exact** for ≥256 tokens.
- Tokenizer round-trips byte-for-byte against HF `tokenizers` over the calibration corpus.

### G0 status — PASSED for the GQA family

| Fixture | logits (max abs) | greedy | tokenizer |
|---|---|---|---|
| OLMoE-1B-7B (MHA, full-width qk-norm) | 1.31e-06 | 256/256 | 36/36 |
| Qwen3-30B-A3B (GQA 16/2, per-head qk-norm) | 1.01e-06 | 256/256 | 36/36 |
| Mixtral-8x7B (GQA 8/2, forced top-k renorm) | 1.79e-06 | 256/256 | refused¹ |
| DeepSeek-V2-Lite, Moonlight | — | — | — (MLA, G4) |
| granite-3.0-3b-a800m | — | — | — (4 arch multipliers) |

Errors are fp32 reassociation noise against logits with σ≈0.4.

¹ **Tokenizer coverage is narrower than model coverage, deliberately.** The compiler *recognizes or
refuses* — it never approximates. Refused today, each with its reason recorded in
`tokenizer.unsupported`: Mixtral's byte-fallback/SentencePiece pipeline, DeepSeek's 5-stage Split chain
with explicit CJK/Latin ranges, granite's legacy `vocab.json`+`merges.txt`, and Moonlight (which ships
no tokenizer files at all). A tokenizer gate that silently approximated would pass while
mis-tokenizing, and mis-tokenization presents at G2 as "the model is subtly stupid" rather than as a
tokenizer fault.

**Known limitation: NFC normalization is not implemented** in the engine (no ICU; a composition table
dwarfs the rest of the format). `compile_tokenizer.py` verifies every calibration string is NFC-stable
and reports how many it dropped — 1 per family — so the gate states its coverage instead of passing by
accident.

**Why tiny-random and not a real checkpoint:** a real model can be *approximately* right in ways that
hide a bug for weeks. A 4-layer random model is either exactly right or obviously wrong, and it runs in
under a second so the check can be per-commit.

**Explicitly deferred:** quantization, streaming, concurrency, MLA, telemetry, serve mode.

---

## G1 — quantized path, token-exact

**Build:** quantized kernels (q4_k, q6_k, q8_0), the per-role quant map, dequant-on-read for experts,
the autotuner and its static dispatch table.

**Gate — AMENDED. The original wording measured the wrong thing.**

- `validate_quant_map` rejects a non-F32 router **and** non-F32 norms. Tested by trying it.
- Codec round-trip error matches each format's theoretical bits/weight.
- **The forward propagates quantization error linearly rather than amplifying it** — relative logit
  error stays within a small factor of relative weight error.
- The autotuner produces `kernel_choice` rows for the model's actual shape set, and the single-row and
  batched families are both exercised.
- Every candidate for an `(op, dtype)` is **numerically equivalent** — asserted before any timing is
  trusted, or the tuner would be selecting on speed while silently varying output.
- **Every extracted shape resolves to a callable implementation**, including shapes the tuner skipped.

### Why the token-exactness criterion was replaced

The original gate read "greedy generation on the tiny quantized model is token-exact against fp32".
Measured, it does not hold — and **it does not hold for a reason that has nothing to do with kernel
correctness**.

A tiny-*random* model has essentially no logit margin. Its output distribution is nearly flat (σ≈0.37
on the fixtures), so any perturbation flips argmax within a few tokens. At q8_0 — a codec demonstrably
correct to 9.00 bits/weight with 5.2e-03 relative error — greedy diverges after ~30 tokens. Nothing is
wrong; the metric is simply not sensitive to what it was supposed to detect.

What *does* separate a correct quantized forward from a broken one is whether end-to-end error **tracks**
the weight error. A mis-packed nibble, a wrong group stride, or a broken accumulation amplifies it by
orders of magnitude.

Token-exactness becomes a real signal again at **G2**, on a trained checkpoint with actual logit margins.

### G1 status — PASSED

Codec, on the fixtures' own weight distribution:

| format | bits/weight | rel_rms |
|---|---|---|
| q8_0 g32 | 9.00 | 5.23e-03 |
| q8_0 g128 | 8.25 | 6.44e-03 |
| q6_g g128 | 6.25 | 2.62e-02 |
| q4_g g128 | 4.50 | 1.00e-01 |
| q4_0 g32 | 5.00 | 9.57e-02 |

Bits/weight match the formats exactly (e.g. q4_g g128 = 4 bits + 64 bits of scale/min per 128 weights =
4.50), which is itself a packing check.

**Error amplification through the full forward, across three families × six configs: 1.57 – 2.90.**
Bounded, consistent, and orders of magnitude below what a packing or stride bug produces. Gate
threshold is 6.0.

> **These are Soma's own formats.** `Q4_G` / `Q6_G` are plain group-scale quants and are **not**
> bit-compatible with llama.cpp's `Q4_K` / `Q6_K` super-block K-quants. They were renamed off the `_K`
> suffix precisely so nobody assumes otherwise — Soma writes its own container, so there is nothing to
> gain from compatibility and a real cost to implying it.

**Failing G0 or G1 → `verdict: reject` → fallback.** That is what these gates are *for*: they are the
admission ladder's stages 1 and 2, not merely development checkpoints.

### Autotuner results — the prior art's finding reproduces, with a caveat

Measured on the dev box across 11 production shapes (`d_model` 2048, 128 experts, 151936 vocab):

| shape | dtype | winner | GF/s | runner-up |
|---|---|---|---|---|
| gemv 1×4096×2048 | q4_g | `q4_g.fused` | 4.40 | `q4_g.dequant` 2.76 |
| gemv 1×128×2048 | **f32** | `f32.unroll4` | **10.80** | `f32.scalar` 5.48 |
| gemv 1×151936×2048 | q8_0 | `q8_0.fused` | 4.90 | `q8_0.dequant` 4.29 |
| gemv 1×2048×768 | q6_g | `q6_g.fused` | 4.09 | `q6_g.dequant` 2.63 |

Two results worth recording:

**1. fp32 is 2.2–2.4× faster than q4_g in raw throughput.** The prior art's observation reproduces. But
it is a *compute* comparison and does not argue against quantization here — q4_g is 4.50 bits/weight
against fp32's 32, so it reads **7× less from disk**. In the disk-bound streaming regime Soma is built
for, FLOP rate is not the binding constraint. The finding argues for measuring streaming economics
per model (§ G2), not for storing weights wide.

**2. Dequantize-first lost every one of 11 shapes** (1.5–1.6× slower). Materializing the row as fp32
costs a full scratch write per output, and the tighter fp32 inner loop never recovers it — not even at
m=32, where the write might have been amortized across the batch. Recorded rather than removed: the
candidate is cheap to keep, and the answer may invert on aarch64 or with a wider SIMD path, which is
precisely why the choice is measured rather than assumed.

`f32.unroll4` beating `f32.scalar` ~2× confirms the measurement is picking up real differences and not
noise — four independent accumulators overlap FMA latency instead of serializing on it.

---

## G2 — streaming from disk, one architecture

**Build:** container writer, sidecar index, `ExpertStore` with aligned/`O_DIRECT` reads, `MemoryHierarchy`
(RAM LRU + pin + heat), async readahead, the bounded load pool, router-lookahead prefetch, the profiler,
the verdict function, `plan --json`.

**First real model: Qwen3-30B-A3B** — GQA, fine-grained (128 experts, top-8, 2.4 MB experts at q4,
6.3% active fraction).

**Gate:**
- Conformance stage 3: logit-KL vs. the fp16 reference under threshold over ≥500 prefill positions on
  the real checkpoint.
- Measured `bytes/token` matches the plan's prediction within 10%.
- Measured disk bandwidth is taken **at this model's expert size**, not from a spec sheet.
- `pilot_profile` has a per-layer recall figure, and prefetch is enabled only above threshold. **A layer
  with poor recall must end up with prefetch off** — verify at least one does, or the threshold is not
  doing anything.
- A cold start followed by a warm start shows the heat bootstrap working: measurably fewer misses.

**Watch for:** at q4 on a large-RAM host the 14.5 GB routed set may simply fit, and the verdict comes
back `resident-only`. That is the correct answer. Use bf16 or a constrained `ram_budget` to exercise the
streaming path, and note that the gate is about *correctness plus predicted-vs-actual economics*, not
about forcing a streaming verdict.

### G2 status — streaming path PASSED; stage 3 pending the real checkpoint

| Check | Result |
|---|---|
| Container round-trips byte-identical to engine quantization | 160/160 experts, 3 families |
| Expert with a live `ExpertRef` survives eviction pressure | 58 evictions, address unchanged |
| LRU evicts the least-recently-used unpinned slot | verified |
| Thrash gate fires above `cap_per_layer` | verified |
| Prefetch OFF by default; ≥1 layer disabled by recall | 2 on / 2 off |
| Prefetch on a disabled layer is a no-op | verified |
| Heat bootstrap: warm start has fewer misses | 573 → 550 |
| **Measured bytes/token vs plan** | **69632 vs 69632 — 0% error** |
| Verdict function vs `schemas/arch-ir.md` §8 | 6/6 cases agree |
| Random-read bandwidth at this model's expert size | measured, 507–1054 MB/s at 4–8 KiB |

### Measured on the real container: the storage medium decides the verdict

Qwen3-30B-A3B converted to `q4_g` gate/up + `q6_g` down: **6144 experts, 2,998,272 B each
(exactly 732 × 4096, so zero padding), 18.42 GB payload, 5 shards.**

The same container, measured on two drives:

| medium | random-read BW at 3 MB | verdict @8 GiB | verdict @16 GiB |
|---|---|---|---|
| network-mounted HDD | **14 MB/s** | `reject` (0.012 tok/s) | `reject` (0.021 tok/s) |
| local SSD | **1230 MB/s** | `stream` (1.07 tok/s) | `stream` (1.85 tok/s) |

An 88× difference in one measurement flips the verdict for the same model at the same quantization on
the same host. This is the strongest argument for the design's insistence that **bandwidth is measured
at admission, at this model's expert size, rather than taken from a spec sheet** — and for the
`kMinProjectedTokS` floor, which fired here for the first time on real numbers and correctly refused a
configuration that would have "worked" at 0.02 tok/s.

It also means a deployment can be broken purely by where the container lives, with nothing else wrong.
The plan document surfaces that rather than burying it.

### Conformance stage 3 — PASSED

Real Qwen3-30B-A3B container (`q4_g` gate/up, `q6_g` down) against a **bf16 reference**, 512 prefill
positions, 8 GiB expert cache:

| | |
|---|---|
| mean KL(ref ‖ engine) | **0.0197 nats** (gate: ≤ 0.05) |
| median | **0.00000** |
| p95 | 0.0618 (gate: ≤ 0.25) |
| max | 4.305 at position 0 |
| **top-1 agreement** | **98.2 %** (503/512) |
| cache | 97.1 % hit, 5614 misses, 2750 evictions, 16.1 GiB read |
| forward | 685 s (1.34 s/token, single-threaded) |

A median of exactly 0 means the engine's distribution matches the reference to fp32 print precision at
most positions. The max sits at **position 0** — the first token, which has no context and is therefore
the least constrained prediction in the sequence; that it is the outlier is expected rather than
concerning.

Note the bar here is distributional, not exact. Stages 1 and 2 demand token-exactness on tiny-random
models; a real checkpoint at q4_g against a bf16 reference cannot and should not reproduce bit-for-bit.

### The bug stage 3 caught, and why the tiny fixtures could not

The first stage-3 run returned **mean KL 11.66** with **0 % top-1 agreement** — and `ln(151936) = 11.93`,
i.e. the engine was emitting a uniform distribution. The cache showed exactly 384 misses (48 layers × 8
experts): every token routed to the same experts.

Cause: a torch→numpy lifetime error in `convert.py`.

```python
handles[i].get_tensor(name).to(torch.float32).numpy()   # dangling view
```

`get_tensor()` returns an mmap-backed tensor. On an **f32** checkpoint `.to(float32)` is a no-op
returning that same tensor, whose storage lives as long as the handle — safe. On a **bf16** checkpoint
it *allocates*, `.numpy()` views the new storage, and the tensor is freed when the function returns.
Large arrays survived by luck; small ones were clobbered. Result: `embed_tokens` and `lm_head` (1.2 GB
each) correct, every per-layer tensor all zeros.

**The tiny fixtures are f32, so this bug is structurally invisible to them.** The container round-trip
test was byte-identical and correct, and could never have caught it. The first real bf16 model exposed
it — which is the argument for stage 3 existing at all, and for it running on a real checkpoint rather
than a scaled-down one.

Two process lessons recorded rather than glossed:

* **Verify the artifact before consuming it.** A two-second `absmax` check over the container's dense
  tensors would have found the zeros immediately, instead of an 11-minute forward and a KL number that
  had to be reverse-engineered.
* **A gate must not misattribute its own failure.** The tool originally printed "This is a QUANTIZATION
  finding" for that run, which would have sent a reader to adjust the quant map while the weights were
  zero. It now discriminates: `KL > 0.8·ln(vocab)` or `top-1 < 5 %` reports *not* a quantization
  finding and points at unloaded weights. Quantization does not flatten a distribution to maximum
  entropy.

### Three bugs the G2 checks surfaced

1. **`np.rint` vs `std::lround`.** numpy rounds half to *even*; C++ rounds half *away from zero*. One
   weight on a `.5` boundary — Mixtral's up-projection, layer 0, expert 3, row 18 — broke byte-identity.
   This is why `convert.py` implements the quant formats a *second* time rather than sharing code: the
   round-trip compares Python's bytes against the engine's, so a divergence surfaces instead of both
   sides agreeing on the same mistake.

2. **The IR did not describe what was loaded.** `load_f32_model` left `arch.quantization` at its all-f32
   default even when loading quantized, so the planner predicted f32 footprints for a q4_g model —
   393216 B/token against a measured 69632, exactly the compression ratio. The IR now records the
   effective map, and `ExpertStore::open` cross-checks the container's expert size against it. That check
   exists because `convert.py` cannot stamp the canonical `arch_hash` (a second hash implementation in
   Python would agree until it did not), so expert size is the strongest available guard.

3. **Flat-element-count sizing was wrong for narrow tensors.** `quantized_bytes` cannot know a row width,
   so it could not apply the group reduction `quantize_tensor` performs when `cols < group`. Added
   `quantized_tensor_bytes`. Invisible on production dimensions (Qwen3's 768 and 2048 both exceed 128)
   and wrong on anything narrower.

One test weakness also worth recording: the eviction-safety check originally ran against the full
24-slot budget while churning 16 experts, so the cache never filled and `evictions` stayed at 0. It
passed while proving nothing. It now runs against a 3-slot cache and asserts eviction actually occurred
before asserting the held expert survived it.

---

## G3 — concurrency, batch-union across sequences

**Build:** the three state tiers as separate types, step-major loop, ragged batch assembly, CSR expert
union, cache-aware admission control, chunked prefill with the fairness cap, KV checkpoint store,
preemption. **And the telemetry text dump.**

**Gate:**
- Token-exactness preserved at `batch == 1` against G1.
- `unique_experts_last_step` is materially below `naive_expert_reads_last_step` at batch > 1. **If the
  ratio is near 1.0 the union is buying nothing** and something upstream is wrong.
- **Throughput conversion** — the three-part gate below.
- The cache-aware gate demonstrably bites: with a deliberately small `ram_budget`, `effective_max_batch`
  drops and throughput **does not** collapse. A run that thrashes instead of throttling is a failed gate.
- Preempt → resume reproduces the same continuation byte-for-byte under `determinism: strict`.
- Chunked prefill: a 32k-token prompt does not starve a concurrent interactive turn beyond the fairness
  cap.

#### The throughput gate, restated

> **Superseded wording.** This gate originally read *"aggregate throughput scales **super-linearly** in
> the disk-bound regime from batch 1 → 4 → 8."* That asks for something the mechanism cannot produce,
> and the arithmetic below shows why. It was replaced after the measurement, not to accommodate a
> disappointing result — the mechanism hit its actual ceiling — but because a gate no correct
> implementation can pass is worse than no gate: it either blocks forever or gets quietly reinterpreted,
> and this one had already been reinterpreted by the tool that reports it.

Let `R_N` be bytes read for a batch of `N` sequences of `T` tokens, and `b_N = R_N / (N·T)` the bytes per
token. In the disk-bound regime time is proportional to bytes, so:

```
TP_N / TP_1  =  [N·T / (R_N/BW)] / [T / (R_1/BW)]  =  N · R_1 / R_N  =  b_1 / b_N
```

**The aggregate-throughput ratio equals the bytes-per-token reduction, exactly.** And since
`b_1/b_N = (rows · top_k) / unique`, that is precisely the **union ratio**. Two consequences:

1. The ceiling is the union ratio, **not `N`**. A bigger batch reads *more* bytes in total (`R_N > R_1`)
   — just fewer per token — so `N · R_1/R_N < N` always. "Super-linear" described an effect no correct
   implementation of this design can exhibit.
2. The union ratio and the throughput ratio are the *same number measured two ways*. If they disagree,
   the disagreement is the finding: either the regime is not disk-bound, or saved bytes are not being
   converted into saved time.

**The gate is therefore three parts, each falsifiable on its own:**

| part | question | passes when |
|---|---|---|
| **mechanism** | does the union reduce work? | `b_1 / b_N > 1` and grows with batch |
| **regime** | is that reduction able to matter? | reads dominate wall time |
| **conversion** | is the reduction actually realised? | `(TP_N/TP_1) / (b_1/b_N)` ≥ 0.5 |

The third is the one the old wording had no room for, and it is the one that catches a real regression:
a mechanism that works and a regime that is right can still fail to produce throughput if the I/O is not
overlapped, and *only* the conversion ratio shows that.

**Measured** (real Qwen3-30B-A3B container, 1 GiB cache, nseq 1 → 8):

| quantity | value |
|---|---:|
| bytes/token reduction `b_1/b_N` | **4.30×** |
| aggregate throughput ratio | **2.65×** (3.40 → 9.02 tok/s) |
| conversion efficiency | **62%** |
| reads as share of achievable bandwidth | **96%** |

Two honest caveats on the 62%:

- The batch-1 baseline gets an unfair advantage. It reads 4489 MiB against 8343 MiB at batch 8, so more
  of it is served from the OS page cache — measured effective bandwidth is 1961 MB/s at nseq=1 versus
  1232 MB/s at nseq=8. The baseline is *faster than the device*, which flatters it and depresses the
  ratio. A cold-cache measurement would score higher; this one is conservative.
- The "reads as share of achievable bandwidth" figure is `observed bytes-per-second ÷ the device
  bandwidth measured at G2`. It saturates and can exceed 100% when the page cache is warm. Above ~90% it
  means "disk-bound", not a literal percentage of wall time.

### The G3 amendment — the debugging view precedes the pretty one

**A minimal text dump of tier occupancy and per-expert heat must exist at G3**, four gates before the
FTXUI panels at G7:

```
$ soma-dump --engine 127.0.0.1:8080
tier   vram      0 experts        0 B
tier   ram    2914 experts   12.0 GiB / 12.0 GiB   (pinned 512)
tier   disk   3230 experts
cache  hit 95.3%  miss 9117  evict 8605  prefetch 5512 hit / 388 wasted
sched  batch 7/12 (limited by expert_cache)  unique 41 / naive 56
layer  0  ████▓▓░░········  hot 12/128   L1 ██████▓░········  hot 19/128
```

Watching expert-load patterns across concurrent sequences is the **primary instrument** for catching
cache thrash. Waiting for polished panels means debugging G3 blind, and G3 is precisely the gate where
the failure mode is subtle: throughput that is merely disappointing rather than obviously broken.

**Speculation is disabled when `batch > 1`.** Grammar-forced drafts stay on.

### G3 status — union and telemetry landed; scheduler still to build

The MoE loop is now **expert-major**. `build_expert_union` groups this step's `(row, expert)` selections
into CSR form and the loop acquires each unique expert **once**, applying it to every row that selected
it. Measured on the tiny Qwen3 fixture (4 MoE layers × 16 experts, top-4), `tests/soma/union_g3.cpp`:

| rows | naive reads | unique reads | ratio | reads/row/layer |
|-----:|------------:|-------------:|------:|----------------:|
| 1    | 16          | 16           | 1.00× | 4.00 |
| 4    | 64          | 28           | 2.29× | 1.75 |
| 16   | 256         | 41           | 6.24× | 0.64 |
| 64   | 1024        | 51           | 20.08× | 0.20 |

At one row the union is **exactly** a no-op, which is the check worth having: a ratio above 1.0 there
would mean the router emitted duplicate experts for a single token.

All seven pre-existing conformance tests still pass unchanged. That matters more than it looks:
expert-major application reassociates the float accumulation for every row, and G0's token-exact
teacher-forced comparison against the `transformers` oracle survived it.

#### Three findings

1. **A working union drives the cache hit rate toward zero, and that is success.** Before the union, the
   LRU absorbed the duplicate `(row, expert)` lookups and reported them as *hits* — the 97.1% figure
   recorded under G2 was mostly counting intra-step duplication. The CSR now removes those before the
   cache is consulted, so what remains is cold-first-touch traffic: misses equal unique reads exactly.
   **The metric moves the wrong way when the system improves.** Bytes per token is the honest measure;
   hit rate is only meaningful across *repeated* steps, where it measures inter-step reuse.

2. **The cache-aware gate's prediction is conservative, by 1.25–1.56×.** The gate sizes `max_batch` from
   a coupon-collector expectation `E·(1−(1−k/E)^rows)`, which assumes uniform expert selection. Real
   routers concentrate, so the formula **overestimates** the unique count:

   | rows | predicted | measured | pred/meas |
   |-----:|----------:|---------:|----------:|
   | 4    | 43.8      | 28       | 1.56× |
   | 16   | 63.4      | 41       | 1.55× |
   | 64   | 64.0      | 51       | 1.25× |

   Overestimating is the safe direction — the gate throttles earlier than strictly necessary. It is
   worth measuring anyway because the unsafe direction is silent: a formula that *under*estimated would
   size `max_batch` above what the cache holds and thrash precisely where the gate exists to prevent it.
   There is throughput being left on the table here, and closing that gap means feeding measured
   per-model router concentration back into the estimate rather than assuming uniformity.

3. **Expert ordering had to be pinned to make output reproducible.** The union emits experts in
   **ascending id**. Any stable order would do; an unstable one (hash iteration order, say) would make
   the float accumulation order — and therefore the low bits of every output row — vary run to run.
   This is a prerequisite for the `determinism: strict` gate later in G3, not a stylistic choice.

#### On the real checkpoint: 35× union saving, stage 3 still passes

Re-running conformance stage 3 against the 18.42 GB Qwen3-30B-A3B container (48 layers × 128 experts,
512 positions, 8 GiB cache):

```
  expert reads  5614 unique of 196608 naive  (35.0x union saving)
  cache         0.0% hit, 5614 misses, 2750 evictions, 16052 MiB read
logit-KL(reference || engine)   mean 0.01972   median 0.00000   p95 0.06184
  top-1 agree   98.2%  (503/512)
stage 3: PASS
```

The KL figures are unchanged from the pre-union run to five decimal places, so the reassociation is
numerically inert at this scale. The hit rate collapsed from 97.1% to 0.0% exactly as finding 1
predicts, while bytes read stayed flat — at 512 rows the expert set is saturated (5614 of 6144 possible
reads), so there is no duplication left for the union to remove that the LRU was not already absorbing.
**The union's win is at moderate batch, not at saturation.**

Wall clock went 685 s → 747 s, about 9% slower. That is a real regression and worth naming: expert-major
order trades expert-weight locality (good, and large) for activation locality (worse). For one expert
the loop now walks a strided scatter across every row that selected it — at 512 rows that is ~4 MB of
hidden state per pass, well past L2. A blocked expert-major loop that tiles rows should recover it; it
is not implemented.

#### The throughput half of the gate is not evaluable on this host

`tests/soma/scaling_g3.cpp` sweeps batch against the real container with a 1 GiB cache:

| nseq | rows | MiB read | KiB/token | unique | union | sec | tok/s |
|-----:|-----:|---------:|----------:|-------:|------:|----:|------:|
| 1 | 8  | 4489 | 574620 | 1570 | 2.0× | 16.3 | 0.49 |
| 2 | 16 | 6247 | 399855 | 2185 | 2.8× | 29.9 | 0.53 |
| 4 | 32 | 7382 | 236253 | 2582 | 4.8× | 57.6 | 0.56 |
| 8 | 64 | 8343 | 133499 | 2918 | 8.4× | 103.4 | 0.62 |

**Bytes per token fell 4.30×** from nseq 1 → 8. The mechanism works exactly as designed.

Aggregate throughput rose only 1.26× for 8× the rows — nowhere near super-linear. The reason is in the
next line of the tool's own output: **implied read bandwidth is 81 MB/s against a device that measured
1230 MB/s at G2, so reads account for roughly 7% of wall time.** This run is compute-bound. The gate
says *"scales super-linearly in the disk-bound regime"*, and this is not that regime, so the gate is
**NOT EVALUABLE** here rather than passed or failed. `scaling_g3` now computes that read fraction and
says so itself, because reporting a bytes win as a throughput win is the easiest self-deception
available at this gate.

The blocker is the kernels, not the scheduler. G1's autotuner already recorded fp32 running **2.2–2.4×
faster than q4_g** — backwards for a format that moves a quarter of the bytes, and a direct statement
that the quantized matvec path is scalar. Until that inverts, every concurrency measurement on this host
measures kernel throughput. **SIMD quantized kernels are therefore a prerequisite for the G3 throughput
gate, not a G6 optimization.** Exit status tracks the mechanism only; the regime is a property of the
host, so failing CI on it would make the result depend on which machine ran it.

#### SIMD quantized kernels — the fp32-beats-q4_g inversion, corrected

`src/soma/kernels_quant_avx2.cpp` implements AVX2+FMA dot products for all four quant formats, behind a
runtime CPUID check (`simd::available()`). The detector deliberately lives in a translation unit built
at the baseline ISA, so the function deciding whether AVX2 may run is never itself compiled with it; the
check includes OSXSAVE/XCR0, because a CPU can advertise AVX2 while the OS declines to preserve YMM.
x86-64 only — aarch64 omits the TU and keeps the scalar path.

Measured by `tests/soma/simd_g3.cpp --bench`, 2048×2048 matvec:

| dtype | scalar | SIMD | speedup |
|-------|-------:|-----:|--------:|
| q4_g | 2.10 ms | 0.24 ms | **8.66×** |
| q6_g | 2.82 ms | 0.57 ms | **4.92×** |
| q8_0 | 1.62 ms | 0.19 ms | **8.56×** |
| q4_0 | 2.14 ms | 0.28 ms | **7.55×** |

The autotuner's verdict inverts, which was the entire point:

| shape | before (scalar) | after (SIMD) |
|---|---:|---:|
| gemv 4096×2048 q4_g | 4.47 GF/s | **35.52 GF/s** |
| gemv 151936×2048 q8_0 | 4.26 GF/s | **44.80 GF/s** |
| gemv 2048×768 q6_g | 3.59 GF/s | **14.95 GF/s** |

> **Correction.** This section first read "q4_g went from 2.4× slower than fp32 to 3.2× faster — the G1
> finding is retired." That was wrong, and wrong in an avoidable way: it compared a *vectorised* q4_g
> kernel against a *scalar* fp32 one and drew a conclusion about the formats. Once fp32 was vectorised
> too (next section), it reached 85.5 GF/s against q4_g's 35.5 — a ratio of **2.41×**, against **2.37×**
> when both were scalar.
>
> **Vectorising both sides left the ratio essentially unchanged.** The gap is the intrinsic cost of
> unpacking sub-byte weights, not an artifact of optimisation level, and it survived an ~8× speedup of
> both sides. The G1 finding stands; what was wrong was calling it a kernel artifact.
>
> This does not weaken the case for quantization, and it is worth being precise about why: GF/s counts
> arithmetic, and quantization does not exist to reduce arithmetic. It exists to reduce **bytes** — q4_g
> moves a quarter of fp32's — and to make the model fit at all. In the memory-bound regime this engine
> is built for, bytes are what set throughput. A format that is 2.4× slower per FLOP while moving 4× less
> data is still the right choice; it just should never have been defended with a FLOP number.

Three things worth keeping:

1. **SIMD is MORE accurate than scalar, not less.** Checked against a double-precision evaluation of the
   same bytes rather than against the scalar kernel: q4_g 1.52e-05 vs scalar's 5.01e-05, q8_0 1.16e-06
   vs 4.95e-06, q4_0 5.49e-07 vs 1.99e-06. An 8-wide accumulator is pairwise summation by accident, and
   pairwise beats sequential. The first version of that check compared SIMD to *scalar* with a fixed
   bound and flagged q4_g as failing — it was measuring scalar's error and blaming the new code.
   q4_g's larger error is inherent to the format: `scale*dot(level,x) + min*sum(x)` is a difference of
   two large cancelling terms, which is a property of asymmetric quantization, not of any kernel.

2. **The autotuner was benchmarking code the engine does not run.** There were two quantized matvec
   implementations — `soma::matvec` on the forward's path, and `q_fused<>` inside the kernel registry —
   and only the first was optimized at first. `kernel_choice` would have gone on faithfully recording
   the ranking of an implementation nothing executes. The SIMD kernels are therefore registered as
   ordinary *candidates* (`q4_g.simd` and friends), so the autotuner measures the real thing and the
   table stays meaningful.

3. **"Candidates must produce the same numbers" had to be read precisely.** The registry's rule is
   agreement to the autotuner's 1e-4 relative tolerance, not bit-identity — otherwise vector kernels are
   inadmissible for a difference far smaller than the quantization they are decoding. The rule protects
   against a candidate computing a *different function* (a permuted row, a dropped tail), and 1e-4
   catches that with orders of magnitude to spare.

End to end on the real container, batch sweep at 1 GiB cache: **103.4 s → 58.8 s** at nseq=8, aggregate
throughput **0.62 → 1.09 tok/s**, read fraction 7% → 12%.

#### SIMD fp32 kernels

The quantized work moved the bottleneck onto fp32 rather than removing it, so `kernels_f32_avx2.cpp`
followed. Four independent accumulators per reduction, not one: a single accumulator serialises the loop
on FMA *latency* instead of *throughput*, which is most of the available speedup and the difference
between vectorising and merely using vector registers.

Three of these are not matvec, and two are not SIMD at all:

- **`axpy` was a loop body inside `gqa.cpp`.** Attention runs it once per (query, key, head), making it
  `O(T² · heads · head_dim)` — the same order as the score dot product, and together they dominate the
  forward at long context far more than the projections do. It is now a `soma::f32` primitive so the
  dispatch lives in one place.
- **RoPE recomputed `pow`/`cos`/`sin` `n_heads` times over.** They depend only on the frequency index,
  not the head. Hoisting is not vectorisation and not an approximation — the identical values are
  computed once instead of 32 times, and three transcendentals per element dwarf the rotation itself.
- **`softmax`'s `exp` was deliberately left scalar.** A polynomial approximation is several times faster,
  but it changes the *values* rather than their summation order. Softmax feeds the router, whose output
  decides **which experts fire** — that is a semantic change, and it needs its own measurement against
  the oracle rather than a quiet substitution inside a SIMD pass.

`f32.simd` measures **85.54 GF/s** against `f32.unroll4`'s 10.81 — 7.9×. And the G0 gate got *more*
accurate, not less, because an 8-wide accumulator is pairwise summation by accident:

| fixture | max logit diff before | after |
|---|---:|---:|
| Mixtral-8x7B | 1.79e-06 | **1.55e-06** |
| OLMoE-1B-7B | 1.31e-06 | **1.07e-06** |
| Qwen3-30B-A3B | 1.01e-06 | **9.54e-07** |

Greedy remains token-exact on all three.

End to end, nseq=8: **58.8 s → 24.4 s**, aggregate **1.09 → 2.62 tok/s**, read fraction **12% → 28%**.
Cumulative against the pre-SIMD baseline: **103.4 s → 24.4 s, 4.2×**.

Two test weaknesses surfaced and were fixed:

1. **`conformance_g0` exited 0 when it found no fixtures at all.** Pointed at the wrong directory it
   printed "0 passed, 0 failed, 0 skipped" and reported success — a gate that has evaluated nothing
   looks exactly like a gate that has passed, right up until someone moves the fixtures. It now returns
   2 when nothing was evaluated.
2. **The `m % 4` tail check was judged against a float threshold.** It passed at 6.79e-04 against a
   1e-3 bound, but the error it was measuring was *cancellation* at small `k`, not the kernel — so the
   verdict sat one RNG seed from meaningless. The structural claim is now tested structurally: with
   every input 1.0, `y[r]` must equal `k` **exactly** for every remainder combination. A dropped row
   reads as 0, a dropped tail element as `k-1`.

`scaling_g3`'s diagnostic no longer names a culprit — two successive versions of that text named the
then-current bottleneck and both were obsolete within a day, because fixing the named bottleneck is what
the tool provokes. It reports the regime and defers attribution to a fresh autotune.

#### Where G3 performance stands

Four passes, same measurement each time — the real container, 1 GiB cache, nseq=8 (64 rows):

| after | nseq=8 | aggregate tok/s | reads as % of wall time |
|---|---:|---:|---:|
| batch union only | 103.4 s | 0.62 | 7% |
| + SIMD quantized kernels | 58.8 s | 1.09 | 12% |
| + SIMD fp32 kernels | 24.4 s | 2.62 | 28% |
| + multi-threading | **14.8 s** | **4.31** | **46%** |

**7.0× end to end, and the read fraction went 7% → 46%.** That last column is the one that matters:
every pass moved the engine closer to the regime the design is actually about, where the batch union's
4.30× reduction in bytes/token can turn into time rather than being masked by compute.

The G3 throughput gate is still **NOT EVALUABLE** — 46% is not "reads dominate" — but it is now marginal
rather than hopeless. Wall-clock here varies by a second or two with page-cache warmth; the read
fraction is the stable number.

#### Multi-threading the forward

`include/soma/threading.hpp` + `src/soma/threading.cpp`. One rule shapes the whole design:

> **Every parallel region partitions OUTPUT elements, never input ranges.** Each output is computed
> start to finish by one worker, in the order a single thread would have used.

That makes results **bit-identical regardless of thread count**, which is not a nicety: splitting a
reduction and combining partials would make output depend on the host's core count and the OS
scheduler, putting `determinism: strict` permanently out of reach and making every conformance number a
property of the machine that produced it. The direct consequence is that reductions over a *single*
output — a dot product, an rmsnorm sum — are deliberately **not** parallelised.

`tests/soma/threading_g3.cpp` asserts it against a real forward rather than arguing it from the source,
re-executing itself with `SOMA_THREADS` set because the pool reads that once at first use. FNV-1a over
the raw logit bytes, not a tolerance — a comparison with any epsilon would pass on precisely the drift
it exists to catch:

```
SOMA_THREADS    workers   logit digest
1               1         17511656933821331073
2               2         17511656933821331073
3               3         17511656933821331073
8               8         17511656933821331073
(default)       32        17511656933821331073
```

Three parallel regions: matvec over output rows (fp32 and quantized), attention over query positions,
and the MoE loop over each expert's row group. `ws.scores` and the four FFN scratch buffers became
per-worker; they were shared, which is exactly the scratch that turns a correct-looking parallel loop
into a silent race.

##### Two bugs worth recording, because both passed every test

**1. The chunking silently disabled the pool.** `min_chunk` was derived from `kParallelMacThreshold`,
giving 512 rows per chunk at k=2048. `parallel_for` requires `n ≥ min_chunk × 2` before it splits, so
every matvec with fewer than 1024 output rows ran **serially** — which is all of the expert
projections, i.e. most of the engine. Everything passed, every number was identical, and 32 cores
delivered **1.2×**. The fix is a separate `kChunkMacs` (~128 K MACs, roughly 7 µs of work): "smallest
chunk worth dispatching" and "total work worth parallelising at all" are different questions.

**2. A load-dependent race in the pool itself.** `parallel_for` originally waited until an *element*
counter reached `n`. That let a straggler still inside `drain()` — between its epoch check and its next
`fetch_add` — observe the *next* region's freshly reset cursor, process a chunk against stale
bookkeeping, and desynchronise the counters so the caller spun forever. **It ran clean at 1, 2, 3, 4 and
8 threads and hung at 16**, which is how this class of bug presents: correct-looking and load-dependent.
The pool now uses a **fixed team** — all `n-1` workers acknowledge every region whether or not they got
a chunk — so setup for region N+1 is provably later than every worker's exit from region N.

Also: the spin loop used `std::this_thread::yield()`, a syscall that enters the scheduler. With 32
workers polling it the scheduler becomes the bottleneck. It now uses `_mm_pause`, which also releases
SMT resources to the sibling thread — relevant because 32 workers on 16 physical cores means every pair
shares a core.

##### Scaling, and its ceiling

| SOMA_THREADS | nseq=8 |
|---:|---:|
| 1 | 31.1 s |
| 4 | 20.0 s |
| 8 | 18.8 s |
| 16 | **16.9 s** |
| 32 | 17.9 s |

**1.84× on 16 cores**, regressing at 32 (SMT contention). That is well short of linear, and the reason
is structural rather than a tuning failure: at nseq=8 there are 64 rows over ~60 unique experts per
layer, so the MoE region has only **~8.5 rows to distribute** — half the workers idle, and the barrier
still waits for all of them. Attention and the projections parallelise fine; the MoE loop, which is
where the work is, is starved of width.

Parallelising across *experts* instead would supply width but two experts write the same output row, so
it needs either a lock or per-thread accumulation and a reduction — and the reduction changes summation
order, which forfeits the bit-identity above. **The row-within-expert ceiling is the price of
determinism at this batch size.**

That explanation makes a falsifiable prediction — scaling should improve with batch, because rows per
expert grows linearly while unique experts saturate — so it was tested rather than asserted. At 32
tokens per sequence (256 rows instead of 64):

| rows | 1 thread | 16 threads | speedup |
|-----:|---------:|-----------:|--------:|
| 64  | 31.1 s | 16.9 s | 1.84× |
| 256 | 95.7 s | 32.3 s | **2.96×** |

Confirmed. The ceiling is the MoE region's width, not the pool, and it lifts as batch grows — which is
the regime the engine is built for. It also means **thread scaling and the batch union reinforce each
other**: the same larger batch that makes the union save more reads also gives the pool more rows to
spread.

#### AVX-512 — a third tier, and a lesson about where width helps

`src/soma/kernels_avx512.cpp`, behind `simd::tier()`. The AVX2 kernels moved into `soma::simd::avx2`,
the new ones into `soma::simd::avx512`, and the public `soma::simd::*` names became dispatchers compiled
at the baseline ISA. CMake probes whether the compiler can emit AVX-512 and builds the TU if so —
dispatch is runtime, so the binary still runs on hosts without it.

Detection checks XCR0 `0xE6` (opmask + both ZMM halves), not just the AVX2 pair. Executing 512-bit code
on a kernel that does not preserve that state corrupts vector registers at arbitrary points.

**In isolation, the quantized kernels gain 1.5–1.7×** — q4_g 1.59×, q8_0 1.49–1.55×, q4_0 1.71×. The
autotuner's q4_g goes **35.5 → 54.6 GF/s** and q8_0 **44.8 → 71.9**.

**fp32 gains nothing, and is dispatched back to AVX2.** `simd_g3 --bench` puts AVX-512 at 0.97–0.99× of
AVX2 for a 2048×2048 fp32 matvec, reproducibly. The split follows from what each family does per byte:
the quantized kernels spend most of their instructions *unpacking* sub-byte levels — shift, mask, widen,
convert — pure ALU work that scales with vector width. An fp32 matvec has no unpacking; it streams
weights and issues one FMA per element, so it is already limited by how fast weights arrive, and a wider
register does not make memory faster. **The tier is therefore per kernel family, decided by
measurement, not a blanket "use the widest thing available."** The AVX-512 fp32 kernels are kept and
still exercised, because that conclusion is a property of this silicon's memory system.

`SOMA_SIMD_TIER=scalar|avx2|avx512` (capped by what the CPU supports) makes the tier an A/B on one
binary. End to end, alternated to rule out drift:

| tier | nseq=8 | reads |
|---|---:|---:|
| avx2 | 16.2 s, 16.4 s | 42%, 41% |
| avx512 | **15.4 s, 15.3 s** | 44%, 44% |

**~6% end to end** from kernels that are 1.5–1.7× faster in isolation. That ratio is the whole story of
this pass: the forward is already ~42% I/O and its compute is spread over 32 threads, so accelerating a
*subset* of the compute by 1.6× moves the total very little. Kept because it is repeatable, free at
runtime, and grows in value as the other bottlenecks fall — but it is not the lever that reaches the
disk-bound regime, and the microbenchmark number should never be quoted as if it were the system number.

Two measurement mistakes were made and corrected here, both the same shape as earlier ones:

1. **The AVX-512-vs-AVX2 accuracy check compared the two kernels to each other**, and flagged f32 matvec
   as failing at 1.16e-03. That was one row out of 2048 landing near zero and cancelling — neither
   kernel was wrong. Comparing both to a double-precision evaluation shows AVX-512 is the *more*
   accurate of the two (5.21e-04 vs 8.56e-04).
2. **The tier benchmark was single-shot** and reported q4_0 at 1.72× and then 0.55× on successive runs of
   the same binary. Best-of-5 made it stable — and also revealed the fp32 parity that a noisy
   measurement had been hiding behind an apparent 1.65× win.

A third claim was withdrawn rather than corrected: an earlier draft justified the fp32 dispatch decision
with an autotuner swing from 85.5 to 75.2 GF/s. That figure varies run to run on unchanged code and is
not evidence of anything; the stable microbenchmark is what supports the decision.

#### Overlapping expert I/O with compute

`MemoryHierarchy::prefetch()` already existed but was **synchronous** — it called `fetch_locked` on the
caller's thread, which moves the read earlier without overlapping anything. Real overlap needed a
background loader, so `Impl` gained a request queue and (by default) two loader threads.

Two threads rather than one per core: loaders do no arithmetic, they wait on the device, and past the
point where the queue stays non-empty more of them only add cache-mutex contention and seek pressure.

**The two prefetches are different in kind**, which is why `prefetch_ahead()` is a separate entry point
rather than a flag on the existing one:

| | `prefetch()` | `prefetch_ahead()` |
|---|---|---|
| source | router lookahead | the step's expert union |
| can be wrong | yes | **no** — these experts are already chosen |
| gated on | measured per-layer recall | nothing; only *when* the read happens is in question |

The union is the one place in the engine where prefetching needs no prediction at all: it already names
every expert the layer will touch, in the order it will touch them.

Two correctness details that are easy to get wrong and expensive to find later:

- **A `loading` flag per slot.** Without it, an `acquire()` arriving while a loader is mid-read finds the
  slot non-resident and starts a *second* read of the same bytes. Both complete correctly —
  `fetch_locked` already handled the double-insert — but doing the I/O twice concurrently is worse than
  not prefetching. `acquire()` now waits on the in-flight read instead of racing it, which is also what
  makes a slightly-too-short prefetch distance still pay: the caller waits for the *remainder* of a read
  that started earlier, not for a whole new one.
- **`close()` joins the loaders before replacing `Impl`.** It previously just reassigned the `unique_ptr`;
  with threads holding a raw pointer to the old one, that is a use-after-free that would only appear
  under load.

Depth is bounded by the cache, not by taste: queue more experts than `cap_per_layer` allows and the
prefetch evicts entries it has already fetched but not yet used, converting a latency win into extra
I/O. `SOMA_PREFETCH_DEPTH` sets it, so "is this paying?" is an A/B on one binary rather than a
comparison across two builds.

##### An LRU bug the A/B caught

First measurement with prefetch on: **21% MORE bytes read** (8343 → 10130 MiB) and a net slowdown.
`fetch_locked` never set `last_used` — on the acquire path the caller had already done it. A prefetched
slot therefore kept whatever timestamp it had (0, if never touched), making it the *oldest* entry in the
cache and the first victim of the very next `make_room`. Experts were being fetched and evicted before
the loop reached them. Touching the LRU clock on prefetch completion fixed it; bytes read returned to
baseline and the run got 17% faster.

That is exactly what a prefetch bug looks like from the outside — more I/O, not less — and it is only
visible if bytes read is reported next to wall time.

##### The race: a shared file position in `ExpertStore::read`

Prefetch was default-off for a while because it made the forward non-deterministic —
`unique_expert_reads` is a pure function of routing and it varied run to run (2898 / 2902 / 2920
against a stable 2918), meaning some expert's *weights* were occasionally wrong.

The cause was not in the MoE loop, the cache, or the `loading` flag. It was in `ExpertStore::read`:

```cpp
auto& f = *impl_->shards[loc.shard];   // a shared std::ifstream
f.seekg(loc.offset);
f.read(dst, loc.length);
```

**A file stream has ONE position.** Two threads interleave as `seek(A)`, `seek(B)`, `read(A)` — and the
first thread receives expert B's bytes. The read *succeeds*; there is no error to check. Wrong weights
flow into that layer's output, change the next layer's routing, and surface as a model that answers
slightly differently on every run.

It had been latent since G2: `store->read` was only ever called from the serial part of the MoE loop, so
one thread at a time. The loader pool made it concurrent for the first time and exposed it.

The fix is stateless positional reads — `pread` on POSIX, `ReadFile` with `OVERLAPPED` on Windows —
which carry the offset in the **call** rather than in the handle, so concurrent reads cannot interfere.
No lock, no per-thread handles. `bytes_read` also became atomic; it was a plain `+=` from every reading
thread.

That fix alone was worth **1.9×** on wall clock (18.05 s → 9.4 s at nseq=8), independently of
correctness: `ifstream` copies through its own buffer, while `ReadFile` writes straight into the
destination.

##### The gate, and a correction to how it was worded

With reads fixed and prefetch on by default, at nseq=8 on the real container (alternated):

| prefetch | nseq=8 | reads as % of wall |
|---|---:|---:|
| off | 9.2 s, 9.6 s | 71–74% |
| depth 3 | **6.6 s, 7.5 s** | 91–103% |

`scaling_g3` now reports:

```
bytes/token fell 4.30x from nseq=1 to nseq=8
aggregate tok/s  3.40 -> 9.02  (2.66x for 8x the rows)
implied read BW  1176 MB/s of 1230 MB/s device  -> reads are ~96% of wall time
G3 throughput gate: PASS
```

**The engine is finally disk-bound** — the regime the whole design is about, reached after four passes
that each moved the read fraction: 7% → 12% → 28% → 44% → **96%**.

The gate this is measured against was **restated** — see "The throughput gate, restated" above. The old
wording asked for super-linear scaling, which the arithmetic shows no correct implementation can produce.
`scaling_g3` now reports the three parts separately:

```
mechanism   bytes/token falls          4.30x   PASS
regime      reads dominate runtime     95%     PASS
conversion  realised / available       2.66x of 4.30x = 62%   PASS

G3 throughput gate: PASS
```

##### The determinism gate that was missing

`tests/soma/threading_g3.cpp` exercises a *resident* model — no ExpertStore, no cache, no loader
threads. The entire streaming path was unguarded, which is why a data race surfaced as an odd column in
a performance table rather than as a failing test. That is luck, not process.

`tests/soma/streamed_determinism_g3.cpp` closes it: the streamed forward under a deliberately tiny
cache (4 experts, so eviction and re-reads are constant), across thread counts and prefetch depths,
compared by FNV-1a over the raw logit bytes. It also reports `unique_expert_reads` as an independent
witness — if that drifts, routing changed, which means weights were wrong rather than merely summed in a
different order.

One caveat worth stating: the gate passes, but its *sensitivity to this specific race* on the tiny
fixture is unverified — I did not re-introduce the bug to confirm it fails. The strong evidence for the
fix is the real-model run, where three consecutive runs now produce byte-identical reads and unique
counts (4489/6247/7382/8343 MiB, 1570/2185/2582/2918) against values that previously varied.

##### A flaky assertion prefetch exposed

Enabling prefetch made `streaming_g2` fail intermittently — 2 runs in 30 — on
`check(cs.misses > 0, "the streamed run actually paged")`, reporting **"0 misses, 43 evictions"**.

Not a regression. When the loader fetches an expert before the compute thread asks for it, the acquire
finds it resident and counts a **hit**. A perfectly-prefetched run therefore has zero acquire-misses
while paging as hard as it possibly can, and which of the two happens is a timing race — hence the
intermittency rather than a clean failure.

This is the same shape as the hit-rate inversion the batch union caused: a metric that moves the wrong
way when the system improves, wired into an assertion. The check now uses **bytes read and evictions**,
which are attributable to the streaming path regardless of which thread issued the read. 0 failures in
40 runs after the change.

#### The step-major scheduler

`include/soma/kv_cache.hpp`, `src/soma/scheduler.cpp`, and a batched-decode attention path.

**The KV cache was the blocker, not the scheduler.** Until now `forward_f32` was teacher-forced — it
recomputed attention over the whole prefix on every call, which is right for conformance and useless for
serving. Batching one decode row from each of N sequences requires each row to attend over *its own*
history, so the history has to be stored per sequence rather than recomputed per call. Everything else
followed from that.

The shape that makes it work is `KvRow`: **a row, not a sequence**. Rows from different sequences, at
different positions, with different visible lengths, sit side by side in one forward and differ only in
that struct. Prefill rows and decode rows are the same type — the scheduler never branches on which it
is, it only decides which token a sequence contributes.

`f32_attention_kv` is a second backend entry point rather than a flag on the first, because the two
differ in what they may assume: `attention` owns the whole sequence and can exploit that; the batched
form owns exactly one position per row and must not. One forward body serves both (`forward_impl`), so
the batched path cannot drift from the path the conformance ladder validates.

**The gate that matters, and it passes:**

```
batched output is identical to solo output          OK
```

A sequence generated alongside three others produces exactly the tokens it produces alone. Without that,
every throughput number describes a different model than the one G0–G2 validated.

Also verified: all four sequences share one step; some step carries **both** prefill and decode rows
(ragged, not phase-separated); and the union across *independent* sequences reaches **1.49×** (naive 64,
unique 43) against 1.00 at batch 1 — which is the case the union was designed for, since adjacent tokens
of one prompt are correlated and flatter it.

##### Two bugs the test caught

1. **The cache-aware gate was inverted.** `compute_gate()` read `cap_per_layer == 0` as "unbounded
   cache" and returned the full `kv_slots`. But `cap_per_layer` is `budget / (expert_bytes × n_moe)`, so
   zero means the cache cannot hold even one expert per layer — *maximum* pressure. The tightest possible
   cache therefore produced the widest possible batch, reversing the gate exactly where it was needed.
   A one-line confusion between "no limit recorded" and "limit is zero"; measured gate now drops to 1
   under a 2-expert cache and 8 under a roomy one.
2. **The test sampled the wrong step.** `SchedulerStats` reports `*_last_step`, which is what a live
   telemetry feed wants and the wrong moment for a test: sequences retire at different times, so the
   final step of a four-sequence run has one row, and reading it reported a union ratio of exactly 1.00
   — the signature of the union being *broken*. Per-step figures now come from the widest step, and
   "did prefill and decode ever mix?" is accumulated across the run rather than sampled from one step.

##### Deliberately not built yet

The header's full surface is not implemented, and the parts left out are named rather than stubbed
silently:

- ~~KV checkpoint store, `preempt()`, `resume()`~~ — **built**, see below.
- ~~Chunked prefill and the fairness cap~~ — **built**, see below.
- ~~Sampling~~ — **built**, see below.
- **`Scheduler::open(ModelState&)`** — returns `Unsupported` pointing at `open_f32()`. The
  `ModelState`-shaped entry belongs with the engine at G5; the fp32 path is what every conformance gate
  is expressed against, so that is what G3 schedules.

#### KV checkpoints and preemption

`src/soma/kv_checkpoint.cpp`. One on-disk format, three callers — scheduler preemption, warm
conversation reopen, cluster slot suspend/restore — because they are the same operation, and unifying
them is *why* preemption is nearly free rather than a feature of its own.

**The gate passes:** preempting mid-generation and resuming reproduces the continuation **byte-for-byte**,
checked at four different points (steps 1, 3, 5, 9) so that a checkpoint taken during prefill and one
taken mid-decode both get exercised. An off-by-one in how many positions are written survives a single
well-chosen preemption point.

Three design decisions worth recording:

- **Live positions are written, not the raw buffer.** Serialising the cache array would bake `max_ctx`
  into the payload, making every checkpoint hostage to the config it was written under — and the cluster
  case (suspend here, restore there) is exactly where that bites. A checkpoint taken at ctx 128 restores
  into an engine configured for 4096.
- **Write-to-temp-then-rename.** A checkpoint is written under memory pressure and read after a crash or
  a migration. A half-written file that still passes the magic check is worse than no file.
- **`preempt()` releases the KV buffer.** Preemption that keeps the memory it was called to reclaim is a
  no-op with extra steps. The sequence stays in the map so its prompt, position and sampler survive, and
  it is removed from the ready queue so nothing can schedule a row with no cache to attend over.

The format tag is **backend-owned** and reached through `resolve_arch_backend`, so this core TU never
names an architecture — the seam check enforces it.

##### The gate that was structurally present and vacuous

The refusal tests passed on the first run. They were also meaningless: `load_f32_model` never populated
`arch.arch_hash`, so every checkpoint carried the empty string and every `arch_hash` comparison in the
system — checkpoints, containers, the registry — was comparing `""` against `""` and accepting
everything. The forged-mismatch test still went green, because `""` differs from a hash of zeros.

`compute_arch_hash` is now called at load, and the test asserts the value is **populated** before
asserting that a mismatch is refused. A gate that is present in the code and vacuous at runtime is worse
than an absent one: it reads as covered.

#### Chunked prefill and the fairness cap

A prefilling sequence now contributes MANY rows per step at consecutive positions, so a 512-token prompt
is one step rather than 512. The step loop tracks a row RANGE per sequence rather than one row.

**Two caps, because they prevent different failures:**

| cap | prevents |
|---|---|
| `max_prefill_rows_per_seq` | one sequence taking the whole step — a 32k prompt starving an interactive turn beside it |
| `prefill_chunk_tokens` | N sequences each under the per-seq cap still multiplying into an unbounded forward. Row count drives workspace size and attention cost; without a shared budget, 8 × 256 is one 2048-row step and every decode row waits behind it |

Decode rows are exempt from both: they are one row each, and they are the latency chunking exists to
protect.

**Why rows of one chunk may attend to each other.** Rows at positions `p..p+k` of the same sequence need
to see each other exactly as they would across `k` separate steps. That is sound because
`f32_attention_kv` writes every row's K/V into its cache *before* any row reads — the write pass and the
read pass are separate loops. Had they been interleaved, a chunked prefill would produce different
numbers from an unchunked one, silently.

Measured on a 40-token prompt sharing a batch with three short ones:

| chunk | steps | widest prefill |
|------:|------:|---------------:|
| 4 | 17 | 7 rows |
| 8 | 12 | 11 rows |
| 64 | 8 | 46 rows |

**Output is identical across all three chunk sizes.** That is the claim that matters: chunk size is a
scheduling decision and must not be visible in the model's output. And with a small chunk, the long
prompt prefills while the short sequences decode — mixed prefill/decode steps, which is the fairness cap
doing its job.

Side effect worth noting: the union ratio across concurrent sequences rose from **1.49× to 3.20×**,
because chunked prefill puts many more rows in a step and the union's payoff grows with row count.

##### A test that was passing for the wrong reason

`scheduler_g3` asserted "some step carried both prefill and decode rows" — and it started failing the
moment chunked prefill landed, because with 1–4 token prompts and a 512-token budget every sequence now
finishes prefill in step 1, leaving no step in which the two could coexist. The old assertion was
quietly relying on prefill being SLOW. It is now reported there and asserted in section 5, against a
long prompt where mixing is the behaviour under test rather than an accident of timing.

#### Sampling

`include/soma/sampler.hpp`, `src/soma/sampler.cpp`. Temperature, top-k, top-p, min-p, repetition and
presence penalties.

**The RNG state is per sequence, and that is the whole design.** A single engine-wide RNG works
perfectly in every single-sequence test and is wrong the moment two sequences share a step: each draw
would depend on how many neighbours drew first, making a sequence's output a function of who else is on
the server. It is invisible until there is concurrency, and the batched-equals-solo gate catches it —
which it did not have to, because the state was put in the right place first.

The PRNG (splitmix64) and the float conversion are **specified here rather than taken from `<random>`**:
`std::uniform_real_distribution` is not required to produce identical values across implementations, and
a seed is part of a request. A request that replays differently on another host is not reproducible.

Stage order is load-bearing and documented at the call site: penalties on **raw** logits (so a penalty
means the same thing at every temperature), then temperature, then top-k, top-p, min-p, then the draw.
Ties break toward the lower token id at every stage.

Three details that would each have been a quiet bug:

- **Greedy consumes no randomness.** `temperature <= 0` returns the argmax without touching the RNG, so
  toggling temperature mid-stream does not shift the sequence that follows.
- **Penalising a negative logit divides the wrong way.** `logit / penalty` moves a negative value
  *toward zero* — i.e. up — rewarding exactly the token being suppressed. The sign branch is tested
  against negative logits specifically.
- **Degenerate settings still return a token.** `min_p` above 1.0 empties the candidate set; the floor of
  one survivor keeps that from returning token 0, which would read as the model emitting padding.

`tests/soma/sampler_g3.cpp` checks the structure and then checks **the distribution**, which the
structural tests cannot substitute for: a sampler drawing from a subtly wrong distribution returns
plausible in-vocabulary tokens, favours high-probability ones, and passes every assertion about shape.
200 000 draws against `{0.5, 0.3, 0.2}` land within **0.0006** — roughly ten standard errors of headroom,
and tight enough that an off-by-one in the cumulative walk (which shifts a whole category's mass) cannot
hide.

One limitation worth stating: `rng_state` lives in the scheduler's in-memory `Seq`, not in the KV
checkpoint. In-process preempt/resume preserves it, so the byte-identity gate holds; a cross-process
restore would resume with a fresh RNG stream. That is a gap in the checkpoint format, not in the
sampler, and it belongs with the `SeqState` overload at G5.

#### The row-tiled expert loop

The union's remaining cost, finally paid off. Expert-major order reads each expert **once from disk** —
the point of the whole design — and then, applying it one row at a time, re-read all ~2.9 MB of its
weights **from memory for every row that selected it**. That was the ~9% regression recorded above.

`matmul_tiled` inverts the loop: weights outermost, inputs innermost. One weight row is held in L1 and
applied to all rows in the tile before moving on, so the weights stream once per *tile* rather than once
per row. Inputs are gathered into a contiguous buffer first — `tile × d_model` floats, 64 KB at tile 8,
against 2.9 MB of weight traffic saved per additional row, and it turns three scattered strided walks
into three dense ones.

Tile size is 8: large enough to amortise the weight traffic, small enough that the gathered inputs and
three intermediates (~200 KB at these dimensions) do not evict the weights they exist to reuse.

**Bit-identical**, and not by luck — every output is still one dot product accumulated in one order.
Only which operand stays in cache changed. G0's token-exact oracle comparison confirms it, and the
parallelism simply moved inside `matmul_tiled` (over weight rows, still disjoint outputs), so the
threading determinism property is untouched.

| | nseq=8 | aggregate tok/s |
|---|---:|---:|
| per-row | 6.6 s, 7.1 s | 9.02–9.69 |
| tiled | **5.7 s, 5.9 s** | **10.84–11.28** |

About **15%**, which is consistent with recovering the 9% the union cost plus the memory traffic the
per-row loop was spending on top of it.

> **Measurement caveat.** The reported *conversion* ratio swung between 77% and 55% across the two runs,
> because its batch-1 baseline moved (3.39 vs 4.61 tok/s) while nseq=8 stayed put. At nseq=1 the run
> reads 4489 MiB — small enough that page-cache warmth dominates. The nseq=8 column is the stable
> comparison; the conversion figure should be read across several runs or against a cold cache, not from
> a single one.

#### The G3 amendment is satisfied

`union_g3` emits the required text dump — tier occupancy, cache counters, and a per-layer expert heat
grid — and asserts its contents rather than merely printing them.

#### Still outstanding at G3

The union and the telemetry dump are the measurable core; the scheduler around them is not built. Not
yet done, and not claimed:

- **The scheduler itself** — step-major loop over ready sequences, ragged batch assembly, the three
  state tiers as separate types. The union is currently exercised through the teacher-forced forward,
  where the "batch" is one sequence's tokens rather than rows drawn from several sequences. The
  arithmetic is identical; the multi-sequence plumbing is not there.
- **KV checkpoint store and preemption** — so the preempt → resume byte-identity gate is untested.
- **Chunked prefill and the fairness cap** — untested.
- **`effective_max_batch` under a deliberately small `ram_budget`** — the formula is validated above, but
  the "throttles instead of collapsing" behaviour needs the scheduler to demonstrate.
- **The last stretch into the disk-bound regime.** Reads sit at ~44% of wall time after vectorising both
  kernel families, threading the forward, and adding the AVX-512 tier. Each pass has bought less than
  the one before, and AVX-512 — expected to be the big mechanical win — returned ~6%. The remaining
  levers are no longer kernel-shaped:
  - **MoE width**, which is the scheduler's job: batching across sequences is what supplies rows per
    expert, and that is the same axis that makes the union save more. It is the one lever that improves
    both halves of the gate at once.
  - **Overlapping I/O with compute.** Expert reads are currently issued serially inside the union loop,
    so the ~44% is *waiting*, not throughput. Prefetching the next unique expert while the current one
    is being applied would convert most of it — and would make the engine read-bound in the useful
    sense rather than the idle one.
  - A **vectorised `exp`** for softmax, which needs its own oracle measurement since it changes values
    rather than their order.

  `scaling_g3`'s read fraction remains the number that says whether any of it worked.
- ~~Blocked (row-tiled) expert-major loop~~ — **built**, see below.

---

## G4 — second architecture through the seam

### G4 status — IR admits DeepSeek; the backend is next

The IR needed **no change at all**. `MlaSpec` (kv_lora_rank, q_lora_rank, qk_nope/rope_head_dim,
v_head_dim) and the RoPE spec's `mscale`/`mscale_all_dim` were co-designed against DeepSeek during the
design pass, and DeepSeek-V2-Lite's config maps onto them without a schema extension. That was the
point of writing `arch.json` for both families on paper before any code existed, and it held.

Three findings so far, all of which the second architecture existed to surface:

1. **The MHA-collapse rule misclassified MLA.** The adapter read
   `family = (n_kv_heads == n_heads) ? Mha : traits.attention`. DeepSeek-V2-Lite has
   `num_key_value_heads == num_attention_heads == 4`, so an MLA model was classified **MHA** and handed
   to the GQA backend. The collapse is a *GQA-family* shortcut and is now guarded on the family. It
   failed loudly here because every tensor name was wrong — but a config whose shapes happened to line
   up would have run and produced wrong numbers.

2. **`head_dim` is a GQA concept.** MLA splits a query head into `qk_nope ++ qk_rope` and its value head
   is a *different width* (`v_head_dim`). One `head_dim` for both is an assumption the first family
   never challenged. The adapter now derives it from the two halves and sets `rope.partial_dim` to the
   rope half, since only that part is rotated.

3. **The seam check rejected my own first attempt.** Guarding the loader with
   `if (family == Mla || family == MlaDsa)` put an architecture's name in core code, and
   `check_seam.py` failed the build. It was right to: "which families can this loader bind weights
   for?" is exactly the knowledge the core must not hold, and encoding it that way means a third
   architecture edits a core line. The check now asks the **backend registry** — `resolve_f32_backend()
   == nullptr` — which answers the same question and stays true as backends are added.

The ladder is green with an accurate status rather than a misleading one: DeepSeek and Moonlight report
*"no fp32 backend for attention family mla"* instead of
*"model.layers.0.self_attn.k_proj.weight not in checkpoint"*, which named a symptom and sent the reader
hunting for a broken checkpoint.

**What remains, and the honest read on the gate.** The G4 gate is *"zero core diffs — only `arch/` TUs
added."* Strictly, that is already not met: `arch_ir.cpp` gained the DeepSeek mapping (it is the
allow-listed adapter, so this is expected and the seam check permits it), and the family-guard fix was a
core correction. Still ahead:

- `src/soma/arch/mla.cpp` — compressed KV, partial RoPE, YaRN `mscale` on the softmax scale,
  `kv_a_layernorm`, shared experts.
- ~~`F32LayerWeights` is GQA-shaped~~ — **resolved: opaque per-layer payload.**

#### The per-layer payload — the seam's real shape

`F32LayerWeights` held `q_proj`/`k_proj`/`v_proj`/`o_proj` and `q_norm`/`k_norm`. That is not "a layer's
attention weights", it is "a **GQA** layer's attention weights", and the first architecture could not
reveal the difference. MLA has none of those tensors; it carries `kv_a_proj_with_mqa`,
`kv_a_layernorm` and `kv_b_proj`.

The easy fix is to widen the core struct to the union of both families. It works, and it means every
future architecture edits a core type — precisely the coupling the seam exists to prevent. So the core
now holds `ArchLayerPayload`: a `void*` and a deleter, allocated and destroyed by the backend, **never
inspected by the core**. `soma::arch::gqa::F32AttnWeights` moved into `gqa.hpp` where it belongs.

The inversion that makes it work is the bind seam. Two halves, neither needing the other's knowledge:

- the **loader** knows how to read and quantize a *named* tensor — `bind_layer_f32`,
  `bind_layer_weight`, exposed on `LayerBindCtx`;
- the **backend** knows *which names exist* — `F32Backend::bind_layer` fills its own payload.

Without that split the core would need a list of every family's tensor names, which is the same coupling
relocated to a different file.

Verified inert: all 14 tests pass and the G0 logit diffs are **unchanged to every digit** (Mixtral
1.43e-06, OLMoE 1.07e-06, Qwen3 9.54e-07). A refactor of where weights live should move no numbers, and
this one moved none.

#### `src/soma/arch/mla.cpp` — running, not yet correct

Written and registered: compressed KV through the `kv_lora_rank` latent, `kv_a_layernorm` before
expansion, per-head K-nope ++ V from `kv_b_proj`, the shared MQA-style RoPE segment, YaRN inverse
frequencies, and DeepSeek's own router (separate from GQA's, so `routed_scaling_factor` — 1.0 on Lite,
16.0 on the full V2 — cannot be silently dropped by sharing).

**The seam held.** Adding MLA required `arch/mla.cpp`, `arch/mla.hpp`, one registry line, and the
adapter mapping. No changes to the memory hierarchy, the scheduler, the kernels, the expert store, the
container, or the forward. That is what the payload refactor bought.

**DeepSeek-V2-Lite PASSES the G0 gate: `max=8.34e-07  mean=1.15e-07`, greedy token-exact.**

The defect, found by the activation harness after four rounds of reasoning had failed to locate it, was
in the **attention factor**. Two formulas exist and they are not the same:

```
DeepSeek remote code   scale *= mscale(factor, mscale_all_dim)^2          -> 1.58962
transformers generic   scale *= mscale(f, mscale) / mscale(f, mscale_all) -> 1.00000
```

With `mscale == mscale_all_dim == 0.707` the ratio form vanishes exactly while the squared form does
not. The engine used the squared one; the oracle came from transformers' native path
(`meta.json: "implementation": "native"`), which uses the ratio.

Its signature is why every earlier attempt missed it: **every projection correct, both rope outputs
correct, only the attention weights wrong** — a correctly-computed set of scores passed through a
too-sharp softmax. Nothing before the softmax is disturbed, so a whole-model logit diff shows only
"positional, growing with context", which is what four rounds of hypotheses were fitted to.

The historical record below is left intact deliberately: it is the honest account of what four rounds of
plausible reasoning bought (nothing) versus what one instrument bought (the answer), and the four
eliminated hypotheses were all *correct* eliminations.

```
DeepSeek-V2-Lite   FAIL   max=6.08e-02@t98  pos0=4.47e-07  mean=8.29e-03
```

The number that localises it is `pos0 = 4.47e-07` — **position 0 is exact**. With a single token the
softmax over one score is 1.0, so RoPE and the softmax scale both cancel; everything that survives that
cancellation is therefore correct: `kv_a_proj`, `kv_a_layernorm`, `kv_b_proj`, `o_proj`, the dense layer
0, and the MoE with shared experts. The error then **grows with position**, which is the signature of a
positional fault.

##### The rope scaling parse was investigated and is NOT the cause

The obvious suspect — `type: "yarn"` failing to reach `RopeScalingKind::Yarn`, which would skip the
`mscale²` correction on the softmax scale and be *invisible at position 0 by construction* — is wrong.
`SOMA_MLA_PROBE=1` prints both YaRN quantities, and they match values computed by hand from the
reference formulas to nine digits:

| quantity | hand-computed | engine |
|---|---|---|
| `softmax_scale` = `1/√24 · mscale²`, `mscale = 0.1·0.707·ln 40 + 1` | 0.324481 | **0.324481100** |
| `inv_freq` (extra `{1, .1, .01, .001}`, ramp `{0, 0, .5, 1}`) | `{1, .1, .005125, .000025}` | **identical** |

So the parse, the correction range (`low=1, high=3`), the interpolate/extrapolate blend, and the
softmax-scale correction are all correct.

The probe stays in the file. It cost a few lines and converted "the rope parse is the likely suspect"
into "the rope parse is ruled out", which reading the code was never going to do.

##### The interleaved-rotation equivalence holds exactly

`tests/soma/mla_g4.cpp` checks the second candidate: `rope_interleaved_at` rotates in interleaved layout
and leaves it there, while the reference de-interleaves first. The claim was that q and k receive the
same permutation and a dot product is invariant under that.

```
the two layouts really do differ elementwise   OK   max elem diff 2.148515
but the dot products agree                     OK   max |dot diff| 0.000000
one operand in each layout gives a different answer  OK   matched -2.200315 vs mixed 0.583307
```

Exact, over 40 positions and query/key pairs at *different* positions — not just `t == j`, which an
equivalence could satisfy while failing in attention. The first check confirms the layouts genuinely
differ (otherwise the test proves nothing), and the third confirms the argument is load-bearing by
showing that applying it to only one operand breaks the result.

**So both positional candidates are eliminated.** Verified correct: the softmax scale, the YaRN
frequencies, the rotation, and everything non-positional (position 0 is exact to 4.47e-07). The residual
is real, grows with position, and is not yet explained.

##### The oracle's provenance was checked, and the hypothesis it suggested is FALSIFIED

`meta.json` already records what produced the reference:

```
"implementation": "native",  "transformers_version": "4.57.6",  "torch_version": "2.13.0+cpu"
```

**Native**, not `trust_remote_code` — so the oracle came from transformers' own DeepseekV2. That made
the standard `rotate_half` pairing the obvious candidate, since HF's native rewrite drops the
de-interleave that the original remote code performs. The two conventions agree only at position zero,
which matches the observed signature exactly.

It was measured, and it is **worse**:

| rope pairing | max | mean |
|---|---:|---:|
| interleaved `(v[2i], v[2i+1])` | 6.08e-02 | 8.29e-03 |
| rotate-half `(v[i], v[i+d/2])` | **9.25e-02** | **1.12e-02** |

So the pairing is not the remaining fault, or not the only one. Interleaved is kept because it is
closer, **not because it is confirmed** — the file says so at the call site, so the next reader does not
inherit it as settled.

##### Where this leaves G4, stated plainly

Four hypotheses raised, four eliminated by measurement: the rope-scaling parse (probe: nine-digit match),
the YaRN frequencies (same), the rotation equivalence (`mla_g4`: 0.000000), and the rope pairing (worse).
Everything non-positional is confirmed correct by `pos0 = 4.47e-07`.

#### The activation diff harness — and what it found immediately

Four rounds of hypothesise-code-build-measure had localised the MLA defect to "somewhere positional",
because the only observable was the final logits. That is the resolution limit of whole-model
comparison, and it was reached four times.

`tools/admission/dump_activations.py` + `tests/soma/actdump_g4.cpp` write the same container from the
transformers reference and from the engine, with three taps per layer: `hidden_in` (the residual stream
entering), `attn_out` (attention's contribution before the residual add), and `hidden_out` (the stream
leaving). The core-side tap is `F32Workspace::Sink` — a C function pointer, null in every production
path, so the cost when off is a branch that always predicts.

First run:

```
 layer  tap                 n    max|diff|   mean|diff|
     0  hidden_in         512    0.000e+00    0.000e+00
     0  attn_out          512    1.160e-02    1.614e-03   <-- DIVERGES
     0  hidden_out        512    1.142e-02    1.926e-03
     ...
logits  logits           4096    3.352e-02    5.988e-03

FIRST DIVERGENCE: layer 0, tap 'attn_out'
```

**Layer 0, `attn_out`.** Input is exactly identical; attention's output is not. And layer 0 is the
DENSE layer (`first_k_dense_replace: 1`), so the MoE, the router, the shared experts, the expert union
and the tiled expert loop are all excluded in one step. The defect is inside `f32_attention` for MLA and
nowhere else.

Compare the cost: four rounds of guessing, each requiring a code change and a full rebuild, versus one
run of a tool that reports the answer directly. The lesson is not about MLA — **the ladder had no
instrument between "logits differ" and "read the code again", and every architecture through the seam
will need one.** It should have been built at G0.

##### One level finer: sub-layer taps

Five more taps inside `f32_attention`, chosen to land on MODULE BOUNDARIES so the Python side can hook
the corresponding submodule directly — a tap mid-way through a fused step is cheap in C++ and unhookable
in torch. `o_proj` is taken on its *input*, because that is the only observable between the attention
math and the output projection.

```
 layer  tap                 n    max|diff|   mean|diff|
     0  hidden_in         512    0.000e+00    0.000e+00
     0  q_proj            768    3.576e-07    3.906e-08
     0  kv_a_proj         320    2.980e-07    4.143e-08
     0  kv_a_layernorm    256    7.749e-07    1.220e-07
     0  kv_b_proj        1024    3.576e-07    4.524e-08
     0  o_proj_in         512    2.491e-02    4.044e-03   <-- DIVERGES
```

**Every projection is exact.** Float noise only, ~1e-7. That clears, in one run: the weight binding, the
`[n_heads·(nope+rope), d_model]` q shape, the `latent ++ shared-rope` split point in
`kv_a_proj_with_mqa`, the `kv_a_layernorm` weight/eps/slice, and — the assumption flagged as most likely
wrong — the per-head `(K-nope ++ V)` h-major layout of `kv_b_proj`.

**The defect is in the ~15 lines between `kv_b_proj` and `o_proj`**: rope application, the score scale,
the softmax, or the value accumulation. Of those, the scale is verified against hand-computed values to
nine digits, the YaRN frequencies likewise, and the softmax is shared code that both GQA families pass
on. That leaves the rope application and the value accumulation.

The search space went from "the whole model" to fifteen lines across two runs of a tool that did not
exist an hour earlier, having previously resisted four rounds of reasoning. The remaining step needs
taps on `q_pe`/`k_pe` *after* rotation, which are intermediate tensors rather than module outputs — so
the reference side needs a monkey-patched `apply_rotary_pos_emb` rather than a hook, which is the next
increment of the same tool.

##### A config key one family states and another derives

The first fix took the mean error from 1.06e-01 to 8.29e-03 and made position 0 exact.
`ffn.shared_intermediate` was read only from `shared_expert_intermediate_size` — a **Qwen2-MoE** key.
DeepSeek ships no such key and *derives* the width as `moe_intermediate_size × n_shared_experts`. The
shared expert was therefore sized zero: tensors bind, the width is wrong, and the logits come out finite
and off by ~0.1 mean — which reads as a subtle attention bug rather than a config one.

This is the fourth finding of the same shape as G0's `rms_norm_eps` and `q_norm` discoveries: **the
config format is not a schema, it is a set of per-family conventions**, and the only way to find the
places they disagree is to run a second family through.

##### Moonlight (deepseek_v3) also passes: `max=7.25e-05 mean=1.06e-05`

V3 needed two things V2 did not.

**1. The router signature could not express its router.** `noaux_tc` scoring adds a per-expert
`e_score_correction_bias` that participates in selection, and `route()` took only logits — there was
nowhere to put it, and no amount of care inside the backend works around a missing argument. It now
takes `const F32LayerWeights&`, so a router's extra tensors travel with the layer. The bias lives in the
backend's per-layer payload, which is the second time that payload has proved to be "the backend's
state" rather than "the backend's attention weights".

The algorithm, with V2 as its degenerate case (softmax, one group, no bias) so there is one router
rather than two to keep in agreement:

- sigmoid scoring rather than softmax;
- the bias added for **selection only** — weights are re-read from the unbiased scores, since the bias
  exists to steer load balancing and would otherwise scale every expert's contribution;
- group-limited top-k where a group's score is the sum of its **top two** experts, and groups outside
  `topk_group` are masked out *before* expert selection. "Pick top-k then filter" selects from the wrong
  pool.

**2. `yarn_inv_freq` ran unconditionally.** It applied YaRN interpolation to models that never asked for
it, and on degenerate inputs produced **NaN**: `log(max_pos / (rotations · 2π))` goes to ±inf when a beta
is zero or the base is 1, and `high - low` then evaluates `inf - inf`. That NaN reached the rope taps
while every projection before them was clean — which is exactly how the harness pointed at it. Now
guarded on `kind == Yarn` with sane parameters, falling back to plain RoPE otherwise.

The second bug is the more interesting one: it had been latent in the MLA backend since it was written,
invisible on V2 (which does have valid YaRN), and would have surfaced on the first non-YaRN MLA model
whenever that arrived.




**Build:** `include/soma/arch/mla.hpp` + `src/soma/arch/mla.cpp`. MLA attention, compressed KV, weight
absorption, partial RoPE, YaRN with mscale, group-limited routing, shared experts.

**Model: DeepSeek-V2-Lite** — 27 layers, `kv_lora_rank` 512, `qk_rope_head_dim` 64, 64 experts top-6,
2 shared, `first_k_dense_replace` 1.

**Gate — the strictest one, and the reason for the sequencing rule:**
- The full ladder (G0→G2 equivalents) passes on **both** families.
- **Zero diffs to `include/soma/*.hpp` and `src/soma/*.cpp` outside `arch/` and `arch_registry.cpp`.**
  A core diff means the seam was shaped to GQA and this gate has found it — which is the entire point of
  running the check here rather than trusting it.
- `tools/ci/check_seam.py` green.
- The absorbed and unabsorbed MLA paths agree (`absorb_weights: false` is a test lever, not dead config).
- A GQA KV checkpoint fed to the MLA engine is **rejected** on `format_id`, not read.

**Expect `resident-only`:** V2-Lite's routed set is 7.2 GB at q4 and fits in RAM on any reasonable host.
That is fine and expected — G4 tests the **seam**, not the economics. Run it with
`backend_override: soma`, which is exactly what that override exists for.

---

## G5 — subprocess integration + verdict routing

**Build:** the supervision rebuild — `EngineProcess`, `EngineSupervisor`, `EngineDescriptor` +
registry, `EngineClient` with virtual `stream_complete`, `KvCheckpointBackend` ×2, `PlacementEngine`,
`ResourceFootprint`, `control.db`. llama.cpp **ported onto** these abstractions.

**Gate:**
- The existing `reliability_tests` suite still passes. The fallback must not regress; this is the gate
  where that risk is real.
- **Mixtral-8x7B admits as `resident-only` and routes to the fallback with no operator action.** The
  negative fixture is a pass condition, not a footnote — a system that cannot recognize what it is bad
  at will happily be bad at it.
- A Soma agent and a fallback agent run **concurrently on the same node**.
- A converted model directory sizes correctly. Assert the old flat-2048 MB path is gone by checking a
  multi-shard fallback model too.
- Killing a Soma process is detected by the watchdog within one poll interval, and the engine transitions
  to `Error` rather than advertising `Ready`.
- `capacity_pressure` as a structured code drives evict-and-retry, with the substring matcher deleted.
- Suspend on a multi-sequence llama.cpp slot returns 409 rather than silently saving sequence 0.

---

## G6 — API surface complete

**Build:** every route in [external-api.md](external-api.md); the three-scope authorizer; the token
store; SSE telemetry with throttling; `/v1/models` reclamation.

**Gate:**
- `require_complete_coverage()` green: every registered handler has a scope-table entry. Startup fails
  otherwise.
- A `read`-scoped token can stream telemetry and **cannot** admit, delete, or chat. Test the negative.
- A `chat`-scoped token cannot admit. Test the negative.
- The legacy flat token still works as all-scopes.
- `hz` clamps to 10; `resolution=full` requires the explicit parameter.
- A client requesting maximum telemetry on a 60k-expert model does not measurably affect chat latency —
  the aggregation-in-engine claim, verified rather than asserted.
- Image content parts return **422**, not a dropped part.

---

## G7 — FTXUI dashboards on that API

**Gate:**
- Every Soma panel's data comes from `/v1/*`. **No panel reaches into in-process engine state.**
  Grep-verifiable, and worth verifying: the existing TUI already mixes direct access and loopback HTTP,
  so the temptation is live.
- The brain grid renders a 48×128 model bucketed to ≤4096 cells at 2 Hz without visible cost.
- The tier bar shows the VRAM tier present and empty — the declared-but-stubbed design, visible and
  honest rather than hidden.
- Panels degrade gracefully when an engine is a fallback and has no tier/heat data.

---

## G8 — admission pipeline self-service

**Gate:**
- A GQA MoE checkpoint **not seen during development** goes from `POST /v1/models/admit` to a served
  agent with **no C++ change**. That is the whole gate.
- An architecture with no backend fails at `validate()` with a clear message, before conversion spends
  hours.
- Re-admission with different quantization produces a new `arch_hash` and invalidates KV checkpoints;
  re-profiling does **not**.

---

## Cross-cutting: CI from day one

Established at G0, extended each gate. Current CI is 3 Release jobs invoking raw `cmake`, one CTest
entry, and **no lint or format job** — there are no format config files in the repo at all.

| Job | From | Purpose |
|---|---|---|
| `soma-header-selftest` | G0 | Every `include/soma/**.hpp` compiles standalone, twice-included, `/W4 /WX` + `-Werror` |
| `seam-check` | G0 | `tools/ci/check_seam.py` — R1 include discipline, R2 no arch names in core code |
| `soma-conformance` | G0 | The full ladder on tiny-random fixtures, **per family**, every commit |
| `format-check` | G0 | `.clang-format` + `.clang-tidy` — neither exists today |
| `aarch64-cross` | G5 | `vcpkg-aarch64-linux-release` exists as a preset and CI has never used it |

**Fixtures and golden oracle logits are committed, not generated in CI.** That keeps `torch` and
`transformers` out of the CI image and makes the oracle reproducible; regeneration is an explicit
`tools/admission/make_oracle.py` run.

Without a per-family ladder on every commit, **the second architecture silently breaks the first within
a month.** Kernel-shape sensitivity makes cross-architecture regressions unusually easy to introduce and
unusually hard to spot — a q4 kernel retuned for MLA's shapes can shift a GQA argmax tie and produce
output that is different but not obviously wrong.

---

## What is deferred, and where it would land

| Deferred | Earliest sensible gate |
|---|---|
| GPU tier residency + GPU kernels | Post-G4. The tier is declared and reported throughout, so this is an implementation, not a migration. |
| Paged KV with a block table | Post-G4. `AttentionBackend` already takes a batch, so it lands without an interface change. |
| Speculation under batching | Post-G4 |
| Multimodal | Not planned. 422 is the contract. |
| Distributed / multi-node single model | Not planned for v1 |
