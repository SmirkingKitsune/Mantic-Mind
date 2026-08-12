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
| **G4** | Second architecture (MLA) through the seam | Full ladder passes on both families with **zero core diffs** — passed on the ladder, **one core diff** (the router signature); see [G4 status](#g4-status--passed-on-both-families-one-core-diff-and-the-gate-is-what-found-it) |
| **G5** | Subprocess integration + verdict routing | Mixtral admits `resident-only` and routes to the fallback, unprompted |
| **G6** | API surface complete | Every capability on `/v1/*`, scope-gated, coverage check green |
| **G7** | FTXUI dashboards on that API | Panels consume only `/v1/*` |
| **G8** | Admission self-service | A new HF repo goes end-to-end with no C++ change |
| ~~**G9**~~ | ~~The read overlaps the compute~~ | **RETIRED** — specified on a false premise; the reads were already overlapped. See [G9](#g9--retired-the-gate-that-was-already-passed) |

Defects found and not yet fixed are tracked in [Open defects](#open-defects), separately from
[what is deferred](#what-is-deferred-and-where-it-would-land) — deliberately-not-built and
known-to-be-wrong are different states and a shared list loses both.

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

One limitation worth stating: `rng_state` lived in the scheduler's in-memory `Seq`, not in the KV
checkpoint. In-process preempt/resume preserved it, so the byte-identity gate held; a cross-process
restore resumed with a fresh RNG stream. That was a gap in the checkpoint format, not in the sampler.
**Closed at G6** by format v3 — see [Checkpoint format v3](#checkpoint-format-v3-carries-the-sampler).

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

Four of the five entries below were written when the union and the telemetry dump were the measurable
core and the scheduler around them did not exist. They have been built since and the section was never
revisited — struck rather than deleted, because a list that quietly loses its completed items stops
being evidence of anything. Verified against the code before striking, not assumed:

- ~~**The scheduler itself**~~ — **built**. `src/soma/scheduler.cpp`, step-major over ready sequences
  with ragged batch assembly; rows are drawn from independent sequences rather than from one prompt's
  tokens, which is the case the union was always for.
- ~~**KV checkpoint store and preemption**~~ — **built**, and the byte-identity gate is no longer
  untested: `checkpoint_g3` asserts "continuation identical after preempt at step N".
- ~~**Chunked prefill and the fairness cap**~~ — **built**, both exercised by `scheduler_g3`.
- ~~**`effective_max_batch` under a deliberately small `ram_budget`**~~ — **built**, covered in
  `scheduler_g3`.
- **The last stretch into the disk-bound regime** — still live, and now the whole of what remains here.
  It is what [G9](#g9--retired-the-gate-that-was-already-passed) is about.
- **The last stretch into the disk-bound regime.** Reads sit at ~44% of wall time after vectorising both
  kernel families, threading the forward, and adding the AVX-512 tier. Each pass has bought less than
  the one before, and AVX-512 — expected to be the big mechanical win — returned ~6%. The remaining
  levers are no longer kernel-shaped:
  - **MoE width**, which is the scheduler's job: batching across sequences is what supplies rows per
    expert, and that is the same axis that makes the union save more. It is the one lever that improves
    both halves of the gate at once.
  - ~~**Overlapping I/O with compute.**~~ **Built, and measured — see
    [G9](#g9--retired-the-gate-that-was-already-passed).** Expert reads are NOT issued serially; the
    union loop queues `prefetch_ahead` and loader threads read while the current expert is applied.
    With it on, `io_wait` is **6.9% of wall at nseq=8** (737 prefetch hits, 0 wasted). The ~44% figure
    below describes the engine with prefetch DISABLED, which is what this bullet was written against and
    never revised.
  - A **vectorised `exp`** for softmax, which needs its own oracle measurement since it changes values
    rather than their order.

  `scaling_g3`'s read fraction remains the number that says whether any of it worked.
- ~~Blocked (row-tiled) expert-major loop~~ — **built**, see below.

---

## G4 — second architecture through the seam

### G4 status — PASSED on both families; ONE core diff, and the gate is what found it

| Model | Teacher-forced vs the `transformers` oracle | Greedy |
|---|---|---|
| DeepSeek-V2-Lite (`deepseek_v2`) | `max=8.34e-07  mean=1.15e-07` | token-exact |
| Moonlight-16B-A3B (`deepseek_v3`) | `max=7.25e-05  mean=1.06e-05` | token-exact |

`tools/ci/check_seam.py` green. The strictest line of this gate was **zero diffs to `include/soma/*.hpp`
and `src/soma/*.cpp` outside `arch/` and `arch_registry.cpp`**, and it was not met: `F32Backend::route`
in `include/soma/f32_model.hpp` gained a `const F32LayerWeights&` parameter for DeepSeek-V3's `noaux_tc`
scoring, whose per-expert correction bias participates in selection and had nowhere to live in the old
signature.

**That is the gate working, not failing.** Its purpose was to reveal where the seam had been shaped to
GQA, and it revealed exactly one place: a router interface that assumed a router's only input is its
logits. No amount of care inside the backend works around a missing argument. Recorded here rather than
rounded down to "zero", because a gate whose result is edited to match its target measures nothing —
and the next architecture will want to know that this is the interface that moved.

Everything else held: no changes to the memory hierarchy, the scheduler, the kernels, the expert store,
the container, or the forward.

The IR needed **no change at all**. `MlaSpec` (kv_lora_rank, q_lora_rank, qk_nope/rope_head_dim,
v_head_dim) and the RoPE spec's `mscale`/`mscale_all_dim` were co-designed against DeepSeek during the
design pass, and DeepSeek-V2-Lite's config maps onto them without a schema extension. That was the
point of writing `arch.json` for both families on paper before any code existed, and it held.

Three findings at the ADAPTER level, all of which the second architecture existed to surface. Four more
came later, from the activation harness, and are recorded further down:

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

**The honest read on the gate.** *"Zero core diffs — only `arch/` TUs added"* was not met, and the
places it was not met are the gate's whole output. Three of them, in increasing order of significance:

- `arch_ir.cpp` gained the DeepSeek mapping. Expected — it is the allow-listed adapter and the seam
  check permits it by construction.
- The family-guard fix was a core correction, and a correction of a rule the first family could not
  have revealed as wrong.
- **`F32Backend::route` gained a parameter.** The one that matters; see
  [G4 status](#g4-status--passed-on-both-families-one-core-diff-and-the-gate-is-what-found-it).

Both items once listed here as "still ahead" are done: ~~`src/soma/arch/mla.cpp`~~ is written and
passing on two families, and ~~`F32LayerWeights` is GQA-shaped~~ resolved into the opaque per-layer
payload described next.

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

#### `src/soma/arch/mla.cpp` — PASSES; the defect was the attention factor

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

> **Everything from here to "What the four rounds established" is a HISTORICAL RECORD of an
> investigation that has since concluded.** The defect was the attention factor, above. These four
> hypotheses were all eliminated correctly and none of them was the cause; they are kept because the
> contrast is the finding — four rounds of plausible reasoning bought nothing, one instrument bought the
> answer. Read them as a log, not as open questions.

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
was real and grew with position, and at the time of writing was unexplained — it was the **attention
factor**, found later by the activation harness. See the heading above.

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

##### What the four rounds established — all correct eliminations

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

### G5 status — verdict routing PASSES; `soma serve` and the node-side supervision layer PASS

The call sites are ported: `SlotManager` is deleted, placement runs on `ResourceFootprint`, `control.db`
exists and an admitted model routes to Soma. Remaining for the gate: the offline admission pipeline
(`admit()` over SSE) and the scope-annotated route table, both of which belong to later gates.

`include/soma/routing.hpp` + `src/soma/routing.cpp`. `select_backend(config, record)` is **pure** — no
node, no placement, no I/O — which is what lets the whole policy be tested exhaustively rather than
observed, and lets `GET /v1/agents/{id}` show the decision without causing a placement.

The policy table, all of it, asserted:

| | choice | reason |
|---|---|---|
| `stream` / `hybrid` | Soma | verdict |
| `resident-only` | fallback | verdict |
| `reject` | fallback | verdict |
| no record | fallback | absence is not evidence of admissibility |
| record for **different weights** | fallback | a verdict for other weights is not a verdict for these |
| override `fallback` | fallback | always honoured — it can only be more conservative |
| override `soma` + `resident-only` | Soma | honoured: an economics call, and what G4 relies on |
| override `soma` + `reject` | **fallback** | **refused** |

**The one refusal is a deliberate asymmetry.** `resident-only` means streaming buys nothing on this
host, so an operator who wants Soma anyway is making a legitimate performance choice. `reject` means the
model FAILED conformance stage 1 or 2 — Soma's output does not match the reference. Honouring the
override there would serve knowingly-wrong tokens because of a config flag, and it would present as
model quality rather than as configuration. The remedy for `reject` is to re-admit with a different
quantization map, which is a different action, and the API should not let one masquerade as the other.

#### The gate, and a contradiction in the first attempt at it

| model | routed | MiB/token | verdict @48 GiB | verdict @8 GiB |
|---|---|---|---|---|
| Mixtral-8x7B | 23 GiB | 6048 | **resident-only** | reject |
| Qwen3-30B-A3B | 17 GiB | 1098 | resident-only | **stream** |

Mixtral admits as `resident-only` and routes to the fallback **unprompted** — no override, no
model-specific case, driven through the real planner rather than a hand-written verdict. Asserting
`select_backend(ResidentOnly) == Fallback` would have proved only the router; the gate is about whether
Mixtral *earns* that verdict.

The first version of the test used one roomy host and expected Mixtral resident-only alongside Qwen3
streaming. **That is unsatisfiable**, and the contradiction was in the test: Qwen3's routed set is
*smaller* than Mixtral's, so every host where Mixtral fits is one where Qwen3 fits too. Worth recording
because "the big model streams, the small one stays resident" is exactly the intuition the verdict
exists to correct.

What separates them is the **active fraction**, not size. Mixtral fires 2 of 8 experts per token (25%);
Qwen3 fires 8 of 128 (6.25%). Streaming reads what fires, so Mixtral moves **5× the bytes per token
while being only 1.35× larger** — asserted directly, since it is the quantity the whole verdict rests
on. On the host where neither can be resident, that difference is what makes one `reject` and the other
`stream`.

#### `soma serve` — the subprocess boundary works

`src/soma/serve.cpp` + `src/soma/main.cpp`. The `soma` executable now has `serve` and `plan`, and the
whole contract a node needs was exercised against a live process:

```
$ soma plan --model-dir tests/fixtures/tiny/Qwen3-30B-A3B
verdict      resident-only
reason       routed set (1 MiB) fits the expert cache (8175 MiB); streaming has nothing to do

$ soma serve --model-dir ... --port 8099
GET  /health              -> {"status":"ok","engine":"soma","verdict":"resident-only","streamed":false}
GET  /v1/models           -> {"data":[{"id":"Qwen3-30B-A3B","owned_by":"soma"}]}
POST /v1/chat/completions -> {"choices":[{"message":{"role":"assistant","content":"..."}}]}
POST ... "stream":true    -> data: {...delta...}  x4  then  data: [DONE]
POST ... image content    -> 422
```

Three things deliberately not inherited from the fallback path:

- **Readiness is an HTTP poll, not a stdout sentinel.** `RuntimeProcess` already works this way and it
  is the right shape; a sentinel is a line-buffering bug waiting to happen on Windows. The probe above
  is literally what the node's `ReadinessProbe::HttpHealth` will do.
- **Capacity pressure is a structured code.** The existing scheduler detects it by substring-matching
  six English phrases against the node's error body (`agent_scheduler.cpp:904`), so a new engine would
  have to emit those exact literals to earn an evict-and-retry. This emits
  `{"error":{"code":"capacity_pressure"}}`.
- **Image parts are refused with 422, not dropped.** Silent dropping is the failure worth designing out:
  the request succeeds, the answer ignores the picture, and nothing says why.

`plan` accepts either a converted container or a raw HF checkpoint — refusing the latter would make the
subcommand useless exactly when it is most wanted, which is *before* conversion. It runs the same
planner the server runs, so an operator asking "what will this do here?" and the engine deciding what to
do cannot disagree.

##### Three limitations, named rather than left to be discovered

1. ~~**Requests serialise on a mutex.**~~ **Fixed** — see "Concurrent turns" below. One thread drives
   `step()`; request threads admit and wait. Session-scoped sequences were the prerequisite: a shared
   step loop needs sequences that are not owned by whichever request holds the mutex.
2. ~~**`CompiledTokenizer::Streamer` is declared and never implemented.**~~ **Fixed** — see
   [Incremental detokenisation](#incremental-detokenisation) below. The claim recorded here, that
   re-decoding "handles the split-multi-byte-codepoint case … because every delta is the difference
   between two complete decodes", was **wrong**, and the test written for the replacement failed against
   the old path on 4 of 72 corpus cases.
3. **The tiny fixtures ship no `tokenizer.soma`**, so the demo above returns raw token ids as text. That
   is the documented no-tokenizer fallback and it exercises the full HTTP path; it is not evidence about
   detokenisation.

#### The node side — supervision, and a watchdog that did not exist

`src/node/engine_process.cpp`, `engine_descriptor.cpp`, `engine_supervisor.cpp`,
`src/common/engine_client.cpp`. Verified by `tests/soma/engine_g5.cpp`, which launches the **real**
`soma` binary rather than a mock — the failure modes that matter here (a child that exits during
startup, a child that dies after reporting ready) do not exist against a stub.

```
1. engines are DATA, not code paths          7/7    registry, capabilities, argv, probe kind
2. launch -> ready -> clean stop              5/5   ready in 0.60s; a clean stop is not a crash
3. an engine that dies is NOTICED             4/4   callback + state=Crashed, by pid
4. a start-up failure fails fast              4/4   1.22s of a 30s budget
5. supervisor: sharing, leases, unknown ids  16/16  attach, lease, refuse-unload, detach
6. a crashed engine stops advertising Ready   6/6   Ready -> Error, and acquire() refuses it
```

Four results worth stating on their own:

- **A dead engine now stops advertising `Ready`.** The pre-existing gap: nothing polled the child after
  readiness, so a crashed engine held `SlotState::Ready` until an inference request happened to fail,
  and the scheduler kept placing work on it. Test 6 kills the process behind the supervisor's back and
  requires the state transition, then requires `acquire()` to refuse the slot. The record is moved to
  `Error` rather than removed — control has to be able to *see* that a placement died; a deleted record
  reads as an engine that was never there.
- **A start-up failure costs child-exit time, not budget time.** The readiness poll checks child liveness
  on *every* iteration, not once at the end. A bad model path fails in 1.22 s against a 30 s budget; the
  naive version would sit for the full 30 s and still pass a boolean assertion.
- **Two agents share one Soma engine at different `ctx_size`.** This is the descriptor abstraction
  earning its keep. `llama_launch_compatible()` gates on `ctx_size` because llama-server carves context
  per slot at launch; Soma's KV slot is per-sequence, so its predicate omits it — one field, and it is
  the difference between one process per agent and one process for all of them.
- **`SlotInfo::backend` comes from the descriptor.** It was hardcoded to `"llama-cpp"` in
  `make_slot_info()`, so every slot reported the same backend regardless of what was running. Likewise
  an unknown engine id now produces a message listing `EngineRegistry::ids()` — accurate by
  construction, not by a maintained literal, which is what the two duplicated
  `backend != "llama-cpp"` checks were.

**Lock discipline is load-bearing, and the watchdog is why.** `on_engine_crash` fires from the watchdog
thread and takes the supervisor's mutex; `EngineProcess::stop()` joins that thread. Holding the mutex
across `stop()` therefore deadlocks. Every mutating path extracts the engine under the lock and does the
blocking work after releasing it — the structure `SlotManager` used for load only, now required
everywhere.

#### KV checkpoints — one format, one parser, two backends

`src/node/kv_checkpoint_backend.cpp`, wired into `descriptor->kv` for both engines.

**`LlamaKvBackend` is complete.** Same wire protocol as before (`POST /slots/0?action=save|restore`,
`{"filename": basename}`) because it is the only one llama-server speaks, with three changes:

- `supports_multi_sequence()` returns **false**, and the supervisor consults it. A suspend on an engine
  holding more than one sequence is now refused with `Unsupported` instead of silently saving sequence 0
  and discarding the rest — a latent data-loss bug in the current system, since the resumed slot comes
  back with one agent's context and several agents' expectations.
- A non-zero `sequence` argument is **refused** rather than quietly rewritten to 0.
- `save()` verifies the file actually appeared. Which exposed a second bug: the llama descriptor never
  passed `--slot-save-path`, so llama-server resolved the basename against its own default and the node
  wrote checkpoints it would never find again. Now passed from `kv_checkpoint_dir`, and a misconfigured
  launch fails at the save rather than at the restore hours later.

**`SomaKvBackend` saves EVERY session, and that is the substantive difference from the fallback.**
llama.cpp is asked for one sequence and gives you sequence 0; Soma is asked for the engine and writes
all of them plus a manifest naming them. `file_extension()` is therefore `.somasession` (the manifest),
not `.somakv` (one sequence's KV) — the two answer different questions, and `stat_sequence()` exists for
the second. The manifest carries `arch_hash`, so the pre-spawn cross-architecture rejection works from
the manifest alone without opening any KV file.

**The header codec now has exactly one owner.** `stat()` has to run *before* an engine is spawned —
rejecting a cross-architecture resume after a 60-second model load is the confusing version of that
error — so the node must read the checkpoint header itself. Rather than a second parser in `src/node`,
the parse moved to `src/soma/kv_checkpoint_header.cpp`, a TU with no dependencies beyond the struct.
`KvCheckpointStore::load/stat/sweep` go through it, and the node links that one object. Static-library
linkage pulls only what is referenced, so this does not drag the engine into the node.

The test writes the header bytes **by hand**, deliberately: this is a wire format between two binaries,
and an independent encoder checking the shared decoder is the point. The other half of the chain — the
engine's writer against that same decoder — is `soma_checkpoint_g3`.

```
7. KV checkpoints: the format is owned in ONE place   17/17
   .kvbin vs .somasession, backend-owned; multi-seq flags; seq!=0 refused;
   arch_hash/format_id/length_tokens round-trip; v1 refused not misread;
   missing, empty, bad-magic and truncated all refused; remove idempotent
```

**A failed suspend no longer kills the engine.** The first version stopped the process and *then*
discovered the checkpoint had not been written, losing both the context it was trying to preserve and
the engine along with it. Save now happens first, behind `SlotState::Suspending` so `acquire()` refuses
during the window; on failure the state goes back to `Ready` and nothing else changes.

#### Session-scoped sequences — what the rest was waiting on

`soma serve` created a sequence per request and finished it with the response, and re-opened the
scheduler on *every* request, discarding every sequence. Two consequences: each turn re-prefilled the
whole conversation, and at suspend time there was no live KV for the node to checkpoint at all.

The scheduler is now opened once, at `ServeServer::open()`, and a finished sequence is **retained**
rather than erased. A conversation key (`"conversation"` in the body, or `X-Conversation-Id`) maps to
the sequence holding its KV; the next turn calls `Scheduler::extend()`, which prefills only the suffix.
An absent key means stateless — the previous behaviour, and still the default. A KV slot is real memory,
so the scheduler REFUSES rather than silently evicting; choosing whose context to drop is policy and
lives in serve, where sessions have names (LRU, then one retry).

```
8. a sequence outlives the request that created it     8/8
   turn 1 -> seq 1, 36 KV tokens
   turn 2 -> seq 1, 70 KV tokens        reused and EXTENDED, not rebuilt
   different key -> its own sequence
   edited prompt -> seq 3, 47 tokens    cold start, not a wrong cache
9. suspend writes EVERY session, not sequence 0        7/7
   manifest stats with no running engine; 2 sessions, 87 tokens, arch_hash present
```

`EngineSupervisor::sequences()` now returns real data, asked of the engine through a new
`EngineDescriptor::fetch_sequences` hook rather than synthesised from an agent count — a fabricated row
looks like per-sequence data while carrying none. llama.cpp has no such route and reports nothing, which
is the truth there.

**Checkpoint format v2 carries the token ids.** v1 did not, and that made every checkpoint unsafe to
replay: a cache of length L attaches to any prompt, and if the first L tokens differ the attention reads
a context nobody supplied. Nothing detects it — the output is fluent and wrong. The ids cost 4 bytes per
position against a payload of `L x n_kv x n_layers x 2 x 4` (0.016% for a 24-layer model with 128 kv
channels) and turn "trust the caller" into a checkable prefix. v1 files refuse to load, and that version
check moved into the shared parser because the node reads headers without a store: every offset in a v1
layout is wrong by `4 x length_tokens`, silently.

Three places check the prefix rather than assume it — `Scheduler::extend` (same process, next turn),
`Scheduler::admit` via `SeqRequest::resume_key` (next process, after a restore), and
`KvCheckpointStore::save`, which refuses a token list whose length disagrees with the cached positions.
A misaligned list makes the check downstream meaningless while still reading as a guarantee.

**Two faults this exposed, both fixed.** The token callback was unfiltered — harmless when the scheduler
held one sequence, and it would now splice another conversation's tokens into a response. And the
no-tokenizer fallback encoded *every* prompt as the single token 0, so prompt length did not track the
prompt; a session cannot be exercised at all under that. It falls back to bytes now, which is not a
tokenizer and does not pretend to be one, but makes length real.

#### Checkpoint format v3 carries the sampler

v2 restored a cache and nothing else. A sequence resumed in a **new process** therefore came back with
the right context and a *fresh draw stream and empty penalty history*: the same prompt resumed twice
produced different text, and a model that had already said something was free to say it again
immediately. Neither surfaces as an error. They read as the model being inconsistent, which is the
category of bug that gets attributed to the weights and never fixed.

v3 adds `rng_state` (u64) and `n_emitted` (u32) to the fixed header, and an emitted-id array after the
token array. The three growing parameters became one struct, because a five-argument `save()` is where
the wrong two get swapped:

```cpp
struct SeqPersistState {
    std::vector<TokenId> tokens;   // ids occupying the cached positions
    std::vector<TokenId> emitted;  // penalty history
    std::uint64_t rng_state = 0;   // splitmix64 position
};
```

**The sampling parameters deliberately stay on the request.** Temperature, `top_p` and the rest are not
persisted and are not restored — a checkpoint that carried them would make a client's change silently
ineffective after a resume, which is worse than the inconsistency v3 exists to remove. Only the *stream
position* is state. `Scheduler::resume` restores `rng_state` alone for exactly this reason.

v2 files refuse to load, as v1 did before them. `parse_kv_checkpoint_header` also lost three out-params:
`tokens_at`, `emitted_at` and `payload_at` are arithmetic from the fixed fields, so they land in the
header struct and `stat()` still reads a bounded 4096-byte prefix of a multi-gigabyte file.

**The test that mattered was the one the existing suite could not be.** `preempt -> resume` runs against
the same `Scheduler`, so the sampler object survives in memory and the continuation is byte-identical
whether or not the RNG was ever written to disk — those four checks would have stayed green through the
entire v2 era, and did. The honest test is a **second `Scheduler`**, which is what a restarted engine
has: everything the resumed sequence knows must have come off disk. With a control — a cold start on the
same prompt, whose cache is identical because prefill rebuilds it — the pair is decisive:

```
   uninterrupted tail: 489 439 156 275
   cold, no rng_state: 47 4 268 323
```

Without the control the check proves nothing; it would pass for any sampler whose stream did not depend
on history, and the premise is that this one's does.

**A real bug, found by writing that test.** `Seq::next_token` — the last token sampled but never fed
back, so it holds no cached position — is not in the checkpoint and cannot be, because it is not part of
the cache. In-process `resume()` never noticed: `next_token` never left memory. `Scheduler::extend()`
never noticed either, because it refuses a prompt that adds nothing beyond the cached prefix. But
`admit()` with a `resume_key` accepts exactly that case, and there is no prefill row to supply the next
input — so the step loop fed a default-constructed **0**, and the conversation continued from a token
nobody sampled. It is recoverable exactly, from `emitted.back()`. The test covers both spellings of a
resume, and with the fix reverted they split precisely as the failure predicts:

```
   fresh scheduler continues the run — cached prefix exactly        FAIL  got 292 137 257 296
   fresh scheduler continues the run — prompt extends past the cache  OK  byte-for-byte
```

A resume that handles only the second looks correct until someone checkpoints at a turn boundary — which
is what cluster suspend/restore does every time. Removed alongside it: `Seq::have_next`, written in two
places and read in none.

#### Incremental detokenisation

`CompiledTokenizer::Streamer` was declared in the header with no implementation. The server re-decoded
the entire emitted prefix on every token and sent the suffix — **O(n²)** in tokens, and inside the
scheduler's lock, so a long answer taxed every *other* sequence in the batch too. That is the reason it
had to go. The reason it was worth writing a test for is different.

The deferred-items list claimed the re-decode "handles the split-multi-byte-codepoint case the Streamer
exists for, because every delta is the difference between two complete decodes." That is false, and the
argument for it is the kind that sounds airtight. `decode()` is a concatenation of vocabulary entries;
a byte-fallback vocabulary has one token per byte; so `decode(["\xE4"])` **is** a complete decode, and
its value is a bare lead byte. Both decodes are complete and the difference between them is still not
text. The invariant the client needs is not "the delta came from a complete decode" but "the delta ends
on a codepoint boundary", and nothing was enforcing it.

The implementation is a hold-back buffer with three bytes of lookback. Three is what makes malformed
input safe rather than merely handled: an incomplete sequence is at most a lead plus two continuations,
so anything further back is complete whatever it is, and bytes that are not a valid prefix of a sequence
are *released* rather than held — otherwise a stream stalls forever waiting for a continuation byte that
is never coming. `flush()` is unconditional, including a partial codepoint: at end of stream the missing
bytes are not late, they are absent, and withholding them would make the streamed text differ from
`decode()`.

**The invariant, and the three ways it is checked.** For any token sequence, the deltas from `push()`
concatenated with `flush()` equal `decode()` of that sequence, byte for byte — streaming changes *when*
bytes are handed over, never *which*.

- `soma_tokenizer_g0` runs every corpus case through the Streamer and compares against `decode()`, with
  an independently written boundary checker (a test that calls the function under test to decide whether
  the function under test is right proves only self-consistency). It reports a `held` count, so a green
  column cannot mean "no split ever happened" — OLMoE holds 8, Qwen3 holds 1.
- A **synthetic split**, because a gate whose interesting case shows up by luck is not a gate. It finds
  the ids that decode to the individual bytes of `U+4E2D`, feeds exactly those, and requires the
  character back in exactly one delta. Zero deltas would be a stall; two would mean a partial codepoint
  went out.
- A **flush check**, which nothing else reaches: the corpus is valid UTF-8, so no case ends
  mid-codepoint and every `flush()` in the loop above returns empty. Pushing two of the three bytes and
  flushing is the deterministic version of a turn that hits `max_tokens` partway through a character.

Reverting the boundary logic to the old always-release behaviour turns all three red, and the corpus
alone catches it — 33/36 and 35/36, with the first failure reported as a delta ending mid-codepoint.
This was a live defect on the streaming path, not a hypothetical one.

**And the wiring, separately.** `soma_engine_g5` §9b streams a greedy turn and a non-streaming one
(`temperature: 0` is argmax and consumes no RNG, which is the only reason the two are comparable) and
requires the concatenated SSE deltas to equal the returned `content`. The unit test proves the class is
exact; this proves `finish()` flushes the tail, and flushes it *before* setting `done` — after that the
request thread returns and a late frame is sent to nobody. The section prints how many frames it got for
24 tokens rather than asserting a split occurred, because whether one does is a fact about the fixture's
random weights: on the committed fixture it is 24 frames, no split, and the boundary check there passes
without being exercised. Saying so is the point — the deterministic coverage is the probe above.

#### Concurrent turns — the union, finally reachable over HTTP

`generate()` held a mutex across the entire turn, so the batch union — the mechanism this whole engine
is built around — only ever had **one** sequence to union. One thread now drives `step()`; request
threads admit and wait on their own completion signal.

```
10. concurrent turns land in ONE forward               6/6
    output IDENTICAL alone vs batched      byte-for-byte
    4 concurrent turns, distinct answers, not spliced
    max observed batch = 4
    union at the widest sampled step: 26 unique / 64 naive = 2.46x
```

**The gate is the identity check, not the batch size.** Batching sequences together must not change what
any of them says — the same property G3 asserts inside the scheduler, now asserted through the HTTP
boundary where the batch is actually assembled. A concurrency win that quietly perturbs output is not a
win.

Three things had to change underneath, and each was a real defect rather than plumbing:

- **`Scheduler::step()` now takes the same lock `admit`/`cancel`/`extend` take.** It read and wrote
  `ready`, `seqs` and `stats` unlocked, which was correct exactly as long as one thread did everything.
  `idle()` likewise, and it lost its `noexcept`. The rule this creates is worth stating once: the token
  and finish callbacks fire from inside that lock, so a callback that calls back into the scheduler
  deadlocks.
- **A finish callback.** `is_last` on the token callback was not enough — a sequence can end without
  producing a token (a prompt that fills the context during prefill does), and a caller keying
  completion off `is_last` waits forever. `last` is also now computed *before* the token goes out, so a
  listener is not told about a token and then, separately, that the turn is over.
- **Stateless turns retire their sequence.** Retaining finished sequences is what makes a session warm;
  doing it for keyless turns too would hold a KV slot per request until the pool was exhausted and every
  later request was refused with `NoKvSlot`. Introduced by the retain change one section above, found by
  the concurrency test.

**Lock order is the one rule this file keeps:** the scheduler's lock is always taken first. `serve`'s
`state_mu` guards only the session and waiter tables, and every path releases it before calling into the
`Scheduler` — the callbacks run the other way round, and holding both in either order in two places is a
deadlock. `on_delta` is likewise called outside the waiter's lock: it writes to a socket, and holding a
lock across a network write is how one slow client stalls every sequence in the batch.

##### Named gaps, not silent ones

1. **Warm continuation needs a tokenizer whose decode-then-encode round-trips.** The tiny fixtures ship
   no `tokenizer.soma`, so the test pins `max_tokens = 1` — the cache then holds exactly the prompt,
   since the single sampled token is never fed back — and sends the full transcript each turn the way a
   real client does. The mechanism is what is under test; the round-trip property belongs to the
   tokenizer.
2. **Restore is lazy.** `/internal/kv/restore` validates the manifest's `arch_hash` and reports what is
   replayable; the caches are attached when each conversation's next request arrives, via `resume_key`.
   Eagerly restoring would hold KV slots for conversations that may never come back.
3. ~~**`SomaEngineClient::stream_telemetry` has no server route yet.**~~ **Fixed** — see "The telemetry
   feed" below.
4. **`estimate_footprint` sizes the directory rather than reading `soma plan --json`.** Recursive
   directory sizing already fixes the live bug it replaces — `fs::file_size()` errors on a directory and
   falls through to a flat 2048 MB, so every multi-shard HF model and every Soma container sized
   identically. `GET /internal/plan` now exists to make the honest version a small change.

**Found while wiring this:** `mm_reliability_tests` could not run under `ctest` on Windows at all — the
loader failed with `0xC0000135` before `main()`, which presents as an exception with no output. Its
vcpkg DLLs were only ever on PATH by accident of the invoking shell. Fixed by copying
`$<TARGET_RUNTIME_DLLS:...>` beside each executable, so the suite is runnable from any shell. 18/18 pass.

#### Porting the live call sites

The abstractions existed and were tested; the running system still used the old ones. This is the pass
that connects them.

**Node.**
- The two duplicated `backend != "llama-cpp"` 400s (`node_api_server.cpp:449` and `:911`) are now
  registry lookups, and the `supported_backends` list in the body is `EngineRegistry::ids()` — accurate
  by construction rather than by a hand-maintained literal that had already been copied once.
- The registry is populated at startup from `llama_server_path` and a new `soma_path`
  (`MM_SOMA_PATH`), and the llama descriptor is **re-registered** when the provisioner resolves a real
  executable, so the placeholder does not shadow it.
- Load and restore failures now carry a structured `code`, and capacity failures answer **503** rather
  than 500.

**Control.**
- `response_indicates_capacity_pressure` reads the structured code. The six-English-phrase substring
  match survives only as a fallback for a stale engine on the far side of a rolling upgrade, and is
  documented as such. A structured code is authoritative: one that says something else means the engine
  decided this is not capacity, and reading its prose for a contradicting hint would undo the point of
  asking.
- The hard `is_llama_backend` gate at `agent_scheduler.cpp:304` — which refused any non-llama agent
  outright — is replaced by `AgentScheduler::resolve_backend()`, which calls the same
  `soma::select_backend()` the engine uses. Control links `mm_soma` for exactly two objects so the
  policy has one implementation rather than a copy that drifts.
- `backend_override` (`auto | soma | fallback`) is persistent on the agent, migration 11, normalised on
  write so a typo cannot reach the router.
- Three hardcoded `{"backend", "llama-cpp"}` bodies and the slot-reuse filter now use the *resolved*
  engine id. The engine fingerprint does too — with a literal there, an agent whose routing changed
  would have kept a slot running the other engine.
- `PUT /v1/agents/{id}/backend` exists, and `GET /v1/placements` reports `backend` and
  `backend_reason`. The override had no route at all, so changing an agent's engine would have meant
  editing the database.

**The flat-2048 bug is fixed.** `estimate_inference_vram_mb()` called `fs::file_size()` on the model
path, which sets an `error_code` on a directory and fell through to a flat 2048 MB — so every
multi-shard HF checkpoint and every converted Soma container reported the same size, and that single
number was what placement consumed. `src/common/footprint.cpp` now exists (it was a declared header with
no implementation) and `measure_model_bytes()` sizes directories recursively.

**One ODR violation found on the way.** `runtime_process.hpp` and `engine_process.hpp` both declared
`enum class ProcessState` in namespace `mm` — the old four-state one and the new five-state one, which
adds `Crashed`. Any translation unit needing both the registry and a slot manager would have failed;
none had, yet. The definition follows the replacement.

**`SlotManager` is gone.** `src/node/slot_manager.{cpp,hpp}` are deleted; the node holds an
`EngineSupervisor`. The renames were mechanical, but three things were not:

- **`load` and `restore` take an engine id.** The handler passes the `backend` from the request
  through; which executable that is, what argv it takes and how to probe it belong to the descriptor.
- **The lease yields an `EngineClient`, so streaming errors arrive structured.** The node forwards the
  code, and control no longer has to read a message to decide whether a failure was capacity.
- **`suspend()` on an engine with no live process now FAILS.** `SlotManager` reported `Ok` with an
  empty cache path, which is a suspend that silently dropped the context it was called to preserve.
  Two tests asserted that old behaviour and now assert the refusal; a third needed a suspended record
  it could no longer produce, so `add_suspended_test_engine()` constructs the end state directly —
  making `suspend()` succeed without a checkpoint to keep a test green would have been the wrong repair.

**And the SSE mapping bug is fixed while the handler was open.** `/api/node/infer` mapped chunks through
an if/else-if priority chain, so a chunk carrying both `thinking_delta` and `delta_content` silently
dropped one, and `tool_result_json` had no branch at all and was never emitted. Both are named in
[mantic-mind-integration.md](mantic-mind-integration.md) as faults the rebuild must not inherit. Every
field the chunk carries is now emitted, and `done` is unconditional — it closes the stream, so it cannot
depend on one of the other branches having matched. A dropped delta presents as the model omitting a
word, not as a transport bug, which is why it survived this long.

#### Placement over three axes

`nodes_with_available_vram(int64_t)` is now `nodes_with_capacity(ResourceFootprint, CapacityPolicy)`.
The policy is unchanged in value — same 1 GiB VRAM and 2 GiB RAM headroom, same 0.60 offload weight,
same 8 GiB minimum GPU for hybrid loads, and a native fit still outranks every offloaded one — but the
four `constexpr`s that lived inside the registry are `CapacityPolicy`'s defaults, so the node and
control cannot disagree about what "fits" means.

**`NodeInfo::disk_free_mb` is read for the first time.** It has been collected on every health poll
since it was written and consulted by nothing, because the scalar the scheduler passed around had
nowhere to put it. `evaluate_fit()` now checks disk headroom on *every* placement, not only when the
footprint asks for disk: a node with no room cannot write a KV checkpoint or spill, whatever the model
costs. `disk_free_mb == 0` is treated as **not reported** rather than full — the field defaults to zero
and an older node never sends it, so enforcing against it would exclude every node predating the field.

llama.cpp's estimate still goes in `vram_mb` and `ram_mb`/`disk_mb` stay zero for it: the estimate folds
weights, KV and overhead into one number that behaves like VRAM, and the offload path inside
`evaluate_fit()` is what trades RAM against it, exactly as before. What changed is that a Soma
footprint — RAM + disk, from its plan — now has somewhere to go.

Two tests cover it, both asserting things that could not be asserted before:

```
multi_shard_directory_sizes_correctly
    a 3-file directory sizes to 9 MiB, nested file included
    two different directories no longer size identically   <- the 2048 MB bug
    a missing path reports 0, not a plausible constant
capacity_fit_across_three_axes
    native outranks offload however much headroom offload has left
    a 4 GiB GPU cannot be offloaded against
    a node with 512 MB free disk is refused, and the reason says "disk"
    disk_free_mb == 0 means not-reported, and still fits
    a RAM+disk footprint with no VRAM fits natively
```

#### control.db — Soma becomes routable

The first control-wide database in this system: until now the only SQLite was per-agent
(`data/agents/{id}/agent.db`), with remembered nodes in `nodes.json` and node model state in a JSON
journal. Migrations follow `AgentDB::run_migrations()` exactly — a `schema_migrations` table and one
`if (!has_version(N)) { Transaction; DDL; INSERT N; commit; }` block per version. That pattern works;
there was no reason to invent a second one.

All seven tables from [001_init.sql](schemas/registry/001_init.sql) are created: `model`,
`expert_heat`, `kernel_choice`, `pilot_profile`, `conformance`, `api_token`, `placement_history`.

**`/v1/models` is reclaimed.** It returned agents wearing model costumes, with an `openai_compat_note`
pointing at the other port for the real thing. The agents catalog stays on the `:9091` OpenAI-compat
listener where it belongs; on `:9090` the route now means the admission registry — `GET` (list and by
id, with conformance stages), `POST` (register an already-converted model), `PUT .../verdict`,
`DELETE`, `.../plan`, `.../heat`. The token-gate test asserted the old meaning and now asserts the new
one.

**The routing loop is closed.** `AgentScheduler::resolve_backend` stays pure — it takes an
`AdmissionRecord` — and `resolve_backend_for()` looks the record up. The two verdict enums
(`mm::ModelVerdict` for storage, `soma::Verdict` for the engine) are mapped with an explicit switch
rather than a cast, so adding a value to one cannot silently reinterpret rows written under the other.

```
model_registry_makes_soma_routable
    before admission: auto -> llama-cpp, AND override=soma -> llama-cpp
    admit with verdict=stream: the same agent config -> soma, reason cites the verdict
    override=fallback still wins (it can only be more conservative)
    resident-only -> llama-cpp, but override=soma IS honoured (economics)
    reject -> llama-cpp, and override=soma is REFUSED (conformance)
    "nonsense" parses to reject, not to a licence to stream
    re-upsert on the same arch_hash updates rather than duplicating
    verdict survives close/reopen; delete cascades; a scheduler with no registry
      still routes to the fallback
```

**Registration is separate from admission, and that is the honest split.** `admit()` — fetch, convert,
compile the tokenizer, run the ladder, profile — is the offline pipeline in `tools/admission/` and
returns an error saying exactly that. What exists is everything *downstream* of it: `upsert()` is the
write primitive `admit()` would end with, so a model converted offline can become evidence here. Keyed
on `arch_hash`, because that is the model's identity — requantized weights are a different row, since
they have a different verdict.

**`plan_for_host()` returns the admission-host verdict and labels it as such.** It cannot compute the
effective one: the verdict is a property of `(model, quantization, host budget)`, which is why
`soma plan --json` runs on the *target node*. Every model JSON carries `verdict_scope:
"admission-host"` for the same reason — a reader who takes the stored verdict as "what this node will
do" will eventually be wrong.

#### Admission, in-process and over SSE

`POST /v1/models/admit` runs the pipeline and streams `AdmissionProgress` frames:
`fetch → convert → tokenize → conformance → profile → finalize`. It ends in exactly one frame with
`done: true`, and `last_error` distinguishes failure from success there — because a stream that simply
goes quiet is indistinguishable from a network fault.

**Orchestration, not reimplementation.** The converter and the tokenizer compiler stay Python: they read
HF checkpoints and that ecosystem is theirs. Control runs them through `run_streamed_command()` and
parses their output. `convert.py` already prints `layer 12/48 3.40 GB` per layer with `flush=True`, so
the fraction tracks the model's real shape rather than an assumed one. `tools/admission/` remains never
a runtime dependency — it is a dependency of *admitting*, which happens once per model.

`process_exec` moved from `node/` to `common/` to make that possible. Nothing in it was ever
node-specific; a subprocess whose output is streamed line by line is the visible alternative to running
something blind, and that is as true of a model conversion as of a llama.cpp build.

Three routes beyond the stream itself, because an hours-long operation needs more than one connection:
`GET /v1/models/admissions` lists every operation this process has run, `GET
/v1/models/admissions/{op}` rejoins one (sending the current state first, so a client joining late — or
after the end — learns where things stand immediately), and `POST .../cancel` stops one. **A client
disconnecting does not cancel anything**: hours of conversion must not be discarded because a browser
tab closed.

```
admission_pipeline_runs_and_reports            (drives the real `soma` binary)
    a nonexistent source fails BEFORE an operation exists — nothing to list
    staged frames arrive: profile, finalize
    exactly one terminal frame, with finished_at >= started_at
    the row carries n_experts/top_k/active_fraction from `soma plan --json`
    the operation is retrievable after it ends; a late watcher gets the outcome
    cancelling a finished operation is refused, not silently accepted
```

**Two gaps this exposed and closed.** `compute_plan(model_dir)` was a stub returning "lands with the
admission converter's arch.json output" — so nothing produced an `arch_hash`, which is the identity the
registry keys every row on. Every unconverted model would have collided on the empty string. It now
reads `config.json` (which an HF checkpoint and a converted container both carry — a second description
file that had to agree with the first is how they drift) and stamps the hash. And the plan document
gained the topology and per-expert economics it was missing: `attention_family`, `n_layers`,
`n_moe_layers`, `n_experts`, `top_k`, `expert_bytes`, `active_fraction`. Control's registry denormalizes
exactly those, and the plan is the only view of the model it has.

**Conformance is not run, and the record says so** rather than implying a pass. Stages 1–2 run against
committed tiny fixtures in CI; stage 3 needs a reference run for *this* model, which is a separate
artifact this pipeline does not have. Writing a `passed` row it did not earn would make the verdict look
validated when it is only computed.

**A failed tokenizer compilation is not fatal.** A container without one serves token ids rather than
text — degraded but honest — and refusing the whole admission over it would discard hours of conversion.

#### Scoped authorization

Auth was ONE flat bearer token compared with a plain `!=`, gating every `/v1/*` path identically, and
entirely opt-in — an empty `external_api_token` left the whole surface open. Three scopes now, and
**the split is by blast radius, not by resource**: `GET /v1/agents/{id}/memories` is `read` while
`POST .../memories` is `chat`, because they touch the same rows and only one can change anything.

| scope | covers |
|---|---|
| `read` | every GET, every telemetry stream — no side effects |
| `chat` | chat, conversations, memories, uploads, speech — bounded, per-agent |
| `operator` | admission, deletes, verdict and backend overrides, token minting |

`operator` implies the other two and `chat` implies `read`: sending a message you cannot then fetch is
not a coherent permission, and a credential that can delete an agent but not read it is a distinction
nobody wants to administer. DELETEs are `operator` even when they are per-agent and small — "bounded"
is about what a mistake costs, not how many rows it touches.

**The legacy token is grandfathered and never becomes a row.** Matched before the table lookup and
granted every scope, so existing deployments keep working unchanged — and rotating it in config takes
effect immediately rather than leaving a stale grant behind, which is exactly what inserting it at
startup would have caused.

**Tokens are stored hashed.** `api_token` keeps `token_sha256` and never the token, so a leaked backup
does not hand over working credentials. `POST /v1/tokens` returns the secret once; a caller that loses
it mints another. Revoking sets `revoked_at` rather than deleting — the row is the audit trail, and a
deleted token cannot answer "what was this credential allowed to do".

**Coverage is checked at startup against the server's own registrations**, not against a second list
someone maintains. `HttpServer` records each route as it is registered, and `listen()` refuses to start
if any `/v1/*` handler has no scope entry: defaulting an unlisted route to `read` would silently
under-protect a new mutation, and defaulting to `operator` would silently break a new GET.

That check earned its keep immediately. `POST /v1/agents/{id}/attachments` is registered through
`PostUpload` rather than `Post`, so it was missing from the first draft of the table — and the server
refused to start until it was added. A grep-built table would have shipped with an unscoped upload route.

```
route_scopes_and_token_store
    the coverage check reports a missing route (not just accepts a good list)
    read !-> chat !-> operator; chat -> read; operator -> everything
    "read,opereator" is REJECTED, not silently reduced to read
    the stored hash is not the token, and the raw token is not a lookup key
    revoke keeps the row; a second revoke reports it is already gone
    a zero-scope token is refused rather than minted useless

scope_negatives_over_http                (a live server, three minted tokens)
    read    GET /v1/{nodes,agents,models} -> 200
            POST /v1/models/admit -> 403 naming required=operator, granted=read
            DELETE /v1/agents/:id -> 403, and the agent is still there
    chat    POST .../conversations -> 200   (chat implies read)
            POST /v1/models/admit -> 403;  GET /v1/tokens -> 403
    operator POST /v1/models/admit -> 400   authorization PASSED; rejected on
                                            its merits, which is the distinction
            DELETE /v1/agents/:id -> 200    the mutation both others were refused
    a revoked token stops working immediately -> 403
    an unknown token -> 403; no token at all -> 401
```

**The negatives are asserted on the wire, not just in the predicate**, because a scope table that is
right in a unit test and unreached by the middleware protects nothing — and that failure is silent, since
every request simply succeeds. The `read` token getting 200 on a GET and 403 on a POST against the same
server is what makes the test self-verifying: auth off would give 200 for both, auth broken-closed 403
for both, and only a working table produces the split.

Confirmed by breaking it: changing `POST /v1/models/admit` to `Scope::Read` turns the test red on exactly
the six assertions that matter, and reverting turns it green. A test that has never been seen to fail is
a test whose passing means nothing.

**One bug this surfaced in my own wiring:** I first configured the authorizer only from
`set_model_registry()`, so any caller that never attached a registry — including the reliability
suite — ran with auth entirely off. Configured from the constructor now, since the legacy token is a
constructor argument and the registry is optional.

#### The telemetry feed

`GET /internal/telemetry` (SSE), `/internal/heat` (snapshot) and `/internal/telemetry/dump` (the G3 text
view). `TelemetryChannel` was a declared header with no implementation; it exists now.

**One channel samples, N watchers read.** Not one sampler per connection — the whole premise is that
aggregation costs the engine once regardless of how many clients are looking, and a per-connection
sampler would make that cost linear in watchers, which is the thing the design says it avoids. The
channel retunes on every attach *and* detach: the fastest watcher sets the rate and full resolution is
sticky only while someone is asking for it, so a departing client's higher rate does not persist.

**The rate ceiling is the engine's.** `?hz=` is a request; `?hz=1000` yields 10 Hz rather than an error,
because the limit is a property of the engine rather than a mistake by the caller. `?hz=banana` falls
back to the default for the same reason. Measured on the wire: 4 frames in 1.5 s at the default, 11 at
`hz=10`, and 15 at `hz=1000` — clamped, not obeyed.

**Downsampling is the default and it conserves counts.** A bucketed grid that dropped counts on the way
down would make the brain view understate load exactly where it is highest, so `bucket_heat()` is tested
for conservation, not just for size:

```
soma_telemetry_g6
    Qwen3-30B-A3B  48x128    6144 -> 1536 cells (2x2), all 6144 counts preserved
    hypothetical   60x256   15360 -> 3840 cells (2x2)
    pathological  128x512   65536 -> 4096 cells (4x4), still a 32x128 GRID
    DeepSeek-V2-Lite 27x64   under the cap: passed through 1:1, bucket factors stay 1
    a mostly-cold bucket reports COLD, not the one warm cell in it
    empty snapshot / zero cap: no crash, no claimed bucketing

engine_g5 §11                                          (against a live engine)
    /internal/heat is bucketed by default; ?resolution=full is an opt-in
    default ~2 Hz; hz=10 faster; hz=1000 CLAMPED; hz=banana falls back
```

**Both axes are reduced, not one.** Bucketing only experts turns 128x512 into 128x1 — every layer keeps
its row and the expert axis vanishes, so the display shows which layers are hot and nothing about which
experts. Reducing both keeps the grid a grid, which is the entire point of the brain view.

Two smaller decisions worth stating. Heat cells go out as **flat parallel arrays** rather than an array
of objects: at 4096 cells the object form is roughly 6× the bytes for identical information, and this
ships on every tick. And a watcher that stops draining has its **oldest** frames dropped at 32 queued —
telemetry is a sample of *now*, so a stale frame delivered late is worse than a gap.

**The container fixtures are usable now.** They carried no `config.json`, so `soma serve` and `soma plan`
both refused them and the streaming path could not be exercised from the repo at all. `convert.py`
already copies it — the committed fixtures simply predate that fix — so backfilling the five containers
from their tiny checkpoints was enough. `soma_engine_g5` now takes the container as a third argument and
CI serves one:

```
the container is served as STREAMED
heat reports the model's real dimensions             4x16
with NON-ZERO counts — experts actually fired        hottest cell = 8
the cache reports a hit rate, so lookups happened
```

That closes a real hole in the coverage. Every telemetry number before this came from a resident model,
where the grid is empty by construction — so the feed was exercised and **the thing it reports was
not**. A telemetry route emitting structurally-correct zeros is indistinguishable from one wired to
nothing, and only a streamed model tells the two apart.

#### Republication — engine → node → control

`GET /v1/engines`, `/v1/engines/{id}/telemetry` (SSE), `/heat` and `/slots`. Cluster-wide an "engine" is
a **(node, slot) pair**, so the id is the slot id and the node is *discovered* rather than supplied — the
answer changes on every eviction, and a client should not have to track it.

**The node forwards what the descriptor names.** `EngineDescriptor` gained `telemetry_path` and
`heat_path`; the node proxies those and never learns that Soma calls one `/internal/telemetry`. An
engine with no telemetry surface — llama.cpp, whose paths are empty — answers **501**, and that survives
both hops: "this engine does not publish that" and "no such engine" are different facts, and only one is
the caller's mistake.

**`hz` is forwarded, never re-clamped.** The ceiling is the engine's, and a second clamp at the node or
at control would be a second number to keep in step with it. Bucketing is likewise the default at every
hop rather than defaulted back to full by a middle layer.

```
engine_telemetry_republication              (control against a live stand-in node)
    GET /v1/engines lists both slots WITH their node, and sums the tier summary
    /heat proxies through and passes the body back
    no ?resolution -> the node sees no query at all (bucketed all the way down)
    ?resolution=full -> the node sees "full"
    /slots returns the sequence list
    an unknown engine -> 404 "no such engine"
    an engine with no telemetry -> 501 SURVIVES the hop, not flattened to 500
    a node that has gone away -> 502 naming the node, not 500
```

The registry is driven through the **real health poll** rather than a setter: `connected` and the slot
list arrive that way in the running system, and faking them would have tested a state it never reaches.

**One bug, and it was in the test.** The stand-in node registered `/api/node/engines/:slot/heat` before
a more specific `/api/node/engines/no-telemetry/heat`, and httplib matches in registration order — so
the parameterised route swallowed both and the 501 case returned 200. The real node has no such problem
because it consults the descriptor *inside* the parameterised handler; the fake now does the same, which
is both correct and a closer model of the thing it stands in for.

##### What is NOT ported yet, stated plainly

1. ~~**Fetching is not implemented.**~~ **Done** — see [The fetch stage](#the-fetch-stage) below.
2. **The OpenAI-compat listener on `:9091` still uses the flat token.** It serves one surface with one
   meaning — chat against agents-as-models — so a scope table for it would have one row. Worth doing
   when it grows a second capability, not before.

**Build:** the supervision rebuild — `EngineProcess`, `EngineSupervisor`, `EngineDescriptor` +
registry, `EngineClient` with virtual `stream_complete`, `KvCheckpointBackend` ×2, `PlacementEngine`,
`ResourceFootprint`, `control.db`. llama.cpp **ported onto** these abstractions.

**Gate:**
- The existing `reliability_tests` suite still passes. The fallback must not regress; this is the gate
  where that risk is real. **Done, continuously** — 24/24 on every run through G8. Marked explicitly
  because a criterion satisfied on every commit and never recorded reads, on a count, exactly like one
  that was never met.
- **Mixtral-8x7B admits as `resident-only` and routes to the fallback with no operator action.**
  **Done** — `routing_g5` §4 asserts `mp.verdict == ResidentOnly` and `select_backend(cfg(), …).choice ==
  Fallback`, where `cfg()` carries no override, which is what "no operator action" means. It also pins
  the reason: on a tight host Mixtral and Qwen3 get DIFFERENT verdicts, and Mixtral moves >3x the bytes
  per token despite comparable size — active fraction, not size, is what makes streaming viable.
  The negative fixture is a pass condition, not a footnote — a system that cannot recognize what it is bad
  at will happily be bad at it.
- A Soma agent and a fallback agent run **concurrently on the same node**. **Done** — `engine_g5` §12
  runs the real `soma` binary and a real `llama-server` under one supervisor: distinct slots, distinct
  ports (8240/8241), both `Ready` simultaneously, each attributed to its own descriptor, each reachable
  through its own client, and unloading one leaves the other Ready. See
  [Two engines, one supervisor](#two-engines-one-supervisor).
- A converted model directory sizes correctly. Assert the old flat-2048 MB path is gone by checking a
  multi-shard fallback model too. **Done** — `measure_model_bytes()` sizes directories recursively and
  `multi_shard_directory_sizes_correctly` asserts two different directories no longer size identically.
- Killing a Soma process is detected by the watchdog ~~within one poll interval~~ **immediately**, and the
  engine transitions to `Error` rather than advertising `Ready`. **Done** — `engine_g5` §6 kills the child
  by port and asserts `SlotState::Ready` -> `SlotState::Error`, that the slot RECORD survives (a removed
  record reads as an engine that was never there), and that `acquire()` then refuses it. The original
  wording described an implementation that was never built: the watchdog is not a poll loop but a blocking
  `WaitForSingleObject(…, INFINITE)` / `waitpid` on its own thread, so there is no interval to be within.
  Struck rather than reworded, because a criterion quietly edited to match the code stops being a check on
  it. §3 asserts the same crash one layer down at `ProcessState::Crashed`; that one is NOT this criterion,
  and reading it as such is the mistake this pass was looking for.
- `capacity_pressure` as a structured code drives evict-and-retry, with the substring matcher deleted.
  **Half done, and the second half is now a deployment decision rather than a code one** — the structured
  code drives eviction and is tested; the matcher survives on purpose. See
  [capacity_pressure](#capacity_pressure-the-signal-with-no-coverage).
- Suspend on a multi-sequence llama.cpp slot returns 409 rather than silently saving sequence 0.
  **Done** — `LlamaKvBackend::supports_multi_sequence()` is false and `EngineSupervisor::suspend`
  refuses with `Unsupported` / `unsupported_content`. The HTTP status mapping lands with the route at G6.

---

## G6 — API surface complete

**Build:** every route in [external-api.md](external-api.md); the three-scope authorizer; the token
store; SSE telemetry with throttling; `/v1/models` reclamation.

**Gate:**
- `require_complete_coverage()` green: every registered handler has a scope-table entry. Startup fails
  otherwise. **Done** — checked against `HttpServer::registered_routes()` in `listen()`, and it caught
  an unscoped upload route on its first run.
- A `read`-scoped token can stream telemetry and **cannot** admit, delete, or chat. Test the negative.
  **Done** over HTTP — `scope_negatives_over_http`. The telemetry half of it lands with the telemetry
  routes; the admit/delete/chat negatives are asserted against a live server.
- A `chat`-scoped token cannot admit. Test the negative. **Done** over HTTP.
- The legacy flat token still works as all-scopes. **Done** — `control_api_external_token_gate` is
  unchanged and passing, which is the regression that matters.
- `hz` clamps to 10; `resolution=full` requires the explicit parameter. **Done** in the ENGINE —
  measured on the wire, `?hz=1000` produces 15 frames in 1.5 s rather than 1500. Control's
  `/v1/engines/{id}/telemetry` re-publishes it and inherits the ceiling.
- The ladder's stage 1 runs at admission rather than reporting `skipped`. **Done** — the pipeline builds
  the oracle itself; `fp32_tiny_tf` passed on OLMoE at `max_abs=1.07e-06` against a `2e-03` tolerance.
- A client requesting maximum telemetry does not measurably affect chat latency — the
  aggregation-in-engine claim, verified rather than asserted. **MEASURED** on the admitted OLMoE
  (16x64 = 1024 expert cells), 10 order-balanced pairs at the 10 Hz ceiling with `resolution=full`:
  **median delta -0.74%**, range -4.39% to +2.07%, against baseline noise of 3.05%. Within noise. Not
  the 60k-expert model the line imagines — see the caveat under
  [Telemetry against a live model](#telemetry-against-a-live-model).
- **The converse was badly false and the gate never asked**: a chat COLLAPSED telemetry, 17.3 frames/s
  idle to 1.3/s during generation. Found by this measurement, **fixed** as D11 — now 16.1/s. (A finding
  rather than a criterion; it sits in this list because it is the other half of the line above it.)
- Image content parts return **422**, not a dropped part. **Done** — the refusal now consults the ENGINE
  that will serve the agent, not only the agent's profile. Verified live: an image sent to a
  vision-enabled agent whose model routes to Soma answers
  `422 {"error":"the 'soma' engine serving this agent does not accept images (…verdict=hybrid)"}`.
  See [Images and the engine that serves them](#images-and-the-engine-that-serves-them).
- A sequence restored into a **different process** continues its conversation rather than starting a new
  one that happens to share a cache. **Done** — checkpoint format v3 carries `rng_state` and the emitted
  history; `soma_checkpoint_g3` §2b resumes into a fresh `Scheduler` and compares against both the
  uninterrupted tail and a cold start. The cold start must *diverge*, or the check is vacuous.
- Every streamed delta is **text on its own** — no frame ends mid-codepoint — and the deltas concatenate
  to exactly what the non-streaming form returns. **Done** — `CompiledTokenizer::Streamer` replaces the
  O(n²) re-decode; checked in `soma_tokenizer_g0` against the corpus, against a synthetic byte-fallback
  split, and against `flush()` with a partial codepoint held, and end-to-end in `soma_engine_g5` §9b.

---

## G7 — FTXUI dashboards on that API

**Gate:**
- Every Soma panel's data comes from `/v1/*`. **No panel reaches into in-process engine state.**
  Grep-verifiable, and worth verifying: the existing TUI already mixes direct access and loopback HTTP,
  so the temptation is live. **Done** — `tools/ci/check_ui_api.py` runs as `mm_ui_api_check`; see
  [The dashboard's data source](#the-dashboards-data-source).
- The brain grid renders a 48×128 model bucketed to ≤4096 cells at 2 Hz without visible cost. **Done** —
  0.017 ms/frame for the layout, against a 500 ms tick, and the panel draws it as tab 8.
- The tier bar shows the VRAM tier present and empty — the declared-but-stubbed design, visible and
  honest rather than hidden. **Done** — and it splits by backend, which `tier_summary` does not.
- Panels degrade gracefully when an engine is a fallback and has no tier/heat data. **Done** — four
  distinct states, each asserted as rendered text.

#### The dashboard's data source

`include/control/soma_dashboard.hpp` + `src/control/soma_dashboard.cpp`. One translation unit that owns
the polling and the layout, and holds a reference to nothing in-process — its only inputs are a base URL
and a token.

**The no-reach-through rule is made mechanical rather than aspirational.** `check_ui_api.py` fails the
build if that TU includes `node_registry.hpp`, `agent_manager.hpp`, `agent_scheduler.hpp`,
`model_registry.hpp`, any node-side engine header, or anything from `soma/`. It also asserts the TU is
actually in `CMakeLists.txt`, because a rule guarding a file nobody compiles passes every check while
shipping nothing. Adding a forbidden include turns it red, which is how it was confirmed.

The reason to bother: a TUI that reads private state is a **second client with privileges no other
client has**, and P1's claim that the API is the single control plane then becomes untestable — every
gap in `/v1/*` stays invisible for exactly as long as the only serious consumer does not need the API.
`ControlUI` already takes `NodeRegistry&`, `AgentManager&` and `AgentScheduler&` by reference, so a panel
needing one number is four keystrokes from taking it directly. A comment asking nicely would not survive.

**The layout half is pure** — no FTXUI, no HTTP, no clock — so the reduction is checkable as arithmetic
rather than by looking at a terminal. That matters more here than usual: a grid that is subtly wrong
looks exactly like a model that routes oddly, and an operator staring at a heat map has no way to tell
those apart. `mm_dashboard_g7` checks that counts are **conserved** through the reduction (against a
deliberately non-uniform fixture, since a flat one passes a broken implementation), that the strides use
ceiling division so a ragged last row is not silently dropped — those are the layers nearest the output,
where an anomaly is most worth seeing — that the **coldest** tier in a bucket wins, and that "fired
once" renders differently from "never fired".

Two bugs the tests caught while being written. The tier reduction was `max()` against a `tier` field
defaulting to `Disk`, so **every** bucket reported Disk — the panel would have said the whole model was
on disk, and a test that only ever looked at a bucket genuinely containing a disk cell would have passed
it; the all-Vram case is what catches it. And `layout_heat` indexed `rows*cols` into `counts` without
checking the frame's arrays matched its dimensions — a segfault on a malformed frame, and on a
merely-short one, plausible-looking heat read from past the end.

Intensity is relative to the frame's hottest cell rather than absolute, so the same grid is readable
after ten tokens and after ten million; the test asserts 1000× the traffic renders an identical shape.

#### The panels

`include/control/soma_panels.hpp` + `src/control/soma_panels.cpp`, wired into `ControlUI` as **tab 8**.
A second guarded TU rather than code in `control_ui.cpp`, for the reason the guard exists: that file
holds `NodeRegistry&`, `AgentManager&` and `AgentScheduler&`, and a panel written there could read any
of them without anyone noticing.

**Two channels for two independent facts.** Colour is the memory TIER, brightness is how often the
expert fired. Collapsing them into one ramp is the obvious simplification and it destroys the panel's
reason to exist — a hot expert on disk and a hot expert in RAM become the same cell, and the first is
the one costing throughput.

**Every renderer takes a snapshot, not the dashboard**, so each is a pure function of data it was
handed. That is what lets the panels be drawn to an off-screen `ftxui::Screen` and asserted as text.
The states worth checking are the ones nobody opens the app in — a fallback engine, a Soma engine that
has not routed a token, a selection pointing at an engine that vanished between frames, an empty list
before the first poll versus after one — and "a human looked at it" checks none of them. Sixteen
assertions cover them; each degenerate case says *which* it is, because an empty grid and a cold grid
are pixel-identical and mean completely different things.

**A `--preview` mode on the test binary** draws the whole tab against a synthetic cluster and prints
it. Assertions prove the panel says the right words; they say nothing about whether the columns line up
or the grid is legible, and a TUI that cannot be looked at without building a cluster does not get
looked at. It found three things immediately:

- **The tier bar was reading `tier_summary`, which sums every engine's VRAM.** A fallback's 22 GB
  appeared under the tier the gate wants shown as declared-and-empty — an operator would have seen Soma
  using VRAM in a release that is CPU-only by design, which is the exact misreading this panel exists to
  prevent. It now splits by backend and reports the fallback's separately, labelled as not Soma.
- The legend truncated mid-word at the width a 64-column grid leaves, and the first thing it lost was
  **"disk"** — the tier the panel exists to make findable. Two lines now.
- Letting the tier bar flex gave it half the terminal for four short rows. Fixed at 46 columns, with the
  grid taking the rest.

The caption reports the model's real shape and the two reductions *separately* — the engine buckets to
bound its payload, the panel reduces to fit a terminal, and conflating them would tell the operator the
model is a shape it is not. Staleness is per field rather than one timestamp for the snapshot: engines
and heat are separate requests, one can fail while the other succeeds, and a single "stale" flag would
blank a panel that is fine.

---

#### The fetch stage

`tools/admission/fetch.py`, driven from `run_admission`. `admit()` now takes a HuggingFace repo id —
optionally `repo@revision` — as well as a local directory. A local path wins if one exists at that name;
anything else has to pass as a repo id, and both are checked *before* the operation exists, so a typo is
a 400 rather than an operation that appears to start and dies a second later.

**The repo id is validated because it becomes a directory name.** `../../etc` is a legal-looking string
and an illegal path, and this field arrives over HTTP from an `operator`-scoped caller. `valid_repo_id`
is a free function rather than a detail of the pipeline, so the rule is tested for itself — "the
download failed" is not evidence that path containment held. It is enforced twice, in C++ and again in
`fetch.py`, and `check_fetch_selection.py` tests the Python half against the same table.

**What it deliberately does not download.** A published checkpoint routinely ships the same weights
three times — safetensors, PyTorch `.bin`, and a TF/Flax copy. Taking everything triples the transfer
and the disk for bytes nothing will ever read; on the synthetic repo in the test, selection cuts 41 GB
to 9 GB. Framework duplicates are never transferred. `.bin` files are skipped when safetensors exist —
**and so is `pytorch_model.bin.index.json`**, which is the subtle half: an index left behind after its
shards were dropped points conversion at files that are not there.

**A repo with no safetensors is refused by default.** Converting `.bin` weights means unpickling them,
which executes code from the repo. `admission_allow_pickle` is the operator saying they meant it, per
deployment; the error names the reason rather than the file. Auth is whatever `huggingface_hub` already
resolves — `HF_TOKEN` or a cached login — and nothing here reads, prints, or stores a credential.

**Progress comes from watching the output directory**, not from hooking the downloader's progress bars.
A 20 GB shard is one file, so per-file granularity reports nothing for twenty minutes; and any
implementation that puts bytes on disk is observable this way, which a `tqdm` hook is not. `fetch` is
the only stage that populates `bytes_done`/`bytes_total`, because it is the only one whose remaining
time a client can estimate — those two fields existed on `AdmissionProgress` and had never been set.

**A bug found on the way past.** `step` and `total_steps` were written independently: a container
admission advertised `total_steps = 2` and then emitted steps 3, 4 and 5, which a progress bar renders
as 250%. Both now come from one ordered stage list per run — `fetch` is present only when fetching, and
a container admission is honestly 3 steps rather than 5-of-2.

**Testing a network stage without a network.** A stub `fetch.py` in a temp tools directory emits exactly
the line protocol the real one promises. `convert.py` is stubbed too — copying the committed container —
so the run reaches the *real* `soma plan --json` and all six stages execute end to end. The three cases
that matter are the ones a mock can actually cover: a fetch that works, one that exits non-zero, and one
that **exits 0 having produced nothing**. The last is why the resolved path is checked rather than
trusted; without it, conversion is handed a path that does not exist and the error names `convert.py`.
Reverting either guard turns the test red, which is how it was confirmed to be load-bearing.

Still not implemented, and named rather than stubbed: resumption across a *control restart* (an
interrupted transfer restarts the operation, though `snapshot_download` skips what is already on disk),
and any bandwidth limit.

#### The conformance stage

`soma conform --model-dir DIR --json`, run from `run_admission` and written to the `conformance` table.
A subcommand of the engine binary for the same reason `plan` is: the codec under test is the one the
engine uses, and a second implementation living in control is how the two come to disagree.

**The gate is not "does it pass". It is that a stage which did not run says so.** Most of this ladder
cannot run on a serving host — `fp32_tiny_tf` and `real_logit_kl` need a `transformers` oracle for the
*specific* model, which is a separate artifact — and the tempting move is to leave them out, since the
ladder reads as incomplete without them. Leaving them out is indistinguishable from passing them. So
`ConformanceEntry` gained a third state, `status ∈ passed | failed | skipped`, and each skip carries
what it would need. `passed` stays as a column for clients written against the old shape, but a boolean
cannot express "did not run", and that is the most common answer here.

Two stages do run, and both are model-specific rather than fixture-specific:

- **`quant_codec`** quantizes the container's own dense weights with the container's own declared
  formats, dequantizes, and checks two things: measured bits/weight against a formula written out
  independently, and round-trip relative RMS against a per-format ceiling. The dense tensors are the
  right subject — F32 on disk, this model's real weight distribution, and *bounded*, because the experts
  are the gigabytes and they are already quantized, so there is no fp32 original to compare them
  against. The bits/weight formula is deliberately not `quantized_tensor_bytes()`: comparing the
  implementation against itself passes through any packing bug that is consistent about its size. The
  rel_rms ceilings are generous by ~3× against the G1 table, on purpose — this is a **packing** check,
  and a mis-packed nibble or wrong group stride lands near 1.0, two orders of magnitude out. A tight
  bound would fail honest models and catch nothing extra.
- **`tokenizer_roundtrip`** finally implements `verify_roundtrip()`, which was declared with "ADMISSION
  IS GATED ON THIS" in its doc comment and had no body. It takes the oracle's **ids** rather than a
  digest of them: a hash can only say "different", while the ids say which case, which position, and
  what was expected — and the digest bought nothing, since both sides are on the same host. Decode is
  checked against HF's ids, not ours, so an encode bug that a decode bug happens to invert cannot pass
  both halves.

**A failed stage is a `reject` verdict, not a failed request.** The operator asked whether Soma can run
this model; "no, and here is the stage that says so" is an answer, and a rejected model is a
successfully admitted record meaning "route this to the fallback". The plan's own reason survives
alongside it — two different findings, neither erasing the other. `soma conform` exits 0 even when a
stage fails, reserving non-zero for *could not run*, which the caller has to be able to tell apart.

**A live bug, found while wiring this up.** `compile_tokenizer.py --out` takes a **directory**, and the
pipeline was handing it `<container>/tokenizer.soma`. The script created a directory of that name and
wrote the tokenizer inside it; the engine looks for a *file* at that path, found a directory, and
**every model admitted through this pipeline served raw token ids**. Nothing failed — the tokenizer was
simply never there. Passing the container directory also puts `tokenizer_oracle.bin` where the
conformance stage needs it, which is how the bug surfaced at all.

**Schema v2** rebuilds the `conformance` table: SQLite cannot alter a CHECK constraint, and the stage
list gained two names. Existing rows carry forward with `status` derived from `passed`, which is the
honest reading of what they meant when they were written.

**Testing it takes two half-complete fixtures**, because neither one alone can show the distinction that
matters. The container fixture has quantized formats and dense weights but no compiled tokenizer — its
model's vocabulary is 512 and the committed tokenizer's is 151936, so they are not the same model, and
copying one in made `soma_engine_g5`'s heat section go to zero. The tokenizer fixture has the reverse.
Each proves one stage runs and the other reports why it could not. A third case — an empty directory —
asserts that *every stage skipped* reports `passed: false`, since "nothing was checked" reading as a
pass is the exact failure this file exists to prevent.

**`fp32_tiny_tf` and `real_logit_kl` both run now** — see [The conformance oracle](#the-conformance-oracle)
and [The stage that looks at the weights](#the-stage-that-looks-at-the-weights). Still recorded as
skipped rather than stubbed: `accuracy_floor`, for which no downstream task harness exists at all.

#### The conformance oracle

Ladder stage 1 — teacher-forced logits against `transformers`, plus greedy token-exactness — was
reported as `skipped` because it needs an oracle for the SPECIFIC architecture and the pipeline had no
way to make one. It does now: a new `oracle` stage runs `make_oracle.py` on the source and lifts the
result into `<container>/conformance/`, so the fixture travels with the admission record.

**What it validates, stated precisely, because the obvious reading is wrong.** The fixture is
tiny-RANDOM weights with the real config — every dimensional field shrunk, every semantic field
preserved verbatim. So this says nothing about the admitted checkpoint. It says the engine implements
this ARCHITECTURE the way `transformers` does. That is the more valuable claim: a real checkpoint can be
approximately right in ways that hide a bug for weeks, while a tiny-random one is either exactly right
or obviously wrong.

Measured on OLMoE-1B-7B-0924, admitted end to end with the pipeline building its own fixture
(16 layers → 4, 64 experts → 16, vocab 50304 → 512):

```
fp32_tiny_tf   passed   max_abs=1.07e-06 (tol 2e-03)  pos0=5.81e-07  greedy=256 exact
```

Six orders of magnitude inside tolerance, and 256 greedy tokens matching exactly. The ladder now reports
**3 of 5 stages ran** for a model admitted from scratch, against 2 before.

**The comparison moved into the library** — `soma/conformance.hpp`, `run_fp32_conformance()`. It had two
callers the moment `soma conform` needed it, and two implementations would have meant two sets of
tolerances, two oracle parsers, and two opinions about what "passes" means. `soma_conformance_g0` keeps
the fixture walk and the reporting, which is what a test is for.

`max_abs_pos0` is reported on every result, pass or fail, because it BISECTS a failure rather than
merely describing one: at t=0 RoPE is the identity and attention is a softmax over one element, so a
divergence already present at 0 cannot be either — it is projection, qk-norm, routing, or the expert
MLP. One clean at 0 and growing with t is the opposite. That single number is what turned four rounds of
failed reasoning about MLA into one afternoon; see G4.

**Failing to build the oracle is not fatal**, for the same reason a failed tokenizer compile is not: the
model is still admissible and still routable, and discarding hours of conversion over a missing fixture
would be the wrong trade. Stage 1 then reports `skipped`, which is the honest result.

#### The architecture check

`soma plan --model-dir <source> --json`, run **before** `convert.py` rather than after it. The pipeline
already planned — it just did so at the end, on the container, which meant an architecture Soma cannot
read was discovered after six hours of writing one.

There are two distinct failures here and they get different answers:

- **`plan` refuses the source.** `adapt_hf_config`'s table *is* the registry of architectures this engine
  understands, and an unknown `model_type` stops there. A container built from a config Soma cannot
  parse is gigabytes nothing can read, so this **fails the admission** with plan's own message. This is
  the case reachable today.
- **`plan` succeeds and reports `arch_supported: false`.** The config parsed, but `resolve_f32_backend`
  has no forward for its attention family — `mla+dsa` is the standing example, deliberately refused
  rather than silently run as dense MLA. That is not an error: it is a **reject record** meaning "route
  this to the fallback", which is a successful admission. Conversion is skipped because no host will
  ever read the container.

`arch_supported` is a new field on `PlanDocument`, checked first in `compute_plan` because it dominates
every economic branch below it — the alternative is a confident throughput projection for a forward that
does not exist. It is deliberately **not** the same signal as a reject verdict: a model rejected on
economics may still be worth converting, since the verdict is a property of `(model, quantization,
host)` and a node with more RAM can reach a different one from the same container. Throwing away a
conversion because *this* host said no would be a category error. Only `arch_supported` short-circuits.

A missing `arch_supported` field is read as `true`, so a control talking to an older `soma` degrades to
the previous behaviour — convert and find out — rather than refusing every model.

**The test asserts the ordering, not just the outcome.** Any check that rejects an unparseable config
passes an "it failed" assertion; the one that matters is that it failed *first*. The stub convert writes
a whole container, so `admission_fetch_stage` requires that directory to be absent and the `tokenize`
stage never to appear. Disabling the check turns both red — the admission succeeds on a nonsense
architecture and records a model, which is precisely the waste this prevents.

---

## G8 — admission pipeline self-service

**Gate:**
- A GQA MoE checkpoint **not seen during development** goes from `POST /v1/models/admit` to a served
  agent with **no C++ change**. That is the whole gate. **`source` may now be a repo id** — see
  [The fetch stage](#the-fetch-stage) — so the gate no longer presumes an operator downloaded the
  weights by hand first.
  **PASSED END TO END** on `allenai/OLMoE-1B-7B-0924`, 2026-08-06 — a real 7B MoE never seen during
  development, from a bare repo id to generated tokens. Admission: 4m21s, verdict `hybrid`. Serving:
  `{"backend":"soma","state":"ready"}` on the container, streaming SSE deltas —
  *"A colour is the colour of a pigment or dye."* No C++ change was needed to ADMIT it; two defects
  (D7, D8) had to be fixed before it would SERVE, and both were in code paths no fixture exercises.
- An architecture with no backend fails with a clear message, before conversion spends hours. **Done** —
  see [The architecture check](#the-architecture-check). `soma plan` now runs on the *source* first, and
  `admission_fetch_stage` proves conversion never started by asserting the container directory does not
  exist.
- The ladder records what it ran and what it did not, and a failed stage yields `reject` without failing
  the request. **Done** — see [The conformance stage](#the-conformance-stage). `quant_codec` and
  `tokenizer_roundtrip` run at admission; the three that need a `transformers` oracle are recorded as
  `skipped` with what they would need.
- Re-admission with different quantization produces a new `arch_hash` and invalidates KV checkpoints;
  re-profiling does **not**. **Done** — see [Requantization is a new admission](#requantization-is-a-new-admission).
  The gate found three real bugs, one of which meant the hash covered a field that was never populated.

#### Requantization is a new admission

The gate line has two halves that pull opposite ways: re-admitting at a different quantization **must**
produce a new `arch_hash` and invalidate KV checkpoints; re-**profiling** must not. Either alone is
easy. A hash over the whole container makes the first true and the second false — every reprofile would
orphan every checkpoint. A hash over the architecture only makes the second true and the first false —
two quantizations share one identity, and a checkpoint written under one replays under the other,
fluently and wrongly.

Writing the test found **three** bugs, in increasing order of how quiet they were.

**1. The hash covered four roles and only their dtype.** `q4_g` at group 128 and `q4_g` at group 64
hashed identically — they dequantize to different weights, so a KV checkpoint written under one and
replayed under the other resumes a conversation the cache does not describe. `expert_up`,
`shared_expert`, `norms` and `draft_head` were not covered at all. It is now the whole `QuantMap`,
every role, dtype **and** group, iterated over a named list so a role added to the struct and forgotten
here is a compile-time omission rather than a silent one.

**2. Two quantizations of one model wrote to the same container directory.** `containers/<name>` for
both, so the second conversion overwrote the first — and the first's registry row then described bytes
that were no longer its quantization, with its verdict, its `expert_bytes` and its KV format all
recorded against a container that had been replaced underneath them. The directory now carries the
quantization.

**3. The quant map was in the hash and the value never arrived.** This is the one the first two were
hiding. `compute_plan` builds the IR from `config.json`, which carries no quantization at all — so for
every converted container, `arch.quantization` was whatever `adapt_hf_config` defaulted to. The field
was hashed and the value was never populated, which means **every container of a given architecture
hashed identically no matter what it was converted at**. Fixing (1) alone would have changed nothing.
`apply_container_quant` now overlays `container_meta.json` — not a second description of the
architecture, but the record of a conversion, and the only place the quantization exists.

A consequence worth stating: a container whose `container_meta.json` is unreadable now **fails the
plan** rather than one stage of the conformance ladder. That is correct — the quantization is
unknowable, so computing an identity from defaults would assign this container the identity of a
differently-quantized one, which is precisely the collision above.

**Quantization is per REQUEST**, not per deployment. `POST /v1/models/admit` accepts
`quantization: {expert_gate, expert_down, group}`, and the override is applied to the *copy* of
`AdmissionTools` the operation runs with — two admissions of the same model can be in flight at once,
and a shared field would let the second rewrite the first's conversion arguments mid-run. The premise
the registry keys on cannot be exercised at all if changing the quantization means editing a config
file and restarting.

**And `resolve()` had to learn to disambiguate.** Two rows now match one name, so "the first row the
scan reaches" is not an answer — it is whichever the b-tree yielded, and it would change under an
unrelated insert. Ranked instead: a verdict that selects Soma first, then more recently profiled. An
operator who wants a *specific* variant passes its `arch_hash`, which is the only identity that cannot
be ambiguous.

The test checks the hash property on the IR directly — a pipeline test that happened to pass would not
say which field made it pass — and then drives the real pipeline twice, asserts two rows with different
hashes and separate containers, writes a KV checkpoint under one and requires `ArchMismatch` loading it
under the other, and finally reprofiles and requires the hash, the directory and the row count all
unchanged. The checkpoint halves use the *same cache geometry* on both sides, differing only in
`arch_hash`: nothing about the bytes would stop the load, so the hash is demonstrably the only thing
that does.

---

## G9 — RETIRED: the gate that was already passed

**This gate was specified on a false premise and is retired rather than deleted.** It asked for expert
reads to be overlapped with compute. They already were, and had been for some time.

The premise came from G3's outstanding list — "expert reads are currently issued serially inside the
union loop, so the ~44% is *waiting*" — which was written before prefetching was built and never revised
after. I read that line, took a throughput measurement that seemed consistent with it, and wrote a gate
around it **without checking whether the mechanism existed**. It did: `prefetch_ahead()` on the
`MemoryHierarchy`, queued from the union loop, depth derived from the per-layer cache cap, loader threads
doing the reads, on by default with `SOMA_PREFETCH_DEPTH` only as an override. The determinism hazard it
introduces — concurrent reads on a shared `ifstream` returning another expert's bytes — had already been
found and fixed with positional reads, and `streamed_determinism_g3` already guards it.

This is the same failure this document has been catching all along, committed here rather than found:
a description that drifted from the code, believed because it was written down.

### What the measurement actually says

`io_wait_ns` had been DECLARED in `CacheStats` since the beginning and never populated, so every claim
about waiting was an inference from throughput. It is populated now, split by cause, and the answer is
unambiguous:

```
prefetch ON (default)
  nseq=1  io_wait 0.23s = 39.5% of wall  [miss  7.9%, depth 92.1%]  427 hit / 0 wasted
  nseq=8  io_wait 0.12s =  6.9% of wall  [miss 27.2%, depth 72.8%]  737 hit / 0 wasted

prefetch OFF (SOMA_PREFETCH_DEPTH=0)
  nseq=1  io_wait 0.50s = 64.9% of wall  [miss 100%, depth 0%]        0 hit / 0 wasted
  nseq=8  io_wait 0.93s = 40.7% of wall  [miss 100%, depth 0%]        0 hit / 0 wasted
```

**The "~44% of wall time is reads" figure describes the prefetch-OFF engine.** With prefetch on it is
6.9% at the widest batch. There is no 45% of waiting to reclaim, and a gate to reclaim it cannot pass
because it is already passed. Throughput at nseq=8 confirms it from the other side: 28.1 tok/s with
prefetch against 18.3 without, a 1.54x standing win.

The split is what makes this actionable rather than merely reassuring. At nseq=8, 72.8% of the remaining
wait is *depth* — finishing a read prefetch had already started — and 27.2% is *miss*, where no prefetch
was attempted. Only the first responds to queueing further ahead, and it is 5% of wall time. That is the
size of the prize, and it is small.

### Why the metric I chose would not have shown it either

G9 was gated on "union conversion rises from 55% toward 80%". Conversion measures realised throughput
scaling against available byte savings — a SCALING ratio. Prefetching speeds up every batch width by
roughly the same factor, so it barely moves: **57% with prefetch and 57% without**. A gate on that number
would have been insensitive to the very mechanism it named.

It is also noisier than the movement it was written to detect. Six runs of the same binary on the same
container: **45, 50, 50, 52, 53, 57%**. The 55% the gate anchored to was near the top of that band, not a
baseline.

### What a successor gate would need first

The conversion gap is real — 2.63x realised of 4.65x available — but it is not I/O wait, which is 6.9%.
It is compute or memory, and this document should not name a mechanism again without evidence.

The blocker is methodological, and `scaling_g3`'s own header already says it: wall-clock here measures
**the page cache, not the device**. The runs above read at 1632 MB/s against a G2-measured NVMe figure of
1230 MB/s — 133%, which is only possible if the OS is serving from RAM. Every number on this page is
warm. Before any performance gate is written:

- A cold-cache measurement, which on this host needs a way to drop the standby list. Until then
  `bytes/token` stays the primary gate, exactly as `scaling_g3` was designed to insist.
- An attribution of the non-wait 93%, which `io_wait_ns` now makes possible for the I/O half and which
  nothing yet does for compute.

`accuracy_floor` remains unclaimed and is unaffected by any of this; it wants its own small gate, for the
reasons recorded when G9 was first drafted.

## GLM-5.2: describable, not servable

The concept this engine implements was originally proven by **Colibri**, on **GLM-5.2** — 744B
parameters, ~372 GB of weights at q4, served from disk with 16 GB of RAM minimum and 24 GB comfortable.
That is a 16x oversubscription, and it is the regime this repository has never once entered.

Worth stating plainly, because the test suite reads as if it had: **no real container has ever earned a
`stream` verdict.** Every committed fixture is shrunk, so all five plan `resident-only`. The single real
admission, OLMoE-1B-7B, earned `hybrid`. Streaming has only ever been reached through synthetic `ArchIr`
in `routing_g5` and `container_g2` — hand-built topologies against a deliberately tight host.

### The adapter needed almost nothing

`adapt_hf_config` already read every key GLM-5.2 has: `first_k_dense_replace`, sigmoid scoring,
`noaux_tc` bias correction, `norm_topk_prob`, `routed_scaling_factor`, `n_shared_experts`, and the whole
MLA block — which already branched on `MlaDsa` because the enum anticipated DSA. The adapter is one
`traits_for` entry. The economics fall out:

```
attention_family  mla+dsa        n_layers 78 (3 dense + 75 MoE)
n_experts         256            top_k 8      shared 1
active_fraction   0.03125        <- 8/256
arch_supported    false
verdict           reject   "no backend for mla+dsa attention in this build"
```

**3.1% active fraction is the lowest of any model this project has measured** — against Qwen3's 6.3%,
DeepSeek-V2-Lite's 9.4%, and Mixtral's 25%, with the streamability ceiling at 15%. By the verdict
function's own criterion GLM-5.2 is the most streamable architecture yet seen, which is what one would
hope given it is the model the concept was proven on. That number is also quantization- and
host-independent, so it is the one part of this answer that needs no further work to trust.

### `arch_supported` finally has a producer

The field had a reader in admission and **nothing that ever set it**. It defaulted to true because an
unadaptable `model_type` failed earlier in `adapt_hf_config` and never reached a plan, so "describable"
and "servable" were the same question and neither needed the distinction.

`glm_moe_dsa` separates them, and the answer is now DERIVED rather than declared:
`arch_supported = (resolve_f32_backend(arch) != nullptr)`. Asking the registry means it cannot drift from
what the engine really resolves at load; a second table would be one more thing to keep in step. The
registry already returned `nullptr` for `MlaDsa` with the reason written down — serving it through the
plain MLA backend would run it as DENSE attention, "finite, plausible, and not the model that was asked
for."

The verdict is forced to `reject`, per the field's own contract, because a verdict is a ROUTING decision
and routing an agent to an engine that cannot execute the model is worse than refusing. `verdict_reason`
distinguishes this from an economic reject, and the distinction is not cosmetic: economics change on a
bigger host, a missing backend does not change on any host. Admission reads `arch_supported` and **skips
conversion entirely** — which is what stops 1.4 TB being converted into a container nothing can read.

### What it caught on the way in

`routing_g5`'s synthetic Mixtral and Qwen3 never set `attention.family`, so both defaulted to `Unknown`
and — once the plan started asking whether a backend exists — planned as `reject`. The fixtures were
describing no real model, and it had not mattered until something read the field. Both now declare GQA,
which is what they are.

### Asking the other two arguments

`soma plan` took `--model-dir` and `--json`. The verdict is a property of **(model, quantization, host)**
— a phrase this document repeats a dozen times and which drives `arch_hash`, staleness detection and
re-admission — and the CLI could vary exactly one of the three. It evaluated that function at one fixed
point: f32 on a hardcoded 16 GiB / 8 GiB, with this box's NVMe figure.

Now: `--quant`, `--expert-down`, `--group`, `--ram`, `--ram-free`, `--disk-bw`, `--ctx`.

Both groups are HYPOTHETICAL and convert nothing. `--quant` mirrors the three fields
`ControlModelRegistry::QuantOverride` already accepts on `POST /v1/models/admit`, so the same values that
DECIDE a conversion can now ASK about one first — which is what a headers-only planner is for. The
overlay is built in `container_meta.json`'s shape and handed to `apply_container_quant`, the same
function the container path uses, rather than mapping dtype names to roles a second time. That mapping
carries a rule — gate and up must share a dtype, because the converter interleaves them into one range —
and a second copy would let `plan` describe a container the converter cannot produce. Asserted directly.

`--ram` alone sets total AND free, because an explicit budget is a statement about what the engine may
have; silently reserving half of a number the operator typed would answer a different question and look
identical. The 16/8 default still models a real machine with an OS on it. Sizes parse `24GiB`/`24G`
(1024-based) and `24GB` (1000-based) distinctly, and an unparseable size is an error rather than a
fallback — a typo'd budget that quietly plans against the default produces an answer indistinguishable
from a real one.

**The claim is now executable.** One model, three verdicts, varying only the arguments:

| OLMoE-1B-7B @ q4_g | verdict |
|---|---|
| `--ram 2GiB` | `stream` — routed set exceeds the cache |
| default 16/8 GiB | `hybrid` |
| `--ram 64GiB` | `resident-only` — streaming buys nothing |

and `--quant q8_0` moves the routed set 3456 -> 6336 MiB at a fixed host. `container_g2` pins both axes.

**`economic_verdict` came out of using them.** With `arch_supported` forcing Reject, GLM-5.2's headline
verdict said nothing about whether it would stream — which is the only reason to plan an unsupported
architecture at all. The plan now reports what the economics alone say, beside what the engine can
actually do. They differ for exactly one model today.

### What the flags found

Asking the Colibri question — q4_g on 24 GiB — produced this:

```
routed        379.7 GiB      (~2% off Colibri's ~372 GB, so the same regime)
bytes/token    11.9 GiB
dense_resident 135.4 GiB     <- and this is what rejects it
expert_cache     0.0 GiB
economic_verdict reject       ("hybrid" only once the host reaches ~512 GiB)
```

The experts are not the problem. **The dense half is 135 GiB at q4**, so no host under ~150 GB can serve
this model whatever the expert streaming does — and Colibri ran it on 16-24 GB. Two separate causes, both
now logged as defects:

- **D16** — the plan's dense sizing, now **fixed**; see below.
- **D17** — dense tensors are F32 in the container by construction, which puts a hard floor under any
  large model regardless of expert quantization.
- **D18** — MLA containers are missing their attention weights entirely, found while checking D16.

None were visible before, because nothing had ever planned a model big enough for the dense half to
matter.

### Fixing the dense estimate found two more bugs

D16 was logged as one error and was three.

**1. MLA sized as GQA.** The planner charged `q + 2*(n_kv_heads x head_dim) + o` for every family. MLA
has no per-head K or V projection — it compresses through `kv_lora_rank` and reconstructs — so two of the
largest tensors in the layer were invented.

**2. Shared experts counted `n_shared^2`.** `shared_intermediate` already carries the count (the adapter
derives `moe_intermediate x n_shared` when config.json omits it, and the tensors really are fused:
DeepSeek-V2-Lite's `shared_experts.gate_proj` is `[64,64]` at moe_intermediate 32 and n_shared 2 — ONE
set, not two). The planner then multiplied by `n_shared_experts` again. Invisible at 1 shared expert
(GLM-5.2, Qwen), 2x over at 2 (DeepSeek, Moonlight) — which is why it hid behind the MLA error for so
long, since the same two models carry both.

**3. `arch::mla::attention_backend()` was declared and never defined.** Found only because fixing (1)
required somewhere to put the formula. `resolve_attention_backend` therefore returned nullptr for MLA,
and `kv_bytes_per_token` — the thing that stops the KV cache and the expert cache fighting over the same
RAM — silently returned **zero for every MLA model**. Now real: DeepSeek-V2-Lite reports 10.5 MB at ctx
4096 x 4 slots against 0 before.

**Corrected measurement, and a correction to what was claimed.** The original "MLA 1.66x" was taken
against the DeepSeek CONTAINER, which turns out to be missing tensors (D18) — so it conflated a formula
error with a conversion bug. Re-measured against the source fixtures, which carry the real tensors, and
now checked in `container_g2` on every commit:

| fixture | family | ratio |
|---|---|---|
| DeepSeek-V2-Lite | mla | 1.00x |
| Moonlight-16B-A3B | mla | 1.00x |
| Qwen3-30B-A3B | gqa | 1.00x |
| Mixtral-8x7B-v0.1 | gqa | 1.00x |
| OLMoE-1B-7B-0924 | mha | 1.00x |

Checked against real BYTES rather than a second formula, because a formula compared to a formula agrees
with itself. Reverting either fix independently turns the two MLA rows red and leaves GQA/MHA at 1.00x.

**The seam check refused the first attempt, correctly.** Branching on `AttentionFamily` inside plan.cpp
tripped R2 — `kv_lora_rank` is architecture knowledge, and core is not allowed to hold it. The error
message named the remedy: express it through the backend pointers. So the formula moved to
`AttentionBackend::weight_bytes_per_layer`, beside `kv_bytes_per_token`, which is the same shape of
question and was already there.

**And the second attempt was wrong in a way only the verdict table caught.** The backend initially
returned bytes and hardcoded `sizeof(float)`. That agrees with the old code on the f32 fixtures — every
dense-sizing row still read 1.00x — and disagrees by ~8x on any quantized plan. `check_verdicts` flipped
Mixtral from `resident-only` to `reject` and said so. The fix is a division of labour worth stating: the
BACKEND owns the shapes, the PLANNER owns the quantization, and the sizer is passed in rather than
assumed.

### The container that could not be served

Found while checking D16's evidence: the ratio had been measured against the DeepSeek CONTAINER, and the
container turned out to be missing tensors. `soma serve` on it dies with `binding attention weights
failed at layer 0`.

`convert.py`'s `DENSE_SUFFIXES` is an allow-list, and it only ever covered what the first three models
happened to carry. Enumerating every source tensor against it found **three** families silently dropped,
which fail in three different ways:

| Dropped | Affects | How it fails |
|---|---|---|
| `kv_a_proj_with_mqa`, `kv_b_proj`, `kv_a_layernorm` | every MLA model | fails to load — loud |
| `mlp.gate_proj/up_proj/down_proj` | any model with `first_k_dense_replace > 0` | fails to load — loud |
| `mlp.gate.e_score_correction_bias` | `noaux_tc` routing: DeepSeek-V3, Moonlight, **GLM-5.2** | **loads and routes to the wrong experts** — silent |

The third is the one worth pausing on. It is the per-expert selection bias, and without it the router
still produces a top-k — just the wrong one. That is the failure mode this whole document keeps trying to
design out, and it was one line from shipping on the model the project was built for.

**The rule was already written down.** The comment above `DENSE_SUFFIXES` says a dropped tensor "is an
error rather than a silent omission — a dropped tensor produces a model that loads and is wrong." It was
never enforced. Now it is: every source tensor must match the allow-list, an explicit `IGNORED_PATTERNS`
entry (routed experts, shared experts, `rotary_emb.inv_freq`, MTP heads), or the conversion REFUSES and
names the unhandled kinds. A refusal rather than a warning, because a conversion costs hours and hundreds
of gigabytes and a warning in that much output is not read.

**Rebuilt and verified.** All five container fixtures reconverted; only the two MLA ones changed, which
is itself the check that the fix added what was missing and disturbed nothing else — Moonlight's dense
half grew 35%, the three GQA containers are byte-identical. All five now start under `soma serve`.

**And the test gap that allowed it.** `container_g2` opened the expert store and compared payload bytes;
it never loaded the container AS A MODEL, so the dense half was unexamined. It does now, and reverting
`convert.py` reproduces the original failure through the test rather than through a server that happens
to be started by hand.

That gap is the reason to qualify G4: the ladder passes for MLA on the fp32 SOURCE path, which is what
`conformance_g0` reads. The CONTAINER path — the one production serves — had never worked for that
family, and every G4 claim on this page was true about the half that was tested.

### The dense half was never a format problem

D17 was logged as "dense tensors are F32-only in a container", implying a container format change. It was
not. `bind_weight` has always read the role's `QuantSpec` and quantized at load — but `apply_container_quant`
only ever set `expert_gate`, `expert_up` and `expert_down`, so embeddings, attention projections and shared
experts stayed F32 **by omission rather than by decision**. The capability existed and nothing could ask
for it.

`dtype_dense` in the overlay, `--quant-dense` on the CLI. One key for the three roles, because they are the
"resident, not routed" family and the reason to quantize any of them is the same. The router is
deliberately excluded — `TensorRole::Router` is documented as MUST be F32, one matrix per layer is
negligible beside the embeddings, and a quantized router changes which experts fire.

**GLM-5.2, q4_g experts, 24 GiB host:**

| dense precision | resident half | expert cache |
|---|---|---|
| f32 (the old behaviour) | 68.6 GiB | 0.0 GiB |
| q8_0 | 18.0 GiB | 3.3 GiB |
| q6_g | 13.7 GiB | 7.5 GiB |
| q4_g | **10.0 GiB** | **11.2 GiB** |

The model fits, with room to stream. That is the 16-24 GB regime Colibri worked in, reached for the first
time by this engine.

**Disk is untouched, and that is the better property.** `dense.safetensors` keeps full precision and the
loader quantizes into RAM, so resident precision can be changed without reconverting a byte — exactly what
the expert half cannot do, since its quantization is baked into the payload. Asserted in `container_g2`
so nobody "helpfully" quantizes the file and takes it away.

**What it did NOT fix, and this is the interesting part.** GLM-5.2 is still `reject`. Not on memory any
more — on throughput: 0.10 tok/s at 24 GiB, and 0.79 tok/s even at 128 GiB with a 7 GB/s disk, against a
`kMinProjectedTokS` of 1.0. So the engine's own floor refuses the model the concept was proven on. Logged
as D21 rather than adjusted, because it is a judgement about what "usefully served" means and that is not
a decision to slip into a defect fix.

### `indexer_types` in the IR, and a hash that moved

Adding `AttentionSpec::dsa` — index_topk, the index head geometry, and a per-layer
`vector<IndexerKind>` of Full/Shared — was the small enabling step both remaining consumers needed. It is
read from `indexer_types` rather than derived from `index_topk_freq`, for the same reason
`Topology::layer_kinds` resolves three upstream spellings once: the stride happens to give the right
answer for GLM-5.2, and the WEIGHTS are what decide. A layer the IR called `Full` without the tensors to
back it would fail at bind time, or worse, borrow nothing. Two malformed configs are refused outright —
a length that disagrees with the layer count, and an all-`Shared` stack with nothing to share from.

That closed D22 immediately: the planner can now amortise the indexer across the stack (`idx * n_full /
n_layers`, which gives the correct total from a per-layer function that has no layer index to consult),
and GLM-5.2's resident half went 0.90x -> **1.00x**. Shapes taken from the committed fixture rather than
inferred — including that `wk` is ONE shared K across index heads, MQA-style, not one per head.

**And then a claim I wrote turned out to be false, which is the part worth recording.** The hash comment
said DSA was emitted conditionally so that "the five families already admitted hash exactly as before;
verified rather than assumed." Checking it — comparing the admitted OLMoE's stored `arch_hash` against a
fresh computation — showed the hash HAD changed, and not because of DSA.

The cause was D17, several increments earlier. `apply_container_quant`'s setter applied `group`
unconditionally, which was harmless while only the three expert roles were settable: a real
`container_meta` always carries `group` beside `dtype_gate_up`, so they arrived together. The moment
`dtype_dense` made embed/attn_proj/shared_expert settable, that same line began stamping `group` onto them
with no dense dtype asked for. `arch_hash` covers dtype AND group for every role, so every admitted
container's hash moved — the registry would have read its own records as `StaleRecord` and routed
everything to the fallback, and KV checkpoints keyed on the hash would have stopped loading. Logged as
D23 and fixed: an unnamed role is left entirely alone.

Nothing would have caught it. No test asserted hash stability across a quant-map change, and the symptom
in production is not a crash but a cluster that quietly stops using Soma. `container_g2` now asserts that
an expert-only overlay leaves the dense roles untouched while still applying to the roles it named.

Worth stating why it surfaced at all: not from a test, but from writing "verified rather than assumed" in
a comment and then going to verify it. The claim was the only reason to look.

### The deployment test

Thirteen commits had landed since the last time the full stack ran, four of them on paths unit tests
cannot reach: the production argv (D14), the node's link structure (D13), an added call on every
image-bearing request (D12), and — by the end — the served output itself (D19). This ran node + control
+ both engines, driven over HTTP.

**Result: passed, and it verified the two things that most needed it.**

`llama-server`'s actual command line, read from the live process:

```
--ctx-size 384  --gpu-layers 0  --threads 3  --batch-size 96
--ubatch-size 48  --flash-attn on  --slot-save-path data/kv_cache
```

Every setting D14 restored, on a real process, from an agent's configured `runtime_settings`. Before that
fix the line was `-m … --port … --host …` and nothing else. `--gpu-layers 0` is the one worth pointing at:
it is the setting whose absence defaulted to *all layers on GPU*, and it is now there because it was
asked for.

Both engines ran **concurrently on one node** — `soma.exe` at 5.1 GB serving the admitted OLMoE,
`llama-server.exe` at 396 MB serving the committed GGUF fixture — launched by the real node binary rather
than by `engine_g5`'s harness. Two concurrent chats both returned `success`, the Soma agent producing
3744 characters of real text, and both engines survived the exchange. The model transfer also worked
unprompted: control copied the GGUF into the node's cache at
`models/tiny-llama-f16.gguf-bdc6021a8bf022ed68c3cf43/`.

**The friction is worth recording, because it is an operator-facing finding rather than a bug.** Standing
a node up from the documented config alone does not work. `control_url` in `mantic-mind.toml` gets the
node running and self-registering, and control refuses it: a new node may only self-register if control
already knows its `api_key`, or if the request carries control's external bearer token — and that token
is empty by default. The node then generates an ephemeral key and does not persist it, so there is
nothing to hand over. What worked was setting `MM_API_KEY` to a known value and calling
`POST /v1/nodes` with it.

That is defensible security — an unpaired node cannot join by asking — but the config file gives no hint
of it, the node logs nothing about the rejection in CLI mode, and control's rejection appears only in its
own log. The symptom an operator sees is a node that runs, answers its own health endpoint, and shows as
`offline` forever.

### The shared expert that was never bound

Pointing the ladder at an MLA container for the first time (possible only after D18) showed all five
containers passing the stage-3 threshold — but not alike. The two MLA fixtures scored mean KL 0.0092 and
0.0111 with top-1 agreement of 35% and 42%, against 0.00000 / 99.6% for the GQA ones.

**Three hypotheses died before the right one.** Quantization: an f32-expert container gave DeepSeek
0.00929 — unchanged — and `conformance_g1` showed source-path sensitivity is uniform across all five
families (max\|dlog\| 0.30-0.43). The converter: container dense tensors are byte-identical to the source,
no source tensor is missing, and the expert round-trip is 48/48. The dense-layer FFN's role: real bug,
fixed below, but not this one.

**What found it was a new measurement**, not more reasoning: load the same model from the source and from
the container, same quantization, same expert bytes, and compare logits. Bit-identical for the three GQA
fixtures; **5.7e-01 max\|logit\|** for the two MLA ones. Then setting `n_shared_experts = 0` on both sides
dropped it to exactly zero, which named the culprit.

```cpp
if (out.experts_are_streamed) continue;   // skips the REST of the MoE branch
```

A container holds no routed experts, so that `continue` is right about them and wrong about everything
after it — and the SHARED expert binding is fifteen lines below. So for every container-served model,
`lw.shared_gate` stayed empty, and the forward's `if (!lw.shared_gate.empty())` skipped the shared expert
without a word. Bound optionally, left empty, silently ignored.

The comment immediately above that line warns that skipping silently "would leave the resident table
empty in a way only visible as wrong output." It was right, one scope further down than it was looking.

**Affects every family whose shared expert fires on every token** — DeepSeek, Moonlight, Qwen2-MoE and
GLM-5.2 — and only when served from a CONTAINER, which is to say only in production.

| | before | after |
|---|---|---|
| DeepSeek-V2-Lite | KL 0.00916, top-1 35.0% | **KL 0.00000, top-1 97.5%** |
| Moonlight-16B-A3B | KL 0.01109, top-1 42.4% | **KL 0.00022, top-1 92.0%** |
| Qwen3-30B-A3B (control) | KL 0.00000, top-1 99.6% | unchanged |

**The coverage gap that allowed it, now closed.** `streaming_g2` compares a streamed forward against a
resident one and finds them bit-identical — while loading the model from `tiny/` in BOTH cases and
attaching only the container's experts. It compares the source path to itself. `container_g2` now loads
the container as a model and compares its output against the source's, which is the check that would have
caught this the day the container format was written.

**One more mis-assignment, found on the way and fixed.** A `first_k_dense_replace` layer's FFN was bound
AND sized with `TensorRole::ExpertGate/Up/Down`, so a quant map applied to it — while `convert.py` writes
it to `dense.safetensors` at F32 and never quantizes it. Loader and planner both now use `SharedExpert`,
the role for "always-active FFN, resident, F32", which is how the converter already treats it. Not the
cause of the divergence above, and a real defect either way.

### GLM-5.2, re-measured

```
dense_resident   66.4 GiB   (was 135.4 GiB before the fix - 2.0x over)
routed          379.7 GiB   @ q4_g
bytes/token      11.9 GiB
```

Still `reject` on economics at any host under ~128 GiB, and still 0.1 tok/s where it does fit, so the
conclusion has not changed — but the number that drives it is now the right one, and half the apparent
problem was arithmetic. The remaining 66 GiB is D17: dense tensors are F32 in a container, and 78 layers
of MLA plus a 155k-token embedding table at fp32 is simply that big. Colibri served this model in 16-24
GB, so it quantized the dense half; Soma cannot yet express that.

## Scoping: DSA and IndexShare

Not a gate yet. This is what serving GLM-5.2 would require, with the facts checked against the real
1.4 TB checkpoint rather than inferred from its config.

### Step zero — the oracle — is DONE

The admission venv runs `transformers` 4.57.6, which does not know `glm_moe_dsa` at all (the config is
absent from `CONFIG_MAPPING_NAMES`). The checkpoint declares 5.12.0. That was the same position D4 is
parked in, with one decisive difference: D4 has no version of anything that reads the fused expert layout,
while this needed only a newer dependency.

**A second venv, not an upgrade.** `tools/admission/.venv-oracle`, pinned via
`requirements-oracle.txt` to `transformers==5.12.1` — the version GLM-5.2's own config names, i.e. the
implementation its weights were exported against. The admission pipeline stays on 4.57 because it WORKS;
4.57 -> 5.x is a major bump under `convert.py`, `make_oracle.py`, `make_reference.py` and the tokenizer
compile, and trading four working tools for an unproven upgrade buys nothing here. CPU-only torch, since
an oracle is a correctness reference and not a throughput one.

**Verified, in this order:**

1. `glm_moe_dsa` is in `CONFIG_MAPPING_NAMES`.
2. GLM-5.2's real config loads as `GlmMoeDsaConfig` — 78 layers, 256 experts, `index_topk` 2048,
   `indexer_types` 21 `full` / 57 `shared`.
3. A shrunk instance (8 layers, d=128, 8 experts, **1.78M params**) builds and forwards finite logits.
4. Its `full` layers carry `indexer.{wk, wq_b, weights_proj, k_norm.weight, k_norm.bias}` and its `shared`
   layers carry **none** — the cross-layer dependency reproduces in the tiny model, which is what makes it
   usable as a fixture rather than merely loadable.

**And the sparsity risk from the section below is now measured, not predicted.** With `index_topk = 8`
against a 40-token prompt, comparing that model to the same weights at `index_topk = 512`:

| positions | max abs logit diff |
|---|---|
| 0-7 (at or below `index_topk`) | **0.0** — bit-identical to dense |
| 20-39 (above it) | **0.484**, greedy agreement 35% |

So below `index_topk` the sparse and dense paths cannot be told apart AT ALL, and an oracle fixture must
shrink `index_topk` as well as the model or it will pass a broken indexer. That was a hypothesis when this
section was written; it is now a number.

Two smaller findings for whoever extends `make_oracle.py`: token ids must be shrunk alongside the vocab
(`pad_token_id` 154820 against a 256-token vocab asserts inside `nn.Embedding`), and `indexer_types` /
`mlp_layer_types` must be TRUNCATED to the new layer count while keeping their pattern — a shrink that
preserves only their length would silently change which layers own an indexer.

### What the checkpoint actually contains

Verified from `model.safetensors.index.json`, not from the config:

| layer | attention tensors |
|---|---|
| 0 (`full`) | `q_a_proj`, `q_b_proj`, `q_a_layernorm`, `kv_a_proj_with_mqa`, `kv_b_proj`, `kv_a_layernorm`, `o_proj` **+ `indexer.wk`, `indexer.wq_b`, `indexer.weights_proj`, `indexer.k_norm.{weight,bias}`** |
| 3 (`shared`) | the same MLA set, and **no indexer tensors at all** |

That is the cross-layer dependency as a fact about the weights: 57 of 78 layers have nothing to compute an
index with. `indexer_types` is 21 `full` and 57 `shared` — one `full` every four layers after the first
three — with 32 index heads of 128 dims and `index_topk = 2048`.

Note `indexer.k_norm.bias`: a rank-1 bias, and the only bias anywhere in this attention block.
`bind_layer_weight` requires rank 2 and rejects anything else, so biases need the rank-1 path that
`bind_layer_f32` already provides for norms.

### The interface question is smaller than previously claimed

Earlier notes on this page said `AttentionBackend` "has no channel for cross-layer state, so it is an
interface question before an implementation one." Half right, and the conclusion was too strong.

The index is per-(row, step): each new token has a new query, so a new top-k selection. It is therefore
recomputed every forward pass and **never has to persist between steps**. Its natural home is
`ExecScratch`, which is per-step, sized for `max_batch`, and already passed to every `prefill`/`decode`
call across every layer. A `full` layer writes it; the next three `shared` layers read it.

`ExecScratch` has no backend-private slot today — but the project already has the exact idiom for one.
`ArchLayerPayload` is an opaque `void*` plus deleter with `adopt()` / `as<T>()`, which is how
`f32_bind_layer` stores `F32AttnWeights` where the core cannot inspect it. One member of that shape on
`ExecScratch` is a small, precedented addition rather than a redesign.

**What this consequently does NOT touch**, each of which would have been expensive:

- **The KV checkpoint format.** The index is not state, so no `persist_format_id` bump, no version
  migration, no effect on cross-process resume.
- **`kv_bytes_per_token`.** DSA selects among cached tokens; it does not shrink the cache. MLA's compressed
  latent is still what gets stored, so the planner's KV arithmetic is already right.
- **`SeqState`.** Nothing per-sequence is added.

### The part worth being blunt about

**DSA will not make this engine faster.** Its published win is ~2.9x fewer attention FLOPs at full
context. Soma at q4_g reads **11.9 GiB of expert weights per token**, and `io_wait` measurements put the
engine at the disk. Attention FLOPs are not the constraint and cutting them will not move
`projected_tok_s` meaningfully.

So implementing DSA is a CORRECTNESS requirement — it is how this model computes attention, and running it
as dense MLA would be a different model that happens to produce plausible text. It is not a throughput
project, and it will not resolve D21: GLM-5.2 projects 0.79 tok/s at 128 GiB with a 7 GB/s disk against a
1.0 floor, and none of that is attention.

### Work breakdown

1. ~~**The oracle.**~~ **DONE, and the fixture is committed.** `tests/fixtures/tiny/GLM-5.2`, 5.3 MB,
   built from the real 1.4 TB checkpoint by `make_oracle.py` under the oracle venv: 78 -> 8 layers, 256 ->
   16 experts, vocab 154880 -> 512, 849,856 params, 512 teacher-forced positions and 256 greedy tokens.
   `conformance_g0` already lists it and reports `SKIP: no fp32 backend for attention family mla+dsa` —
   waiting rather than failing, so it starts checking the day step 5 lands.

   Its `indexer_types` is `full,full,full,shared,shared,shared,full,shared`: a full->shared borrow AND a
   second group boundary at layer 6, which is where the `index_topk_freq` off-by-one named below would
   show. `index_topk` is 64 against 512 evaluated positions, so 87.5% of the sequence exercises real
   selection.

   Four shrink bugs surfaced doing it, all versions of the generated fixture disagreeing with the model
   that produced its oracle:
   * `dtype` must be normalized to `float32` alongside `torch_dtype`. The generator initializes, runs and
     saves F32 weights, but Transformers 5.x treats `dtype` as authoritative on reload; leaving the source
     checkpoint's BF16 declaration silently cast the saved file and moved logits by 0.51. A clean reload
     now reproduces the committed oracle at max absolute difference 0.0.
   * `qk_head_dim` is stated explicitly by GLM (most MLA families leave it implied) and must be recomputed
     as nope ++ rope, or the forward dies in `torch.split` with "expects split_sizes to sum exactly to 256
     ... but got [16, 8]".
   * `indexer_types` / `mlp_layer_types` must be TRUNCATED, never regenerated, and the shrink now refuses
     outright if the truncation leaves no `shared` layer — a fixture that cannot exercise IndexShare
     should not be written at all.
   * `layer_kinds()` now believes `mlp_layer_types` when a family states it, rather than re-deriving from
     `first_k_dense_replace` + `moe_layer_freq`. Those agree for GLM-5.2, and that coincidence is the
     argument: an irregular pattern would be silently mis-derived with nothing to catch it.

   One honest weakness: the fixture's greedy sequence is `[378] * 8` — a tiny-random model repeating one
   token. Greedy token-exactness is therefore a WEAK check here and the teacher-forced logit comparison is
   the real one. Worth knowing before anyone reads a green greedy row as meaningful.
2. ~~**Conversion.**~~ **DONE, and it turned out to depend on D4.** Attempting it produced the first real
   test of D18's enforcement against a genuinely new family, and the answer was not the indexer: the
   conversion refused on the FUSED expert layout, because anything `make_oracle.py` writes for this family
   carries it. So step 2 could not be finished without D4, and D4 had just become finishable.

   The gate/up split was settled by measurement rather than reading. `modeling_glm_moe_dsa.py` applies the
   parameter as `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` — contiguous halves of the output, hence
   contiguous ROWS, gate first. Confirmed against the real fixture: reconstructing from rows `[0:inter]`
   and `[inter:]` reproduces that `chunk` at **0.00e+00**, while the interleaved reading is off by
   **1.35**. gpt_oss stores the same concept interleaved, which is exactly why guessing was refused.

   `convert.py` now reads both layouts, resolving which once per layer rather than per expert.
   `DENSE_SUFFIXES` gained the five indexer tensors — `wk`, `wq_b`, `weights_proj`, `k_norm.weight` and
   `k_norm.bias`. GLM-5.2's fixture converts: 8 layers, 5 MoE, 129 dense tensors, `expert layout: fused`.

   **And the correctness check is independent of the engine, which it had to be.** GLM-5.2 cannot be
   loaded — no backend for `mla+dsa` — so `container_g2`'s round-trip skips it and would never have
   verified the new reader. Instead `tools/ci/check_fused_experts.py` (ctest `mm_fused_experts`) rewrites a
   committed PER-EXPERT fixture into the fused layout, converts both, and requires the expert payloads to
   be BYTE-IDENTICAL. The per-expert container is already checked against the engine's own quantizer, so
   byte-equality inherits that transitively. Reversing the split makes it fail, with a message naming the
   contiguous-vs-interleaved hazard.

   It also refuses to pass vacuously: it asserts each run REPORTED the layout it was supposed to take, so
   a fixture that quietly stopped being fused cannot satisfy the check by converting the same way twice.

   No rank-1 binding path was needed after all. `k_norm.bias` is rank 1 and `bind_weight` requires rank 2,
   but `bind_layer_f32` already handles rank-1 tensors for norms and is what the backend will use at step 4.
3. ~~**`ExecScratch` payload.**~~ **DONE** — and it landed on `F32Workspace`, not `ExecScratch`. The fp32
   reference path is what conformance exercises and what steps 6-7 gate on, and it threads `F32Workspace`
   through every layer, not `ExecScratch`. One `ArchLayerPayload arch_state` member, the same opaque
   `adopt`/`as<T>` idiom `F32LayerWeights::attn` already uses, plus a `reset_arch_state()` the forward calls
   before layer 0 — a selection left over from a previous prompt has the wrong length and the wrong
   contents, and reusing it would be plausible rather than loud.
4. ~~**The indexer.**~~ **DONE.** Two details were traps rather than choices, and both are in the code with
   the evidence attached. `k_norm` is a true **LayerNorm** — mean-centred, with a bias — not the RMSNorm
   this model uses everywhere else; `indexer.k_norm.bias` being the only bias in the attention block is the
   tell. And the indexer's RoPE is **half-split** `(i, i+R/2)` where the main MLA path rotates
   **interleaved** `(2i, 2i+1)` off identical frequencies: two conventions in one attention block, and the
   wrong pairing yields finite, plausible scores and a differently-ordered top-k. ReLU is applied per head
   BEFORE the head mix, so a negative head cannot cancel a positive one.
5. ~~**Sparse attention.**~~ **DONE.** The softmax runs over the selected keys only — they are absent from
   the normalisation rather than scored down, which is why a merely-reordered selection still changes the
   distribution. A `shared` layer with no published selection is a hard error, not a fall back to dense:
   57 of 78 layers own no indexer weights, so attending to everything would be a different model that still
   produces text.
6. ~~**Registry.**~~ **DONE**, and `arch_supported` flipped on its own exactly as predicted. GLM-5.2's
   verdict on a 24 GiB host moved from `no backend for mla+dsa attention` to `projected 0.0877 tok/s is
   below the 1 floor` — i.e. it is now refused only by D21, a policy constant.
7. ~~**Conformance.**~~ **DONE — GLM-5.2 PASSES logits and greedy**, `max=1.25e-06 mean=1.70e-07`, the same
   order as all five other fixtures, over 512 teacher-forced positions and 256 greedy tokens with real
   selection active for seven eighths of the sequence.

   **The pass is not vacuous, and that was verified rather than argued.** Forcing the key list back to
   dense makes the same fixture fail at `max=1.15e+00`, greedy diverging at token 57. The sparse path is
   load-bearing.

   Four defects had to be fixed before the gate could even be read, and only one of them was in the DSA
   code: D34 (the loader could not read the fused layout it converts), D33 (the tap tool's rope taps were
   confounded by DSA, in two independent ways), D29 (MLA's latent norms used the wrong eps — a pre-existing
   defect that Moonlight had been silently paying 7.25e-05 for), and D30 (`rope_theta` read from the top
   level only, so GLM-5.2 rotated at 10000 instead of 8000000). The scope's own bisection rule is what
   separated them: `pos0` was clean once the norms were fixed, and a positional divergence with clean
   projections is the rope, not the indexer.

Steps 1-2 are prerequisites and independently useful. Steps 3-5 are the actual work. Step 7 is what makes
any of it believable — and it earned that description: three of the four defects it caught were older than
DSA and none would have been found by reading the code.

### Two risks worth naming now

**The sparse path can be right on average and wrong at the boundary.** `index_skip_topk_offset = 3` and
`index_topk_freq = 4` are off-by-one hazards, and a selection that is correct for most positions and wrong
at layer-group boundaries would pass a mean-error check while failing token-exactness. Stage 1's
token-exact bar is the right gate precisely because it does not average.

**Contexts below 2048 make the sparse path unobservable.** With fewer tokens than `index_topk`, top-k
selects everything and DSA degenerates to dense MLA — so a tiny fixture will pass whether or not the
indexer works. The oracle needs a prompt longer than 2048 tokens, or a shrunk `index_topk`, and shrinking
it is the better choice since a tiny-random model with a 4k prompt is otherwise cheap to run.

## Cross-cutting: CI from day one

Established at G0, extended each gate. Current CI is 3 Release jobs invoking raw `cmake`, one CTest
entry, and **no lint or format job** — there are no format config files in the repo at all.

| Job | From | Purpose |
|---|---|---|
| `soma-header-selftest` | G0 | Every `include/soma/**.hpp` compiles standalone, twice-included, `/W4 /WX` + `-Werror` |
| `seam-check` | G0 | `tools/ci/check_seam.py` — R1 include discipline, R2 no arch names in core code |
| `soma-conformance` | G0 | The full ladder on tiny-random fixtures, **per family**, every commit |
| `ui-api-check` | G7 | `tools/ci/check_ui_api.py` — no Soma panel includes a header carrying in-process state |
| `api-docs-check` | G8 | `tools/ci/check_api_docs.py` — every documented route exists, or is marked `(planned)` |
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

## Open defects

Found and real. Distinct from the table below: that is work deliberately not built, this is work that is
wrong. A defect leaves this list by being fixed or by being reclassified with a reason — never by going
quiet, and struck-through rather than deleted, because how a defect was found is usually worth more than
the fix.

**Every defect is closed except D4, which is parked with its reason recorded and its error message
fixed.** Eleven found, ten resolved; five of the eleven came from a single run of the G8 gate against a
real model, and none of those five was reachable by any fixture in the repo. All of D4–D8 were found by running the G8 gate
against a real model — five defects that every unit test in the repo passed straight over, and the two
that blocked serving were each invisible until the one before it was fixed. That is the argument for
running a gate rather than reasoning about it: D8 could not be seen until D7 stopped hiding it. Two of the resolved ones turned out worse than logged: D3 was
recorded as stale documentation and was hiding a false claim about the G4 gate; D1 was recorded as a
possible auth fault and was a retry budget written six times with three different numbers.

| # | Defect | Found | Severity |
|---|---|---|---|
| D1 | `control_api_external_token_gate` fails intermittently at the transport layer, not the auth gate. **Recurred after the first fix**, which had been applied to only half the file; disambiguation now covers both paths. The flake itself is environmental and remains open. | G8 requantization work | Low — diagnosable now rather than eliminated; see below. |
| ~~D2~~ | The brain grid's tier channel saturated under reduction. **Fixed** — the split now weights by traffic rather than membership; see below. | G7 preview | Resolved |
| ~~D3~~ | §G4's headings were stale. **Fixed** — and fixing them surfaced a claim that was not merely stale but wrong; see below. | Status review | Resolved |
| ~~D8~~ | `load-model` and `restore` applied llama.cpp's preconditions to every backend. **Fixed** — `EngineDescriptor::validate_model_ref`. | G8 gate run, after D7 | Resolved |
| ~~D7~~ | Placement sent the agent's `model_path` to the node instead of the registry's `model_dir`. **Fixed** — `AgentScheduler::model_location()`; the node now receives the container path. Serving is still blocked, by D8. | G8 gate run | Resolved |
| ~~D11~~ | A chat starved the telemetry feed — 1.3 frames/s during generation against 17.3 idle. **Fixed** — the sampler no longer waits on the work it measures; **16.1/s** during generation now. | #11, on the real OLMoE | Resolved |
| ~~D10~~ | Every proxied SSE stream closed instantly with a clean, empty 200: `set_read_timeout(0, 0)` is ZERO seconds in cpp-httplib, not "no limit". Three sites. **Fixed**. | #11, on the real OLMoE | Resolved |
| ~~D9~~ | The heat frame declared a dense `rows x cols` grid and carried a SPARSE array — 878 entries for a 16x64 grid on a real model. **Fixed** — `bucket_heat`'s passthrough branch densifies. | #11, on the real OLMoE | Resolved |
| ~~D6~~ | `GET /v1/models/{id}/conformance` was documented and never registered. **Fixed** — registered, plus `check_api_docs.py` walking the direction `require_complete_coverage()` cannot. | G8 gate run | Resolved |
| ~~D5~~ | One transfer reported two sizes: decimal GB in Python, binary GiB labelled "GB" in C++. **Fixed** — one formatter, `util::bytes_label`, binary with binary labels. | G8 gate run | Resolved |
| ~~D13~~ | No test ran a real Soma engine and a real llama.cpp engine on one supervisor. **Fixed** — `engine_g5` §12, with a committed 244 KiB GGUF fixture. | G8 criteria confirmation | Resolved |
| ~~D14~~ | The descriptor launch path dropped `n_gpu_layers` — and, on inspection, eight more settings. **Fixed** — `build_launch` now calls `build_llama_server_args()`, which its own documentation always claimed it called. | D13 work | Resolved |
| ~~D12~~ | Nothing stopped an image part from being routed to Soma; the 422 gates tested only the agent profile. **Fixed** — one shared capability table, four gates routed through one rule, verified live. | G8 criteria confirmation | Resolved |
| ~~D15~~ | `BackendDecision::explain()` stuttered — `soma (verdict, verdict=hybrid)`. **Fixed**, along with the doubled name it produced in the D12 refusal, and pinned by whole-string assertions. | D12 work | Resolved |
| ~~D16~~ | `compute_plan` sized MLA attention with the GQA formula, and shared experts were counted `n_shared^2`. **Both fixed**, behind `AttentionBackend::weight_bytes_per_layer`; all five families now size to 1.00x of the real tensors. A third bug fell out: `arch::mla::attention_backend()` was declared and never defined, so `kv_bytes_at_ctx` was **zero** for every MLA model. | GLM-5.2 planning | Resolved |
| ~~D18~~ | `convert.py`'s `DENSE_SUFFIXES` was a GQA-shaped allow-list, silently dropping MLA attention, dense-layer MLPs and the `noaux_tc` router bias. **Fixed**, and the "everything must be accounted for" rule the comment always claimed is now enforced as a refusal. All five container fixtures rebuilt and verified to serve. | GLM-5.2 planning | Resolved |
| ~~D19~~ | Container-served models with SHARED EXPERTS silently dropped the shared expert's contribution: `if (out.experts_are_streamed) continue;` skipped the rest of the MoE-layer binding, and the shared-expert binding sat below it. **Fixed** — a guard rather than a `continue`. Also closed the coverage gap that hid it: nothing had ever compared a container's OUTPUT to anything. | MLA container conformance | Resolved |
| ~~D22~~ | The planner did not size DSA's indexer — 0.90x of the real tensors. **Fixed** with `AttentionSpec::dsa`, which gives the IR the per-layer indexer map it lacked; GLM-5.2 now sizes at 1.00x. | GLM-5.2 oracle fixture | Resolved |
| ~~D23~~ | **`apply_container_quant` stamped `group` onto roles nobody asked about**, so D17's three new dense roles began receiving it — and `arch_hash` covers dtype AND group for every role, which silently changed the hash of every already-admitted container. The registry would have read its own records as `StaleRecord` and routed everything to the fallback, and KV checkpoints keyed on the hash would have stopped loading. **Fixed** — an unnamed role is left entirely alone. Found by checking a claim rather than by a test: the comment asserted existing hashes were untouched, and comparing a real admitted model's stored hash to a freshly computed one showed they were not. | D17 | Resolved |
| ~~D24~~ | **`convert.py` segfaulted on a 282-shard checkpoint.** It opened every shard at once — `handles = [safe_open(f) for f in shard_files]` — which is fine for the 1-to-5-shard fixtures and fatal for GLM-5.2: it mapped all 1.4 TB, indexed 59,585 tensors, read six expert tensors and died on the seventh. Reading the same tensors one handle at a time runs indefinitely, which is what identified the handle COUNT rather than any tensor as the cause. **Fixed** — the tensor->shard map comes from `model.safetensors.index.json` and handles open on demand behind a 4-entry cache; both committed containers still convert byte-identically. **How it presented is the lesson**: the crash left a 46 MB partial container and the shell reported exit 0, because the invocation chained a `date` after it. A conversion that dies 0.01% in and looks successful is the worst available outcome, and the fix to that half was to stop chaining anything after the converter. | GLM-5.2 conversion | Resolved |
| ~~D25~~ | **The completeness check ran last, so a 4.5-hour conversion refused after doing all the work.** GLM-5.2 converted all 78 layers and 439 GiB of expert payload, then exited 3: 20 tensors matched no known role. They were the MULTI-TOKEN-PREDICTION head — `num_hidden_layers=78` with `num_nextn_predict_layers=1` means layer 78 exists, carries a full attention block, its own 256 experts and the MTP-specific `eh_proj`/`enorm`/`hnorm`, and is not part of the served stack. `IGNORED_PATTERNS` held a `.mtp` entry that had never matched anything, because the head is identified by layer INDEX (`>= n_layers`), not by any substring. **Two fixes.** The head is now excluded by index. And the check MOVED to before the expert loop: it is pure name arithmetic over the shard index's key list, opens no tensor, and now refuses in **2 seconds** — verified by disabling the index rule and watching the same 20 tensors come back at t=2s instead of t=4.5h. The output file is opened after the check too, so a refusal leaves nothing behind that could pass for a partial success. The allow-list is now built once as an ordered list and used for both the check and the copy loop, since two transcriptions of one rule is how a check ends up asserting something the loop does not do; all six convertible fixtures verify byte-identical to the previous converter. **The lesson is the ordering, not the tensor**: the cost of a validation is not what it computes but where it sits relative to the work it guards. | GLM-5.2 conversion | Resolved |
| ~~D26~~ | **The dense half was read whole into RAM, so D17's saving was cancelled by the load that applied it.** `SafeTensors::ingest` calls `read_whole_file` — the entire F32 `dense.safetensors` becomes one `std::vector<std::byte>`. **Measured**: `soma conform` on the GLM-5.2 container peaked at **73.54 GiB** against 69.30 GiB of F32 dense on disk, and took 255 s to do it. The plan for that same container with `--quant-dense q4_g` reports `dense_resident_bytes` of 10.1 GiB and blesses a 24 GiB host. Worse than a transient spike: non-quantized roles bind as `WeightRef::from_f32(tv->f32(), …)`, a VIEW into that buffer, nothing in the load path ever closes it, and `TensorRole::Router` is deliberately F32 forever — so at least one view always survives and the whole 69.3 GiB stays resident for the model's lifetime *alongside* the ~10.1 GiB of quantized copies. The design note is right that F32-on-disk is a feature ("resident precision can be changed without reconverting a single byte"); what it does not say is that the loader must therefore hold the unquantized bytes to quantize from, and the current ingest holds ALL of them at once instead of one tensor at a time. Invisible until now because every fixture's dense half is a few MB. **Fixed** — `SafeTensors` maps the file instead of committing it. Committing was never necessary: quantized roles read each tensor ONCE and never look at the source again, and the roles that stay F32 keep a `WeightRef` into those bytes for the model's lifetime. A mapping serves both, because pages consumed by quantization are clean and reclaimable the instant the OS wants them while the few still referenced stay by virtue of being referenced. `TensorView` is untouched — `bytes` is still a span over a stable address range — which is why this is a mapping swap rather than a loader redesign. **Measured on the GLM-5.2 container**: peak PRIVATE (committed) memory **73.54 GiB -> 4.38 GiB**, and `conform` got 15% faster besides (255.6 s -> 217.7 s). Reclaimability was proven rather than asserted: under a **hard 6 GiB working-set cap** (`SetProcessWorkingSetSizeEx`, HARDWS_MAX_ENABLE) the same 69.3 GiB dense half runs to **exit 0** with `quant_codec` still passing, peak working set pinned at exactly 6.00 GiB, for a 27% slowdown from paging. A 69 GiB private commit could not have done that at any cap. 26/26 tests pass; clean under `/W4 /WX`. | GLM-5.2 conversion | Resolved |
| ~~D27~~ | **460 GB of quantized experts were written and admitted without one byte being checked against the source.** `quant_codec` round-trips the **dense F32 tensors** through each declared codec and never opens `experts-*.bin`, so it validated the CODEC and said nothing about the payload. **Fixed** — `tools/admission/verify_payload.py`, three passes kept deliberately separate because they prove different things. STRUCTURE covers **100%** of the payload from the index alone: every offset, length, 4 KB alignment, shard membership, intra-shard packing continuity and shard file size — 19,200 slots and 460.535 GB in **0.167 s**, and it is what catches a truncated final shard. EXACT re-quantizes the source and demands byte equality, proving PLACEMENT (it reuses `quantize_rows`, so a codec bug would cancel on both sides — stated in the tool, not glossed). DECODE unpacks with a decoder written against the layout and compares to source, reporting rel_rms against the correct tensor AND against decoys; the **decoy margin** is the real assertion, and it is self-invalidating — a broken decoder makes correct and decoy equally bad and the margin collapses rather than passing vacuously. **GLM-5.2 result**: structure passed, 117 experts sampled across all 75 MoE layers (every shard-first expert, both ends, 8 random), **all byte-exact**, q4_g rel_rms 0.1001 mean / 0.1069 max against a 0.30 ceiling, q6_g 0.0267 / 0.0336 against 0.08, decoy margin 11.4x–14.2x. Cross-check worth noting: the Python decoder's error agrees with the C++ `quant_codec`'s independent measurement to ~2% (0.1001 vs 0.1022, 0.0267 vs 0.0275). `mm_verify_payload` breaks a good container five ways and requires the RIGHT pass to catch each. | GLM-5.2 conversion | Resolved |
| ~~D28~~ | **`mm_fused_experts` had never once run.** Found while registering `mm_verify_payload`: the new test passed in 0.13 s, which is impossible for something that converts a fixture six times. `ctest -V` said `numpy is not installed` — `find_package(Python3)` returns the SYSTEM interpreter, the admission deps live in `tools/admission/.venv`, and both scripts skip on ImportError by returning **0**. So ctest printed `mm_fused_experts ... Passed 0.11 sec` while importing nothing, for the test's entire existence, and D4's "requires the two readers to agree byte-for-byte" was enforced only when run by hand. **Fixed** twice over, because either fix alone still hides the next one: the tests now resolve `tools/admission/.venv` first (CMake logs which interpreter it chose), and a skip returns **77** under `SKIP_RETURN_CODE`, so a missing dependency reads as *Skipped* rather than green. `mm_fused_experts` went 0.11 s -> 6.27 s on the same machine; nothing about it changed except that it started running. **The same shape as the vacuous `arch_hash` comparison in D19's note**: a gate that is structurally present and does nothing reads as covered, which is worse than one that is absent. | D27 work | Resolved |
| ~~D29~~ | **MLA's two latent norms used the wrong epsilon, and one fixture had been quietly paying for it for the backend's whole life.** `arch_ir.hpp` states the RMSNorm epsilon "applies to attention q/k norms, layer norms, and the output norm alike". That is false for this family in BOTH reference implementations: `q_a_layernorm` and `kv_a_layernorm` are constructed as `RMSNorm(rank)` with the class DEFAULT of 1e-6, while only `input_layernorm`/`post_attention_layernorm`/`norm` receive `config.rms_norm_eps`. DeepSeek-V2-Lite hid it perfectly because its `rms_norm_eps` IS 1e-6, so the two agreed by coincidence. **The witness was Moonlight**: `rms_norm_eps=1e-5`, conformance error 7.25e-05 — seventy times every other fixture, under the threshold, passing, and unexplained. **Fixed**; Moonlight fell to 9.54e-07, in line with the rest, which is the independent confirmation that this was the cause rather than a plausible story. GLM-5.2's `pos0` fell 3.16e-05 -> 1.25e-06. | DSA backend | Resolved |
| ~~D30~~ | **`rope_theta` was read from the top level only, so GLM-5.2 silently rotated at 10000 instead of 8000000.** transformers migrated `rope_scaling` to `rope_parameters` and moved `rope_theta` INSIDE it. The adapter followed half the migration: it learned the new name for the scaling block and kept reading theta from the top level, where GLM-5.2 does not state it — so the 10000 default applied to a model that says 8e6. **Quiet in the worst way**: every projection, norm and shape stays correct and only the rotation angles are wrong, so the model loads, runs, produces finite logits, and matches its reference EXACTLY at position 0, where rotation is the identity for any theta. It surfaced as `max\|diff\| = 1.8` on the `k_pe_rot` tap with every tensor feeding it clean at 4e-07, and was confirmed in one line by `SOMA_MLA_PROBE`, which printed `1.0 0.1 0.01 0.001` — theta 10000 written out. **Fixed**: the nested value wins when present, because `standardize_rope_params` writes the authoritative value there. | DSA backend | Resolved |
| ~~D31~~ | **`F32Model::quantized` was reserved one element short of its own worst case.** The budget was `n_layers * (3*n_experts + 10)`, and a shared-expert MoE layer already needed `3*E + 11` (5 attention + 3E routed + 3 shared); DSA's three indexer tensors took it to `3*E + 14`. Every `WeightRef` points INTO that vector, so overflowing it does not degrade performance, it dangles every weight bound so far — at once, silently, in a release build. Never observed, because the fixtures that exercise quantized dense roles are small; found by counting rather than by a failure. **Fixed** with the worst case enumerated per role in a comment, so the next role added is added to a number that shows its work. | DSA backend | Resolved |
| ~~D32~~ | **The oracle fixture shrank `index_n_heads` 32 -> 4, which manufactured ties and made token-exactness untestable at exactly the positions DSA exists to exercise.** An index score is `sum_h w[h]*relu(q[h].k)`, so it is exactly 0.0 only when ReLU zeroes EVERY head at once — probability ~2^-H. Measured on the fixture: 50.69% of scores at 1 head, 27.12% at 2, 13.52% at 3, 6.99% at 4, a clean halving, extrapolating to ~2e-8% at the real 32. At 4 heads, 47 of 768 selective queries had the top-k cut land on a tie, and ties are resolved by `torch.topk`'s internals — which reproduce neither ascending nor descending index order. Soma matched the reference on **721 of 721** queries where the cut was untied and 7 of 47 where it was tied: a correct implementation being graded on a coin flip. **Fixed** in `make_oracle.py` — `index_n_heads` is preserved, costing ~49k parameters. `index_head_dim` still shrinks, since head WIDTH does not affect the sign statistics. The generator's rule is "SHRINK DIMENSIONS, PRESERVE SEMANTICS", and this is a dimension that turned out to be semantic. | DSA backend | Resolved |
| ~~D33~~ | **`dump_activations.py`'s rope taps were confounded by DSA in two independent ways, and it is the tool that finds bugs.** (1) It attributes taps to layers with a counter incremented per call, sound only because "apply_rotary_pos_emb is called exactly once per layer, in order" — DSA calls it TWICE on a `full` layer, once for attention and once for the indexer, so every tap after GLM-5.2's layer 0 was filed against the wrong layer, visible as `q_pe_rot` alternating between 4096 and 32768 elements (4 attention heads vs 32 indexer heads). (2) `apply_rotary_pos_emb_interleave` READS interleaved pairs and WRITES them concatenated — same values, permuted layout — which attention cannot see, because q and k are permuted identically and their dot product is unchanged, but which showed as a 1.6 divergence no amount of reading the rotation code explains. **Both fixed**: the indexer's call is counted separately and not recorded, and the interleave output is un-permuted to the engine's layout. Only after this did the tool point at the real defect, D30. | DSA backend | Resolved |
| ~~D34~~ | **The engine could convert GLM-5.2 and not load the checkpoint it converted from.** D4 taught `convert.py` the FUSED expert layout; the fp32 loader still read per-expert names only. Since conformance runs against the fp32 SOURCE, the one model the fused reader existed for could not be conformance-tested at all — it failed at `model.layers.3.mlp.experts.0.gate_proj.weight not in checkpoint`. **Fixed**: the loader reads both layouts, with the shapes checked against the IR rather than inferred from the tensor, so a checkpoint disagreeing with its own config fails at bind instead of computing nonsense. | DSA backend | Resolved |
| ~~D35~~ | **MLA's `kv_bytes_per_token` counted one cache plane while two are allocated — a 2x under-report, in the optimistic direction.** `KvCache::open` allocates a K and a V plane of equal width for EVERY family. GQA's accounting says `2 * n_kv_heads * head_dim` and matches; MLA's said `kv_lora_rank + qk_rope_head_dim` and did not. Nothing caught it because nothing could: the cached path was a stub, so no MLA model ever allocated a cache it then had to fit inside. On GLM-5.2 at 4096 context that is a reported 2.94 GB against a real 5.89 GB, next to a 9 GB expert cache on a 24 GiB host — the same failure shape as D26, a planner number wrong in the direction that says yes. **Fixed**, and the planner now shrinks the expert cache from 11.95 GB to 9.01 GB to stay inside the budget, which is the arithmetic working rather than a new problem. | MLA/DSA cached decode | Resolved |
| ~~D36~~ | **Core computed every family's KV cache geometry with GQA's formula.** `KvCache::open` hardcoded `hkv_ = n_kv_heads * head_dim`. MLA does not store per-head K and V at all — it stores a `kv_lora_rank` latent plus one shared RoPE segment — so there was no width at which the cached path could have been correct, which is precisely why it had been left as a stub returning `Unsupported`. The stub was the right call and the comment naming the cause was accurate; what was missing was anywhere to record it (see D37). **Fixed** with `F32Backend::kv_floats_per_layer`, the same seam argument as `weight_bytes_per_layer` in D16 — null means the GQA default, so no existing backend had to change. GQA's aggregate initialiser DID have to change, and the compiler caught it only because the shifted member types happened to disagree; both backends now name their members. | MLA/DSA cached decode | Resolved |
| ~~D37~~ | **The scoping note said DSA adds nothing per-sequence. It adds an indexer key cache.** The claim was that the top-k index is recomputed every forward and never persists between steps — true, and the wrong noun. The INDICES are per-step; the KEYS behind them are per-sequence. `k = k_norm(wk(hidden))` depends on a past token's hidden state AT THAT LAYER, which is gone by the next step, so it cannot be recomputed and must be stored. Invisible while only prefill existed, because prefill has every hidden state in hand. **Fixed** by caching the indexer key in the second cache plane, which MLA leaves empty. The claim that this needs no `persist_format_id` bump still holds — the layout changed before any DSA checkpoint was written — but the claim that `kv_bytes_per_token` was "already right" did not survive, and is D35. | MLA/DSA cached decode | Resolved |
| ~~D38~~ | **`MlaSpec::absorb_weights` described a feature nothing implemented, and the conformance A/B it documented had never run.** The field's comment says weight absorption is "THE reason AttentionBackend has this hook" and that it is "skipped when absorb_weights is false, which the conformance harness uses to check the absorbed and unabsorbed paths against each other" — an accurate description of a thing that did not exist, since `prepare_weights` never folded anything and no harness ever toggled the flag. Another case of the description outliving the code. **Fixed** — the flag now selects between two real implementations of the fp32 cached decode, and `soma_decode_kv_g4` runs BOTH for every fixture, which is the A/B the comment promised. The unabsorbed form is deliberately kept: it is the reference absorption was checked against, and deleting it would remove the thing that makes the fast path believable. | absorbed decode | Resolved |
| ~~D41~~ | **`soma serve` hardcoded the quantization it expected, so the configuration `soma plan` blesses was not reachable through it.** main.cpp's header says `plan` is a subcommand of the same binary because "the planner it runs is the one the server runs... must not be able to disagree". Serve set `q4_g/q4_g/q6_g @128` in three literal lines and never read container_meta.json. Two consequences: a container converted at any other setting was REFUSED with "container experts are 4864 B but the IR's quantization map implies 4352 B", which reads as a corrupt container rather than a wrong assumption; and the resident half could only ever be F32, so GLM-5.2 served with a 69.3 GiB dense half and a 0-byte expert cache while `plan --quant-dense q4_g` reported 10.9 GiB and `stream`. **Fixed** by extracting `resolve_arch(dir, overlay, out)` — adapt config.json, apply the container's own record, then the caller's overlay — and having BOTH `compute_plan(dir, ...)` and serve call it, so there is one answer to what a model is. `--quant-dense` / `SOMA_QUANT_DENSE` is expressed as a container_meta-shaped overlay through the same applier, so it cannot mean something a container could not be. The previously-refused container now serves, and `dense_resident_bytes` agrees between plan and serve to the byte. | generate a token | Resolved |
| ~~D42~~ | **The loader's `arch_hash` covered the all-F32 default map, not the quantization it actually loaded at.** `compute_arch_hash` ran three lines ABOVE `out.arch.quantization = quant`, so every quantization of one architecture hashed identically at load. That defeats the reason QuantMap is inside the hash: the same weights at two quantizations are two models, with two verdicts and two sets of KV checkpoints. A checkpoint written under q4_g would replay under q8_0 with nothing detecting it — the exact failure plan.cpp's comment says this hash exists to prevent, and which it records as already having been observed once. A subtler second version of the fault the same comment block fixed: first the hash was empty and every comparison passed vacuously; then it was populated from the wrong state. **Found by checking the HASH and not just the bytes** — D41's fix made `dense_resident_bytes` agree between plan and serve to the byte while the hashes still differed. **Fixed** by ordering the assignment before the stamp; the two now match exactly, and 29/29 tests pass unchanged. | D41 | Resolved |
| ~~D40~~ | **The scheduler kept a second copy of the KV cache geometry, and it segfaulted the 744B model on its first request.** D36 moved the cache width behind `F32Backend::kv_floats_per_layer` because core had it hardcoded as `n_kv_heads * head_dim` — GQA's formula. The fix landed in `KvCache::open`; `scheduler.cpp` computed the SAME formula independently for `KvRow::stride` and `KvRow::hkv`, and that is the copy the serving path uses. So the allocation followed the backend and the addressing did not. **Found by generating a token**: `soma serve` loaded the real GLM-5.2 container, reported `verdict=stream`, listened, and died with a segfault 4 s into the first `/v1/chat/completions`. Nothing in the suite at the time could have caught it — `soma_decode_kv_g4` drives `forward_step_f32` with rows it builds from the cache itself, which is correct by construction and therefore blind to a caller that builds them wrong. **The MLA reading is subtler than it first looks**: writes and reads used the same wrong formula, so while offsets stayed inside the allocation the arithmetic was self-consistent and the output correct — DeepSeek-V2-Lite served the identical token `'31 '` before and after the fix. Past the end it is heap corruption, and GLM-5.2's 78 layers reached 2.1x beyond. **Fixed** by giving `KvCache` a `stride()` accessor and having the scheduler take both numbers FROM the buffer it is about to point into, so there is no second formula to disagree. `soma_scheduler_g3` now runs through GQA, MLA and MLA+DSA fixtures, making the scheduler's row construction part of the gate rather than testing only rows assembled by the cache test itself. | generate a token | Resolved |
| ~~D39~~ | **`arch::mla::prepare_weights` was declared, never defined, and documented in detail — found by auditing my OWN increment for the drift it had just created.** Its comment described folding the KV up-projections into Q at load, running once from `load_model()`, and being skipped when `absorb_weights` is false. None of it happened: the function had no definition, and `mla::attention_backend()` never wired the pointer. Benign only because nothing called it — and that is precisely the state `attention_backend()` was in at D16, where a declared-never-defined function meant the planner silently sized MLA with GQA's formula. The absorbed decode made the drift worse rather than better, since absorption now genuinely exists and does NOT happen at load. **Fixed** — defined and wired as an explicit no-op that says where absorption really lives and why load time was the wrong place: folding at load means a resident transposed fp32 copy of the up-projection, 1.96 GB on GLM-5.2, to save arithmetic that was never the bottleneck (6.3e6 element reads per layer per step, against the 1.5e8 the absorbed attention already does). `MlaSpec::absorb_weights` said "at load" too, and now says per step. | absorbed decode | Resolved |
| ~~D21~~ | **The throughput floor rejected the model the concept was proven on.** `kMinProjectedTokS = 1.0` refused GLM-5.2 at every host size — 0.087 tok/s on a 24 GiB workstation, 0.79 at 128 GiB with a 7 GB/s disk — while Colibri had served those same 744B weights on 16-24 GB to someone who found the result useful. The constant's reasoning was sound as written; what it could not express is that "usefully served" depends on who is asking, and for a 744B model on a workstation 0.1 tok/s may be the entire point. **Resolved by the user as a POLICY call**: it is now `HostBudget::min_tok_s`, beside `ram_total_bytes` and `disk_bandwidth`, because the verdict was already documented as a property of (model, quantization, host budget) and this belongs in the third. `--min-tok-s` on the CLI. **0 means unstated and resolves to 1.0, not to "no floor"** — a default-constructed budget guards exactly as before, so nothing admits that did not admit before and lowering the bar takes a deliberate statement. The refusal names the figure AND whether it was chosen or inherited, because "raise your tolerance" and "this host is too small" are different answers. GLM-5.2 at `--min-tok-s 0.05` now plans as **stream**. Covered by `container_g2`, which also asserts the floor moves the verdict and NOTHING the verdict is computed from; verified load-bearing by making compute_plan ignore the budget field. | D17 | Resolved |
| ~~D20~~ | A node could not be brought up from its config file alone, and the refusal was invisible: the node's warning died in a buffer (spdlog flushed only on `err`) and CLI mode has the console sink off. **Fixed** — `flush_on(warn)`, a once-only console diagnostic naming the remedy, and the pairing path documented in `tools/mantic-mind.toml`. The refusal itself is unchanged; only its discoverability was wrong. | deployment test | Resolved |
| ~~D17~~ | The dense half was F32 by omission, not by design — the loader could always quantize it, and only the three EXPERT roles were settable. **Fixed** with `dtype_dense` / `--quant-dense`. GLM-5.2's resident half falls 68.6 -> 10.0 GiB and it now fits a 24 GiB host. | GLM-5.2 planning | Resolved |
| ~~D4~~ | `convert.py` could not read the FUSED expert layout, and was parked because no oracle could verify an implementation. **Fixed** — transformers 5.12.1 supplied the reference, the split was settled by measurement (CONTIGUOUS, not interleaved), and `mm_fused_experts` now requires the two readers to agree byte-for-byte. | G8 gate run | Resolved |

**D1 — resolved, and it was never an auth bug.** The cause was already written down in the test, three
lines above the failure:

> `mm::HttpClient` opens a fresh connection per request; rapid sequential connect/close cycles on
> Windows loopback occasionally fail at the transport level (`status == 0`). Retry those.

A known environment flake with a retry to match — except the retry budget had been written **six times
with three different numbers**. `with_retry` allowed 8 attempts; two SSE helpers allowed 3; another
`with_retry` allowed 3. The SSE path is the one carrying the auth negatives, so the thinnest budget sat
directly under the assertions where a transport failure is hardest to tell from a wrong verdict:
`status == 403` fails identically whether authorization returned 200 or the request never arrived.

That is precisely how it presented — one intermittent failure, at the auth assertion, in the suite's
most alarming test. The diagnosis cost more than it should have because **the test could not distinguish
the two findings**, which is the more interesting defect of the two.

Three changes:

- **One budget, `kTransportRetries`, stated once** with the reason, replacing all six sites. 6 attempts
  with linear backoff (50, 100, … 300 ms; 1.05 s total) rather than a flat interval — the failure is a
  socket in TIME_WAIT or a listener mid-accept, and a fixed short retry re-hits the same window every
  time, which is why the old 3×50 ms "failed" identically three times and read as a verdict.
- **`status != 0` is asserted before the status is compared**, so a transport failure fails on its own
  terms rather than as a wrong status code.
- **A diagnostic that names the cause**: `TRANSPORT FAILURE on invalid_chat after 6 attempts: status=0`
  followed by "the auth assertions below are about to fail for a reason that has nothing to do with
  auth". Verified by forcing the condition.

**It returned — and the banner did not fire, because the fix had only been applied to half the file.**

The recurrence came during the conformance-oracle increment: one failure at `reliability_tests.cpp:1407`,
clean on re-run. Line 1407 is inside the shared `expect_error` lambda, so the report named the assertion
helper rather than the call, and it printed a bare status mismatch — precisely the ambiguity D1 was
supposed to have removed.

The reason is worth recording, because it is a failure mode of the fix rather than of the diagnosis.
`kTransportRetries` was unified across all six sites, but the other two changes — the loud banner and
**asserting `status != 0` before comparing the status** — were only made on the SSE helper
(`reached_server`). The plain-request path got the shared budget and none of the disambiguation. So a
fix written to make transport failures self-identifying left the other half of the file exactly as
misleading as before, while the roadmap recorded it as done.

Both are now on both paths. `expect_error` checks `status == 0` first, prints the same banner naming the
**call site** via `__LINE__` rather than the lambda, and records `status != 0` as its own assertion.
Verified by forcing the condition — pointing one request at a dead port:

```
  TRANSPORT FAILURE at line 1426 after 6 attempts: status=0
  (the assertion below is about to fail for a reason that
   has nothing to do with authorization)
CHECK failed at line 1411: resp.status != 0
```

**Still not reproduced deliberately**, and the underlying flake is not fixed — retries make it rare, they
do not make it impossible. What has changed is only that the next occurrence will say which of the two
findings it is. That is the whole claim; the earlier "resolved" was stronger than the evidence, which is
why the recurrence is recorded here rather than quietly re-fixed.

#### A string nobody asserted

`BackendDecision::explain()` rendered `soma (verdict, verdict=hybrid)`. The reason enum's name and the
field it qualifies are the same word, so naming both said it twice.

It survived because **nothing asserted its text**. `routing_g5` called `explain()` nine times — every one
as the `detail` argument of `check()`, which prints but does not compare. The function was exercised
constantly and verified never. It only became worth fixing when D12 started quoting it inside a 422, at
which point a wart in a log line became a sentence users read.

`Verdict` now renders as `verdict=hybrid` alone. Every other reason keeps both halves, because they are a
DIFFERENT fact from the verdict — `fallback (override_refused_conformance, verdict=reject)` says what was
asked for and why it was refused, and collapsing that would lose one of them.

Pinned by whole-string equality rather than substring. A `contains` check would have passed the stutter
too, which is presumably how nine call sites managed to look like coverage.

**The composite had a second copy of the same problem.** With the stutter gone the refusal read
`the 'soma' engine serving this agent does not accept images (soma (verdict=hybrid))` — the reason
already names the engine, so quoting the id as well said "soma" twice inside nested parens. Colon form
now, and the reason carries the name:

```
the engine serving this agent does not accept images: soma (verdict=hybrid)
```

The id is still quoted when there is no reason to carry it, because otherwise the message would not say
WHICH engine at all. Both shapes asserted, including the negatives — no `'soma'` when a reason is
present, and no `((` in any form.

A footnote on layering that the assertions made explicit: `explain()` says `fallback`, not `llama-cpp`.
The choice is a ROLE, and `AgentScheduler::resolve_backend` maps it to an engine id one layer up. My
first draft of the test asserted `llama-cpp` and failed — correctly, and the expectation was what was
wrong, not the code.

#### The stage that looks at the weights

Stage 1 proves the ARCHITECTURE against tiny-random weights and says nothing about the checkpoint an
operator ships. Stage 2 is the other half, and until now it reported `skipped` — which meant the strongest
guarantee in the routing system was unreachable. `reject` is the one verdict that overrides an explicit
`backend_override: soma`, precisely because it means conformance failed; with stage 2 never running,
nothing could produce it from real weights. The interlock existed and had no way to trip.

Both halves already existed and had never been connected: `tools/admission/make_reference.py` writes a
bf16 forward over the real checkpoint into the same `SOMAORCL` container the tiny oracles use, and
`tests/soma/stage3_g2.cpp` had the KL comparison. Neither was reachable from `soma conform` or the
pipeline.

**Measured on OLMoE-1B-7B-0924 at q4_g/q6_g/g128**, admitted end to end with the pipeline building its
own reference:

```
real_logit_kl  passed   mean=0.0367 (tol 0.05)   p95=0.1435 (tol 0.25)   top-1 94.1%
```

The ladder now reports **4 of 5 stages ran**, against 3.

**Why two metrics rather than one.** The perturbation tests make the case better than argument does. A
reference sharpened 3x — identical ranking, wrong distribution shape — fails at p95 0.99 while top-1
agreement is **95.8%, HIGHER than the passing run's 94.1%**. A gate on argmax alone would have called
that a pass. Conversely KL can stay small while top-1 drifts on near-ties. They fail differently, so both
are reported.

**The failure path is real, and classified.** Two deliberate corruptions, both caught:

| Perturbed reference | Result | Classified |
|---|---|---|
| vocab axis shuffled — unrelated | FAIL, mean KL 19.0, top-1 0.0% | not a quantization finding |
| logits sharpened 3x — same ranking | FAIL, p95 0.99, top-1 95.8% | quantization finding |

That distinction is the point of the stage. A quantization finding says requantize a role or tighten a
group; a degenerate one says the container or the reference is wrong and the quant map is innocent.
Sending an operator to re-run conversion for the second costs an hour and arrives back in the same place.

**And the degenerate message was wrong, which the measurement showed.** It said "essentially uniform" for
a case scoring mean KL 19.0 against `ln(vocab) = 10.8`. That is arithmetically impossible for a flat
engine: `KL(ref||uniform) = ln(vocab) - H(ref)`, so it can never EXCEED `ln(vocab)`. A mean above that is
a CONFIDENT engine pointing somewhere else — a mismatched checkpoint or vocab ordering, not dead weights.
The two now say different things, split on a hard bound rather than a judgement call.

**One real defect fixed on the way.** The harness hardcoded `q4_g/q4_g/q6_g@128` instead of reading the
container's own map — so a model admitted through `QuantOverride` would have been dequantized with the
wrong map and scored as a different model than the one anyone would run. It now reads
`container_meta.json`. A mismatched map is caught before the forward, by a size check:
`container experts are 3997696 B but the IR's quantization map implies 3538944 B`.

**Cost, stated plainly.** The reference pass is the slowest thing in admission — it loads the source at
bf16 — and it is bounded to 256 positions in-pipeline so a distributional check does not dominate a
pipeline that also converts weights. It is non-fatal, like the oracle: on a host that cannot hold the
checkpoint it fails and stage 2 reports `skipped`. That distinction is load-bearing in the other
direction too, and the header says so — no evidence must never read as adverse evidence, because only one
of those should ever become a `reject`.

#### The launch path that ignored its settings

Logged as "`n_gpu_layers` is dropped". It was nine settings: **ctx_size, n_gpu_layers, n_threads,
n_threads_http, parallel, batch_size, ubatch_size, flash_attn, and the operator's `extra_args`**. An
engine started through `EngineSupervisor` ran on llama.cpp's defaults no matter what was configured.

The sharpest way to state it is that the code contradicted its own documentation.
`EngineDescriptor::build_launch` is declared with:

> "For llama.cpp this wraps the existing, unit-tested `build_llama_server_args()`"

It did not. It hand-rolled a five-flag argv — model, port, host, mmproj, slot-save-path — beside a pure,
already-unit-tested builder that produced the full one. The doc comment was right and the implementation
was wrong, which is the failure mode a doc comment is worst at surviving.

**Why nothing caught it.** `engine_g5` §1 asserted `build_launch`'s argv for **soma** and never for
llama.cpp — so the one descriptor that had a rich argv to get wrong was the one not checked. Fixed by
asserting the fallback's argv with deliberately distinctive values, so a match cannot be a coincidence of
some other flag carrying the same number. Reverting to the hand-rolled argv fails all seven.

One of those assertions is not about a value surviving but about it being TRANSFORMED: llama-server hosts
a single shared context of `ctx_size * parallel`, so the number on the wire is not the number in the
settings. It is asserted at 4608 (1536 x 3) rather than 1536, because a builder that passed the raw
ctx_size would under-provision every slot but one and still look correct to a naive check.

**The severity is the direction.** `n_gpu_layers` defaults to `-1`, meaning *all layers on GPU*. So the
dropped setting did not fall back to something conservative — it fell back to the most aggressive
possible value, and an operator who had turned layers DOWN to fit a busy host got them turned back up.
That fails worst exactly when the host is most loaded.

Confirmed on a live llama-server that the flags change behaviour rather than merely appearing in argv:
no flags gives `n_ctx=512, n_parallel=4`; with the computed flags, `n_ctx=256`. The old descriptor sent
no flags.

**Structural note.** `llama_runtime.cpp` moved from the `mantic-mind` executable into `mm_node_engine`
so the descriptor can reach the builder — the same move `kv_checkpoint_header.cpp` already makes for the
KV codec, and for the same reason: a wire format, or an argv contract, should not acquire a second
definition because of a link boundary. It is a light TU (util + filesystem, no HTTP), and static-library
linkage pulls only what is referenced.

#### Images and the engine that serves them

The criterion — "image content parts return 422, not a dropped part" — read as done because a 422 existed
and was tested. What it tested was the AGENT PROFILE's `vision_settings.enabled`, the operator's intent.
Whether the engine on the far end could accept an image was never asked, and Soma is text-only by
construction. So an agent with vision switched on, serving a model that earned a streamable verdict, sent
image parts to an engine that would never look at them.

That is the failure mode worth naming: not a crash, a silently text-only answer to a question about a
picture. The request succeeds, the model responds, and nothing anywhere says the image was discarded.

**Why it could not be fixed where it was found.** `EngineDescriptor::supports_vision` already held the
right values — `false` for Soma, `true` for llama.cpp — and was never read outside the file that set
them. Control, which is where a request carrying an image actually arrives, cannot reach the node's
descriptors. The tempting fix is a literal `engine_id == "soma"` in control, which is precisely what the
descriptor pattern exists to prevent and exactly how the backend-attribution bug got in.

So the fact moved to `common/engine_capabilities.hpp`, where the node descriptors and the control API both
read it and neither owns it. The descriptors now derive `supports_vision` from it rather than asserting
it, and a test asserts the node's view EQUALS the shared function — which is what makes "single source"
true rather than merely intended. Host-level facts deliberately stay out: whether a machine has a GPU is
`NodeCapabilities` and varies per host; this is what the engine SOFTWARE can do.

**One rule, four gates.** The OpenAI-compat route, the SSE chat route, its attachment path, and the local
chat helper each carried their own copy of a one-condition check — which is how the first condition came
to be checked everywhere and the second nowhere. All four now call `image_refusal()`. The rule itself is
pure and lives in `common/`, so it is asserted without standing up a server, a registry and an admitted
record just to discover what a two-condition predicate does; the server method is only the lookup.

An unknown engine refuses. A new engine that has said nothing about vision gets a clear refusal an
operator can act on, where the other default silently drops the image and answers as though it had been
read. API-backed agents are exempt — they own no node-local engine and the remote provider's own
capabilities govern.

**Verified live, all three cases**, against the admitted OLMoE:

| Agent | Result |
|---|---|
| vision on, model routes to **Soma** | `422 the 'soma' engine serving this agent does not accept images` |
| vision on, model routes to **llama.cpp** | `200` — no regression |
| vision **off** | `422 this agent profile does not accept images` — unchanged |

The middle row is the one that had to keep working: a capability check that refused everything would also
have "closed" D12.

Two mutations confirm the assertions hold it. Restoring the old profile-only rule fails five of them,
including every message-content check; letting a descriptor drift back to a literal fails the
equality assertion and nothing else, which is the drift it exists to catch.

#### Two engines, one supervisor

The coexistence criterion had no test, and the nearest thing to one asserted the opposite arrangement
(§5: two Soma agents SHARING a process — that is Soma's per-sequence KV slot, not coexistence). So the
"Soma is a peer, not a replacement" claim rested on code nothing ran.

**The obstacle was a fixture, not a test.** llama-server needs a GGUF and nothing in the tree could
supply one: `tests/fixtures/tiny/*` are weights+config only, carrying no tokenizer, because the fp32
oracle feeds raw token ids and never needed one; llama.cpp's own `ggml-vocab-*.gguf` are the mirror
image, vocab with no weights, and llama-server dies on them with "missing tensor 'token_embd.weight'".
The real 13 GB source converts, and a multi-gigabyte artifact is not a fixture — a test whose fixture
cannot be committed is a test that only ever runs on one machine.

So `tools/testing/make_tiny_gguf.py` writes the smallest thing that is still genuinely a llama model:
2 layers, d=64, a real SPM vocab with byte fallback, random weights. **244 KiB**, committed. Verified
loadable end to end before being used — llama-server reports `arch = llama`, `n_vocab = 296`, tokenizes
via byte fallback, and generates from token ids.

Random weights are the point rather than a compromise. §12 asserts that two engine types COEXIST; no
logit participates in that. What it deliberately does NOT assert is anything llama.cpp generates — random
logits produce random tokens, which are invalid UTF-8, and llama-server errors building a response string
from them. That is inherent to a random model, and asserting around it would be asserting noise.

**§12 was already load-bearing before it was written.** `engine_supervisor.cpp` carries the comment
"From the descriptor, never the literal `llama-cpp`. `make_slot_info()` used to hardcode it, so every
slot reported the same backend regardless." That was a real bug, and until §12 no test could have caught
it regressing, because no test ran two different engine types. Re-introducing it — pinning
`info.backend` to a literal — turns §12 red with `soma + soma`.

**It also destabilised a neighbour, which is worth recording rather than quietly fixing.** With §12 in
its original position, §10 ("the engine really did batch them") failed 1 run in 3 with
`max observed batch = 1`; with §12 skipped, and on the commit before it, 4 of 4 passed. llama-server
loads the CUDA backend at startup, and that plus its teardown closes the timing window §10 needs to
observe a real batch. The fix is ordering — §12 runs last — rather than a sleep, because coexistence has
no dependency on running early and a sleep only moves the same race. Measured after the move: 4 of 4,
batch 3-4.

**The fixture nearly was not committed at all.** `.gitignore` carries a blanket `*.gguf`, which silently
swallowed it — `git status` simply did not list the new directory. Caught by checking `git check-ignore`
rather than trusting the claim already written into this document, which at that moment said "committed"
and was false. A negation rule now keeps exactly this path, and the test distinguishes the two absences:
a missing llama-server is "not configured on this host", a missing GGUF is "the committed fixture is
MISSING — build it with make_tiny_gguf.py", because those want different reactions.

**When the binary is absent it skips, loudly.** llama-server is provisioned per host, so CMake passes
`MM_LLAMA_SERVER` and §12 reports `SKIPPED` plus "the G8 coexistence criterion is NOT covered by this
run", and the verdict line reads `OK (1 SECTION SKIPPED)`. A criterion that reports green because its
fixture was missing is worse than one that reports nothing. A path that was GIVEN but does not exist is
reported as a bad path rather than silently downgraded to "not configured".

#### `capacity_pressure`: the signal with no coverage

This is the code that decides whether a failed placement gets a second chance, and it had **no test at
all** — implemented across both engines, the node proxy and the scheduler, asserted nowhere. It was found
by checking the G8 criteria against the suite rather than against their names, which is the same check
that should have caught the half-applied D1 fix.

`capacity_pressure_is_structured` now covers it. The happy path is one string compare and is not the
point; the rule worth pinning is **precedence**, and three mutations confirm the assertions are
load-bearing rather than decorative:

| Mutation | Caught by |
|---|---|
| Structured code no longer authoritative (fall through to prose when the code isn't capacity) | the three authoritative-negative assertions |
| Drop the top-level `{"code":…}` branch | a positive AND a negative, in opposite directions |
| Drop the 503 status-derived fallback in `EngineError::parse` | both `through_client(503, …)` assertions |

The middle one is the instructive failure. Deleting one branch simultaneously stopped the node's real
wire shape from earning a retry **and** stopped the code from overriding "out of memory" prose — a hard
failure where a retry was correct, and a spurious eviction where it was not, from a single deletion.

**Three things the test found that reading the code did not.**

*The wire shape is neither of the two you would guess.* The node's load handlers emit
`{"error":"<prose>","code":"capacity_pressure",…}` — `error` is a STRING with the code beside it, not the
nested object `soma serve` produces. Both shapes are real and both are now asserted.

*The matcher cannot be deleted yet, and the reason in the comment was wrong.* It claimed to cover "a
stale llama-server on the far side of a rolling upgrade". llama.cpp's prose never arrives unlabelled —
the node translates it to a code at the boundary in `engine_error_code_for`. The fallback covers a stale
**node**, which is a different component and a different upgrade story. Corrected in place. Deleting it is
safe exactly when no pre-code node can still be in the cluster: a deployment fact, so the criterion stays
open rather than being quietly reworded.

*The two coded responses are exactly the two endpoints the scheduler reads.* Only 2 of the node's 28
error responses carry a structured code — which looks alarming until you check which two: `load-model`
and `restore-slot`, and those are precisely the two the scheduler calls
`response_indicates_capacity_pressure` on. That is coherent, not accidental, and the 26 uncoded responses
are on paths this function never sees. Worth writing down because the raw ratio invites the wrong fix.

**One honest limit.** `EngineError::is_capacity_pressure()` has **no production consumer** — the
scheduler uses its own matcher, and the only callers of the `EngineClient` parser are these new
assertions. That section pins the path `EngineClient` is being built toward, not a live one, and the test
says so rather than implying coverage it does not have. The shape divergence between the two parsers is
free to fix now and expensive to discover once something depends on it, which is why it is pinned early.

Two small structural changes fell out: `EngineError::parse` is now a static member rather than a function
hidden in a translation unit — it constructs an `EngineError`, so it belongs on the type — and
`response_indicates_capacity_pressure` is public, for the reason `model_location` already is. Reaching it
through `ensure_agent_running` would need a node that refuses on demand, which tests the harness more
than the rule.

#### Telemetry against a live model

The G6 line asked whether a client at maximum telemetry measurably costs chat latency. Measured on the
admitted OLMoE — 16 layers x 64 experts, 1024 cells — with a watcher at `hz=10&resolution=full`, ten
pairs, the order of the two arms alternating so any drift cancels:

```
baseline median  3.384s   sd 0.103s
loaded   median  3.314s   sd 0.126s
per-pair delta   median -0.74%   range -4.39% to +2.07%
baseline noise   sd is 3.05% of median
```

**Within noise.** The aggregation-in-engine claim holds at this scale.

**The caveat, stated rather than glossed:** the line says "a 60k-expert model" and this is 1024 cells.
That is three orders of magnitude closer than the 4x16 fixtures and still not what was asked. What the
measurement does establish is that the cost is not per-token — it is per-frame and bounded by grid size,
which is the mechanism the claim rests on. A 6144-cell model is 6x the frame payload at the same tick
rate, not 6x the interference.

**The converse is what the measurement actually found, and the gate never asked about it.** A chat
collapses the feed: 17.3 frames/s idle, **1.3/s during generation**, 18.7/s after. Logged as D11.
Telemetry does not slow chat; chat all but stops telemetry.

**Three attempts, two of them invalid, and the honest part is why.** The first measured baseline against
baseline — the watcher received zero frames because of D10 — and reported "within noise" from data where
the treatment was never applied. It also drifted 5.8s to 16.8s as one conversation grew turn by turn.
The second fixed the drift by passing a fresh `conversation_id`, which is a hint for an EXISTING
conversation, so every chat answered "conversation not found" in milliseconds and all ten pairs were
discarded. Only the discard guard — added after the first failure, on the principle that a measurement
which cannot tell whether its treatment was applied is not a measurement — kept a run of instant
failures from being reported as a very fast result.

**D9 — a wire format that promised density and delivered a sparse list.**

`MemoryHierarchy::heat()` returns a SPARSE snapshot: it skips every expert that
has neither fired nor been made resident, and each `HeatCell` carries its own
`(layer, expert)`. The wire format has no coordinates — `heat_frame_json` emits
`counts` and `tiers` as flat arrays beside `rows` and `cols`, so a consumer reads
them as dense and indexes `r * cols + c`.

`bucket_heat` has two branches. The bucketed one always built a dense grid and
scattered sources into it by coordinate. The passthrough one — taken when the
grid is already under the 4096 cap — did `out.cells = snapshot.cells`, copying
the sparse list straight through. On the real OLMoE that produced a frame
declaring **16x64** and carrying **878** entries: every count on the wrong
expert, and a length-checking consumer rejecting the frame outright. The G7 brain
grid would have shown "no heat frame yet" against every real model, forever.

Two branches of one function disagreeing about their own output format, with
only one of them tested — because `make_snapshot()` in `soma_telemetry_g6` builds
every cell. A dense fixture cannot exercise a densify step. The regression now
feeds a deliberately sparse snapshot through BOTH branches and checks that counts
land at their own coordinates and that totals are conserved.

Fixed alongside it: a bucket that received no sources kept its `Vram` seed and
reported the WARMEST tier for the cells we know least about — in a CPU-only v1
where the VRAM tier is always empty. Absent now reads as Disk, which is what
`heat()` skipping a cell actually means.

**D11 — an observer that waits on what it observes stops being an observer.**

The sampler called `MemoryHierarchy::occupancy()`, `stats()`, `heat()` and
`Scheduler::stats()`. All four take a mutex; the step loop holds the scheduler's
across an entire forward and the hierarchy's across expert reads and evictions,
which is most of what a streamed model does. So the telemetry thread sampled at
whatever rate the engine happened to be idle:

```
idle    17.3 frames/s
during   1.3 frames/s     <- 13x collapse
after   18.7 frames/s
```

The feed went quiet at exactly the moment worth watching. Nothing was WRONG — no
number was wrong, no frame was corrupt — the feed simply was not there.

**The fix is a rule, not a patch.** Each accessor gained a `try_` form that
returns false instead of waiting, and the telemetry path uses only those. The
blocking forms stay for callers that genuinely need an exact current value; both
share one `_locked` body so the two cannot drift.

Counters and grids are then handled DIFFERENTLY, because they are read
differently:

- **Counters** carry over from the previous tick when contended, and the frame
  says `stale: true`. A slightly old number labelled old is worth more than a
  frame that never arrives.
- **The heat grid** is skipped entirely when contended. A grid is read
  spatially — republishing the previous one at tick rate would animate a still
  image, and there is no honest way to shade a cell "this is last tick's".

Measured after, on the same live engine:

```
idle    18.7 frames/s
during  16.1 frames/s     <- 86% of idle, 38 of 65 frames marked stale
after   18.7 frames/s
```

and chat latency unmoved: -2.20% median across 6 pairs against 0.92% noise. The
residual 14% is the skipped heat frames, which is the deliberate half of the fix.

The regression test is STRUCTURAL, not timed — a test that races a lock is a
flaky test. It asserts the `try_` forms exist, succeed uncontended, and report
the same numbers as the blocking forms, which is what stops a second source of
truth appearing.

**D10 — every proxied SSE stream closed before its first frame.**

`cli.set_read_timeout(0, 0)` reads as ZERO seconds in cpp-httplib, not as "no limit". On an SSE proxy
that presents as a clean, empty 200: headers go out, the inner `Get` times out immediately, `!ok` fires
`sink.done()`, and the client sees a stream that ends without a frame. No error anywhere.

Three sites, all with comments stating the opposite intent. The best of them read *"No read timeout: the
feed is long-lived by design... A timeout here would look like the engine going quiet"* — directly above
the line that made it go quiet instantly.

Fixed with a long FINITE timeout rather than an attempt at infinity: a wedged engine should eventually
release the socket. An hour is far beyond any tick interval and far short of forever.

`engine_telemetry_republication` passes and always did — it runs against a stand-in node whose handler
answers directly, so the proxy path it was written to cover was never executed by it.

**D8 — the node validates llama.cpp's preconditions for every engine.**

Found immediately after fixing D7, by the same request. The node now receives the right path and still
refuses it:

```
model file not found on this node: Z:/AI/soma-data/containers\OLMoE-1B-7B-0924-q4_g-q6_g-g128
```

The directory is there. `POST /api/node/load-model` reads `backend`, checks it against the engine
registry (line 485), and dispatches on it (line 568) — but between those two points it runs the
llama.cpp runtime-ready check and `fs::is_regular_file(model_path)` for **every** request. The comment
above that block says "llama.cpp always loads a local GGUF file", which is true and is not a statement
about Soma. A Soma container is a directory; `is_regular_file` is false; the load is refused before the
dispatch that would have sent it to the right engine.

This is the same shape as the SlotManager work at G5: a handler written when there was one engine, with
its assumptions still standing after a second arrived. The registry lookup at 485 was already converted
from a string literal for exactly this reason — the conversion stopped one check too early.

**Fixed** by `EngineDescriptor::validate_model_ref`. llama.cpp answers "a node-local GGUF file"; Soma
answers "a directory containing `container_meta.json`" — a stricter check than mere existence, and worth
making: `soma serve` refuses an unconverted checkpoint, and refusing it here costs a stat instead of a
process spawn and a 30-second startup timeout. The handler asks the descriptor and no longer knows what
either answer is.

**The same defect was in the RESTORE handler**, which is how a suspended slot comes back — so a
suspended Soma slot could never have been restored either. Found by grepping for the check rather than
by hitting it, since nothing in this run suspended anything. Both handlers now go through the
descriptor, and `is_regular_file(model_path)` appears nowhere in the file.

The mmproj rules stay gated on `backend` rather than becoming descriptor fields: vision is llama.cpp's
alone — Soma answers 422 for image parts by contract — and a field only one engine would ever set is
worse than an `if` that says why.

**D7 — the registry knows where the model is and placement never asks.**

`AgentScheduler::resolve_backend_for` resolves the agent's `model_path` against the registry and takes
exactly two fields off the record — `arch_hash` and `verdict`. `prepared.model_path` is then set to
`cfg.model_path`, the agent's own string, and that is what the node receives. The node looks it up by
name under `models_dir` and does not find it:

```
model file not found on this node: OLMoE-1B-7B-0924
container on disk:                 OLMoE-1B-7B-0924-q4_g-q6_g-g128
```

**This was working by coincidence and today's requantization fix removed the coincidence.** Containers
used to be written to `containers/<name>`, which happened to equal the agent's `model_path`, so the
node's own name lookup found them. Putting the quantization in the directory name — necessary, because
two quantizations of one model were overwriting each other — broke the accident. The fragility was
always there: the registry has been the authority on where a model's bytes live since control.db
existed, and placement has never asked it.

**Fixed** by `AgentScheduler::model_location()`, which asks the registry where the bytes are and falls
back to the agent's own path when nothing admitted the model — the fallback's GGUF path, which must keep
working. `prepare_model_for_node` now takes the location as an explicit parameter rather than deriving it
from `cfg`, so the identity/location distinction is visible at every call site.

Proven live: the same chat request that failed with `model file not found: OLMoE-1B-7B-0924` now sends
`Z:/AI/soma-data/containers\OLMoE-1B-7B-0924-q4_g-q6_g-g128`. It still fails, for a different reason —
D8.

**The regression test needed its fixture changed to be able to fail.** `model_registry_makes_soma_routable`
set `model_dir = "/containers/" + name`, which is exactly the coincidence that hid the defect: with those
two strings equal, every assertion passes whichever one the scheduler uses. The fixture now carries the
quantization suffix, and reverting the fix turns it red.

**The question this run answered:** `POST /v1/agents` reporting `inference_backend: "llama-cpp"` was a
red herring. That field is the agent's API-vs-node-local class, not the engine choice;
`resolve_backend_for` picks the engine at schedule time and the load request carries it. The node
received `backend: soma` and refused it anyway — see D8.

**D6 — resolved, and the check written for it found a second one.**

`GET /v1/models/{id}/conformance` was described in `docs/external-api.md` from the day the API surface
was written and never registered. It answered 404 for its entire life. Found by reading the document and
calling what it said.

Registered rather than struck from the docs — a client written against the document should work, and a
caller who wants only the ladder should not have to fetch the whole record to read it. Both routes now
share one `conformance_json()` builder, because two would drift and the thing they would drift about is
which field a client should read.

**The structural half matters more.** `require_complete_coverage()` walks REGISTERED handlers and
asserts each has a scope. Nothing walked from the documentation to the router, so a documented route
that was never built raised nothing, forever. `tools/ci/check_api_docs.py` closes it:

```
external-api.md  ->  route_scope.cpp's table   (check_api_docs.py, at build)
route_scope.cpp  ->  registered handlers       (require_complete_coverage, at startup)
```

Together those are bidirectional; neither alone is.

**On its first run it found `POST /v1/engines/{id}/slots/{n}/suspend`** — also documented, also
unregistered. That one is different: its own section says "there is no `/v1/*` route for either.
Promoted because P1 says a capability the system has is a capability the API exposes." It is a DESIGN
entry, not a broken promise. But nothing in the heading let a reader tell a shipped route from a
proposed one without calling it, which is the same ambiguity in a milder form. Headings now carry an
explicit `(planned)` marker, the check reports those separately rather than failing them, and the
document opens by saying what the marker means. Anything unmarked must exist.

**D5 — resolved, and it was four formatters rather than two.**

`fetch.py` divided by `1e9` and wrote "GB". Two separate C++ functions — both named `bytes_label`, one in
`model_registry.cpp` and one in `node_ui.cpp` — divided by 1024³ and also wrote "GB". A fourth,
`tui::mb_str`, divided megabytes by 1024 and suffixed "G". The same OLMoE transfer therefore announced
**13.84 GB** and then counted up to **12.89 GB**: one number, two units, one label, and an operator
watching a 14 GB download see a gigabyte disappear.

The same shape as D1's retry budget — one concept written several times, drifting because nothing forced
them to agree. Now one `util::bytes_label`, binary because every C++ site already was and because
Windows reports file sizes the same way, with labels that say what the arithmetic does: `GiB`, `MiB`,
`KiB`. `mb_str` keeps its narrow-column format and gains honest `Gi`/`Mi` suffixes. `fetch.py` matches.

Verified on a live transfer: `6 files, 0.02 GiB` then `16.1 MiB / 16.1 MiB`. Both ends of one download,
one unit.

**D4 — the admission pipeline reads only the per-expert layout.**

Found by pointing the G8 gate at `hf-internal-testing/tiny-random-OlmoeForCausalLM` as a fast rehearsal
before the 13.84 GB run. Fetch succeeded; convert refused:

```
REFUSED  missing model.layers.0.mlp.experts.0.gate_proj.weight
```

The checkpoint has no such tensor and never would. It carries the **fused** layout that recent
`transformers` produces — one 3-D tensor per role across all experts:

```
model.layers.0.mlp.experts.gate_up_proj     [n_experts, d_model, 2 * intermediate]
model.layers.0.mlp.experts.down_proj        [n_experts, intermediate, d_model]
```

rather than the per-expert `experts.{i}.gate_proj.weight` the converter expects.

**The real `allenai/OLMoE-1B-7B-0924` is not affected** — checked from its
`model.safetensors.index.json` alone, 0.3 MB rather than 13.84 GB: 3219 tensors, per-expert names, which
is what a 2024 upload would have. The tiny-random repos are regenerated against current `transformers`,
so the probe was *newer* than the model it stands in for. That is worth remembering about
`hf-internal-testing/*` fixtures generally: they track the library, not the era of the model they mimic.

**The error is fixed.** `fused_expert_diagnosis()` looks for the fused names before reporting a missing
one, and the pipeline now says:

```
REFUSED  model.layers.0.mlp.experts.<i>.* not found, but this checkpoint carries the
         FUSED expert layout (gate_up_proj(4, 256, 64), down_proj(4, 64, 128)) - every
         expert stacked into one 3-D tensor. This converter reads the per-expert layout only.
         ...
         Re-export in the per-expert layout, or add support here together with an oracle
         that can verify it.
```

**Reading the layout is deliberately NOT implemented, and the reason is the interesting part.**

Investigating it to write the fix turned up three things, in order of how much each changed the plan:

1. **The shapes are not what a reasonable person would assume.** I expected `[experts, in, out]` — the
   ordering a fused matmul wants. The OLMoE checkpoint is `gate_up_proj[E, 2*inter, in]`, out-major,
   the same convention as `nn.Linear.weight`. Checking rather than assuming caught that.
2. **There is more than one fused convention, and they disagree on both axes.** `gpt_oss` in the same
   `transformers` version declares `nn.Parameter(num_experts, hidden_size, 2 * expert_dim)` — in-major,
   the opposite of this checkpoint — and splits gate from up **interleaved**, `gate_up[..., ::2]` and
   `[..., 1::2]`, not as blocked halves. Two conventions, differing in axis order AND in split. A
   converter that picks one and applies it to the other produces a container that loads, runs, and is
   wrong.
3. **`transformers` 4.57.6 cannot read this checkpoint either — and does not say so loudly.** Loading
   `hf-internal-testing/tiny-random-OlmoeForCausalLM` with the pinned version leaves the fused tensors
   unused and **randomly initialises** every per-expert weight, announcing it in a warning that reads
   like boilerplate. So the reference implementation this project defines correctness against would, on
   this class of checkpoint, be comparing against noise.

That is what settles it. The conformance ladder exists to stop "plausible and wrong" from shipping, and
implementing this layout would mean writing a path that cannot be verified against any oracle available
here, for a convention that is family-specific and unsettled. The error message costs a reader nothing
and tells them the truth; a guess would cost them a silently wrong model.

**What would unblock it:** a checkpoint in this layout whose reference output can be produced — either a
`transformers` new enough to load it, or a per-expert export of the same weights to diff against. Then
the implementation and its verification land together, which is the only order that is worth anything.

**D2 — resolved. The channel was measuring membership when the question is traffic.**

`GridCell::tier` held the coldest tier PRESENT in a bucket and ignored counts entirely, so a bucket of
six experts with one disk-resident expert that never fires reported Disk — identically to a bucket whose
disk-resident expert is the hot one. On a streamed model, where most experts are on disk by
construction, the whole grid went one colour the moment any reduction happened and got *more* uniform
the more it reduced.

**An expert on disk that never fires costs nothing; one that fires constantly costs everything.** The
single `tier` field is replaced by `tier_count[3]` — routing traffic attributed to each tier — and the
colour channel carries `cold_fraction()`, the share of a cell's traffic served from disk. At full
resolution, one expert per cell, that is 0 or 1 and the rendering degenerates exactly to the old
per-expert tier, which is what makes this a generalisation rather than a different metric.

Colour is now banded by disk share — none / some / most / all — rather than by tier name, because the
band is what an operator acts on. Brightness still carries volume, so hot-and-streamed remains visually
distinct from cold-and-streamed; that separation was the one thing the old rule got right.

Measured on a synthetic streamed model (1 expert in 8 cached, carrying most of the traffic) reduced
16×16 → 4×4: `cold_fraction` spans **0.03 to 1.00**. Under coldest-wins every one of those buckets
contained a disk expert, so every one reported Disk and the span was exactly **0**. Restoring the old
rule turns four of the six checks red — and leaves "a disk expert carrying the traffic DOES colour its
bucket" green, which is precisely why the original looked correct: coldest-wins is right for that one
case and wrong for every other.

**D3 — resolved, and it was hiding a wrong claim rather than only a stale one.**

The headings were the visible half: "the backend is next" when it is done, "running, not yet correct"
directly above its own paragraph reading "PASSES the G0 gate", "the residual is not yet explained" for a
residual explained four subsections later. Retitled, with the gate's actual numbers moved to the top and
a banner marking the eliminated-hypotheses log as historical. The log itself is kept — the section
already argued for keeping it, and the contrast is the finding: four rounds of plausible reasoning
bought nothing, one instrument bought the answer.

The half worth recording: writing an accurate status line meant asserting **"zero core diffs"**, the
strictest line of the G4 gate — and it is **false**. `F32Backend::route` in `include/soma/f32_model.hpp`
gained a `const F32LayerWeights&` parameter for DeepSeek-V3's `noaux_tc` scoring. That is a core header,
changed for the second architecture, which is exactly what the gate was written to detect.

**The gate worked; the summary rounded its result down.** A prior pass had noticed ("strictly, that is
already not met") and listed two smaller diffs, but not the router signature — the one that matters,
because it is an interface that assumed a router's only input is its logits. The status section now
states one core diff and names it. A gate whose result is edited toward its target measures nothing, and
the next architecture through the seam will want to know which interface moved.

---

## What is deferred, and where it would land

| Deferred | Earliest sensible gate |
|---|---|
| GPU tier residency + GPU kernels | Post-G4. The tier is declared and reported throughout, so this is an implementation, not a migration. |
| ~~**MLA/DSA decode against a cached latent**~~ | **DONE.** `f32_attention_kv` decodes against the compressed cache for both Mla and MlaDsa. The cache width now comes from the backend (D36); the K plane holds the normalised latent plus the shared RoPE segment, and the V plane — which MLA leaves empty, since its V is derived rather than stored — holds DSA's indexer key (D37). Under DSA only the SELECTED latents are expanded through `kv_b`, so the sparse path pays for itself here rather than merely being honoured: at `index_topk` 2048 against a 32k context that is a sixteenth of the expansion, and the saving grows with context. Verified by `soma_decode_kv_g4`, which requires the cached path to agree with the teacher-forced one at EVERY position across all six fixtures (max 4.77e-07), and which was itself checked by breaking the code two ways. **Absorption landed separately and is now the default** — see the row below. One follow-up remains: plain MLA allocates a V plane it never uses, 2.94 GB of waste on GLM-5.2 at 4k context. |
| ~~**A compiled tokenizer for GLM-5.2**~~ | **DONE.** `compile_tokenizer.py` refused its Split pattern, so every token the engine produced from GLM-5.2 came from serve's byte fallback: the forward pass was real and the TEXT was meaningless. The pattern turned out to be Qwen3's with one alternative changed — `\p{N}{1,3}` for `\p{N}`, digits grouped in runs of up to three, the GPT-4/cl100k convention. **No engine change was needed**: the compiled program already carries `(kind, payload, min, max)` per item and the C++ matcher already honours `max_count` generically, so a bounded repeat was representable all along — the compiler simply had no pattern that used one. `program_qwen` is now parameterized on the digit run rather than copied, since a second near-identical program is how the two drift. Round-trip against HF `tokenizers`: **36/36 encode, decode and stream**. Verified load-bearing by compiling GLM as though it were Qwen — encode falls to 34/36 and the failing string is exactly `"3.14159 and 42 and 0x1F and 1,000,000"`, 28 ids where 20 are wanted. Committed as a fixture (6.6 MB, alongside Qwen3's 3.7 MB) so `soma_tokenizer_g0` covers it permanently. **The result**: GLM-5.2 served from its 499 GiB container at `--quant-dense q4_g` on a 24 GiB budget, prompted "The capital of France is", answers " Paris." |
| **The indexer's own O(context) scoring** | After absorption. DSA scores EVERY cached key to pick its top-k, so that pass is linear in context while the attention it feeds is now capped at `index_topk`. On GLM-5.2 at 32k the two are within 10% of each other (1.34e8 vs 1.49e8 MACs per layer per step), so the indexer is no longer the cheap part. Measured on the fixture: GLM-5.2's absorbed/expanded ratio stays flat at ~1.3x from 128 to 512 positions while DeepSeek-V2-Lite's grows 2.0x -> 4.5x, because DSA caps the j loop and V2-Lite does not — the growth GLM does show is the indexer, shared by both paths. |
| **The production `ArchBackend` execution path** | Not started, and recorded here because it was recorded NOWHERE. `arch_backend.hpp` and `attention_backend.hpp` declare `prefill`, `decode`, `route`, `apply_expert`, `dense_ffn`, `shared_experts`, `apply_norm`, `init_kv_region` and `apply_rope` for both families, each with a doc comment describing behaviour, and **neither family defines any of them** — verified: zero definitions and zero qualified references across `src/`. What actually serves is the fp32 `F32Backend` path. Harmless today, since nothing calls them and the backend structs wire only description members; the reason to write it down is that this exact pattern has now cost twice — D16, where a declared-never-defined `attention_backend()` left the planner sizing MLA with GQA's formula, and D39. A declaration with a confident comment and no definition reads as implemented. |
| Paged KV with a block table | Post-G4. `AttentionBackend` already takes a batch, so it lands without an interface change. |
| Speculation under batching | Post-G4 |
| ~~**DSA + IndexShare attention**~~ (GLM-5.2) | **DONE** — implemented and token-exact against the oracle (`max=1.25e-06`, greedy exact). The concern recorded here, that `AttentionBackend` is per-layer function pointers with no channel for cross-layer state, turned out to be the wrong shape of problem: the index is per-(row, step) and never persists between steps, so it needed a per-FORWARD slot, not a per-sequence one. One `ArchLayerPayload` on `F32Workspace` — the same opaque idiom the per-layer weights already use — and the seam was untouched. See the DSA scoping section. |
| Multimodal | Not planned. 422 is the contract. |
| Distributed / multi-node single model | Not planned for v1 |
