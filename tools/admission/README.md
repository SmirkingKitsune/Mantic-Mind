# Admission tooling — offline only

Converts, validates, profiles, and issues a **verdict** for a model. Output lands in the converted model
directory and in `control.db`; clients read it through `/v1/models/*`.

> **These scripts are never a runtime dependency.** The engine does not spawn Python, link it, or
> require it to be installed. A serving host needs none of this. That is what makes "admission is
> offline" a structural fact rather than a stated intention.

**Status:** design pass. This directory documents the pipeline; the scripts land at the gates noted
below. See [../../docs/roadmap.md](../../docs/roadmap.md).

---

## Pipeline

```
HF repo or local weights
    │
    ├─ fetch.py              repo id → a local directory                      [G6]
    │                        safetensors only unless --allow-pickle; never
    │                        the TF/Flax/GGUF copies of the same weights
    │                        SKIPPED entirely when `source` is already a path
    │
    ├─ convert.py            tensor names → canonical roles → our layout      [G2]
    │                        one expert = one contiguous range (gate/up/down
    │                        interleaved), 4 KB-aligned, sidecar index,
    │                        fused 3D expert tensors PRE-TRANSPOSED
    │
    ├─ compile_tokenizer.py  tokenizer.json → merge table + byte-class NFA     [G0]
    │                        + special tokens + chat_template.py's compiled
    │                        chat template, resolved to TOKEN IDS
    │                        GATED on a byte-exact round-trip vs HF tokenizers,
    │                        and on the template reproducing HF's own renderer
    │                        id-for-id over a battery of conversations
    │
    ├─ make_oracle.py        tiny-random model with the REAL arch config       [G0]
    │                        + transformers teacher-forcing / greedy oracle
    │
    ├─ conformance ladder    stage 1 fp32 tiny, token-exact TF                 [G0]
    │                        stage 2 quantized tiny, token-exact greedy        [G1]
    │                        stage 3 real checkpoint, logit-KL vs fp16         [G2]
    │                        stage 4 accuracy floor                            [G2]
    │
    ├─ profile_streaming.py  bytes/token vs MEASURED disk bandwidth            [G2]
    ├─ profile_lookahead.py  per-layer router-lookahead recall                 [G2]
    ├─ autotune.py           kernel choice over the model's shape set          [G1]
    ├─ bootstrap_heat.py     calibration-corpus routing histogram              [G2]
    │
    └─ verdict               stream | hybrid | resident-only | reject
```

Failing stage 1 or 2 → `reject` → the model routes to the llama.cpp fallback. **That is a successful
admission**, not a failed one: the operator asked whether Soma can run this, and "no, here is why" is an
answer.

Failing stage 3 while 1 and 2 pass is a **quantization finding**, not a correctness bug. Different
remediation — requantize a role, raise group-scale granularity — and conflating the two costs days.

---

## The chat template is measured, not interpreted

The prompt framing is the part of serving that is invisible when it is wrong. A model handed
a flattened `user:` / `assistant:` transcript instead of `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>`
still answers, still streams, still reads fluently — and is not the model that was trained. Nothing
downstream catches it: the weights are right, the tokenizer round-trips, and the logit-KL is computed
against whatever prompt was actually built.

`chat_template.py` therefore does not reimplement Jinja. It **runs** the checkpoint's own template
against probe conversations built from sentinels, reads the scaffolding off the text around each
sentinel, and resolves it to token ids. The engine concatenates; it has no renderer and must not grow
one, because a second renderer has to stay bug-for-bug identical with the first forever.

Two gates, and the second is not a restatement of the first:

- the assembled TEXT must equal what the template rendered, over a battery of conversations chosen for
  the seams — a run of tool results, an assistant turn with and without reasoning, padded content,
  multibyte content, content that looks like scaffolding;
- the assembled IDS must equal the ids of that same string tokenized whole. Assembling from
  precompiled pieces is sound only where BPE cannot merge across a seam, and that is a property of the
  TOKENIZER. Qwen3 fails exactly here, on tool conversations: `<tool_response>` is ordinary text in
  its vocabulary rather than a special token, so the pieces come to 59 ids where the whole comes to
  57. Its template is refused with the case named.

**Recognize or refuse.** A template this shape cannot express produces no compiled template and a
reason in `chat_template.unsupported`; `soma serve` falls back to flattening messages, which is
visibly worse rather than subtly wrong.

**Nothing here is read off template source.** GLM-5.2 and GLM-5.3 implement `clear_thinking` with
opposite defaults — 5.2 drops prior reasoning unless told not to, 5.3 keeps it unless told to — and a
compiler written by reading either would have been confidently wrong about the other. The battery
found that. It also found that GLM-5.2's `enable_thinking: false` removes its entire `Reasoning
Effort` system block, that Qwen3 removes historical `<think>` blocks rather than re-emitting them, and
that GLM-5.3 has no `enable_thinking` switch at all.

---

## What `convert.py` will read, and what it refuses

`f32`, `bf16`, and **blockwise fp8** — `F8_E4M3` weights beside a `<tensor>_scale_inv` carrying one f32
multiplier per `weight_block_size` tile, which is what `quantization_config.quant_method == "fp8"`
declares. Everything else that arrives pre-quantized is refused by name: compressed-tensors, AWQ and
GPTQ pack sub-byte levels in layouts of their own, and quantizing their packed bytes yields a container
that loads, streams and generates noise.

The fp8 exception is on the merits, not for convenience. Blockwise fp8 dequantizes **exactly** — one
multiply per tile, nothing unpacked, no layout inferred — so `quantize_rows` sees the same fp32 matrix
the bf16 upload would have produced, less the rounding the publisher had already applied. Whether a
tensor is dequantized is decided by its **dtype**, not by matching `modules_to_not_convert` by name: an
fp8 upload publishes its norms, routers, embeddings and output head unquantized, the file already says
which is which, and a second copy of that fact is a second thing that can go stale. An fp8 tensor with
no scale beside it **refuses** rather than converting a matrix ~400× too small.

GLM-5.3 is why this exists. It is the same base model as GLM-5.2 — the two `config.json` files differ
only in `transformers_version` — but `zai-org/GLM-5.3` is fp8 at 756 GB while `zai-org/GLM-5.3-BF16` is
a separate 1.5 TB repo, so refusing the primary upload meant supporting GLM-5.3 by fetching it twice and
keeping the larger copy. `container_meta.json` records which one a container came from, in
`source_quantization`.

`tools/ci/check_fp8_source.py` (ctest `mm_fp8_source`) is what holds this: it quantizes a committed
fixture blockwise, writes both the fp8 halves and their exact f32 product as two checkpoints, and
requires the two conversions to be **byte-identical** — no tolerance, because the two inputs hold the
same numbers by construction.

---

## Everything here is a measurement

Four findings are carried forward from the prior art (`JustVugg/colibri`) as **things to re-measure**,
never as constants:

| Finding | Why it cannot be inherited |
|---|---|
| Router-lookahead recall ~72% on one checkpoint | Per-model **and per-layer**. Prefetch is enabled per layer only above threshold; a wrong prefetch evicts something useful, so a poor-recall layer gets none. |
| Expert size | Architecture-specific — 2.4 MB (Qwen3) vs 88 MB (Mixtral) at q4. It sets the per-read cost that concurrency amortizes. |
| Which kernel wins at a shape | Empirical. int4 single-row measured *slower* than fp32 there. |
| Whether streaming pays at all | Depends on granularity. This is what the verdict is for. |

`profile_streaming.py` measures bandwidth with reads **the size of this model's experts**. A 2.4 MB read
and an 88 MB read do not achieve the same bandwidth on the same drive, and using one headline figure is
how a verdict ends up confidently wrong.

---

## The verdict is not a property of the model

It is a property of `(model, quantization, host budget)`:

| | Qwen3-30B-A3B | DeepSeek-V2-Lite | Mixtral-8x7B |
|---|---|---|---|
| Active fraction | 6.3 % | 9.4 % | **25 %** |
| Expert bytes @q4 | 2.36 MB | 4.33 MB | **88.1 MB** |
| Routed set @q4 | 14.5 GB | 7.2 GB | 22.6 GB |
| Verdict @q4, 32 GB | `resident-only` | `resident-only` | `resident-only` |
| Verdict @bf16 | `stream` | `hybrid` | `reject` |

Mixtral is `resident-only` because of its **shape** — 25 % active fraction is disqualifying at any size.
Qwen3 flips purely on quantization and host RAM.

So the registry stores the measurements plus an admission-host verdict and its `verdict_basis`, and
`soma plan --json` re-derives the effective verdict for the actual target node. A stored scalar alone
would mis-route the same model between two nodes with different RAM, silently.

Full derivation: [../../schemas/arch-ir.md §8](../../schemas/arch-ir.md).

---

## Conventions

Follows the existing `tools/qwen_tts_service.py` precedent: `#!/usr/bin/env python3`, module docstring,
`from __future__ import annotations`, `argparse`, full type annotations, leading-underscore private
helpers, invoked by an explicit `python tools/admission/<name>.py`.

**One departure:** unlike every other Python file in this repo, admission needs `torch` and
`transformers` for the oracle. It therefore gets `tools/admission/requirements.txt` — the first in this
repo — and it is never installed on a serving host.

CI does **not** install them. Tiny-random fixtures and golden oracle logits are **committed** under
`tests/fixtures/tiny/`, so the conformance ladder runs per-commit without torch in the image and the
oracle stays reproducible. Regenerating a fixture is an explicit `make_oracle.py` run by a human.
