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
    ├─ convert.py            tensor names → canonical roles → our layout      [G2]
    │                        one expert = one contiguous range (gate/up/down
    │                        interleaved), 4 KB-aligned, sidecar index,
    │                        fused 3D expert tensors PRE-TRANSPOSED
    │
    ├─ compile_tokenizer.py  tokenizer.json → merge table + byte-class NFA     [G0]
    │                        + special tokens + token-level chat template
    │                        GATED on a byte-exact round-trip vs HF tokenizers
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
