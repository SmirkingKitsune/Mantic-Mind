"""Deterministic CPU substitutes for DeepSeek V4's TileLang kernels.

This module deliberately implements the *interfaces* imported by the pinned
``inference/model.py``.  It never imports TileLang and it never relies on native
FP8/FP4 execution.  Low-precision values are represented as ordinary FP32
tensors plus explicit scales, so the reference runs on any PyTorch CPU build.

The implementation is intentionally literal rather than fast.  It is an oracle
for a five-token DSpark block, not a serving kernel.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

import torch


Trace = Callable[[str, torch.Tensor], None]
_trace: Optional[Trace] = None


def set_trace(trace: Optional[Trace]) -> None:
    global _trace
    _trace = trace


def _emit(point: str, value: torch.Tensor) -> None:
    if _trace is not None:
        _trace(point, value.detach().float().cpu().contiguous())


def _pow2_scale(amax: torch.Tensor, maximum: float, minimum: float) -> torch.Tensor:
    amax = torch.clamp(amax.float(), min=minimum)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / maximum)))


def _fp8_e4m3fn_round(x: torch.Tensor) -> torch.Tensor:
    """Round to finite E4M3 values without using a native float8 dtype."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=448.0, neginf=-448.0)
    sign = torch.sign(x)
    a = x.abs().clamp(max=448.0)
    sub = torch.round(a * 512.0) / 512.0
    safe = torch.clamp(a, min=2.0**-9)
    exponent = torch.floor(torch.log2(safe))
    step = torch.pow(2.0, exponent - 3.0)
    normal = torch.round(a / step) * step
    rounded = torch.where(a < 2.0**-6, sub, normal).clamp(max=448.0)
    return torch.copysign(rounded, x)


_FP4_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp4_e2m1_round(x: torch.Tensor) -> torch.Tensor:
    levels = _FP4_LEVELS.to(x.device)
    distance = (x.float().unsqueeze(-1) - levels).abs()
    nearest = distance.amin(dim=-1, keepdim=True)
    tied = distance == nearest
    encoding = torch.arange(levels.numel(), device=x.device)
    even_tied = tied & ((encoding & 1) == 0)
    # IEEE round-to-nearest-even: if a midpoint has an even encoded endpoint,
    # prefer it; otherwise the ordinary unique minimum wins.
    preferred = torch.where(even_tied, distance, torch.full_like(distance, float("inf")))
    have_even = even_tied.any(dim=-1)
    index = torch.where(have_even, preferred.argmin(dim=-1), distance.argmin(dim=-1))
    return levels[index]


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block FP8 quantization, including the official in-place Q/DQ form."""
    del scale_dtype  # E8M0 is represented exactly by an FP32 power of two here.
    z = x.float().contiguous()
    n = z.shape[-1]
    flat = z.reshape(-1, n)
    q = torch.empty_like(flat)
    scales = torch.empty(flat.shape[0], (n + block_size - 1) // block_size)
    for group, begin in enumerate(range(0, n, block_size)):
        end = min(begin + block_size, n)
        amax = flat[:, begin:end].abs().amax(dim=1)
        if scale_fmt is not None:
            scale = _pow2_scale(amax, 448.0, 1.0e-4)
        else:
            scale = torch.clamp(amax, min=1.0e-4) / 448.0
        q[:, begin:end] = _fp8_e4m3fn_round(flat[:, begin:end] / scale[:, None])
        scales[:, group] = scale
    q = q.reshape(z.shape)
    scales = scales.reshape(*z.shape[:-1], scales.shape[-1])
    if inplace:
        dequant = dequant_fp8_activation(q, scales, block_size).to(x.dtype)
        x.copy_(dequant)
        _emit("fp8_qdq", x)
        return x
    return q, scales


def dequant_fp8_activation(q: torch.Tensor, scales: torch.Tensor, block_size: int = 128):
    out = q.float().clone()
    n = out.shape[-1]
    for group, begin in enumerate(range(0, n, block_size)):
        end = min(begin + block_size, n)
        out[..., begin:end] *= scales[..., group, None].float()
    return out


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Block E2M1 quantization with E8M0 (power-of-two) scales."""
    z = x.float().contiguous()
    n = z.shape[-1]
    flat = z.reshape(-1, n)
    q = torch.empty_like(flat)
    scales = torch.empty(flat.shape[0], (n + block_size - 1) // block_size)
    for group, begin in enumerate(range(0, n, block_size)):
        end = min(begin + block_size, n)
        amax = flat[:, begin:end].abs().amax(dim=1)
        scale = _pow2_scale(amax, 6.0, 6.0 * 2.0**-126)
        q[:, begin:end] = _fp4_e2m1_round(flat[:, begin:end] / scale[:, None])
        scales[:, group] = scale
    q = q.reshape(z.shape)
    scales = scales.reshape(*z.shape[:-1], scales.shape[-1])
    if inplace:
        out = q.float()
        for group, begin in enumerate(range(0, n, block_size)):
            end = min(begin + block_size, n)
            out[..., begin:end] *= scales[..., group, None]
        x.copy_(out.to(x.dtype))
        _emit("fp4_qdq", x)
        return x
    return q, scales


def quantize_fp8_weight(weight: torch.Tensor, block_size: int = 128):
    """Return logical E4M3 values and the official 128x128 block scales."""
    w = weight.detach().float().contiguous()
    rows, cols = w.shape
    q = torch.empty_like(w)
    scales = torch.empty((rows + block_size - 1) // block_size,
                         (cols + block_size - 1) // block_size)
    for rb, r0 in enumerate(range(0, rows, block_size)):
        r1 = min(r0 + block_size, rows)
        for cb, c0 in enumerate(range(0, cols, block_size)):
            c1 = min(c0 + block_size, cols)
            amax = w[r0:r1, c0:c1].abs().amax()
            scale = _pow2_scale(amax.reshape(1), 448.0, 1.0e-4)[0]
            q[r0:r1, c0:c1] = _fp8_e4m3fn_round(w[r0:r1, c0:c1] / scale)
            scales[rb, cb] = scale
    return q, scales


def dequant_fp8_weight(q: torch.Tensor, scales: torch.Tensor, block_size: int = 128):
    out = q.float().clone()
    for rb, r0 in enumerate(range(0, out.shape[0], block_size)):
        r1 = min(r0 + block_size, out.shape[0])
        for cb, c0 in enumerate(range(0, out.shape[1], block_size)):
            c1 = min(c0 + block_size, out.shape[1])
            out[r0:r1, c0:c1] *= scales[rb, cb]
    return out


def quantize_fp4_weight(weight: torch.Tensor, block_size: int = 32):
    w = weight.detach().float().contiguous()
    q = torch.empty_like(w)
    scales = torch.empty(w.shape[0], (w.shape[1] + block_size - 1) // block_size)
    for group, begin in enumerate(range(0, w.shape[1], block_size)):
        end = min(begin + block_size, w.shape[1])
        amax = w[:, begin:end].abs().amax(dim=1)
        scale = _pow2_scale(amax, 6.0, 6.0 * 2.0**-126)
        q[:, begin:end] = _fp4_e2m1_round(w[:, begin:end] / scale[:, None])
        scales[:, group] = scale
    return q, scales


def dequant_fp4_weight(q: torch.Tensor, scales: torch.Tensor, block_size: int = 32):
    out = q.float().clone()
    for group, begin in enumerate(range(0, out.shape[1], block_size)):
        end = min(begin + block_size, out.shape[1])
        out[:, begin:end] *= scales[:, group, None]
    return out


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    del scale_dtype
    ad = dequant_fp8_activation(a, a_s, 128)
    bd = dequant_fp8_weight(b, b_s, 128)
    out = torch.matmul(ad.float(), bd.float().transpose(-1, -2))
    _emit("fp8_gemm", out)
    return out.to(torch.bfloat16)


def fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    del scale_dtype
    ad = dequant_fp8_activation(a, a_s, 128)
    bd = dequant_fp4_weight(b, b_s, 32)
    out = torch.matmul(ad.float(), bd.float().transpose(-1, -2))
    _emit("fp4_gemm", out)
    return out.to(torch.bfloat16)


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Literal gather/softmax implementation of DeepSeek's sparse kernel."""
    bsz, seqlen, heads, dim = q.shape
    out = torch.zeros_like(q)
    for b in range(bsz):
        for t in range(seqlen):
            idx = topk_idxs[b, t].long()
            valid = idx >= 0
            keys = kv[b, idx[valid]].float()
            for h in range(heads):
                if keys.numel() == 0:
                    continue
                score = torch.mv(keys, q[b, t, h].float()) * softmax_scale
                maximum = torch.maximum(score.max(), attn_sink[h].float())
                exp_score = torch.exp(score - maximum)
                denom = exp_score.sum() + torch.exp(attn_sink[h].float() - maximum)
                # The TileLang kernel casts the probabilities to BF16 before AV.
                prob = (exp_score / denom).to(torch.bfloat16).float()
                out[b, t, h] = torch.mv(keys.transpose(0, 1), prob).to(out.dtype)
    _emit("sparse_q", q)
    _emit("sparse_kv", kv)
    _emit("sparse_topk", topk_idxs.float())
    _emit("sparse_out", out)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor,
                      hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1.0e-6):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult:].reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].reshape(hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    _emit("hc_pre", pre)
    _emit("hc_post", post)
    _emit("hc_comb", comb)
    return pre, post, comb


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform used by source FP4 Q/DQ paths."""
    n = x.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    y = x.float().clone()
    width = 1
    while width < n:
        y = y.reshape(*y.shape[:-1], -1, 2, width)
        left = y[..., 0, :].clone()
        right = y[..., 1, :].clone()
        y[..., 0, :] = left + right
        y[..., 1, :] = left - right
        y = y.reshape(*x.shape[:-1], n)
        width *= 2
    return (y * (n ** -0.5)).to(x.dtype)


def self_test() -> None:
    """Small deterministic checks which do not compare an implementation to itself."""
    x = torch.tensor([-7.0, -5.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25,
                       0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0])
    got = _fp4_e2m1_round(x)
    want = torch.tensor([-6.0, -4.0, -4.0, -2.0, -2.0, -1.0, -1.0, -0.0,
                          0.0,  1.0,  1.0,  2.0,  2.0,  4.0,  4.0,  6.0])
    if not torch.equal(got, want):
        raise AssertionError(f"E2M1 table mismatch: {got.tolist()}")
    fp8_in = torch.tensor([0.0, 2.0**-9, 1.0625, 1.1875, 448.0, 500.0])
    fp8_want = torch.tensor([0.0, 2.0**-9, 1.0, 1.25, 448.0, 448.0])
    fp8_got = _fp8_e4m3fn_round(fp8_in)
    if not torch.equal(fp8_got, fp8_want):
        raise AssertionError(f"E4M3 table mismatch: {fp8_got.tolist()}")
    h = fwht(torch.eye(8))
    if not torch.allclose(h @ h.T, torch.eye(8), atol=1.0e-6, rtol=0.0):
        raise AssertionError("normalized CPU Hadamard is not orthogonal")
