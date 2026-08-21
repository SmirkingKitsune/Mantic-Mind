"""Build a deterministic tiny DSpark oracle from DeepSeek's pinned model.py.

The target base model is deliberately not executed.  DSpark's public contract is
three exported target-layer hidden streams, so this generator feeds deterministic
streams directly to both implementations.  That makes every difference in the
result attributable to DSpark itself.

Example (from the repository root)::

  uv run --with torch --with numpy --with safetensors \
    python tools/admission/make_dspark_oracle.py \
      --pinned-model Z:/.../DeepSeek-V4-Pro-0813 \
      --out tests/fixtures/tiny/DeepSeek-V4-Pro-0813-DSpark
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

import deepseek_dspark_cpu as cpu


PINNED_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
PINNED_MODEL_SHA256 = "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
PINNED_KERNEL_SHA256 = "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"
SEED = 20260820


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned_model(root: Path):
    model_py = root / "inference" / "model.py"
    kernel_py = root / "inference" / "kernel.py"
    if sha256(model_py) != PINNED_MODEL_SHA256 or sha256(kernel_py) != PINNED_KERNEL_SHA256:
        raise RuntimeError("pinned inference/model.py or inference/kernel.py hash mismatch")

    shim = types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                 "sparse_attn", "hc_split_sinkhorn"):
        setattr(shim, name, getattr(cpu, name))
    old = sys.modules.get("kernel")
    sys.modules["kernel"] = shim
    try:
        spec = importlib.util.spec_from_file_location("deepseek_v4_pinned_oracle", model_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {model_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            del sys.modules["kernel"]
        else:
            sys.modules["kernel"] = old
    module.rotate_activation = cpu.fwht
    return module


def write_records(path: Path, records: list[tuple[int, str, torch.Tensor]]) -> None:
    with path.open("wb") as out:
        out.write(b"SOMAACT1")
        out.write(struct.pack("<I", len(records)))
        for layer, point, value in records:
            data = value.detach().float().cpu().contiguous().reshape(-1).numpy()
            encoded = point.encode("utf-8")
            out.write(struct.pack("<II", layer, len(encoded)))
            out.write(encoded)
            out.write(struct.pack("<I", data.size))
            out.write(data.astype("<f4", copy=False).tobytes())


def initialize_dspark(model, mod, generator: torch.Generator) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("mtp."):
                continue
            if name.endswith("norm.weight"):
                value = 1.0 + 0.01 * torch.randn(parameter.shape, generator=generator)
            elif name.endswith("gate.bias"):
                # Keep scored-routing choices away from accidental random ties
                # while still exercising bias-for-selection/score-for-weight.
                value = torch.linspace(-0.25, 0.25, parameter.numel()).reshape(parameter.shape)
            elif name.endswith("hc_attn_scale") or name.endswith("hc_ffn_scale"):
                value = torch.tensor([0.35, 0.45, 0.25]).reshape(parameter.shape)
            elif name.endswith("hc_head_scale"):
                value = torch.tensor([0.4]).reshape(parameter.shape)
            elif name.endswith("hc_attn_base") or name.endswith("hc_ffn_base"):
                value = torch.linspace(-0.08, 0.08, parameter.numel()).reshape(parameter.shape)
            elif name.endswith("hc_head_base"):
                value = torch.linspace(-0.06, 0.06, parameter.numel()).reshape(parameter.shape)
            elif name.endswith("attn_sink"):
                value = torch.linspace(-0.2, 0.2, parameter.numel()).reshape(parameter.shape)
            else:
                fan_in = parameter.shape[-1] if parameter.ndim >= 2 else parameter.numel()
                value = torch.randn(parameter.shape, generator=generator) * (0.18 / fan_in**0.5)
            parameter.copy_(value.to(parameter.dtype))

        # A strong, explicit Markov chain makes exact proposal identity a stable
        # semantic gate. Random untrained logits otherwise differ at near-ties
        # while every activation remains within BF16 tolerance.
        markov = model.mtp[-1].markov_head
        markov.markov_w1.weight.zero_()
        markov.markov_w2.weight.zero_()
        chain = (37, 101, 202, 303, 404, 505)
        for ordinal, (previous, following) in enumerate(zip(chain, chain[1:])):
            markov.markov_w1.weight[previous, ordinal] = 2.0
            markov.markov_w2.weight[following, ordinal] = 4.0

    # Run the official linear dispatch through explicit CPU FP8/FP4 kernels.
    with torch.no_grad():
        for name, module in model.mtp.named_modules():
            if not isinstance(module, mod.Linear) or module.weight is None:
                continue
            if name.endswith("confidence_head.proj") or name.endswith("attn.wo_a"):
                continue
            if ".ffn.experts." in name:
                q, scale = cpu.quantize_fp4_weight(module.weight)
                module.weight.copy_(cpu.dequant_fp4_weight(q, scale).to(module.weight.dtype))
                module.weight._dspark_oracle_format = "fp4"
            else:
                q, scale = cpu.quantize_fp8_weight(module.weight)
                module.weight.copy_(cpu.dequant_fp8_weight(q, scale).to(module.weight.dtype))
                module.weight._dspark_oracle_format = "fp8"
            module.weight._dspark_oracle_q = q
            module.weight._dspark_oracle_scale = scale

    def oracle_linear(x: torch.Tensor, weight: torch.Tensor, bias=None):
        if bias is not None:
            raise AssertionError("the pinned V4 graph has no biased Linear modules")
        fmt = getattr(weight, "_dspark_oracle_format", "")
        if fmt == "fp8":
            aq, a_scale = cpu.act_quant(x, 128, "ue8m0", torch.float32)
            return cpu.fp8_gemm(aq, a_scale, weight._dspark_oracle_q,
                                weight._dspark_oracle_scale)
        if fmt == "fp4":
            aq, a_scale = cpu.act_quant(x, 128, "ue8m0", torch.float32)
            return cpu.fp4_gemm(aq, a_scale, weight._dspark_oracle_q,
                                weight._dspark_oracle_scale)
        return F.linear(x, weight)

    mod.linear = oracle_linear
    mod.scale_fmt = "ue8m0"
    mod.scale_dtype = torch.float32


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinned-model", type=Path, required=True)
    parser.add_argument("--base-fixture", type=Path,
                        default=Path("tests/fixtures/tiny/DeepSeek-V4-Pro-0813"))
    parser.add_argument("--out", type=Path,
                        default=Path("tests/fixtures/tiny/DeepSeek-V4-Pro-0813-DSpark"))
    args = parser.parse_args()

    cpu.self_test()
    mod = load_pinned_model(args.pinned_model)
    base_cfg = json.loads((args.base_fixture / "config.json").read_text(encoding="utf-8"))
    cfg = dict(base_cfg)
    cfg.update({
        "dspark_block_size": 5,
        "dspark_markov_rank": 32,
        "dspark_noise_token_id": 511,
        "dspark_target_layer_ids": [1, 2, 3],
        "num_nextn_predict_layers": 3,
        "semantic_fp8_quant_dequant": True,
        "semantic_fp4_quant_dequant": True,
        "dspark_oracle": "pinned-model-py-cpu-v1",
    })

    model_args = mod.ModelArgs(
        max_batch_size=1,
        max_seq_len=128,
        temperature=0,
        dtype="bf16",
        scale_fmt="ue8m0",
        expert_dtype=None,
        scale_dtype="fp32",
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_layers=cfg["num_hidden_layers"],
        n_hash_layers=cfg["num_hash_layers"],
        n_mtp_layers=3,
        n_heads=cfg["num_attention_heads"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"],
        route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"],
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        norm_eps=cfg["rms_norm_eps"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        # The pinned constructor indexes this tuple with absolute layer ids,
        # including the three appended MTP blocks. DSpark itself is uncompressed.
        compress_ratios=tuple(cfg["compress_ratios"]) + (0, 0, 0),
        compress_rope_theta=cfg["compress_rope_theta"],
        original_seq_len=cfg["rope_scaling"]["original_max_position_embeddings"],
        rope_theta=cfg["rope_theta"],
        rope_factor=cfg["rope_scaling"]["factor"],
        beta_fast=cfg["rope_scaling"]["beta_fast"],
        beta_slow=cfg["rope_scaling"]["beta_slow"],
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        hc_mult=cfg["hc_mult"],
        hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        hc_eps=cfg["hc_eps"],
        dspark_block_size=cfg["dspark_block_size"],
        dspark_noise_token_id=cfg["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(cfg["dspark_target_layer_ids"]),
        dspark_markov_rank=cfg["dspark_markov_rank"],
    )

    old_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mod.Transformer(model_args).eval()
    finally:
        torch.set_default_dtype(old_default)

    with safe_open(str(args.base_fixture / "model.safetensors"), framework="pt",
                   device="cpu") as source:
        embed = source.get_tensor("model.embed_tokens.weight")
        head = source.get_tensor("lm_head.weight")
    with torch.no_grad():
        model.embed.weight.copy_(embed.to(model.embed.weight.dtype))
        model.head.weight.copy_(head.to(model.head.weight.dtype))

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    initialize_dspark(model, mod, generator)

    records: list[tuple[int, str, torch.Tensor]] = []
    counters: dict[str, int] = {}

    def record(layer: int, point: str, value: torch.Tensor) -> None:
        # Hooks often expose a tensor the pinned graph mutates in-place later
        # (notably base logits before Markov add_). Golden records must snapshot.
        records.append((layer, point, value.detach().float().cpu().contiguous().clone()))

    def trace(point: str, value: torch.Tensor) -> None:
        if point.startswith("sparse_"):
            call = counters.get("sparse", 0)
            record(call // 4, f"cpu_{point}", value)
            counters["sparse"] = call + 1
        elif point.startswith("hc_"):
            call = counters.get("hc", 0)
            record(call // 6, f"cpu_{point}_{(call // 3) % 2}", value)
            counters["hc"] = call + 1

    cpu.set_trace(trace)

    def capture_output(layer: int, point: str):
        def hook(_module, _inputs, output):
            record(layer, point, output)
        return hook

    def capture_gate(layer: int):
        def hook(_module, _inputs, output):
            record(layer, "router_weights", output[0])
            record(layer, "router_ids", output[1].float())
        return hook

    def capture_head_norm(_module, inputs, output):
        record(2, "head_hidden", inputs[0])
        record(2, "head_norm", output)

    def capture_markov(_module, _inputs, output):
        ordinal = counters.setdefault("markov", 0)
        record(2, f"markov_bias_{ordinal}", output[0])
        record(2, f"markov_embed_{ordinal}", output[1])
        counters["markov"] = ordinal + 1

    hooks = []
    for stage, layer in enumerate(model.mtp):
        hooks.append(layer.attn_norm.register_forward_hook(capture_output(stage, "attn_norm")))
        hooks.append(layer.attn.wq_a.register_forward_hook(capture_output(stage, "q_a")))
        hooks.append(layer.attn.q_norm.register_forward_hook(capture_output(stage, "q_norm")))
        hooks.append(layer.attn.wq_b.register_forward_hook(capture_output(stage, "q_b")))
        hooks.append(layer.attn.register_forward_hook(capture_output(stage, "attn_branch")))
        hooks.append(layer.ffn_norm.register_forward_hook(capture_output(stage, "ffn_norm")))
        hooks.append(layer.ffn.gate.register_forward_hook(capture_gate(stage)))
        hooks.append(layer.ffn.register_forward_hook(capture_output(stage, "ffn_branch")))
        hooks.append(layer.register_forward_hook(capture_output(stage, "stage_streams")))
    hooks.append(model.mtp[0].main_proj.register_forward_hook(capture_output(0, "main_proj")))
    hooks.append(model.mtp[0].main_norm.register_forward_hook(capture_output(0, "main_norm")))
    hooks.append(model.mtp[-1].norm.register_forward_hook(capture_head_norm))
    hooks.append(model.head.register_forward_hook(capture_output(2, "base_logits")))
    hooks.append(model.mtp[-1].markov_head.register_forward_hook(capture_markov))

    prompt_len = 12
    n_targets = len(cfg["dspark_target_layer_ids"])
    prompt_hidden = torch.randn((1, prompt_len, n_targets * cfg["hidden_size"]),
                                generator=generator, dtype=torch.float32) * 0.3
    decode_hidden = torch.randn((1, 1, n_targets * cfg["hidden_size"]),
                                generator=generator, dtype=torch.float32) * 0.3
    prompt_hidden = prompt_hidden.to(torch.bfloat16)
    decode_hidden = decode_hidden.to(torch.bfloat16)
    anchor = torch.tensor([37], dtype=torch.long)

    with torch.inference_mode():
        model.forward_spec(anchor, prompt_hidden, 0)
        # Discard prefill hooks: it intentionally computes only main projections
        # and KV state, while the oracle records below describe the draft graph.
        records.clear()
        counters.clear()
        output_ids, logits, confidence = model.forward_spec(
            anchor, decode_hidden, prompt_len)
    cpu.set_trace(None)
    for hook in hooks:
        hook.remove()

    record(2, "final_logits", logits)
    record(2, "confidence", confidence)

    args.out.mkdir(parents=True, exist_ok=True)
    base_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(args.base_fixture / "model.safetensors"), framework="pt",
                   device="cpu") as source:
        for name in source.keys():
            base_tensors[name] = source.get_tensor(name).contiguous()
    parameters = dict(model.named_parameters())
    for name, value in model.state_dict().items():
        if (name.startswith("mtp.") and ".embed.weight" not in name
                and ".head.weight" not in name):
            parameter = parameters.get(name)
            fmt = getattr(parameter, "_dspark_oracle_format", "") if parameter is not None else ""
            if fmt == "fp8":
                value = cpu.dequant_fp8_weight(parameter._dspark_oracle_q,
                                                parameter._dspark_oracle_scale)
            elif fmt == "fp4":
                value = cpu.dequant_fp4_weight(parameter._dspark_oracle_q,
                                                parameter._dspark_oracle_scale)
            base_tensors[name] = value.detach().float().cpu().contiguous()
    save_file(base_tensors, str(args.out / "model.safetensors"))
    (args.out / "config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(args.base_fixture / "native_to_soma.json", args.out / "native_to_soma.json")
    write_records(args.out / "dspark_oracle.somaact", records)

    reference = {
        "format": 1,
        "seed": SEED,
        "source_revision": PINNED_REVISION,
        "model_py_sha256": PINNED_MODEL_SHA256,
        "kernel_py_sha256": PINNED_KERNEL_SHA256,
        "cpu_kernels": {
            "fp8": "explicit-e4m3fn-e8m0",
            "fp4": "explicit-e2m1-e8m0",
            "sparse_attention": "literal-gather-sink-softmax",
        },
        "prompt_length": prompt_len,
        "target_layer_ids": cfg["dspark_target_layer_ids"],
        "anchor": int(anchor[0]),
        "prompt_hidden": prompt_hidden.float().reshape(-1).tolist(),
        "decode_hidden": decode_hidden.float().reshape(-1).tolist(),
        "proposal_tokens": output_ids[0, 1:].tolist(),
        "confidence": confidence[0].float().tolist(),
        "confidence_probability": torch.sigmoid(confidence[0].float()).tolist(),
        "logits": logits[0].float().reshape(-1).tolist(),
        "record_count": len(records),
    }
    (args.out / "dspark_reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "fixture_version": 1,
        "implementation": "pinned DeepSeek inference/model.py with CPU kernel replacements",
        "seed": SEED,
        "source_revision": PINNED_REVISION,
        "model_py_sha256": PINNED_MODEL_SHA256,
        "kernel_py_sha256": PINNED_KERNEL_SHA256,
        "weights_sha256": sha256(args.out / "model.safetensors"),
        "oracle_sha256": sha256(args.out / "dspark_oracle.somaact"),
        "proposal_tokens": reference["proposal_tokens"],
    }
    (args.out / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote DSpark source fixture and {len(records)} oracle records to {args.out}")
    print("proposal:", reference["proposal_tokens"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
