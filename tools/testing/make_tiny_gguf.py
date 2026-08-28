#!/usr/bin/env python3
"""Build a minimal, REAL llama-architecture GGUF that llama-server can serve.

Why this exists
---------------
The G8 criterion "a Soma agent and a fallback agent run concurrently on the same
node" needs two DIFFERENT engines as real processes under one supervisor. Soma
runs against a converted container; llama.cpp needs a GGUF, and nothing in the
tree could supply one:

  * tests/fixtures/tiny/* are weights+config only. They carry no tokenizer,
    because the fp32 conformance oracle feeds raw token ids and never needed one.
    llama-server requires a vocab, so those fixtures cannot be converted.
  * llama.cpp's own ggml-vocab-*.gguf are vocab-only — llama-server starts,
    then dies with "missing tensor 'token_embd.weight'".
  * The real 13 GB source converts, and a multi-gigabyte artifact is not a test
    fixture. A test whose fixture cannot be committed is a test that only ever
    runs on one machine.

So this writes the smallest thing that is still genuinely a llama model: random
weights, real architecture, real SPM vocab with byte fallback. ~300 KB.

The weights are random, which is the point rather than a compromise. The test
this feeds asserts that two engine types COEXIST — ports, slots, descriptor
dispatch, independent lifecycle. None of that reads a logit. A real checkpoint
would make the fixture 10,000x larger and test exactly the same supervisor code.

Usage: make_tiny_gguf.py OUT.gguf
"""

import sys
from pathlib import Path

import numpy as np

from gguf import GGUFWriter

# Small enough to commit, large enough to be a real forward pass.
N_LAYERS = 2
N_EMBD = 64
N_HEAD = 4
N_HEAD_KV = 4
N_FF = 128
N_CTX = 512
HEAD_DIM = N_EMBD // N_HEAD


def build_vocab():
    """An SPM vocab with byte fallback, which is what llama.cpp expects.

    Byte tokens are not decoration: without <0x00>..<0xFF> llama.cpp cannot
    represent input it has no merge for, and rejects the model at load.
    """
    tokens, scores, toktypes = [], [], []

    def add(text, ttype, score=0.0):
        tokens.append(text)
        scores.append(score)
        toktypes.append(ttype)

    # gguf.TokenType: 2=UNKNOWN 3=CONTROL 1=NORMAL 6=BYTE
    add("<unk>", 2)
    add("<s>", 3)
    add("</s>", 3)
    for b in range(256):
        add(f"<0x{b:02X}>", 6)
    # A handful of ordinary pieces so the vocab is not purely bytes. The
    # leading U+2581 is SPM's space marker.
    for i, piece in enumerate(
        ["▁the", "▁a", "▁of", "▁and", "▁to",
         "▁in", "▁is", "▁it", "▁that", "▁for",
         "▁on", "▁with", "▁as", "▁was", "▁at",
         "in", "er", "on", "at", "en", "or", "an", "ar", "al", "re",
         "s", "e", "t", "a", "o", "i", "n", "h", "r", "d", "l", "u"]
    ):
        add(piece, 1, -float(i + 1))
    return tokens, scores, toktypes


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)

    tokens, scores, toktypes = build_vocab()
    n_vocab = len(tokens)

    # Deterministic: a fixture that differs run to run is not a fixture.
    rng = np.random.default_rng(0)

    def mat(rows, cols):
        # Small values keep the random forward pass numerically tame; llama.cpp
        # warns on non-finite logits and a wild init can trip it.
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float16)

    def norm(n):
        return np.ones(n, dtype=np.float32)

    w = GGUFWriter(str(out), "llama")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYERS)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(1)  # F16

    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)

    # Shapes follow torch's nn.Linear/nn.Embedding convention (out, in);
    # GGUFWriter reverses them into ggml order.
    w.add_tensor("token_embd.weight", mat(n_vocab, N_EMBD))
    for i in range(N_LAYERS):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "attn_q.weight", mat(N_EMBD, N_EMBD))
        w.add_tensor(p + "attn_k.weight", mat(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_v.weight", mat(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_output.weight", mat(N_EMBD, N_EMBD))
        w.add_tensor(p + "ffn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", mat(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", mat(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", mat(N_EMBD, N_FF))
    w.add_tensor("output_norm.weight", norm(N_EMBD))
    w.add_tensor("output.weight", mat(n_vocab, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size = out.stat().st_size
    print(f"wrote {out}  ({size / 1024:.0f} KiB, vocab {n_vocab}, "
          f"{N_LAYERS}L x {N_EMBD}d)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
