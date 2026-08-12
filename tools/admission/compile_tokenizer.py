#!/usr/bin/env python3
"""Compile a HuggingFace tokenizer.json into Soma's runtime format + its oracle.

Admission is GATED on the round-trip: the compiled tokenizer must reproduce HF
`tokenizers` byte-for-byte over the calibration corpus. This script emits both
halves — the compiled form and the golden ids to check it against.

DESIGN STANCE: recognize or reject. Never approximate.

The pretokenizer is the hard part. Rather than shipping a general Unicode regex
engine, this compiles a small set of EXACTLY RECOGNIZED patterns into an ordered
alternation program, with character classes expanded to codepoint-range tables
here (Python has full Unicode data; the engine should not carry its own).

An unrecognized pattern is refused with the pattern printed. A tokenizer gate
that silently approximated would pass while mis-tokenizing, and mis-tokenization
presents at G2 as "the model is subtly stupid" rather than as a tokenizer fault.

Supported today:
  * ByteLevel BPE, GPT-2 default pattern      (OLMoE)
  * ByteLevel BPE, Qwen3 Split pattern        (Qwen3-MoE, Qwen2-MoE)
  * ByteLevel BPE, GLM Split pattern          (GLM-5.2) — Qwen3's, with digits
    grouped in runs of up to three instead of one at a time

Refused, with the reason, until their families land:
  * multi-Split chains with explicit CJK/Latin ranges   (DeepSeek — G4)
  * byte_fallback / SentencePiece pipelines             (Mixtral)
  * legacy vocab.json + merges.txt, no tokenizer.json   (granite)

Usage:
    compile_tokenizer.py <model_dir> --out <fixture_dir>
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import unicodedata
from pathlib import Path

FORMAT_VERSION = 1
MAGIC = b"SOMATOK\0"

FLAG_NFC = 1 << 0
FLAG_BYTE_FALLBACK = 1 << 1
FLAG_ADD_PREFIX_SPACE = 1 << 2

# Item kinds in the compiled pretokenizer program.
ITEM_CLASS = 0
ITEM_LITERAL_CI = 1

# Special alternative behaviours that plain (class, quantifier) cannot express.
ALT_PLAIN = 0
ALT_WS_NOT_FOLLOWED_BY_NONSPACE = 1  # \s+(?!\S)
ALT_WS_THEN_NEWLINES = 2             # \s*[\r\n]+

GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# GLM-5.2. Identical to Qwen3 except `\p{N}{1,3}` for `\p{N}` — digits group in
# runs of up to three instead of one at a time, the GPT-4/cl100k convention.
GLM_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]


# ── Unicode class tables ─────────────────────────────────────────────────────
# Built from Python's unicodedata so the engine carries no Unicode tables of its
# own. BMP + astral up to 0x110000; ranges are coalesced, so these stay small.

MAX_CP = 0x110000


def ranges_for(predicate) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for cp in range(MAX_CP):
        if predicate(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CP - 1))
    return out


def _cat(cp: int) -> str:
    try:
        return unicodedata.category(chr(cp))
    except ValueError:
        return "Cn"


_WS = set(" \t\n\r\v\f\x1c\x1d\x1e\x1f\x85\xa0     　")
_WS |= {chr(c) for c in range(0x2000, 0x200B)}


def is_letter(cp: int) -> bool:
    return _cat(cp).startswith("L")


def is_number(cp: int) -> bool:
    return _cat(cp).startswith("N")


def is_space(cp: int) -> bool:
    return chr(cp) in _WS


class ClassTable:
    """Deduplicated pool of codepoint-range sets."""

    def __init__(self) -> None:
        self.classes: list[tuple[list[tuple[int, int]], bool]] = []
        self._index: dict[tuple, int] = {}

    def add(self, ranges: list[tuple[int, int]], negated: bool) -> int:
        key = (tuple(ranges), negated)
        if key in self._index:
            return self._index[key]
        self._index[key] = len(self.classes)
        self.classes.append((ranges, negated))
        return self._index[key]


def build_classes(pool: ClassTable) -> dict[str, int]:
    letters = ranges_for(is_letter)
    numbers = ranges_for(is_number)
    spaces = ranges_for(is_space)

    def union(*sets: list[tuple[int, int]]) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = sorted(r for s in sets for r in s)
        merged: list[tuple[int, int]] = []
        for lo, hi in pts:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    crlf = [(0x0A, 0x0A), (0x0D, 0x0D)]
    return {
        "L": pool.add(letters, False),
        "N": pool.add(numbers, False),
        "S": pool.add(spaces, False),
        "SPACE_LITERAL": pool.add([(0x20, 0x20)], False),
        "CRLF": pool.add(crlf, False),
        # [^\s\p{L}\p{N}]
        "NOT_SLN": pool.add(union(spaces, letters, numbers), True),
        # [^\r\n\p{L}\p{N}]
        "NOT_CRLF_LN": pool.add(union(crlf, letters, numbers), True),
    }


# ── Pretokenizer programs ────────────────────────────────────────────────────
# An alternative is (behaviour, [items]); an item is (kind, payload, min, max).
INF = 0xFFFFFFFF


def program_gpt2(c: dict[str, int]) -> list:
    return [
        (ALT_PLAIN, [(ITEM_LITERAL_CI, CONTRACTIONS, 0, 0)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["SPACE_LITERAL"], 0, 1), (ITEM_CLASS, c["L"], 1, INF)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["SPACE_LITERAL"], 0, 1), (ITEM_CLASS, c["N"], 1, INF)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["SPACE_LITERAL"], 0, 1), (ITEM_CLASS, c["NOT_SLN"], 1, INF)]),
        (ALT_WS_NOT_FOLLOWED_BY_NONSPACE, [(ITEM_CLASS, c["S"], 1, INF)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["S"], 1, INF)]),
    ]


def program_qwen(c: dict[str, int], digit_run: int = 1) -> list:
    """Qwen3's alternation, parameterized on the DIGIT RUN only.

    `digit_run` is the `{1,n}` bound on `\\p{N}`: 1 for Qwen3, 3 for GLM-5.2. That
    single number is the entire difference between the two patterns —

        Qwen3    ...|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|...
        GLM-5.2  ...|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|...

    — and it changes tokenization materially rather than cosmetically: "2024"
    becomes ["2024"]-ish groupings of up to three digits instead of four separate
    tokens, so a model trained one way and pretokenized the other is being fed
    numbers it has never seen. It is the GPT-4/cl100k convention.

    Parameterized rather than copied, because a second near-identical program is
    how the two drift: a fix to the whitespace or contraction alternatives would
    have to be made twice, and the compiler would keep passing either way.
    """
    return [
        (ALT_PLAIN, [(ITEM_LITERAL_CI, CONTRACTIONS, 0, 0)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["NOT_CRLF_LN"], 0, 1), (ITEM_CLASS, c["L"], 1, INF)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["N"], 1, digit_run)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["SPACE_LITERAL"], 0, 1),
                     (ITEM_CLASS, c["NOT_SLN"], 1, INF),
                     (ITEM_CLASS, c["CRLF"], 0, INF)]),
        (ALT_WS_THEN_NEWLINES, [(ITEM_CLASS, c["S"], 0, INF), (ITEM_CLASS, c["CRLF"], 1, INF)]),
        (ALT_WS_NOT_FOLLOWED_BY_NONSPACE, [(ITEM_CLASS, c["S"], 1, INF)]),
        (ALT_PLAIN, [(ITEM_CLASS, c["S"], 1, INF)]),
    ]


# ── Byte-level codec ─────────────────────────────────────────────────────────

def byte_decoder() -> dict[str, int]:
    """Inverse of HF's bytes_to_unicode: byte-level char -> raw byte.

    BPE is run in the BYTE domain rather than on these codepoints. The mapping is
    a bijection, so the two are equivalent — and working in bytes means the
    engine needs no Unicode handling in the merge loop at all.
    """
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_to_bytes(tok: str, dec: dict[str, int]) -> bytes | None:
    out = bytearray()
    for ch in tok:
        if ch not in dec:
            return None
        out.append(dec[ch])
    return bytes(out)


# ── Calibration corpus ───────────────────────────────────────────────────────
# Chosen to exercise every alternative in both programs, not to be representative
# text. A corpus that avoids the hard cases makes the gate meaningless.

def corpus(added_tokens: list[str]) -> list[str]:
    base = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "it's a test, isn't it? I'd say they've won and we'll see.",
        "IT'S UPPERCASE AND It'S Mixed",                      # (?i:) on contractions
        "3.14159 and 42 and 0x1F and 1,000,000",
        "  leading spaces",
        "trailing spaces   ",
        "internal    run    of    spaces",
        "tabs\there\tand\tthere",
        "line one\nline two\r\nline three\n\n\nmany blanks",
        "   \n   \n  mixed whitespace and newlines   \n",
        "trailing newline\n",
        "\n",
        "   ",
        "",
        "punctuation!!!???...---===+++",
        "def f(x: int) -> int:\n    return x * 2  # comment\n",
        '{"json": [1, 2, 3], "nested": {"k": null}}',
        "emoji: \U0001F600\U0001F680 and \U0001F1EC\U0001F1E7 flag",
        "accents: café naïve über ça va Renée",
        "greek: αβγδ cyrillic: добро",
        "cjk: 你好世界 こんにちは 안녕하세요",
        "arabic: مرحبا hebrew: שלום",
        "mixed 你好 world 123 éà !!! \n\t end",
        "a" * 200,
        "é" * 50,
        "NFC test: é vs é",                        # composed vs decomposed
        "zero​width​space and non breaking",
    ]
    # Added/special tokens must survive verbatim — they are matched before
    # pretokenization and must never be split by BPE.
    base.extend(added_tokens[:8])
    if added_tokens:
        base.append(f"text {added_tokens[0]} more text")
    return base


# ── Recognition ──────────────────────────────────────────────────────────────

class Unsupported(Exception):
    pass


def recognize(tj: dict) -> tuple[str, bool]:
    """Return (program_name, add_prefix_space). Raises Unsupported with a reason."""
    model = tj.get("model", {})
    if model.get("type") != "BPE":
        raise Unsupported(f"model type {model.get('type')!r}; only BPE is compiled today")
    if model.get("byte_fallback"):
        raise Unsupported(
            "byte_fallback BPE (SentencePiece-converted pipeline) needs a different "
            "encode path — unk fusing and byte tokens — and is not implemented")

    pre = tj.get("pre_tokenizer") or {}
    ptype = pre.get("type")

    if ptype == "ByteLevel":
        if not pre.get("use_regex", True):
            raise Unsupported("bare ByteLevel with use_regex=false has no pretokenizer to compile")
        return "gpt2", bool(pre.get("add_prefix_space", False))

    if ptype == "Sequence":
        subs = pre.get("pretokenizers", [])
        splits = [s for s in subs if s.get("type") == "Split"]
        bl = [s for s in subs if s.get("type") == "ByteLevel"]
        others = [s.get("type") for s in subs if s.get("type") not in ("Split", "ByteLevel")]
        if others:
            raise Unsupported(f"pretokenizer stages {others} are not compiled")
        if len(splits) != 1:
            raise Unsupported(
                f"{len(splits)} Split stages; only a single recognized Split is compiled "
                "(DeepSeek's 5-stage chain with explicit CJK/Latin ranges lands with G4)")
        pattern = (splits[0].get("pattern") or {}).get("Regex")
        add_prefix = bool(bl[0].get("add_prefix_space", False)) if bl else False
        if pattern == QWEN_PATTERN:
            return "qwen", add_prefix
        if pattern == GLM_PATTERN:
            return "glm", add_prefix
        if pattern == GPT2_PATTERN:
            return "gpt2", add_prefix
        raise Unsupported("unrecognized Split pattern:\n    " + str(pattern))

    raise Unsupported(f"pretokenizer type {ptype!r} is not compiled")


# ── Writer ───────────────────────────────────────────────────────────────────

def w_u32(fh, v: int) -> None:
    fh.write(struct.pack("<I", v))


def w_bytes(fh, b: bytes) -> None:
    w_u32(fh, len(b))
    fh.write(b)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv[1:])

    src = Path(args.model_dir)
    tj_path = src / "tokenizer.json"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tj_path.exists():
        legacy = (src / "vocab.json").exists() and (src / "merges.txt").exists()
        reason = ("legacy vocab.json + merges.txt with no tokenizer.json; the slow-tokenizer "
                  "path is not compiled" if legacy else "no tokenizer files at all")
        print(f"  REFUSED  {src.name}: {reason}")
        (out_dir / "tokenizer.unsupported").write_text(reason + "\n", encoding="utf-8")
        return 3

    tj = json.loads(tj_path.read_text(encoding="utf-8"))

    try:
        program_name, add_prefix_space = recognize(tj)
    except Unsupported as e:
        print(f"  REFUSED  {src.name}: {e}")
        (out_dir / "tokenizer.unsupported").write_text(str(e) + "\n", encoding="utf-8")
        return 3

    norm = tj.get("normalizer") or {}
    norm_type = norm.get("type")
    if norm_type not in (None, "NFC"):
        print(f"  REFUSED  {src.name}: normalizer {norm_type!r} is not implemented")
        (out_dir / "tokenizer.unsupported").write_text(
            f"normalizer {norm_type}\n", encoding="utf-8")
        return 3

    model = tj["model"]
    dec = byte_decoder()

    added = tj.get("added_tokens", [])
    added_content = {a["content"] for a in added}

    # vocab: id -> raw bytes
    #
    # Added tokens BYPASS byte-level encoding: they appear in model.vocab in
    # literal form, so a token like 24 consecutive spaces has no Ġ-encoding and
    # must be taken as raw UTF-8. OLMoE ships several of these as ordinary
    # (non-special) added tokens.
    vocab_items: dict[int, bytes] = {}
    for tok, tid in model["vocab"].items():
        raw = token_to_bytes(tok, dec)
        if raw is None:
            if tok in added_content:
                raw = tok.encode("utf-8")
            else:
                print(f"  REFUSED  {src.name}: vocab token {tok!r} is not byte-level encodable")
                return 3
        vocab_items[int(tid)] = raw

    for a in added:
        vocab_items[int(a["id"])] = a["content"].encode("utf-8")

    n_vocab = (max(vocab_items) + 1) if vocab_items else 0

    # merges, in rank order
    merges: list[tuple[bytes, bytes]] = []
    for m in model["merges"]:
        # Two serializations for the same thing: OLMoE writes "Ġ t", Qwen3 writes
        # ["Ġ", "t"]. Another per-family spelling the adapter absorbs.
        if isinstance(m, str):
            left, _, right = m.partition(" ")
        else:
            left, right = m[0], m[1]
        lb, rb = token_to_bytes(left, dec), token_to_bytes(right, dec)
        if lb is None or rb is None:
            continue
        merges.append((lb, rb))

    pool = ClassTable()
    classes = build_classes(pool)
    if program_name == "gpt2":
        program = program_gpt2(classes)
    else:
        # The digit run is the only thing that differs, so it is the only thing
        # selected here. See program_qwen.
        program = program_qwen(classes, digit_run=3 if program_name == "glm" else 1)

    flags = 0
    if norm_type == "NFC":
        flags |= FLAG_NFC
    if add_prefix_space:
        flags |= FLAG_ADD_PREFIX_SPACE

    # ── tokenizer.soma ───────────────────────────────────────────────────────
    tok_path = out_dir / "tokenizer.soma"
    with open(tok_path, "wb") as fh:
        fh.write(MAGIC)
        w_u32(fh, FORMAT_VERSION)
        w_u32(fh, flags)
        w_u32(fh, n_vocab)
        w_u32(fh, len(merges))
        w_u32(fh, len(added))
        w_u32(fh, len(pool.classes))
        w_u32(fh, len(program))

        for tid in range(n_vocab):
            w_bytes(fh, vocab_items.get(tid, b""))
        for lb, rb in merges:
            w_bytes(fh, lb)
            w_bytes(fh, rb)
        for a in added:
            w_bytes(fh, a["content"].encode("utf-8"))
            w_u32(fh, int(a["id"]))
            w_u32(fh, 1 if a.get("special") else 0)
        for ranges, negated in pool.classes:
            w_u32(fh, len(ranges))
            w_u32(fh, 1 if negated else 0)
            for lo, hi in ranges:
                w_u32(fh, lo)
                w_u32(fh, hi)
        for behaviour, items in program:
            w_u32(fh, behaviour)
            w_u32(fh, len(items))
            for kind, payload, lo, hi in items:
                w_u32(fh, kind)
                if kind == ITEM_LITERAL_CI:
                    w_u32(fh, len(payload))
                    for lit in payload:
                        w_bytes(fh, lit.encode("utf-8"))
                else:
                    w_u32(fh, payload)
                    w_u32(fh, lo)
                    w_u32(fh, hi)

    # ── oracle: HF's own answer ──────────────────────────────────────────────
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(str(tj_path))

    texts = corpus([a["content"] for a in added])

    # NFC normalization is not implemented in the engine (C++ has no Unicode
    # normalization without ICU, and a composition table is far larger than the
    # rest of this format). The gate must therefore be explicit about what it
    # covers rather than passing by accident.
    #
    # Every corpus string is checked to be NFC-stable. If one is not, it is
    # dropped and COUNTED — a silently-shrinking corpus would make the gate
    # weaker over time without anyone noticing.
    nfc_unstable = [t for t in texts if unicodedata.normalize("NFC", t) != t]
    texts = [t for t in texts if unicodedata.normalize("NFC", t) == t]
    if nfc_unstable:
        print(f"           NOTE: dropped {len(nfc_unstable)} NFC-unstable corpus string(s); "
              f"engine-side NFC is not implemented")

    encoded = [hf.encode(t, add_special_tokens=False).ids for t in texts]

    oracle_path = out_dir / "tokenizer_oracle.bin"
    with open(oracle_path, "wb") as fh:
        fh.write(b"SOMATORC")
        w_u32(fh, FORMAT_VERSION)
        w_u32(fh, len(texts))
        for text, ids in zip(texts, encoded):
            w_bytes(fh, text.encode("utf-8"))
            w_u32(fh, len(ids))
            for i in ids:
                w_u32(fh, i)

    meta = {
        "source": str(src),
        "program": program_name,
        "normalizer": norm_type,
        "add_prefix_space": add_prefix_space,
        "n_vocab": n_vocab,
        "n_merges": len(merges),
        "n_added": len(added),
        "n_classes": len(pool.classes),
        "corpus_size": len(texts),
        "nfc_implemented_in_engine": False,
        "nfc_unstable_dropped": len(nfc_unstable),
        "total_oracle_tokens": sum(len(e) for e in encoded),
    }
    (out_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"  OK       {src.name:<32} program={program_name:<5} vocab={n_vocab} "
          f"merges={len(merges)} classes={len(pool.classes)} "
          f"corpus={len(texts)} tokens={meta['total_oracle_tokens']} "
          f"({tok_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
