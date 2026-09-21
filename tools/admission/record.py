#!/usr/bin/env python3
"""The conversion record: one schema, one writer, one validator.

`container_meta.json` is what a conversion DID — not a second description of the
architecture, which is `config.json`'s job and `config.json`'s alone. It is read
once at open, never on the hot path, which is why it stays JSON and stays
inspectable rather than moving into the binary index.

What it did not have was a schema, and the cost was measurable. Three dialects
had grown from two writers with no declaration of which was which:

    general converter   21 keys
    DeepSeek-V4         30 keys
    DeepSeek-V4+DSpark  45 keys

Twelve keys were written by somebody and read by nobody. Several were read by
somebody and written by nobody for containers that reader would actually see —
and because every C++ reader used `j.value(key, default)`, **a key nobody wrote
was indistinguishable from a key absent on purpose**. That is the defect shape
this module exists to remove: `dspark_profiled_speedup` gated `--speculative
auto` for months while no writer emitted it, so the branch could not be taken and
nothing said so.

THE RULE, which is the actual deliverable:

    Every fact has exactly one authority. Any repetition must be (a) deliberate,
    (b) checked at open, and (c) documented as such at both sites.

The binary index's per-role quantization descriptor passes all three — it is
repeated so the runtime can check it before decoding a byte, and `open()` does
exactly that. A quant map recorded four times and reconciled once does not pass.

HOW ABSENCE IS EXPRESSED. Never by leaving a key out. A field that does not apply
carries an explicit empty value (`""`, `[]`, `0`) or is governed by a
DISCRIMINATOR — a field that is always present and whose value says which other
fields must be. There are two:

    dense_storage  "source" | "f32" | "quantized"   -> dtype_dense iff quantized
    dspark         "present" | "omitted" | "not-present"  -> dspark_* iff present

so a reader never has to guess whether a missing key means "not applicable here"
or "the writer broke". Both readings were available before, and they differ.

DELIBERATELY FLAT. `dspark_` already namespaces its group; nesting it under an
object would be tidier and would buy no checkable property, while rewriting every
reader in two languages. The schema below is what makes the record legible, not
the punctuation.
"""

from __future__ import annotations

RECORD_VERSION = 1

# `when`:
#   "always"            — required in every record, no exceptions
#   ("key", "value")    — required exactly when record["key"] == "value"
#
# `why` is not decoration. A field nobody can justify in one line is a field
# nobody reads, and seven of those were deleted to write this file. Two more
# looked dead and were not — `validate_deepseek_v4_full.py` reads them across a
# line break, which the first scan for readers did not survive. A grep is not a
# proof that nothing reads a key.
FIELDS: tuple[tuple, ...] = (
    # ── identity of the record itself ────────────────────────────────────────
    ("record_version", int, "always",
     "this schema's version; bumped when a required field is added or removed"),
    ("container_version", int, "always",
     "the BINARY index's format version, repeated so a reader that has only the "
     "record can tell whether it can parse the index at all"),

    # ── provenance: what this was built FROM ─────────────────────────────────
    ("source", str, "always",
     "the checkpoint directory this conversion read, as given on the command line"),
    ("source_repo", str, "always",
     "upstream repository id, or '' when the source was a local directory with no "
     "recorded origin. Explicitly empty rather than absent"),
    ("source_revision", str, "always",
     "upstream revision, or '' when the source carried no HF download metadata"),
    ("config_sha256", str, "always",
     "sha256 of the config.json this conversion read. The V4 fixture carried one "
     "matching no config.json anywhere, which is how a stale provenance field "
     "looks: worse than not having one"),
    ("index_sha256", str, "always",
     "sha256 of the source's safetensors index, or '' for a single-file checkpoint"),
    ("source_quantization", str, "always",
     "what the SOURCE was stored as ('none' or 'fp8-e4m3-block-HxW'). A q4_g "
     "container built from a bf16 upload and one built from that upload's fp8 twin "
     "are not the same artifact, and nothing else records which one this is"),
    ("omitted_namespaces", list, "always",
     "tensor name prefixes present in the source and deliberately left out — a "
     "vision tower, a projector, an MTP head. [] means nothing was dropped"),
    ("omitted_tensors", int, "always",
     "how many tensors those namespaces accounted for"),

    # ── shape of the routed half ─────────────────────────────────────────────
    ("model_type", str, "always", "config.json's model_type, as converted"),
    ("n_layers", int, "always", "layers in the container's index"),
    ("n_experts", int, "always", "routed experts per MoE layer"),
    ("n_shards", int, "always", "experts-*.bin files"),
    ("expert_bytes", int, "always",
     "one expert's payload, or 0 when they are not uniform. The planner's "
     "economics are built on this"),
    ("total_expert_bytes", int, "always", "the routed payload, unpadded"),

    # ── quantization, as written ─────────────────────────────────────────────
    ("dtype_gate_up", str, "always",
     "gate and up share one field because the converter interleaves them into one "
     "range; the binary role descriptor is what lets a reader check them apart"),
    ("dtype_down", str, "always", "the down projection's codec"),
    ("group", int, "always",
     "the group size REQUESTED. What was used per role is effective_groups_by_role, "
     "which can be smaller when a row is narrower than the request"),
    ("effective_groups_by_role", dict, "always",
     "{gate, up, down} -> the group each role was actually written at. Keyed by "
     "ROLE, not by dtype: two roles can share a dtype and not a row width, which "
     "is exactly how the dtype-keyed version was wrong by construction"),

    # ── the resident half ────────────────────────────────────────────────────
    ("dense_storage", str, "always",
     "DISCRIMINATOR. 'source' keeps the checkpoint's bf16/f16, 'f32' widens it, "
     "'quantized' packs it with a codec and requires dtype_dense"),
    ("dense_narrow_tensors", int, "always",
     "how many resident tensors kept a narrow source dtype instead of being widened"),
    ("dtype_dense", str, ("dense_storage", "quantized"),
     "the codec the resident half was packed with. Present only when it was"),
    # These two are REPETITION, and they pass the rule because something checks
    # them: validate_deepseek_v4_full.py compares each against the `total_size`
    # its safetensors index declares. A byte count recorded twice and reconciled
    # is a cross-check; recorded twice and reconciled nowhere is drift waiting.
    ("lossless_resident_bytes", int, ("dense_storage", "quantized"),
     "resident bytes left at source precision, checked against "
     "dense.safetensors.index.json"),
    ("quantized_resident_bytes", int, ("dense_storage", "quantized"),
     "resident bytes packed with dtype_dense, checked against "
     "dense.qweights.index.json"),

    # ── the tokenizer ────────────────────────────────────────────────────────
    ("tokenizer", str, "always",
     "'compiled' or 'unsupported'. Recorded rather than inferred from which files "
     "exist, so 'no tokenizer was possible for this family' is distinguishable "
     "from 'someone deleted one'"),

    # ── the auxiliary draft model ────────────────────────────────────────────
    ("dspark", str, "always",
     "DISCRIMINATOR. 'present' means the draft model is in this container and "
     "every dspark_* field below is required; 'omitted' means the source had one "
     "and it was left out; 'not-present' means the source had none"),
    ("dspark_format", int, ("dspark", "present"), "the draft payload's own version"),
    ("dspark_tensors", int, ("dspark", "present"), "tensors in the draft model"),
    ("dspark_stages", int, ("dspark", "present"), "draft stages"),
    ("dspark_target_layer_ids", list, ("dspark", "present"),
     "which base layers the draft stages shadow"),
    ("dspark_trained_block_size", int, ("dspark", "present"),
     "the block length the draft was trained at"),
    ("dspark_noise_token_id", int, ("dspark", "present"), "the draft's noise token"),
    ("dspark_markov_rank", int, ("dspark", "present"), "the draft's Markov rank"),
    ("dspark_confidence_head", bool, ("dspark", "present"),
     "whether the draft carries a confidence head"),
    ("dspark_expert_bytes", int, ("dspark", "present"), "one draft expert's payload"),
    ("dspark_total_expert_bytes", int, ("dspark", "present"),
     "the draft's routed payload, unpadded"),
    ("dspark_resident_bytes", int, ("dspark", "present"),
     "the draft's resident half, total"),
    ("dspark_lossless_resident_bytes", int, ("dspark", "present"),
     "the draft's resident bytes at source precision, checked against "
     "dspark.safetensors.index.json"),
    ("dspark_quantized_resident_bytes", int, ("dspark", "present"),
     "the draft's packed resident bytes, checked against dspark.qweights.index.json"),
    ("dspark_kv_bytes_per_sequence", int, ("dspark", "present"),
     "what one sequence costs the draft's KV cache"),
    ("dtype_dspark", str, ("dspark", "present"), "the draft's routed codec"),
)

BY_NAME = {f[0]: f for f in FIELDS}


def omitted_namespace(name: str) -> str:
    """The namespace a dropped tensor belongs to, spelled one way.

    The first two dotted components, which is the rule the V4 converter already
    used for its MTP heads (`mtp.0`, `mtp.1`, `mtp.2`) — with one refinement.
    An MTP head identified by LAYER INDEX rather than by prefix would collapse to
    `model.layers`, which is also where every kept layer lives and therefore says
    nothing; those keep their index, so GLM-5.2's extra stack reads
    `model.layers.78`.
    """
    parts = name.split(".")
    head = ".".join(parts[:2])
    if head == "model.layers" and len(parts) > 2:
        return ".".join(parts[:3])
    return head


class RecordError(ValueError):
    pass


def required(record: dict) -> set:
    """The fields this particular record must carry, per its discriminators."""
    out = set()
    for name, _types, when, _why in FIELDS:
        if when == "always":
            out.add(name)
        elif record.get(when[0]) == when[1]:
            out.add(name)
    return out


def validate(record: dict) -> list[str]:
    """Every complaint, rather than the first.

    A validator that stops at the first problem turns one reconversion into
    several, and the whole point is to find out what the writer got wrong in one
    pass.
    """
    problems = []
    if record.get("record_version") != RECORD_VERSION:
        problems.append(f"record_version is {record.get('record_version')!r}, "
                        f"this build writes and reads {RECORD_VERSION}")
    want = required(record)
    for name in sorted(want - set(record)):
        problems.append(f"missing required field {name!r} "
                        f"({BY_NAME[name][3].split(';')[0]})")
    for name in sorted(set(record) - {f[0] for f in FIELDS}):
        problems.append(f"unknown field {name!r}: the schema in record.py does not "
                        f"declare it, so nothing reads it")
    for name in sorted(set(record) & want):
        types = BY_NAME[name][1]
        value = record[name]
        # bool is an int in Python and the two mean different things here.
        if types is int and isinstance(value, bool):
            problems.append(f"{name!r} is a bool, expected int")
        elif not isinstance(value, types):
            problems.append(f"{name!r} is {type(value).__name__}, "
                            f"expected {types.__name__}")
    # A field present but NOT required by its discriminator is a writer bug: it
    # says something the record has just declared does not apply.
    for name in sorted(set(record) - want):
        if name in BY_NAME:
            when = BY_NAME[name][2]
            problems.append(f"{name!r} is present but {when[0]}={record.get(when[0])!r} "
                            f"says it does not apply")
    return problems


def build(**values) -> dict:
    """The one writer. Both converters go through here or they drift again."""
    record = {"record_version": RECORD_VERSION, **values}
    problems = validate(record)
    if problems:
        raise RecordError("the conversion record does not satisfy its own schema:\n  " +
                          "\n  ".join(problems))
    # Sorted, because the record is diffed by humans and by
    # tools/ci/check_fixture_containers.py, and key order is not a fact about
    # the conversion.
    return dict(sorted(record.items()))
