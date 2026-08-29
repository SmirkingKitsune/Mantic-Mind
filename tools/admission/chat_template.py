#!/usr/bin/env python3
"""Compile a checkpoint's chat template into token ids, or refuse.

Imported by compile_tokenizer.py. Offline only; never a runtime dependency.

WHY THIS IS NOT A JINJA INTERPRETER. GLM-5.3's template is 10,465 bytes of
macros, namespaces, `break`, and tool-call plumbing. Reimplementing that in C++
would be a second renderer that has to stay bug-for-bug identical with the first
one forever, and the failure mode when it drifts is not a crash — it is a
correctly-served model answering a subtly differently-framed prompt.

So the template is not interpreted. It is MEASURED. The real Jinja renderer runs
here, at admission, against probe conversations built from sentinels that cannot
occur in the template's own text; the scaffolding falls out as the text around
each sentinel, and the engine's whole job at runtime is to concatenate.

WHY THAT IS SAFE TO CONCATENATE. Every seam this extraction cuts falls on a
special token — `<|user|>`, `<think>`, `<|im_start|>` — which both this tokenizer
and HF's treat as an atomic unit that BPE may not merge across. That is a claim
about a specific tokenizer, not a general truth, so `verify()` proves it per
model: it renders whole conversations, tokenizes them as one string, and requires
the piecewise assembly to match id for id.

RECOGNIZE OR REFUSE, never approximate — the rule compile_tokenizer.py already
follows for pretokenizers. A template whose behaviour this shape cannot express
produces no compiled template at all and a stated reason; `soma serve` then falls
back to flattening messages, which is honest and visibly worse, rather than
emitting a prompt that is subtly wrong and looks fine.

WHAT THE BATTERY IS FOR. Every rule below was measured against a real template
and several were wrong the first time. GLM-5.2 and GLM-5.3 spell `clear_thinking`
with OPPOSITE defaults — 5.2 drops prior reasoning unless told not to, 5.3 keeps
it unless told to — and reading either one's source would have produced a
compiler that was confidently wrong about the other. The battery is what found
that, and it is why nothing here is inferred from template source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Roles, in the order soma::MessageRole declares them. The C++ enum is the
# authority; this is a second transcription, and the chat oracle is what stops
# the two from silently renumbering past each other.
ROLES = ("system", "user", "assistant", "tool")

# Sentinels. ASCII, no whitespace at either end (so a template that `.strip()`s
# content cannot eat part of one), and containing no character that Jinja, JSON
# encoding, or a BPE merge table treats specially.
S_CONTENT = "%%SOMACONTENT%%"
S_CONTENT2 = "%%SOMACONTENTTWO%%"
S_REASON = "%%SOMAREASONING%%"

# `reasoning_effort` values worth probing: OpenAI's four plus the two GLM names.
# Probing a value the template ignores is free — it renders as the default and is
# then not stored — while missing one is not, because the request would quietly
# get a different system prologue than it asked for.
EFFORT_KEYS = ("minimal", "low", "medium", "high", "max", "none")

FLAG_SUPPORTS_THINKING = 1 << 0
FLAG_ASSISTANT_SPLITS_THINK = 1 << 1
FLAG_ASSISTANT_STRIPS = 1 << 2
FLAG_CLEAR_THINKING_SETTABLE = 1 << 3
FLAG_CLEAR_THINKING_DEFAULT = 1 << 4
FLAG_ENABLE_THINKING = 1 << 5
FLAG_REASONING_EFFORT = 1 << 6
# The template REMOVES a `<think>...</think>` block from a historical assistant
# turn instead of re-emitting it. Qwen3 does this; GLM does not. It is a third
# state, not the absence of the first: an engine that passed the block through
# would feed the model its own scratchpad back as if it were an answer.
FLAG_ASSISTANT_DROPS_THINK = 1 << 7


class Unsupported(Exception):
    """The template does not fit the shape the engine can assemble."""


@dataclass
class ChatScaffold:
    """The template, as strings. compile_tokenizer.py turns these into ids."""

    # The prologue, and the variants of it selected by `reasoning_effort` and
    # `enable_thinking`. GLM-5.2 varies on both — turning thinking off removes
    # its whole `Reasoning Effort` system block — so one BOS is not enough.
    bos: str = ""
    prologues: dict[tuple[str, bool], str] = field(default_factory=dict)

    # Four strings per role, not two, because a RUN of same-role messages is
    # framed once as a run and once per message. GLM-5.3 opens a run of tool
    # results with `<|observation|>` and wraps each in `<tool_response>`; Qwen3
    # closes the run with `<|im_end|>` and separates each with a newline. A
    # model with only prefix/suffix gets one of those two families wrong.
    run_prefix: list[str] = field(default_factory=lambda: [""] * 4)
    prefix: list[str] = field(default_factory=lambda: [""] * 4)
    suffix: list[str] = field(default_factory=lambda: [""] * 4)
    run_suffix: list[str] = field(default_factory=lambda: [""] * 4)

    # The assistant prefix has two spellings and BOTH are measured rather than
    # composed. `prefix[assistant]` is what a turn with no reasoning emits — on
    # GLM that is `<|assistant|><think></think>`, an empty thinking block whose
    # inner whitespace differs by family — while `assistant_prefix_thinking` is
    # what a turn WITH reasoning emits and `thinking_close` is what sits between
    # the reasoning and the content.
    assistant_prefix_thinking: str = ""
    thinking_close: str = ""

    generation_prompt: str = ""
    generation_prompt_nothink: str = ""
    flags: int = 0

    def supports(self, flag: int) -> bool:
        return bool(self.flags & flag)

    def prologue(self, effort: str | None, enable_thinking: bool) -> str:
        """Exact match, then the default effort at this thinking setting, then BOS.

        The middle step is what makes an unrecognized effort behave the way the
        template behaves: GLM-5.3 renders `medium` as `Max` rather than erroring,
        and an engine that refused it would be stricter than the model.
        """
        for key in ((effort or "", enable_thinking), ("", enable_thinking)):
            if key in self.prologues:
                return self.prologues[key]
        return self.bos


# ── the renderer ─────────────────────────────────────────────────────────────


def jinja_env():
    """A Jinja environment configured the way `transformers` configures its own.

    `trim_blocks` and `lstrip_blocks` are not cosmetic: they decide whether the
    newline after `{%- endif %}` reaches the prompt, and a template rendered with
    different settings than the model was trained against is wrong in a way that
    still produces fluent output. Mirrored explicitly rather than imported from
    `transformers`, whose import path for it has moved more than once — a
    ModuleNotFoundError is a better failure than a silently different default.
    """
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # jinja2's own `tojson` is HTML-safe and escapes `<`, `>`, `&` and `'`.
        # A chat template full of `<tool_call>` markup would come out mangled;
        # `transformers` replaces the filter for exactly that reason.
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise Unsupported(f"template raised: {message}")

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def load_template(src) -> str | None:
    """The template text: a `chat_template.jinja` file, or the tokenizer_config key.

    Both spellings are in the wild — GLM ships the file, Qwen3 embeds the string
    — and a compiler that read only one would report "no chat template" about
    half the checkpoints that have one.
    """
    path = src / "chat_template.jinja"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    config = src / "tokenizer_config.json"
    if config.is_file():
        value = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(value, str):
            return value
        # A LIST of named templates (`{"name": ..., "template": ...}`) is the
        # multi-template spelling. Refused rather than guessed at: picking one
        # by position would pick the tool-use variant for some checkpoints.
        if isinstance(value, list):
            raise Unsupported("tokenizer_config.json carries a LIST of named chat "
                              "templates; which one to serve is not this "
                              "compiler's decision to make")
    return None


def make_renderer(template_text: str):
    template = jinja_env().from_string(template_text)

    def render(messages, add_generation_prompt=False, **kwargs) -> str:
        return template.render(messages=messages,
                               add_generation_prompt=add_generation_prompt,
                               **kwargs)

    return render


# ── extraction ───────────────────────────────────────────────────────────────


def _find(text: str, sentinel: str, what: str) -> int:
    at = text.find(sentinel)
    if at < 0:
        raise Unsupported(f"{what}: the probe content does not survive rendering, "
                          f"so the template rewrites or drops message content")
    return at


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    head = strings[0]
    for other in strings[1:]:
        n = 0
        while n < min(len(head), len(other)) and head[n] == other[n]:
            n += 1
        head = head[:n]
    return head


def extract(render) -> ChatScaffold:
    """Derive the scaffolding by rendering probes and reading off the sentinels."""
    sc = ChatScaffold()

    # Per role: one message, then two, which separates the once-per-RUN opener
    # from the per-message framing. GLM-5.3 emits `<|observation|>` once at the
    # start of a run of tool results and `<tool_response>` around each; without
    # the split, a two-result turn gets two observation markers.
    #
    # `head` is the prologue plus that role's run opener, still fused — the two
    # cannot be told apart from one role's renderings alone.
    heads: list[str] = []
    for idx, role in enumerate(ROLES):
        one = render([{"role": role, "content": S_CONTENT}])
        two = render([{"role": role, "content": S_CONTENT},
                      {"role": role, "content": S_CONTENT2}])
        if not two.startswith(one[:_find(one, S_CONTENT, role)]):
            raise Unsupported(f"a second {role} message changes how the first is "
                              f"framed; this shape cannot express that")
        at_two = _find(two, S_CONTENT2, f"{role} (second of two)")
        at_one_in_two = _find(two, S_CONTENT, f"{role} (first of two)")
        # What separates two same-role messages is this role's per-message
        # suffix followed by its per-message prefix, fused.
        between = two[at_one_in_two + len(S_CONTENT):at_two]

        at_one = _find(one, S_CONTENT, role)
        # A lone message is both the first and the LAST of its run, so what
        # follows it is the per-message suffix plus the run's closer.
        last_suffix = one[at_one + len(S_CONTENT):]

        # The per-message suffix is what those two agree on. Splitting `between`
        # by the lone message's suffix instead would be wrong for any family
        # that closes a run — Qwen3's `</tool_response><|im_end|>\n` shares only
        # its first half with what sits between two tool results.
        sc.suffix[idx] = _common_prefix([between, last_suffix])
        sc.run_suffix[idx] = last_suffix[len(sc.suffix[idx]):]
        sc.prefix[idx] = between[len(sc.suffix[idx]):]
        if not one[:at_one].endswith(sc.prefix[idx]):
            raise Unsupported(
                f"the {role} framing for a first message ({one[:at_one]!r}) does "
                f"not end with the framing for a later one ({sc.prefix[idx]!r})")
        heads.append(one[:at_one - len(sc.prefix[idx])] if sc.prefix[idx]
                     else one[:at_one])

    # The prologue is what every role's head shares. `render([])` is the direct
    # question and is preferred when the template tolerates it — Qwen3's does
    # not, indexing `messages[0]` unconditionally — but it is only trusted when
    # every head actually starts with it, because a template that emits a
    # default system turn for an empty conversation answers a different
    # question than the one being asked.
    sc.bos = _common_prefix(heads)
    try:
        empty = render([])
        if all(head.startswith(empty) for head in heads):
            sc.bos = empty
    except Exception:
        pass
    for idx in range(len(ROLES)):
        sc.run_prefix[idx] = heads[idx][len(sc.bos):]

    _extract_prologues(render, sc)
    _extract_thinking(render, sc)
    _extract_generation_prompt(render, sc)
    _extract_content_transform(render, sc)
    _extract_clear_thinking(render, sc)
    return sc


def _extract_prologues(render, sc: ChatScaffold) -> None:
    """The prologue over `reasoning_effort` x `enable_thinking`.

    A product rather than two independent axes because they are not independent:
    GLM-5.2 suppresses its whole `Reasoning Effort` system block when thinking is
    off, so the effort it was asked for stops mattering. Storing them separately
    would have to pick one to win.
    """
    probe = [{"role": "user", "content": S_CONTENT}]
    a_user = ROLES.index("user")

    def prologue_for(**kwargs) -> str | None:
        try:
            text = render(probe, **kwargs)
        except Exception:
            return None
        at = text.find(S_CONTENT)
        if at < 0:
            return None
        rendered_head = text[:at]
        # Everything before this message's own framing is the prologue.
        tail = sc.run_prefix[a_user] + sc.prefix[a_user]
        if tail and not rendered_head.endswith(tail):
            raise Unsupported(
                f"the user framing changes with {kwargs!r}; the prologue and the "
                f"per-message framing are not separable for this template")
        return rendered_head[:len(rendered_head) - len(tail)]

    table: dict[tuple[str, bool], str] = {}
    for effort in ("",) + EFFORT_KEYS:
        for thinking in (True, False):
            kwargs = {}
            if effort:
                kwargs["reasoning_effort"] = effort
            if not thinking:
                kwargs["enable_thinking"] = False
            value = prologue_for(**kwargs)
            if value is not None:
                table[(effort, thinking)] = value

    # Only what DIFFERS from the plain prologue is stored; a template that
    # ignores both parameters stores nothing and the engine then has nothing to
    # get wrong.
    for key, value in table.items():
        if value != sc.bos:
            sc.prologues[key] = value

    # The FLAGS are comparisons along one axis at a time, and that distinction is
    # not pedantic. Setting `enable_thinking` merely because some `(effort,
    # False)` entry differs from the plain prologue marks GLM-5.3 as having a
    # thinking switch it does not have — the entry differs because of the EFFORT
    # — and the engine then accepts `enable_thinking: false` and silently
    # changes nothing, which is the exact failure the refusal exists to prevent.
    for (effort, thinking), value in table.items():
        if effort and value != table.get(("", thinking)):
            sc.flags |= FLAG_REASONING_EFFORT
        if thinking and value != table.get((effort, False)):
            sc.flags |= FLAG_ENABLE_THINKING


def _extract_thinking(render, sc: ChatScaffold) -> None:
    """Separate the two assistant prefixes, and the closer between them.

    `extract` already read `prefix[assistant]` off a turn with plain content, so
    it necessarily swallowed whatever empty thinking block the template emits
    there. That is the right thing to have: one measured string, rather than an
    open marker and a close marker reassembled on a guess about the whitespace
    between them.
    """
    a, u = ROLES.index("assistant"), ROLES.index("user")

    # The probe is `user, assistant`, not a lone assistant turn, and that is not
    # tidiness. Templates that keep reasoning only for turns AFTER the last user
    # message — Qwen3 is one — put a lone assistant turn on the wrong side of
    # that boundary, so probing with one measures the history spelling and calls
    # it the live one. The reasoning is then recorded as dropped for every turn,
    # including the one the model is about to continue.
    convo = [{"role": "user", "content": S_CONTENT2},
             {"role": "assistant",
              # Reasoning supplied the way an inference server actually holds
              # it: inside the assistant's own content, which is what the
              # previous turn streamed.
              "content": f"<think>{S_REASON}</think>{S_CONTENT}"}]
    inline = render(convo)
    head = (sc.bos + sc.run_prefix[u] + sc.prefix[u] + S_CONTENT2 + sc.suffix[u]
            + sc.run_suffix[u] + sc.run_prefix[a])
    if not inline.startswith(head):
        raise Unsupported("an assistant turn carrying reasoning is framed "
                          "differently from one without it")
    rest = inline[len(head):]
    body = rest[:_find(rest, S_CONTENT, "assistant reasoning")]
    at = body.find(S_REASON)
    if at < 0:
        if S_REASON not in inline:
            # Reasoning is REMOVED from history rather than re-emitted. Recorded
            # as its own state: treating it as "no thinking channel" and passing
            # the block through would hand the model its own scratchpad back as
            # though it were the answer it gave.
            sc.flags |= FLAG_ASSISTANT_DROPS_THINK
            return
        raise Unsupported(
            "assistant reasoning is rendered somewhere other than between the "
            "turn's prefix and its content; this shape cannot express that")
    sc.assistant_prefix_thinking = body[:at]
    sc.thinking_close = body[at + len(S_REASON):]
    sc.flags |= FLAG_SUPPORTS_THINKING

    # The same reasoning offered as a separate `reasoning_content` field must
    # render identically. The engine has ONE channel for reasoning; two
    # spellings that disagreed would mean it had to choose, and choosing is what
    # this file exists to avoid.
    separate = render([convo[0], {"role": "assistant", "content": S_CONTENT,
                                  "reasoning_content": S_REASON}])
    if separate == inline:
        sc.flags |= FLAG_ASSISTANT_SPLITS_THINK


def _extract_generation_prompt(render, sc: ChatScaffold) -> None:
    base = render([{"role": "user", "content": S_CONTENT}])
    with_prompt = render([{"role": "user", "content": S_CONTENT}],
                         add_generation_prompt=True)
    if not with_prompt.startswith(base):
        raise Unsupported(
            "add_generation_prompt does not APPEND to the conversation, it "
            "rewrites it; this shape cannot express that")
    sc.generation_prompt = with_prompt[len(base):]

    nothink_base = render([{"role": "user", "content": S_CONTENT}],
                          enable_thinking=False)
    nothink = render([{"role": "user", "content": S_CONTENT}],
                     add_generation_prompt=True, enable_thinking=False)
    sc.generation_prompt_nothink = (nothink[len(nothink_base):]
                                    if nothink.startswith(nothink_base)
                                    else sc.generation_prompt)
    if sc.generation_prompt_nothink != sc.generation_prompt:
        sc.flags |= FLAG_ENABLE_THINKING


def _extract_content_transform(render, sc: ChatScaffold) -> None:
    """Does the template strip whitespace off assistant content? GLM does."""
    a = ROLES.index("assistant")
    head = sc.bos + sc.run_prefix[a] + sc.prefix[a]
    padded = render([{"role": "assistant", "content": f"  {S_CONTENT}  "}])
    if not padded.startswith(head):
        raise Unsupported("assistant framing changes with padded content")
    lead = padded[len(head):_find(padded, S_CONTENT, "assistant padding")]
    if lead == "":
        sc.flags |= FLAG_ASSISTANT_STRIPS
    elif lead != "  ":
        raise Unsupported(
            f"assistant content is transformed in a way that is neither "
            f"pass-through nor strip (leading {lead!r})")

    # The other three roles must be pass-through. A template that rewrote user
    # content would need that rewrite reimplemented in C++, which is the thing
    # this file refuses to do.
    for idx, role in enumerate(ROLES):
        if role == "assistant":
            continue
        text = render([{"role": role, "content": f"  {S_CONTENT}  "}])
        role_head = sc.bos + sc.run_prefix[idx] + sc.prefix[idx]
        if not text.startswith(role_head):
            raise Unsupported(f"{role} framing changes with padded content")
        at = _find(text, S_CONTENT, f"{role} padding")
        if text[len(role_head):at] != "  " or not text[at + len(S_CONTENT):].startswith("  "):
            raise Unsupported(f"{role} content is not passed through verbatim")


def _extract_clear_thinking(render, sc: ChatScaffold) -> None:
    """Whether prior turns keep their reasoning, by default and when asked.

    Both halves are measured, and that is the whole point. GLM-5.2 and GLM-5.3
    implement the same option with OPPOSITE defaults — 5.2 drops prior reasoning
    unless explicitly told not to, 5.3 keeps it unless explicitly told to — and a
    compiler that assumed either would produce a prompt for the other model that
    silently omits or invents a reasoning block.
    """
    if not sc.supports(FLAG_SUPPORTS_THINKING):
        return
    convo = [
        {"role": "assistant", "content": f"<think>{S_REASON}</think>{S_CONTENT}"},
        {"role": "user", "content": S_CONTENT2},
        {"role": "assistant", "content": f"<think>{S_REASON}</think>{S_CONTENT}"},
    ]

    def kept(**kwargs) -> int:
        return render(convo, **kwargs).count(S_REASON)

    # Two blocks survive when nothing is dropped; one when the turn at or before
    # the last user message loses its reasoning. Anything else is a rule this
    # engine does not implement, and saying so beats implementing it wrongly.
    for label, value in (("default", kept()),
                         ("clear_thinking=True", kept(clear_thinking=True)),
                         ("clear_thinking=False", kept(clear_thinking=False))):
        if value not in (1, 2):
            raise Unsupported(
                f"with {label} the template keeps {value} of 2 reasoning blocks; "
                f"the engine implements only 'keep all' and 'drop at or before "
                f"the last user message'")

    if kept() == 1:
        sc.flags |= FLAG_CLEAR_THINKING_DEFAULT
    if kept(clear_thinking=True) != kept(clear_thinking=False):
        sc.flags |= FLAG_CLEAR_THINKING_SETTABLE


# ── assembly, in Python, so it can be compared with the renderer ─────────────


def split_reasoning(sc: ChatScaffold, message: dict, index: int, last_user: int,
                    clear_thinking: bool) -> tuple[str, str]:
    """`(content, reasoning)` for one message, applying the clear_thinking rule.

    Mirrors `split_reasoning` in src/soma/tokenizer.cpp. Shared by both
    assemblers below so the text comparison and the id comparison cannot
    disagree about which half is which — if they could, one would be grading the
    other's bug.
    """
    content = message["content"]
    if message["role"] != "assistant":
        return content, ""
    reasoning = ""
    if "</think>" in content and (sc.supports(FLAG_SUPPORTS_THINKING)
                                  or sc.supports(FLAG_ASSISTANT_DROPS_THINK)):
        head, _, content = content.partition("</think>")
        reasoning = head.split("<think>")[-1]
    if sc.supports(FLAG_ASSISTANT_DROPS_THINK):
        reasoning = ""
    if clear_thinking and index <= last_user:
        reasoning = ""
    if sc.supports(FLAG_ASSISTANT_STRIPS):
        content = content.strip()
    return content, reasoning


def effective_clear_thinking(sc: ChatScaffold, requested: bool | None) -> bool:
    if requested is None or not sc.supports(FLAG_CLEAR_THINKING_SETTABLE):
        return sc.supports(FLAG_CLEAR_THINKING_DEFAULT)
    return requested


def assemble(sc: ChatScaffold, messages: list[dict], add_generation_prompt: bool,
             enable_thinking: bool = True, clear_thinking: bool | None = None,
             reasoning_effort: str | None = None) -> str:
    """Rebuild a prompt from the scaffolding — the C++ algorithm, in Python.

    Deliberately a SECOND implementation of what tokenizer.cpp does. verify()
    grades this against the Jinja renderer and the chat oracle grades the C++
    against the renderer's ids, so a disagreement shows up as a failure on one
    side rather than as both agreeing on the same mistake.
    """
    out = sc.prologue(reasoning_effort, enable_thinking)
    clear = effective_clear_thinking(sc, clear_thinking)
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"),
                    default=-1)
    for i, m in enumerate(messages):
        idx = ROLES.index(m["role"])
        if i == 0 or messages[i - 1]["role"] != m["role"]:
            out += sc.run_prefix[idx]
        content, reasoning = split_reasoning(sc, m, i, last_user, clear)
        # The two assistant prefixes are ALTERNATIVES, not a base plus an
        # insertion: `prefix[assistant]` already carries the empty thinking
        # block, so composing them would emit it twice.
        out += (sc.assistant_prefix_thinking + reasoning + sc.thinking_close
                if reasoning else sc.prefix[idx])
        out += content + sc.suffix[idx]
        if i + 1 == len(messages) or messages[i + 1]["role"] != m["role"]:
            out += sc.run_suffix[idx]
    if add_generation_prompt:
        out += (sc.generation_prompt if enable_thinking
                else sc.generation_prompt_nothink)
    return out


def assemble_ids(sc: ChatScaffold, case: dict, encode) -> list[int]:
    """The same assembly, concatenating ids — exactly what the engine will do."""
    messages = case["messages"]
    enable_thinking = case.get("enable_thinking", True)
    clear = effective_clear_thinking(sc, case.get("clear_thinking"))
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"),
                    default=-1)

    ids = list(encode(sc.prologue(case.get("reasoning_effort"), enable_thinking)))
    for i, m in enumerate(messages):
        idx = ROLES.index(m["role"])
        if i == 0 or messages[i - 1]["role"] != m["role"]:
            ids += encode(sc.run_prefix[idx])
        content, reasoning = split_reasoning(sc, m, i, last_user, clear)
        if reasoning:
            ids += (encode(sc.assistant_prefix_thinking) + encode(reasoning)
                    + encode(sc.thinking_close))
        else:
            ids += encode(sc.prefix[idx])
        ids += encode(content) + encode(sc.suffix[idx])
        if i + 1 == len(messages) or messages[i + 1]["role"] != m["role"]:
            ids += encode(sc.run_suffix[idx])
    if case.get("add_generation_prompt", False):
        ids += encode(sc.generation_prompt if enable_thinking
                      else sc.generation_prompt_nothink)
    return ids


# ── the battery ──────────────────────────────────────────────────────────────


def battery(sc: ChatScaffold) -> list[dict]:
    """Conversations chosen for the seams, not for coverage of English.

    Every one is a shape a piecewise assembler can get wrong while still
    producing something that reads correctly.

    Takes the scaffold because the option cases have to be REACHABLE. The engine
    refuses `enable_thinking: false` on a template with no such switch — that
    refusal is the feature — so recording such a case in the oracle would ask the
    engine to reproduce ids for a request it is right to reject, and the gate
    would fail on its own correct behaviour. Whether the unsupported options are
    refused is checked separately, engine-side.
    """
    U = {"role": "user", "content": "What is the capital of France?"}
    A = {"role": "assistant", "content": "<think>Recall geography.</think>Paris."}
    U2 = {"role": "user", "content": "And Germany?"}
    cases: list[dict] = [
        {"messages": [U], "add_generation_prompt": True},
        {"messages": [U], "add_generation_prompt": False},
        {"messages": [{"role": "system", "content": "Be terse."}, U],
         "add_generation_prompt": True},
        # A prior assistant turn WITH reasoning, which is where the two assistant
        # prefixes and the clear_thinking default all meet.
        {"messages": [U, A, U2], "add_generation_prompt": True},
        # ...and one without, which uses the empty-thinking spelling instead.
        {"messages": [U, {"role": "assistant", "content": "Paris."}, U2],
         "add_generation_prompt": True},
        # Whitespace either side of assistant content: stripped by some
        # templates, passed through by others.
        {"messages": [U, {"role": "assistant", "content": "  Paris.  "}, U2],
         "add_generation_prompt": True},
        # A RUN of tool results: the once-per-run opener versus per-message
        # framing.
        {"messages": [U, {"role": "assistant", "content": "<think>Look it up.</think>"},
                      {"role": "tool", "content": "{\"city\": \"Paris\"}"},
                      {"role": "tool", "content": "{\"pop\": 2100000}"},
                      {"role": "user", "content": "Thanks."}],
         "add_generation_prompt": True},
        # Consecutive same-role messages for the roles that are not tools.
        {"messages": [U, U], "add_generation_prompt": True},
        {"messages": [{"role": "system", "content": "One."},
                      {"role": "system", "content": "Two."}, U],
         "add_generation_prompt": True},
        # Multibyte content, so a byte-offset bug shows up as a mismatch rather
        # than as mojibake nobody reads.
        {"messages": [{"role": "user", "content": "Comment ça va, 世界? 🙂"}],
         "add_generation_prompt": True},
        # Content that LOOKS like scaffolding. Both sides must treat it the same
        # way; whatever that way is, it must not be an accident.
        {"messages": [{"role": "user", "content": "Say <|assistant|> and <think>."}],
         "add_generation_prompt": True},
    ]
    if sc.supports(FLAG_ENABLE_THINKING):
        cases.append({"messages": [U], "add_generation_prompt": True,
                      "enable_thinking": False})
        cases.append({"messages": [U, A, U2], "add_generation_prompt": True,
                      "enable_thinking": False})
    if sc.supports(FLAG_REASONING_EFFORT):
        for key in ("low", "high", "max", "medium"):
            cases.append({"messages": [U], "add_generation_prompt": True,
                          "reasoning_effort": key})
    if sc.supports(FLAG_CLEAR_THINKING_SETTABLE):
        for flag in (True, False):
            cases.append({"messages": [U, A, U2, A], "add_generation_prompt": True,
                          "clear_thinking": flag})
    return cases


def verify(sc: ChatScaffold, render, encode) -> list[dict]:
    """Prove the scaffolding reproduces the renderer, in text AND in token ids.

    Returns the oracle cases. Raises Unsupported on the first disagreement,
    naming the case: "the chat template does not round-trip" without saying which
    conversation is a bug report nobody can act on.

    The id comparison is not a restatement of the text one. Assembling from
    precompiled pieces is sound only if every seam falls where BPE cannot merge
    across it, and that is a property of the TOKENIZER, not of the template. A
    family whose merges reach across `<|user|>` fails here rather than in
    production.
    """
    cases = []
    for case in battery(sc):
        kwargs = {k: v for k, v in case.items()
                  if k not in ("messages", "add_generation_prompt")}
        want_text = render(case["messages"],
                           add_generation_prompt=case.get("add_generation_prompt", False),
                           **kwargs)
        got_text = assemble(sc, case["messages"],
                            case.get("add_generation_prompt", False),
                            enable_thinking=case.get("enable_thinking", True),
                            clear_thinking=case.get("clear_thinking"),
                            reasoning_effort=case.get("reasoning_effort"))
        if got_text != want_text:
            raise Unsupported(_diff(case, want_text, got_text))

        want_ids = list(encode(want_text))
        got_ids = assemble_ids(sc, case, encode)
        if got_ids != want_ids:
            raise Unsupported(
                f"the same text tokenizes differently assembled piecewise "
                f"({len(got_ids)} ids) than whole ({len(want_ids)} ids) for "
                f"{_label(case)}; a BPE merge reaches across a scaffolding seam, "
                f"so this template cannot be compiled for this tokenizer")
        cases.append({**case, "ids": want_ids})
    return cases


def _label(case: dict) -> str:
    bits = [f"{len(case['messages'])} message(s)"]
    for key in ("add_generation_prompt", "enable_thinking", "clear_thinking",
                "reasoning_effort"):
        if key in case:
            bits.append(f"{key}={case[key]!r}")
    return ", ".join(bits)


def _diff(case: dict, want: str, got: str) -> str:
    at = 0
    while at < min(len(want), len(got)) and want[at] == got[at]:
        at += 1
    return (f"the compiled scaffolding does not reproduce the template for "
            f"{_label(case)}: they agree for {at} characters, then the template "
            f"has {want[at:at + 48]!r} and the assembly has {got[at:at + 48]!r}")
