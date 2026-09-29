# CLAUDE.md — Mantic-Mind

Distributed LLM inference cluster, plus **Soma**, its own streaming mixture-of-experts inference
engine. C++20, CMake + vcpkg.

This file is for working in **this** repository. Its two siblings carry their own, because a shared
guidance file that no repository owns is a file that drifts from the code with no diff to show for
it — which is what happened to the last one, and why `../CLAUDE.md` spent months pointing at a
document that had been deleted.

## Where the answers already live

Do not re-derive these, and do not restate them here. Each has one owner, and a second copy is a
second thing to be wrong.

| Question | Authority |
|---|---|
| How do I build, install, package? | `README.md` |
| What is in this tree and why? | `docs/repo-layout.md` |
| How does Soma work? | `docs/architecture.md` |
| What is the `/v1/*` surface? | `docs/external-api.md` |
| Where does Soma end and the cluster begin? | `docs/mantic-mind-integration.md` |
| What does a container hold, byte for byte? | `schemas/container.md` |
| What is the IR, and what does `arch_hash` cover? | `schemas/arch-ir.md` |
| Why is this the way it is? | `docs/roadmap.md` — a log, written after the fact |

`docs/roadmap.md` is history. An entry describes what was true when it was written, so read it for
reasoning and never as a statement about the tree today.

## Building and testing

`README.md` has the full story. Two things it does not say, both of which cost time to rediscover:

**Build both toolchains before believing a change is done.** MSVC and GCC each catch what the other
does not, and not hypothetically: in one recent series MSVC alone caught a use-after-free and a
parameter shadow, and GCC alone caught two functions that had become dead. A change that compiles on
one is a change that has been checked once.

**`cmake --build … | grep` hides the build's exit code.** The pipeline reports grep's status, so a
build that failed on a warning-as-error reads as success. Redirect to a file and check `$?`, or read
the log.

**Seven tests fail under WSL for environment reasons and are not regressions.** The admission venv at
`tools/admission/.venv` is Windows-only and the Linux build tree shares it through `/mnt/z`, so
`MM_ADMISSION_PY` resolves to a `python.exe` that cannot run. Affected: `mm_fused_experts`,
`mm_verify_payload`, `mm_minimax_dialect`, `mm_kimi_dialect`, `mm_fp8_source`, `mm_bf16_source`, and
**`mm_reliability_tests`** — the last is the trap, because it looks like a pure C++ test and is not:
`admission_fetch_stage` drives the real admission ladder, which shells out to that interpreter.
Windows is the only tree where the whole suite runs. Use WSL for the `soma_*` C++ suite, which is
genuinely unaffected.

## Working rules

**Adding a model never removes one.** A family that used to serve must still serve. This is the rule
the tree is shaped around; `docs/repo-layout.md` explains the seams that enforce it, and
`tools/ci/check_seam.py` fails a change that reaches across one.

**`tools/ci/` is where invariants go to be enforced.** All sixteen run under ctest, and they exist
because each one caught something real. Fourteen are registered in `tests/CMakeLists.txt` and two
next to what they check — `check_seam.py` in `src/soma/`, `check_ui_api.py` in `src/control/` — so
counting only the first file undercounts them. If you find yourself writing "we should remember to…", write
a check instead. `check_test_registry.py` will notice a test you declared and did not register — a
suite that silently runs fewer tests than it defines is indistinguishable from a green run.

**Every fact has exactly one authority.** Any repetition must be (a) deliberate, (b) checked, and
(c) documented as such at both sites. The conversion record's schema is transcribed twice, in
`tools/admission/record.py` and `src/soma/arch_ir.cpp`, because they are in different languages —
and `tools/ci/check_container_record.py` drives both against the same mutations so they cannot
drift. That is what makes the repetition legitimate. A value recorded in four places and reconciled
in none is not.

**Containers are written once.** Conversion produces a finished container, identity and all; nothing
rewrites an index afterwards, and the engine has no serializer to do it with. This is not tidiness:
node cache identity is a hash over each file's `(relpath, size, mtime)`, so touching one byte of an
index re-transfers every shard to every node holding it. A repack writes a **new** directory.

**`soma` is copied beside the node and control binaries at build time.** Both resolve it by name —
the node spawns it, control runs `soma plan`, `soma verify` and `soma conform` during admission —
and neither directory is on the other's PATH. The copy is what makes "soma ships with the node" true
rather than merely documented.

## The siblings

Five projects live beside this one and are **clients** of its REST surface. None is in this
repository, and a session started in one of them does not load this file. `../CLAUDE.md` names only
the three that sit directly beside it; these are all of them — **and two are no longer live**, which
is worth knowing before you treat one as a constraint.

| Client | What | Standing |
|---|---|---|
| `../Mafia-Machine/` | LLM Mafia game, Python | **live.** git; has its own `CLAUDE.md` |
| `../Zenith-Crown/` | murder-mystery game (movement, murder, investigation, trial) with a web viewer, Python | **live.** *Not* a git repository |
| `../Patriot-Skyline/` | planetary-defense bunker simulation; Python owns the state, a Three.js client renders it | **live.** *Not* a git repository |
| `../Yappy-Agent/` | SvelteKit web UI, Node/TypeScript | **deprecated**, and has been for several Mantic-Mind versions. Not a git repository; has its own `CLAUDE.md` |
| `../Code-Aid/` | terminal coding assistant, Python | **discontinued** — it never worked as intended. git; reached the `:9091` listener |

The three games are not versioned, so a change there leaves no diff to review. Snapshot one before
editing it.

**Do not spend effort keeping the last two working.** Both were carried through the users overhaul
and are green, and that is where it stops: a change that would cost real work to carry into Yappy or
Code-Aid should be made without them, with the breakage noted rather than fixed. What replaces Yappy
is undecided — an HTML interface served by Mantic-Mind itself is one option, and much of what both
projects did may instead be delegated to **Hermes Agent** (NousResearch), a third-party consumer
this repository does not own.

That last possibility is what the `:9091` OpenAI-compatible listener is for. An agent framework
arriving from outside speaks chat completions and owns its own conversation; it cannot use the
user-scoped half of the surface — conversations, memories, voice, curation — because that half is
shaped for the games, which want the server to remember. So keep §11 of `docs/external-api.md` true:
`tools`, history and sampling pass through **unchanged**. That property is the whole of what an
outside agent depends on, and nothing in the games would notice if it broke.

If you change a `/v1/*` route's shape, you have changed the live clients' contract.
`docs/external-api.md` is the authority on that surface, and `tools/ci/check_api_docs.py` asserts
that every route it describes is one that exists — a documented route that answers 404 is a promise, and one of those
survived from the day the surface was written until someone read the docs and called it.
