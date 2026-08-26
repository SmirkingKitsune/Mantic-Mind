// Soma — G3: supplied embeddings, and the prefix check that survives them.
//
// The claim under test:
//
//   A CACHE MUST NOT ATTACH TO A PROMPT WHOSE SUPPLIED EMBEDDINGS DIFFER FROM
//   THE ONES THAT BUILT IT — EVEN WHEN EVERY TOKEN ID MATCHES.
//
// v2 made a restore checkable by carrying the token ids the cached positions
// hold. That check rests on token ids DETERMINING the hidden state, which stops
// being true the moment a position can be built from a supplied row instead of
// the embedding table. Two different images occupy the same placeholder ids and
// produce byte-identical token arrays: the prefix check passes, the cache is
// attached, and the model answers fluently about a picture nobody sent. That is
// the exact failure the token array exists to prevent, one layer beneath it and
// invisible to it.
//
// There is no vision tower here, and that is deliberate. Every property below is
// a property of the PLUMBING — the forward, the digest, the checkpoint, the two
// prefix gates — and each is provable with synthetic rows. A tower would add a
// large dependency and prove none of it.
//
// Usage: media_prefix_g3 <fixtures_root> [fixture]

#include "soma/kv_checkpoint.hpp"
#include "soma/media_digest.hpp"
#include "soma/quant_format.hpp"
#include "soma/scheduler.hpp"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(56) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

/// A deterministic row that is not any embedding-table row.
std::vector<float> synthetic_row(std::uint32_t d, float seed) {
    std::vector<float> r(d);
    for (std::uint32_t i = 0; i < d; ++i)
        r[i] = seed + 0.001f * static_cast<float>(i % 97);
    return r;
}

/// Fill one single-row KV batch, matching what the scheduler builds.
void one_row(soma::KvCache& kv, soma::KvRow& r) {
    if (kv.is_opaque()) {
        r.opaque_base = kv.opaque_data();
        r.opaque_bytes = kv.opaque_size();
        r.max_ctx = kv.capacity();
    } else {
        r.k_base = kv.k_at(0, 0);
        r.v_base = kv.v_at(0, 0);
    }
    r.k_stride = kv.k_stride();
    r.v_stride = kv.v_stride();
    r.k_hkv = kv.k_hkv();
    r.v_hkv = kv.v_hkv();
    r.pos = 0;
    r.len = 1;
}

/// One single-row forward, optionally with the embedding lookup overridden.
bool one_row_logits(const soma::F32Model& model,
                    soma::TokenId tok,
                    const float* override_row,
                    std::vector<float>& out) {
    soma::KvCache kv;
    if (!kv.open(model.arch, 8).ok()) return false;
    soma::F32Workspace ws;

    const std::vector<soma::TokenId> tokens{tok};
    soma::KvRow r{};
    one_row(kv, r);
    const std::vector<soma::KvRow> rows{r};

    std::vector<soma::EmbeddingOverride> ov;
    if (override_row != nullptr) ov.push_back(soma::EmbeddingOverride{0, override_row});
    return soma::forward_step_f32(model, tokens, rows, ws, out, nullptr, ov).ok();
}

/// Drive a scheduler until every sequence retires. Bounded, so a scheduler bug
/// fails the test rather than hanging the suite.
void drain(soma::Scheduler& sched) {
    for (std::uint32_t i = 0; i < 4000 && !sched.idle(); ++i) {
        if (auto st = sched.step(); !st.ok()) {
            std::cerr << "step failed: " << st.message() << "\n";
            return;
        }
    }
}

/// Hand-written checkpoint headers, for the versions this build must still READ.
///
/// Written by hand on purpose: the header crosses a process boundary, so the
/// decoder deserves a check that does not share its encoder. A helper shared with
/// the writer would agree with itself whatever the layout became.
std::vector<unsigned char> handmade_header(std::uint32_t version,
                                           std::uint32_t flags,
                                           bool write_flags_word,
                                           std::uint64_t auxiliary_bytes,
                                           bool write_auxiliary) {
    std::vector<unsigned char> b;
    const auto u32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i)
            b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFF));
    };
    const auto u64 = [&](std::uint64_t v) {
        for (int i = 0; i < 8; ++i)
            b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFF));
    };
    const std::string magic = "SOMAKV01";
    for (const char c : magic)
        b.push_back(static_cast<unsigned char>(c));
    u32(version);
    const std::string arch_hash = "arch-hash-abc";
    u32(static_cast<std::uint32_t>(arch_hash.size()));
    for (const char c : arch_hash)
        b.push_back(static_cast<unsigned char>(c));
    u32(7);    // format_id
    u32(3);    // length_tokens
    u32(64);   // d_model
    u64(4096); // payload_bytes
    u64(1700000000000);
    u64(0xabcdef); // rng_state
    u32(2);        // n_emitted
    if (write_flags_word) u32(flags);
    if (write_auxiliary) u64(auxiliary_bytes);
    return b;
}

soma::Status parse_bytes(const std::vector<unsigned char>& b, soma::KvCheckpointHeader& out) {
    return soma::parse_kv_checkpoint_header(reinterpret_cast<const std::byte*>(b.data()),
                                            b.size(),
                                            out);
}

} // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures");
    const std::string name = (argc > 2) ? argv[2] : "Qwen3-30B-A3B";

    soma::F32Model model;
    if (auto st = soma::load_f32_model((root / "tiny" / name).string(), model); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }
    const auto d = model.arch.topology.d_model;
    std::cout << "== " << name << "  d_model=" << d << "\n";

    // ── the digest is arithmetic ─────────────────────────────────────────────
    //
    // Everything downstream rests on one property: what a save FOLDED IN and
    // what a restore RECOMPUTES are the same value. Checked directly, because
    // every other test here would still pass if the two drifted — they would
    // simply never attach a cache again, and "warm reopen quietly stopped
    // working" is not a failure anything reports.
    std::cout << "\n-- digest --\n";
    {
        const auto row0 = synthetic_row(d, 0.25f);
        const auto row1 = synthetic_row(d, -1.5f);

        const std::vector<std::uint32_t> positions{2, 5};
        std::vector<float> values;
        values.insert(values.end(), row0.begin(), row0.end());
        values.insert(values.end(), row1.begin(), row1.end());

        soma::MediaDigest incremental{};
        check(incremental.empty(), "a fresh digest is empty");
        check(soma::media_digest_prefix(positions, values, d, 0).empty(),
              "a zero-length prefix is empty");

        soma::media_digest_fold(incremental, 2, std::span<const float>(row0));
        const auto one = soma::media_digest_prefix(positions, values, d, 3);
        check(incremental == one, "fold(1) == prefix(past the 1st)", one.hex().substr(0, 16));

        soma::media_digest_fold(incremental, 5, std::span<const float>(row1));
        const auto two = soma::media_digest_prefix(positions, values, d, 6);
        check(incremental == two, "fold(2) == prefix(past the 2nd)", two.hex().substr(0, 16));
        check(!(one == two), "a longer prefix is a different digest");
        check(!two.empty(), "a digest over real rows is not empty");

        // One float. The whole point is that a change nothing else can see is
        // visible here.
        auto tweaked = values;
        tweaked[d + 3] += 1e-3f;
        check(!(soma::media_digest_prefix(positions, tweaked, d, 6) == two),
              "one changed float changes the digest");

        // The same rows at different positions are a different context.
        const std::vector<std::uint32_t> moved{2, 6};
        check(!(soma::media_digest_prefix(moved, values, d, 7) == two),
              "the same rows at a moved position differ");

        // Order matters, or "the digest of a prefix" means nothing.
        soma::MediaDigest reversed{};
        soma::media_digest_fold(reversed, 5, std::span<const float>(row1));
        soma::media_digest_fold(reversed, 2, std::span<const float>(row0));
        check(!(reversed == two), "folding in the other order differs");
    }

    // ── the shape rules are enforced at the boundary ─────────────────────────
    std::cout << "\n-- validation --\n";
    {
        soma::PromptEmbeddings e;
        check(soma::validate_prompt_embeddings(e, d, 4).ok(), "no embeddings is valid");

        e.positions = {1};
        check(!soma::validate_prompt_embeddings(e, d, 4).ok(), "positions without values refused");

        e.values = synthetic_row(d, 1.0f);
        check(soma::validate_prompt_embeddings(e, d, 4).ok(), "one position, one row is valid");

        e.positions = {1, 2};
        check(!soma::validate_prompt_embeddings(e, d, 4).ok(), "a missing row is refused");

        auto two_rows = e.values;
        two_rows.insert(two_rows.end(), e.values.begin(), e.values.end());
        e.values = two_rows;
        check(soma::validate_prompt_embeddings(e, d, 4).ok(), "two positions, two rows is valid");

        e.positions = {2, 1};
        check(!soma::validate_prompt_embeddings(e, d, 4).ok(), "descending positions refused");
        e.positions = {1, 1};
        check(!soma::validate_prompt_embeddings(e, d, 4).ok(), "duplicate positions refused");
        e.positions = {1, 9};
        check(!soma::validate_prompt_embeddings(e, d, 4).ok(), "a position past the prompt refused");
    }

    // ── the override REPLACES the lookup ─────────────────────────────────────
    //
    // The strongest statement available without a tower: run token A while
    // supplying token B's OWN embedding row, and demand the logits equal a plain
    // run of token B. Anything weaker — "the output changed" — would also pass if
    // the override were added to the gathered row, or scaled, or landed on the
    // wrong position.
    //
    // The equality holds because nothing downstream of the embedding reads the
    // token id on this family. That is not luck: it is why an overridden position
    // still carries a REAL id rather than a sentinel. A family whose begin-forward
    // hook derived state from ids would legitimately differ here, and would be
    // telling the truth.
    std::cout << "\n-- the forward --\n";
    {
        const soma::TokenId a = 3, b = 11;
        std::vector<float> row_b(d);
        const auto src = soma::row_block(model.embed, b, 1);
        const bool got_row = !src.empty() && soma::dequantize(src, std::span<float>(row_b)).ok();
        check(got_row, "token B's embedding row is readable");

        std::vector<float> plain_a, plain_b, a_with_b, a_with_synth;
        check(one_row_logits(model, a, nullptr, plain_a), "plain forward of token A");
        check(one_row_logits(model, b, nullptr, plain_b), "plain forward of token B");
        check(one_row_logits(model, a, row_b.data(), a_with_b), "forward of A overridden with B");

        check(!plain_a.empty() && plain_a.size() == plain_b.size(), "logits are the same shape");
        check(a_with_b == plain_b, "A overridden with B's row IS B, bit for bit");
        check(!(a_with_b == plain_a), "and is not A");

        const auto synth = synthetic_row(d, 0.75f);
        check(one_row_logits(model, a, synth.data(), a_with_synth), "forward with a synthetic row");
        check(!(a_with_synth == plain_a) && !(a_with_synth == plain_b),
              "a row from no token is neither");

        // A row index outside the batch is a caller bug, and one that would
        // otherwise write d_model floats past the end of the workspace.
        soma::KvCache kv;
        soma::F32Workspace ws;
        std::vector<float> logits;
        if (kv.open(model.arch, 8).ok()) {
            const std::vector<soma::TokenId> tokens{a};
            soma::KvRow r{};
            one_row(kv, r);
            const std::vector<soma::KvRow> rows{r};
            const std::vector<soma::EmbeddingOverride> bad{{7, synth.data()}};
            check(!soma::forward_step_f32(model, tokens, rows, ws, logits, nullptr, bad).ok(),
                  "an out-of-range override row is refused");
            const std::vector<soma::EmbeddingOverride> null_row{{0, nullptr}};
            check(!soma::forward_step_f32(model, tokens, rows, ws, logits, nullptr, null_row).ok(),
                  "an override with no values is refused");
        }
    }

    // ── the checkpoint carries it ────────────────────────────────────────────
    std::cout << "\n-- the format --\n";
    const auto dir = fs::temp_directory_path() / ("soma_media_g3_" + name);
    std::error_code ec;
    fs::remove_all(dir, ec);
    soma::KvCheckpointStore store;
    if (auto st = store.open(dir.string(), model.arch); !st.ok()) {
        std::cerr << "store open failed: " << st.message() << "\n";
        return 2;
    }
    {
        soma::KvCache kv;
        if (!kv.open(model.arch, 8).ok()) return 2;

        soma::MediaDigest digest{};
        const auto row = synthetic_row(d, 3.5f);
        soma::media_digest_fold(digest, 1, std::span<const float>(row));

        soma::SeqPersistState state;
        state.media = digest;
        check(store.save("with-media", kv, state).ok(), "a checkpoint with media saves");

        soma::KvCheckpointHeader h;
        check(store.stat("with-media", h).ok() && h.version == soma::kKvCheckpointVersion &&
                  (h.flags & soma::kKvFlagMedia) != 0 &&
                  (h.flags & soma::kKvFlagAuxiliary) == 0,
              "the header sets the media flag alone");
        check(h.media_digest == digest, "the header carries the digest verbatim");

        soma::KvCache into;
        soma::SeqPersistState loaded;
        if (into.open(model.arch, 8).ok()) {
            check(store.load("with-media", into, loaded).ok() && loaded.media == digest,
                  "the digest round-trips through load()");
        }

        // Both optional fields at once. The flags word exists so these stay
        // independent; a version-per-variant scheme is what made them a menu.
        soma::SeqPersistState both;
        both.media = digest;
        both.auxiliary = {std::byte{0x01}, std::byte{0x02}};
        check(store.save("both", kv, both).ok(), "media and auxiliary save together");
        soma::KvCheckpointHeader hb;
        check(store.stat("both", hb).ok() && (hb.flags & soma::kKvFlagMedia) != 0 &&
                  (hb.flags & soma::kKvFlagAuxiliary) != 0 &&
                  hb.auxiliary_bytes == both.auxiliary.size() && hb.media_digest == digest,
              "both flags set, both extents right");
        soma::KvCache into2;
        soma::SeqPersistState loaded2;
        if (into2.open(model.arch, 8).ok()) {
            check(store.load("both", into2, loaded2).ok() && loaded2.media == digest &&
                      loaded2.auxiliary == both.auxiliary,
                  "both round-trip without treading on each other");
        }
        (void)store.remove("with-media");
        (void)store.remove("both");
    }

    // ── versions this build must still read ──────────────────────────────────
    {
        soma::KvCheckpointHeader h3;
        const auto v3 = handmade_header(soma::kKvCheckpointVersionSampler, 0, false, 0, false);
        check(parse_bytes(v3, h3).ok() && h3.flags == 0 && h3.auxiliary_bytes == 0 &&
                  h3.media_digest.empty() && h3.tokens_at == v3.size(),
              "a v3 header parses, with synthesised flags");

        soma::KvCheckpointHeader h4;
        const auto v4 = handmade_header(soma::kKvCheckpointVersionSpeculative, 0, false, 24, true);
        check(parse_bytes(v4, h4).ok() && h4.flags == soma::kKvFlagAuxiliary &&
                  h4.auxiliary_bytes == 24 && h4.tokens_at == v4.size(),
              "a v4 header parses into the auxiliary flag");

        // Every field after the flags word sits at an offset derived from it, so
        // an unrecognised bit is not a field to skip — it is the rest of the file.
        soma::KvCheckpointHeader h5;
        const auto future = handmade_header(soma::kKvCheckpointVersion, 0x80u, true, 0, false);
        const auto st = parse_bytes(future, h5);
        check(!st.ok() && st.code() == soma::StatusCode::VersionMismatch,
              "an unknown header flag is refused, not skipped");

        // Truncation must name truncation. Before the version gate moved ahead of
        // the variable fields, a short file reported "unsupported version 0".
        soma::KvCheckpointHeader h6;
        const std::string magic = "SOMAKV01";
        std::vector<unsigned char> stub(magic.begin(), magic.end());
        stub.push_back(5);
        const auto trunc = parse_bytes(stub, h6);
        check(!trunc.ok() && trunc.code() == soma::StatusCode::InvalidArgument,
              "a truncated header reports truncation");
    }

    // ── the gate, through the scheduler ──────────────────────────────────────
    //
    // The whole reason for the rest of this file.
    std::cout << "\n-- warm reopen --\n";
    {
        soma::SchedulerConfig cfg;
        cfg.kv_slots = 8;
        cfg.ctx_size = 64;
        soma::Scheduler sched;
        if (auto st = sched.open_f32(model, nullptr, cfg, &store); !st.ok()) {
            std::cerr << "scheduler open failed: " << st.message() << "\n";
            return 2;
        }

        const std::vector<soma::TokenId> prompt{3, 11, 29, 7};
        const auto row_a = synthetic_row(d, 0.5f);
        auto row_b = row_a;
        row_b[2] += 0.125f; // one float, one position — a different picture

        const auto make = [&](const std::vector<float>& row) {
            soma::PromptEmbeddings e;
            e.positions = {1};
            e.values = row;
            return e;
        };

        soma::SeqRequest first;
        first.prompt = prompt;
        first.embeddings = make(row_a);
        first.max_tokens = 1;
        soma::SeqId id1 = 0;
        soma::AdmitRejection why{};
        check(sched.admit(std::move(first), id1, why).ok(), "a sequence with supplied rows admits");
        drain(sched);
        check(sched.checkpoint(id1, "warm").ok(), "it checkpoints");

        soma::KvCheckpointHeader warm;
        check(store.stat("warm", warm).ok() && (warm.flags & soma::kKvFlagMedia) != 0 &&
                  !warm.media_digest.empty(),
              "the checkpoint records a media digest");

        std::vector<soma::TokenId> history;
        check(sched.sequence_tokens(id1, history).ok() && history == prompt,
              "the cached positions are the prompt");
        auto longer = history;
        longer.push_back(19);

        // Same rows: the cache is the prefix, and attaches.
        soma::SeqRequest same;
        same.prompt = longer;
        same.embeddings = make(row_a);
        same.max_tokens = 1;
        same.resume_key = "warm";
        soma::SeqId id2 = 0;
        check(sched.admit(std::move(same), id2, why).ok(), "a matching request admits");
        std::vector<soma::TokenId> attached;
        check(sched.sequence_tokens(id2, attached).ok() && attached == history,
              "matching rows attach the cache");

        // One float different, every token identical. THIS is the gate.
        soma::SeqRequest changed;
        changed.prompt = longer;
        changed.embeddings = make(row_b);
        changed.max_tokens = 1;
        changed.resume_key = "warm";
        soma::SeqId id3 = 0;
        check(sched.admit(std::move(changed), id3, why).ok(), "a changed-media request admits");
        std::vector<soma::TokenId> cold;
        check(sched.sequence_tokens(id3, cold).ok() && cold.empty(),
              "changed rows COLD START despite identical tokens");

        // And the same for a caller that forgot to resend them at all — which is
        // the shape the mistake actually takes.
        soma::SeqRequest forgot;
        forgot.prompt = longer;
        forgot.max_tokens = 1;
        forgot.resume_key = "warm";
        soma::SeqId id4 = 0;
        check(sched.admit(std::move(forgot), id4, why).ok(), "a request with no rows admits");
        std::vector<soma::TokenId> cold2;
        check(sched.sequence_tokens(id4, cold2).ok() && cold2.empty(),
              "omitting the rows cold starts too");

        // extend() carries the same gate, and REFUSES rather than cold starting:
        // it is attaching to a live session, so there is no cold path to fall to.
        //
        // Built from the sequence's OWN history rather than from `longer`: id2 has
        // since prefilled the extra token, so its cache now covers more than the
        // prompt it was admitted with, and a stale prompt would fail the ordinary
        // "adds no tokens" check before reaching the one under test.
        drain(sched);
        std::vector<soma::TokenId> so_far;
        check(sched.sequence_tokens(id2, so_far).ok() && so_far.size() > history.size(),
              "the session cached its extension");
        auto next_turn = so_far;
        next_turn.push_back(23);

        const auto changed_extend = sched.extend(id2, next_turn, 1, make(row_b));
        check(!changed_extend.ok() && changed_extend.code() == soma::StatusCode::ArchMismatch,
              "extend refuses changed rows");
        check(sched.extend(id2, next_turn, 1, make(row_a)).ok(), "extend accepts matching rows");

        // A preempt/resume round trip carries the digest through the same file.
        drain(sched);
        check(sched.preempt(id2).ok(), "the sequence preempts");
        check(sched.resume(id2).ok(), "and resumes with its digest intact");
    }

    fs::remove_all(dir, ec);
    std::cout << "\n" << (g_failures == 0 ? "PASS" : "FAIL") << "  " << g_failures
              << " failure(s)\n";
    return g_failures == 0 ? 0 : 1;
}
