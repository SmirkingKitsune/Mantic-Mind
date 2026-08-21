// Soma — G4 speculative scheduling invariants.
//
// A scripted draft backend supplies every possible acceptance depth.  The
// target scheduler must still emit exactly the ordinary autoregressive stream:
// accepted tokens, the first correction, and the all-accepted bonus token are
// sampled by the same target sampler and committed through the same KV cache.

#include "soma/f32_model.hpp"
#include "soma/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Script {
    std::vector<soma::TokenId> target;
    std::uint32_t matching = 0;
    bool proposed = false;
};

struct State {};

struct StreamCapture {
    std::vector<float> values;
};

void capture_streams(void* opaque,
                     std::uint32_t layer,
                     const char* point,
                     const float* data,
                     std::size_t n) {
    if (layer != 0 || std::strcmp(point, "hc_ffn_streams") != 0) return;
    auto* capture = static_cast<StreamCapture*>(opaque);
    capture->values.assign(data, data + n);
}

void delete_script(void* p) { delete static_cast<Script*>(p); }
void delete_state(void* p) { delete static_cast<State*>(p); }

soma::StatusCode open_sequence(const soma::F32Model&,
                               std::uint32_t,
                               soma::ArchLayerPayload& out) noexcept {
    out.adopt(new State{}, delete_state);
    return soma::StatusCode::Ok;
}

soma::StatusCode observe_target(const soma::F32Model&,
                                const soma::ArchLayerPayload&,
                                soma::ArchLayerPayload&,
                                const soma::HiddenStateTaps&,
                                std::uint32_t,
                                std::uint32_t,
                                std::uint32_t) noexcept {
    return soma::StatusCode::Ok;
}

soma::StatusCode propose(const soma::F32Model& model,
                         const soma::ArchLayerPayload& payload,
                         soma::ArchLayerPayload&,
                         soma::TokenId,
                         std::uint32_t cap,
                         float,
                         soma::SpeculativeProposal& out) noexcept {
    auto* script = payload.as<Script>();
    if (script == nullptr || script->proposed || script->target.size() < cap) {
        return soma::StatusCode::Ok;
    }
    script->proposed = true;
    out.tokens.assign(script->target.begin(), script->target.begin() + cap);
    if (script->matching < cap) {
        auto& mismatch = out.tokens[script->matching];
        mismatch = static_cast<soma::TokenId>((mismatch + 1) % model.vocab());
    }
    out.confidence.assign(out.tokens.size(), 1.0f);
    return soma::StatusCode::Ok;
}

const soma::SpeculativeBackend kScripted{
    "scripted",
    nullptr,
    nullptr,
    open_sequence,
    observe_target,
    propose,
    nullptr,
    nullptr,
};

struct Run {
    std::vector<soma::TokenId> tokens;
    std::vector<bool> last;
    soma::SchedulerStats stats{};
};

bool append_rows(soma::F32Model& model,
                 soma::KvCache& kv,
                 std::span<const soma::TokenId> tokens,
                 soma::F32Workspace& workspace,
                 bool tentative,
                 std::uint32_t transaction_row = 0) {
    std::vector<soma::KvRow> rows(tokens.size());
    for (std::uint32_t i = 0; i < rows.size(); ++i) {
        auto& row = rows[i];
        row.opaque_base = kv.opaque_data();
        row.opaque_bytes = kv.opaque_size();
        row.max_ctx = kv.capacity();
        row.k_base = kv.is_opaque() ? nullptr : kv.k_at(0, 0);
        row.v_base = kv.is_opaque() ? nullptr : kv.v_at(0, 0);
        row.k_stride = kv.k_stride();
        row.v_stride = kv.v_stride();
        row.k_hkv = kv.k_hkv();
        row.v_hkv = kv.v_hkv();
        // Tentative length advances only at prefix commit, so separately issued
        // verification rows use their transaction ordinal as the live offset.
        row.pos = kv.length() + (tentative ? transaction_row : 0u) + i;
        row.len = row.pos + 1;
        row.transaction = tentative ? kv.transaction() : nullptr;
        row.transaction_row = transaction_row + i;
    }
    std::vector<float> logits;
    const auto st = soma::forward_step_f32(model, tokens, rows, workspace, logits);
    if (!st.ok()) {
        std::cerr << "cache forward: " << st.message() << '\n';
        return false;
    }
    if (!tentative) kv.commit(static_cast<std::uint32_t>(tokens.size()));
    return true;
}

bool generate(soma::F32Model& model, bool speculative, Run& out) {
    soma::SchedulerConfig cfg;
    cfg.ctx_size = 128;
    cfg.kv_slots = 1;
    cfg.max_batch = 1;
    cfg.enable_speculation = speculative;
    cfg.speculative_tokens = 7;

    soma::Scheduler scheduler;
    const auto st = scheduler.open_f32(model, nullptr, cfg);
    if (!st.ok()) {
        std::cerr << "open: " << st.message() << '\n';
        return false;
    }
    scheduler.set_token_callback([&](soma::SeqId, soma::TokenId token, bool last) {
        out.tokens.push_back(token);
        out.last.push_back(last);
    });

    soma::SeqRequest request;
    request.prompt = {3, 11, 29, 7};
    request.max_tokens = 16;
    request.sampler.temperature = 0.0f;
    soma::SeqId id = 0;
    soma::AdmitRejection reason{};
    if (const auto admit = scheduler.admit(std::move(request), id, reason); !admit.ok()) {
        std::cerr << "admit: " << admit.message() << '\n';
        return false;
    }
    for (std::uint32_t steps = 0; !scheduler.idle() && steps < 128; ++steps) {
        if (const auto step = scheduler.step(); !step.ok()) {
            std::cerr << "step: " << step.message() << '\n';
            return false;
        }
    }
    out.stats = scheduler.stats();
    return scheduler.idle();
}

} // namespace

int main(int argc, char** argv) {
    const fs::path model_dir = argc > 1
                                   ? fs::path(argv[1])
                                   : fs::path("tests/fixtures/tiny/DeepSeek-V4-Pro-0813");
    soma::F32Model model;
    if (const auto st = soma::load_f32_model(model_dir.string(), model); !st.ok()) {
        std::cerr << "load: " << st.message() << '\n';
        return 2;
    }

    // DSpark's official conditioning tensor is h.mean(dim=2), i.e. the mean of
    // all four post-block hyper-connection streams. The generic tap hook must
    // export that representation rather than the core's stream-0 working row.
    {
        soma::KvCache kv;
        if (!kv.open(model.arch, 8).ok()) return 1;
        soma::KvRow row;
        row.opaque_base = kv.opaque_data();
        row.opaque_bytes = kv.opaque_size();
        row.max_ctx = kv.capacity();
        row.pos = 0;
        row.len = 1;
        soma::F32Workspace ws;
        StreamCapture capture;
        ws.sink.emit = capture_streams;
        ws.sink.ctx = &capture;
        const soma::LayerIndex layer = 0;
        soma::HiddenStateTaps taps{{&layer, 1}};
        std::vector<float> logits;
        const soma::TokenId token = 3;
        if (!soma::forward_step_f32(model, {&token, 1}, {&row, 1}, ws, logits, &taps).ok())
            return 1;
        const auto d = model.d_model(), hc = model.arch.hyper_connections.multiplier;
        if (capture.values.size() != static_cast<std::size_t>(d) * hc ||
            taps.values.size() != d) return 1;
        for (std::uint32_t k = 0; k < d; ++k) {
            double sum = 0.0;
            for (std::uint32_t h = 0; h < hc; ++h)
                sum += capture.values[static_cast<std::size_t>(h) * d + k];
            if (std::abs(taps.values[k] - static_cast<float>(sum / hc)) > 1e-7f) {
                std::cerr << "target hidden tap is not the HC stream mean\n";
                return 1;
            }
        }
    }

    model.speculative_backend = nullptr;
    Run baseline;
    if (!generate(model, false, baseline) || baseline.tokens.size() < 8) return 1;

    auto* script = new Script{};
    // The final prefill row emits baseline[0].  The first speculative proposal
    // therefore begins with the token after that already-live anchor.
    script->target.assign(baseline.tokens.begin() + 1, baseline.tokens.end());
    model.speculative_payload.adopt(script, delete_script);
    model.speculative_backend = &kScripted;

    for (std::uint32_t accepted = 0; accepted <= 7; ++accepted) {
        script->matching = accepted;
        script->proposed = false;
        Run actual;
        if (!generate(model, true, actual)) return 1;
        if (actual.tokens != baseline.tokens) {
            std::cerr << "acceptance depth " << accepted << " changed target output\n";
            return 1;
        }
        if (actual.stats.speculative_draft_tokens != 7 ||
            actual.stats.speculative_accepted_tokens != accepted ||
            actual.stats.speculative_verifications != 1) {
            std::cerr << "acceptance depth " << accepted << " reported wrong counters\n";
            return 1;
        }
        if (actual.last.size() != actual.tokens.size() || actual.last.empty() ||
            !actual.last.back() ||
            std::count(actual.last.begin(), actual.last.end(), true) != 1) {
            std::cerr << "acceptance depth " << accepted << " reported wrong finish marker\n";
            return 1;
        }
    }

    // The V4 cache combines a circular window with ratio-4/ratio-128
    // compressor histories, indexer history, and carry state.  Verify the byte
    // reverse journal at the boundary where all of those can mutate: writing
    // positions 127..134, then retaining every possible prefix, must equal a
    // cache that only ever saw that prefix.
    std::vector<soma::TokenId> cache_tokens(135);
    for (std::uint32_t i = 0; i < cache_tokens.size(); ++i)
        cache_tokens[i] = static_cast<soma::TokenId>((i * 17 + 3) % model.vocab());

    for (std::uint32_t keep = 0; keep <= 8; ++keep) {
        soma::KvCache actual;
        if (!actual.open(model.arch, 160).ok()) return 1;
        soma::F32Workspace actual_ws;
        actual_ws.reserve(model.arch, 135);
        const auto base = std::span<const soma::TokenId>(cache_tokens).first(127);
        if (!append_rows(model, actual, base, actual_ws, false)) return 1;
        if (!actual.begin_tentative(8).ok()) return 1;
        std::vector<std::byte> accepted_snapshot;
        if (keep == 0) {
            accepted_snapshot.assign(actual.opaque_data(),
                                     actual.opaque_data() + actual.opaque_size());
        }
        for (std::uint32_t row = 0; row < 8; ++row) {
            if (!append_rows(model,
                             actual,
                             std::span<const soma::TokenId>(cache_tokens).subspan(127 + row, 1),
                             actual_ws,
                             true,
                             row)) return 1;
            if (row + 1 == keep) {
                accepted_snapshot.assign(actual.opaque_data(),
                                         actual.opaque_data() + actual.opaque_size());
            }
        }
        if (!actual.commit_tentative_prefix(keep).ok()) return 1;
        const bool snapshot_restored = accepted_snapshot.size() == actual.opaque_size() &&
            std::memcmp(actual.opaque_data(), accepted_snapshot.data(), actual.opaque_size()) == 0;
        if (!snapshot_restored || actual.length() != 127 + keep) {
            std::size_t first = 0, changed = 0;
            while (first < actual.opaque_size() &&
                   actual.opaque_data()[first] == accepted_snapshot[first]) ++first;
            for (std::size_t i = first; i < actual.opaque_size(); ++i)
                changed += actual.opaque_data()[i] != accepted_snapshot[i];
            std::cerr << "KV rollback prefix " << keep
                      << " differs at compression/window boundary (first byte " << first
                      << ", changed " << changed << ", snapshot "
                      << (snapshot_restored ? "restored" : "DIFFERS") << ")\n";
            return 1;
        }
    }

    std::cout << "speculative_g4: OK (acceptance depths 0..7, KV prefixes 0..8)\n";
    return 0;
}
