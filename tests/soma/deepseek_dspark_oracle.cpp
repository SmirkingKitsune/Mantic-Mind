// Deterministic DSpark conformance against the pinned DeepSeek model.py oracle.

#include "soma/f32_model.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Key {
    std::uint32_t layer = 0;
    std::string point;
    auto operator<=>(const Key&) const = default;
};

using Records = std::map<Key, std::vector<float>>;

void capture(void* opaque, std::uint32_t layer, const char* point,
             const float* data, std::size_t n) {
    auto& records = *static_cast<Records*>(opaque);
    records[{layer, point}] = {data, data + n};
}

bool read_u32(std::ifstream& in, std::uint32_t& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

bool read_records(const fs::path& path, Records& out) {
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, "SOMAACT1", 8) != 0)
        return false;
    std::uint32_t count = 0;
    if (!read_u32(in, count)) return false;
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t layer = 0, name_len = 0, n = 0;
        if (!read_u32(in, layer) || !read_u32(in, name_len)) return false;
        std::string name(name_len, '\0');
        if (!in.read(name.data(), name_len) || !read_u32(in, n)) return false;
        std::vector<float> values(n);
        if (!in.read(reinterpret_cast<char*>(values.data()),
                     static_cast<std::streamsize>(n * sizeof(float)))) return false;
        out[{layer, std::move(name)}] = std::move(values);
    }
    return true;
}

std::vector<float> layer_major(const std::vector<float>& row_major,
                               std::uint32_t rows, std::uint32_t layers,
                               std::uint32_t width) {
    std::vector<float> out(row_major.size());
    for (std::uint32_t layer = 0; layer < layers; ++layer)
        for (std::uint32_t row = 0; row < rows; ++row)
            std::copy_n(row_major.data() + (static_cast<std::size_t>(row) * layers + layer) * width,
                        width,
                        out.data() + (static_cast<std::size_t>(layer) * rows + row) * width);
    return out;
}

struct Delta {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

Delta delta(std::span<const float> a, std::span<const float> b) {
    Delta out;
    if (a.size() != b.size() || a.empty()) {
        out.max_abs = out.mean_abs = INFINITY;
        return out;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        out.max_abs = std::max(out.max_abs, d);
        sum += d;
    }
    out.mean_abs = static_cast<float>(sum / a.size());
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: deepseek_dspark_oracle <container> <oracle-source>\n";
        return 2;
    }
    const fs::path container(argv[1]), oracle_dir(argv[2]);
    json ref;
    {
        std::ifstream in(oracle_dir / "dspark_reference.json");
        if (!in || !(in >> ref)) {
            std::cerr << "cannot read DSpark reference JSON\n";
            return 2;
        }
    }
    if (ref.value("source_revision", "") !=
            "72e1d3230f6c080a530b0a1d46f8eb4602340597" ||
        ref.value("model_py_sha256", "") !=
            "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f" ||
        ref.value("kernel_py_sha256", "") !=
            "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2") {
        std::cerr << "oracle is not pinned to the supported revision\n";
        return 2;
    }

    soma::F32Model model;
    if (const auto st = soma::load_f32_model(container.string(), model); !st.ok()) {
        std::cerr << "load: " << st.message() << '\n';
        return 2;
    }
    const auto* backend = soma::resolve_speculative_backend(model.arch);
    if (backend == nullptr || backend->bind_model == nullptr || backend->start_runtime == nullptr) {
        std::cerr << "DSpark backend was not resolved\n";
        return 2;
    }
    if (backend->bind_model(model, container.string()) != soma::StatusCode::Ok ||
        backend->start_runtime(model, container.string(), 8 * 1024 * 1024) != soma::StatusCode::Ok) {
        std::cerr << "DSpark payload/runtime initialization failed\n";
        return 2;
    }
    soma::ArchLayerPayload state;
    if (backend->open_sequence(model, 128, state) != soma::StatusCode::Ok) return 2;

    const auto ids_json = ref.at("target_layer_ids").get<std::vector<std::uint32_t>>();
    std::vector<soma::LayerIndex> layer_ids(ids_json.begin(), ids_json.end());
    const auto prompt_rows = ref.at("prompt_length").get<std::uint32_t>();
    const auto d = model.d_model(), n_layers = static_cast<std::uint32_t>(layer_ids.size());

    soma::HiddenStateTaps prompt;
    prompt.layers = layer_ids;
    prompt.n_rows = prompt_rows;
    prompt.d_model = d;
    prompt.values = layer_major(ref.at("prompt_hidden").get<std::vector<float>>(),
                                prompt_rows, n_layers, d);
    if (backend->observe_target(model, model.speculative_payload, state, prompt,
                                0, prompt_rows, 0) != soma::StatusCode::Ok) {
        std::cerr << "DSpark prompt observation failed\n";
        return 2;
    }

    soma::HiddenStateTaps decode;
    decode.layers = layer_ids;
    decode.n_rows = 1;
    decode.d_model = d;
    decode.values = layer_major(ref.at("decode_hidden").get<std::vector<float>>(),
                                1, n_layers, d);
    if (backend->observe_target(model, model.speculative_payload, state, decode,
                                0, 1, prompt_rows) != soma::StatusCode::Ok) {
        std::cerr << "DSpark decode observation failed\n";
        return 2;
    }

    Records actual;
    soma::SpeculativeProposal proposal;
    proposal.sink.emit = capture;
    proposal.sink.ctx = &actual;
    const auto anchor = ref.at("anchor").get<soma::TokenId>();
    if (backend->propose(model, model.speculative_payload, state, anchor, 5, 0.0f,
                         proposal) != soma::StatusCode::Ok) {
        std::cerr << "DSpark proposal failed\n";
        return 2;
    }

    const auto expected_tokens = ref.at("proposal_tokens").get<std::vector<soma::TokenId>>();
    const auto expected_confidence =
        ref.at("confidence_probability").get<std::vector<float>>();
    bool ok = proposal.tokens == expected_tokens;
    if (!ok) {
        std::cerr << "proposal mismatch\n  oracle:";
        for (const auto t : expected_tokens) std::cerr << ' ' << t;
        std::cerr << "\n  soma:  ";
        for (const auto t : proposal.tokens) std::cerr << ' ' << t;
        std::cerr << '\n';
    }
    const auto confidence_delta = delta(proposal.confidence, expected_confidence);
    std::cout << "confidence max|d|=" << confidence_delta.max_abs
              << " mean|d|=" << confidence_delta.mean_abs << '\n';

    Records expected;
    if (!read_records(oracle_dir / "dspark_oracle.somaact", expected)) {
        std::cerr << "cannot read DSpark activation oracle\n";
        return 2;
    }
    const std::vector<std::string> points = {
        "attn_norm", "q_a", "q_norm", "q_b", "attn_branch",
        "ffn_norm", "router_ids", "router_weights",
        "ffn_branch", "stage_streams",
    };
    const std::map<std::string, float> tolerances = {
        {"attn_norm", 0.05f},      {"q_a", 0.025f},
        {"q_norm", 0.125f},        {"q_b", 0.03f},
        {"attn_branch", 0.004f},   {"ffn_norm", 0.07f},
        {"router_ids", 0.0f},      {"router_weights", 0.003f},
        {"ffn_branch", 0.002f},    {"stage_streams", 0.004f},
        {"sparse_q", 0.23f},       {"sparse_out", 0.11f},
        {"head_hidden", 0.007f},   {"base_logits", 0.05f},
        {"final_logits", 0.05f},   {"confidence", 0.001f},
    };
    for (std::uint32_t stage = 0; stage < 3; ++stage) {
        for (const auto& [actual_point, oracle_point] :
             {std::pair{"sparse_q", "cpu_sparse_q"},
              std::pair{"sparse_out", "cpu_sparse_out"}}) {
            const auto ai = actual.find({stage, actual_point});
            const auto ei = expected.find({stage, oracle_point});
            if (ai != actual.end() && ei != expected.end()) {
                const auto dlt = delta(ai->second, ei->second);
                std::cout << "stage " << stage << ' ' << actual_point
                          << " max|d|=" << dlt.max_abs
                          << " mean|d|=" << dlt.mean_abs << '\n';
                if (dlt.max_abs > tolerances.at(actual_point)) ok = false;
            } else {
                std::cerr << "stage " << stage << ' ' << actual_point
                          << ": missing activation record\n";
                ok = false;
            }
        }
        for (const auto& point : points) {
            const Key key{stage, point};
            const auto ai = actual.find(key), ei = expected.find(key);
            if (ai == actual.end() || ei == expected.end()) {
                std::cout << "stage " << stage << ' ' << point << ": missing on "
                          << (ai == actual.end() ? "Soma" : "oracle") << '\n';
                ok = false;
                continue;
            }
            const auto dlt = delta(ai->second, ei->second);
            std::cout << "stage " << stage << ' ' << point << " max|d|=" << dlt.max_abs
                      << " mean|d|=" << dlt.mean_abs << '\n';
            if (dlt.max_abs > tolerances.at(point)) ok = false;
        }
    }
    for (const auto& point : {"head_hidden", "base_logits", "final_logits", "confidence"}) {
        const Key key{2, point};
        const auto ai = actual.find(key), ei = expected.find(key);
        if (ai == actual.end() || ei == expected.end()) {
            std::cerr << point << ": missing activation record\n";
            ok = false;
            continue;
        }
        const auto dlt = delta(ai->second, ei->second);
        std::cout << point << " max|d|=" << dlt.max_abs
                  << " mean|d|=" << dlt.mean_abs << '\n';
        if (dlt.max_abs > tolerances.at(point)) ok = false;
    }

    // Exact proposal identity is the first useful gate. Activation thresholds
    // are added point-by-point as the oracle localizes remaining semantic drift.
    if (!ok) {
        std::cerr << "DSpark activation/proposal conformance failed\n";
        return 1;
    }
    if (confidence_delta.max_abs > 0.02f) {
        std::cerr << "confidence probability exceeds oracle tolerance\n";
        return 1;
    }
    std::cout << "deepseek_dspark_oracle: OK\n";
    return 0;
}
