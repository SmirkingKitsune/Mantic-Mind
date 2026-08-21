#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/kv_cache.hpp"
#include "soma/kv_checkpoint.hpp"
#include "soma/plan.hpp"
#include "soma/quant_format.hpp"
#include "soma/safetensors.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool check(bool value, const char* expression, int line) {
    if (value) return true;
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    return false;
}

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if (!check((expr), #expr, __LINE__)) return 1;                                             \
    } while (false)

constexpr const char* kConfig = R"json({
  "model_type":"deepseek_v4","hidden_size":7168,"vocab_size":129280,
  "num_hidden_layers":61,"num_attention_heads":128,"num_key_value_heads":1,
  "head_dim":512,"q_lora_rank":1536,"qk_rope_head_dim":64,
  "o_groups":16,"o_lora_rank":1024,"sliding_window":128,
  "max_position_embeddings":1048576,"eos_token_id":1,
  "compress_rope_theta":160000,"index_n_heads":64,"index_head_dim":128,
  "index_topk":1024,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6,
  "n_routed_experts":384,"n_shared_experts":1,"num_experts_per_tok":6,
  "num_hash_layers":3,"moe_intermediate_size":3072,"hidden_act":"silu",
  "norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sqrtsoftplus",
  "topk_method":"noaux_tc","swiglu_limit":10.0,"rms_norm_eps":1e-6,
  "rope_theta":10000,"rope_scaling":{"type":"yarn","factor":16,
  "original_max_position_embeddings":65536,"beta_fast":32,"beta_slow":1},
  "dspark_block_size":5,"dspark_noise_token_id":128799,
  "dspark_target_layer_ids":[58,59,60],"dspark_markov_rank":512,
  "compress_ratios":[128,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,
  128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,
  128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]
})json";

} // namespace

int main(int argc, char** argv) {
    soma::ArchIr arch;
    CHECK(soma::adapt_hf_config(kConfig, arch).ok());
    CHECK(arch.schema_version == soma::kArchIrSchemaVersionV2);
    CHECK(arch.attention.family == soma::AttentionFamily::CompressedSparse);
    CHECK(arch.topology.n_layers == 61);
    CHECK(arch.router.n_experts == 384);
    CHECK(arch.router.top_k == 6);
    CHECK(arch.router.n_hash_layers == 3);
    CHECK(arch.topology.max_position_embeddings == 1048576);
    CHECK(arch.topology.eos_token_ids == std::vector<std::uint32_t>{1});
    CHECK(arch.attention.compressed.compress_ratios.size() == 61);
    CHECK(arch.attention.compressed.compress_ratios[2] == 4);
    CHECK(arch.attention.compressed.compress_ratios[3] == 128);
    CHECK(arch.speculative.source_declared);
    CHECK(!arch.speculative.present); // config declaration is not a converted capability
    CHECK(arch.speculative.n_layers == 3);
    CHECK(arch.speculative.target_layer_ids == std::vector<soma::LayerIndex>({58, 59, 60}));
    CHECK(arch.speculative.trained_block_size == 5);
    CHECK(arch.speculative.noise_token_id == 128799);
    CHECK(arch.speculative.markov_rank == 512);

    std::string omitted_hash;
    CHECK(soma::compute_arch_hash(arch, omitted_hash).ok());
    auto capable = arch;
    CHECK(soma::apply_container_quant(
              R"json({"group":128,"dtype_dense":"q4_g","dtype_dspark":"q4_g",
              "dspark":"present","dspark_confidence_head":true,
              "dspark_total_expert_bytes":48356130816,
              "dspark_resident_bytes":123456789,"dspark_expert_bytes":41975808,
              "dspark_kv_bytes_per_sequence":393216})json",
              capable)
              .ok());
    CHECK(capable.speculative.present && capable.speculative.confidence_head);
    CHECK(capable.speculative.kv_bytes_per_sequence == 393216);
    CHECK(soma::validate_arch_ir(capable).ok());
    CHECK(soma::compute_arch_hash(capable, capable.arch_hash).ok());
    CHECK(capable.arch_hash != omitted_hash);

    const auto* backend = soma::resolve_attention_backend(arch.attention.family);
    CHECK(backend != nullptr);
    CHECK(backend->kv_bytes_for_context != nullptr);
    CHECK(backend->resident_weight_bytes != nullptr);
    CHECK(backend->serialize_kv != nullptr);
    CHECK(backend->restore_kv != nullptr);
    const auto kv_4k = backend->kv_bytes_for_context(arch, 4096);
    const auto kv_1m = backend->kv_bytes_for_context(arch, 1048576);
    CHECK(kv_1m > kv_4k);

    CHECK(soma::compute_arch_hash(arch, arch.arch_hash).ok());
    soma::HostBudget roomy;
    roomy.ram_total_bytes = 3ull << 40;
    roomy.ram_free_bytes = 2ull << 40;
    roomy.ctx_size = 1048576;
    roomy.kv_slots = 1;
    roomy.disk_bandwidth = 10ull << 30;
    roomy.min_tok_s = 0.01f;
    soma::PlanDocument plan;
    CHECK(soma::compute_plan(arch, roomy, plan).ok());
    CHECK(plan.n_layers == 61 && plan.n_experts == 384 && plan.top_k == 6);
    CHECK(plan.ctx_size == 1048576 && plan.max_context == 1048576 && plan.kv_slots == 1);
    CHECK(plan.kv_bytes_at_ctx == kv_1m);
    CHECK(plan.arch_supported);

    auto speculative_host = roomy;
    speculative_host.speculative = true;
    soma::PlanDocument capable_base_plan;
    CHECK(soma::compute_plan(capable, roomy, capable_base_plan).ok());
    soma::PlanDocument speculative_plan;
    CHECK(soma::compute_plan(capable, speculative_host, speculative_plan).ok());
    CHECK(speculative_plan.speculative_available && speculative_plan.speculative_selected);
    CHECK(speculative_plan.speculative_method == "dspark");
    CHECK(speculative_plan.speculative_stages == 3);
    CHECK(speculative_plan.speculative_trained_block_size == 5);
    CHECK(speculative_plan.speculative_default_tokens == 7);
    CHECK(speculative_plan.speculative_routed_bytes == 48356130816ull);
    CHECK(speculative_plan.speculative_resident_bytes == 123456789ull);
    CHECK(speculative_plan.speculative_kv_bytes_per_slot == 393216ull);
    CHECK(speculative_plan.speculative_kv_bytes_at_ctx == 393216ull);
    CHECK(speculative_plan.kv_bytes_at_ctx == kv_1m + 393216ull);
    CHECK(speculative_plan.dense_resident_bytes ==
          capable_base_plan.dense_resident_bytes + 123456789ull);
    // Selection changes RAM, not the bytes already present in a capable
    // container on disk.
    CHECK(speculative_plan.disk_footprint_bytes == capable_base_plan.disk_footprint_bytes);
    CHECK(speculative_plan.disk_footprint_bytes >= 48356130816ull + 123456789ull);

    // The heterogeneous whole-model callback must be byte-exact, not a rounded
    // per-layer average. The committed native fixture makes the independent
    // reference simple: sum every non-routed tensor payload in SafeTensors.
    if (argc > 1) {
        soma::SafeTensors fixture;
        CHECK(fixture.open_dir(argv[1]).ok());
        std::uint64_t resident_payload = 0;
        for (const auto& name : fixture.names()) {
            if (name.find(".experts.") != std::string::npos) continue;
            const auto* tensor = fixture.find(name);
            CHECK(tensor != nullptr);
            resident_payload += tensor->bytes.size();
        }
        soma::HostBudget fixture_host;
        fixture_host.ram_total_bytes = 8ull << 30;
        fixture_host.ram_free_bytes = 8ull << 30;
        fixture_host.ctx_size = 2048;
        fixture_host.kv_slots = 1;
        fixture_host.disk_bandwidth = 1ull << 30;
        fixture_host.min_tok_s = 0.01f;
        soma::PlanDocument fixture_plan;
        CHECK(soma::compute_plan(argv[1], fixture_host, fixture_plan).ok());
        CHECK(fixture_plan.dense_resident_bytes == resident_payload);
    }

    auto too_long = roomy;
    too_long.ctx_size = 1048577;
    CHECK(!soma::compute_plan(arch, too_long, plan).ok());

    auto tight = roomy;
    tight.ram_total_bytes = kv_1m;
    tight.ram_free_bytes = kv_1m;
    tight.kv_slots = 2;
    CHECK(soma::compute_plan(arch, tight, plan).ok());
    CHECK(plan.verdict == soma::Verdict::Reject);
    CHECK(plan.verdict_reason.find("KV slot") != std::string::npos);

    // A compact V4 cache exercises live-state serialization without allocating
    // the production dimensions. Both compression modes and the sparse indexer
    // remain present, and restoration targets a different configured context.
    soma::ArchIr tiny = arch;
    tiny.topology.n_layers = 2;
    tiny.topology.d_model = 16;
    tiny.topology.vocab_size = 32;
    tiny.topology.layer_kinds = {soma::LayerKind::Moe, soma::LayerKind::Moe};
    tiny.attention.n_heads = 4;
    tiny.attention.n_kv_heads = 1;
    tiny.attention.head_dim = 8;
    tiny.attention.compressed.q_lora_rank = 8;
    tiny.attention.compressed.rope_head_dim = 2;
    tiny.attention.compressed.o_groups = 2;
    tiny.attention.compressed.o_lora_rank = 4;
    tiny.attention.compressed.index_n_heads = 2;
    tiny.attention.compressed.index_head_dim = 4;
    tiny.attention.compressed.index_topk = 8;
    tiny.attention.compressed.compress_ratios = {4, 128};
    tiny.router.n_experts = 8;
    tiny.router.top_k = 2;
    tiny.ffn.expert_intermediate = 8;
    CHECK(soma::compute_arch_hash(tiny, tiny.arch_hash).ok());

    soma::KvCache source;
    CHECK(source.open(tiny, 140).ok());
    for (std::size_t i = 0; i < source.opaque_size(); ++i) {
        source.opaque_data()[i] = static_cast<std::byte>((i * 37u + 11u) & 0xffu);
    }
    CHECK(source.set_length(133).ok());

    std::vector<std::byte> live_before;
    CHECK(backend->serialize_kv(tiny,
                                {source.opaque_data(), source.opaque_size()},
                                source.capacity(),
                                source.length(),
                                live_before)
              .ok());
    // 128 live window rows per layer, completed compressed/index rows, and
    // only the previous+partial ratio-4 carry or partial ratio-128 carry.
    // Inactive carry slots must not leak into a V4 checkpoint.
    CHECK(live_before.size() == 6184);
    CHECK(live_before.size() < source.opaque_size());

    const auto temp = fs::temp_directory_path() / "soma-v4-kv-roundtrip";
    std::error_code ec;
    fs::remove_all(temp, ec);
    soma::KvCheckpointStore store;
    CHECK(store.open(temp.string(), tiny).ok());
    soma::SeqPersistState state;
    state.tokens.resize(source.length());
    std::iota(state.tokens.begin(), state.tokens.end(), soma::TokenId{0});
    state.emitted = {7, 8, 9};
    state.rng_state = 1234567;
    CHECK(store.save("live", source, state).ok());

    soma::KvCheckpointHeader header;
    CHECK(store.stat("live", header).ok());
    CHECK(header.format_id == backend->persist_format_id);
    CHECK(header.payload_bytes == live_before.size());

    soma::KvCache restored;
    CHECK(restored.open(tiny, 256).ok());
    soma::SeqPersistState restored_state;
    CHECK(store.load("live", restored, restored_state).ok());
    CHECK(restored.length() == source.length());
    CHECK(restored_state.tokens == state.tokens);
    CHECK(restored_state.emitted == state.emitted);
    CHECK(restored_state.rng_state == state.rng_state);

    std::vector<std::byte> live_after;
    CHECK(backend->serialize_kv(tiny,
                                {restored.opaque_data(), restored.opaque_size()},
                                restored.capacity(),
                                restored.length(),
                                live_after)
              .ok());
    CHECK(live_after == live_before);
    fs::remove_all(temp, ec);

    // The v2 resident sidecar maps existing QTensor bytes directly; it must not
    // silently reinterpret them as SafeTensors or requantize them on load.
    const auto qdir = fs::temp_directory_path() / "soma-v4-qweight-sidecar";
    fs::remove_all(qdir, ec);
    fs::create_directories(qdir, ec);
    CHECK(!ec);
    std::vector<float> source_weight(4 * 16);
    for (std::size_t i = 0; i < source_weight.size(); ++i)
        source_weight[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 7.0f;
    soma::QTensor packed;
    CHECK(soma::quantize_tensor(source_weight, 4, 16, soma::DType::Q4_G, 128, packed).ok());
    {
        std::ofstream out(qdir / "dense-q-00000.bin", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(packed.data.data()),
                  static_cast<std::streamsize>(packed.data.size()));
    }
    {
        std::ofstream out(qdir / "dense.qweights.index.json", std::ios::binary | std::ios::trunc);
        out << "{\"format\":1,\"weight_map\":{\"probe.weight\":{"
               "\"file\":\"dense-q-00000.bin\",\"offset\":0,\"length\":"
            << packed.data.size()
            << ",\"dtype\":\"q4_g\",\"group\":" << packed.group
            << ",\"shape\":[4,16]}}}";
    }
    soma::SafeTensors qweights;
    CHECK(qweights.open_dir(qdir.string()).ok());
    const auto* probe = qweights.find("probe.weight");
    CHECK(probe != nullptr && probe->dtype == soma::DType::Q4_G);
    CHECK(probe->group == packed.group && probe->bytes.size() == packed.data.size());
    const auto view = soma::WeightRef::from_quantized_bytes(
        probe->bytes, probe->dtype, probe->group, 4, 16);
    std::vector<float> expected(source_weight.size()), actual(source_weight.size());
    CHECK(soma::dequantize_tensor(packed, expected).ok());
    CHECK(soma::dequantize(view, actual).ok());
    CHECK(actual == expected);
    fs::remove_all(qdir, ec);

    std::cout << "deepseek_v4_plan: OK (1M KV " << kv_1m << " bytes)\n";
    return 0;
}
