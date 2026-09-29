// mm_models_tests — the served-model catalog, without a server.
//
// Its own executable rather than more of mm_reliability_tests: that runner
// stops at the first failure, and under WSL `admission_fetch_stage` fails for
// environment reasons (the admission venv is Windows-only), so every test
// registered after it never runs there. The rules in this file are pure or
// SQLite-only and should be checked on both toolchains.

#include "common/served_model.hpp"
#include "common/util.hpp"
#include "control/model_registry.hpp"
#include "control/served_model_catalog.hpp"
#include "control/served_model_validator.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check(bool ok, const char* expr, int line) {
    if (!ok) std::cerr << "CHECK failed at line " << line << ": " << expr << "\n";
    return ok;
}

#define CHECK(expr) do { if (!check((expr), #expr, __LINE__)) return false; } while (0)

std::filesystem::path temp_test_dir(const std::string& name) {
    return std::filesystem::temp_directory_path() /
           ("mantic-mind-" + name + "-" + mm::util::generate_uuid());
}

bool remove_tree(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return !ec;
}

mm::AdmittedModel admission(std::int64_t id,
                            const std::string& hash,
                            const std::string& name,
                            const std::string& dir,
                            mm::ModelVerdict verdict) {
    mm::AdmittedModel m;
    m.id = id;
    m.arch_hash = hash;
    m.name = name;
    m.model_dir = dir;
    m.attention_family = "gqa";
    m.verdict = verdict;
    m.verdict_basis = "{}";
    m.admitted_at_ms = 1000;
    m.profiled_at_ms = 2000;
    return m;
}

const mm::CatalogEntry* entry(const std::vector<mm::CatalogEntry>& list, const std::string& id) {
    const auto it = std::find_if(list.begin(), list.end(),
                                 [&](const mm::CatalogEntry& e) { return e.model.id == id; });
    return it == list.end() ? nullptr : &*it;
}

// ── ids ───────────────────────────────────────────────────────────────────────

bool test_served_model_ids_and_slugs() {
    CHECK(mm::util::is_valid_served_model_id("qwen3-30b-a3b"));
    CHECK(mm::util::is_valid_served_model_id("qwen2.5-7b"));
    CHECK(mm::util::is_valid_served_model_id("a"));
    CHECK(mm::util::is_valid_served_model_id("m_1.q4"));
    // Lowercase only, so two spellings cannot name one model.
    CHECK(!mm::util::is_valid_served_model_id("Qwen3"));
    CHECK(!mm::util::is_valid_served_model_id(""));
    CHECK(!mm::util::is_valid_served_model_id("-lead"));
    CHECK(!mm::util::is_valid_served_model_id(".lead"));
    CHECK(!mm::util::is_valid_served_model_id("has space"));
    CHECK(!mm::util::is_valid_served_model_id("a:b"));
    CHECK(!mm::util::is_valid_served_model_id("a/b"));
    CHECK(!mm::util::is_valid_served_model_id(std::string(129, 'a')));
    CHECK(mm::util::is_valid_served_model_id(std::string(128, 'a')));

    CHECK(mm::util::slugify("Qwen3-30B-A3B-q4_g-q6_g-g128") == "qwen3-30b-a3b-q4_g-q6_g-g128");
    CHECK(mm::util::slugify("  My Model (v2)! ") == "my-model-v2");
    CHECK(mm::util::slugify("a//b\\\\c") == "a-b-c");
    CHECK(mm::util::slugify("__x__") == "x");
    CHECK(mm::util::slugify("!!!").empty());
    CHECK(mm::util::slugify("abcdef", 3) == "abc");
    CHECK(mm::util::slugify("ab-cdef", 3) == "ab"); // never ends on a separator
    return true;
}

// ── sampling overrides ────────────────────────────────────────────────────────

bool test_sampling_overrides() {
    mm::SamplingOverrides o;
    std::string err;
    CHECK(mm::parse_sampling_overrides(nlohmann::json::parse(R"({"temperature":0.2,"max_tokens":64})"),
                                       o, err));
    CHECK(o.temperature && *o.temperature == 0.2f);
    CHECK(o.max_tokens && *o.max_tokens == 64);
    CHECK(!o.top_p);

    // A misspelled key is REPORTED, not dropped: silently ignoring "temprature"
    // reads as the model ignoring the setting.
    CHECK(!mm::parse_sampling_overrides(nlohmann::json::parse(R"({"temprature":0.2})"), o, err));
    CHECK(err.find("temprature") != std::string::npos);
    CHECK(!mm::parse_sampling_overrides(nlohmann::json::parse(R"({"top_k":1.5})"), o, err));
    CHECK(!mm::parse_sampling_overrides(nlohmann::json::parse(R"({"top_p":"high"})"), o, err));
    CHECK(!mm::parse_sampling_overrides(nlohmann::json::parse("[1]"), o, err));
    // null clears; an absent object is no overrides at all.
    CHECK(mm::parse_sampling_overrides(nlohmann::json::parse(R"({"temperature":null})"), o, err));
    CHECK(o.empty());
    CHECK(mm::parse_sampling_overrides(nlohmann::json(), o, err));
    CHECK(o.empty());

    // Request over conversation, field by field.
    mm::SamplingOverrides conv;
    conv.temperature = 0.9f;
    conv.top_p = 0.5f;
    mm::SamplingOverrides req;
    req.temperature = 0.1f;
    const auto merged = mm::merge_overrides(conv, req);
    CHECK(*merged.temperature == 0.1f);
    CHECK(*merged.top_p == 0.5f);

    // Launch settings are untouched: they are engine identity.
    mm::RuntimeSettings base;
    base.ctx_size = 8192;
    base.parallel = 4;
    const auto applied = mm::apply_overrides(base, merged);
    CHECK(applied.temperature == 0.1f);
    CHECK(applied.top_p == 0.5f);
    CHECK(applied.ctx_size == 8192);
    CHECK(applied.parallel == 4);
    CHECK(applied.max_tokens == base.max_tokens);

    // One set of ranges for defaults and overrides.
    mm::SamplingOverrides bad;
    bad.top_p = 0.0f;
    bad.max_tokens = 0;
    bad.min_p = -1.0f; // -1 is "runtime default", allowed
    const auto problems = mm::sampling_override_problems(bad);
    CHECK(problems.size() == 2);
    CHECK(mm::sampling_override_problems(mm::sampling_of(mm::RuntimeSettings{})).empty());
    return true;
}

// ── JSON ──────────────────────────────────────────────────────────────────────

bool test_served_model_json_never_carries_the_key() {
    mm::ServedModel m;
    m.id = "remote";
    m.inference_backend = "api";
    m.api_settings.api_key = "sk-secret";
    m.runtime_settings.ctx_size = 32768;
    const nlohmann::json j = m;
    CHECK(j.dump().find("sk-secret") == std::string::npos);
    CHECK(j.at("api_settings").at("api_key_configured") == true);

    mm::ServedModel back = j.get<mm::ServedModel>();
    CHECK(back.id == "remote");
    CHECK(back.inference_backend == "api");
    CHECK(back.runtime_settings.ctx_size == 32768);
    CHECK(back.api_settings.api_key.empty());

    // Absent fields keep defaults.
    const auto partial = nlohmann::json::parse(R"({"model_path":"/m.gguf"})").get<mm::ServedModel>();
    CHECK(partial.inference_backend == "node");
    CHECK(partial.backend_override == "auto");
    CHECK(partial.enabled);
    CHECK(partial.tool_calling);
    return true;
}

// What a request may carry depends on the model AND the engine it lands on:
// the same conversation can be answered by models that differ in both.
bool test_capabilities_for() {
    mm::ServedModel m;
    m.id = "vision-gguf";
    m.vision_settings.enabled = true;
    m.tool_calling = true;

    auto caps = mm::capabilities_for(m, "llama-cpp", false);
    CHECK(caps.vision && caps.tools);
    // Soma is text-only and renders no tools, whatever the model says.
    caps = mm::capabilities_for(m, "soma", false);
    CHECK(!caps.vision && !caps.tools);
    // vLLM calls tools only when launched with auto tool choice.
    CHECK(!mm::capabilities_for(m, "vllm", false).tools);
    CHECK(mm::capabilities_for(m, "vllm", true).tools);
    // An engine nobody knows is refused both, not granted them on a guess.
    caps = mm::capabilities_for(m, "", false);
    CHECK(!caps.vision && !caps.tools);

    // The model's own settings are necessary as well as sufficient.
    m.vision_settings.enabled = false;
    m.tool_calling = false;
    caps = mm::capabilities_for(m, "llama-cpp", true);
    CHECK(!caps.vision && !caps.tools);

    // An API model has no engine here: its provider governs.
    mm::ServedModel api;
    api.inference_backend = "api";
    api.vision_settings.enabled = true;
    api.tool_calling = true;
    caps = mm::capabilities_for(api, "soma", false);
    CHECK(caps.vision && caps.tools);
    return true;
}

// ── the catalog rules ─────────────────────────────────────────────────────────

bool test_compute_catalog_rules() {
    using mm::ModelVerdict;
    const std::vector<mm::AdmittedModel> admitted = {
        admission(1, "hash-qwen-q4", "Qwen3-30B-A3B", "/c/Qwen3-30B-A3B-q4_g-q6_g-g128",
                  ModelVerdict::Stream),
        admission(2, "hash-dsv2", "DeepSeek-V2-Lite", "/c/DeepSeek-V2-Lite-q4_g-q6_g-g128",
                  ModelVerdict::Hybrid),
        // Not servable by Soma: never listed as admitted. (An operator can
        // still list a GGUF of it explicitly.)
        admission(3, "hash-mixtral", "Mixtral-8x7B", "/c/Mixtral-8x7B-q4_g-q6_g-g128",
                  ModelVerdict::ResidentOnly),
        admission(4, "hash-bad", "Broken", "/c/Broken-q4_g-q6_g-g128", ModelVerdict::Reject),
    };
    const auto resolve = [&](const std::string& ref) -> std::optional<mm::AdmittedModel> {
        for (const auto& a : admitted)
            if (a.arch_hash == ref || a.name == ref) return a;
        return std::nullopt;
    };

    // Admitted only.
    auto list = mm::compute_catalog({}, admitted, resolve, true);
    CHECK(list.size() == 2);
    const auto* qwen = entry(list, "qwen3-30b-a3b-q4_g-q6_g-g128");
    CHECK(qwen != nullptr);
    CHECK(qwen->source == mm::CatalogSource::Admitted);
    CHECK(qwen->available);
    CHECK(qwen->admission_id == 1);
    // Addressed by HASH (names are ambiguous across quantizations) and pinned
    // to Soma (under [llama-cpp, soma] `auto` would hand llama.cpp a container).
    CHECK(qwen->model.model_path == "hash-qwen-q4");
    CHECK(qwen->model.backend_override == "soma");
    CHECK(qwen->model.inference_backend == "node");
    CHECK(!qwen->model.tool_calling);
    CHECK(!qwen->model.vision_settings.enabled);
    CHECK(qwen->model.display_name == "Qwen3-30B-A3B (q4_g-q6_g-g128)");
    CHECK(entry(list, "mixtral-8x7b-q4_g-q6_g-g128") == nullptr);
    CHECK(entry(list, "broken-q4_g-q6_g-g128") == nullptr);
    // Sorted by id.
    CHECK(list[0].model.id < list[1].model.id);

    // No Soma in the engine policy: listed, but unavailable, with a reason.
    list = mm::compute_catalog({}, admitted, resolve, false);
    CHECK(list.size() == 2);
    CHECK(!list[0].available);
    CHECK(!list[0].unavailable_reason.empty());

    // An explicit row that resolves to an admission SHADOWS it.
    mm::ServedModel override_row;
    override_row.id = "qwen-tuned";
    override_row.model_path = "hash-qwen-q4";
    override_row.runtime_settings.ctx_size = 32768;
    // An explicit GGUF row, unrelated to any admission.
    mm::ServedModel gguf;
    gguf.id = "llama-3-8b";
    gguf.model_path = "/models/llama-3-8b.Q4_K_M.gguf";
    // A disabled row stays listed, unavailable.
    mm::ServedModel off;
    off.id = "old";
    off.model_path = "/models/old.gguf";
    off.enabled = false;
    // An API row never resolves, even if its model_path happens to match.
    mm::ServedModel api;
    api.id = "remote";
    api.inference_backend = "api";
    api.model_path = "DeepSeek-V2-Lite";

    list = mm::compute_catalog({override_row, gguf, off, api}, admitted, resolve, true);
    CHECK(entry(list, "qwen3-30b-a3b-q4_g-q6_g-g128") == nullptr); // shadowed
    const auto* tuned = entry(list, "qwen-tuned");
    CHECK(tuned != nullptr);
    CHECK(tuned->source == mm::CatalogSource::Explicit);
    CHECK(tuned->admission_id == 1);
    CHECK(entry(list, "llama-3-8b")->admission_id == 0);
    CHECK(!entry(list, "old")->available);
    CHECK(entry(list, "remote")->admission_id == 0);
    CHECK(entry(list, "deepseek-v2-lite-q4_g-q6_g-g128") != nullptr); // not shadowed by API row
    CHECK(list.size() == 5);

    // Explicit rows win id collisions; the admitted entry gets a hash suffix.
    mm::ServedModel squatter;
    squatter.id = "deepseek-v2-lite-q4_g-q6_g-g128";
    squatter.model_path = "/models/unrelated.gguf";
    list = mm::compute_catalog({squatter}, admitted, resolve, true);
    CHECK(entry(list, "deepseek-v2-lite-q4_g-q6_g-g128")->source ==
          mm::CatalogSource::Explicit);
    const auto* suffixed = entry(list, "deepseek-v2-lite-q4_g-q6_g-g128-hash-dsv");
    CHECK(suffixed != nullptr);
    CHECK(suffixed->source == mm::CatalogSource::Admitted);
    CHECK(suffixed->admission_id == 2);
    return true;
}

// ── storage and the catalog class ─────────────────────────────────────────────

bool test_registry_served_model_rows() {
    const auto dir = temp_test_dir("served-rows");
    {
        mm::ControlModelRegistry reg;
        std::string err;
        CHECK(reg.open(dir.string(), err));
        CHECK(reg.schema_version() == 4);
        CHECK(reg.list_served_models().empty());

        mm::ServedModel m;
        m.id = "remote";
        m.display_name = "Remote";
        m.model_path = "gpt-x";
        m.inference_backend = "api";
        m.api_settings.base_url = "https://example.test";
        m.api_settings.api_key = "sk-secret";
        m.runtime_settings.temperature = 0.3f;
        m.vision_settings.enabled = true;
        m.tool_calling = false;
        CHECK(reg.upsert_served_model(m, err));

        auto row = reg.find_served_model("remote");
        CHECK(row.has_value());
        CHECK(row->display_name == "Remote");
        CHECK(row->api_settings.base_url == "https://example.test");
        CHECK(row->api_settings.api_key.empty()); // never at rest
        CHECK(row->runtime_settings.temperature == 0.3f);
        CHECK(row->vision_settings.enabled);
        CHECK(!row->tool_calling);
        CHECK(row->created_at_ms > 0);
        const auto created = row->created_at_ms;

        // A replace is an edit: created_at survives.
        m.display_name = "Remote 2";
        m.created_at_ms = 0;
        CHECK(reg.upsert_served_model(m, err));
        row = reg.find_served_model("remote");
        CHECK(row->display_name == "Remote 2");
        CHECK(row->created_at_ms == created);

        // The CHECK constraint is the last line of defence.
        mm::ServedModel bad = m;
        bad.id = "bad";
        bad.inference_backend = "llama-cpp";
        CHECK(!reg.upsert_served_model(bad, err));
        CHECK(!reg.find_served_model("bad"));

        CHECK(reg.list_served_models().size() == 1);
        CHECK(reg.delete_served_model("remote", err));
        CHECK(!reg.delete_served_model("remote", err));
        CHECK(reg.list_served_models().empty());
        reg.close();
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_catalog_create_update_remove() {
    const auto dir = temp_test_dir("served-catalog");
    {
        mm::ControlModelRegistry reg;
        std::string err;
        CHECK(reg.open(dir.string(), err));
        auto a = admission(0, "hash-qwen", "Qwen3-30B-A3B", "/c/Qwen3-30B-A3B-q4_g-q6_g-g128",
                           mm::ModelVerdict::Stream);
        CHECK(reg.upsert(a, err));

        mm::ServedModelCatalog catalog(reg);
        auto list = catalog.list();
        CHECK(list.size() == 1);
        const std::string admitted_id = list[0].model.id;
        CHECK(admitted_id == "qwen3-30b-a3b-q4_g-q6_g-g128");

        // An explicit row may not take an admitted entry's id by CREATE...
        mm::ServedModel clash;
        clash.id = admitted_id;
        clash.model_path = "/models/other.gguf";
        CHECK(catalog.create(clash, err) == mm::ServedModelCatalog::PutResult::Conflict);

        // ...but UPDATE on an admitted id writes the operator's override under
        // the SAME id, so conversations that named it keep working.
        auto edited = list[0].model;
        edited.runtime_settings.ctx_size = 65536;
        CHECK(catalog.update(admitted_id, edited, err) ==
              mm::ServedModelCatalog::PutResult::Updated);
        list = catalog.list();
        CHECK(list.size() == 1);
        CHECK(list[0].model.id == admitted_id);
        CHECK(list[0].source == mm::CatalogSource::Explicit);
        CHECK(list[0].model.runtime_settings.ctx_size == 65536);
        CHECK(list[0].admission_id > 0);

        // Removing the override brings the admitted entry back, same id.
        CHECK(catalog.remove(admitted_id, err) == mm::ServedModelCatalog::RemoveResult::Removed);
        list = catalog.list();
        CHECK(list.size() == 1);
        CHECK(list[0].model.id == admitted_id);
        CHECK(list[0].source == mm::CatalogSource::Admitted);
        // And an admitted entry itself cannot be removed.
        CHECK(catalog.remove(admitted_id, err) ==
              mm::ServedModelCatalog::RemoveResult::NotExplicit);
        CHECK(catalog.remove("nope", err) == mm::ServedModelCatalog::RemoveResult::NotFound);
        CHECK(catalog.update("nope", edited, err) ==
              mm::ServedModelCatalog::PutResult::NotFound);

        // API keys: process memory only, and an edit without one keeps it.
        mm::ServedModel remote;
        remote.id = "remote";
        remote.model_path = "gpt-x";
        remote.inference_backend = "api";
        remote.api_settings.api_key = "sk-1";
        CHECK(catalog.create(remote, err) == mm::ServedModelCatalog::PutResult::Created);
        CHECK(catalog.find("remote")->model.api_settings.api_key == "sk-1");
        CHECK(reg.find_served_model("remote")->api_settings.api_key.empty());
        remote.api_settings.api_key.clear();
        remote.display_name = "renamed";
        CHECK(catalog.update("remote", remote, err) ==
              mm::ServedModelCatalog::PutResult::Updated);
        CHECK(catalog.find("remote")->model.api_settings.api_key == "sk-1");
        CHECK(catalog.find("remote")->model.display_name == "renamed");

        // Soma absent from the engine policy: listed, unavailable.
        catalog.set_soma_configured([] { return false; });
        CHECK(!catalog.find(admitted_id)->available);
        CHECK(catalog.find("remote")->available);

        // Free ids.
        CHECK(catalog.unique_id("Remote") == "remote-2");
        CHECK(catalog.unique_id("Brand New") == "brand-new");
        CHECK(catalog.unique_id("!!!").empty());
        reg.close();
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_validate_served_model() {
    const auto dir = temp_test_dir("served-validate");
    {
        mm::ControlModelRegistry reg;
        std::string err;
        CHECK(reg.open(dir.string(), err));
        auto a = admission(0, "hash-qwen", "Qwen3-30B-A3B", "/c/Qwen3-30B-A3B-q4_g-q6_g-g128",
                           mm::ModelVerdict::Stream);
        CHECK(reg.upsert(a, err));

        const auto has = [](const mm::ValidationResult& r, const std::string& field,
                            mm::ValidationSeverity sev) {
            return std::any_of(r.issues.begin(), r.issues.end(), [&](const auto& i) {
                return i.field == field && i.severity == sev;
            });
        };

        mm::ServedModel m;
        m.id = "gguf";
        m.model_path = "/models/m.gguf";
        CHECK(mm::validate_served_model(m, nullptr, "", &reg).ok());

        // An admitted reference is a container, not a GGUF: no GGUF warning.
        m.model_path = "hash-qwen";
        auto r = mm::validate_served_model(m, nullptr, "", &reg);
        CHECK(r.ok());
        CHECK(!has(r, "model_path", mm::ValidationSeverity::Warning));
        // Without the registry the same ref looks like a malformed GGUF path.
        r = mm::validate_served_model(m, nullptr, "", nullptr);
        CHECK(has(r, "model_path", mm::ValidationSeverity::Warning));

        mm::ServedModel bad;
        bad.id = "Bad Id";
        bad.inference_backend = "llama-cpp"; // the agent spelling is not accepted
        bad.backend_override = "sometimes";
        bad.runtime_settings.top_p = 0.0f;
        r = mm::validate_served_model(bad, nullptr, "", &reg);
        CHECK(!r.ok());
        CHECK(has(r, "id", mm::ValidationSeverity::Error));
        CHECK(has(r, "model_path", mm::ValidationSeverity::Error));
        CHECK(has(r, "inference_backend", mm::ValidationSeverity::Error));
        CHECK(has(r, "backend_override", mm::ValidationSeverity::Error));
        CHECK(has(r, "runtime_settings.top_p", mm::ValidationSeverity::Error));

        mm::ServedModel api;
        api.id = "remote";
        api.model_path = "gpt-x";
        api.inference_backend = "api";
        api.api_settings.chat_completions_path = "v1/chat";
        api.vision_settings.mmproj_path = "/models/mmproj-x.gguf";
        api.preferred_node_id = "node-a";
        r = mm::validate_served_model(api, nullptr, "", &reg);
        CHECK(!r.ok());
        CHECK(has(r, "api_settings.chat_completions_path", mm::ValidationSeverity::Error));
        CHECK(has(r, "vision_settings.mmproj_path", mm::ValidationSeverity::Error));
        CHECK(has(r, "preferred_node_id", mm::ValidationSeverity::Warning));
        reg.close();
    }
    CHECK(remove_tree(dir));
    return true;
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        bool (*fn)();
    };
    const Test tests[] = {
        {"served_model_ids_and_slugs", test_served_model_ids_and_slugs},
        {"sampling_overrides", test_sampling_overrides},
        {"served_model_json_never_carries_the_key", test_served_model_json_never_carries_the_key},
        {"capabilities_for", test_capabilities_for},
        {"compute_catalog_rules", test_compute_catalog_rules},
        {"registry_served_model_rows", test_registry_served_model_rows},
        {"catalog_create_update_remove", test_catalog_create_update_remove},
        {"validate_served_model", test_validate_served_model},
    };
    const std::string filter = argc > 1 ? argv[1] : std::string{};
    int failed = 0;
    bool ran_any = false;
    // Unlike mm_reliability_tests, this runner does NOT stop at the first
    // failure: every test runs, so one red test cannot hide the rest.
    for (const auto& t : tests) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) continue;
        ran_any = true;
        if (t.fn()) {
            std::cout << "PASSED: " << t.name << "\n";
        } else {
            std::cerr << "FAILED: " << t.name << "\n";
            ++failed;
        }
    }
    if (!filter.empty() && !ran_any) {
        std::cerr << "No tests matched filter: " << filter << "\n";
        return 1;
    }
    return failed == 0 ? 0 : 1;
}
