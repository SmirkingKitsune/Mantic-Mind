#include "node/model_puller.hpp"

#include "node/model_storage.hpp"
#include "node/node_state.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/model_catalog.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace mm {

namespace {

std::string select_control_auth_key(const std::string& control_url,
                                    const std::vector<std::string>& api_keys) {
    for (const auto& key : api_keys) {
        if (key.empty()) continue;
        HttpClient cli(control_url);
        cli.set_bearer_token(key);
        auto resp = cli.get("/api/control/models");
        if (resp.ok()) return key;
    }
    return {};
}

} // namespace

ModelPuller::ModelPuller(ModelStorage& storage,
                         NodeState& state,
                         std::string control_url)
    : storage_(storage)
    , state_(state)
    , control_url_(std::move(control_url))
{}

bool ModelPuller::pull_model(const std::string& model_filename,
                             bool force,
                             nlohmann::json* out_result) {
    nlohmann::json result = {
        {"status", "failed"},
        {"model_filename", canonical_model_filename(model_filename)},
        {"force", force},
        {"downloaded", 0},
        {"skipped", 0},
        {"errors", nlohmann::json::array()},
        {"files", nlohmann::json::array()}
    };

    auto fail = [&](const std::string& reason) -> bool {
        result["error"] = reason;
        if (out_result) *out_result = result;
        return false;
    };

    const std::string canonical = canonical_model_filename(model_filename);
    if (!is_safe_model_filename(canonical)) {
        return fail("invalid model_filename");
    }
    if (control_url_.empty()) {
        return fail("control_url not configured");
    }

    const auto api_keys = state_.get_api_keys();
    const std::string auth_key = select_control_auth_key(control_url_, api_keys);
    if (auth_key.empty()) {
        return fail("no valid node api key could authenticate with control");
    }

    HttpClient catalog_cli(control_url_);
    catalog_cli.set_bearer_token(auth_key);
    auto catalog_resp = catalog_cli.get("/api/control/models");
    if (!catalog_resp.ok()) {
        return fail("failed to fetch control model catalog");
    }

    std::vector<StoredModel> control_models;
    try {
        auto j = nlohmann::json::parse(catalog_resp.body);
        if (j.contains("models")) {
            control_models = j["models"].get<std::vector<StoredModel>>();
        }
    } catch (const std::exception& e) {
        return fail(std::string("invalid control catalog response: ") + e.what());
    }

    auto get_control_model = [&](const std::string& filename) -> std::optional<StoredModel> {
        for (const auto& m : control_models) {
            if (m.model_path == filename) return m;
        }
        return std::nullopt;
    };

    const auto shard_files = expand_model_shards(canonical);
    if (shard_files.empty()) return fail("invalid shard expansion");

    auto [host, port] = util::parse_url(control_url_);
    httplib::Client raw_cli(host, port);
    raw_cli.set_connection_timeout(10);
    raw_cli.set_read_timeout(300);
    raw_cli.set_write_timeout(30);

    for (const auto& filename : shard_files) {
        auto cm = get_control_model(filename);
        if (!cm) {
            result["errors"].push_back("control model not found: " + filename);
            continue;
        }
        const auto& control_meta = *cm;

        if (!force && storage_.has_model(filename, control_meta.sha256)) {
            result["skipped"] = result.value("skipped", 0) + 1;
            result["files"].push_back(nlohmann::json{
                {"filename", filename},
                {"status", "already_present"},
                {"size_bytes", control_meta.size_bytes},
                {"sha256", control_meta.sha256}
            });
            continue;
        }

        if (force) {
            storage_.delete_model(filename);
        }

        std::string temp_name = filename + ".partial";
        const std::string temp_path = storage_.model_path(temp_name);
        std::error_code ec;
        fs::remove(temp_path, ec);

        int64_t offset = 0;
        bool write_failed = false;
        httplib::Headers headers = {
            {"Authorization", "Bearer " + auth_key}
        };
        std::string path = "/api/control/models/" + filename + "/content";

        auto http_res = raw_cli.Get(path.c_str(), headers,
            [&](const char* data, size_t data_length) {
                if (write_failed) return false;
                if (!storage_.write_chunk(temp_name, data, data_length, offset)) {
                    write_failed = true;
                    return false;
                }
                offset += static_cast<int64_t>(data_length);
                return true;
            });

        if (!http_res || http_res->status != 200 || write_failed) {
            fs::remove(temp_path, ec);
            const int code = http_res ? http_res->status : 0;
            result["errors"].push_back("download failed for " + filename +
                                        " (HTTP " + std::to_string(code) + ")");
            continue;
        }

        if (offset != control_meta.size_bytes) {
            fs::remove(temp_path, ec);
            result["errors"].push_back("size mismatch for " + filename);
            continue;
        }

        if (!storage_.verify_hash(temp_name, control_meta.sha256)) {
            fs::remove(temp_path, ec);
            result["errors"].push_back("sha256 mismatch for " + filename);
            continue;
        }

        const std::string final_path = storage_.model_path(filename);
        fs::remove(final_path, ec);
        fs::rename(temp_path, final_path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
            result["errors"].push_back("failed to move file into place: " + filename);
            continue;
        }

        result["downloaded"] = result.value("downloaded", 0) + 1;
        result["files"].push_back(nlohmann::json{
            {"filename", filename},
            {"status", "downloaded"},
            {"size_bytes", control_meta.size_bytes},
            {"sha256", control_meta.sha256}
        });
    }

    const int downloaded = result.value("downloaded", 0);
    const size_t errors = result["errors"].size();
    if (downloaded > 0 && errors == 0) {
        result["status"] = "downloaded";
    } else if (downloaded > 0) {
        result["status"] = "partial";
    } else {
        result["status"] = "failed";
    }

    if (out_result) *out_result = result;
    return result["status"] == "downloaded" || result["status"] == "partial";
}

} // namespace mm

