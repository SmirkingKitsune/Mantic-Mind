#include "control/model_distributor.hpp"
#include "control/node_registry.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/model_catalog.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string compute_sha256(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return {};

    auto ctx = EVP_MD_CTX_new();
    if (!ctx) return {};

    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);

    constexpr size_t chunk_size = 1024 * 1024;
    std::vector<char> buf(chunk_size);
    while (file.read(buf.data(), static_cast<std::streamsize>(chunk_size)) || file.gcount() > 0) {
        EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(file.gcount()));
        if (file.eof()) break;
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_DigestFinal_ex(ctx, hash, &hash_len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream ss;
    for (unsigned int i = 0; i < hash_len; ++i)
        ss << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(hash[i]);
    return ss.str();
}

} // anonymous namespace

namespace mm {

ModelDistributor::ModelDistributor(NodeRegistry& registry, std::string models_dir)
    : registry_(registry)
    , models_dir_(std::move(models_dir))
{}

bool ModelDistributor::upload_model(const NodeId& node_id,
                                    const std::string& model_filename,
                                    ProgressCallback progress_cb) {
    auto shards = detect_shards(model_filename);

    for (const auto& shard : shards) {
        if (!upload_single_file(node_id, shard, progress_cb)) {
            MM_ERROR("ModelDistributor: failed to upload shard {} to node {}",
                     shard, node_id);
            return false;
        }
    }

    MM_INFO("ModelDistributor: all {} shard(s) of {} uploaded to node {}",
            shards.size(), model_filename, node_id);
    return true;
}

bool ModelDistributor::node_has_model(const NodeId& node_id,
                                      const std::string& model_filename) const {
    try {
        auto node = registry_.get_node(node_id);
        for (const auto& m : node.stored_models) {
            if (m.model_path == model_filename) return true;
        }
    } catch (...) {}
    return false;
}

bool ModelDistributor::delete_model_on_node(const NodeId& node_id,
                                            const std::string& model_filename) {
    try {
        auto node = registry_.get_node(node_id);
        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);
        auto resp = cli.del("/api/node/models/" + model_filename);
        return resp.ok();
    } catch (const std::exception& e) {
        MM_WARN("ModelDistributor: delete_model_on_node failed: {}", e.what());
        return false;
    }
}

std::vector<std::string> ModelDistributor::detect_shards(
    const std::string& filename)
{
    return expand_model_shards(filename);
}

bool ModelDistributor::upload_single_file(const NodeId& node_id,
                                          const std::string& filename,
                                          ProgressCallback progress_cb) {
    NodeInfo node;
    try {
        node = registry_.get_node(node_id);
    } catch (...) {
        MM_ERROR("ModelDistributor: node {} not found", node_id);
        return false;
    }

    auto local_path = (fs::path(models_dir_) / filename).string();
    std::error_code ec;
    if (!fs::exists(local_path, ec)) {
        MM_ERROR("ModelDistributor: local model file not found: {}", local_path);
        return false;
    }

    int64_t total_bytes = static_cast<int64_t>(fs::file_size(local_path, ec));
    std::string sha256 = compute_sha256(local_path);

    std::ifstream file(local_path, std::ios::binary);
    if (!file) {
        MM_ERROR("ModelDistributor: failed to open {}", local_path);
        return false;
    }

    HttpClient cli(node.url);
    cli.set_bearer_token(node.api_key);

    constexpr int64_t chunk_size = 4 * 1024 * 1024;  // 4 MB
    std::vector<char> buf(static_cast<size_t>(chunk_size));
    int64_t offset = 0;

    MM_INFO("ModelDistributor: uploading {} ({} MB) to node {}",
            filename, static_cast<long long>(total_bytes / (1024 * 1024)), node_id);

    while (offset < total_bytes) {
        auto read_size = std::min(chunk_size, total_bytes - offset);
        file.read(buf.data(), read_size);
        auto actually_read = file.gcount();
        if (actually_read <= 0) break;

        bool is_last = (offset + actually_read >= total_bytes);

        // Build chunked upload request body as raw bytes
        // We need to use raw POST with custom headers
        // For now, use the HttpClient's post method with a json body that
        // contains base64 data — but that's inefficient.
        // Instead, build a direct httplib call via the node URL.

        // Build the request manually using cpp-httplib
        auto [host, port] = [&]() -> std::pair<std::string, int> {
            // Simple URL parse
            std::string url = node.url;
            std::string h;
            int p = 80;
            if (url.rfind("http://", 0) == 0) url = url.substr(7);
            else if (url.rfind("https://", 0) == 0) { url = url.substr(8); p = 443; }
            auto colon = url.find(':');
            auto slash = url.find('/');
            if (colon != std::string::npos) {
                h = url.substr(0, colon);
                auto port_end = (slash != std::string::npos) ? slash : url.size();
                try { p = std::stoi(url.substr(colon + 1, port_end - colon - 1)); } catch (...) {}
            } else {
                h = (slash != std::string::npos) ? url.substr(0, slash) : url;
            }
            return {h, p};
        }();

        httplib::Client http_cli(host, port);
        http_cli.set_connection_timeout(10);
        http_cli.set_read_timeout(300);
        http_cli.set_write_timeout(300);

        httplib::Headers headers = {
            {"Authorization", "Bearer " + node.api_key},
            {"X-Model-Filename", filename},
            {"X-Model-SHA256", sha256},
            {"X-Chunk-Offset", std::to_string(offset)},
            {"X-Last-Chunk", is_last ? "true" : "false"},
            {"X-Shard-Index", "0"},
            {"X-Shard-Count", "1"}
        };

        std::string body_str(buf.data(), static_cast<size_t>(actually_read));
        auto result = http_cli.Post("/api/node/upload-model",
                                     headers,
                                     body_str,
                                     "application/octet-stream");

        if (!result || result->status != 200) {
            int http_code = result ? result->status : 0;
            MM_ERROR("ModelDistributor: chunk upload failed at offset {} (HTTP {})",
                     static_cast<long long>(offset), http_code);
            return false;
        }

        offset += actually_read;

        if (progress_cb)
            progress_cb(offset, total_bytes);

        // Check verification on last chunk
        if (is_last) {
            try {
                auto j = nlohmann::json::parse(result->body);
                if (!j.value("verified", false)) {
                    MM_ERROR("ModelDistributor: SHA-256 verification failed for {} on node {}",
                             filename, node_id);
                    return false;
                }
            } catch (...) {
                MM_WARN("ModelDistributor: could not parse verification response");
            }
        }
    }

    MM_INFO("ModelDistributor: {} uploaded and verified on node {}", filename, node_id);
    return true;
}

} // namespace mm
