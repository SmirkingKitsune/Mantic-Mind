#include "node/model_storage.hpp"
#include "common/logger.hpp"

#include <openssl/evp.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <regex>

namespace fs = std::filesystem;

namespace mm {

ModelStorage::ModelStorage(std::string models_dir)
    : models_dir_(std::move(models_dir))
{
    std::error_code ec;
    fs::create_directories(models_dir_, ec);
}

std::vector<StoredModel> ModelStorage::list_models() const {
    std::vector<StoredModel> result;
    std::error_code ec;

    if (!fs::exists(models_dir_, ec))
        return result;

    for (const auto& entry : fs::recursive_directory_iterator(models_dir_, ec)) {
        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
        // case-insensitive .gguf check
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".gguf") continue;

        StoredModel m;
        m.model_path  = entry.path().filename().string();
        m.size_bytes   = static_cast<int64_t>(entry.file_size(ec));
        m.sha256       = compute_sha256(entry.path().string());

        // Detect shards: pattern like model-00001-of-00003.gguf
        static const std::regex shard_re(
            R"(.*-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
        std::smatch match;
        std::string fname = m.model_path;
        if (std::regex_match(fname, match, shard_re)) {
            m.shard_count = std::stoi(match[2].str());
        } else {
            m.shard_count = 1;
        }

        result.push_back(std::move(m));
    }

    return result;
}

bool ModelStorage::has_model(const std::string& filename,
                             const std::string& expected_sha256) const {
    auto path = model_path(filename);
    std::error_code ec;
    if (!fs::exists(path, ec))
        return false;

    if (!expected_sha256.empty())
        return compute_sha256(path) == expected_sha256;

    return true;
}

std::string ModelStorage::model_path(const std::string& filename) const {
    return (fs::path(models_dir_) / filename).string();
}

bool ModelStorage::delete_model(const std::string& filename) {
    // Also delete shards if this looks like a sharded model base name.
    static const std::regex shard_re(
        R"((.*)-\d{5}-of-(\d{5})\.gguf$)", std::regex::icase);
    std::smatch match;
    std::string fname = filename;

    if (std::regex_match(fname, match, shard_re)) {
        // Delete all shards matching the base pattern.
        std::string base = match[1].str();
        int count = std::stoi(match[2].str());
        bool ok = true;
        for (int i = 1; i <= count; ++i) {
            std::ostringstream ss;
            ss << base << "-" << std::setw(5) << std::setfill('0') << i
               << "-of-" << std::setw(5) << std::setfill('0') << count << ".gguf";
            auto p = model_path(ss.str());
            std::error_code ec;
            if (fs::exists(p, ec)) {
                if (!fs::remove(p, ec)) ok = false;
            }
        }
        return ok;
    }

    auto path = model_path(filename);
    std::error_code ec;
    return fs::remove(path, ec);
}

int64_t ModelStorage::free_space_mb() const {
    std::error_code ec;
    auto si = fs::space(models_dir_, ec);
    if (ec) return 0;
    return static_cast<int64_t>(si.available / (1024 * 1024));
}

std::string ModelStorage::compute_sha256(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return {};

    auto ctx = EVP_MD_CTX_new();
    if (!ctx) return {};

    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);

    constexpr size_t chunk_size = 1024 * 1024;  // 1 MB
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

bool ModelStorage::write_chunk(const std::string& filename,
                               const char* data, size_t size, int64_t offset) {
    auto path = model_path(filename);

    // Ensure parent directory exists.
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);

    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file) {
        // File doesn't exist yet — create it.
        file.open(path, std::ios::out | std::ios::binary);
        if (!file) {
            MM_ERROR("ModelStorage: failed to create {}", path);
            return false;
        }
        file.close();
        file.open(path, std::ios::in | std::ios::out | std::ios::binary);
        if (!file) return false;
    }

    file.seekp(offset);
    file.write(data, static_cast<std::streamsize>(size));
    return file.good();
}

bool ModelStorage::verify_hash(const std::string& filename,
                               const std::string& expected_sha256) const {
    return compute_sha256(model_path(filename)) == expected_sha256;
}

} // namespace mm
