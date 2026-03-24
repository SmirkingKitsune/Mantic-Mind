#include "common/model_catalog.hpp"

#include "common/util.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace mm {

namespace {

std::string sanitize_token(std::string s) {
    s = util::trim(s);
    if (s.size() >= 2) {
        if ((s.front() == '"' && s.back() == '"') ||
            (s.front() == '\'' && s.back() == '\'')) {
            s = util::trim(s.substr(1, s.size() - 2));
        }
    }
    return s;
}

std::string compute_sha256(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return {};

    auto* ctx = EVP_MD_CTX_new();
    if (!ctx) return {};

    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);

    constexpr size_t kChunkSize = 1024 * 1024;
    std::vector<char> buf(kChunkSize);
    while (file.read(buf.data(), static_cast<std::streamsize>(kChunkSize)) || file.gcount() > 0) {
        EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(file.gcount()));
        if (file.eof()) break;
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_DigestFinal_ex(ctx, hash, &hash_len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream ss;
    for (unsigned int i = 0; i < hash_len; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(hash[i]);
    }
    return ss.str();
}

} // namespace

std::string canonical_model_filename(const std::string& model_ref) {
    const std::string clean = sanitize_token(model_ref);
    if (clean.empty()) return {};
    return fs::path(clean).filename().string();
}

bool is_safe_model_filename(const std::string& filename) {
    const std::string clean = sanitize_token(filename);
    if (clean.empty()) return false;

    if (clean.find('/') != std::string::npos ||
        clean.find('\\') != std::string::npos) {
        return false;
    }
    if (clean.find("..") != std::string::npos) return false;
    if (clean.find(':') != std::string::npos) return false;

    fs::path p(clean);
    if (p.has_parent_path() || p.has_root_path()) return false;

    static const std::regex allowed_re(R"(^[A-Za-z0-9._+-]+$)");
    if (!std::regex_match(clean, allowed_re)) return false;

    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext != ".gguf") return false;
    return true;
}

std::vector<std::string> expand_model_shards(const std::string& filename) {
    std::string clean = canonical_model_filename(filename);
    if (clean.empty()) return {};

    static const std::regex shard_re(
        R"((.*)-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
    std::smatch match;
    if (!std::regex_match(clean, match, shard_re)) {
        return {clean};
    }

    std::string base = match[1].str();
    int count = 0;
    try {
        count = std::stoi(match[3].str());
    } catch (...) {
        return {clean};
    }
    if (count <= 1 || count > 10000) return {clean};

    std::vector<std::string> shards;
    shards.reserve(static_cast<size_t>(count));
    for (int i = 1; i <= count; ++i) {
        std::ostringstream ss;
        ss << base << "-" << std::setw(5) << std::setfill('0') << i
           << "-of-" << std::setw(5) << std::setfill('0') << count << ".gguf";
        shards.push_back(ss.str());
    }
    return shards;
}

std::vector<StoredModel> list_models_in_dir(const std::string& models_dir) {
    std::vector<StoredModel> result;
    std::error_code ec;

    if (models_dir.empty()) return result;
    if (!fs::exists(models_dir, ec)) return result;

    for (const auto& entry : fs::recursive_directory_iterator(models_dir, ec)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".gguf") continue;

        StoredModel m;
        m.model_path = entry.path().filename().string();
        m.size_bytes = static_cast<int64_t>(entry.file_size(ec));
        m.sha256 = compute_sha256(entry.path().string());

        static const std::regex shard_re(
            R"(.*-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
        std::smatch match;
        if (std::regex_match(m.model_path, match, shard_re)) {
            try {
                m.shard_count = std::stoi(match[2].str());
            } catch (...) {
                m.shard_count = 1;
            }
        } else {
            m.shard_count = 1;
        }

        result.push_back(std::move(m));
    }

    std::sort(result.begin(), result.end(),
              [](const StoredModel& a, const StoredModel& b) {
                  return a.model_path < b.model_path;
              });
    return result;
}

std::optional<StoredModel> find_model_in_dir(const std::string& models_dir,
                                             const std::string& filename) {
    const std::string needle = canonical_model_filename(filename);
    if (!is_safe_model_filename(needle)) return std::nullopt;

    auto models = list_models_in_dir(models_dir);
    auto it = std::find_if(models.begin(), models.end(),
                           [&](const StoredModel& m) {
                               return m.model_path == needle;
                           });
    if (it == models.end()) return std::nullopt;
    return *it;
}

} // namespace mm
