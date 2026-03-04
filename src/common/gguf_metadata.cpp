#include "common/gguf_metadata.hpp"

#include "common/util.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace mm {

namespace {

std::string sanitize_model_path(std::string path) {
    path = util::trim(path);
    if (path.size() >= 2) {
        if ((path.front() == '"' && path.back() == '"') ||
            (path.front() == '\'' && path.back() == '\'')) {
            path = util::trim(path.substr(1, path.size() - 2));
        }
    }
    return path;
}

bool filename_suggests_reasoning(const std::string& fname) {
    std::string low = fname;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    return low.find("qwq")      != std::string::npos ||
           low.find("-r1")      != std::string::npos ||
           low.find("_r1")      != std::string::npos ||
           low.find("sky-t1")   != std::string::npos ||
           low.find("marco-o1") != std::string::npos ||
           low.find("thinking") != std::string::npos;
}

bool filename_suggests_tools(const std::string& fname) {
    std::string low = fname;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    return low.find("hermes")      != std::string::npos ||
           low.find("functionary") != std::string::npos ||
           low.find("xlam")        != std::string::npos ||
           low.find("hammer")      != std::string::npos;
}

} // namespace

std::string resolve_model_path_for_metadata(const std::string& model_path,
                                            const std::string& models_dir) {
    const std::string cleaned = sanitize_model_path(model_path);
    if (cleaned.empty()) return {};

    std::error_code ec;
    fs::path direct(cleaned);
    if (fs::exists(direct, ec) && fs::is_regular_file(direct, ec)) {
        return direct.lexically_normal().string();
    }

    if (!models_dir.empty()) {
        fs::path fallback = fs::path(models_dir) / direct.filename();
        ec.clear();
        if (fs::exists(fallback, ec) && fs::is_regular_file(fallback, ec)) {
            return fallback.lexically_normal().string();
        }
    }

    return {};
}

ModelCapabilityInfo inspect_model_capabilities(const std::string& model_path,
                                               const std::string& models_dir) {
    ModelCapabilityInfo info;

    const std::string cleaned = sanitize_model_path(model_path);
    const std::string resolved = resolve_model_path_for_metadata(cleaned, models_dir);
    if (resolved.empty()) {
        if (!cleaned.empty()) {
            info.warnings.push_back("Model file could not be resolved for metadata inspection.");
            const auto stem = fs::path(cleaned).stem().string();
            if (filename_suggests_tools(stem) || filename_suggests_reasoning(stem)) {
                info.used_filename_heuristics = true;
                info.supports_tool_calls = filename_suggests_tools(stem);
                info.supports_reasoning = filename_suggests_reasoning(stem);
            }
        }
        return info;
    }

    info.source_path = resolved;

    std::ifstream f(resolved, std::ios::binary);
    if (!f) {
        info.warnings.push_back("Model file could not be opened for metadata inspection.");
        return info;
    }

    char magic[4] = {};
    f.read(magic, 4);
    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        info.warnings.push_back("Resolved model file is not a GGUF file.");
        return info;
    }

    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version < 1 || version > 3) {
        info.warnings.push_back("Unsupported GGUF version.");
        return info;
    }

    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    f.read(reinterpret_cast<char*>(&n_tensors), 8);
    f.read(reinterpret_cast<char*>(&n_kv), 8);

    info.metadata_found = true;

    for (uint64_t i = 0; i < n_kv && f.good(); ++i) {
        uint64_t key_len = 0;
        f.read(reinterpret_cast<char*>(&key_len), 8);
        if (key_len > 512 || !f.good()) break;

        std::string key(key_len, '\0');
        f.read(key.data(), static_cast<std::streamsize>(key_len));

        uint32_t vtype = 0;
        f.read(reinterpret_cast<char*>(&vtype), 4);
        if (!f.good()) break;

        const bool is_ctx = key.find(".context_length") != std::string::npos;
        const bool is_tmpl = key == "tokenizer.chat_template" ||
                             key == "tokenizer.ggml.chat_template";

        auto skip_str = [&]() -> bool {
            uint64_t slen = 0;
            f.read(reinterpret_cast<char*>(&slen), 8);
            if (slen > 4u * 1024u * 1024u || !f.good()) return false;
            f.seekg(static_cast<std::streamoff>(slen), std::ios::cur);
            return f.good();
        };

        auto skip_val = [&]() -> bool {
            switch (vtype) {
                case 0:
                case 1:
                case 7:
                    f.seekg(1, std::ios::cur);
                    return true;
                case 2:
                case 3:
                    f.seekg(2, std::ios::cur);
                    return true;
                case 4:
                case 5:
                case 6:
                    f.seekg(4, std::ios::cur);
                    return true;
                case 8:
                    return skip_str();
                case 10:
                case 11:
                case 12:
                    f.seekg(8, std::ios::cur);
                    return true;
                default:
                    return false;
            }
        };

        if (is_ctx) {
            if (vtype == 4) {
                uint32_t v = 0;
                f.read(reinterpret_cast<char*>(&v), 4);
                info.n_ctx_train = v;
            } else if (vtype == 5) {
                int32_t v = 0;
                f.read(reinterpret_cast<char*>(&v), 4);
                info.n_ctx_train = v;
            } else if (vtype == 10) {
                uint64_t v = 0;
                f.read(reinterpret_cast<char*>(&v), 8);
                info.n_ctx_train = static_cast<int64_t>(v);
            } else if (vtype == 11) {
                int64_t v = 0;
                f.read(reinterpret_cast<char*>(&v), 8);
                info.n_ctx_train = v;
            } else if (!skip_val()) {
                break;
            }
        } else if (is_tmpl && vtype == 8) {
            uint64_t slen = 0;
            f.read(reinterpret_cast<char*>(&slen), 8);
            if (slen > 4u * 1024u * 1024u || !f.good()) break;
            std::string tmpl(slen, '\0');
            f.read(tmpl.data(), static_cast<std::streamsize>(slen));

            if (tmpl.find("tool_call") != std::string::npos ||
                tmpl.find("[TOOL_CALLS]") != std::string::npos ||
                tmpl.find("python_tag") != std::string::npos ||
                tmpl.find("function_call") != std::string::npos) {
                info.supports_tool_calls = true;
            }
            if (tmpl.find("<think>") != std::string::npos ||
                tmpl.find("<thinking>") != std::string::npos) {
                info.supports_reasoning = true;
            }
        } else if (!skip_val()) {
            break;
        }

        if (!f.good()) break;
    }

    const auto stem = fs::path(resolved).stem().string();
    const bool tool_hint = filename_suggests_tools(stem);
    const bool reasoning_hint = filename_suggests_reasoning(stem);
    if ((!info.supports_tool_calls && tool_hint) || (!info.supports_reasoning && reasoning_hint)) {
        info.used_filename_heuristics = true;
        info.supports_tool_calls = info.supports_tool_calls || tool_hint;
        info.supports_reasoning = info.supports_reasoning || reasoning_hint;
    }

    if (info.metadata_found && info.n_ctx_train <= 0) {
        info.warnings.push_back("GGUF metadata did not expose a usable training context length.");
    }

    return info;
}

} // namespace mm
