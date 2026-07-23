#include "aio/aio_config.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <system_error>
#include <utility>

namespace mm {

namespace {

std::string trim(std::string_view value) {
    std::size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    std::size_t last = value.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }
    return std::string(value.substr(first, last - first));
}

std::string lower(std::string_view value) {
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string strip_inline_comment(std::string_view line) {
    char quote = 0;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote != 0) {
            if (quote == '"' && c == '\\' && i + 1 < line.size()) {
                ++i;
            } else if (c == quote) {
                quote = 0;
            }
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#') {
            return std::string(line.substr(0, i));
        }
    }
    return std::string(line);
}

bool decode_scalar(std::string_view raw, std::string* value) {
    std::string parsed = trim(raw);
    if (parsed.empty()) {
        *value = {};
        return true;
    }
    if (parsed.front() != '"' && parsed.front() != '\'') {
        *value = std::move(parsed);
        return true;
    }
    if (parsed.size() < 2 || parsed.back() != parsed.front()) return false;
    *value = parsed.substr(1, parsed.size() - 2);
    return true;
}

bool parse_int64(std::string_view value, int64_t* out) {
    const std::string text = trim(value);
    if (text.empty()) return false;
    int64_t parsed = 0;
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return false;
    }
    *out = parsed;
    return true;
}

bool parse_bool(std::string_view value, bool* out) {
    const std::string text = lower(trim(value));
    if (text == "true" || text == "yes" || text == "on" || text == "1") {
        *out = true;
        return true;
    }
    if (text == "false" || text == "no" || text == "off" || text == "0") {
        *out = false;
        return true;
    }
    return false;
}

bool parse_string_list(std::string_view value,
                       std::vector<std::string>* out) {
    std::string text = trim(value);
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
        text = trim(std::string_view(text).substr(1, text.size() - 2));
    }
    if (text.empty()) {
        out->clear();
        return true;
    }

    std::vector<std::string> parsed;
    std::size_t item_start = 0;
    char quote = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const bool at_end = i == text.size();
        const char c = at_end ? ',' : text[i];
        if (!at_end && quote != 0) {
            if (quote == '"' && c == '\\' && i + 1 < text.size()) {
                ++i;
            } else if (c == quote) {
                quote = 0;
            }
            continue;
        }
        if (!at_end && (c == '"' || c == '\'')) {
            quote = c;
            continue;
        }
        if (c != ',') continue;
        std::string item;
        if (!decode_scalar(
                std::string_view(text).substr(item_start, i - item_start),
                &item)) {
            return false;
        }
        if (item.empty()) return false;
        parsed.push_back(std::move(item));
        item_start = i + 1;
    }
    if (quote != 0) return false;
    *out = std::move(parsed);
    return true;
}

void add_issue(std::vector<AioConfigIssue>* issues,
               std::string source,
               std::string key,
               std::string message) {
    issues->push_back(AioConfigIssue{
        std::move(source), std::move(key), std::move(message)});
}

bool apply_setting(AioConfig* config,
                   const std::string& section,
                   const std::string& key,
                   const std::string& value,
                   const std::string& source,
                   std::vector<AioConfigIssue>* issues) {
    const std::string dotted = section + "." + key;
    auto invalid = [&](const std::string& expected) {
        add_issue(issues, source, dotted,
                  "invalid value '" + value + "' (expected " + expected + ")");
    };
    auto assign_bool = [&](bool* target) {
        bool parsed = false;
        if (!parse_bool(value, &parsed)) {
            invalid("boolean");
            return;
        }
        *target = parsed;
    };
    auto assign_integer = [&](int64_t minimum,
                              int64_t maximum,
                              auto setter) {
        int64_t parsed = 0;
        if (!parse_int64(value, &parsed) || parsed < minimum || parsed > maximum) {
            invalid("integer in " + std::to_string(minimum) + ".." +
                    std::to_string(maximum));
            return;
        }
        setter(parsed);
    };
    auto assign_port = [&](uint16_t* target, bool zero_allowed = false) {
        assign_integer(zero_allowed ? 0 : 1, 65535,
                       [&](int64_t parsed) {
                           *target = static_cast<uint16_t>(parsed);
                       });
    };

    if (section == "shared") {
        if (key == "data_dir") config->shared.data_dir = value;
        else if (key == "models_dir") config->shared.models_dir = value;
        else if (key == "log_file") config->shared.log_file = value;
        else return false;
        return true;
    }

    if (section == "control") {
        if (key == "bind_host") config->control.bind_host = value;
        else if (key == "listen_port") assign_port(&config->control.listen_port);
        else if (key == "openai_compat_port")
            assign_port(&config->control.openai_compat_port, true);
        else if (key == "node_health_poll_interval_s") {
            assign_integer(1, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->control.node_health_poll_interval_s =
                                   static_cast<uint32_t>(parsed);
                           });
        } else if (key == "node_offline_after_s") {
            assign_integer(1, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->control.node_offline_after_s =
                                   static_cast<uint32_t>(parsed);
                           });
        } else if (key == "external_api_token") {
            config->control.external_api_token = value;
        } else if (key == "tts_enabled") {
            assign_bool(&config->control.tts.enabled);
        } else if (key == "tts_service_url") {
            config->control.tts.service_url = value;
        } else if (key == "tts_service_command") {
            config->control.tts.service_command = value;
        } else if (key == "tts_cache_dir") {
            config->control.tts.cache_dir = value;
        } else if (key == "tts_voice_design_model_id") {
            config->control.tts.voice_design_model_id = value;
        } else if (key == "tts_clone_model_id") {
            config->control.tts.clone_model_id = value;
        } else if (key == "tts_custom_voice_model_id") {
            config->control.tts.custom_voice_model_id = value;
        } else if (key == "tts_cache_ttl_ms") {
            assign_integer(0, std::numeric_limits<int64_t>::max(),
                           [&](int64_t parsed) {
                               config->control.tts.cache_ttl_ms = parsed;
                           });
        } else if (key == "tts_timeout_s") {
            assign_integer(1, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->control.tts.timeout_s =
                                   static_cast<int>(parsed);
                           });
        } else {
            return false;
        }
        return true;
    }

    if (section == "node") {
        if (key == "llama_server_path") config->node.llama_server_path = value;
        else if (key == "llama_auto_provision")
            assign_bool(&config->node.llama_auto_provision);
        else if (key == "llama_provision_dir")
            config->node.llama_provision_dir = value;
        else if (key == "llama_install_method")
            config->node.llama_install_method = lower(trim(value));
        else if (key == "llama_version") config->node.llama_version = value;
        else if (key == "llama_accelerator")
            config->node.llama_accelerator = lower(trim(value));
        else if (key == "llama_cuda_arch") config->node.llama_cuda_arch = value;
        else if (key == "llama_cmake_args") {
            if (!parse_string_list(value, &config->node.llama_cmake_args)) {
                invalid("a TOML string array or comma-separated string list");
            }
        } else if (key == "llama_build_jobs") {
            assign_integer(0, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->node.llama_build_jobs =
                                   static_cast<int>(parsed);
                           });
        } else if (key == "llama_update_policy") {
            config->node.llama_update_policy = lower(trim(value));
        } else if (key == "llama_update_check") {
            assign_bool(&config->node.llama_update_check);
        } else if (key == "llama_update_check_interval_hours") {
            assign_integer(1, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->node.llama_update_check_interval_hours =
                                   static_cast<int>(parsed);
                           });
        } else if (key == "runtime_network_policy") {
            const auto policy = aio_runtime_network_policy_from_string(value);
            if (!policy) invalid("prompt, auto, or offline");
            else config->node.runtime_network_policy = *policy;
        } else if (key == "node_gpu_count") {
            assign_integer(0, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->node.node_gpu_count =
                                   static_cast<int>(parsed);
                           });
        } else if (key == "runtime_port_range_start") {
            assign_port(&config->node.runtime_port_range_start);
        } else if (key == "runtime_port_range_end") {
            assign_port(&config->node.runtime_port_range_end);
        } else if (key == "max_slots") {
            assign_integer(1, std::numeric_limits<int>::max(),
                           [&](int64_t parsed) {
                               config->node.max_slots = static_cast<int>(parsed);
                           });
        } else if (key == "kv_cache_dir") {
            config->node.kv_cache_dir = value;
        } else if (key == "model_cache_min_free_mb") {
            assign_integer(0, std::numeric_limits<int64_t>::max(),
                           [&](int64_t parsed) {
                               config->node.model_cache_min_free_mb = parsed;
                           });
        } else if (key == "model_cache_clear_on_shutdown") {
            assign_bool(&config->node.model_cache_clear_on_shutdown);
        } else {
            return false;
        }
        return true;
    }

    if (section == "cluster") {
        if (key == "enabled") assign_bool(&config->cluster.enabled);
        else if (key == "discovery_enabled")
            assign_bool(&config->cluster.discovery_enabled);
        else if (key == "discovery_port")
            assign_port(&config->cluster.discovery_port);
        else if (key == "pairing_key") config->cluster.pairing_key = value;
        else return false;
        return true;
    }

    return false;
}

bool known_section(const std::string& section) {
    return section == "shared" || section == "control" ||
           section == "node" || section == "cluster";
}

void parse_file(const std::filesystem::path& path,
                AioConfig* config,
                std::vector<AioConfigIssue>* issues) {
    std::ifstream input(path);
    if (!input.is_open()) {
        add_issue(issues, path.string(), {}, "could not open AIO config file");
        return;
    }

    std::string section;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string source =
            path.string() + ":" + std::to_string(line_number);
        const std::string text = trim(strip_inline_comment(line));
        if (text.empty()) continue;

        if (text.front() == '[') {
            if (text.size() < 3 || text.back() != ']') {
                add_issue(issues, source, {}, "malformed section header");
                section.clear();
                continue;
            }
            section = lower(trim(
                std::string_view(text).substr(1, text.size() - 2)));
            if (!known_section(section)) {
                add_issue(issues, source, section,
                          "unknown AIO config section");
                section.clear();
            }
            continue;
        }

        const auto equals = text.find('=');
        if (equals == std::string::npos) {
            add_issue(issues, source, {}, "expected key = value");
            continue;
        }
        if (section.empty()) {
            add_issue(issues, source, {},
                      "AIO config keys must be inside [shared], [control], "
                      "[node], or [cluster]");
            continue;
        }

        const std::string key = lower(trim(
            std::string_view(text).substr(0, equals)));
        std::string value;
        if (key.empty() ||
            !decode_scalar(std::string_view(text).substr(equals + 1), &value)) {
            add_issue(issues, source, section + "." + key,
                      "malformed key or quoted value");
            continue;
        }
        if (!apply_setting(config, section, key, value, source, issues)) {
            add_issue(issues, source, section + "." + key,
                      "unknown AIO config key");
        }
    }
}

struct SettingName {
    const char* section;
    const char* key;
};

const std::vector<SettingName>& setting_names() {
    static const std::vector<SettingName> names = {
        {"shared", "data_dir"},
        {"shared", "models_dir"},
        {"shared", "log_file"},
        {"control", "bind_host"},
        {"control", "listen_port"},
        {"control", "openai_compat_port"},
        {"control", "node_health_poll_interval_s"},
        {"control", "node_offline_after_s"},
        {"control", "external_api_token"},
        {"control", "tts_enabled"},
        {"control", "tts_service_url"},
        {"control", "tts_service_command"},
        {"control", "tts_cache_dir"},
        {"control", "tts_voice_design_model_id"},
        {"control", "tts_clone_model_id"},
        {"control", "tts_custom_voice_model_id"},
        {"control", "tts_cache_ttl_ms"},
        {"control", "tts_timeout_s"},
        {"node", "llama_server_path"},
        {"node", "llama_auto_provision"},
        {"node", "llama_provision_dir"},
        {"node", "llama_install_method"},
        {"node", "llama_version"},
        {"node", "llama_accelerator"},
        {"node", "llama_cuda_arch"},
        {"node", "llama_cmake_args"},
        {"node", "llama_build_jobs"},
        {"node", "llama_update_policy"},
        {"node", "llama_update_check"},
        {"node", "llama_update_check_interval_hours"},
        {"node", "runtime_network_policy"},
        {"node", "node_gpu_count"},
        {"node", "runtime_port_range_start"},
        {"node", "runtime_port_range_end"},
        {"node", "max_slots"},
        {"node", "kv_cache_dir"},
        {"node", "model_cache_min_free_mb"},
        {"node", "model_cache_clear_on_shutdown"},
        {"cluster", "enabled"},
        {"cluster", "discovery_enabled"},
        {"cluster", "discovery_port"},
        {"cluster", "pairing_key"},
    };
    return names;
}

std::string environment_name(const SettingName& setting) {
    std::string name = "MM_AIO_";
    name += setting.section;
    name += '_';
    name += setting.key;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return name;
}

void apply_environment(AioConfig* config,
                       std::vector<AioConfigIssue>* issues) {
    for (const auto& setting : setting_names()) {
        const std::string name = environment_name(setting);
        const char* raw = std::getenv(name.c_str());
        if (!raw) continue;
        if (!apply_setting(config, setting.section, setting.key, raw, name, issues)) {
            add_issue(issues, name,
                      std::string(setting.section) + "." + setting.key,
                      "unsupported AIO environment override");
        }
    }
}

std::filesystem::path absolute_normalized(const std::filesystem::path& path) {
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    return (ec ? path : absolute).lexically_normal();
}

std::optional<std::filesystem::path> find_upward(
    const std::filesystem::path& search_start,
    std::size_t limit,
    std::vector<AioConfigIssue>* issues) {
    if (limit == 0) return std::nullopt;

    std::error_code ec;
    std::filesystem::path directory = search_start;
    if (directory.empty()) directory = std::filesystem::current_path(ec);
    if (ec) {
        add_issue(issues, "upward-search", {},
                  "could not determine current directory: " + ec.message());
        return std::nullopt;
    }
    directory = absolute_normalized(directory);

    for (std::size_t i = 0; i < limit; ++i) {
        const auto candidate = directory / "mantic-mind-aio.toml";
        if (std::filesystem::exists(candidate, ec)) {
            return absolute_normalized(candidate);
        }
        ec.clear();
        const auto parent = directory.parent_path();
        if (parent.empty() || parent == directory) break;
        directory = parent;
    }
    return std::nullopt;
}

bool has_issue_for_key(const std::vector<AioConfigIssue>& issues,
                       std::string_view key) {
    return std::any_of(issues.begin(), issues.end(), [&](const auto& issue) {
        return issue.key == key;
    });
}

void validation_issue(std::vector<AioConfigIssue>* issues,
                      std::string key,
                      std::string message) {
    add_issue(issues, "validation", std::move(key), std::move(message));
}

bool valid_raw_bind_host(std::string_view value) {
    const std::string host = trim(value);
    if (host.empty() || host.find("://") != std::string::npos ||
        host.find_first_of("/?#@[]") != std::string::npos) {
        return false;
    }
    const auto first_colon = host.find(':');
    // A single colon denotes host:port, while multiple colons are a raw IPv6
    // literal. The port belongs in control.listen_port, not bind_host.
    return first_colon == std::string::npos || first_colon != host.rfind(':');
}

} // namespace

std::string_view to_string(AioRuntimeNetworkPolicy policy) noexcept {
    switch (policy) {
        case AioRuntimeNetworkPolicy::Allow: return "auto";
        case AioRuntimeNetworkPolicy::Deny: return "offline";
        case AioRuntimeNetworkPolicy::Prompt:
        default: return "prompt";
    }
}

std::optional<AioRuntimeNetworkPolicy> aio_runtime_network_policy_from_string(
    std::string_view value) {
    const std::string normalized = lower(trim(value));
    if (normalized == "prompt") return AioRuntimeNetworkPolicy::Prompt;
    if (normalized == "auto" || normalized == "allow") {
        return AioRuntimeNetworkPolicy::Allow;
    }
    if (normalized == "offline" || normalized == "deny") {
        return AioRuntimeNetworkPolicy::Deny;
    }
    return std::nullopt;
}

bool aio_host_is_loopback(std::string_view host_or_url) {
    std::string host = lower(trim(host_or_url));
    const auto scheme = host.find("://");
    if (scheme != std::string::npos) host.erase(0, scheme + 3);
    const auto path = host.find_first_of("/?#");
    if (path != std::string::npos) host.resize(path);
    const auto at = host.rfind('@');
    if (at != std::string::npos) host.erase(0, at + 1);

    if (!host.empty() && host.front() == '[') {
        const auto close = host.find(']');
        if (close == std::string::npos) return false;
        host = host.substr(1, close - 1);
    } else {
        const auto first_colon = host.find(':');
        if (first_colon != std::string::npos &&
            first_colon == host.rfind(':')) {
            host.resize(first_colon);
        }
    }
    if (!host.empty() && host.back() == '.') host.pop_back();

    if (host == "localhost" || host == "::1" ||
        host.rfind("::1%", 0) == 0) {
        return true;
    }
    if (host.rfind("::ffff:127.", 0) == 0) return true;
    if (host.rfind("127.", 0) != 0) return false;

    // Fail closed on malformed 127/8 text rather than treating an arbitrary
    // hostname beginning with "127." as loopback.
    int octets = 0;
    std::size_t start = 0;
    while (start <= host.size()) {
        const auto dot = host.find('.', start);
        const auto end = dot == std::string::npos ? host.size() : dot;
        if (end == start) return false;
        int64_t octet = 0;
        if (!parse_int64(std::string_view(host).substr(start, end - start), &octet) ||
            octet < 0 || octet > 255) {
            return false;
        }
        ++octets;
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return octets == 4;
}

std::vector<AioConfigIssue> validate_aio_config(const AioConfig& config) {
    std::vector<AioConfigIssue> issues;

    if (trim(config.shared.data_dir).empty())
        validation_issue(&issues, "shared.data_dir", "must not be empty");
    if (trim(config.shared.models_dir).empty())
        validation_issue(&issues, "shared.models_dir", "must not be empty");
    if (trim(config.shared.log_file).empty())
        validation_issue(&issues, "shared.log_file", "must not be empty");
    if (trim(config.control.bind_host).empty()) {
        validation_issue(&issues, "control.bind_host", "must not be empty");
    } else if (!valid_raw_bind_host(config.control.bind_host)) {
        validation_issue(&issues, "control.bind_host",
                         "must be a raw hostname or IP address without a scheme or port");
    }

    const bool loopback = aio_host_is_loopback(config.control.bind_host);
    if (!config.cluster.enabled && !loopback) {
        validation_issue(
            &issues, "control.bind_host",
            "must be loopback when cluster.enabled is false");
    }
    if (!config.cluster.enabled && config.cluster.discovery_enabled) {
        validation_issue(
            &issues, "cluster.discovery_enabled",
            "must be false when cluster.enabled is false");
    }
    if (!loopback) {
        if (trim(config.control.external_api_token).empty()) {
            validation_issue(
                &issues, "control.external_api_token",
                "is required when control.bind_host is non-loopback");
        }
        if (trim(config.cluster.pairing_key).empty()) {
            validation_issue(
                &issues, "cluster.pairing_key",
                "is required when control.bind_host is non-loopback");
        }
    }
    if (config.control.listen_port == 0) {
        validation_issue(&issues, "control.listen_port", "must be in 1..65535");
    }
    if (config.control.openai_compat_port != 0 &&
        config.control.openai_compat_port == config.control.listen_port) {
        validation_issue(
            &issues, "control.openai_compat_port",
            "must differ from control.listen_port, or be 0 to disable it");
    }
    const auto maximum_poll_seconds =
        static_cast<uint32_t>(std::numeric_limits<int>::max());
    if (config.control.node_health_poll_interval_s < 1 ||
        config.control.node_health_poll_interval_s > maximum_poll_seconds) {
        validation_issue(
            &issues, "control.node_health_poll_interval_s",
            "must be in 1.." + std::to_string(maximum_poll_seconds));
    }
    if (config.control.node_offline_after_s < 1 ||
        config.control.node_offline_after_s > maximum_poll_seconds) {
        validation_issue(
            &issues, "control.node_offline_after_s",
            "must be in 1.." + std::to_string(maximum_poll_seconds));
    }

    const uint16_t range_start = config.node.runtime_port_range_start;
    const uint16_t range_end = config.node.runtime_port_range_end;
    if (range_start == 0 || range_end == 0 || range_start > range_end) {
        validation_issue(
            &issues, "node.runtime_port_range_start",
            "runtime port range must be nonzero and start must not exceed end");
    } else {
        const auto range_size = static_cast<uint32_t>(range_end) -
                                static_cast<uint32_t>(range_start) + 1U;
        if (config.node.max_slots <= 0 ||
            static_cast<uint64_t>(config.node.max_slots) > range_size) {
            validation_issue(
                &issues, "node.max_slots",
                "must be positive and fit within the runtime port range");
        }
        auto range_contains = [&](uint16_t port) {
            return port != 0 && port >= range_start && port <= range_end;
        };
        if (range_contains(config.control.listen_port)) {
            validation_issue(
                &issues, "control.listen_port",
                "collides with the node runtime port range");
        }
        if (range_contains(config.control.openai_compat_port)) {
            validation_issue(
                &issues, "control.openai_compat_port",
                "collides with the node runtime port range");
        }
        if (config.cluster.enabled && config.cluster.discovery_enabled &&
            range_contains(config.cluster.discovery_port)) {
            validation_issue(
                &issues, "cluster.discovery_port",
                "collides with the node runtime port range");
        }
    }

    if (config.cluster.enabled && config.cluster.discovery_enabled) {
        if (config.cluster.discovery_port == 0) {
            validation_issue(
                &issues, "cluster.discovery_port", "must be in 1..65535");
        }
        if (config.cluster.discovery_port == config.control.listen_port ||
            (config.control.openai_compat_port != 0 &&
             config.cluster.discovery_port == config.control.openai_compat_port)) {
            validation_issue(
                &issues, "cluster.discovery_port",
                "must differ from active control listener ports");
        }
    }

    const std::string install_method = lower(trim(config.node.llama_install_method));
    if (install_method != "auto" && install_method != "release" &&
        install_method != "source") {
        validation_issue(
            &issues, "node.llama_install_method",
            "must be auto, release, or source");
    }
    const std::string update_policy = lower(trim(config.node.llama_update_policy));
    if (update_policy != "prompt" && update_policy != "auto" &&
        update_policy != "manual") {
        validation_issue(
            &issues, "node.llama_update_policy",
            "must be prompt, auto, or manual");
    }
    if (trim(config.node.llama_server_path).empty()) {
        validation_issue(
            &issues, "node.llama_server_path", "must not be empty");
    }
    if (config.node.llama_build_jobs < 0) {
        validation_issue(
            &issues, "node.llama_build_jobs", "must not be negative");
    }
    if (config.node.llama_update_check_interval_hours < 1) {
        validation_issue(
            &issues, "node.llama_update_check_interval_hours",
            "must be at least 1");
    }
    if (config.node.node_gpu_count < 0) {
        validation_issue(
            &issues, "node.node_gpu_count", "must not be negative");
    }
    if (config.node.model_cache_min_free_mb < 0) {
        validation_issue(
            &issues, "node.model_cache_min_free_mb", "must not be negative");
    }
    if (config.control.tts.enabled &&
        trim(config.control.tts.service_url).empty() &&
        trim(config.control.tts.service_command).empty()) {
        validation_issue(
            &issues, "control.tts_service_url",
            "tts_service_url or tts_service_command is required when TTS is enabled");
    }
    if (config.control.tts.timeout_s < 1) {
        validation_issue(
            &issues, "control.tts_timeout_s", "must be at least 1");
    }
    if (config.control.tts.cache_ttl_ms < 0) {
        validation_issue(
            &issues, "control.tts_cache_ttl_ms", "must not be negative");
    }

    return issues;
}

AioConfigLoadResult load_aio_config(const AioConfigLoadOptions& options) {
    AioConfigLoadResult result;
    std::optional<std::filesystem::path> selected;

    if (options.explicit_path.has_value()) {
        if (options.explicit_path->empty()) {
            add_issue(&result.issues, "--config", {},
                      "explicit AIO config path must not be empty");
        } else {
            selected = absolute_normalized(*options.explicit_path);
        }
    } else if (options.use_environment) {
        const char* env_path = std::getenv("MM_AIO_CONFIG_FILE");
        if (env_path && *env_path) {
            selected = absolute_normalized(std::filesystem::path(env_path));
        }
    }

    if (!selected && !options.explicit_path.has_value()) {
        selected = find_upward(
            options.search_start, options.upward_search_limit, &result.issues);
    }

    if (selected) {
        result.source_path = *selected;
        std::error_code ec;
        if (!std::filesystem::is_regular_file(*selected, ec)) {
            add_issue(&result.issues, selected->string(), {},
                      "AIO config file does not exist or is not a regular file");
        } else {
            parse_file(*selected, &result.config, &result.issues);
        }
    }

    if (options.use_environment) {
        apply_environment(&result.config, &result.issues);
    }

    auto validation = validate_aio_config(result.config);
    for (auto& issue : validation) {
        // Avoid making one malformed parsed value unnecessarily noisy by also
        // reporting the default-retained field as invalid.
        if (!has_issue_for_key(result.issues, issue.key)) {
            result.issues.push_back(std::move(issue));
        }
    }
    return result;
}

ControlConfig make_control_config(const AioConfig& config) {
    ControlConfig out;
    out.listen_port = config.control.listen_port;
    out.openai_compat_port = config.control.openai_compat_port;
    out.node_health_poll_interval_s =
        config.control.node_health_poll_interval_s;
    out.node_offline_after_s = config.control.node_offline_after_s;
    out.external_api_token = config.control.external_api_token;
    out.tts = config.control.tts;
    out.data_dir = config.shared.data_dir;
    out.models_dir = config.shared.models_dir;
    out.log_file = config.shared.log_file;
    out.pairing_key = config.cluster.pairing_key;
    out.discovery_port = config.cluster.discovery_port;
    return out;
}

NodeConfig make_node_config(const AioConfig& config) {
    NodeConfig out;
    // AIO never starts the legacy node listener/registration/discovery paths.
    out.control_url.clear();
    out.control_api_key.clear();
    out.listen_port = 0;
    out.pairing_key.clear();
    out.discovery_port = 0;

    out.llama_server_path = config.node.llama_server_path;
    // Startup must remain network-silent unless the user explicitly selected
    // the auto policy. Prompt/offline still resolve an existing executable;
    // consent-gated actions are handled by LocalNodeOperations.
    out.llama_auto_provision =
        config.node.runtime_network_policy == AioRuntimeNetworkPolicy::Allow &&
        config.node.llama_auto_provision;
    out.llama_provision_dir = config.node.llama_provision_dir.empty()
        ? (std::filesystem::path(config.shared.data_dir) /
           "runtimes" / "llama.cpp").string()
        : config.node.llama_provision_dir;
    out.llama_install_method = config.node.llama_install_method;
    out.llama_version = config.node.llama_version;
    out.llama_accelerator = config.node.llama_accelerator;
    out.llama_cuda_arch = config.node.llama_cuda_arch;
    out.llama_cmake_args = config.node.llama_cmake_args;
    out.llama_build_jobs = config.node.llama_build_jobs;
    out.llama_update_policy = config.node.llama_update_policy;
    out.llama_update_check =
        config.node.runtime_network_policy == AioRuntimeNetworkPolicy::Allow &&
        config.node.llama_update_check;
    out.llama_update_check_interval_hours =
        config.node.llama_update_check_interval_hours;
    out.node_gpu_count = config.node.node_gpu_count;
    out.runtime_port_range_start = config.node.runtime_port_range_start;
    out.runtime_port_range_end = config.node.runtime_port_range_end;
    out.max_slots = config.node.max_slots;

    out.models_dir = config.shared.models_dir;
    out.data_dir = config.shared.data_dir;
    out.log_file = config.shared.log_file;
    out.kv_cache_dir = config.node.kv_cache_dir.empty()
        ? (std::filesystem::path(config.shared.data_dir) / "kv_cache").string()
        : config.node.kv_cache_dir;
    out.model_cache_min_free_mb = config.node.model_cache_min_free_mb;
    out.model_cache_clear_on_shutdown =
        config.node.model_cache_clear_on_shutdown;
    return out;
}

} // namespace mm
