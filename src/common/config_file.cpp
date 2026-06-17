#include "common/config_file.hpp"
#include "common/logger.hpp"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace mm {

// ── Helpers ────────────────────────────────────────────────────────────────────

static std::string trim_ws(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// Remove an inline `#` comment, but only when the `#` is outside a quoted
// string — values like passwords or URL fragments may legitimately contain it.
static std::string strip_inline_comment(const std::string& line) {
    char quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote != 0) {
            if (quote == '"' && c == '\\') { ++i; continue; } // skip escaped char
            if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#') {
            return line.substr(0, i);
        }
    }
    return line;
}

// ── ConfigFile::load ───────────────────────────────────────────────────────────

bool ConfigFile::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        line = strip_inline_comment(line);
        line = trim_ws(line);
        if (line.empty()) continue;
        if (line.front() == '[') continue; // section header — ignored

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim_ws(line.substr(0, eq));
        std::string val = trim_ws(line.substr(eq + 1));
        val = strip_quotes(val);

        if (!key.empty()) data_[key] = val;
    }
    return true;
}

// ── Accessors ──────────────────────────────────────────────────────────────────

bool ConfigFile::has(const std::string& key) const {
    return data_.count(key) > 0;
}

std::string ConfigFile::get(const std::string& key,
                             const std::string& def) const {
    auto it = data_.find(key);
    return it != data_.end() ? it->second : def;
}

int ConfigFile::get_int(const std::string& key, int def) const {
    auto it = data_.find(key);
    if (it == data_.end()) return def;
    try { return std::stoi(it->second); }
    catch (const std::exception& e) { MM_DEBUG("config get_int('{}'): {}", key, e.what()); return def; }
}

float ConfigFile::get_float(const std::string& key, float def) const {
    auto it = data_.find(key);
    if (it == data_.end()) return def;
    try { return std::stof(it->second); }
    catch (const std::exception& e) { MM_DEBUG("config get_float('{}'): {}", key, e.what()); return def; }
}

bool ConfigFile::get_bool(const std::string& key, bool def) const {
    auto it = data_.find(key);
    if (it == data_.end()) return def;
    const std::string& v = it->second;
    if (v == "true"  || v == "yes" || v == "1") return true;
    if (v == "false" || v == "no"  || v == "0") return false;
    return def;
}

} // namespace mm
