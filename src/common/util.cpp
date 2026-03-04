#include "common/util.hpp"

#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cctype>

namespace mm::util {

// ── UUID v4 ───────────────────────────────────────────────────────────────────
std::string generate_uuid() {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<uint32_t> dist32;

    auto r = [&]() -> uint32_t { return dist32(rng); };

    uint32_t d[4] = { r(), r(), r(), r() };

    // Version 4: bits 12-15 of d[1] = 0100
    d[1] = (d[1] & 0xFFFF0FFFu) | 0x00004000u;
    // Variant 1: bits 30-31 of d[2] = 10
    d[2] = (d[2] & 0x3FFFFFFFu) | 0x80000000u;

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << std::setw(8) << d[0]                      << '-';
    oss << std::setw(4) << ((d[1] >> 16) & 0xFFFFu)  << '-';
    oss << std::setw(4) << ( d[1]        & 0xFFFFu)  << '-';
    oss << std::setw(4) << ((d[2] >> 16) & 0xFFFFu)  << '-';
    oss << std::setw(4) << ( d[2]        & 0xFFFFu);
    oss << std::setw(8) << d[3];
    return oss.str();
}

// ── Timestamp ─────────────────────────────────────────────────────────────────
int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int64_t now_s() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

// ── String helpers ────────────────────────────────────────────────────────────
std::string trim_left(const std::string& s) {
    auto it = std::find_if_not(s.begin(), s.end(),
                               [](unsigned char c){ return std::isspace(c); });
    return std::string(it, s.end());
}

std::string trim_right(const std::string& s) {
    auto it = std::find_if_not(s.rbegin(), s.rend(),
                               [](unsigned char c){ return std::isspace(c); });
    return std::string(s.begin(), it.base());
}

std::string trim(const std::string& s) {
    return trim_right(trim_left(s));
}

std::string to_lower(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return r;
}

std::string to_upper(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
    return r;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, delim))
        result.push_back(token);
    return result;
}

std::string join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) result += sep;
        result += parts[i];
    }
    return result;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string replace_all(const std::string& s, const std::string& from, const std::string& to) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.size(), to);
        pos += to.size();
    }
    return result;
}

// ── Agent ID validation ───────────────────────────────────────────────────────
bool is_valid_agent_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    if (!std::isalnum(static_cast<unsigned char>(id[0]))) return false;
    for (unsigned char c : id) {
        if (!std::isalnum(c) && c != '-' && c != '_') return false;
    }
    return true;
}

// ── API key generation ────────────────────────────────────────────────────────
std::string generate_api_key(size_t length) {
    static constexpr char kCharset[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    static constexpr size_t kCharsetSize = sizeof(kCharset) - 1;

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, kCharsetSize - 1);

    std::string key;
    key.reserve(length);
    for (size_t i = 0; i < length; ++i)
        key += kCharset[dist(rng)];
    return key;
}

// ── URL parsing ──────────────────────────────────────────────────────────────
std::pair<std::string, int> parse_url(const std::string& url) {
    std::string s = url;
    if (s.rfind("https://", 0) == 0)      s = s.substr(8);
    else if (s.rfind("http://", 0) == 0)  s = s.substr(7);
    auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    auto colon = s.rfind(':');
    if (colon != std::string::npos)
        return { s.substr(0, colon), std::stoi(s.substr(colon + 1)) };
    return { s, 80 };
}

// ── SSE line draining ────────────────────────────────────────────────────────
std::vector<std::string> drain_sse_lines(std::string& buf) {
    std::vector<std::string> out;
    size_t pos;
    while ((pos = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, pos);
        buf.erase(0, pos + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data: ", 0) == 0)
            out.push_back(line.substr(6));
    }
    return out;
}

} // namespace mm::util
