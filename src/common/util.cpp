#include "common/util.hpp"

#include <openssl/rand.h>

#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <stdexcept>

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

// ── HF repo id classification ─────────────────────────────────────────────────
bool is_hf_repo_id(const std::string& ref) {
    // HF repo id: exactly "org/name", each segment [A-Za-z0-9._-]. Anything
    // that looks like a filesystem path is treated as a local model dir.
    if (ref.empty()) return false;
    if (ref.find('\\') != std::string::npos) return false;   // Windows path
    if (ref.front() == '/' || ref.front() == '.') return false;
    if (ref.size() > 1 && ref[1] == ':') return false;        // drive letter

    const size_t slash = ref.find('/');
    if (slash == std::string::npos) return false;             // need org/name
    if (ref.find('/', slash + 1) != std::string::npos) return false; // one slash only

    auto valid_segment = [](const std::string& s) {
        if (s.empty()) return false;
        return std::all_of(s.begin(), s.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '.' || c == '_' || c == '-';
        });
    };
    return valid_segment(ref.substr(0, slash)) && valid_segment(ref.substr(slash + 1));
}

bool model_ref_is_local_path(const std::string& ref) {
    if (ref.empty()) return false;
    if (is_hf_repo_id(ref)) return false;
    if (ref.find('/') != std::string::npos || ref.find('\\') != std::string::npos) return true;
    if (ref.size() >= 2 && std::isalpha(static_cast<unsigned char>(ref[0])) && ref[1] == ':')
        return true;
    return false;
}

std::string model_id_from_ref(const std::string& ref) {
    std::string r = trim(ref);
    if (r.size() >= 2 &&
        ((r.front() == '"' && r.back() == '"') || (r.front() == '\'' && r.back() == '\'')))
        r = trim(r.substr(1, r.size() - 2));
    while (!r.empty() && (r.back() == '/' || r.back() == '\\')) r.pop_back();

    const size_t pos = r.find_last_of("/\\");
    const std::string base = (pos == std::string::npos) ? r : r.substr(pos + 1);

    std::string id;
    id.reserve(base.size());
    for (unsigned char c : base) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == '-')
            id.push_back(static_cast<char>(c));
        else
            id.push_back('_');
    }
    if (id.empty() || id == "." || id == "..") id = "model";
    return id;
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
// API keys are bearer credentials, so they must come from a CSPRNG. Rejection
// sampling keeps the charset distribution unbiased (62 does not divide 256).
std::string generate_api_key(size_t length) {
    static constexpr char kCharset[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    static constexpr size_t kCharsetSize = sizeof(kCharset) - 1;
    static constexpr unsigned char kRejectAbove =
        static_cast<unsigned char>(256 / kCharsetSize * kCharsetSize); // 248

    std::string key;
    key.reserve(length);
    unsigned char buf[64];
    while (key.size() < length) {
        if (RAND_bytes(buf, static_cast<int>(sizeof(buf))) != 1)
            throw std::runtime_error("generate_api_key: RAND_bytes failed");
        for (unsigned char b : buf) {
            if (b >= kRejectAbove) continue;
            key += kCharset[b % kCharsetSize];
            if (key.size() == length) break;
        }
    }
    return key;
}

// ── URL parsing ──────────────────────────────────────────────────────────────
std::pair<std::string, int> parse_url(const std::string& url) {
    std::string s = url;
    int default_port = 80;
    if (s.rfind("https://", 0) == 0) {
        s = s.substr(8);
        default_port = 443;
    } else if (s.rfind("http://", 0) == 0) {
        s = s.substr(7);
    }
    auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);

    auto parse_port = [&](const std::string& text) -> int {
        try {
            int p = std::stoi(text);
            if (p > 0 && p <= 65535) return p;
        } catch (const std::exception&) {}
        return default_port;
    };

    // Bracketed IPv6 literal: [::1] or [::1]:8080
    if (!s.empty() && s.front() == '[') {
        auto close = s.find(']');
        if (close != std::string::npos) {
            std::string host = s.substr(1, close - 1);
            if (close + 1 < s.size() && s[close + 1] == ':')
                return { host, parse_port(s.substr(close + 2)) };
            return { host, default_port };
        }
    }

    auto colon = s.rfind(':');
    // A single colon separates host:port. Multiple colons mean an unbracketed
    // IPv6 address with no port.
    if (colon != std::string::npos && s.find(':') == colon)
        return { s.substr(0, colon), parse_port(s.substr(colon + 1)) };
    return { s, default_port };
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
