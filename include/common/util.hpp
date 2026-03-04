#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace mm::util {

// ── Identity / randomness ────────────────────────────────────────────────────
std::string generate_uuid();          // UUID v4
std::string generate_api_key(size_t length = 32);

// Returns true iff id is a valid agent/resource identifier:
//   - 1–128 characters
//   - Only [a-zA-Z0-9_-]
//   - First character must be alphanumeric
bool is_valid_agent_id(const std::string& id);

// ── Time ─────────────────────────────────────────────────────────────────────
int64_t now_ms();   // milliseconds since Unix epoch
int64_t now_s();    // seconds since Unix epoch

// ── String helpers ────────────────────────────────────────────────────────────
std::string trim(const std::string& s);
std::string trim_left(const std::string& s);
std::string trim_right(const std::string& s);
std::string to_lower(const std::string& s);
std::string to_upper(const std::string& s);

std::vector<std::string> split(const std::string& s, char delim);
std::string join(const std::vector<std::string>& parts, const std::string& sep);

bool starts_with(const std::string& s, const std::string& prefix);
bool ends_with(const std::string& s, const std::string& suffix);
std::string replace_all(const std::string& s, const std::string& from, const std::string& to);

// ── URL / SSE helpers ───────────────────────────────────────────────────────
// Parse "http://host:port/path" → {host, port}.  Defaults to port 80.
std::pair<std::string, int> parse_url(const std::string& url);

// Drain complete SSE "data: ..." lines from buf, returning payloads (after
// the "data: " prefix).  Incomplete trailing data stays in buf.
std::vector<std::string> drain_sse_lines(std::string& buf);

} // namespace mm::util
