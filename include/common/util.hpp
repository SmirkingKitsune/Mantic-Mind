#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <optional>

namespace mm::util {

// ── Identity / randomness ────────────────────────────────────────────────────
std::string generate_uuid();          // UUID v4
std::string generate_api_key(size_t length = 32);
std::string hostname();

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

// True when ref looks like a Hugging Face repo id ("org/name", each segment
// [A-Za-z0-9._-]) rather than a local filesystem path.
bool is_hf_repo_id(const std::string& ref);

// True when ref is syntactically a local filesystem path. GGUF filenames are
// local model references even when bare or shaped like "models/name.gguf";
// this check deliberately takes precedence over the legacy org/name heuristic.
bool model_ref_is_local_path(const std::string& ref);

// Resolve a model reference to an existing local file. The reference itself is
// checked first, followed by paths beneath models_dir (including the common
// "models/name.gguf" spelling when models_dir is absolute). Returns a normalized
// absolute path, or nullopt when the reference is node-local/nonexistent here.
std::optional<std::string> resolve_existing_local_model_path(
    const std::string& ref,
    const std::string& models_dir = {});

// A stable, filesystem-safe identity for a model reference: the final path
// component (a file name like "Qwen3-8B.gguf" or a directory name), with any
// character outside [A-Za-z0-9._-] replaced by '_'. Both control and node
// compute the same id from the same ref so they agree on the node-local
// destination folder and the cache dedup key. Handles '/' and '\\'.
std::string model_id_from_ref(const std::string& ref);

// ── URL / SSE helpers ───────────────────────────────────────────────────────
// Parse "http://host:port/path" → {host, port}.  Defaults to port 80.
std::pair<std::string, int> parse_url(const std::string& url);

// Drain complete SSE "data: ..." lines from buf, returning payloads (after
// the "data: " prefix).  Incomplete trailing data stays in buf.
std::vector<std::string> drain_sse_lines(std::string& buf);

// Truncate s to at most max_bytes bytes WITHOUT splitting a multi-byte UTF-8
// codepoint: if the cut would land inside a continuation-byte sequence, back up to
// the preceding codepoint boundary. ASCII input is returned byte-for-byte. Does not
// append an ellipsis — callers add "..." themselves.
inline std::string utf8_truncate(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    size_t n = max_bytes;
    while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
    return s.substr(0, n);
}

} // namespace mm::util
