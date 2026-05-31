#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mm::cli {

// Tokenize one REPL command line with support for:
// - whitespace-separated tokens
// - single/double-quoted tokens
// - backslash escapes inside and outside quotes
// Returns true on success. On parse error, false and out_error is populated.
bool tokenize_command_line(const std::string& line,
                           std::vector<std::string>* out_tokens,
                           std::string* out_error = nullptr);

// Join tokens from [start_idx, end) with spaces.
std::string join_tokens(const std::vector<std::string>& tokens,
                        std::size_t start_idx);

// Parse a bool token used by CLI flags/options.
// Accepted true values:  true, 1, yes, y, on
// Accepted false values: false, 0, no, n, off
bool parse_bool_token(const std::string& token, bool* out_value);

} // namespace mm::cli

