#include "common/cli_repl.hpp"

#include "common/util.hpp"

#include <cctype>

namespace mm::cli {

bool tokenize_command_line(const std::string& line,
                           std::vector<std::string>* out_tokens,
                           std::string* out_error) {
    if (!out_tokens) return false;
    out_tokens->clear();
    if (out_error) out_error->clear();

    std::string cur;
    bool in_quote = false;
    char quote_ch = '\0';
    bool escaping = false;

    auto flush_cur = [&]() {
        if (!cur.empty()) {
            out_tokens->push_back(cur);
            cur.clear();
        }
    };

    for (char ch : line) {
        if (escaping) {
            cur.push_back(ch);
            escaping = false;
            continue;
        }

        if (ch == '\\') {
            escaping = true;
            continue;
        }

        if (in_quote) {
            if (ch == quote_ch) {
                in_quote = false;
                quote_ch = '\0';
            } else {
                cur.push_back(ch);
            }
            continue;
        }

        if (ch == '"' || ch == '\'') {
            in_quote = true;
            quote_ch = ch;
            continue;
        }

        if (std::isspace(static_cast<unsigned char>(ch))) {
            flush_cur();
            continue;
        }

        cur.push_back(ch);
    }

    if (escaping) {
        if (out_error) *out_error = "trailing escape (\\) at end of input";
        return false;
    }
    if (in_quote) {
        if (out_error) *out_error = "unterminated quoted string";
        return false;
    }

    flush_cur();
    return true;
}

std::string join_tokens(const std::vector<std::string>& tokens,
                        std::size_t start_idx) {
    if (start_idx >= tokens.size()) return {};
    std::string out;
    for (std::size_t i = start_idx; i < tokens.size(); ++i) {
        if (!out.empty()) out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

bool parse_bool_token(const std::string& token, bool* out_value) {
    if (!out_value) return false;
    const std::string t = mm::util::to_lower(mm::util::trim(token));
    if (t == "1" || t == "true" || t == "yes" || t == "y" || t == "on") {
        *out_value = true;
        return true;
    }
    if (t == "0" || t == "false" || t == "no" || t == "n" || t == "off") {
        *out_value = false;
        return true;
    }
    return false;
}

} // namespace mm::cli

