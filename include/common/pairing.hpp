#pragma once

#include <string>
#include <cstddef>

namespace mm::pairing {

// Generate a random 6-digit decimal PIN string (e.g. "042391").
std::string generate_pin();

// Generate `bytes` random bytes encoded as a lowercase hex string.
// Default 32 bytes → 64 hex chars.
std::string generate_nonce(size_t bytes = 32);

// Compute HMAC-SHA256(key, data) and return as a 64-char lowercase hex string.
std::string hmac_sha256_hex(const std::string& key, const std::string& data);

// Plain SHA-256 as a 64-char lowercase hex string.
//
// For API tokens, which are stored HASHED: the api_token table keeps
// token_sha256 and never the token, so a leaked database backup does not hand
// over working credentials. Unkeyed rather than HMAC because there is no secret
// to key it with that would not have to be stored beside the hashes it protects.
std::string sha256_hex(const std::string& data);

} // namespace mm::pairing
