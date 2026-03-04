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

} // namespace mm::pairing
