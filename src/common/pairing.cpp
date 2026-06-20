#include "common/pairing.hpp"

#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mm::pairing {

// PINs and nonces are authentication material, so they must come from a
// CSPRNG. Rejection sampling keeps the PIN uniform over [0, 1000000).
std::string generate_pin() {
    // Largest multiple of 1'000'000 that fits in uint32_t.
    constexpr uint32_t kRejectAbove = 4294000000u;
    uint32_t v = 0;
    do {
        unsigned char buf[4];
        if (RAND_bytes(buf, sizeof(buf)) != 1)
            throw std::runtime_error("generate_pin: RAND_bytes failed");
        v = (static_cast<uint32_t>(buf[0]) << 24) |
            (static_cast<uint32_t>(buf[1]) << 16) |
            (static_cast<uint32_t>(buf[2]) << 8)  |
             static_cast<uint32_t>(buf[3]);
    } while (v >= kRejectAbove);

    char buf[8];
    snprintf(buf, sizeof(buf), "%06u", v % 1000000u);
    return std::string(buf);
}

std::string generate_nonce(size_t bytes) {
    std::vector<unsigned char> raw(bytes);
    if (!raw.empty() && RAND_bytes(raw.data(), static_cast<int>(raw.size())) != 1)
        throw std::runtime_error("generate_nonce: RAND_bytes failed");

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned char b : raw)
        oss << std::setw(2) << static_cast<unsigned int>(b);
    return oss.str();
}

std::string hmac_sha256_hex(const std::string& key, const std::string& data) {
    unsigned char out[EVP_MAX_MD_SIZE];
    unsigned int  len = 0;
    HMAC(EVP_sha256(),
         key.data(),  static_cast<int>(key.size()),
         reinterpret_cast<const unsigned char*>(data.data()),
         data.size(),
         out, &len);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < len; ++i)
        oss << std::setw(2) << static_cast<unsigned int>(out[i]);
    return oss.str();
}

} // namespace mm::pairing
