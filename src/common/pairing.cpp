#include "common/pairing.hpp"

#include <openssl/hmac.h>
#include <openssl/evp.h>

#include <iomanip>
#include <random>
#include <sstream>

namespace mm::pairing {

std::string generate_pin() {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, 999999);
    char buf[8];
    snprintf(buf, sizeof(buf), "%06d", dist(rng));
    return std::string(buf);
}

std::string generate_nonce(size_t bytes) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<unsigned int> dist(0, 255u);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes; ++i)
        oss << std::setw(2) << dist(rng);
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
