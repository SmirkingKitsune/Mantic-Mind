#include "common/pairing.hpp"

#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <cstdint>
#include <fstream>
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

std::string sha256_hex(const std::string& data) {
    unsigned char out[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) throw std::runtime_error("sha256_hex: EVP_MD_CTX_new failed");
    const bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(ctx, data.data(), data.size()) == 1 &&
                    EVP_DigestFinal_ex(ctx, out, &len) == 1;
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("sha256_hex: digest failed");

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < len; ++i)
        oss << std::setw(2) << static_cast<unsigned int>(out[i]);
    return oss.str();
}

std::string sha256_file_hex(const std::string& path, std::string* error) {
    if (error) error->clear();

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) *error = "cannot open " + path;
        return {};
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) {
        if (error) *error = "EVP_MD_CTX_new failed";
        return {};
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        if (error) *error = "digest init failed";
        return {};
    }

    std::vector<char> buf(64 * 1024);
    while (in.good()) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = in.gcount();
        if (got <= 0) break;
        if (EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(got)) != 1) {
            EVP_MD_CTX_free(ctx);
            if (error) *error = "digest update failed";
            return {};
        }
    }
    // Distinguish a read fault from a clean EOF: a truncated read that hashed
    // the prefix would produce a confident digest of the wrong bytes.
    if (in.bad()) {
        EVP_MD_CTX_free(ctx);
        if (error) *error = "read error on " + path;
        return {};
    }

    unsigned char out[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    const bool ok = EVP_DigestFinal_ex(ctx, out, &len) == 1;
    EVP_MD_CTX_free(ctx);
    if (!ok) {
        if (error) *error = "digest final failed";
        return {};
    }

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < len; ++i)
        oss << std::setw(2) << static_cast<unsigned int>(out[i]);
    return oss.str();
}

} // namespace mm::pairing
