// Soma — the KV checkpoint HEADER codec, and nothing else.
//
// Its own translation unit on purpose. The node has to read this header before
// spawning an engine (rejecting a cross-architecture resume after a 60-second
// model load is the confusing version of that error), so it links this object —
// and only this object. Nothing here references the model, the kernels, or the
// arch registry, so pulling it in costs the node one small .obj rather than the
// whole engine.
//
// The alternative — a second parser in src/node — is how a format acquires two
// definitions that agree until the day they don't.

#include "soma/kv_checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace soma {

namespace {

constexpr char kMagic[8] = {'S', 'O', 'M', 'A', 'K', 'V', '0', '1'};
constexpr char kExtension[] = ".somakv";

/// Enough for the fixed fields plus an arch_hash of any sane length. A header
/// that does not fit is truncated by definition, and reading more would mean
/// reading payload.
constexpr std::size_t kMaxHeaderBytes = 4096;

struct Cursor {
    const std::byte* p = nullptr;
    std::size_t n = 0, at = 0;
    bool ok = true;

    std::uint32_t u32() {
        if (at + 4 > n) {
            ok = false;
            return 0;
        }
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(static_cast<unsigned char>(p[at + i])) << (8 * i);
        }
        at += 4;
        return v;
    }

    std::uint64_t u64() {
        if (at + 8 > n) {
            ok = false;
            return 0;
        }
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<std::uint64_t>(static_cast<unsigned char>(p[at + i])) << (8 * i);
        }
        at += 8;
        return v;
    }

    std::string str() {
        const auto len = u32();
        if (!ok || at + len > n) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(p + at), len);
        at += len;
        return s;
    }

    void raw(std::uint8_t* out, std::size_t count) {
        if (at + count > n) {
            ok = false;
            return;
        }
        for (std::size_t i = 0; i < count; ++i)
            out[i] = static_cast<std::uint8_t>(p[at + i]);
        at += count;
    }
};

std::string hex32(std::uint32_t v) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s = "0x";
    for (int i = 7; i >= 0; --i)
        s.push_back(kHex[(v >> (4 * i)) & 0xF]);
    return s;
}

} // namespace

const char* kv_checkpoint_extension() noexcept {
    return kExtension;
}

Status
parse_kv_checkpoint_header(const std::byte* data, std::size_t size, KvCheckpointHeader& out) {
    if (data == nullptr || size < sizeof(kMagic) ||
        std::memcmp(data, kMagic, sizeof(kMagic)) != 0) {
        return {StatusCode::InvalidArgument, "not a Soma KV checkpoint (bad magic)"};
    }
    Cursor c{data, size, sizeof(kMagic), true};
    out.version = c.u32();
    out.arch_hash = c.str();
    out.format_id = c.u32();
    out.length_tokens = c.u32();
    out.d_model = c.u32();
    out.payload_bytes = c.u64();
    out.written_at_ms = c.u64();
    out.rng_state = c.u64();
    out.n_emitted = c.u32();
    // Before the version gate: a file too short to hold the fixed fields has a
    // version of 0 by truncation, and "unsupported checkpoint version 0" names
    // the wrong fault.
    if (!c.ok) return {StatusCode::InvalidArgument, "truncated KV checkpoint header"};

    // The version gate comes BEFORE the variable fields, not after. Reading a
    // flags word out of a v4 file's auxiliary extent, or an extent out of a v3
    // file's token array, is how a version check that ran last would let a
    // rejected file still corrupt every offset it computed on the way there.
    //
    // Checked here rather than only in the store's gate, because the node reads
    // this header without a store and must not interpret a v1 layout — where the
    // payload starts immediately and the token array does not exist — as a v2
    // one. Every offset below would be wrong by 4 x length_tokens.
    if (out.version != kKvCheckpointVersion && out.version != kKvCheckpointVersionSpeculative &&
        out.version != kKvCheckpointVersionSampler) {
        return {StatusCode::VersionMismatch,
                "unsupported checkpoint version " + std::to_string(out.version)};
    }

    // v3 and v4 predate the flags word. Synthesising theirs here is what lets
    // every reader below — and every caller — branch on flags alone rather than
    // re-deriving the layout from the version a second time.
    if (out.version == kKvCheckpointVersionSampler) {
        out.flags = 0;
    } else if (out.version == kKvCheckpointVersionSpeculative) {
        out.flags = kKvFlagAuxiliary;
        out.auxiliary_bytes = c.u64();
    } else {
        out.flags = c.u32();
        if ((out.flags & ~kKvFlagsKnown) != 0) {
            // Written by a newer build. Every field after this word is at an
            // offset derived from it, so this is not a field to skip — it is the
            // whole rest of the file.
            return {StatusCode::VersionMismatch,
                    "checkpoint carries unknown header flags " + hex32(out.flags & ~kKvFlagsKnown)};
        }
        if ((out.flags & kKvFlagAuxiliary) != 0) out.auxiliary_bytes = c.u64();
        if ((out.flags & kKvFlagMedia) != 0) {
            c.raw(out.media_digest.bytes.data(), out.media_digest.bytes.size());
        }
    }
    if (!c.ok) return {StatusCode::InvalidArgument, "truncated KV checkpoint header"};

    // Arithmetic, not consumed: stat() reads a bounded prefix and the token
    // arrays of a full context do not fit in it.
    out.tokens_at = c.at;
    out.emitted_at = out.tokens_at + static_cast<std::size_t>(out.length_tokens) * 4;
    out.payload_at = out.emitted_at + static_cast<std::size_t>(out.n_emitted) * 4;
    out.auxiliary_at = out.payload_at + static_cast<std::size_t>(out.payload_bytes);
    return {};
}

Status read_kv_checkpoint_header(const std::string& path, KvCheckpointHeader& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "no checkpoint at " + path};

    std::vector<std::byte> prefix(kMaxHeaderBytes);
    in.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    // gcount(), not the read's failbit: a header-only file is shorter than the
    // buffer, so a short read is normal here rather than an error.
    prefix.resize(static_cast<std::size_t>(in.gcount()));

    if (auto st = parse_kv_checkpoint_header(prefix.data(), prefix.size(), out); !st.ok()) {
        return {st.code(), path + ": " + st.message()};
    }
    return {};
}

} // namespace soma
