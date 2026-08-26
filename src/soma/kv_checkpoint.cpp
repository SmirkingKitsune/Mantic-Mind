// Soma — KV persistence. ONE format, THREE callers.
//
// The format writes only LIVE positions, layer by layer, rather than the raw
// cache buffer. That is what lets a checkpoint taken at ctx_size 128 restore
// into an engine configured for 4096: baking max_ctx into the payload would make
// every checkpoint hostage to the config it was written under, and the cluster
// case (suspend here, restore there) is exactly where that bites.
//
// Every load is gated on version, arch_hash AND format_id, and refuses rather
// than reads. A checkpoint replayed into a different attention family has the
// wrong cache shape; the bytes are all readable and the output is quietly
// degraded, which is the single most confusing bug report this system could
// produce.

#include "soma/kv_checkpoint.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

namespace soma {

namespace {

constexpr char kMagic[8] = {'S', 'O', 'M', 'A', 'K', 'V', '0', '1'};

void put_u32(std::vector<std::byte>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
        b.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
}

void put_u64(std::vector<std::byte>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i)
        b.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
}

void put_str(std::vector<std::byte>& b, const std::string& s) {
    put_u32(b, static_cast<std::uint32_t>(s.size()));
    for (const char c : s)
        b.push_back(static_cast<std::byte>(c));
}

void put_raw(std::vector<std::byte>& b, const std::uint8_t* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        b.push_back(static_cast<std::byte>(p[i]));
}

// The reading half of this codec lives in kv_checkpoint_header.cpp. Writing
// stays here because only the store writes; the node only ever reads.

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

} // namespace

struct KvCheckpointStore::Impl {
    fs::path dir;
    ArchIr arch;
    const AttentionBackend* attention = nullptr;
    KvFormatId format_id = kKvFormatInvalid;
    bool open = false;

    fs::path path_for(const std::string& key) const {
        return dir / (key + kv_checkpoint_extension());
    }

    /// Read and validate a header. Shared by load(), stat() and sweep(), so the
    /// gating cannot drift between "what a load checks" and "what a sweep keeps".
    Status
    read_header(const fs::path& p, KvCheckpointHeader& out, std::vector<std::byte>& raw) const {
        std::ifstream in(p, std::ios::binary | std::ios::ate);
        if (!in) return {StatusCode::NotFound, "no checkpoint at " + p.string()};
        const auto size = static_cast<std::streamsize>(in.tellg());
        in.seekg(0);
        raw.resize(static_cast<std::size_t>(std::max<std::streamsize>(0, size)));
        if (!raw.empty()) {
            in.read(reinterpret_cast<char*>(raw.data()), size);
            if (!in) return {StatusCode::IoError, "short read from " + p.string()};
        }
        // The parse itself lives in kv_checkpoint_header.cpp, which the node also
        // links: the header is a wire format between two binaries, and two
        // parsers is how it acquires two meanings.
        if (auto st = parse_kv_checkpoint_header(raw.data(), raw.size(), out); !st.ok()) {
            return {st.code(), p.string() + ": " + st.message()};
        }
        return {};
    }

    Status gate(const KvCheckpointHeader& h) const {
        if (h.version != kKvCheckpointVersion &&
            h.version != kKvCheckpointVersionSpeculative &&
            h.version != kKvCheckpointVersionSampler) {
            return {StatusCode::VersionMismatch,
                    "unsupported checkpoint version " + std::to_string(h.version)};
        }
        if (h.arch_hash != arch.arch_hash) {
            return {StatusCode::ArchMismatch,
                    "checkpoint arch_hash " + h.arch_hash + " != " + arch.arch_hash};
        }
        if (h.format_id != format_id) {
            // A different attention family. The bytes would all read; the cache
            // shape would be wrong and the output quietly degraded.
            return {StatusCode::ArchMismatch,
                    "checkpoint KV format " + std::to_string(h.format_id) +
                        " != " + std::to_string(format_id)};
        }
        return {};
    }
};

KvCheckpointStore::KvCheckpointStore() : impl_(std::make_unique<Impl>()) {}

KvCheckpointStore::~KvCheckpointStore() = default;

Status KvCheckpointStore::open(const std::string& checkpoint_dir, const ArchIr& arch) {
    close();
    auto& im = *impl_;
    im.dir = checkpoint_dir;
    im.arch = arch;

    // The format tag is BACKEND-OWNED and reached through the registry, so this
    // core TU never names an architecture. tools/ci/check_seam.py enforces that.
    const auto* attention = resolve_attention_backend(arch.attention.family);
    if (attention == nullptr) {
        return {StatusCode::Unsupported, "no attention backend for this architecture"};
    }
    im.format_id = attention->persist_format_id;
    im.attention = attention;
    if (im.format_id == kKvFormatInvalid) {
        return {StatusCode::Unsupported, "attention backend declares no KV persist format"};
    }

    std::error_code ec;
    fs::create_directories(im.dir, ec);
    if (ec) return {StatusCode::IoError, "cannot create " + im.dir.string()};
    im.open = true;
    return {};
}

void KvCheckpointStore::close() {
    impl_ = std::make_unique<Impl>();
}

Status
KvCheckpointStore::save(const std::string& key, const KvCache& kv, const SeqPersistState& state) {
    auto& im = *impl_;
    if (!im.open) return {StatusCode::InvalidArgument, "checkpoint store is not open"};

    const auto len = kv.length();
    if (state.tokens.size() != len) {
        // Refused rather than padded or truncated. A token list that does not
        // line up with the cached positions makes the prefix check on the other
        // side meaningless, and a meaningless check is worse than none because
        // it reads as a guarantee.
        return {StatusCode::InvalidArgument,
                "token count " + std::to_string(state.tokens.size()) + " != cached positions " +
                    std::to_string(len)};
    }
    // Each plane at ITS width. They were the same number until MLA's V plane
    // stopped existing.
    const auto k_hkv = kv.k_hkv();
    const auto v_hkv = kv.v_hkv();
    const auto layers = kv.n_layers();
    const auto payload_floats =
        static_cast<std::uint64_t>(len) * (static_cast<std::uint64_t>(k_hkv) + v_hkv) * layers;
    std::vector<std::byte> opaque_payload;
    if (kv.is_opaque()) {
        if (im.attention == nullptr || im.attention->serialize_kv == nullptr) {
            return {StatusCode::Unsupported,
                    "opaque KV backend does not provide checkpoint serialization"};
        }
        if (auto st = im.attention->serialize_kv(
                im.arch,
                std::span<const std::byte>(kv.opaque_data(), kv.opaque_size()),
                kv.capacity(),
                len,
                opaque_payload);
            !st.ok())
            return st;
    }
    const auto cache_payload_bytes = kv.is_opaque()
                                         ? static_cast<std::uint64_t>(opaque_payload.size())
                                         : payload_floats * sizeof(float);

    std::vector<std::byte> buf;
    buf.reserve(static_cast<std::size_t>(cache_payload_bytes) + 256);
    for (const char c : kMagic)
        buf.push_back(static_cast<std::byte>(c));
    // One version, always the current one. What varies is the FLAGS word — see
    // kv_checkpoint.hpp for why the two optional fields stopped being two
    // version numbers.
    const std::uint32_t flags =
        (state.auxiliary.empty() ? 0u : kKvFlagAuxiliary) |
        (state.media.empty() ? 0u : kKvFlagMedia);
    put_u32(buf, kKvCheckpointVersion);
    put_str(buf, im.arch.arch_hash);
    put_u32(buf, im.format_id);
    put_u32(buf, len);
    put_u32(buf, im.arch.topology.d_model);
    put_u64(buf, cache_payload_bytes);
    put_u64(buf, now_ms());
    // v3: the sampler's stream position, so a resumed sequence continues the
    // draw it was on rather than starting a fresh one.
    put_u64(buf, state.rng_state);
    put_u32(buf, static_cast<std::uint32_t>(state.emitted.size()));
    put_u32(buf, flags);
    if ((flags & kKvFlagAuxiliary) != 0)
        put_u64(buf, static_cast<std::uint64_t>(state.auxiliary.size()));
    // v5: what the cached positions were built from beyond their token ids. A
    // text-only sequence sets no bit and writes no bytes, so the ONLY cost of
    // this field is borne by the checkpoints that need it.
    if ((flags & kKvFlagMedia) != 0)
        put_raw(buf, state.media.bytes.data(), state.media.bytes.size());
    // v2: the ids occupying the cached positions, so a restore can be CHECKED
    // against the prompt it is attached to rather than trusted.
    for (const auto t : state.tokens)
        put_u32(buf, static_cast<std::uint32_t>(t));
    // v3: what the sequence has said. The repetition and presence penalties read
    // this, so a resume without it lets the model immediately repeat something it
    // had already been penalised for.
    for (const auto t : state.emitted)
        put_u32(buf, static_cast<std::uint32_t>(t));

    // Live positions only, layer by layer — see the file header for why the raw
    // buffer is not written.
    auto& mut = const_cast<KvCache&>(kv);
    const auto append = [&](const float* src, std::size_t n_floats) {
        const auto* b = reinterpret_cast<const std::byte*>(src);
        buf.insert(buf.end(), b, b + n_floats * sizeof(float));
    };
    if (kv.is_opaque()) {
        buf.insert(buf.end(), opaque_payload.begin(), opaque_payload.end());
    } else {
        for (std::uint32_t l = 0; l < layers; ++l) {
            append(mut.k_at(l, 0), static_cast<std::size_t>(len) * k_hkv);
        }
        for (std::uint32_t l = 0; l < layers && v_hkv > 0; ++l) {
            append(mut.v_at(l, 0), static_cast<std::size_t>(len) * v_hkv);
        }
    }
    buf.insert(buf.end(), state.auxiliary.begin(), state.auxiliary.end());

    // Write to a temporary and rename. A checkpoint is written under memory
    // pressure and read after a crash or a migration; a half-written file that
    // passes the magic check is worse than no file.
    const auto final_path = im.path_for(key);
    const auto tmp = fs::path(final_path.string() + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return {StatusCode::IoError, "cannot write " + tmp.string()};
        out.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(buf.size()));
        if (!out) return {StatusCode::IoError, "short write to " + tmp.string()};
    }
    std::error_code ec;
    fs::rename(tmp, final_path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return {StatusCode::IoError, "cannot rename into " + final_path.string()};
    }
    return {};
}

Status KvCheckpointStore::load(const std::string& key, KvCache& kv, SeqPersistState& out) {
    auto& im = *impl_;
    if (!im.open) return {StatusCode::InvalidArgument, "checkpoint store is not open"};

    KvCheckpointHeader h;
    std::vector<std::byte> raw;
    if (auto st = im.read_header(im.path_for(key), h, raw); !st.ok()) return st;
    if (auto st = im.gate(h); !st.ok()) return st;

    const auto at = h.payload_at;
    if (raw.size() < at) return {StatusCode::InvalidArgument, "checkpoint token arrays are short"};

    const auto read_ids = [&](std::size_t offset, std::uint32_t count) {
        std::vector<TokenId> ids(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto* p = raw.data() + offset + static_cast<std::size_t>(i) * 4;
            std::uint32_t v = 0;
            for (int b = 0; b < 4; ++b) {
                v |= static_cast<std::uint32_t>(static_cast<unsigned char>(p[b])) << (8 * b);
            }
            ids[i] = static_cast<TokenId>(v);
        }
        return ids;
    };
    out.tokens = read_ids(h.tokens_at, h.length_tokens);
    out.emitted = read_ids(h.emitted_at, h.n_emitted);
    out.rng_state = h.rng_state;
    // Returned, never enforced here. The store cannot know what media the
    // caller is about to attach this cache to — admit() and extend() can, and
    // they are where the comparison belongs.
    out.media = h.media_digest;
    if (h.auxiliary_bytes > 0) {
        if (h.auxiliary_at > raw.size() || h.auxiliary_bytes > raw.size() - h.auxiliary_at) {
            return {StatusCode::InvalidArgument, "checkpoint auxiliary state is truncated"};
        }
        out.auxiliary.assign(raw.begin() + h.auxiliary_at,
                             raw.begin() + h.auxiliary_at + h.auxiliary_bytes);
    } else {
        out.auxiliary.clear();
    }

    const auto k_hkv = kv.k_hkv();
    const auto v_hkv = kv.v_hkv();
    const auto layers = kv.n_layers();
    if (kv.is_opaque()) {
        if (h.length_tokens > kv.capacity()) {
            return {StatusCode::InvalidArgument, "opaque checkpoint exceeds destination capacity"};
        }
        if (raw.size() - at < h.payload_bytes) {
            return {StatusCode::InvalidArgument, "opaque checkpoint payload is truncated"};
        }
        if (im.attention == nullptr || im.attention->restore_kv == nullptr) {
            return {StatusCode::Unsupported,
                    "opaque KV backend does not provide checkpoint restoration"};
        }
        if (auto st = im.attention->restore_kv(
                im.arch,
                std::span<const std::byte>(raw.data() + at,
                                           static_cast<std::size_t>(h.payload_bytes)),
                h.length_tokens,
                std::span<std::byte>(kv.opaque_data(), kv.opaque_size()),
                kv.capacity());
            !st.ok())
            return st;
        return kv.set_length(h.length_tokens);
    }
    if (k_hkv == 0 || layers == 0) {
        return {StatusCode::InvalidArgument, "destination cache is not open"};
    }
    if (h.length_tokens > kv.capacity()) {
        return {StatusCode::InvalidArgument,
                "checkpoint holds " + std::to_string(h.length_tokens) +
                    " tokens, destination capacity is " + std::to_string(kv.capacity())};
    }
    const auto payload_floats = static_cast<std::uint64_t>(h.length_tokens) *
                                (static_cast<std::uint64_t>(k_hkv) + v_hkv) * layers;
    const auto payload_bytes = payload_floats * sizeof(float);
    if (h.payload_bytes != payload_bytes) {
        return {StatusCode::InvalidArgument,
                "checkpoint declares " + std::to_string(h.payload_bytes) +
                    " payload bytes, destination geometry requires " +
                    std::to_string(payload_bytes)};
    }
    if (raw.size() - at < payload_bytes) {
        return {StatusCode::InvalidArgument, "checkpoint payload is short"};
    }

    const auto* src = reinterpret_cast<const float*>(raw.data() + at);
    const auto k_run = static_cast<std::size_t>(h.length_tokens) * k_hkv;
    for (std::uint32_t l = 0; l < layers; ++l) {
        std::copy_n(src + static_cast<std::size_t>(l) * k_run, k_run, kv.k_at(l, 0));
    }
    // The V plane may not exist. `format_id` is what stops a v1 checkpoint — which
    // always had one — reaching this code with a cache that does not.
    if (v_hkv > 0) {
        src += static_cast<std::size_t>(layers) * k_run;
        const auto v_run = static_cast<std::size_t>(h.length_tokens) * v_hkv;
        for (std::uint32_t l = 0; l < layers; ++l) {
            std::copy_n(src + static_cast<std::size_t>(l) * v_run, v_run, kv.v_at(l, 0));
        }
    }
    return kv.set_length(h.length_tokens);
}

Status KvCheckpointStore::stat(const std::string& key, KvCheckpointHeader& out) const {
    std::vector<std::byte> raw;
    return impl_->read_header(impl_->path_for(key), out, raw);
}

bool KvCheckpointStore::exists(const std::string& key) const noexcept {
    std::error_code ec;
    return fs::exists(impl_->path_for(key), ec);
}

Status KvCheckpointStore::remove(const std::string& key) {
    std::error_code ec;
    if (!fs::remove(impl_->path_for(key), ec)) {
        return {StatusCode::NotFound, "no checkpoint for " + key};
    }
    return {};
}

Status KvCheckpointStore::sweep(std::uint64_t max_age_ms, std::uint32_t& out_removed) {
    auto& im = *impl_;
    out_removed = 0;
    if (!im.open) return {StatusCode::InvalidArgument, "checkpoint store is not open"};

    const auto now = now_ms();
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(im.dir, ec)) {
        if (ec) break;
        if (e.path().extension() != kv_checkpoint_extension()) continue;

        KvCheckpointHeader h;
        std::vector<std::byte> raw;
        // Unreadable, stale-arch and expired all get removed, for one reason:
        // none of them can ever be loaded again, so keeping them only grows the
        // directory. A requantization invalidates every checkpoint at once, and
        // without this they would accumulate forever.
        const bool bad = !im.read_header(e.path(), h, raw).ok() || !im.gate(h).ok();
        const bool old =
            (max_age_ms > 0 && h.written_at_ms > 0 && now > h.written_at_ms + max_age_ms);
        if (bad || old) {
            std::error_code rm;
            if (fs::remove(e.path(), rm)) ++out_removed;
        }
    }
    return {};
}

std::uint64_t KvCheckpointStore::total_bytes() const noexcept {
    std::uint64_t total = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(impl_->dir, ec)) {
        if (ec) break;
        if (e.path().extension() == ".somakv") {
            std::error_code se;
            total += fs::file_size(e.path(), se);
        }
    }
    return total;
}

} // namespace soma
