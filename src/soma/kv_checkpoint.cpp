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

#include "soma/arch_backend.hpp"

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
};

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

} // namespace

struct KvCheckpointStore::Impl {
    fs::path dir;
    ArchIr arch;
    KvFormatId format_id = kKvFormatInvalid;
    bool open = false;

    fs::path path_for(const std::string& key) const { return dir / (key + ".somakv"); }

    /// Read and validate a header. Shared by load(), stat() and sweep(), so the
    /// gating cannot drift between "what a load checks" and "what a sweep keeps".
    Status read_header(const fs::path& p,
                       KvCheckpointHeader& out,
                       std::vector<std::byte>& raw,
                       std::size_t& payload_at) const {
        std::ifstream in(p, std::ios::binary | std::ios::ate);
        if (!in) return {StatusCode::NotFound, "no checkpoint at " + p.string()};
        const auto size = static_cast<std::streamsize>(in.tellg());
        in.seekg(0);
        raw.resize(static_cast<std::size_t>(std::max<std::streamsize>(0, size)));
        if (!raw.empty()) {
            in.read(reinterpret_cast<char*>(raw.data()), size);
            if (!in) return {StatusCode::IoError, "short read from " + p.string()};
        }
        if (raw.size() < sizeof(kMagic) || std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) {
            return {StatusCode::InvalidArgument, p.string() + ": bad magic"};
        }
        Cursor c{raw.data(), raw.size(), sizeof(kMagic), true};
        out.version = c.u32();
        out.arch_hash = c.str();
        out.format_id = c.u32();
        out.length_tokens = c.u32();
        out.d_model = c.u32();
        out.payload_bytes = c.u64();
        out.written_at_ms = c.u64();
        if (!c.ok) return {StatusCode::InvalidArgument, p.string() + ": truncated header"};
        payload_at = c.at;
        return {};
    }

    Status gate(const KvCheckpointHeader& h) const {
        if (h.version != kKvCheckpointVersion) {
            return {StatusCode::VersionMismatch,
                    "checkpoint version " + std::to_string(h.version) +
                        " != " + std::to_string(kKvCheckpointVersion)};
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
    const auto* backend = resolve_arch_backend(arch);
    if (backend == nullptr || backend->attention == nullptr) {
        return {StatusCode::Unsupported, "no attention backend for this architecture"};
    }
    im.format_id = backend->attention->persist_format_id;
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

Status KvCheckpointStore::save(const std::string& key, const KvCache& kv) {
    auto& im = *impl_;
    if (!im.open) return {StatusCode::InvalidArgument, "checkpoint store is not open"};

    const auto len = kv.length();
    const auto hkv = kv.hkv();
    const auto layers = kv.n_layers();
    const auto per_plane = static_cast<std::uint64_t>(len) * hkv * layers;

    std::vector<std::byte> buf;
    buf.reserve(static_cast<std::size_t>(per_plane) * 2 * sizeof(float) + 256);
    for (const char c : kMagic)
        buf.push_back(static_cast<std::byte>(c));
    put_u32(buf, kKvCheckpointVersion);
    put_str(buf, im.arch.arch_hash);
    put_u32(buf, im.format_id);
    put_u32(buf, len);
    put_u32(buf, im.arch.topology.d_model);
    put_u64(buf, per_plane * 2 * sizeof(float));
    put_u64(buf, now_ms());

    // Live positions only, layer by layer — see the file header for why the raw
    // buffer is not written.
    auto& mut = const_cast<KvCache&>(kv);
    const auto append = [&](const float* src, std::size_t n_floats) {
        const auto* b = reinterpret_cast<const std::byte*>(src);
        buf.insert(buf.end(), b, b + n_floats * sizeof(float));
    };
    for (std::uint32_t l = 0; l < layers; ++l) {
        append(mut.k_at(l, 0), static_cast<std::size_t>(len) * hkv);
    }
    for (std::uint32_t l = 0; l < layers; ++l) {
        append(mut.v_at(l, 0), static_cast<std::size_t>(len) * hkv);
    }

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

Status KvCheckpointStore::load(const std::string& key, KvCache& kv) {
    auto& im = *impl_;
    if (!im.open) return {StatusCode::InvalidArgument, "checkpoint store is not open"};

    KvCheckpointHeader h;
    std::vector<std::byte> raw;
    std::size_t at = 0;
    if (auto st = im.read_header(im.path_for(key), h, raw, at); !st.ok()) return st;
    if (auto st = im.gate(h); !st.ok()) return st;

    const auto hkv = kv.hkv();
    const auto layers = kv.n_layers();
    if (hkv == 0 || layers == 0) {
        return {StatusCode::InvalidArgument, "destination cache is not open"};
    }
    if (h.length_tokens > kv.capacity()) {
        return {StatusCode::InvalidArgument,
                "checkpoint holds " + std::to_string(h.length_tokens) +
                    " tokens, destination capacity is " + std::to_string(kv.capacity())};
    }
    const auto per_plane = static_cast<std::uint64_t>(h.length_tokens) * hkv * layers;
    if (raw.size() - at < per_plane * 2 * sizeof(float)) {
        return {StatusCode::InvalidArgument, "checkpoint payload is short"};
    }

    const auto* src = reinterpret_cast<const float*>(raw.data() + at);
    const auto run = static_cast<std::size_t>(h.length_tokens) * hkv;
    for (std::uint32_t l = 0; l < layers; ++l) {
        std::copy_n(src + static_cast<std::size_t>(l) * run, run, kv.k_at(l, 0));
    }
    src += static_cast<std::size_t>(layers) * run;
    for (std::uint32_t l = 0; l < layers; ++l) {
        std::copy_n(src + static_cast<std::size_t>(l) * run, run, kv.v_at(l, 0));
    }
    return kv.set_length(h.length_tokens);
}

Status KvCheckpointStore::save(const std::string& key, const SeqState& seq) {
    (void)key;
    (void)seq;
    return {StatusCode::Unsupported,
            "SeqState checkpointing lands with the engine at G5; the KvCache overload "
            "writes the same format"};
}

Status KvCheckpointStore::load(const std::string& key, SeqState& seq) {
    (void)key;
    (void)seq;
    return {StatusCode::Unsupported,
            "SeqState checkpointing lands with the engine at G5; the KvCache overload "
            "reads the same format"};
}

Status KvCheckpointStore::stat(const std::string& key, KvCheckpointHeader& out) const {
    std::vector<std::byte> raw;
    std::size_t at = 0;
    return impl_->read_header(impl_->path_for(key), out, raw, at);
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
        if (e.path().extension() != ".somakv") continue;

        KvCheckpointHeader h;
        std::vector<std::byte> raw;
        std::size_t at = 0;
        // Unreadable, stale-arch and expired all get removed, for one reason:
        // none of them can ever be loaded again, so keeping them only grows the
        // directory. A requantization invalidates every checkpoint at once, and
        // without this they would accumulate forever.
        const bool bad = !im.read_header(e.path(), h, raw, at).ok() || !im.gate(h).ok();
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
