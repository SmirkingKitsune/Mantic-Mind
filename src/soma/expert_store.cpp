// Soma — the expert container reader.
//
// Format: schemas/container.md
//
// The whole design goal is a cheap miss: one contiguous read at a 4 KB-aligned
// offset, located by a sidecar lookup rather than a header parse.
//
// G2 scope: synchronous reads plus the bandwidth probe the verdict depends on.
// The async load pool and readahead land with MemoryHierarchy, which is what
// actually has a policy to overlap against — an async read with no cache behind
// it has nothing useful to do while it waits.

#include "soma/expert_store.hpp"

#include "soma/quant_format.hpp"

#include <openssl/evp.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace soma {

namespace {

/// Bytes one expert projection occupies under the IR's quantization map.
/// Mirrors the same helper in plan.cpp; both derive from the QuantMap so they
/// cannot disagree about a role.
std::uint64_t
expert_bytes_for(const ArchIr& arch, std::uint32_t rows, std::uint32_t cols, TensorRole role) {
    const auto& spec = arch.quantization.for_role(role);
    // Row-aware: the effective group is the largest divisor of `cols` not
    // exceeding the requested one, matching quantize_tensor(). A flat
    // element-count calculation disagrees for any tensor narrower than the group.
    return quantized_tensor_bytes(spec.dtype, rows, cols, spec.group ? spec.group : kDefaultGroup);
}

constexpr char kMagic[8] = {'S', 'O', 'M', 'A', 'C', 'T', 'N', 'R'};

struct Cursor {
    const std::byte* p = nullptr;
    const std::byte* end = nullptr;
    bool ok = true;

    template <typename T>
    T read() noexcept {
        T v{};
        if (!ok || static_cast<std::size_t>(end - p) < sizeof(T)) {
            ok = false;
            return v;
        }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    std::string bytes(std::size_t n) {
        if (!ok || n > static_cast<std::size_t>(end - p)) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }

    bool empty() const noexcept { return ok && p == end; }
    std::size_t remaining() const noexcept {
        return ok ? static_cast<std::size_t>(end - p) : 0;
    }
};

} // namespace

ExpertDigest expert_digest(LayerIndex layer, ExpertId expert, CByteSpan bytes) noexcept {
    // A tagged, length-bound preimage, spelled out here because Python reproduces
    // it byte for byte and "the SHA-256 of the expert" did not say enough.
    //
    //   "soma/expert\0" ‖ u32le(layer) ‖ u32le(expert) ‖ u64le(length) ‖ payload
    //
    // The tag keeps this from colliding with a bare SHA-256 of the same bytes
    // taken for some other purpose. The identity keeps two experts with identical
    // payloads — plausible for zeroed or pruned slots, and routine in the tiny
    // fixtures — from having identical digests, which is what makes the table a
    // placement check and not only a corruption check. The length is in the
    // preimage as well as implied by it so that a truncated range cannot hash as
    // a shorter expert that legitimately ends there.
    static constexpr char kTag[] = "soma/expert";
    std::array<unsigned char, sizeof(kTag) + 16> prefix{};
    std::memcpy(prefix.data(), kTag, sizeof(kTag)); // includes the NUL
    const auto put32 = [&](std::size_t at, std::uint32_t v) {
        for (std::size_t i = 0; i < 4; ++i)
            prefix[at + i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
    };
    const auto put64 = [&](std::size_t at, std::uint64_t v) {
        for (std::size_t i = 0; i < 8; ++i)
            prefix[at + i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
    };
    put32(sizeof(kTag), layer);
    put32(sizeof(kTag) + 4, expert);
    put64(sizeof(kTag) + 8, static_cast<std::uint64_t>(bytes.size()));

    // EVP rather than the one-shot SHA256(): the preimage is a small prefix
    // followed by a payload that can be megabytes, and concatenating them to use
    // the one-shot form would memcpy every expert an extra time on the read path.
    // The SHA256_Init/Update/Final trio would stream too, but it is deprecated in
    // OpenSSL 3.0 and this build treats that as an error.
    ExpertDigest out{};
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) return out;
    unsigned int written = 0;
    const bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(ctx, prefix.data(), prefix.size()) == 1 &&
                    (bytes.empty() || EVP_DigestUpdate(ctx, bytes.data(), bytes.size()) == 1) &&
                    EVP_DigestFinal_ex(ctx, out.bytes.data(), &written) == 1;
    EVP_MD_CTX_free(ctx);
    // An all-zero digest is the dense-slot sentinel, so a silent failure here
    // would make a live expert look like an empty slot. Nothing can throw from a
    // noexcept function, but the caller compares digests and an all-zero one can
    // only ever mismatch — which is the safe direction.
    if (!ok || written != kExpertDigestBytes) out = ExpertDigest{};
    return out;
}

namespace {

bool dtype_from_id(std::uint32_t id, DType& out) noexcept {
    if (id > static_cast<std::uint32_t>(DType::Q4_0)) return false;
    out = static_cast<DType>(id);
    return true;
}

/// Everything the index file holds, parsed once.
///
/// Shared by open_indexed() and verify_identity() so two readers asking different
/// questions of the same file cannot disagree about its layout — which is the
/// failure this whole change is about.
struct ParsedIndex {
    ContainerHeader header;
    /// The legacy single-dtype pair. Preserved verbatim on rewrite: it cannot
    /// describe the default map (gate/up and down differ), which is why the
    /// per-role descriptor exists, but silently dropping a field a v1 reader may
    /// still consult is not this function's decision to make.
    std::uint32_t dtype_id = 0;
    std::uint32_t group = 0;
    std::uint64_t total_bytes = 0;
    std::vector<ExpertLocation> entries;
    /// Parallel to `entries`, or empty when the container carries no table.
    std::vector<ExpertDigest> digests;
};

Status parse_index(const std::string& path, std::string_view raw, ParsedIndex& out) {
    if (raw.size() < sizeof(kMagic) || std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) {
        return {StatusCode::InvalidArgument, path + ": bad magic"};
    }

    const auto* base = reinterpret_cast<const std::byte*>(raw.data());
    Cursor c{base + sizeof(kMagic), base + raw.size(), true};

    auto& h = out.header;
    h.version = c.read<std::uint32_t>();
    if (h.version != kLegacyContainerVersion && h.version != kContainerVersion) {
        return {StatusCode::VersionMismatch,
                "container version " + std::to_string(h.version) +
                    " is not supported (expected " + std::to_string(kLegacyContainerVersion) +
                    " or " + std::to_string(kContainerVersion) + ")"};
    }

    // MUST-UNDERSTAND. An unknown bit means a field this build cannot see sits
    // somewhere in the header, so every offset after it is guesswork — and the
    // guess would not fail, it would produce a plausible index pointing at the
    // wrong bytes. Refusing is the only safe reading.
    h.flags = c.read<std::uint32_t>();
    if ((h.flags & ~kKnownContainerFlags) != 0) {
        return {StatusCode::VersionMismatch,
                path + ": container sets flags 0x" +
                    [&] {
                        char b[16];
                        std::snprintf(b, sizeof(b), "%08x", h.flags & ~kKnownContainerFlags);
                        return std::string(b);
                    }() +
                    " that this build does not understand; it was written by a newer converter"};
    }

    const auto hash_len = c.read<std::uint32_t>();
    h.arch_hash = c.bytes(hash_len);

    h.n_layers = c.read<std::uint32_t>();
    h.n_experts = c.read<std::uint32_t>();
    h.n_shards = c.read<std::uint32_t>();
    out.dtype_id = c.read<std::uint32_t>();
    out.group = c.read<std::uint32_t>();

    if ((h.flags & kFlagPerRoleQuant) != 0) {
        const auto n_roles = c.read<std::uint32_t>();
        if (n_roles != 3) {
            return {StatusCode::InvalidArgument,
                    path + ": role descriptor must contain exactly gate, up, and down; got " +
                        std::to_string(n_roles) + " entries"};
        }
        bool saw_gate = false, saw_up = false, saw_down = false;
        for (std::uint32_t i = 0; i < n_roles && c.ok; ++i) {
            const auto role_id = c.read<std::uint32_t>();
            const auto dt_id = c.read<std::uint32_t>();
            const auto grp = c.read<std::uint32_t>();
            DType dt{};
            if (!dtype_from_id(dt_id, dt)) {
                return {StatusCode::InvalidArgument,
                        path + ": role descriptor holds unknown dtype id " + std::to_string(dt_id)};
            }
            if (grp == 0) {
                return {StatusCode::InvalidArgument,
                        path + ": role descriptor holds a zero effective group"};
            }
            const RoleQuant rq{dt, grp};
            switch (role_id) {
            case static_cast<std::uint32_t>(TensorRole::ExpertGate):
                if (saw_gate)
                    return {StatusCode::InvalidArgument, path + ": duplicate gate descriptor"};
                h.gate = rq;
                saw_gate = true;
                break;
            case static_cast<std::uint32_t>(TensorRole::ExpertUp):
                if (saw_up)
                    return {StatusCode::InvalidArgument, path + ": duplicate up descriptor"};
                h.up = rq;
                saw_up = true;
                break;
            case static_cast<std::uint32_t>(TensorRole::ExpertDown):
                if (saw_down)
                    return {StatusCode::InvalidArgument, path + ": duplicate down descriptor"};
                h.down = rq;
                saw_down = true;
                break;
            default:
                return {StatusCode::InvalidArgument,
                        path + ": unknown expert role id " + std::to_string(role_id)};
            }
        }
        if (!c.ok) return {StatusCode::InvalidArgument, path + ": truncated role descriptor"};
        if (!saw_gate || !saw_up || !saw_down) {
            return {StatusCode::InvalidArgument,
                    path + ": role descriptor is missing gate, up, or down"};
        }
        h.has_role_quant = true;

        // V2 retains the legacy gate pair so the common prefix stays parseable.
        // It is redundant now, so contradictory values are corruption rather
        // than an alternate source of truth.
        DType legacy_dtype{};
        if (!dtype_from_id(out.dtype_id, legacy_dtype)) {
            return {StatusCode::InvalidArgument,
                    path + ": legacy expert dtype id " + std::to_string(out.dtype_id) +
                        " is not known"};
        }
        if (h.version == kContainerVersion &&
            (legacy_dtype != h.gate.dtype || out.group != h.gate.group)) {
            return {StatusCode::InvalidArgument,
                    path + ": legacy expert dtype/group contradicts the gate descriptor"};
        }
    }

    h.expert_bytes = c.read<std::uint64_t>();
    out.total_bytes = c.read<std::uint64_t>();
    if (!c.ok) return {StatusCode::InvalidArgument, path + ": truncated header"};

    if (h.n_experts != 0 &&
        static_cast<std::size_t>(h.n_layers) >
            std::numeric_limits<std::size_t>::max() / h.n_experts) {
        return {StatusCode::InvalidArgument, path + ": index dimensions overflow"};
    }
    const std::size_t entries = static_cast<std::size_t>(h.n_layers) * h.n_experts;
    if (h.n_shards > entries || h.n_shards > 100000) {
        return {StatusCode::InvalidArgument, path + ": shard count exceeds index bounds"};
    }
    constexpr std::size_t kEntryBytes = sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                                         sizeof(std::uint32_t);
    if (entries > c.remaining() / kEntryBytes) {
        return {StatusCode::InvalidArgument,
                path + ": index dimensions exceed the bytes remaining in the file"};
    }
    out.entries.resize(entries);
    for (auto& e : out.entries) {
        e.shard = c.read<std::uint32_t>();
        e.offset = c.read<std::uint64_t>();
        e.length = c.read<std::uint32_t>();
    }
    if (!c.ok) return {StatusCode::InvalidArgument, path + ": truncated index"};

    // The retired 8-byte table. Refused rather than read: its digests bind no
    // identity, so they cannot answer the question the reader now asks of them,
    // and silently accepting a weaker table would leave a container looking
    // checked when a swap would slip through it.
    if ((h.flags & kFlagExpertDigestsLegacy) != 0) {
        return {StatusCode::VersionMismatch,
                path + " carries the retired 8-byte digest table, which binds no expert "
                       "identity. Reconvert it with the current converter; nothing can "
                       "upgrade this table in place"};
    }

    // The digest table, after the entries rather than widened into them: the entry
    // stride stays 16 bytes, so every offset computation in the reader and in the
    // Python tools keeps working unchanged.
    if ((h.flags & kFlagExpertDigestsV2) != 0) {
        if (entries > c.remaining() / kExpertDigestBytes) {
            return {StatusCode::InvalidArgument,
                    path + ": digest table does not fit in the bytes remaining"};
        }
        out.digests.resize(entries);
        for (auto& digest : out.digests) {
            const auto stored = c.bytes(kExpertDigestBytes);
            if (!c.ok) break;
            std::memcpy(digest.bytes.data(), stored.data(), kExpertDigestBytes);
        }
        if (!c.ok) return {StatusCode::InvalidArgument, path + ": truncated digest table"};
        h.has_digests = true;
    }

    if (!c.empty()) return {StatusCode::InvalidArgument, path + ": trailing bytes after index"};
    return {};
}

// Validate the exact parsed snapshot the caller is about to use.
/// The half of the layout invariant that needs no architecture.
///
/// Split out because the two callers hold different amounts of context and were
/// enforcing different rules as a result. `validate_ranges()` has an IR and can
/// also check topology and the uniform expert length; `verify_payload()`
/// deliberately has none, so that a node holding a copied container can check it
/// without being able to resolve — or even support — its architecture.
///
/// That asymmetry had quietly become a disagreement: `soma verify` and
/// `verify_payload.py` both accepted a PERMUTED container that `open()` refuses,
/// so the post-transfer check blessed a container serve would reject. Liveness
/// comes from `length != 0` here rather than from the layer kind, which is the
/// only thing the IR was needed for in this walk.
Status validate_layout(const ParsedIndex& ix, const std::string& dir, const std::string& prefix) {
    const auto& h = ix.header;
    std::vector<std::uint64_t> ends(h.n_shards, 0);
    std::uint64_t total = 0;
    for (const auto& e : ix.entries) {
        if (e.length == 0) continue;
        // Index order IS the layout order today, and that is what makes a swapped
        // pair a structural failure rather than one only a digest can see. When
        // that rule is relaxed to a tiling invariant, it must be relaxed HERE, in
        // one place, with digests mandatory first.
        if (e.shard >= ends.size() || e.offset != ends[e.shard]) {
            return {StatusCode::InvalidArgument, "expert ranges are aliased or noncanonical"};
        }
        const auto end = e.offset + e.length;
        if (end < e.offset || end > std::numeric_limits<std::uint64_t>::max() - kDirectIoAlign) {
            return {StatusCode::InvalidArgument, "expert range overflows"};
        }
        ends[e.shard] = (end + kDirectIoAlign - 1) / kDirectIoAlign * kDirectIoAlign;
        if (total > std::numeric_limits<std::uint64_t>::max() - e.length)
            return {StatusCode::InvalidArgument, "expert total overflows"};
        total += e.length;
    }
    if (total != ix.total_bytes)
        return {StatusCode::InvalidArgument, "indexed expert total disagrees with header"};
    for (std::size_t s = 0; s < ends.size(); ++s) {
        char number[24];
        std::snprintf(number, sizeof(number), "%05u.bin", static_cast<unsigned>(s));
        std::error_code ec;
        const auto size = fs::file_size(fs::path(dir) / (prefix + number), ec);
        if (ec || ends[s] == 0 || size != ends[s])
            return {StatusCode::InvalidArgument, "shard size disagrees with canonical expert ranges"};
    }
    return {};
}

Status validate_ranges(const ParsedIndex& ix, const ArchIr& arch,
                       const std::string& dir, const std::string& prefix) {
    const auto& h = ix.header;
    const auto d = arch.routed_expert_width();
    const auto fi = arch.ffn.expert_intermediate;
    if (h.n_layers != arch.topology.n_layers || h.n_experts != arch.router.n_experts ||
        arch.topology.layer_kinds.size() != h.n_layers || d == 0 || fi == 0) {
        return {StatusCode::ArchMismatch, "container requires complete matching expert topology"};
    }
    const auto expected = expert_bytes_for(arch, fi, d, TensorRole::ExpertGate) +
                          expert_bytes_for(arch, fi, d, TensorRole::ExpertUp) +
                          expert_bytes_for(arch, d, fi, TensorRole::ExpertDown);
    if (expected == 0 || expected > std::numeric_limits<std::uint32_t>::max() ||
        h.expert_bytes != expected) {
        return {StatusCode::ArchMismatch, "container requires the exact uniform expert length"};
    }
    // What only an IR can say: which slots are supposed to hold an expert at all.
    for (std::size_t i = 0; i < ix.entries.size(); ++i) {
        const auto& e = ix.entries[i];
        const bool moe = arch.is_moe_layer(static_cast<LayerIndex>(i / h.n_experts));
        if ((moe && e.length != expected) || (!moe && e.length != 0)) {
            return {StatusCode::InvalidArgument, "expert slot presence/length disagrees with layer kind"};
        }
    }
    return validate_layout(ix, dir, prefix);
}


/// The role's quantization as the IR describes it, reduced to what a row of that
/// shape would actually have been written at.
RoleQuant role_from_ir(const ArchIr& arch, TensorRole role, std::uint32_t cols) {
    const auto& spec = arch.quantization.for_role(role);
    const auto requested = spec.group ? spec.group : kDefaultGroup;
    return {spec.dtype, effective_group(cols, requested)};
}


} // namespace

/// A shard opened for POSITIONAL reads.
///
/// Not an ifstream, and that is the entire point. A file stream carries ONE
/// stateful position, so two threads reading the same shard interleave as:
///
///     A: seekg(offset_A)
///     B: seekg(offset_B)      <- clobbers A's position
///     A: read(...)            <- returns expert B's bytes
///
/// The read SUCCEEDS. There is no error to check; the caller simply receives the
/// wrong expert's weights, which flows into that layer's output, changes the next
/// layer's routing, and surfaces as a model that gives slightly different answers
/// on every run. That is precisely the non-determinism the prefetch loader
/// exposed — the bug was latent for as long as reads happened on one thread.
///
/// pread/ReadFile-with-offset carry the position in the CALL rather than in the
/// handle, so concurrent reads cannot interfere. No lock, no per-thread handles.
class ShardFile {
public:
    ShardFile() = default;

    ~ShardFile() { close(); }

    ShardFile(ShardFile&& o) noexcept : h_(o.h_) { o.h_ = kInvalid; }

    ShardFile& operator=(ShardFile&& o) noexcept {
        if (this != &o) {
            close();
            h_ = o.h_;
            o.h_ = kInvalid;
        }
        return *this;
    }

    ShardFile(const ShardFile&) = delete;
    ShardFile& operator=(const ShardFile&) = delete;

    enum class Io : std::uint8_t { Buffered, Unbuffered };

    /// Unbuffered opens FAIL, routinely, and that is expected rather than
    /// exceptional: tmpfs refuses O_DIRECT outright, and so do several network
    /// and overlay filesystems — including the 9p mount a WSL build reads its own
    /// fixtures through. Callers treat a false return as "not available here" and
    /// fall back, which is why this reports rather than throws.
    bool open(const std::filesystem::path& p, Io io = Io::Buffered) noexcept {
        close();
#if defined(_WIN32)
        h_ = ::CreateFileW(p.wstring().c_str(),
                           GENERIC_READ,
                           FILE_SHARE_READ,
                           nullptr,
                           OPEN_EXISTING,
                           io == Io::Unbuffered
                               ? (FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
                               : FILE_ATTRIBUTE_NORMAL,
                           nullptr);
#elif defined(O_DIRECT)
        h_ = ::open(p.c_str(), io == Io::Unbuffered ? (O_RDONLY | O_DIRECT) : O_RDONLY);
#else
        // macOS has no O_DIRECT; F_NOCACHE is the equivalent, applied after the
        // open rather than as a flag.
        h_ = ::open(p.c_str(), O_RDONLY);
        if (h_ != kInvalid && io == Io::Unbuffered) {
#if defined(F_NOCACHE)
            if (::fcntl(h_, F_NOCACHE, 1) != 0) {
                close();
                return false;
            }
#else
            close();
            return false;
#endif
        }
#endif
        return valid();
    }

    /// Ask the OS to forget a range it has cached. Returns false when the
    /// platform offers no way to say it.
    ///
    /// Advice is best effort, even after flushing recent conversion writes.
    bool flush_for_probe() const noexcept {
#if defined(POSIX_FADV_DONTNEED)
        return valid() && ::fsync(h_) == 0;
#else
        return false;
#endif
    }

    bool drop_from_cache(std::uint64_t offset, std::uint64_t len) const noexcept {
#if defined(POSIX_FADV_DONTNEED)
        return valid() && ::posix_fadvise(h_, static_cast<off_t>(offset),
                                          static_cast<off_t>(len), POSIX_FADV_DONTNEED) == 0;
#else
        (void)offset;
        (void)len;
        return false;
#endif
    }

    bool valid() const noexcept { return h_ != kInvalid; }

    /// Thread-safe by construction: the offset lives in the call.
    bool read_at(std::uint64_t offset, void* dst, std::uint32_t len) const noexcept {
        auto* p = static_cast<unsigned char*>(dst);
        std::uint32_t done = 0;
        while (done < len) {
            const auto want = len - done;
#if defined(_WIN32)
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>((offset + done) & 0xFFFFFFFFull);
            ov.OffsetHigh = static_cast<DWORD>((offset + done) >> 32);
            DWORD got = 0;
            if (!::ReadFile(h_, p + done, static_cast<DWORD>(want), &got, &ov)) return false;
#else
            const auto got = ::pread(h_, p + done, want, static_cast<off_t>(offset + done));
            if (got < 0) return false;
#endif
            // A short read is legal for both APIs and must be looped, not
            // assumed away: treating one as complete would silently leave the
            // tail of an expert as whatever the buffer held before.
            if (got == 0) return false;
            done += static_cast<std::uint32_t>(got);
        }
        return true;
    }

private:
    void close() noexcept {
        if (!valid()) return;
#if defined(_WIN32)
        ::CloseHandle(h_);
#else
        ::close(h_);
#endif
        h_ = kInvalid;
    }

#if defined(_WIN32)
    using Handle = HANDLE;
    static inline const Handle kInvalid = INVALID_HANDLE_VALUE;
#else
    using Handle = int;
    static constexpr Handle kInvalid = -1;
#endif
    Handle h_ = kInvalid;
};

namespace {

/// A page-aligned buffer, which an unbuffered read requires and `new`/`vector` do
/// not give: both align to alignof(max_align_t), typically 16.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    ~AlignedBuffer() { reset(); }

    bool allocate(std::size_t bytes) noexcept {
        reset();
        if (bytes == 0) return true;
        const auto rounded = (bytes + kDirectIoAlign - 1) / kDirectIoAlign * kDirectIoAlign;
#if defined(_WIN32)
        p_ = ::_aligned_malloc(rounded, kDirectIoAlign);
#else
        p_ = std::aligned_alloc(kDirectIoAlign, rounded);
#endif
        size_ = (p_ != nullptr) ? rounded : 0;
        return p_ != nullptr;
    }

    void* data() const noexcept { return p_; }
    std::size_t size() const noexcept { return size_; }

private:
    void reset() noexcept {
        if (p_ == nullptr) return;
#if defined(_WIN32)
        ::_aligned_free(p_);
#else
        std::free(p_);
#endif
        p_ = nullptr;
        size_ = 0;
    }

    void* p_ = nullptr;
    std::size_t size_ = 0;
};

/// Force the probe off its unbuffered path.
///
/// Same shape and the same reason as SOMA_SIMD_TIER: the only way to know what a
/// mechanism is worth is to be able to turn it off under a fixed workload. Here
/// that answers a question an operator will actually ask — "is this host's
/// bandwidth figure the drive or the page cache?" — by letting them measure both
/// and compare. It is also the only way to exercise the fallback on a machine
/// whose filesystems all accept O_DIRECT.
bool probe_direct_disabled() noexcept {
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    const bool have = (_dupenv_s(&buf, &len, "SOMA_PROBE_NO_DIRECT") == 0 && buf != nullptr);
    const std::string v = have ? std::string(buf) : std::string();
    std::free(buf);
#else
    const char* raw = std::getenv("SOMA_PROBE_NO_DIRECT");
    const std::string v = raw ? std::string(raw) : std::string();
#endif
    return !v.empty() && v != "0" && v != "false";
}

/// The shard-prefix an index file's payload lives under.
///
/// One place, because every reader has to agree and a container
/// whose auxiliary index was checked against the base shards would report a
/// mismatch that says nothing about either.
const char* shard_prefix_for(const std::string& index_file) {
    return index_file == "soma.dspark" ? "dspark-experts-" : "experts-";
}

/// Read every live expert once, in index order, and hand its bytes to `sink`.
///
/// Opens the shards directly rather than going through ExpertStore, because both
/// callers run when the container may not be fit to open through the front door:
/// they are the checks that decide whether it is, and verify in particular has to
/// work on a container whose digests are the thing in doubt.
///
/// Index order, not random: this is a sequential sweep of files that are about to
/// be read end to end, and it is the one place in this file where readahead is
/// what you want.
template <typename Sink>
Status walk_experts(const ParsedIndex& ix,
                    const std::string& dir,
                    const std::string& index_file,
                    Sink&& sink) {
    const auto& h = ix.header;
    std::vector<ShardFile> shards(h.n_shards);
    for (std::uint32_t i = 0; i < h.n_shards; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "%s%05u.bin", shard_prefix_for(index_file), i);
        if (!shards[i].open(fs::path(dir) / name)) {
            return {StatusCode::NotFound, std::string("missing shard ") + name};
        }
    }

    std::vector<std::byte> buf;
    for (std::size_t slot = 0; slot < ix.entries.size(); ++slot) {
        const auto& e = ix.entries[slot];
        if (e.length == 0) continue; // a dense layer's empty slot
        if (e.shard >= shards.size()) {
            return {StatusCode::InvalidArgument,
                    "expert " + std::to_string(slot) + " names shard " + std::to_string(e.shard)};
        }
        if (buf.size() < e.length) buf.resize(e.length);
        if (!shards[e.shard].read_at(e.offset, buf.data(), e.length)) {
            return {StatusCode::IoError,
                    "cannot read expert " + std::to_string(slot) + " from shard " +
                        std::to_string(e.shard)};
        }
        sink(slot, CByteSpan(buf.data(), e.length));
    }
    return {};
}

} // namespace

struct ExpertStore::Impl {
    ContainerHeader header{};
    std::vector<ExpertLocation> index;
    std::vector<ShardFile> shards;
    /// Atomic: incremented from every thread that reads, with no lock held.
    std::atomic<std::uint64_t> bytes_read{0};
    std::string dir;
    /// Kept so the bandwidth probe can open its own handles onto the same shards.
    std::string shard_prefix;

    PayloadPolicy payload = PayloadPolicy::Trust;
    std::vector<ExpertDigest> digests;
    /// One byte per slot, not a bitset: a bitset would need read-modify-write on
    /// a shared word, and the whole point of this array is that it is touched
    /// from every reading thread with no lock held. A byte per expert is 15 KB
    /// for the largest model in the roadmap.
    ///
    /// The race is benign by construction — two threads may both verify the same
    /// expert and both store 1 — so this needs atomicity, not ordering.
    std::unique_ptr<std::atomic<std::uint8_t>[]> verified;

    std::size_t slot(LayerIndex layer, ExpertId expert) const noexcept {
        return static_cast<std::size_t>(layer) * header.n_experts + expert;
    }
};

struct ExpertStore::Pending::Impl {
    StatusCode code = StatusCode::Ok;
};

ExpertStore::Pending::Pending(Pending&&) noexcept = default;
ExpertStore::Pending& ExpertStore::Pending::operator=(Pending&&) noexcept = default;
ExpertStore::Pending::~Pending() = default;

ExpertStore::Pending::operator bool() const noexcept {
    return impl_ != nullptr;
}

StatusCode ExpertStore::Pending::wait() noexcept {
    return impl_ ? impl_->code : StatusCode::Internal;
}

ExpertStore::ExpertStore() : impl_(std::make_unique<Impl>()) {}

ExpertStore::~ExpertStore() = default;

void ExpertStore::close() {
    impl_ = std::make_unique<Impl>();
}

const ContainerHeader& ExpertStore::header() const noexcept {
    return impl_->header;
}

std::uint64_t ExpertStore::bytes_read() const noexcept {
    return impl_->bytes_read;
}

Status ExpertStore::open(const std::string& model_dir, const ArchIr& arch, OpenOptions opts) {
    return open_indexed(model_dir, arch, "soma.container", "experts-", opts);
}

Status ExpertStore::open_indexed(const std::string& model_dir,
                                 const ArchIr& arch,
                                 const std::string& index_file,
                                 const std::string& shard_prefix,
                                 OpenOptions opts) try {
    close();
    impl_->dir = model_dir;
    impl_->shard_prefix = shard_prefix;

    const fs::path index_path = fs::path(model_dir) / index_file;
    std::ifstream in(index_path, std::ios::binary);
    if (!in) {
        return {StatusCode::NotFound, "no " + index_file + " in " + model_dir};
    }
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    ParsedIndex parsed;
    if (auto st = parse_index(index_path.string(), raw, parsed); !st.ok()) return st;
    impl_->header = parsed.header;
    auto& h = impl_->header;

    // arch_hash gate.
    //
    // An empty hash means the container was never finished. The converter asks
    // the engine for the identity through `soma arch-hash` while it writes, so
    // the only ways to produce one now are `--no-identity` or a converter older
    // than that change. Python still does not compute the hash itself: it is
    // defined by the C++ IR canonicalization, and a second implementation would
    // agree until it did not.
    //
    // UNSTAMPED IS REFUSED, where it used to be accepted silently. The old
    // reading was that an unstamped container is merely un-gated, and everything
    // else the gate covers — the IR moving under a container that did not move,
    // which is what requantization and a changed family default both look like —
    // then goes uncaught, because the checks below compare the container against a
    // map read out of that same container's own meta.
    //
    // DSpark's auxiliary container uses the typed SkipArchHash policy below;
    // clearing a string is never treated as authority to bypass identity.
    //
    // Compared against the CONTAINER's identity, not the model's: `--quant-dense`
    // moves arch_hash without moving a shard byte, and gating on that would make
    // a stamped container refuse the very serve invocation the F32-on-disk dense
    // half exists to permit. Falls back to arch_hash for an IR that came from
    // somewhere other than a container directory, where the two are the same.
    const std::string& want_hash =
        arch.container_arch_hash.empty() ? arch.arch_hash : arch.container_arch_hash;

    if (opts.identity != IdentityPolicy::SkipArchHash && want_hash.empty()) {
        return {StatusCode::InvalidArgument,
                "the resolved model IR carries no arch_hash; refusing an unbound container open"};
    }

    // V1 pre-dates the mandatory role descriptor. Even a populated hash cannot
    // prove which same-sized role owns which encoding, so strict serving accepts
    // only V2. The developer escape remains explicit and auditable.
    if (h.version != kContainerVersion && opts.identity != IdentityPolicy::AllowUnstamped) {
        return {StatusCode::VersionMismatch,
                index_path.string() +
                    " is a legacy v1 container whose role layout is not mandatory. Reconvert it "
                    "with the current converter, or use --allow-unstamped for a development-only "
                    "open"};
    }
    if (h.version == kContainerVersion && !h.has_role_quant) {
        return {StatusCode::InvalidArgument,
                index_path.string() +
                    ": v2 container is missing the mandatory gate/up/down quant descriptor"};
    }

    if (h.arch_hash.empty() && opts.identity == IdentityPolicy::RequireStamped) {
        return {StatusCode::ArchMismatch,
                index_path.string() +
                    " carries no arch_hash: it was written by a converter that did not "
                    "stamp it, or with --no-identity. Reconvert it with the current "
                    "converter, or pass --allow-unstamped to use the development-only "
                    "structural and role checks without identity"};
    }

    // MISMATCHED is refused. Reading q4 bytes as q6 produces finite, wrong
    // numbers rather than an error, so this is the only place it can be caught.
    if (opts.identity != IdentityPolicy::SkipArchHash && !h.arch_hash.empty() &&
        h.arch_hash != want_hash) {
        return {StatusCode::ArchMismatch,
                "container arch_hash " + h.arch_hash.substr(0, 16) + "... does not match model " +
                    want_hash.substr(0, 16) + "...; requantization changes the hash"};
    }

    if (h.n_layers != arch.topology.n_layers && arch.topology.n_layers != 0) {
        return {StatusCode::ArchMismatch,
                "container has " + std::to_string(h.n_layers) + " layers, model has " +
                    std::to_string(arch.topology.n_layers)};
    }
    if (arch.router.n_experts != 0 && h.n_experts != arch.router.n_experts) {
        return {StatusCode::ArchMismatch,
                "container has " + std::to_string(h.n_experts) + " experts/layer, model has " +
                    std::to_string(arch.router.n_experts)};
    }

    // Per-role quantization check.
    //
    // The byte total below is a PROXY: it says the three sections add up, not that
    // each role holds the format the reader is about to decode it as. gate is
    // [expert_intermediate, d_model], up is the same, and down is its transpose —
    // all three hold the same element count — so any map that permutes dtypes
    // between roles sums identically, and the reader would slice the expert range
    // at boundaries that do not exist (q4_g is 72 B/group against q6_g's 100 at
    // G=128) with nothing to raise.
    //
    // No path produces such a map TODAY: `container_meta.json` has one
    // `dtype_gate_up` field, so gate and up cannot differ, and with them welded the
    // total is unique to the map. That is a property of a JSON schema, though, not
    // of the format — and the header could not describe the default map at all,
    // carrying one `expert_dtype` that the reader discarded into `(void)`. This
    // checks the thing directly instead of relying on the coincidence.
    if (h.has_role_quant && arch.ffn.expert_intermediate > 0 && arch.topology.d_model > 0) {
        const auto fi = arch.ffn.expert_intermediate;
        const auto d = arch.routed_expert_width();
        const auto check = [&](const char* name, TensorRole role, std::uint32_t cols,
                               const RoleQuant& on_disk) -> Status {
            const auto want = role_from_ir(arch, role, cols);
            if (want.dtype != on_disk.dtype) {
                return {StatusCode::ArchMismatch,
                        std::string("container's ") + name + " experts are " +
                            std::string(to_string(on_disk.dtype)) +
                            " but the model's quant map says " +
                            std::string(to_string(want.dtype)) +
                            "; the same total size does not make them the same bytes"};
            }
            if (want.group != on_disk.group) {
                return {StatusCode::ArchMismatch,
                        std::string("container's ") + name + " experts use group " +
                            std::to_string(on_disk.group) + " but the model's quant map implies " +
                            std::to_string(want.group)};
            }
            return {};
        };
        if (auto st = check("gate", TensorRole::ExpertGate, d, h.gate); !st.ok()) return st;
        if (auto st = check("up", TensorRole::ExpertUp, d, h.up); !st.ok()) return st;
        if (auto st = check("down", TensorRole::ExpertDown, fi, h.down); !st.ok()) return st;
    } else if (!h.has_role_quant && arch.ffn.expert_intermediate > 0 &&
               arch.topology.d_model > 0) {
        // Best-effort guard for the explicit legacy escape. V1 cannot describe
        // all three roles, but its retained pair still claims to describe gate.
        DType legacy_dtype{};
        if (!dtype_from_id(parsed.dtype_id, legacy_dtype)) {
            return {StatusCode::InvalidArgument,
                    index_path.string() + ": unknown legacy expert dtype id " +
                        std::to_string(parsed.dtype_id)};
        }
        const auto gate = role_from_ir(arch, TensorRole::ExpertGate, arch.topology.d_model);
        const auto& gate_spec = arch.quantization.for_role(TensorRole::ExpertGate);
        const auto requested_group = gate_spec.group ? gate_spec.group : kDefaultGroup;
        // Historical writers disagreed on whether this legacy field held the
        // requested or effective group. Accept either on the explicit legacy
        // path; V2's role descriptors have one unambiguous effective meaning.
        if (legacy_dtype != gate.dtype ||
            (parsed.group != requested_group && parsed.group != gate.group)) {
            return {StatusCode::ArchMismatch,
                    "legacy container's expert dtype/group does not match the model's gate "
                    "quantization"};
        }
    }

    // Expert-size cross-check.
    //
    // Independent of the descriptor above, and kept for containers written before
    // it existed — for those it is the only guard that the IR's quantization map
    // describes the bytes actually on disk. Without it, opening a q4_g container
    // with an all-f32 map succeeds and every downstream size calculation — plan
    // footprint, cap_per_layer, bytes_per_token — is wrong by the compression
    // ratio while the reads themselves still work.
    //
    // Found exactly that way: a plan predicted 393216 B/token against a measured
    // 69632, and nothing had objected.
    if (h.expert_bytes > 0 && arch.ffn.expert_intermediate > 0 && arch.topology.d_model > 0) {
        const auto fi = arch.ffn.expert_intermediate;
        const auto d = arch.routed_expert_width();
        const auto implied = expert_bytes_for(arch, fi, d, TensorRole::ExpertGate) +
                             expert_bytes_for(arch, fi, d, TensorRole::ExpertUp) +
                             expert_bytes_for(arch, d, fi, TensorRole::ExpertDown);
        if (implied != h.expert_bytes) {
            return {StatusCode::ArchMismatch,
                    "container experts are " + std::to_string(h.expert_bytes) +
                        " B but the IR's quantization map implies " + std::to_string(implied) +
                        " B; the model's quant map does not describe this container"};
        }
    }

    if (auto st = validate_ranges(parsed, arch, model_dir, shard_prefix); !st.ok()) return st;
    impl_->index = std::move(parsed.entries);

    // A digest table is now REQUIRED, where it used to be additive.
    //
    // It stopped being optional the moment it became the only thing standing
    // between a mis-ordered container and the model. `validate_layout()` catches
    // a permutation today because index order is layout order; when that is
    // relaxed to allow heat-ordered placement, the digests are what remains —
    // and a check that some containers carry and others do not is no check at
    // all. Making it mandatory now means the relaxation is a change to one
    // validator rather than a change to the threat model.
    //
    // `PayloadPolicy::Trust` is the escape, and it already says the right thing:
    // the caller is asking to read without checking. Nothing can add a table to a
    // container that lacks one — that is a rewrite, and the index is written once.
    if (!h.has_digests && opts.payload != PayloadPolicy::Trust) {
        return {StatusCode::Unsupported,
                index_path.string() +
                    " carries no digest table, so nothing would check the bytes it hands "
                    "the model. Reconvert it with the current converter"};
    }

    if (h.has_digests && opts.payload == PayloadPolicy::VerifyOnFirstRead) {
        impl_->digests = std::move(parsed.digests);
        impl_->verified = std::make_unique<std::atomic<std::uint8_t>[]>(impl_->index.size());
        for (std::size_t i = 0; i < impl_->index.size(); ++i)
            impl_->verified[i].store(0, std::memory_order_relaxed);
        impl_->payload = PayloadPolicy::VerifyOnFirstRead;
    }

    std::vector<std::uint64_t> shard_sizes;
    shard_sizes.reserve(h.n_shards);
    for (std::uint32_t s = 0; s < h.n_shards; ++s) {
        char name[32];
        std::snprintf(name, sizeof(name), "%s%05u.bin", shard_prefix.c_str(), s);
        const auto shard_path = fs::path(model_dir) / name;
        ShardFile f;
        if (!f.open(shard_path)) {
            return {StatusCode::NotFound, std::string("missing shard ") + name};
        }
        std::error_code size_ec;
        const auto size = fs::file_size(shard_path, size_ec);
        if (size_ec) {
            return {StatusCode::IoError,
                    "cannot stat shard " + shard_path.string() + ": " + size_ec.message()};
        }
        shard_sizes.push_back(size);
        impl_->shards.push_back(std::move(f));
    }

    // Validate every range once at open. A bad shard id or a range beyond EOF
    // otherwise survives admission and fails only if routing happens to select
    // that expert; a wrong nonzero length is worse, because it can be read and
    // decoded under the wrong section boundaries.
    std::uint64_t indexed_bytes = 0;
    for (std::size_t i = 0; i < impl_->index.size(); ++i) {
        const auto& e = impl_->index[i];
        if (e.length == 0) continue;
        if (e.shard >= shard_sizes.size()) {
            return {StatusCode::InvalidArgument,
                    "expert " + std::to_string(i) + " names missing shard " +
                        std::to_string(e.shard)};
        }
        if (e.offset % kDirectIoAlign != 0) {
            return {StatusCode::InvalidArgument,
                    "expert " + std::to_string(i) + " is at unaligned offset " +
                        std::to_string(e.offset)};
        }
        if (h.expert_bytes != 0 && e.length != h.expert_bytes) {
            return {StatusCode::InvalidArgument,
                    "expert " + std::to_string(i) + " has length " +
                        std::to_string(e.length) + ", expected " +
                        std::to_string(h.expert_bytes)};
        }
        if (e.offset > std::numeric_limits<std::uint64_t>::max() - e.length ||
            e.offset + e.length > shard_sizes[e.shard]) {
            return {StatusCode::InvalidArgument,
                    "expert " + std::to_string(i) + " extends beyond shard " +
                        std::to_string(e.shard)};
        }
        if (indexed_bytes > std::numeric_limits<std::uint64_t>::max() - e.length) {
            return {StatusCode::InvalidArgument, "indexed expert byte total overflows"};
        }
        indexed_bytes += e.length;
    }
    if (indexed_bytes != parsed.total_bytes) {
        return {StatusCode::InvalidArgument,
                "index accounts for " + std::to_string(indexed_bytes) +
                    " expert bytes but the header declares " +
                    std::to_string(parsed.total_bytes)};
    }
    return {};
}

catch (const std::bad_alloc&) {
    return {StatusCode::IoError, "insufficient memory for container index"};
}

ExpertLocation ExpertStore::locate(LayerIndex layer, ExpertId expert) const noexcept {
    const auto s = impl_->slot(layer, expert);
    if (s >= impl_->index.size()) return {};
    return impl_->index[s];
}

StatusCode ExpertStore::read(LayerIndex layer, ExpertId expert, ByteSpan dst) noexcept {
    const auto s = impl_->slot(layer, expert);
    if (s >= impl_->index.size()) return StatusCode::NotFound;
    const auto loc = impl_->index[s];
    if (loc.shard >= impl_->shards.size()) return StatusCode::NotFound;
    if (dst.size() < loc.length) return StatusCode::InvalidArgument;

    const auto& f = impl_->shards[loc.shard];
    if (!f.valid()) return StatusCode::IoError;
    if (!f.read_at(loc.offset, dst.data(), loc.length)) return StatusCode::IoError;
    impl_->bytes_read.fetch_add(loc.length, std::memory_order_relaxed);

    // Verified ONCE per expert, over the bytes already sitting in the caller's
    // buffer — no second read, and no cost at all on a cache hit, because a hit
    // never reaches this function.
    //
    // Per read rather than per expert would roughly double miss latency (SHA-256
    // runs at about the speed of a good NVMe read) and buy nothing: the bytes on
    // disk do not change between two reads of the same expert within one process,
    // and if they did, the next process would catch it.
    if (impl_->payload == PayloadPolicy::VerifyOnFirstRead && impl_->verified != nullptr &&
        loc.length > 0 && impl_->verified[s].load(std::memory_order_relaxed) == 0) {
        if (expert_digest(layer, expert, CByteSpan(dst.data(), loc.length)) !=
            impl_->digests[s]) {
            // Deliberately NOT latched into the store. The caller is told this
            // expert is corrupt on every read of it, because a store that answered
            // "corrupt" once and then went quiet would let a retry loop feed the
            // bad bytes straight into the model.
            return StatusCode::DataCorruption;
        }
        impl_->verified[s].store(1, std::memory_order_relaxed);
    }
    return StatusCode::Ok;
}

ExpertStore::Pending
ExpertStore::read_async(LayerIndex layer, ExpertId expert, ByteSpan dst) noexcept {
    // Synchronous behind an async-shaped interface, deliberately.
    //
    // The bounded load pool lands with MemoryHierarchy: an async read is only
    // useful if there is a cache with a policy to compute against while it is in
    // flight, and there is not one yet. Shipping a thread pool now would mean
    // shipping a pool whose only observable effect is overhead.
    Pending p;
    p.impl_ = std::make_unique<Pending::Impl>();
    p.impl_->code = read(layer, expert, dst);
    return p;
}

namespace {

/// Read an index file whole. Both entry points below want the bytes, not a
/// stream, because the file is small by design and parsing it twice from two
/// different reads is how a checker and a writer end up describing two files.
Status read_index_file(const fs::path& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "cannot open " + path.string()};
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return {};
}

} // namespace

Status verify_payload(const std::string& model_dir,
                      PayloadReport& out,
                      const std::string& index_file) try {
    out = {};
    const fs::path index_path = fs::path(model_dir) / index_file;
    std::string raw;
    if (auto st = read_index_file(index_path, raw); !st.ok()) {
        return {StatusCode::NotFound, "no " + index_file + " in " + model_dir};
    }

    ParsedIndex ix;
    if (auto st = parse_index(index_path.string(), raw, ix); !st.ok()) return st;
    if (!ix.header.has_digests) {
        return {StatusCode::Unsupported,
                index_path.string() +
                    " carries no digest table, so there is nothing to check the payload "
                    "against. Reconvert it with the current converter"};
    }
    // Structure before payload, and the SAME structure open() requires.
    //
    // Without this, `soma verify` reported OK on a container `open()` refuses:
    // a swapped pair of experts whose payloads moved with them has valid digests
    // and a perfect tiling, and only index order tells it apart. The post-transfer
    // check must not bless what serving will reject.
    if (auto st = validate_layout(ix, model_dir, shard_prefix_for(index_file)); !st.ok()) {
        return st;
    }

    const auto n_experts = ix.header.n_experts;
    auto st = walk_experts(ix, model_dir, index_file, [&](std::size_t slot, CByteSpan bytes) {
        ++out.experts_checked;
        out.bytes_checked += bytes.size();
        const auto l = static_cast<LayerIndex>(slot / n_experts);
        const auto e = static_cast<ExpertId>(slot % n_experts);
        if (expert_digest(l, e, bytes) == ix.digests[slot]) return;
        if (out.mismatches == 0) {
            out.first_bad_layer = static_cast<LayerIndex>(slot / n_experts);
            out.first_bad_expert = static_cast<ExpertId>(slot % n_experts);
        }
        ++out.mismatches;
    });
    if (!st.ok()) return st;

    if (out.mismatches > 0) {
        return {StatusCode::DataCorruption,
                std::to_string(out.mismatches) + " of " + std::to_string(out.experts_checked) +
                    " experts do not match their digest, first at layer " +
                    std::to_string(out.first_bad_layer) + " expert " +
                    std::to_string(out.first_bad_expert) +
                    "; the shards were damaged after they were written"};
    }
    return {};
} catch (const std::bad_alloc&) {
    return {StatusCode::OutOfMemory, "verify_payload ran out of memory"};
}

Status verify_identity(const std::string& model_dir,
                       const ArchIr& arch,
                       const std::string& index_file,
                       IdentityReport* report) try {
    if (report != nullptr) *report = {};
    const fs::path index_path = fs::path(model_dir) / index_file;
    std::string raw;
    {
        std::ifstream in(index_path, std::ios::binary);
        if (!in) return {StatusCode::NotFound, "no " + index_file + " in " + model_dir};
        raw.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    ParsedIndex ix;
    if (auto st = parse_index(index_path.string(), raw, ix); !st.ok()) return st;

    // The container's identity, which is what open() compares against.
    const std::string& identity =
        arch.container_arch_hash.empty() ? arch.arch_hash : arch.container_arch_hash;
    if (identity.empty()) {
        return {StatusCode::InvalidArgument,
                "the resolved IR carries no arch_hash, so there is nothing to compare against"};
    }
    if (arch.ffn.expert_intermediate == 0 || arch.topology.d_model == 0) {
        return {StatusCode::InvalidArgument,
                "the resolved IR does not describe expert shapes, so the container's "
                "quantization cannot be verified"};
    }
    if (!ix.header.has_role_quant) {
        return {StatusCode::ArchMismatch,
                "descriptorless v1 container: its byte total cannot prove which same-sized "
                "expert role owns which dtype. Reconvert it with the current converter, or "
                "use --allow-unstamped only for development"};
    }
    // Empty is a FAILURE, not an invitation. This function used to fill it, and a
    // container that reached here unstamped simply got one; now the converter
    // writes it, so an empty hash means the container was never finished — and
    // nothing here can finish it without rewriting the index, which is the
    // rewrite this change exists to remove.
    if (ix.header.arch_hash.empty()) {
        return {StatusCode::ArchMismatch,
                index_path.string() + " carries no arch_hash: it was written by a converter "
                "that did not stamp it, or with --no-identity. Reconvert it with the current "
                "converter, which asks the engine for the identity while it writes"};
    }

    const auto fi = arch.ffn.expert_intermediate;
    const auto d = arch.routed_expert_width();
    const auto gate = role_from_ir(arch, TensorRole::ExpertGate, d);
    const auto up = role_from_ir(arch, TensorRole::ExpertUp, d);
    const auto down = role_from_ir(arch, TensorRole::ExpertDown, fi);

    // The byte total the IR implies, against the one the index declares.
    const auto implied = expert_bytes_for(arch, fi, d, TensorRole::ExpertGate) +
                         expert_bytes_for(arch, fi, d, TensorRole::ExpertUp) +
                         expert_bytes_for(arch, d, fi, TensorRole::ExpertDown);
    if (ix.header.expert_bytes > 0 && implied != ix.header.expert_bytes) {
        return {StatusCode::ArchMismatch,
                "the container's experts are " +
                    std::to_string(ix.header.expert_bytes) +
                    " B but the IR's quantization map implies " + std::to_string(implied) +
                    " B. container_meta.json does not describe these shards"};
    }
    // The descriptor is evidence rather than something to overwrite:
    // disagreeing with it means the IR moved under bytes that did not move.
    const auto same = [](const RoleQuant& a, const RoleQuant& b) {
        return a.dtype == b.dtype && a.group == b.group;
    };
    if (!same(ix.header.gate, gate) || !same(ix.header.up, up) ||
        !same(ix.header.down, down)) {
        return {StatusCode::ArchMismatch,
                "this container describes expert roles as " +
                    std::string(to_string(ix.header.gate.dtype)) + "/" +
                    std::string(to_string(ix.header.up.dtype)) + "/" +
                    std::string(to_string(ix.header.down.dtype)) +
                    " and the IR says " + std::string(to_string(gate.dtype)) + "/" +
                    std::string(to_string(up.dtype)) + "/" +
                    std::string(to_string(down.dtype)) +
                    "; an equal byte total cannot prove those layouts equivalent"};
    }
    if (ix.header.arch_hash != identity) {
        return {StatusCode::ArchMismatch,
                "this container carries arch_hash " + ix.header.arch_hash.substr(0, 16) +
                    "... and the IR hashes to " + identity.substr(0, 16) +
                    "...; requantization changes the hash, and these are the same bytes"};
    }

    if (auto st = validate_ranges(ix, arch, model_dir, shard_prefix_for(index_file)); !st.ok())
        return st;

    // PAYLOAD. The one pass in this function that reads a shard byte.
    //
    // Everything above proves the container is SHAPED right: ranges pack
    // canonically, shard files are exactly the implied size, the roles carry the
    // dtypes the IR names, the identity matches. A shard of exactly the right
    // size full of wrong bytes satisfies all of it.
    if (!ix.header.has_digests) {
        return {StatusCode::Unsupported,
                index_path.string() + " carries no digest table, so its payload cannot be "
                "checked at all. Reconvert it with the current converter"};
    }
    const auto n_experts = std::max<std::uint32_t>(ix.header.n_experts, 1);
    std::size_t checked = 0;
    Status mismatch;
    if (auto st = walk_experts(ix, model_dir, index_file,
                               [&](std::size_t slot, CByteSpan bytes) {
                                   if (!mismatch.ok()) return;
                                   ++checked;
                                   const auto layer = static_cast<LayerIndex>(slot / n_experts);
                                   const auto expert = static_cast<ExpertId>(slot % n_experts);
                                   if (expert_digest(layer, expert, bytes) == ix.digests[slot])
                                       return;
                                   mismatch = {StatusCode::DataCorruption,
                                               "layer " + std::to_string(layer) + " expert " +
                                                   std::to_string(expert) +
                                                   " does not match the digest the converter "
                                                   "recorded; the shards changed after they "
                                                   "were written"};
                               });
        !st.ok())
        return st;
    if (!mismatch.ok()) return mismatch;

    if (report != nullptr) report->experts_checked = checked;
    return {};
}

catch (const std::bad_alloc&) {
    return {StatusCode::IoError, "insufficient memory to verify container index"};
}

const char* to_string(BandwidthMethod method) noexcept {
    switch (method) {
    case BandwidthMethod::Unbuffered:
        return "unbuffered";
    case BandwidthMethod::CacheEvicted:
        return "cache-eviction-advised";
    case BandwidthMethod::Buffered:
        return "buffered";
    }
    return "unknown";
}

Status ExpertStore::measure_bandwidth(std::uint64_t& bytes_per_second, BandwidthReport* report) {
    bytes_per_second = 0;
    if (report != nullptr) *report = {};
    if (impl_->index.empty()) {
        return {StatusCode::InvalidArgument, "no container open"};
    }

    // Measured with reads THE SIZE OF THIS MODEL'S EXPERTS, in a random order,
    // and COLD.
    //
    // All three matter. A 2.4 MB read and an 88 MB read do not achieve the same
    // bandwidth on the same drive; a sequential sweep measures readahead rather
    // than the random access routing actually produces; and a warm read measures
    // the page cache, which on a container the converter has just written is most
    // of it. The third was the one that was missing, and it fails in the
    // dangerous direction — a probe reporting memcpy speed says streaming is
    // affordable on a host where it is not.
    const std::size_t n = impl_->index.size();

    std::vector<std::size_t> order;
    order.reserve(n);
    std::uint32_t maxlen = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (impl_->index[i].length == 0) continue; // a dense layer's empty slot
        order.push_back(i);
        maxlen = std::max(maxlen, impl_->index[i].length);
    }
    if (order.empty()) return {StatusCode::InvalidArgument, "container holds no routed experts"};
    if (maxlen > std::numeric_limits<std::uint32_t>::max() - (kDirectIoAlign - 1))
        return {StatusCode::InvalidArgument, "padded expert read exceeds supported length"};

    std::mt19937 rng(20260729);
    std::shuffle(order.begin(), order.end(), rng);
    const std::size_t samples = std::min<std::size_t>(order.size(), 64);

    // The probe's OWN handles and its OWN buffer, which is what lets it read
    // unbuffered at all. An unbuffered read needs an aligned destination and a
    // length rounded up to the block size; ExpertStore::read() has neither, since
    // it writes into an exact-length buffer the memory tier allocated. Here both
    // ends belong to the probe.
    //
    // Reading into the PADDING is safe by construction: every shard is padded to
    // kDirectIoAlign and validate_ranges() has already required the file to be
    // exactly that size, so the rounded-up read never crosses EOF.
    auto method =
        probe_direct_disabled() ? BandwidthMethod::CacheEvicted : BandwidthMethod::Unbuffered;
    std::vector<ShardFile> probe(method == BandwidthMethod::Unbuffered ? impl_->header.n_shards
                                                                      : 0u);
    const auto& prefix = impl_->shard_prefix;
    for (std::uint32_t i = 0; i < probe.size() && method == BandwidthMethod::Unbuffered; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "%s%05u.bin", prefix.c_str(), i);
        if (!probe[i].open(fs::path(impl_->dir) / name, ShardFile::Io::Unbuffered)) {
            method = BandwidthMethod::CacheEvicted;
        }
    }

    AlignedBuffer buf;
    if (method != BandwidthMethod::Unbuffered) {
        // Fall back to the store's own buffered handles, dropping each range from
        // the page cache immediately before reading it. Whether the platform has
        // any way to say that is checked on the first range rather than assumed,
        // so a system with no fadvise reports `buffered` and not a stronger claim.
        //
        // The check is that the advice was ACCEPTED, not that pages were provably
        // evicted — nothing portable reports the latter. On a RAM-backed
        // filesystem it will be accepted and do nothing, but there the number is
        // not misleading either: the file really is memory, and that is what the
        // storage costs.
        probe.clear();
        bool flushed = true;
        for (const auto& shard : impl_->shards)
            if (!shard.flush_for_probe()) flushed = false;
        const auto& first = impl_->index[order[0]];
        method = flushed && impl_->shards[first.shard].drop_from_cache(
                                first.offset, (first.length + kDirectIoAlign - 1) /
                                                  kDirectIoAlign * kDirectIoAlign)
                     ? BandwidthMethod::CacheEvicted
                     : BandwidthMethod::Buffered;
    }
    if (!buf.allocate(maxlen)) {
        return {StatusCode::OutOfMemory, "cannot allocate the bandwidth probe buffer"};
    }

    const auto read_one = [&](const ExpertLocation& loc) noexcept {
        if (loc.shard >= impl_->shards.size()) return false;
        if (!probe.empty()) {
            const auto padded = static_cast<std::uint32_t>(
                (loc.length + kDirectIoAlign - 1) / kDirectIoAlign * kDirectIoAlign);
            return probe[loc.shard].valid() &&
                   probe[loc.shard].read_at(loc.offset, buf.data(), padded);
        }
        if (method == BandwidthMethod::CacheEvicted &&
            !impl_->shards[loc.shard].drop_from_cache(
                loc.offset, (loc.length + kDirectIoAlign - 1) / kDirectIoAlign * kDirectIoAlign))
            method = BandwidthMethod::Buffered;
        return impl_->shards[loc.shard].read_at(loc.offset, buf.data(), loc.length);
    };

    // Probe directly so concurrent model reads retain verification. Probe I/O
    // neither changes the shared policy nor marks any expert verified.
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t moved = 0;
    for (std::size_t i = 0; i < samples; ++i) {
        const auto& loc = impl_->index[order[i]];
        if (!read_one(loc)) return {StatusCode::IoError, "bandwidth probe read failed"};
        moved += loc.length;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // The probe's own traffic is not a cache miss the engine caused, so it does
    // not belong in the hit-rate accounting.

    if (secs <= 0.0) return {StatusCode::Internal, "bandwidth probe took no measurable time"};
    bytes_per_second = static_cast<std::uint64_t>(static_cast<double>(moved) / secs);
    if (report != nullptr) {
        report->bytes_per_second = bytes_per_second;
        report->method = method;
        report->bytes_moved = moved;
        report->samples = static_cast<std::uint32_t>(samples);
    }
    return {};
}

} // namespace soma
