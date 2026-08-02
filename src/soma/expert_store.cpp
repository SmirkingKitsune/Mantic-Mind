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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
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
        if (!ok || p + sizeof(T) > end) {
            ok = false;
            return v;
        }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    std::string bytes(std::size_t n) {
        if (!ok || p + n > end) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
};

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

    bool open(const std::filesystem::path& p) noexcept {
        close();
#if defined(_WIN32)
        h_ = ::CreateFileW(p.wstring().c_str(),
                           GENERIC_READ,
                           FILE_SHARE_READ,
                           nullptr,
                           OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL,
                           nullptr);
#else
        h_ = ::open(p.c_str(), O_RDONLY);
#endif
        return valid();
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

struct ExpertStore::Impl {
    ContainerHeader header{};
    std::vector<ExpertLocation> index;
    std::vector<ShardFile> shards;
    /// Atomic: incremented from every thread that reads, with no lock held.
    std::atomic<std::uint64_t> bytes_read{0};
    std::string dir;

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

Status ExpertStore::open(const std::string& model_dir, const ArchIr& arch) {
    close();
    impl_->dir = model_dir;

    const fs::path index_path = fs::path(model_dir) / "soma.container";
    std::ifstream in(index_path, std::ios::binary);
    if (!in) {
        return {StatusCode::NotFound, "no soma.container in " + model_dir};
    }
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() < sizeof(kMagic) || std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) {
        return {StatusCode::InvalidArgument, index_path.string() + ": bad magic"};
    }

    const auto* base = reinterpret_cast<const std::byte*>(raw.data());
    Cursor c{base + sizeof(kMagic), base + raw.size(), true};

    auto& h = impl_->header;
    h.version = c.read<std::uint32_t>();
    if (h.version != kContainerVersion) {
        return {StatusCode::VersionMismatch,
                "container version " + std::to_string(h.version) +
                    " != " + std::to_string(kContainerVersion)};
    }
    (void)c.read<std::uint32_t>(); // flags
    const auto hash_len = c.read<std::uint32_t>();
    h.arch_hash = c.bytes(hash_len);

    h.n_layers = c.read<std::uint32_t>();
    h.n_experts = c.read<std::uint32_t>();
    h.n_shards = c.read<std::uint32_t>();
    const auto dtype_id = c.read<std::uint32_t>();
    const auto group = c.read<std::uint32_t>();
    h.expert_bytes = c.read<std::uint64_t>();
    const auto total = c.read<std::uint64_t>();
    (void)dtype_id;
    (void)group;
    (void)total;

    if (!c.ok) return {StatusCode::InvalidArgument, index_path.string() + ": truncated header"};

    // arch_hash gate.
    //
    // An empty hash means the container was written but never STAMPED: convert.py
    // cannot compute it, because the canonical hash is defined by the C++ IR
    // canonicalization and a second implementation in Python would agree until it
    // did not. Stamping is `soma admit-verify`'s job.
    //
    // Unstamped is accepted; MISMATCHED is refused. Reading q4 bytes as q6
    // produces finite, wrong numbers rather than an error, so this is the only
    // place it can be caught.
    if (!h.arch_hash.empty() && !arch.arch_hash.empty() && h.arch_hash != arch.arch_hash) {
        return {StatusCode::ArchMismatch,
                "container arch_hash " + h.arch_hash.substr(0, 16) + "... does not match model " +
                    arch.arch_hash.substr(0, 16) + "...; requantization changes the hash"};
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

    // Expert-size cross-check.
    //
    // The container is stamped with no arch_hash today (convert.py cannot compute
    // the canonical one), so this is the strongest available guard that the IR's
    // quantization map describes the bytes actually on disk. Without it, opening
    // a q4_g container with an all-f32 map succeeds and every downstream
    // size calculation — plan footprint, cap_per_layer, bytes_per_token — is
    // wrong by the compression ratio while the reads themselves still work.
    //
    // Found exactly that way: a plan predicted 393216 B/token against a measured
    // 69632, and nothing had objected.
    if (h.expert_bytes > 0 && arch.ffn.expert_intermediate > 0 && arch.topology.d_model > 0) {
        const auto fi = arch.ffn.expert_intermediate;
        const auto d = arch.topology.d_model;
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

    const std::size_t entries = static_cast<std::size_t>(h.n_layers) * h.n_experts;
    impl_->index.resize(entries);
    for (auto& e : impl_->index) {
        e.shard = c.read<std::uint32_t>();
        e.offset = c.read<std::uint64_t>();
        e.length = c.read<std::uint32_t>();
    }
    if (!c.ok) return {StatusCode::InvalidArgument, index_path.string() + ": truncated index"};

    for (std::uint32_t s = 0; s < h.n_shards; ++s) {
        char name[32];
        std::snprintf(name, sizeof(name), "experts-%05u.bin", s);
        ShardFile f;
        if (!f.open(fs::path(model_dir) / name)) {
            return {StatusCode::NotFound, std::string("missing shard ") + name};
        }
        impl_->shards.push_back(std::move(f));
    }

    // Alignment is a format guarantee, so it is worth asserting once at open
    // rather than discovering via a failed O_DIRECT read on the hot path.
    for (std::size_t i = 0; i < impl_->index.size(); ++i) {
        if (impl_->index[i].offset % kDirectIoAlign != 0) {
            return {StatusCode::InvalidArgument,
                    "expert " + std::to_string(i) + " is at unaligned offset " +
                        std::to_string(impl_->index[i].offset)};
        }
    }
    return {};
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

Status ExpertStore::measure_bandwidth(std::uint64_t& bytes_per_second) {
    bytes_per_second = 0;
    if (impl_->index.empty()) {
        return {StatusCode::InvalidArgument, "no container open"};
    }

    // Measured with reads THE SIZE OF THIS MODEL'S EXPERTS, and in a random
    // order.
    //
    // Both matter. A 2.4 MB read and an 88 MB read do not achieve the same
    // bandwidth on the same drive, and a sequential sweep measures readahead
    // rather than the random-access pattern routing actually produces. Using a
    // spec-sheet number, or a sequential benchmark, is how a verdict ends up
    // confidently wrong.
    const std::size_t n = impl_->index.size();
    const std::size_t samples = std::min<std::size_t>(n, 64);

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(20260729);
    std::shuffle(order.begin(), order.end(), rng);

    std::uint32_t maxlen = 0;
    for (const auto& e : impl_->index)
        maxlen = std::max(maxlen, e.length);
    std::vector<std::byte> buf(maxlen);

    const auto before = impl_->bytes_read.load(std::memory_order_relaxed);
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t moved = 0;
    for (std::size_t i = 0; i < samples; ++i) {
        const auto slot = order[i];
        const auto layer = static_cast<LayerIndex>(slot / impl_->header.n_experts);
        const auto expert = static_cast<ExpertId>(slot % impl_->header.n_experts);
        if (read(layer, expert, buf) != StatusCode::Ok) {
            return {StatusCode::IoError, "bandwidth probe read failed"};
        }
        moved += impl_->index[slot].length;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // The probe's own traffic is not a cache miss the engine caused, so it does
    // not belong in the hit-rate accounting.
    impl_->bytes_read.store(before, std::memory_order_relaxed);

    if (secs <= 0.0) return {StatusCode::Internal, "bandwidth probe took no measurable time"};
    bytes_per_second = static_cast<std::uint64_t>(static_cast<double>(moved) / secs);
    return {};
}

} // namespace soma
