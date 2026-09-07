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
#include <cstdio>
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

/// The index file's counterpart to Cursor. Only stamp_container() writes, and it
/// writes the WHOLE file rather than patching a field, because `arch_hash` is
/// length-prefixed: stamping an empty hash moves every byte after it.
struct Writer {
    std::string out;

    template <typename T>
    void put(T v) {
        char buf[sizeof(T)];
        std::memcpy(buf, &v, sizeof(T));
        out.append(buf, sizeof(T));
    }

    void raw(std::string_view s) { out.append(s); }
};

bool dtype_from_id(std::uint32_t id, DType& out) noexcept {
    if (id > static_cast<std::uint32_t>(DType::Q4_0)) return false;
    out = static_cast<DType>(id);
    return true;
}

/// Everything the index file holds, parsed once.
///
/// Shared by open_indexed() and stamp_container() so a reader and the writer that
/// rewrites what it read cannot disagree about the layout — which is the failure
/// this whole change is about.
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
    if (!c.empty()) return {StatusCode::InvalidArgument, path + ": trailing bytes after index"};
    return {};
}

// Validate the exact parsed snapshot that will be used or stamped.
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
    std::vector<std::uint64_t> ends(h.n_shards, 0);
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < ix.entries.size(); ++i) {
        const auto& e = ix.entries[i];
        const bool moe = arch.is_moe_layer(static_cast<LayerIndex>(i / h.n_experts));
        if ((moe && e.length != expected) || (!moe && e.length != 0)) {
            return {StatusCode::InvalidArgument, "expert slot presence/length disagrees with layer kind"};
        }
        if (!moe) continue;
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

std::string serialize_index(const ParsedIndex& ix) {
    Writer w;
    w.raw(std::string_view(kMagic, sizeof(kMagic)));
    w.put<std::uint32_t>(ix.header.version);
    w.put<std::uint32_t>(ix.header.flags);
    w.put<std::uint32_t>(static_cast<std::uint32_t>(ix.header.arch_hash.size()));
    w.raw(ix.header.arch_hash);
    w.put<std::uint32_t>(ix.header.n_layers);
    w.put<std::uint32_t>(ix.header.n_experts);
    w.put<std::uint32_t>(ix.header.n_shards);
    w.put<std::uint32_t>(ix.dtype_id);
    w.put<std::uint32_t>(ix.group);
    if ((ix.header.flags & kFlagPerRoleQuant) != 0) {
        w.put<std::uint32_t>(3u);
        const auto role = [&](TensorRole r, const RoleQuant& q) {
            w.put<std::uint32_t>(static_cast<std::uint32_t>(r));
            w.put<std::uint32_t>(static_cast<std::uint32_t>(q.dtype));
            w.put<std::uint32_t>(q.group);
        };
        role(TensorRole::ExpertGate, ix.header.gate);
        role(TensorRole::ExpertUp, ix.header.up);
        role(TensorRole::ExpertDown, ix.header.down);
    }
    w.put<std::uint64_t>(ix.header.expert_bytes);
    w.put<std::uint64_t>(ix.total_bytes);
    for (const auto& e : ix.entries) {
        w.put<std::uint32_t>(e.shard);
        w.put<std::uint64_t>(e.offset);
        w.put<std::uint32_t>(e.length);
    }
    return std::move(w.out);
}

/// The role's quantization as the IR describes it, reduced to what a row of that
/// shape would actually have been written at.
RoleQuant role_from_ir(const ArchIr& arch, TensorRole role, std::uint32_t cols) {
    const auto& spec = arch.quantization.for_role(role);
    const auto requested = spec.group ? spec.group : kDefaultGroup;
    return {spec.dtype, effective_group(cols, requested)};
}

Status replace_index(const fs::path& from, const fs::path& to) {
#if defined(_WIN32)
    const auto from_w = from.wstring();
    const auto to_w = to.wstring();
    if (!::MoveFileExW(from_w.c_str(),
                       to_w.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const std::error_code ec(static_cast<int>(::GetLastError()), std::system_category());
        std::error_code ignored;
        fs::remove(from, ignored);
        return {StatusCode::IoError,
                "cannot replace " + to.string() + ": " + ec.message()};
    }
#else
    std::error_code ec;
    fs::rename(from, to, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(from, ignored);
        return {StatusCode::IoError,
                "cannot replace " + to.string() + ": " + ec.message()};
    }
#endif
    return {};
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
    // An empty hash means the container was written but never STAMPED: convert.py
    // cannot compute it, because the canonical hash is defined by the C++ IR
    // canonicalization and a second implementation in Python would agree until it
    // did not. Stamping is `soma stamp`'s job.
    //
    // UNSTAMPED IS NOW REFUSED, where it used to be accepted silently. The old
    // reading was that an unstamped container is merely un-gated, and everything
    // else the gate covers — the IR moving under a container that did not move,
    // which is what requantization and a changed family default both look like —
    // then goes uncaught, because the checks below compare the container against a
    // map read out of that same container's own meta. Accepting unstamped by
    // default also made `stamp` optional, and an optional integrity check is
    // the one that does not run.
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
                    " carries no arch_hash: it was converted but never stamped. Run "
                    "`soma stamp " + model_dir +
                    "` to bind it to the IR it was built from, or pass --allow-unstamped to "
                    "use the development-only structural and role checks without identity"};
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

Status stamp_container(const std::string& model_dir,
                       const ArchIr& arch,
                       const std::string& index_file) try {
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
                "the resolved IR carries no arch_hash, so there is nothing to stamp"};
    }
    if (arch.ffn.expert_intermediate == 0 || arch.topology.d_model == 0) {
        return {StatusCode::InvalidArgument,
                "the resolved IR does not describe expert shapes, so the container's "
                "quantization cannot be verified"};
    }
    if (!ix.header.has_role_quant) {
        return {StatusCode::ArchMismatch,
                "refusing to stamp a descriptorless v1 container: its byte total cannot prove "
                "which same-sized expert role owns which dtype. Reconvert it with the current "
                "converter, or use --allow-unstamped only for development"};
    }

    const auto fi = arch.ffn.expert_intermediate;
    const auto d = arch.routed_expert_width();
    const auto gate = role_from_ir(arch, TensorRole::ExpertGate, d);
    const auto up = role_from_ir(arch, TensorRole::ExpertUp, d);
    const auto down = role_from_ir(arch, TensorRole::ExpertDown, fi);

    // VERIFY, then stamp.
    //
    // A stamp asserts that this container is what the IR says it is. Stamping
    // past a disagreement would launder precisely the error the hash exists to
    // catch — and it would do so permanently, because from then on the container
    // opens without objection.
    const auto implied = expert_bytes_for(arch, fi, d, TensorRole::ExpertGate) +
                         expert_bytes_for(arch, fi, d, TensorRole::ExpertUp) +
                         expert_bytes_for(arch, d, fi, TensorRole::ExpertDown);
    if (ix.header.expert_bytes > 0 && implied != ix.header.expert_bytes) {
        return {StatusCode::ArchMismatch,
                "refusing to stamp: the container's experts are " +
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
                "refusing to stamp: this container describes expert roles as " +
                    std::string(to_string(ix.header.gate.dtype)) + "/" +
                    std::string(to_string(ix.header.up.dtype)) + "/" +
                    std::string(to_string(ix.header.down.dtype)) +
                    " and the IR says " + std::string(to_string(gate.dtype)) + "/" +
                    std::string(to_string(up.dtype)) + "/" +
                    std::string(to_string(down.dtype)) +
                    "; an equal byte total cannot prove those layouts equivalent"};
    }
    if (!ix.header.arch_hash.empty() && ix.header.arch_hash != identity) {
        return {StatusCode::ArchMismatch,
                "refusing to re-stamp: this container already carries arch_hash " +
                    ix.header.arch_hash.substr(0, 16) + "... and the IR now hashes to " +
                    identity.substr(0, 16) +
                    "...; requantization changes the hash, and these are the same bytes"};
    }

    if (auto st = validate_ranges(ix, arch, model_dir,
                                  index_file == "soma.dspark" ? "dspark-experts-" : "experts-");
        !st.ok()) return st;

    // Exact repeat is a true no-op: admission may safely run stamp on every
    // pass without changing mtimes or exposing a replacement race to readers.
    const auto gate_id = static_cast<std::uint32_t>(gate.dtype);
    if (ix.header.version == kContainerVersion && ix.header.arch_hash == identity &&
        ix.header.flags == kFlagPerRoleQuant && ix.dtype_id == gate_id &&
        ix.group == gate.group) {
        return {};
    }

    ix.header.version = kContainerVersion;
    ix.header.arch_hash = identity;
    ix.header.flags = kFlagPerRoleQuant;
    ix.header.gate = gate;
    ix.header.up = up;
    ix.header.down = down;
    ix.dtype_id = gate_id;
    ix.group = gate.group;
    const auto bytes = serialize_index(ix);

    // Temp plus rename, matching what the V4 converter does for the same file.
    // An interrupted stamp that left a half-written index would destroy a
    // container the shards of which are perfectly intact.
    static std::atomic<std::uint64_t> stamp_sequence{0};
#if defined(_WIN32)
    const auto process_id = static_cast<std::uint64_t>(::GetCurrentProcessId());
#else
    const auto process_id = static_cast<std::uint64_t>(::getpid());
#endif
    const auto stamp_id = stamp_sequence.fetch_add(1, std::memory_order_relaxed);
    const fs::path tmp = index_path.string() + ".tmp." + std::to_string(process_id) + "." +
                         std::to_string(stamp_id);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return {StatusCode::IoError, "cannot write " + tmp.string()};
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        out.close();
        if (!out) {
            std::error_code ignored;
            fs::remove(tmp, ignored);
            return {StatusCode::IoError, "short write to " + tmp.string()};
        }
    }
    return replace_index(tmp, index_path);
}

catch (const std::bad_alloc&) {
    return {StatusCode::IoError, "insufficient memory to stamp container index"};
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
