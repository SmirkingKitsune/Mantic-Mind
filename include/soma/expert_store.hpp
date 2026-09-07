#pragma once

// Soma — the on-disk expert container and its sidecar index.
//
// Streaming imposes hard requirements on the container. All of them are
// satisfied at ADMISSION, never at runtime:
//
//   * one expert = one contiguous byte range, gate/up/down interleaved so a
//     single read fetches the whole SwiGLU triple
//   * 4 KB-aligned offsets, for O_DIRECT
//   * a sidecar expert_id -> (shard, offset, len) index, so a cache miss never
//     parses a safetensors header
//   * fused 3D expert tensors pre-transposed
//
// The last one matters more than it looks: transposing at runtime would mutate
// state the model tier promises is immutable (model.hpp), and the lock-free
// read of tier 1 depends on that promise being literal.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace soma {

inline constexpr std::uint32_t kLegacyContainerVersion = 1;
inline constexpr std::uint32_t kContainerVersion = 2;

/// Container `flags` bits, and the set this build understands.
///
/// MUST-UNDERSTAND: a v2 reader that sees a bit it does not know refuses the
/// container rather than parsing on past a field it cannot see. V1 readers
/// ignored this word, so the role descriptor also requires a version bump: a
/// flag cannot retroactively make an old reader fail closed.
inline constexpr std::uint32_t kFlagPerRoleQuant = 1u << 0;
inline constexpr std::uint32_t kKnownContainerFlags = kFlagPerRoleQuant;

/// One expert role's quantization, as the shards were actually written.
///
/// `group` is the EFFECTIVE group — the reduced value quantize_tensor() used for
/// that role's row width — not the group the operator asked for. Storing the
/// effective one means the reader compares what is on disk against what the IR
/// implies, rather than comparing two requests that were both honoured
/// differently.
struct RoleQuant {
    DType dtype = DType::F32;
    std::uint32_t group = 0;
};

/// Wire ids for the three routed roles. Pinned to TensorRole so there is no
/// second mapping table to drift, and asserted so that REORDERING the enum
/// breaks the build instead of silently redefining the format.
static_assert(static_cast<std::uint32_t>(TensorRole::ExpertGate) == 2, "container role wire id");
static_assert(static_cast<std::uint32_t>(TensorRole::ExpertUp) == 3, "container role wire id");
static_assert(static_cast<std::uint32_t>(TensorRole::ExpertDown) == 4, "container role wire id");

// DType is serialized as its enum ordinal. Pin every value used by the format so
// an enum insertion is a compile failure rather than a silent wire-format change.
static_assert(static_cast<std::uint32_t>(DType::F32) == 0, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::F16) == 1, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::BF16) == 2, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::Q8_0) == 3, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::Q6_G) == 4, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::Q5_G) == 5, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::Q4_G) == 6, "container dtype wire id");
static_assert(static_cast<std::uint32_t>(DType::Q4_0) == 7, "container dtype wire id");

enum class IdentityPolicy : std::uint8_t {
    RequireStamped,
    AllowUnstamped,
    /// Auxiliary indexes whose identity is owned by a parent container. This is
    /// deliberately distinct from an empty ArchIr hash: omitting a hash must not
    /// turn the default policy off by accident.
    SkipArchHash,
};

/// How strictly to open a container.
struct OpenOptions {
    /// Open a container whose `arch_hash` was never stamped.
    ///
    /// Off by default, and that is the entire point. `soma stamp` is what
    /// writes the hash; for as long as an unstamped container opened silently,
    /// stamping was optional in practice, and an optional integrity check is the
    /// one that does not run. Without the stamp the remaining checks compare the
    /// container against a map read out of its own meta, so they cannot see the IR
    /// moving under a container that did not. It also enables the explicitly
    /// documented legacy-v1 reader; a hash mismatch and an empty expected
    /// identity are still errors.
    IdentityPolicy identity = IdentityPolicy::RequireStamped;
};

/// Sidecar index entry. Fixed-size and POD so the index loads with one read.
struct ExpertLocation {
    std::uint32_t shard = 0;
    std::uint64_t offset = 0; ///< kDirectIoAlign-aligned
    std::uint32_t length = 0;
};

struct ContainerHeader {
    std::uint32_t version = kContainerVersion;
    std::uint32_t flags = 0;
    std::string arch_hash;
    std::uint32_t n_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint32_t n_shards = 0;
    std::uint64_t expert_bytes = 0;

    /// Present only when `flags & kFlagPerRoleQuant`. A v1 container written
    /// before the descriptor existed carries a single expert dtype that cannot
    /// even EXPRESS the default map (q4_g gate/up with q6_g down), so there is
    /// nothing to fall back to except the byte total.
    bool has_role_quant = false;
    RoleQuant gate;
    RoleQuant up;
    RoleQuant down;
};

/// Write a container's identity into its index, in place.
///
/// V2 separates the two pieces of evidence that used to be absent or
/// inexpressible:
///
///   * `arch_hash`, computed by compute_arch_hash() from the SAME canonical IR
///     the engine loads. convert.py leaves it empty because a second hash
///     implementation in Python would agree until it did not — which is correct,
///     and left the gate permanently dormant because nothing else stamped it.
///   * the per-role quantization descriptor, written by the converter, so the
///     reader can check gate, up and down individually instead of comparing one
///     byte total. stamp verifies this evidence; it does not manufacture it.
///
/// Refuses rather than stamps when the IR and the payload disagree: a stamp is
/// an assertion that this container is what the IR says it is, and stamping past
/// a disagreement would launder exactly the error the hash exists to catch.
///
/// A descriptor-less v1 index is refused: its single dtype plus byte total cannot
/// prove which format occupies each role, so manufacturing a descriptor from the
/// same metadata would certify the ambiguity this function exists to remove.
///
/// Rewrites the small index file only — the shards are not touched, and the
/// write goes through a temporary plus rename so an interrupted stamp leaves the
/// previous index intact.
Status stamp_container(const std::string& model_dir,
                       const ArchIr& arch,
                       const std::string& index_file = "soma.container");

/// Reads expert bytes from disk. Owns the file handles, the sidecar index, and
/// the bounded background load pool.
///
/// Does no caching — that is MemoryHierarchy's job. The split exists so eviction
/// policy and I/O mechanics can be tested independently, and so that the OS page
/// cache sits naturally underneath as a free L2.
class ExpertStore {
public:
    /// Completion handle for an async read. `wait()` is what MemoryHierarchy
    /// blocks on inside acquire().
    class Pending {
    public:
        Pending() noexcept = default;
        Pending(const Pending&) = delete;
        Pending& operator=(const Pending&) = delete;
        Pending(Pending&&) noexcept;
        Pending& operator=(Pending&&) noexcept;
        ~Pending();

        explicit operator bool() const noexcept;
        StatusCode wait() noexcept;

    private:
        friend class ExpertStore;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;
    ~ExpertStore();

    /// Opens the container and loads the sidecar index. Refuses to open when the
    /// container's arch_hash does not match the model being loaded.
    Status open(const std::string& model_dir, const ArchIr& arch, OpenOptions opts = {});
    /// Open an additional container using the same wire format but distinct
    /// index/shard names. Optional speculative backends use this without making
    /// the ordinary store or memory hierarchy aware of a model family.
    Status open_indexed(const std::string& model_dir,
                        const ArchIr& arch,
                        const std::string& index_file,
                        const std::string& shard_prefix,
                        OpenOptions opts = {});
    void close();

    const ContainerHeader& header() const noexcept;
    ExpertLocation locate(LayerIndex layer, ExpertId expert) const noexcept;

    /// Synchronous read into a caller-provided, aligned destination.
    StatusCode read(LayerIndex layer, ExpertId expert, ByteSpan dst) noexcept;

    /// Async read via the bounded load pool, so resident experts compute while
    /// cold ones load. Returns a falsy Pending when the pool is saturated —
    /// callers fall back to a synchronous read rather than queueing unboundedly.
    Pending read_async(LayerIndex layer, ExpertId expert, ByteSpan dst) noexcept;

    /// Measured with reads the size of THIS model's experts, not a spec-sheet
    /// number. A 2.4 MB read and an 88 MB read do not achieve the same bandwidth
    /// on the same drive, and using one headline figure is how a verdict ends up
    /// confidently wrong.
    Status measure_bandwidth(std::uint64_t& bytes_per_second);

    std::uint64_t bytes_read() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace soma
