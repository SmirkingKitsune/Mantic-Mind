#pragma once

// Soma — the on-disk expert container and its sidecar index.
//
// Streaming imposes hard requirements on the container. All of them are
// satisfied at ADMISSION, never at runtime:
//
//   * one expert = one contiguous byte range, gate/up/down interleaved so a
//     single read fetches the whole SwiGLU triple
//   * 4 KB-aligned offsets, so no expert range shares a page with its neighbour
//     and an unbuffered read of one is legal
//   * a sidecar expert_id -> (shard, offset, len) index, so a cache miss never
//     parses a safetensors header
//   * fused 3D expert tensors pre-transposed
//
// The last one matters more than it looks: transposing at runtime would mutate
// state the model tier promises is immutable (model.hpp), and the lock-free
// read of tier 1 depends on that promise being literal.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <array>
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

/// A truncated, unbound per-expert digest table follows the index. **Retired.**
///
/// Written by converters up to `8a1ee20d`. Superseded by `kFlagExpertDigestsV2`
/// below; the bit is kept reserved so it can never be reused for something else
/// and quietly reinterpret an old table.
inline constexpr std::uint32_t kFlagExpertDigestsLegacy = 1u << 1;

/// A per-expert digest table follows the index: 32 bytes per slot, each binding
/// the expert's identity as well as its bytes.
///
/// A new bit rather than a wider table under the old one. Changing a block's
/// stride in place is exactly the silent mis-parse the version gate exists for,
/// whereas a new must-understand bit makes an older build refuse the container
/// outright — which is the correct answer, since it cannot check what it cannot
/// parse.
inline constexpr std::uint32_t kFlagExpertDigestsV2 = 1u << 2;

inline constexpr std::uint32_t kKnownContainerFlags =
    kFlagPerRoleQuant | kFlagExpertDigestsLegacy | kFlagExpertDigestsV2;

/// The SHA-256 of one expert's identity and bytes, in full.
///
/// Two things changed from the 8-byte form, and the second is the one that
/// matters:
///
///   * **Full width.** The truncation was never paying for itself — the hash is
///     computed in full either way and 24 bytes were thrown away, so the saving
///     was 24 B per expert of INDEX (375 KB on DeepSeek V4's 15,616) in exchange
///     for a permanent "is 64 bits enough" question. It is also the wrong
///     substrate to build provenance on later.
///   * **Domain separation.** The old digest hashed bytes alone, so two experts
///     with identical bytes had identical digests and remained interchangeable —
///     which is precisely the check that has to hold once index order stops being
///     structural. Binding `(layer, expert, length)` makes the digest a PLACEMENT
///     check as well as a corruption check.
///
/// Still a corruption check, not a tamper check: it answers "did these bytes
/// survive being written, copied between nodes, and sat on for a month", which is
/// the question a cluster that streams container directories actually has. It
/// binds an expert to its slot within one container, not a container to an
/// origin — that needs a signature, and a key story this project does not have.
///
/// SHA-256 because OpenSSL is already linked for arch_hash and Python's hashlib
/// is in the standard library: one obvious implementation on each side.
struct ExpertDigest {
    std::array<std::uint8_t, 32> bytes{};

    friend bool operator==(const ExpertDigest& a, const ExpertDigest& b) noexcept {
        return a.bytes == b.bytes;
    }
};

/// Zero-length slots — the dense layers — store all-zero rather than the digest
/// of an empty string, so a missing entry and a real one are never confusable.
inline constexpr std::size_t kExpertDigestBytes = 32;

/// Compute one expert's digest. Exposed so the verifier and the writer cannot
/// disagree about what is being hashed.
///
/// The preimage is `"soma/expert\0"` then little-endian `u32 layer`, `u32 expert`,
/// `u64 length`, then the payload. Spelled out because a second implementation in
/// Python has to reproduce it exactly, and "hash the bytes" left no room to say
/// which bytes.
ExpertDigest expert_digest(LayerIndex layer,
                           ExpertId expert,
                           CByteSpan bytes) noexcept;

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

/// What to do with the digest table on the read path.
enum class PayloadPolicy : std::uint8_t {
    /// Verify each expert once, the first time it is read.
    ///
    /// The cost is one SHA-256 pass over bytes already in the destination buffer,
    /// paid once per expert per process — not per read, which would roughly double
    /// miss latency for no added coverage. What it buys is that a bad byte is
    /// caught at the moment it would otherwise enter the model, on the node that
    /// has it, rather than surfacing as one cluster member quietly giving
    /// different answers.
    VerifyOnFirstRead,
    /// Read without checking. For a container with no digest table this is the
    /// only available behaviour, and it is what the bandwidth probe uses so that
    /// it measures the disk rather than the hash.
    Trust,
};

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

    /// Ignored when the container carries no digest table.
    PayloadPolicy payload = PayloadPolicy::VerifyOnFirstRead;
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

    /// Present only when `flags & kFlagExpertDigestsV2`. Nothing before this
    /// checked a single payload byte: validate_ranges() proves the ranges pack
    /// canonically and the shard files are exactly the right size, which a
    /// perfectly sized file full of wrong bytes satisfies completely.
    bool has_digests = false;
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
/// What a stamp actually did, so the caller can state which guarantee it holds.
///
/// "Digests confirmed" and "digests recorded" are different claims: the first
/// says the shards still hash to what the converter measured while it had the
/// tensors in memory, the second only pins whatever is on disk right now. A
/// command that printed the same line for both would overstate the weaker one.
struct StampReport {
    bool had_digests = false;      ///< the converter had already recorded them
    bool wrote = false;            ///< false when the stamp was an exact repeat
    std::uint64_t experts_checked = 0;
};

Status stamp_container(const std::string& model_dir,
                       const ArchIr& arch,
                       const std::string& index_file = "soma.container",
                       StampReport* report = nullptr);

/// How measure_bandwidth() got its number.
///
/// Reported rather than assumed, because the three differ by more than a
/// constant. A buffered re-read of a file the converter has just written measures
/// memcpy from the page cache — often 10x the drive — and a verdict derived from
/// that number says `stream` is cheap on a host where it is not. The probe cannot
/// always avoid it, so it says which one it got.
enum class BandwidthMethod : std::uint8_t {
    /// O_DIRECT / FILE_FLAG_NO_BUFFERING. Bypasses the local OS page cache;
    /// device and remote-server caches can still affect the measurement.
    Unbuffered,
    /// Buffered, after flushing writes and advising eviction for every range.
    /// Advice success does not prove eviction; this remains an estimate.
    CacheEvicted,
    /// Neither was available. TREAT THIS NUMBER AS AN UPPER BOUND: it may be
    /// page-cache speed, and on a freshly converted container it usually is.
    Buffered,
};

const char* to_string(BandwidthMethod method) noexcept;

struct BandwidthReport {
    std::uint64_t bytes_per_second = 0;
    BandwidthMethod method = BandwidthMethod::Buffered;
    std::uint64_t bytes_moved = 0;
    std::uint32_t samples = 0;
};

/// What a full payload check found.
struct PayloadReport {
    std::uint64_t experts_checked = 0;
    std::uint64_t bytes_checked = 0;
    std::uint64_t mismatches = 0;
    /// The first expert whose bytes disagreed, so the report names one place to
    /// look rather than a count.
    LayerIndex first_bad_layer = 0;
    ExpertId first_bad_expert = 0;
};

/// Read every expert and check it against the container's digest table.
///
/// The check nothing performed before it. `validate_ranges()` proves the index
/// packs canonically and that each shard file is exactly the size those ranges
/// imply — all of which a correctly sized file full of wrong bytes satisfies. A
/// container is written once and then COPIED: control streams it to a node, it
/// sits on that node's disk, and it is read months later. Every step after the
/// conversion can damage it, and until this existed none of them was checked.
///
/// Refuses a container with no digest table rather than reporting success over a
/// check it did not make.
///
/// Takes no ArchIr, deliberately. "Do these bytes still hash to what was
/// recorded" is well posed without one, and requiring an IR would put this out of
/// reach of the two callers that need it most: a node that holds a copied
/// container and cannot resolve its architecture, and the auxiliary DSpark index,
/// whose IR only the speculative backend can build. Whether the container matches
/// an IR is a different question, and open() and stamp_container() ask it.
Status verify_payload(const std::string& model_dir,
                      PayloadReport& out,
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
    ///
    /// Reads COLD wherever the platform allows it. The probe opens its own
    /// unbuffered handles and reads into its own aligned buffer, so it can do what
    /// the ordinary read path cannot: `read()` writes into a destination the
    /// memory tier owns, which is neither aligned nor padded, while the probe owns
    /// both ends. Where an unbuffered open is refused, each range is advised out
    /// of the page cache first. `report` says which happened, and a caller
    /// deriving a verdict from this number should look at it — a buffered figure
    /// on a freshly converted container is memcpy speed, not disk speed.
    Status measure_bandwidth(std::uint64_t& bytes_per_second, BandwidthReport* report = nullptr);

    std::uint64_t bytes_read() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace soma
