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

inline constexpr std::uint32_t kContainerVersion = 1;

/// Sidecar index entry. Fixed-size and POD so the index loads with one read.
struct ExpertLocation {
    std::uint32_t shard = 0;
    std::uint64_t offset = 0; ///< kDirectIoAlign-aligned
    std::uint32_t length = 0;
};

struct ContainerHeader {
    std::uint32_t version = kContainerVersion;
    std::string arch_hash;
    std::uint32_t n_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint32_t n_shards = 0;
    std::uint64_t expert_bytes = 0;
};

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
    Status open(const std::string& model_dir, const ArchIr& arch);
    /// Open an additional container using the same wire format but distinct
    /// index/shard names. Optional speculative backends use this without making
    /// the ordinary store or memory hierarchy aware of a model family.
    Status open_indexed(const std::string& model_dir,
                        const ArchIr& arch,
                        const std::string& index_file,
                        const std::string& shard_prefix);
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
