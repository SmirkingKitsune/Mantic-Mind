#pragma once

// Soma — safetensors container reader.
//
// Format:
//   [0..8)      u64 little-endian header length N
//   [8..8+N)    UTF-8 JSON: { "name": {dtype, shape, data_offsets:[a,b]}, ... }
//               plus an optional "__metadata__" string map
//   [8+N..)     tensor bytes; data_offsets are relative to THIS point
//
// Used at G0/G1 to read HF checkpoints directly. From G2 the streaming path
// reads Soma's own container instead (expert_store.hpp) — one expert per
// contiguous, 4 KB-aligned range with a sidecar index, which safetensors does
// not provide. This reader stays, because admission has to read the source.

#include "soma/quant.hpp"
#include "soma/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace soma {

struct TensorView {
    std::string name;
    DType dtype = DType::F32;
    std::vector<std::int64_t> shape;
    CByteSpan bytes;

    std::size_t elements() const noexcept;
    std::int64_t dim(std::size_t i) const noexcept;

    std::size_t rank() const noexcept { return shape.size(); }

    /// Typed view. Returns an empty span when dtype is not F32 — callers check
    /// rather than reinterpreting whatever is there.
    std::span<const float> f32() const noexcept;
};

class SafeTensors {
public:
    SafeTensors();
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    SafeTensors(SafeTensors&&) noexcept;
    SafeTensors& operator=(SafeTensors&&) noexcept;
    ~SafeTensors();

    /// Read one .safetensors file into memory.
    Status open(const std::string& path);

    /// Read a sharded checkpoint. Accepts either a directory containing
    /// model.safetensors, or one containing model.safetensors.index.json plus
    /// its shards. Tensor names are unioned across shards.
    Status open_dir(const std::string& dir);

    void close();

    const TensorView* find(std::string_view name) const noexcept;

    /// Lookup that reports what it wanted. Every miss in a model loader is a
    /// bug in the name adapter, and "tensor not found" without the name costs
    /// more time than the check saves.
    Status require(std::string_view name, const TensorView*& out) const;

    std::vector<std::string> names() const;
    std::size_t size() const noexcept;
    std::uint64_t bytes_loaded() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// safetensors dtype strings ("F32", "BF16", …) → DType.
Status parse_safetensors_dtype(std::string_view text, DType& out);

} // namespace soma
