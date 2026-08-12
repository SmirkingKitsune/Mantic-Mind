#include "soma/quant_format.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace soma {

namespace {

void put_f32(std::byte* p, float v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}

float get_f32(const std::byte* p) noexcept {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// Largest divisor of `cols` that does not exceed `want`.
///
/// Padding a short final group would fold invented zeros into that group's
/// scale, widening its range and costing resolution on the real values. Shrinking
/// the group instead costs a little space and nothing in accuracy.
std::uint32_t usable_group(std::uint32_t cols, std::uint32_t want) noexcept {
    if (want == 0) want = kDefaultGroup;
    std::uint32_t g = std::min(want, cols);
    while (g > 1 && cols % g != 0)
        --g;
    return std::max<std::uint32_t>(g, 1);
}

} // namespace

bool is_quantized(DType dtype) noexcept {
    switch (dtype) {
    case DType::Q8_0:
    case DType::Q4_0:
    case DType::Q4_G:
    case DType::Q6_G:
        return true;
    default:
        return false;
    }
}

std::size_t group_bytes(DType dtype, std::uint32_t group) noexcept {
    switch (dtype) {
    case DType::Q8_0:
        return 4 + group;
    case DType::Q4_0:
        return 4 + (group + 1) / 2;
    case DType::Q4_G:
        return 8 + (group + 1) / 2;
    case DType::Q6_G:
        return 4 + (static_cast<std::size_t>(group) * 3 + 3) / 4;
    default:
        return 0;
    }
}

std::size_t quantized_bytes(DType dtype, std::size_t n, std::uint32_t group) noexcept {
    if (!is_quantized(dtype) || group == 0) return 0;
    const auto groups = (n + group - 1) / group;
    return groups * group_bytes(dtype, group);
}

std::uint32_t effective_group(std::uint32_t cols, std::uint32_t group) noexcept {
    return usable_group(cols, group);
}

std::size_t quantized_tensor_bytes(DType dtype,
                                   std::uint32_t rows,
                                   std::uint32_t cols,
                                   std::uint32_t group) noexcept {
    if (!is_quantized(dtype)) {
        return static_cast<std::size_t>(rows) * cols * (dtype_bits(dtype) / 8);
    }
    const auto g = usable_group(cols, group);
    return static_cast<std::size_t>(rows) * (cols / g) * group_bytes(dtype, g);
}

std::size_t QTensor::bytes_per_row() const noexcept {
    if (group == 0) return 0;
    return (static_cast<std::size_t>(cols) / group) * group_bytes(dtype, group);
}

std::size_t quantized_footprint(const QTensor& t) noexcept {
    return t.data.size();
}

WeightRef
WeightRef::from_f32(std::span<const float> data, std::uint32_t rows, std::uint32_t cols) noexcept {
    WeightRef w;
    w.f32 = data;
    w.rows = rows;
    w.cols = cols;
    return w;
}

WeightRef WeightRef::from_q(const QTensor& t) noexcept {
    return from_quantized_bytes(
        CByteSpan(t.data.data(), t.data.size()), t.dtype, t.group, t.rows, t.cols);
}

WeightRef WeightRef::from_quantized_bytes(CByteSpan data,
                                          DType dtype,
                                          std::uint32_t group,
                                          std::uint32_t rows,
                                          std::uint32_t cols) noexcept {
    WeightRef w;
    w.bytes = data;
    w.dtype = dtype;
    w.group = group ? group : effective_group(cols, kDefaultGroup);
    w.rows = rows;
    w.cols = cols;
    return w;
}

// ── quantize ─────────────────────────────────────────────────────────────────

namespace {

void quantize_group_symmetric(
    const float* src, std::uint32_t n, int max_level, float& out_scale, int* out_levels) noexcept {
    float amax = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
        amax = std::max(amax, std::fabs(src[i]));
    const float scale = (amax > 0.0f) ? amax / static_cast<float>(max_level) : 0.0f;
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    out_scale = scale;
    for (std::uint32_t i = 0; i < n; ++i) {
        const int q = static_cast<int>(std::lround(src[i] * inv));
        out_levels[i] = std::clamp(q, -max_level, max_level);
    }
}

void quantize_group_asymmetric(const float* src,
                               std::uint32_t n,
                               int levels,
                               float& out_scale,
                               float& out_min,
                               int* out_levels) noexcept {
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (std::uint32_t i = 0; i < n; ++i) {
        lo = std::min(lo, src[i]);
        hi = std::max(hi, src[i]);
    }
    const float scale = (hi > lo) ? (hi - lo) / static_cast<float>(levels) : 0.0f;
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    out_scale = scale;
    out_min = lo;
    for (std::uint32_t i = 0; i < n; ++i) {
        const int q = static_cast<int>(std::lround((src[i] - lo) * inv));
        out_levels[i] = std::clamp(q, 0, levels);
    }
}

} // namespace

Status quantize_tensor(std::span<const float> src,
                       std::uint32_t rows,
                       std::uint32_t cols,
                       DType dtype,
                       std::uint32_t group,
                       QTensor& out) {
    if (!is_quantized(dtype)) {
        return {StatusCode::InvalidArgument,
                std::string("quantize_tensor called with unquantized dtype ") + to_string(dtype)};
    }
    if (src.size() != static_cast<std::size_t>(rows) * cols) {
        return {StatusCode::InvalidArgument, "quantize_tensor: shape does not match data"};
    }

    out.dtype = dtype;
    out.group = usable_group(cols, group);
    out.rows = rows;
    out.cols = cols;

    const auto g = out.group;
    const auto gb = group_bytes(dtype, g);
    const auto per_row = (static_cast<std::size_t>(cols) / g) * gb;
    out.data.assign(static_cast<std::size_t>(rows) * per_row, std::byte{0});

    std::vector<int> levels(g);

    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = src.data() + static_cast<std::size_t>(r) * cols;
        std::byte* dst = out.data.data() + static_cast<std::size_t>(r) * per_row;

        for (std::uint32_t c = 0; c < cols; c += g) {
            const float* blk = row + c;
            float scale = 0.0f;
            float minv = 0.0f;

            switch (dtype) {
            case DType::Q8_0: {
                quantize_group_symmetric(blk, g, 127, scale, levels.data());
                put_f32(dst, scale);
                for (std::uint32_t i = 0; i < g; ++i) {
                    dst[4 + i] = static_cast<std::byte>(static_cast<std::int8_t>(levels[i]));
                }
                break;
            }
            case DType::Q4_0: {
                quantize_group_symmetric(blk, g, 7, scale, levels.data());
                put_f32(dst, scale);
                for (std::uint32_t i = 0; i < g; i += 2) {
                    const auto a = static_cast<std::uint8_t>(levels[i] + 8) & 0x0F;
                    const auto b =
                        static_cast<std::uint8_t>((i + 1 < g ? levels[i + 1] : 0) + 8) & 0x0F;
                    dst[4 + i / 2] = static_cast<std::byte>(a | (b << 4));
                }
                break;
            }
            case DType::Q4_G: {
                quantize_group_asymmetric(blk, g, 15, scale, minv, levels.data());
                put_f32(dst, scale);
                put_f32(dst + 4, minv);
                for (std::uint32_t i = 0; i < g; i += 2) {
                    const auto a = static_cast<std::uint8_t>(levels[i]) & 0x0F;
                    const auto b = static_cast<std::uint8_t>(i + 1 < g ? levels[i + 1] : 0) & 0x0F;
                    dst[8 + i / 2] = static_cast<std::byte>(a | (b << 4));
                }
                break;
            }
            case DType::Q6_G: {
                quantize_group_symmetric(blk, g, 31, scale, levels.data());
                put_f32(dst, scale);
                // Four 6-bit values per three bytes.
                std::byte* q = dst + 4;
                for (std::uint32_t i = 0; i < g; i += 4) {
                    std::uint32_t v[4]{};
                    for (std::uint32_t k = 0; k < 4; ++k) {
                        const int lv = (i + k < g) ? levels[i + k] : 0;
                        v[k] = static_cast<std::uint32_t>(lv + 32) & 0x3F;
                    }
                    const std::uint32_t packed = v[0] | (v[1] << 6) | (v[2] << 12) | (v[3] << 18);
                    q[0] = static_cast<std::byte>(packed & 0xFF);
                    q[1] = static_cast<std::byte>((packed >> 8) & 0xFF);
                    q[2] = static_cast<std::byte>((packed >> 16) & 0xFF);
                    q += 3;
                }
                break;
            }
            default:
                return {StatusCode::Unsupported, "unreachable"};
            }
            dst += gb;
        }
    }
    return {};
}

namespace {

/// The decode loop, against a raw description rather than a container.
///
/// Split out so an owning QTensor and a borrowed WeightRef share one
/// implementation: two copies of a format decoder is two chances to update only
/// one of them.
Status dequantize_raw(CByteSpan data,
                      DType dtype,
                      std::uint32_t group,
                      std::uint32_t rows,
                      std::uint32_t cols,
                      std::span<float> out) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    if (out.size() < n) {
        return {StatusCode::InvalidArgument, "dequantize: destination too small"};
    }
    if (!is_quantized(dtype)) {
        return {StatusCode::Unsupported, "dequantize: unsupported quantized dtype"};
    }
    if (group == 0 || cols == 0 || cols % group != 0) {
        return {StatusCode::InvalidArgument, "dequantize: cols is not a multiple of the group"};
    }
    const auto g = group;
    const auto gb = group_bytes(dtype, g);
    if (gb == 0) {
        return {StatusCode::Unsupported, "dequantize: unsupported quantized dtype"};
    }
    const auto per_row = (static_cast<std::size_t>(cols) / g) * gb;
    if (data.size() < static_cast<std::size_t>(rows) * per_row) {
        return {StatusCode::InvalidArgument, "dequantize: source too small"};
    }

    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::byte* p = data.data() + static_cast<std::size_t>(r) * per_row;
        float* dst = out.data() + static_cast<std::size_t>(r) * cols;
        for (std::uint32_t c = 0; c < cols; c += g) {
            const float scale = get_f32(p);
            switch (dtype) {
            case DType::Q8_0:
                for (std::uint32_t i = 0; i < g; ++i) {
                    dst[c + i] = scale * static_cast<float>(static_cast<std::int8_t>(p[4 + i]));
                }
                break;
            case DType::Q4_0:
                for (std::uint32_t i = 0; i < g; ++i) {
                    const auto byte = static_cast<std::uint8_t>(p[4 + i / 2]);
                    const int nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                    dst[c + i] = scale * static_cast<float>(nib - 8);
                }
                break;
            case DType::Q4_G: {
                const float minv = get_f32(p + 4);
                for (std::uint32_t i = 0; i < g; ++i) {
                    const auto byte = static_cast<std::uint8_t>(p[8 + i / 2]);
                    const int nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                    dst[c + i] = minv + scale * static_cast<float>(nib);
                }
                break;
            }
            case DType::Q6_G: {
                const std::byte* q = p + 4;
                for (std::uint32_t i = 0; i < g; i += 4) {
                    const std::uint32_t packed =
                        static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
                        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
                        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16);
                    for (std::uint32_t k = 0; k < 4 && i + k < g; ++k) {
                        const int lv = static_cast<int>((packed >> (6 * k)) & 0x3F) - 32;
                        dst[c + i + k] = scale * static_cast<float>(lv);
                    }
                    q += 3;
                }
                break;
            }
            default:
                return {StatusCode::Unsupported,
                        std::string("dequantize: unsupported dtype ") + to_string(dtype)};
            }
            p += gb;
        }
    }
    return {};
}

} // namespace

Status dequantize_tensor(const QTensor& src, std::span<float> out) {
    return dequantize_raw(
        CByteSpan(src.data.data(), src.data.size()), src.dtype, src.group, src.rows, src.cols, out);
}

WeightRef row_block(const WeightRef& w, std::uint32_t first_row, std::uint32_t n_rows) noexcept {
    if (n_rows == 0 || first_row >= w.rows || n_rows > w.rows - first_row || w.cols == 0) return {};

    if (!w.quantized()) {
        const auto off = static_cast<std::size_t>(first_row) * w.cols;
        const auto len = static_cast<std::size_t>(n_rows) * w.cols;
        if (w.f32.size() < off + len) return {};
        return WeightRef::from_f32(w.f32.subspan(off, len), n_rows, w.cols);
    }

    const auto g = w.group;
    if (!is_quantized(w.dtype) || g == 0 || w.cols % g != 0) return {};
    const auto gb = group_bytes(w.dtype, g);
    if (gb == 0) return {};
    const auto per_row = (static_cast<std::size_t>(w.cols) / g) * gb;
    const auto off = static_cast<std::size_t>(first_row) * per_row;
    const auto len = static_cast<std::size_t>(n_rows) * per_row;
    if (w.bytes.size() < off + len) return {};
    return WeightRef::from_quantized_bytes(w.bytes.subspan(off, len), w.dtype, g, n_rows, w.cols);
}

Status dequantize(const WeightRef& w, std::span<float> out) {
    const std::size_t n = static_cast<std::size_t>(w.rows) * w.cols;
    if (w.rows == 0 || w.cols == 0 || w.empty()) {
        return {StatusCode::InvalidArgument, "dequantize: empty weight"};
    }
    if (out.size() < n) return {StatusCode::InvalidArgument, "dequantize: destination too small"};
    if (!w.quantized()) {
        if (w.f32.size() < n) return {StatusCode::InvalidArgument, "dequantize: source too small"};
        std::copy_n(w.f32.data(), n, out.data());
        return {};
    }
    return dequantize_raw(w.bytes, w.dtype, w.group, w.rows, w.cols, out);
}

} // namespace soma
