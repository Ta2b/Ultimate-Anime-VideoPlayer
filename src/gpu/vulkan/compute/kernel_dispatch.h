// Interne aux noyaux de KernelRunner : blocs push_constant de chaque shader (même ordre, mêmes
// types que leur bloc GLSL), constantes de spécialisation des formats YUV, tables de poids
// figées dans l'enregistrement, et lancement d'un shader nommé. Inclus par les .cpp seulement.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/kernel_types.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/recording.h"
#include "gpu/vulkan/compute/weights.h"
#include "kernels/spec/color.h"

namespace ajp::gpu::vk::detail {

// Groupes 16 × 16 de tous les shaders, sauf block_diff (un groupe par bloc).
inline constexpr std::uint32_t kGroup = 16;

constexpr std::uint32_t groups_for(std::uint32_t n, std::uint32_t group = kGroup) {
  return (n + group - 1U) / group;
}

template <class T>
std::span<const std::byte> bytes_of(const T& value) {
  return std::as_bytes(std::span<const T, 1>(&value, 1));
}

// ingest_chroma_h, output_chroma_v.
struct ChromaPush {
  std::uint32_t cb_offset;
  std::uint32_t cb_pitch;
  std::uint32_t cr_offset;
  std::uint32_t cr_pitch;
  std::uint32_t width;
  std::uint32_t rows;
  std::uint32_t out;  // ingest : colonnes de sortie ; sortie : lignes de chroma
  std::uint32_t taps;
};

// ingest_combine, output_matrix.
struct PlanesPush {
  std::uint32_t y_offset;
  std::uint32_t y_pitch;
  std::uint32_t cb_offset;
  std::uint32_t cb_pitch;
  std::uint32_t cr_offset;
  std::uint32_t cr_pitch;
  std::uint32_t width;
  std::uint32_t height;
};

// resample_h (src_width, dst_width, rows, taps) et resample_v (width, src_rows, dst_rows, taps).
struct ResamplePush {
  std::uint32_t a;
  std::uint32_t b;
  std::uint32_t c;
  std::uint32_t taps;
};

struct SizePush {
  std::uint32_t width;
  std::uint32_t height;
};

struct YuvWindowPush {
  std::uint32_t src_offset;
  std::uint32_t src_pitch;
  std::uint32_t src_width;
  std::uint32_t src_height;
  std::uint32_t dst_offset;
  std::uint32_t dst_pitch;
  std::uint32_t dst_width;
  std::uint32_t dst_height;
  std::int32_t shift_x;
  std::int32_t shift_y;
  std::uint32_t edge;
};

struct RgbWindowPush {
  std::uint32_t src_width;
  std::uint32_t src_height;
  std::uint32_t dst_width;
  std::uint32_t dst_height;
  std::int32_t shift_x;
  std::int32_t shift_y;
  float fill;
};

struct RifeConstantsPush {
  std::uint32_t width;
  std::uint32_t height;
  float t;
  std::uint32_t planes;  // 11, ou 7 (v2) : l'instant seul
};

struct WarpPush {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t flow_width;
  std::uint32_t flow_height;
  std::uint32_t area_x;
  std::uint32_t area_y;
  std::uint32_t area_width;
  std::uint32_t area_height;
  float t;
  std::uint32_t keep_poses;
  std::uint32_t has_previous;
};

struct TransferPush {
  std::uint32_t width;
  std::uint32_t height;
  float hlg_peak;
  float hlg_gamma;
  float bt1886_a;
  float bt1886_b;
  float bt1886_black;
  float bt1886_white;
};

struct RifeFramesPush {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t padded_width;
  std::uint32_t padded_height;
  std::uint32_t left;
  std::uint32_t top;
  float black;
};

struct RgbPairPush {
  std::uint32_t width;
  std::uint32_t height;
  float levels;
};

struct FlowMaxPush {
  std::uint32_t plane_width;
  std::uint32_t plane_height;
  std::uint32_t area_x;
  std::uint32_t area_y;
  std::uint32_t area_width;
  std::uint32_t area_height;
};

struct SplatPush {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
  std::uint32_t group;  // plans par groupe de l'axe z
  float s;
};

struct SplatNormalizePush {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

// scene_diff, block_diff.
struct PairPush {
  std::uint32_t a_offset;
  std::uint32_t a_pitch;
  std::uint32_t b_offset;
  std::uint32_t b_pitch;
  std::uint32_t width;
  std::uint32_t height;
};

static_assert(sizeof(ChromaPush) == 32 && sizeof(PlanesPush) == 32 && sizeof(ResamplePush) == 16);
static_assert(sizeof(YuvWindowPush) == 44 && sizeof(RgbWindowPush) == 28 && sizeof(RifeConstantsPush) == 16);
static_assert(sizeof(RifeFramesPush) == 28 && sizeof(RgbPairPush) == 12 && sizeof(FlowMaxPush) == 24);
static_assert(sizeof(WarpPush) == 44 && sizeof(TransferPush) == 32 && sizeof(PairPush) == 24);
struct FillPush {
  std::uint32_t count;
  float value;
};

static_assert(sizeof(SplatPush) == 20 && sizeof(SplatNormalizePush) == 12 && sizeof(FillPush) == 8);

// Constantes 0 à 5 de yuv.glsl : format, matrice, plage, octets par échantillon, chroma
// entrelacée, chroma 4:2:0, tirées de kernels::info.
std::array<std::uint32_t, 6> yuv_constants(kernels::Format format, kernels::Matrix matrix,
                                           kernels::Range range);

// Énumérations connues : InputMismatch sinon.
Done check_color(const ColorSpec& color);
Done check_filter(kernels::Filter filter);

// Une table de poids figée dans l'enregistrement.
struct WeightViews {
  BufferView start;
  BufferView weights;
  std::uint32_t taps = 0;
};
std::expected<WeightViews, Refusal> record_weights(Recorder& recorder, const WeightTable& table);

// Lance le shader nommé, dans sa variante « _f16 » si variant vaut F16.
Done run(PipelineCache& pipelines, Recorder& recorder, std::string_view shader, Precision variant,
         std::span<const std::uint32_t> constants, std::span<const BufferView> buffers,
         std::span<const VkImageView> images, std::span<const std::byte> push,
         std::array<std::uint32_t, 3> groups, DispatchOrder order = DispatchOrder::AfterPrevious);

}  // namespace ajp::gpu::vk::detail
