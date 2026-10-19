#include "gpu/vulkan/compute/kernel_types.h"

#include <algorithm>
#include <format>

namespace ajp::gpu::vk {
namespace {

constexpr VkDeviceSize kWordAddressable = VkDeviceSize{1} << 32U;

std::uint32_t align4(std::uint32_t bytes) {
  return (bytes + 3U) & ~3U;
}

bool known(kernels::Format format) {
  return static_cast<std::uint8_t>(format) <= static_cast<std::uint8_t>(kernels::Format::Yuv444p16);
}

Done check_dimensions(kernels::Format format, std::uint32_t width, std::uint32_t height,
                      std::string_view what) {
  if (!known(format)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : format {} inconnu", what, static_cast<int>(format)));
  }
  const auto w = static_cast<int>(std::min(width, static_cast<std::uint32_t>(kernels::kMaxDimension) + 1U));
  const auto h = static_cast<int>(std::min(height, static_cast<std::uint32_t>(kernels::kMaxDimension) + 1U));
  const bool subsampled = kernels::info(format).subsampled;
  const bool valid = subsampled ? kernels::valid_frame_dimension(w) && kernels::valid_frame_dimension(h)
                                : kernels::valid_plane_dimension(w) && kernels::valid_plane_dimension(h);
  if (!valid) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : {}x{} hors de [{}, {}]{}", what, width, height, subsampled ? 2 : 1,
                              kernels::kMaxDimension, subsampled ? ", pair en 4:2:0" : ""));
  }
  return {};
}

// Un plan de rows lignes de row_bytes octets tient dans la vue, aligné sur 4 octets.
Done check_plane(const PlaneLayout& plane, std::uint32_t row_bytes, std::uint32_t rows,
                 VkDeviceSize view_size, std::string_view what, std::string_view name) {
  if (plane.offset % 4 != 0 || plane.pitch % 4 != 0) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : plan {} à l'offset {}, pas {} : multiples de 4 attendus", what, name,
                              plane.offset, plane.pitch));
  }
  if (plane.pitch < row_bytes) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : plan {} au pas {} pour des lignes de {} o",
                                                          what, name, plane.pitch, row_bytes));
  }
  const VkDeviceSize end =
      VkDeviceSize{plane.offset} + VkDeviceSize{plane.pitch} * (rows - 1U) + align4(row_bytes);
  if (end > view_size) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : plan {} jusqu'à l'octet {}, vue de {} o", what, name, end, view_size));
  }
  return {};
}

// Octets [begin, end) qu'un plan occupe dans la vue, bourrage des lignes compris.
struct Extent {
  VkDeviceSize begin = 0;
  VkDeviceSize end = 0;
  bool overlaps(const Extent& other) const { return begin < other.end && other.begin < end; }
};

Extent extent(const PlaneLayout& plane, std::uint32_t row_bytes, std::uint32_t rows) {
  return {.begin = plane.offset,
          .end = VkDeviceSize{plane.offset} + VkDeviceSize{plane.pitch} * (rows - 1U) + align4(row_bytes)};
}

}  // namespace

std::uint32_t chroma_width(kernels::Format format, std::uint32_t width) {
  return kernels::info(format).subsampled ? width / 2U : width;
}

std::uint32_t chroma_height(kernels::Format format, std::uint32_t height) {
  return kernels::info(format).subsampled ? height / 2U : height;
}

std::expected<YuvLayout, Refusal> tight_layout(kernels::Format format, std::uint32_t width,
                                               std::uint32_t height) {
  if (auto ok = check_dimensions(format, width, height, "disposition YUV"); !ok)
    return std::unexpected(ok.error());
  const kernels::FormatInfo info = kernels::info(format);
  const auto bytes = static_cast<std::uint32_t>(info.bytes);
  const std::uint32_t cw = chroma_width(format, width);
  const std::uint32_t ch = chroma_height(format, height);
  YuvLayout layout;
  layout.luma = {.offset = 0, .pitch = align4(width * bytes)};
  const std::uint32_t luma_bytes = layout.luma.pitch * height;
  if (info.interleaved) {
    layout.cb = {.offset = luma_bytes, .pitch = align4(2U * cw * bytes)};
    layout.cr = layout.cb;
    layout.bytes = luma_bytes + layout.cb.pitch * ch;
  } else {
    const std::uint32_t pitch = align4(cw * bytes);
    layout.cb = {.offset = luma_bytes, .pitch = pitch};
    layout.cr = {.offset = luma_bytes + pitch * ch, .pitch = pitch};
    layout.bytes = luma_bytes + 2U * pitch * ch;
  }
  return layout;
}

Done check_yuv(const YuvView& view, std::string_view what) {
  if (auto ok = check_dimensions(view.format, view.width, view.height, what); !ok) return ok;
  if (view.buffer.buffer == VK_NULL_HANDLE || view.buffer.size == 0 || view.buffer.size >= kWordAddressable) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : vue de {} o, de 1 o à 4 Gio attendue", what, view.buffer.size));
  }
  const kernels::FormatInfo info = kernels::info(view.format);
  const auto bytes = static_cast<std::uint32_t>(info.bytes);
  const std::uint32_t cw = chroma_width(view.format, view.width);
  const std::uint32_t ch = chroma_height(view.format, view.height);
  const YuvLayout& l = view.layout;
  if (auto ok = check_plane(l.luma, view.width * bytes, view.height, view.buffer.size, what, "Y"); !ok)
    return ok;
  const std::uint32_t chroma_row = (info.interleaved ? 2U : 1U) * cw * bytes;
  if (auto ok = check_plane(l.cb, chroma_row, ch, view.buffer.size, what, info.interleaved ? "CbCr" : "Cb");
      !ok)
    return ok;
  if (!info.interleaved) {
    if (auto ok = check_plane(l.cr, chroma_row, ch, view.buffer.size, what, "Cr"); !ok) return ok;
  }
  // Des plans qui se chevauchent : la sortie écrirait l'un sur l'autre, l'entrée serait ambiguë.
  const Extent y = extent(l.luma, view.width * bytes, view.height);
  const Extent cb = extent(l.cb, chroma_row, ch);
  const Extent cr = extent(l.cr, chroma_row, ch);
  const bool overlap = y.overlaps(cb) || (!info.interleaved && (y.overlaps(cr) || cb.overlaps(cr)));
  if (overlap) {
    return refuse(
        RefusalCode::InputMismatch,
        std::format("{} : plans Y ({}, pas {}), Cb ({}, pas {}) et Cr ({}, pas {}) qui se chevauchent", what,
                    l.luma.offset, l.luma.pitch, l.cb.offset, l.cb.pitch, l.cr.offset, l.cr.pitch));
  }
  return {};
}

Done check_tensor(const TensorView& tensor, std::uint32_t planes, const Features& enabled,
                  std::string_view what) {
  if (tensor.planes != planes) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : {} plans, {} attendus", what, tensor.planes, planes));
  }
  const auto limit = static_cast<std::uint32_t>(kernels::kMaxDimension);
  if (tensor.width == 0 || tensor.height == 0 || tensor.width > limit || tensor.height > limit) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : {}x{} hors de [1, {}]", what, tensor.width, tensor.height, limit));
  }
  if (tensor.precision != Precision::F32 && tensor.precision != Precision::F16) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : précision inconnue", what));
  }
  if (tensor.precision == Precision::F16 && !enabled.storage_buffer_16bit) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} : tenseur float16 sans storageBuffer16BitAccess", what));
  }
  const VkDeviceSize needed = tensor_bytes(tensor.width, tensor.height, tensor.planes, tensor.precision);
  if (tensor.buffer.buffer == VK_NULL_HANDLE || tensor.buffer.size < needed ||
      tensor.buffer.size >= kWordAddressable) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : vue de {} o pour un tenseur de {} o", what, tensor.buffer.size, needed));
  }
  return {};
}

Done check_same_size(std::uint32_t w0, std::uint32_t h0, std::uint32_t w1, std::uint32_t h1,
                     std::string_view what) {
  if (w0 != w1 || h0 != h1) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : {}x{} contre {}x{}", what, w0, h0, w1, h1));
  }
  return {};
}

}  // namespace ajp::gpu::vk
