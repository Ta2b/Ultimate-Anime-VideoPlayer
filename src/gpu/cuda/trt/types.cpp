#include "gpu/cuda/trt/types.h"

#include <format>
#include <limits>

namespace ajp::gpu::trt {

std::string_view name(DataType type) {
  switch (type) {
    case DataType::F32:
      return "F32";
    case DataType::F16:
      return "F16";
    case DataType::BF16:
      return "BF16";
    case DataType::I8:
      return "I8";
    case DataType::U8:
      return "U8";
    case DataType::I32:
      return "I32";
    case DataType::I64:
      return "I64";
    case DataType::Bool:
      return "Bool";
    case DataType::Other:
      break;
  }
  return "Other";
}

std::size_t element_bytes(DataType type) {
  switch (type) {
    case DataType::F32:
    case DataType::I32:
      return 4;
    case DataType::F16:
    case DataType::BF16:
      return 2;
    case DataType::I8:
    case DataType::U8:
    case DataType::Bool:
      return 1;
    case DataType::I64:
      return 8;
    case DataType::Other:
      break;
  }
  return 0;
}

std::string dims_text(std::span<const std::int64_t> dims) {
  std::string text;
  for (const std::int64_t d : dims) {
    if (!text.empty()) text += 'x';
    text += std::to_string(d);
  }
  return text;
}

std::expected<std::size_t, Refusal> byte_count(DataType type, std::span<const std::int64_t> dims) {
  std::size_t bytes = element_bytes(type);
  if (bytes == 0) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   std::format("type de tenseur {} sans taille connue", name(type))});
  }
  for (const std::int64_t d : dims) {
    const auto extent = static_cast<std::size_t>(d);
    if (d < 0 || (extent != 0 && bytes > std::numeric_limits<std::size_t>::max() / extent)) {
      return std::unexpected(
          Refusal{RefusalCode::ShapeInvalid,
                  std::format("forme {} : taille du tenseur impossible", dims_text(dims))});
    }
    bytes *= extent;
  }
  return bytes;
}

}  // namespace ajp::gpu::trt
