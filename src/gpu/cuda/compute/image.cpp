#include "gpu/cuda/compute/image.h"

#include <cstdint>
#include <format>

namespace ajp::gpu::cuda {
namespace {

Result<> check_plane(const Plane& p, std::size_t row_bytes, std::size_t sample, int index,
                     std::string_view what) {
  if (p.ptr == 0 || p.ptr % sample != 0 || p.pitch % sample != 0 || p.pitch < row_bytes) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : plan {} d'adresse {:#x} et de pas {} pour des lignes de {} octets", what,
                              index, p.ptr, p.pitch, row_bytes));
  }
  return {};
}

}  // namespace

std::size_t elem_size(Elem elem) {
  return elem == Elem::F16 ? 2 : 4;
}

dev::Sample sample_of(kernels::Format format) {
  return kernels::info(format).bytes == 1 ? dev::Sample::U8 : dev::Sample::U16;
}

Tensor channels(const Tensor& tensor, int first, int count) {
  const std::size_t plane = static_cast<std::size_t>(tensor.width) * static_cast<std::size_t>(tensor.height);
  Tensor out = tensor;
  out.ptr = tensor.ptr + static_cast<std::size_t>(first) * plane * elem_size(tensor.elem);
  out.channels = count;
  return out;
}

Result<> check_frame(int width, int height, std::string_view what) {
  for (const int n : {width, height}) {
    if (!kernels::valid_frame_dimension(n)) {
      return refuse(RefusalCode::ShapeInvalid, std::format("{} : {}x{} hors de [2, {}] ou impair", what,
                                                           width, height, kernels::kMaxDimension));
    }
  }
  return {};
}

Result<> check_tensor(const Tensor& tensor, int width, int height, int channels, std::string_view what) {
  if (tensor.width != width || tensor.height != height || tensor.channels != channels) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("{} : tenseur {}x{}x{}, {}x{}x{} attendu", what, tensor.channels, tensor.height,
                              tensor.width, channels, height, width));
  }
  if (tensor.ptr == 0 || tensor.ptr % elem_size(tensor.elem) != 0) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : adresse {:#x} nulle ou mal alignée", what, tensor.ptr));
  }
  return {};
}

Result<> check_yuv(const YuvPlanes& image, kernels::Format format, int width, int height,
                   std::string_view what) {
  if (image.format != format || image.width != width || image.height != height) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("{} : image {}x{} de format {}, {}x{} de format {} attendu", what, image.width,
                    image.height, static_cast<int>(image.format), width, height, static_cast<int>(format)));
  }
  const kernels::FormatInfo info = kernels::info(format);
  const auto sample = static_cast<std::size_t>(info.bytes);
  const auto luma_row = static_cast<std::size_t>(width) * sample;
  if (auto ok = check_plane(image.planes[0], luma_row, sample, 0, what); !ok) return ok;
  const auto chroma_width = static_cast<std::size_t>(info.subsampled ? width / 2 : width);
  if (info.interleaved) return check_plane(image.planes[1], 2 * chroma_width * sample, sample, 1, what);
  for (const int i : {1, 2}) {
    if (auto ok =
            check_plane(image.planes[static_cast<std::size_t>(i)], chroma_width * sample, sample, i, what);
        !ok) {
      return ok;
    }
  }
  return {};
}

Result<> check_luma(const YuvPlanes& image, kernels::Format format, int width, int height,
                    std::string_view what) {
  if (image.format != format || image.width != width || image.height != height) {
    return refuse(RefusalCode::ShapeInvalid, std::format("{} : luma {}x{}, {}x{} attendu", what, image.width,
                                                         image.height, width, height));
  }
  const auto sample = static_cast<std::size_t>(kernels::info(format).bytes);
  return check_plane(image.planes[0], static_cast<std::size_t>(width) * sample, sample, 0, what);
}

}  // namespace ajp::gpu::cuda
