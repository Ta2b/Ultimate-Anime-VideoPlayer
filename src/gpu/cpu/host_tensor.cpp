#include "gpu/cpu/host_tensor.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "kernels/ref/half.h"
#include "kernels/spec/color.h"
#include "kernels/spec/resample.h"

namespace ajp::gpu::cpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

constexpr std::size_t kMaxElements = std::size_t{1} << 31U;

std::expected<kernels::Format, Refusal> format_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::Nv12:
      return kernels::Format::Nv12;
    case PixelFormat::P010:
      return kernels::Format::P010;
    case PixelFormat::P016:
      return kernels::Format::P016;
    case PixelFormat::Yuv420p:
      return kernels::Format::Yuv420p;
    case PixelFormat::Yuv420p10:
      return kernels::Format::Yuv420p10;
    case PixelFormat::Yuv444p16:
      return kernels::Format::Yuv444p16;
    case PixelFormat::Rgba16Unorm:
    case PixelFormat::Rgba8Unorm:
      break;
  }
  return refuse(RefusalCode::FormatUnsupported, "image décodée : format RGB, YUV attendu");
}

// rows lignes de samples échantillons de `bytes` octets, au pas du plan : chaque octet lu doit
// être dans le plan.
template <class Store>
std::expected<void, Refusal> read_rows(const FrameData& data, int plane, int samples, int rows, int bytes,
                                       Store store) {
  const std::span<const std::byte> memory = data.host_plane(plane);
  const std::size_t stride = data.host_stride(plane);
  const auto row_bytes = static_cast<std::size_t>(samples) * static_cast<std::size_t>(bytes);
  if (stride < row_bytes || memory.size() < stride * static_cast<std::size_t>(rows - 1) + row_bytes) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("plan {} : {} octets au pas de {}, {} lignes de {} attendues", plane,
                              memory.size(), stride, rows, row_bytes));
  }
  for (int y = 0; y < rows; ++y) {
    const std::byte* row = memory.data() + stride * static_cast<std::size_t>(y);
    for (int x = 0; x < samples; ++x) {
      std::uint16_t value = 0;
      if (bytes == 1) {
        value = static_cast<std::uint16_t>(row[x]);
      } else {
        std::memcpy(&value, row + 2 * static_cast<std::ptrdiff_t>(x), sizeof value);
      }
      store(x, y, value);
    }
  }
  return {};
}

}  // namespace

std::size_t element_size(TensorType type) {
  switch (type) {
    case TensorType::F32:
      return 4;
    case TensorType::F16:
      return 2;
    case TensorType::U8:
    case TensorType::I8:
      return 1;
  }
  return 0;
}

std::expected<std::size_t, Refusal> element_count(const Shape& shape) {
  std::size_t count = 1;
  for (const std::int64_t dim : shape.dims) {
    if (dim <= 0 || static_cast<std::size_t>(dim) > kMaxElements / count) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("forme [{}, {}, {}, {}] : dimension nulle ou total trop grand", shape.dims[0],
                                shape.dims[1], shape.dims[2], shape.dims[3]));
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

void HostMemory::add(std::byte* base, std::size_t size) {
  const std::scoped_lock lock(mutex_);
  ranges_.push_back({base, size});
}

void HostMemory::remove(const std::byte* base) {
  const std::scoped_lock lock(mutex_);
  std::erase_if(ranges_, [base](const Range& range) { return range.base == base; });
}

std::expected<std::span<std::byte>, Refusal> HostMemory::bytes(const Tensor& tensor) const {
  auto count = element_count(tensor.shape);
  if (!count) return std::unexpected(std::move(count.error()));
  const std::size_t size = *count * element_size(tensor.type);
  const std::uintptr_t begin = tensor.handle + tensor.offset;
  const std::scoped_lock lock(mutex_);
  for (const Range& range : ranges_) {
    const auto base = reinterpret_cast<std::uintptr_t>(range.base);
    if (tensor.handle != 0 && begin >= base && begin - base <= range.size &&
        size <= range.size - (begin - base)) {
      // Le tampon est à nous et vivant : le handle redevient son adresse.
      return std::span<std::byte>(range.base + (begin - base), size);
    }
  }
  return refuse(RefusalCode::InputMismatch, std::format("tenseur {:#x}+{} : hors des tampons du backend CPU",
                                                        tensor.handle, tensor.offset));
}

HostBuffer::HostBuffer(HostMemory& memory, TensorType type, const Shape& shape, std::size_t bytes)
    : memory_(memory), data_(bytes) {
  tensor_ = {.type = type, .shape = shape, .handle = reinterpret_cast<std::uintptr_t>(data_.data())};
  memory_.add(data_.data(), data_.size());
}

HostBuffer::~HostBuffer() {
  memory_.remove(data_.data());
}

std::expected<ImageShape, Refusal> image_shape(const Tensor& tensor, std::string_view what) {
  if (tensor.type != TensorType::F32 && tensor.type != TensorType::F16) {
    return refuse(RefusalCode::FormatUnsupported, std::format("{} : image en F32 ou F16", what));
  }
  const auto& d = tensor.shape.dims;
  const auto in_range = [](std::int64_t n, std::int64_t max) { return n >= 1 && n <= max; };
  if (d[0] != 1 || !in_range(d[1], 64) || !in_range(d[2], kernels::kMaxDimension) ||
      !in_range(d[3], kernels::kMaxDimension)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("{} : forme [{}, {}, {}, {}], image 1 × c × h × w attendue", what, d[0], d[1],
                              d[2], d[3]));
  }
  return ImageShape{
      .channels = static_cast<int>(d[1]), .height = static_cast<int>(d[2]), .width = static_cast<int>(d[3])};
}

std::expected<void, Refusal> expect_image(const Tensor& tensor, int channels, int height, int width,
                                          std::string_view what) {
  auto shape = image_shape(tensor, what);
  if (!shape) return std::unexpected(std::move(shape.error()));
  if (shape->channels != channels || shape->height != height || shape->width != width) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("{} : {} × {} × {}, attendu {} × {} × {}", what, shape->channels, shape->height,
                              shape->width, channels, height, width));
  }
  return {};
}

std::expected<std::vector<float>, Refusal> read_values(const HostMemory& memory, const Tensor& tensor) {
  if (tensor.type != TensorType::F32 && tensor.type != TensorType::F16) {
    return refuse(RefusalCode::FormatUnsupported, "lecture : tenseur F32 ou F16 attendu");
  }
  auto bytes = memory.bytes(tensor);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  std::vector<float> values(bytes->size() / element_size(tensor.type));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::byte* at = bytes->data() + i * element_size(tensor.type);
    if (tensor.type == TensorType::F32) {
      std::memcpy(&values[i], at, sizeof(float));
    } else {
      std::uint16_t half = 0;
      std::memcpy(&half, at, sizeof half);
      values[i] = kernels::ref::half_to_float(half);
    }
  }
  return values;
}

std::expected<void, Refusal> write_values(const HostMemory& memory, const Tensor& tensor,
                                          std::span<const float> values) {
  if (tensor.type != TensorType::F32 && tensor.type != TensorType::F16) {
    return refuse(RefusalCode::FormatUnsupported, "écriture : tenseur F32 ou F16 attendu");
  }
  auto bytes = memory.bytes(tensor);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  if (values.size() * element_size(tensor.type) != bytes->size()) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("écriture : {} valeurs pour un tenseur de {}", values.size(),
                              bytes->size() / element_size(tensor.type)));
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::byte* at = bytes->data() + i * element_size(tensor.type);
    if (tensor.type == TensorType::F32) {
      std::memcpy(at, &values[i], sizeof(float));
    } else {
      const std::uint16_t half = kernels::ref::float_to_half(values[i]);
      std::memcpy(at, &half, sizeof half);
    }
  }
  return {};
}

std::expected<std::vector<kernels::PlaneF>, Refusal> read_planes(const HostMemory& memory,
                                                                 const Tensor& tensor) {
  auto shape = image_shape(tensor, "lecture");
  if (!shape) return std::unexpected(std::move(shape.error()));
  auto values = read_values(memory, tensor);
  if (!values) return std::unexpected(std::move(values.error()));
  std::vector<kernels::PlaneF> planes(static_cast<std::size_t>(shape->channels),
                                      kernels::PlaneF(shape->width, shape->height));
  const auto area = static_cast<std::ptrdiff_t>(planes.front().area());
  for (std::size_t c = 0; c < planes.size(); ++c) {
    const auto first = values->begin() + static_cast<std::ptrdiff_t>(c) * area;
    std::copy(first, first + area, planes[c].data.begin());
  }
  return planes;
}

std::expected<void, Refusal> write_planes(const HostMemory& memory, const Tensor& tensor,
                                          std::span<const kernels::PlaneF> planes) {
  if (planes.empty()) return refuse(RefusalCode::ShapeInvalid, "écriture : aucun plan");
  if (auto ok = expect_image(tensor, static_cast<int>(planes.size()), planes.front().height,
                             planes.front().width, "écriture");
      !ok) {
    return ok;
  }
  std::vector<float> values;
  values.reserve(planes.size() * planes.front().area());
  for (const kernels::PlaneF& plane : planes) {
    if (plane.data.size() != planes.front().area()) {
      return refuse(RefusalCode::ShapeInvalid, "écriture : plans de tailles différentes");
    }
    values.insert(values.end(), plane.data.begin(), plane.data.end());
  }
  return write_values(memory, tensor, values);
}

std::expected<kernels::RgbImage, Refusal> read_rgb(const HostMemory& memory, const Tensor& tensor) {
  auto planes = read_planes(memory, tensor);
  if (!planes) return std::unexpected(std::move(planes.error()));
  if (planes->size() != 3)
    return refuse(RefusalCode::ShapeInvalid, std::format("{} canaux, RGB attendu", planes->size()));
  kernels::RgbImage image;
  for (std::size_t c = 0; c < 3; ++c) image.planes[c] = std::move((*planes)[c]);
  return image;
}

std::expected<void, Refusal> write_rgb(const HostMemory& memory, const Tensor& tensor,
                                       const kernels::RgbImage& image) {
  return write_planes(memory, tensor, image.planes);
}

std::expected<kernels::YuvImage, Refusal> read_frame(const DecodedFrame& frame) {
  if (!frame.data || frame.data->memory() != FrameMemory::Host) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("image {} : le backend CPU ne lit que la mémoire hôte", frame.id.value));
  }
  auto format = format_of(frame.format);
  if (!format) return std::unexpected(std::move(format.error()));
  if (!kernels::valid_frame_dimension(frame.width) || !kernels::valid_frame_dimension(frame.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("image {}×{} : dimensions paires dans [2, {}]",
                                                         frame.width, frame.height, kernels::kMaxDimension));
  }
  const kernels::FormatInfo info = kernels::info(*format);
  if (frame.data->plane_count() < (info.interleaved ? 2 : 3)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("image {} : {} plans", frame.id.value, frame.data->plane_count()));
  }
  kernels::YuvImage image(*format, frame.width, frame.height, 0, 0, 0);
  const FrameData& data = *frame.data;
  auto ok = read_rows(data, 0, frame.width, frame.height, info.bytes,
                      [&](int x, int y, std::uint16_t v) { image.y.at(x, y) = v; });
  const int cw = image.u.width;
  const int ch = image.u.height;
  if (ok && info.interleaved) {
    ok = read_rows(data, 1, 2 * cw, ch, info.bytes,
                   [&](int x, int y, std::uint16_t v) { (x % 2 == 0 ? image.u : image.v).at(x / 2, y) = v; });
  } else if (ok) {
    ok = read_rows(data, 1, cw, ch, info.bytes, [&](int x, int y, std::uint16_t v) { image.u.at(x, y) = v; });
    if (ok)
      ok = read_rows(data, 2, cw, ch, info.bytes,
                     [&](int x, int y, std::uint16_t v) { image.v.at(x, y) = v; });
  }
  if (!ok) return std::unexpected(std::move(ok.error()));
  return image;
}

std::expected<kernels::ref::ColorSpec, Refusal> color_of(const DecodedFrame& frame) {
  std::optional<kernels::Matrix> matrix;
  switch (frame.color.matrix) {
    case ColorMatrix::Bt601:
      matrix = kernels::Matrix::Bt601;
      break;
    case ColorMatrix::Bt709:
      matrix = kernels::Matrix::Bt709;
      break;
    case ColorMatrix::Bt2020Ncl:
      matrix = kernels::Matrix::Bt2020Ncl;
      break;
    case ColorMatrix::Rgb:
      return refuse(RefusalCode::FormatUnsupported, "matrice RGB sur une image YUV");
  }
  const kernels::Range range =
      frame.color.range == ColorRange::Full ? kernels::Range::Full : kernels::Range::Limited;
  kernels::Siting siting = kernels::Siting::Left;
  if (frame.color.siting == ChromaSiting::Center) siting = kernels::Siting::Center;
  if (frame.color.siting == ChromaSiting::TopLeft) siting = kernels::Siting::TopLeft;
  return kernels::ref::ColorSpec{.matrix = kernels::effective_matrix(matrix, frame.height),
                                 .range = kernels::effective_range(range),
                                 .siting = kernels::effective_siting(siting)};
}

Refusal refusal_of(const kernels::Error& error, std::string_view operation) {
  RefusalCode code = RefusalCode::FieldOutOfRange;
  switch (error.code) {
    case kernels::ErrorCode::DimensionOutOfRange:
    case kernels::ErrorCode::OddDimension:
    case kernels::ErrorCode::SizeMismatch:
      code = RefusalCode::ShapeInvalid;
      break;
    case kernels::ErrorCode::SampleOutOfRange:
      code = RefusalCode::InputMismatch;
      break;
    case kernels::ErrorCode::FilterDirection:
    case kernels::ErrorCode::ParameterOutOfRange:
    case kernels::ErrorCode::InvalidConstant:
      code = RefusalCode::FieldOutOfRange;
      break;
  }
  return {code, std::format("{} : {} : {}", operation, kernels::name(error.code), error.detail)};
}

}  // namespace ajp::gpu::cpu
