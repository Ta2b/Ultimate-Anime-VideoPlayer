#include "gpu/cuda/trt/bench_input.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>

#include "kernels/ref/half.h"
#include "kernels/ref/rife.h"

namespace ajp::gpu::trt::bench {

std::vector<std::byte> noise_input(DataType type, std::size_t bytes) {
  std::vector<std::byte> data(bytes);
  const std::size_t width = element_bytes(type);
  for (std::size_t k = 0; width > 0 && k + width <= bytes; k += width) {
    const auto mix = static_cast<std::uint32_t>((k / width) * 2654435761U);
    if (type == DataType::F16) {
      const auto bits = static_cast<std::uint16_t>(0x3800U | (mix >> 22U));
      std::memcpy(&data[k], &bits, sizeof bits);
    } else if (type == DataType::F32) {
      const std::uint32_t bits = 0x3F000000U | (mix >> 9U);
      std::memcpy(&data[k], &bits, sizeof bits);
    }
  }
  return data;
}

std::expected<std::vector<std::byte>, std::string> rife_input(DataType type, const Dims& dims) {
  if (type != DataType::F16 || dims.size() != 4 || dims[0] != 1 || dims[1] != kernels::kRifeInputChannels) {
    return std::unexpected(std::format("entrée RIFE F16 1x{}xHxW attendue, pas {} {}",
                                       kernels::kRifeInputChannels, name(type), dims_text(dims)));
  }
  const auto height = static_cast<int>(dims[2]);
  const auto width = static_cast<int>(dims[3]);
  kernels::RgbImage a(width, height);
  kernels::RgbImage b(width, height);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const auto value = [&](int shift) {
          const double u = static_cast<double>(x + shift + (7 * c)) / 37.0;
          return static_cast<float>(0.5 + (0.25 * std::sin(u) * std::cos(static_cast<double>(y) / 23.0)));
        };
        a.planes[static_cast<std::size_t>(c)].data[a.planes[0].index(x, y)] = value(0);
        b.planes[static_cast<std::size_t>(c)].data[b.planes[0].index(x, y)] = value(3);
      }
    }
  }
  auto planes = kernels::ref::rife_input(a, b, 0.5F);
  if (!planes) return std::unexpected(std::format("rife_input : {}", kernels::name(planes.error().code)));
  std::vector<std::byte> data;
  data.reserve(static_cast<std::size_t>(kernels::kRifeInputChannels) * a.planes[0].area() * 2);
  for (const kernels::PlaneF& plane : *planes) {
    for (const float v : plane.data) {
      const std::uint16_t bits = kernels::ref::float_to_half(v);
      data.push_back(static_cast<std::byte>(bits & 0xFFU));
      data.push_back(static_cast<std::byte>(bits >> 8U));
    }
  }
  return data;
}

}  // namespace ajp::gpu::trt::bench
