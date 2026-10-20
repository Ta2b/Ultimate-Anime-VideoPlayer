#include "backend_rig.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/buffer.h"
#include "kernels/ref/half.h"
#include "net_rig.h"

namespace ajp::gpu::vk::test {
namespace {

// SplitMix64 : la même suite partout.
std::uint64_t mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

class HostFrame final : public FrameData {
 public:
  HostFrame(std::vector<std::vector<std::byte>> planes, std::vector<std::size_t> strides)
      : planes_(std::move(planes)), strides_(std::move(strides)) {}
  FrameMemory memory() const override { return FrameMemory::Host; }
  int plane_count() const override { return static_cast<int>(planes_.size()); }
  std::span<const std::byte> host_plane(int plane) const override {
    return planes_[static_cast<std::size_t>(plane)];
  }
  std::size_t host_stride(int plane) const override { return strides_[static_cast<std::size_t>(plane)]; }
  std::uintptr_t native_plane(int /*plane*/) const override { return 0; }
  std::size_t native_pitch(int /*plane*/) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }

 private:
  std::vector<std::vector<std::byte>> planes_;
  std::vector<std::size_t> strides_;
};

void put_code(std::vector<std::byte>& out, std::size_t at, std::uint16_t code, int bytes) {
  out[at] = static_cast<std::byte>(code & 0xFFU);
  if (bytes == 2) out[at + 1] = static_cast<std::byte>(code >> 8U);
}

void copy_buffers(VulkanBackend& backend, VkBuffer from, VkDeviceSize from_offset, VkBuffer to,
                  VkDeviceSize to_offset, VkDeviceSize size) {
  VkContext& context = backend.context();
  require_ok(context.wait_on_host(backend.queue().last()));
  require_ok(context.run_once([&](VkCommandBuffer cb) {
    const VkBufferCopy region{.srcOffset = from_offset, .dstOffset = to_offset, .size = size};
    context.fn().vkCmdCopyBuffer(cb, from, to, 1, &region);
  }));
}

}  // namespace

TempDir::TempDir(std::string_view name) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  path_ = std::filesystem::temp_directory_path() / std::format("ajp-{}-{}", name, stamp);
  std::filesystem::create_directories(path_);
}

TempDir::~TempDir() {
  std::error_code error;
  std::filesystem::remove_all(path_, error);
}

std::unique_ptr<VulkanBackend> make_backend(const std::filesystem::path& cache_dir, BackendOptions options) {
  auto backend = VulkanBackend::create(BackendInit{.cache_dir = cache_dir}, options);
  if (!backend) {
    const RefusalCode code = backend.error().code;
    if (code == RefusalCode::LibraryUnavailable || code == RefusalCode::DeviceUnavailable) {
      SKIP("aucun Vulkan : " << describe(backend.error()));
    }
    FAIL(describe(backend.error()));
  }
  return std::move(*backend);
}

void write_tensor(VulkanBackend& backend, const Tensor& tensor, std::span<const std::byte> bytes) {
  Buffer stage =
      require_value(Buffer::create(backend.context(), backend.memory(), bytes.size(), MemoryUse::Upload));
  std::memcpy(stage.mapped(), bytes.data(), bytes.size());
  copy_buffers(backend, stage.handle(), 0, detail::to_buffer(tensor.handle), tensor.offset, bytes.size());
}

std::vector<std::byte> read_tensor(VulkanBackend& backend, const Tensor& tensor, std::size_t bytes) {
  Buffer stage =
      require_value(Buffer::create(backend.context(), backend.memory(), bytes, MemoryUse::Readback));
  copy_buffers(backend, detail::to_buffer(tensor.handle), tensor.offset, stage.handle(), 0, bytes);
  std::vector<std::byte> out(bytes);
  std::memcpy(out.data(), stage.mapped(), bytes);
  return out;
}

std::vector<std::byte> planes_bytes(std::span<const kernels::PlaneF> planes, TensorType type) {
  const std::size_t element = type == TensorType::F16 ? 2 : 4;
  std::vector<std::byte> out;
  for (const kernels::PlaneF& p : planes) {
    for (const float v : p.data) {
      std::array<std::byte, 4> b{};
      if (type == TensorType::F16) {
        const std::uint16_t h = kernels::ref::float_to_half(v);
        std::memcpy(b.data(), &h, 2);
      } else {
        std::memcpy(b.data(), &v, 4);
      }
      out.insert(out.end(), b.begin(), b.begin() + static_cast<std::ptrdiff_t>(element));
    }
  }
  return out;
}

std::vector<std::byte> rgb_bytes(const kernels::RgbImage& image, TensorType type) {
  return planes_bytes(image.planes, type);
}

kernels::RgbImage rgb_from(std::span<const std::byte> bytes, int width, int height, TensorType type) {
  kernels::RgbImage out(width, height);
  std::vector<kernels::PlaneF> planes = planes_from(bytes, width, height, 3, type);
  std::ranges::move(planes, out.planes.begin());
  return out;
}

std::vector<kernels::PlaneF> planes_from(std::span<const std::byte> bytes, int width, int height, int count,
                                         TensorType type) {
  std::vector<kernels::PlaneF> out(static_cast<std::size_t>(count), kernels::PlaneF(width, height));
  const std::size_t element = type == TensorType::F16 ? 2 : 4;
  std::size_t at = 0;
  for (kernels::PlaneF& p : out) {
    for (float& v : p.data) {
      if (type == TensorType::F16) {
        std::uint16_t h = 0;
        std::memcpy(&h, bytes.data() + at, 2);
        v = kernels::ref::half_to_float(h);
      } else {
        std::memcpy(&v, bytes.data() + at, 4);
      }
      at += element;
    }
  }
  return out;
}

std::unique_ptr<TensorBuffer> rgb_tensor(VulkanBackend& backend, int width, int height, TensorType type,
                                         int planes) {
  return require_value(backend.alloc_tensor(type, Shape{{1, planes, height, width}}));
}

kernels::RgbImage stored(const kernels::RgbImage& image, TensorType type) {
  return type == TensorType::F16 ? kernels::ref::round_to_half(image) : image;
}

kernels::YuvImage synthetic_yuv(kernels::Format format, int width, int height, std::uint32_t seed) {
  kernels::YuvImage image(format, width, height, 0, 0, 0);
  const int max = static_cast<int>(kernels::max_code(format));
  const int step = format == kernels::Format::P010 ? 64 : 1;
  const auto fill = [&](kernels::PlaneU16& p, std::uint32_t channel) {
    for (int y = 0; y < p.height; ++y) {
      for (int x = 0; x < p.width; ++x) {
        const std::uint64_t r = mix((std::uint64_t{seed} << 40U) ^ (std::uint64_t{channel} << 32U) ^
                                    (static_cast<std::uint64_t>(y) << 16U) ^ static_cast<std::uint64_t>(x));
        // Un quart des pixels en rampe, quelques-uns aux extrêmes, le reste en bruit.
        int code = static_cast<int>(r % static_cast<std::uint64_t>(max / step + 1)) * step;
        if (x % 4 == 0) code = (x * max / std::max(1, p.width - 1)) / step * step;
        if ((x + y) % 17 == 0) code = (r & 1U) != 0 ? max : 0;
        p.at(x, y) = static_cast<std::uint16_t>(code);
      }
    }
  };
  fill(image.y, 0);
  fill(image.u, 1);
  fill(image.v, 2);
  return image;
}

kernels::RgbImage synthetic_rgb(int width, int height, std::uint32_t seed, float lo, float hi) {
  kernels::RgbImage image(width, height);
  for (std::size_t c = 0; c < 3; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::uint64_t r = mix((std::uint64_t{seed} << 40U) ^ (std::uint64_t{c} << 32U) ^
                                    (static_cast<std::uint64_t>(y) << 16U) ^ static_cast<std::uint64_t>(x));
        image.planes[c].at(x, y) =
            lo + ((hi - lo) * static_cast<float>(r >> 40U) / static_cast<float>(1U << 24U));
      }
    }
  }
  return image;
}

PixelFormat pixel_format(kernels::Format format) {
  switch (format) {
    case kernels::Format::Nv12:
      return PixelFormat::Nv12;
    case kernels::Format::P010:
      return PixelFormat::P010;
    case kernels::Format::P016:
      return PixelFormat::P016;
    case kernels::Format::Yuv420p:
      return PixelFormat::Yuv420p;
    case kernels::Format::Yuv420p10:
      return PixelFormat::Yuv420p10;
    case kernels::Format::Yuv444p16:
      return PixelFormat::Yuv444p16;
  }
  return PixelFormat::Nv12;
}

DecodedFrame host_frame(const kernels::YuvImage& image, const ColorInfo& color, std::uint32_t id, int pad) {
  const kernels::FormatInfo info = kernels::info(image.format);
  const int bytes = info.bytes;
  std::vector<std::vector<std::byte>> planes;
  std::vector<std::size_t> strides;
  const auto add_plane = [&](int width, int height, int samples_per_texel, auto&& code_at) {
    const std::size_t stride = static_cast<std::size_t>(width * samples_per_texel * bytes + pad);
    std::vector<std::byte> data(stride * static_cast<std::size_t>(height), std::byte{0xCD});
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int s = 0; s < samples_per_texel; ++s) {
          const std::size_t at = (static_cast<std::size_t>(y) * stride) +
                                 static_cast<std::size_t>(((x * samples_per_texel) + s) * bytes);
          put_code(data, at, code_at(x, y, s), bytes);
        }
      }
    }
    planes.push_back(std::move(data));
    strides.push_back(stride);
  };
  add_plane(image.y.width, image.y.height, 1, [&](int x, int y, int) { return image.y.at(x, y); });
  if (info.interleaved) {
    add_plane(image.u.width, image.u.height, 2,
              [&](int x, int y, int s) { return s == 0 ? image.u.at(x, y) : image.v.at(x, y); });
  } else {
    add_plane(image.u.width, image.u.height, 1, [&](int x, int y, int) { return image.u.at(x, y); });
    add_plane(image.v.width, image.v.height, 1, [&](int x, int y, int) { return image.v.at(x, y); });
  }
  return DecodedFrame{.id = SourceId{id},
                      .width = image.width(),
                      .height = image.height(),
                      .format = pixel_format(image.format),
                      .color = color,
                      .pts_us = 0,
                      .data = std::make_shared<HostFrame>(std::move(planes), std::move(strides))};
}

float max_abs_diff(const kernels::RgbImage& a, const kernels::RgbImage& b) {
  float worst = 0.0F;
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t k = 0; k < a.planes[c].data.size(); ++k) {
      const float d = std::fabs(a.planes[c].data[k] - b.planes[c].data[k]);
      worst = std::isnan(d) ? INFINITY : std::max(worst, d);
    }
  }
  return worst;
}

ModelSpec write_model(const std::filesystem::path& dir, std::string_view name,
                      std::span<const std::byte> bytes, TensorType precision) {
  const std::filesystem::path path = dir / std::format("{}.onnx", name);
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(out.good());
  return ModelSpec{
      .id = std::string(name), .onnx_path = path, .sha256 = std::string(64, '0'), .precision = precision};
}

}  // namespace ajp::gpu::vk::test
