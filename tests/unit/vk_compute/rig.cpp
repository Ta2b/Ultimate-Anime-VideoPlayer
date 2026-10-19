#include "rig.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "kernels/ref/half.h"

namespace ajp::gpu::vk::test {
namespace {

// SplitMix64 : la même suite sur toutes les plateformes, contrairement aux distributions de <random>.
std::uint64_t mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

std::uint32_t noise(std::uint32_t seed, int x, int y, int channel) {
  const std::uint64_t key =
      (std::uint64_t{seed} << 40U) ^ (std::uint64_t(static_cast<std::uint32_t>(channel)) << 32U) ^
      (std::uint64_t(static_cast<std::uint32_t>(y)) << 16U) ^ static_cast<std::uint32_t>(x);
  return static_cast<std::uint32_t>(mix(key) >> 32U);
}

void put_sample(std::vector<std::byte>& out, std::size_t byte, std::uint16_t code, int bytes) {
  out[byte] = static_cast<std::byte>(code & 0xFFU);
  if (bytes == 2) out[byte + 1] = static_cast<std::byte>(code >> 8U);
}

std::uint16_t get_sample(std::span<const std::byte> in, std::size_t byte, int bytes) {
  const auto low = std::to_integer<std::uint16_t>(in[byte]);
  if (bytes == 1) return low;
  return static_cast<std::uint16_t>(low | (std::to_integer<std::uint16_t>(in[byte + 1]) << 8U));
}

// Un plan de codes vers les octets : stride échantillons entre deux pixels, first le premier.
void pack_plane(std::vector<std::byte>& out, const kernels::PlaneU16& plane, const PlaneLayout& layout,
                int bytes, int stride, int first) {
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      const std::size_t byte = layout.offset + static_cast<std::size_t>(y) * layout.pitch +
                               static_cast<std::size_t>((x * stride + first) * bytes);
      put_sample(out, byte, plane.at(x, y), bytes);
    }
  }
}

void unpack_plane(std::span<const std::byte> in, kernels::PlaneU16& plane, const PlaneLayout& layout,
                  int bytes, int stride, int first) {
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      const std::size_t byte = layout.offset + static_cast<std::size_t>(y) * layout.pitch +
                               static_cast<std::size_t>((x * stride + first) * bytes);
      plane.at(x, y) = get_sample(in, byte, bytes);
    }
  }
}

}  // namespace

std::string describe(const Refusal& refusal) {
  return std::format("{} : {}", name(refusal.code), refusal.detail);
}

Rig make_rig(const CreateOptions& options) {
  // La couche de validation, quand elle est installée, surveille chaque test : run() vérifie
  // qu'elle n'a signalé aucune erreur. Sans elle, l'appareil est créé tel quel.
  CreateOptions validated = options;
  validated.validation = true;
  auto context = VkContext::create(validated);
  if (!context && context.error().code == RefusalCode::FeatureUnavailable && !options.validation) {
    context = VkContext::create(options);
  }
  if (!context) {
    const RefusalCode code = context.error().code;
    if (code == RefusalCode::LibraryUnavailable || code == RefusalCode::DeviceUnavailable) {
      SKIP("aucun Vulkan : " << describe(context.error()));
    }
    FAIL(describe(context.error()));
  }
  Rig rig;
  rig.context = std::move(*context);
  rig.memory = require_value(MemoryAllocator::create(*rig.context));
  rig.pipelines = require_value(PipelineCache::create(*rig.context));
  rig.runner.emplace(*rig.pipelines);
  return rig;
}

void require_f16(const Rig& rig) {
  if (!rig.context->caps().enabled.storage_buffer_16bit) {
    SKIP(rig.context->caps().device_name
         << " : pas de storageBuffer16BitAccess, variantes float16 non exercées");
  }
}

std::unique_ptr<Recording> record(Rig& rig, const std::function<Done(Recorder&)>& body) {
  Recorder recorder = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_ok(body(recorder));
  return require_value(recorder.finish());
}

void run(Rig& rig, Recording& recording) {
  require_ok(rig.context->submit(recording));
  require_ok(rig.context->wait_on_host(recording.last_run()));
  require_no_validation_errors(rig);
}

void require_no_validation_errors(const Rig& rig) {
  CHECK(rig.context->validation_errors() == 0);
}

void record_and_run(Rig& rig, const std::function<Done(Recorder&)>& body) {
  const std::unique_ptr<Recording> recording = record(rig, body);
  run(rig, *recording);
}

Buffer make_buffer(Rig& rig, VkDeviceSize bytes, MemoryUse use) {
  return require_value(Buffer::create(*rig.context, *rig.memory, bytes, use));
}

void write_bytes(Rig& rig, const Buffer& buffer, std::span<const std::byte> bytes, VkDeviceSize offset) {
  require_ok(upload(*rig.context, *rig.memory, buffer, offset, bytes));
}

std::vector<std::byte> read_bytes(Rig& rig, const Buffer& buffer, std::size_t count, VkDeviceSize offset) {
  std::vector<std::byte> out(count);
  require_ok(download(*rig.context, *rig.memory, buffer, offset, out));
  return out;
}

std::vector<std::byte> pack(const kernels::YuvImage& image, const YuvLayout& layout) {
  const kernels::FormatInfo info = kernels::info(image.format);
  std::vector<std::byte> out(layout.bytes, std::byte{0});
  pack_plane(out, image.y, layout.luma, info.bytes, 1, 0);
  if (info.interleaved) {
    pack_plane(out, image.u, layout.cb, info.bytes, 2, 0);
    pack_plane(out, image.v, layout.cb, info.bytes, 2, 1);
  } else {
    pack_plane(out, image.u, layout.cb, info.bytes, 1, 0);
    pack_plane(out, image.v, layout.cr, info.bytes, 1, 0);
  }
  return out;
}

kernels::YuvImage unpack(std::span<const std::byte> bytes, const YuvView& view) {
  const kernels::FormatInfo info = kernels::info(view.format);
  kernels::YuvImage image(view.format, static_cast<int>(view.width), static_cast<int>(view.height), 0, 0, 0);
  unpack_plane(bytes, image.y, view.layout.luma, info.bytes, 1, 0);
  if (info.interleaved) {
    unpack_plane(bytes, image.u, view.layout.cb, info.bytes, 2, 0);
    unpack_plane(bytes, image.v, view.layout.cb, info.bytes, 2, 1);
  } else {
    unpack_plane(bytes, image.u, view.layout.cb, info.bytes, 1, 0);
    unpack_plane(bytes, image.v, view.layout.cr, info.bytes, 1, 0);
  }
  return image;
}

YuvBuffer upload_yuv(Rig& rig, const kernels::YuvImage& image) {
  const auto width = static_cast<std::uint32_t>(image.width());
  const auto height = static_cast<std::uint32_t>(image.height());
  const YuvLayout layout = require_value(tight_layout(image.format, width, height));
  Buffer buffer = make_buffer(rig, layout.bytes);
  write_bytes(rig, buffer, pack(image, layout));
  const YuvView view{
      .buffer = buffer.view(), .format = image.format, .width = width, .height = height, .layout = layout};
  return {.buffer = std::move(buffer), .view = view};
}

YuvBuffer yuv_target(Rig& rig, kernels::Format format, std::uint32_t width, std::uint32_t height) {
  const YuvLayout layout = require_value(tight_layout(format, width, height));
  Buffer buffer = make_buffer(rig, layout.bytes);
  write_bytes(rig, buffer, std::vector<std::byte>(layout.bytes, std::byte{0xA5}));
  const YuvView view{
      .buffer = buffer.view(), .format = format, .width = width, .height = height, .layout = layout};
  return {.buffer = std::move(buffer), .view = view};
}

kernels::YuvImage download_yuv(Rig& rig, const YuvBuffer& image) {
  return unpack(read_bytes(rig, image.buffer, image.view.layout.bytes), image.view);
}

TensorBuffer upload_planes(Rig& rig, std::span<const kernels::PlaneF> planes, Precision precision) {
  const auto width = static_cast<std::uint32_t>(planes.front().width);
  const auto height = static_cast<std::uint32_t>(planes.front().height);
  const auto count = static_cast<std::uint32_t>(planes.size());
  std::vector<std::byte> bytes;
  for (const kernels::PlaneF& p : planes) {
    for (const float v : p.data) {
      if (precision == Precision::F16) {
        const std::uint16_t half = kernels::ref::float_to_half(v);
        const auto* raw = reinterpret_cast<const std::byte*>(&half);
        bytes.insert(bytes.end(), raw, raw + sizeof(half));
      } else {
        const auto* raw = reinterpret_cast<const std::byte*>(&v);
        bytes.insert(bytes.end(), raw, raw + sizeof(v));
      }
    }
  }
  Buffer buffer = make_buffer(rig, bytes.size());
  write_bytes(rig, buffer, bytes);
  const TensorView view{
      .buffer = buffer.view(), .width = width, .height = height, .planes = count, .precision = precision};
  return {.buffer = std::move(buffer), .view = view};
}

TensorBuffer upload_rgb(Rig& rig, const kernels::RgbImage& image, Precision precision) {
  return upload_planes(rig, image.planes, precision);
}

TensorBuffer tensor_target(Rig& rig, std::uint32_t width, std::uint32_t height, std::uint32_t planes,
                           Precision precision) {
  const VkDeviceSize bytes = tensor_bytes(width, height, planes, precision);
  Buffer buffer = make_buffer(rig, bytes);
  write_bytes(rig, buffer, std::vector<std::byte>(bytes, std::byte{0x7F}));
  const TensorView view{
      .buffer = buffer.view(), .width = width, .height = height, .planes = planes, .precision = precision};
  return {.buffer = std::move(buffer), .view = view};
}

std::vector<kernels::PlaneF> download_planes(Rig& rig, const TensorBuffer& tensor) {
  const TensorView& v = tensor.view;
  const std::size_t area = std::size_t{v.width} * v.height;
  const std::vector<std::byte> bytes =
      read_bytes(rig, tensor.buffer, tensor_bytes(v.width, v.height, v.planes, v.precision));
  std::vector<kernels::PlaneF> planes;
  for (std::size_t p = 0; p < v.planes; ++p) {
    kernels::PlaneF plane(static_cast<int>(v.width), static_cast<int>(v.height));
    for (std::size_t i = 0; i < area; ++i) {
      const std::size_t at = p * area + i;
      if (v.precision == Precision::F16) {
        std::uint16_t half = 0;
        std::memcpy(&half, bytes.data() + at * 2, 2);
        plane.data[i] = kernels::ref::half_to_float(half);
      } else {
        std::memcpy(&plane.data[i], bytes.data() + at * 4, 4);
      }
    }
    planes.push_back(std::move(plane));
  }
  return planes;
}

kernels::RgbImage download_rgb(Rig& rig, const TensorBuffer& tensor) {
  std::vector<kernels::PlaneF> planes = download_planes(rig, tensor);
  kernels::RgbImage image;
  for (std::size_t c = 0; c < 3; ++c) image.planes[c] = std::move(planes[c]);
  return image;
}

kernels::YuvImage synthetic_yuv(kernels::Format format, int width, int height, std::uint32_t seed) {
  const kernels::FormatInfo info = kernels::info(format);
  const std::uint32_t top = kernels::max_code(format) >> static_cast<std::uint32_t>(info.shift);
  kernels::YuvImage image(format, width, height, 0, 0, 0);
  int channel = 0;
  for (kernels::PlaneU16* plane : {&image.y, &image.u, &image.v}) {
    for (int y = 0; y < plane->height; ++y) {
      for (int x = 0; x < plane->width; ++x) {
        // Bruit, une rampe sur la première ligne et les deux codes extrêmes aux coins.
        std::uint32_t code = noise(seed, x, y, channel) % (top + 1U);
        if (y == 0)
          code =
              static_cast<std::uint32_t>(x) * top / static_cast<std::uint32_t>(std::max(1, plane->width - 1));
        if (x == 0 && y == plane->height - 1) code = 0;
        if (x == plane->width - 1 && y == plane->height - 1) code = top;
        plane->at(x, y) = static_cast<std::uint16_t>(code << static_cast<std::uint32_t>(info.shift));
      }
    }
    ++channel;
  }
  return image;
}

kernels::PlaneF synthetic_plane(int width, int height, std::uint32_t seed, float lo, float hi) {
  kernels::PlaneF plane(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto unit = static_cast<float>(noise(seed, x, y, 7) >> 8U) / static_cast<float>(1U << 24U);
      plane.at(x, y) = lo + (hi - lo) * unit;
    }
  }
  return plane;
}

kernels::RgbImage synthetic_rgb(int width, int height, std::uint32_t seed, float lo, float hi) {
  kernels::RgbImage image;
  for (std::uint32_t c = 0; c < 3; ++c)
    image.planes[c] = synthetic_plane(width, height, seed * 3U + c, lo, hi);
  return image;
}

kernels::RgbImage as_stored(const kernels::RgbImage& image, Precision precision) {
  return precision == Precision::F16 ? kernels::ref::round_to_half(image) : image;
}

float max_abs_diff(const kernels::PlaneF& a, const kernels::PlaneF& b) {
  REQUIRE(a.width == b.width);
  REQUIRE(a.height == b.height);
  float worst = 0.0F;
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    const float d = std::fabs(a.data[i] - b.data[i]);
    // NaN compte comme l'écart le plus grand.
    worst = std::isnan(d) ? std::numeric_limits<float>::infinity() : std::max(worst, d);
  }
  return worst;
}

float max_abs_diff(const kernels::RgbImage& a, const kernels::RgbImage& b) {
  float worst = 0.0F;
  for (std::size_t c = 0; c < 3; ++c) worst = std::max(worst, max_abs_diff(a.planes[c], b.planes[c]));
  return worst;
}

int max_code_diff(const kernels::PlaneU16& a, const kernels::PlaneU16& b) {
  REQUIRE(a.width == b.width);
  REQUIRE(a.height == b.height);
  int worst = 0;
  for (std::size_t i = 0; i < a.data.size(); ++i)
    worst = std::max(worst, std::abs(int{a.data[i]} - int{b.data[i]}));
  return worst;
}

}  // namespace ajp::gpu::vk::test
