#include "rig.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/half.h"

namespace ajp::gpu::cuda::test {
namespace {

// Pas de ligne d'un plan sur l'appareil : la ligne, plus de quoi la décaler, arrondie à 256
// octets comme les surfaces de NVDEC.
std::size_t pitch_for(std::size_t row_bytes) {
  return (row_bytes + 64 + 255) / 256 * 256;
}

struct PlaneLayout {
  int width = 0;  // échantillons par ligne (CbCr entrelacés comptent double)
  int height = 0;
};

std::vector<PlaneLayout> layouts(kernels::Format format, int width, int height) {
  const kernels::FormatInfo info = kernels::info(format);
  const int cw = info.subsampled ? width / 2 : width;
  const int ch = info.subsampled ? height / 2 : height;
  if (info.interleaved) return {{width, height}, {2 * cw, ch}};
  return {{width, height}, {cw, ch}, {cw, ch}};
}

}  // namespace

DeviceBuffer Rig::upload(const void* host, std::size_t bytes) {
  auto buffer = pool->allocate(bytes, "test");
  REQUIRE(buffer);
  REQUIRE(cudaMemcpyAsync(buffer->data(), host, bytes, cudaMemcpyHostToDevice, s()) == cudaSuccess);
  REQUIRE(stream.synchronize());
  return std::move(*buffer);
}

std::vector<std::byte> Rig::download(const void* source, std::size_t bytes) {
  std::vector<std::byte> out(bytes);
  REQUIRE(cudaMemcpyAsync(out.data(), source, bytes, cudaMemcpyDeviceToHost, s()) == cudaSuccess);
  REQUIRE(stream.synchronize());
  return out;
}

RigOrReason open_rig(std::size_t pool_bytes) {
  auto devices = list_devices();
  if (!devices) return {.rig = std::nullopt, .reason = "pas de pilote CUDA : " + devices.error().detail};
  if (devices->empty()) return {.rig = std::nullopt, .reason = "aucun appareil CUDA"};
  auto device = Device::open(devices->front().uuid);
  REQUIRE(device);
  auto pool = DevicePool::create(pool_bytes);
  REQUIRE(pool);
  auto stream = Stream::create();
  REQUIRE(stream);
  RigOrReason out;
  out.rig.emplace(Rig{.device = std::move(*device), .pool = std::move(*pool), .stream = std::move(*stream)});
  return out;
}

DeviceYuv alloc_yuv(Rig& rig, kernels::Format format, int width, int height) {
  const auto sample = static_cast<std::size_t>(kernels::info(format).bytes);
  DeviceYuv out;
  out.planes = {.format = format, .width = width, .height = height};
  std::size_t i = 0;
  for (const PlaneLayout& l : layouts(format, width, height)) {
    const std::size_t pitch = pitch_for(static_cast<std::size_t>(l.width) * sample);
    auto buffer = rig.pool->allocate(pitch * static_cast<std::size_t>(l.height), "test : plan YUV");
    REQUIRE(buffer);
    REQUIRE(cudaMemsetAsync(buffer->data(), 0xA5, buffer->size(), rig.s()) == cudaSuccess);
    out.planes.planes[i++] = {.ptr = buffer->address(), .pitch = pitch};
    out.buffers.push_back(std::move(*buffer));
  }
  REQUIRE(rig.stream.synchronize());
  return out;
}

DeviceYuv upload_yuv(Rig& rig, const kernels::YuvImage& image) {
  const kernels::Format format = image.format;
  const kernels::FormatInfo info = kernels::info(format);
  DeviceYuv out = alloc_yuv(rig, format, image.width(), image.height());
  const auto sample = static_cast<std::size_t>(info.bytes);
  std::vector<std::vector<std::uint16_t>> rows;
  rows.push_back(image.y.data);
  if (info.interleaved) {
    std::vector<std::uint16_t> uv(image.u.data.size() * 2);
    for (std::size_t i = 0; i < image.u.data.size(); ++i) {
      uv[2 * i] = image.u.data[i];
      uv[2 * i + 1] = image.v.data[i];
    }
    rows.push_back(std::move(uv));
  } else {
    rows.push_back(image.u.data);
    rows.push_back(image.v.data);
  }
  const std::vector<PlaneLayout> ls = layouts(format, image.width(), image.height());
  for (std::size_t p = 0; p < ls.size(); ++p) {
    const auto w = static_cast<std::size_t>(ls[p].width);
    std::vector<std::byte> bytes(w * sample * static_cast<std::size_t>(ls[p].height));
    for (std::size_t i = 0; i < rows[p].size(); ++i) {
      const std::uint16_t code = rows[p][i];
      if (sample == 1) {
        bytes[i] = static_cast<std::byte>(code);
      } else {
        std::memcpy(&bytes[2 * i], &code, 2);
      }
    }
    const Plane& dst = out.planes.planes[p];
    REQUIRE(cudaMemcpy2DAsync(reinterpret_cast<void*>(dst.ptr), dst.pitch, bytes.data(), w * sample,
                              w * sample, static_cast<std::size_t>(ls[p].height), cudaMemcpyHostToDevice,
                              rig.s()) == cudaSuccess);
    REQUIRE(rig.stream.synchronize());
  }
  return out;
}

kernels::YuvImage download_yuv(Rig& rig, const DeviceYuv& image) {
  const kernels::Format format = image.planes.format;
  const kernels::FormatInfo info = kernels::info(format);
  const auto sample = static_cast<std::size_t>(info.bytes);
  kernels::YuvImage out(format, image.planes.width, image.planes.height, 0, 0, 0);
  const std::vector<PlaneLayout> ls = layouts(format, image.planes.width, image.planes.height);
  std::vector<std::vector<std::uint16_t>> codes;
  for (std::size_t p = 0; p < ls.size(); ++p) {
    const auto w = static_cast<std::size_t>(ls[p].width);
    const auto h = static_cast<std::size_t>(ls[p].height);
    std::vector<std::byte> bytes(w * sample * h);
    const Plane& src = image.planes.planes[p];
    REQUIRE(cudaMemcpy2DAsync(bytes.data(), w * sample, reinterpret_cast<const void*>(src.ptr), src.pitch,
                              w * sample, h, cudaMemcpyDeviceToHost, rig.s()) == cudaSuccess);
    REQUIRE(rig.stream.synchronize());
    std::vector<std::uint16_t> plane(w * h);
    for (std::size_t i = 0; i < plane.size(); ++i) {
      if (sample == 1) {
        plane[i] = static_cast<std::uint16_t>(bytes[i]);
      } else {
        std::memcpy(&plane[i], &bytes[2 * i], 2);
      }
    }
    codes.push_back(std::move(plane));
  }
  out.y.data = codes[0];
  if (info.interleaved) {
    for (std::size_t i = 0; i < out.u.data.size(); ++i) {
      out.u.data[i] = codes[1][2 * i];
      out.v.data[i] = codes[1][2 * i + 1];
    }
  } else {
    out.u.data = codes[1];
    out.v.data = codes[2];
  }
  return out;
}

DeviceTensor alloc_tensor(Rig& rig, Elem elem, int width, int height, int channels) {
  const std::size_t count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
  auto buffer = rig.pool->allocate(count * elem_size(elem), "test : tenseur");
  REQUIRE(buffer);
  // Des NaN partout : une case que le noyau oublie d'écrire se voit.
  REQUIRE(cudaMemsetAsync(buffer->data(), 0xFF, buffer->size(), rig.s()) == cudaSuccess);
  REQUIRE(rig.stream.synchronize());
  const Tensor t{
      .ptr = buffer->address(), .elem = elem, .width = width, .height = height, .channels = channels};
  return {.buffer = std::move(*buffer), .tensor = t};
}

DeviceTensor upload_planes(Rig& rig, const std::vector<kernels::PlaneF>& planes, Elem elem) {
  REQUIRE(!planes.empty());
  const int w = planes[0].width;
  const int h = planes[0].height;
  DeviceTensor out = alloc_tensor(rig, elem, w, h, static_cast<int>(planes.size()));
  std::vector<float> f32;
  std::vector<std::uint16_t> f16;
  for (const kernels::PlaneF& p : planes) {
    REQUIRE(p.width == w);
    REQUIRE(p.height == h);
    for (const float v : p.data) {
      f32.push_back(v);
      f16.push_back(kernels::ref::float_to_half(v));
    }
  }
  const void* host = elem == Elem::F32 ? static_cast<const void*>(f32.data()) : f16.data();
  REQUIRE(cudaMemcpyAsync(out.buffer.data(), host, f32.size() * elem_size(elem), cudaMemcpyHostToDevice,
                          rig.s()) == cudaSuccess);
  REQUIRE(rig.stream.synchronize());
  return out;
}

DeviceTensor upload_rgb(Rig& rig, const kernels::RgbImage& image, Elem elem) {
  return upload_planes(rig, {image.planes.begin(), image.planes.end()}, elem);
}

std::vector<kernels::PlaneF> download_planes(Rig& rig, const DeviceTensor& t) {
  const Tensor& d = t.tensor;
  const std::size_t plane = static_cast<std::size_t>(d.width) * static_cast<std::size_t>(d.height);
  const std::size_t count = plane * static_cast<std::size_t>(d.channels);
  const std::vector<std::byte> bytes = rig.download(t.buffer.data(), count * elem_size(d.elem));
  std::vector<kernels::PlaneF> out;
  for (int c = 0; c < d.channels; ++c) {
    kernels::PlaneF p(d.width, d.height);
    for (std::size_t i = 0; i < plane; ++i) {
      const std::size_t at = static_cast<std::size_t>(c) * plane + i;
      if (d.elem == Elem::F32) {
        std::memcpy(&p.data[i], &bytes[4 * at], 4);
      } else {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &bytes[2 * at], 2);
        p.data[i] = kernels::ref::half_to_float(bits);
      }
    }
    out.push_back(std::move(p));
  }
  return out;
}

kernels::RgbImage download_rgb(Rig& rig, const DeviceTensor& t) {
  REQUIRE(t.tensor.channels == 3);
  std::vector<kernels::PlaneF> planes = download_planes(rig, t);
  kernels::RgbImage out;
  for (std::size_t c = 0; c < 3; ++c) out.planes[c] = std::move(planes[c]);
  return out;
}

kernels::YuvImage random_yuv(kernels::Format format, int width, int height, std::uint32_t seed) {
  const kernels::FormatInfo info = kernels::info(format);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint32_t> code(0, (1U << info.bits) - 1U);
  kernels::YuvImage image(format, width, height, 0, 0, 0);
  for (kernels::PlaneU16* p : {&image.y, &image.u, &image.v}) {
    for (std::uint16_t& v : p->data) v = static_cast<std::uint16_t>(code(rng) << info.shift);
  }
  return image;
}

kernels::PlaneF random_plane(int width, int height, float lo, float hi, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> value(lo, hi);
  kernels::PlaneF p(width, height);
  for (float& v : p.data) v = value(rng);
  return p;
}

kernels::RgbImage random_rgb(int width, int height, float lo, float hi, std::uint32_t seed) {
  kernels::RgbImage image;
  for (std::size_t c = 0; c < 3; ++c) {
    image.planes[c] = random_plane(width, height, lo, hi, seed + static_cast<std::uint32_t>(c) * 7919U);
  }
  return image;
}

float max_diff(const kernels::PlaneF& a, const kernels::PlaneF& b) {
  REQUIRE(a.width == b.width);
  REQUIRE(a.height == b.height);
  float m = 0.0F;
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    const float x = a.data[i];
    const float y = b.data[i];
    if (std::isnan(x) && std::isnan(y)) continue;
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<float>::infinity();
    m = std::max(m, std::fabs(x - y));
  }
  return m;
}

float max_diff(const kernels::RgbImage& a, const kernels::RgbImage& b) {
  float m = 0.0F;
  for (std::size_t c = 0; c < 3; ++c) m = std::max(m, max_diff(a.planes[c], b.planes[c]));
  return m;
}

}  // namespace ajp::gpu::cuda::test
