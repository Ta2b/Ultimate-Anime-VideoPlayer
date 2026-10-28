// Les images des tests du backend TensorRT : une image NV12 dans la mémoire CUDA, comme NVDEC la
// livre, une rampe RGB, et la DecodedFrame qui les porte. Partagées par backend_test.cpp et
// backend_motion_test.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/backend.h"
#include "kernels/image.h"

namespace ajp::test {

// Image NV12 dans la mémoire CUDA, comme NVDEC la livre (plans à pas de ligne).
class CudaNv12 final : public FrameData {
 public:
  CudaNv12(const kernels::YuvImage& image) {
    const auto w = static_cast<std::size_t>(image.width());
    const auto h = static_cast<std::size_t>(image.height());
    REQUIRE(cudaMallocPitch(&luma_, &luma_pitch_, w, h) == cudaSuccess);
    REQUIRE(cudaMallocPitch(&chroma_, &chroma_pitch_, w, h / 2) == cudaSuccess);
    std::vector<std::uint8_t> y(w * h);
    std::vector<std::uint8_t> uv(w * h / 2);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = static_cast<std::uint8_t>(image.y.data[i]);
    for (std::size_t i = 0; i < image.u.data.size(); ++i) {
      uv[2 * i] = static_cast<std::uint8_t>(image.u.data[i]);
      uv[2 * i + 1] = static_cast<std::uint8_t>(image.v.data[i]);
    }
    REQUIRE(cudaMemcpy2D(luma_, luma_pitch_, y.data(), w, w, h, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy2D(chroma_, chroma_pitch_, uv.data(), w, w, h / 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  ~CudaNv12() override {
    cudaFree(luma_);
    cudaFree(chroma_);
  }
  FrameMemory memory() const override { return FrameMemory::Cuda; }
  int plane_count() const override { return 2; }
  std::span<const std::byte> host_plane(int) const override { return {}; }
  std::size_t host_stride(int) const override { return 0; }
  std::uintptr_t native_plane(int plane) const override {
    return reinterpret_cast<std::uintptr_t>(plane == 0 ? luma_ : chroma_);
  }
  std::size_t native_pitch(int plane) const override { return plane == 0 ? luma_pitch_ : chroma_pitch_; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }

 private:
  void* luma_ = nullptr;
  void* chroma_ = nullptr;
  std::size_t luma_pitch_ = 0;
  std::size_t chroma_pitch_ = 0;
};

// Image YUV 4:2:0 en mémoire hôte, comme le décodage logiciel la rend : trois plans (YUV 4:2:0
// planaire) ou deux (NV12, P010, P016), échantillons de 8 ou 16 bits, lignes à marge.
class HostYuv final : public FrameData {
 public:
  explicit HostYuv(const kernels::YuvImage& image) {
    const kernels::FormatInfo info = kernels::info(image.format);
    sample_ = static_cast<std::size_t>(info.bytes);
    planes_ = info.interleaved ? 2 : 3;
    add(image.y);
    if (planes_ == 3) {
      add(image.u);
      add(image.v);
      return;
    }
    // Entrelacée : Cb et Cr alternent sur une ligne de chroma.
    const auto cw = static_cast<std::size_t>(image.u.width);
    const auto ch = static_cast<std::size_t>(image.u.height);
    strides_.push_back(2 * cw * sample_ + 32);
    bytes_.emplace_back(strides_.back() * ch);
    for (std::size_t y = 0; y < ch; ++y) {
      for (std::size_t x = 0; x < cw; ++x) {
        const auto at = static_cast<int>(x);
        put(bytes_.back(), y * strides_.back() + 2 * x * sample_, image.u.at(at, static_cast<int>(y)));
        put(bytes_.back(), y * strides_.back() + (2 * x + 1) * sample_, image.v.at(at, static_cast<int>(y)));
      }
    }
  }
  FrameMemory memory() const override { return FrameMemory::Host; }
  int plane_count() const override { return planes_; }
  std::span<const std::byte> host_plane(int plane) const override {
    return bytes_[static_cast<std::size_t>(plane)];
  }
  std::size_t host_stride(int plane) const override { return strides_[static_cast<std::size_t>(plane)]; }
  std::uintptr_t native_plane(int) const override { return 0; }
  std::size_t native_pitch(int) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }

 private:
  void put(std::vector<std::byte>& bytes, std::size_t at, std::uint16_t value) const {
    bytes[at] = static_cast<std::byte>(value & 0xFF);
    if (sample_ == 2) bytes[at + 1] = static_cast<std::byte>(value >> 8);
  }
  void add(const kernels::PlaneU16& plane) {
    const auto w = static_cast<std::size_t>(plane.width);
    const auto h = static_cast<std::size_t>(plane.height);
    strides_.push_back(w * sample_ + 32);
    bytes_.emplace_back(strides_.back() * h);
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x)
        put(bytes_.back(), y * strides_.back() + x * sample_,
            plane.at(static_cast<int>(x), static_cast<int>(y)));
    }
  }
  std::size_t sample_ = 1;
  int planes_ = 2;
  std::vector<std::vector<std::byte>> bytes_;
  std::vector<std::size_t> strides_;
};

// Image RGB w × h : plan c, pixel (x, y) à 0,1 × (c + 1) + 0,001 × (x + y).
inline std::vector<float> ramp(int w, int h) {
  std::vector<float> v(static_cast<std::size_t>(3 * w * h));
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        v[static_cast<std::size_t>((c * h + y) * w + x)] =
            0.1F * static_cast<float>(c + 1) + 0.001F * static_cast<float>(x + y);
      }
    }
  }
  return v;
}

inline DecodedFrame frame_of(std::shared_ptr<const FrameData> data, int w, int h) {
  return DecodedFrame{
      .id = {1}, .width = w, .height = h, .format = PixelFormat::Nv12, .data = std::move(data)};
}

}  // namespace ajp::test
