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
