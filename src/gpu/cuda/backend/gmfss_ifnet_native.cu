#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "gpu/cuda/backend/gmfss_ifnet_native.h"
#include "gpu/cuda/backend/rife_native_layer.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda {
namespace {

constexpr int kPitchIn = 16;
constexpr int kPitchOut = 24;
constexpr int kThreads = 128;

struct Images {
  const __half* a;
  const __half* b;
  const __half* time;
  int h;
  int w;
};

struct Blocks {
  const __half* out[4];
};

// La transposée écrit 24 canaux HWC. PixelShuffle CRD relit quatre de ces canaux par plan.
__device__ float shuffled(const __half* out, int x, int y, int width, int c) {
  const int phase = 2 * (y & 1) + (x & 1);
  const auto p = static_cast<std::size_t>(y / 2) * (width / 2) + x / 2;
  return __half2float(out[p * kPitchOut + 4 * c + phase]);
}

struct Relift {
  warp::Axis x[4];
  warp::Axis y[4];
  int widths[4];
};

template <int Count>
__device__ Relift relift(int x, int y, int w, int h) {
  Relift r;
#pragma unroll
  for (int k = 0; k < Count; ++k) {
    const int d = 8 >> k;
    r.widths[k] = w / d;
    r.x[k] = warp::coarse_axis(x, 1.F / d, r.widths[k]);
    r.y[k] = warp::coarse_axis(y, 1.F / d, h / d);
  }
  return r;
}

template <int Count>
__device__ float accumulated(const Blocks& blocks, const Relift& r, int c) {
  float flow = 0.F;
#pragma unroll
  for (int k = 0; k < Count; ++k) {
    const warp::Axis ax = r.x[k], ay = r.y[k];
    const __half* src = blocks.out[k];
    const int w = r.widths[k];
    const float resized = warp::rounded(
        warp::lerp2(shuffled(src, ax.i0, ay.i0, w, c), shuffled(src, ax.i1, ay.i0, w, c),
                    shuffled(src, ax.i0, ay.i1, w, c), shuffled(src, ax.i1, ay.i1, w, c), ax, ay));
    const float delta = c == 4 ? resized : warp::rounded(resized * static_cast<float>(8 >> k));
    flow = k == 0 ? delta : warp::rounded(flow + delta);
  }
  return flow;
}

// La grille de l'ONNX est en FP32 : [−1, 1], align_corners et bord répété.
__device__ float grid_position(int p, float flow, int size) {
  const float denom = static_cast<float>(size - 1);
  const float grid = __fadd_rn(__fmul_rn(static_cast<float>(p), 2.F / denom), -1.F);
  const float displaced = __fadd_rn(grid, flow / (denom * .5F));
  return __fmul_rn(__fmul_rn(__fadd_rn(displaced, 1.F), .5F), denom);
}

template <int Count>
__device__ warp::Tap motion_tap(const Images& im, const Blocks& blocks, const Relift& r, int x, int y,
                                int source) {
  const int c = 2 * source;
  const float u = grid_position(x, accumulated<Count>(blocks, r, c), im.w);
  const float v = grid_position(y, accumulated<Count>(blocks, r, c + 1), im.h);
  return warp::tap_at(0, 0, u, v, {0, 0, im.w, im.h});
}

// Resize demi-pixel par un entier pair : seuls les quatre pixels autour du centre contribuent.
template <int K>
__global__ void ifnet_input(Images im, Blocks blocks, __half* out) {
  constexpr int d = 8 >> K;
  const int w = im.w / d, h = im.h / d;
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= w * h) return;
  const int x0 = (p % w) * d + (d - 1) / 2;
  const int y0 = (p / w) * d + (d - 1) / 2;
  const int plane = im.h * im.w;
  float sum[12]{};
  const __half* source[]{im.a, im.b};
  constexpr int taps = d == 1 ? 1 : 4;
#pragma unroll
  for (int j = 0; j < taps; ++j) {
    const int x = x0 + (j & 1), y = y0 + (j >> 1);
    if constexpr (K == 0) {
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int c = 0; c < 3; ++c) sum[3 * s + c] += __half2float(source[s][c * plane + y * im.w + x]);
    } else {
      const Relift r = relift<K>(x, y, im.w, im.h);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int c = 2 * s;
        const float u = accumulated<K>(blocks, r, c);
        const float v = accumulated<K>(blocks, r, c + 1);
        const warp::Tap t =
            warp::tap_at(0, 0, grid_position(x, u, im.w), grid_position(y, v, im.h), {0, 0, im.w, im.h});
#pragma unroll
        for (int rgb = 0; rgb < 3; ++rgb)
          sum[3 * s + rgb] += warp::rounded(warp::bilinear(source[s] + rgb * plane, t, im.w));
        sum[8 + c] += u;
        sum[9 + c] += v;
      }
      sum[7] += accumulated<K>(blocks, r, 4);
    }
  }
  __half* to = out + static_cast<std::size_t>(p) * kPitchIn;
#pragma unroll
  for (int c = 0; c < 6; ++c) to[c] = __float2half_rn(sum[c] / taps);
  to[6] = *im.time;
  if constexpr (K > 0) {
    to[7] = __float2half_rn(sum[7] / taps);
#pragma unroll
    for (int c = 8; c < 12; ++c) to[c] = __float2half_rn(warp::rounded(sum[c] / taps) / d);
  }
#pragma unroll
  for (int c = K == 0 ? 7 : 12; c < kPitchIn; ++c) to[c] = __float2half_rn(0.F);
}

__global__ void ifnet_output(Images im, Blocks blocks, __half* out) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int plane = im.h * im.w;
  if (p >= plane) return;
  const int x = p % im.w, y = p / im.w;
  const Relift r = relift<4>(x, y, im.w, im.h);
  const warp::Tap a = motion_tap<4>(im, blocks, r, x, y, 0);
  const warp::Tap b = motion_tap<4>(im, blocks, r, x, y, 1);
  const float alpha = warp::rounded(1.F / (1.F + expf(-accumulated<4>(blocks, r, 4))));
  const float beta = warp::rounded(1.F - alpha);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int offset = c * plane;
    const float l = warp::rounded(warp::bilinear(im.a + offset, a, im.w));
    const float r = warp::rounded(warp::bilinear(im.b + offset, b, im.w));
    out[offset + p] = __float2half_rn(warp::rounded(l * alpha) + warp::rounded(r * beta));
  }
}

class Ifnet final : public GmfssIfnetNative {
 public:
  Ifnet(std::unique_ptr<RifeNativeNet> net, int h, int w) : net_(std::move(net)), h_(h), w_(w) {}

  cudaError_t allocate() {
    cudaError_t result = workspace_.allocate(static_cast<std::size_t>(net_->workspace_bytes()));
    if (result != cudaSuccess) return result;
    const auto plane = static_cast<std::size_t>(h_) * w_;
    result = input_.allocate(plane * kPitchIn * sizeof(__half));
    for (std::size_t k = 0; k < out_.size() && result == cudaSuccess; ++k) {
      const int d = 2 * RifeNativeNet::kIfnetScales[k];
      result = out_[k].allocate((plane / (d * d)) * kPitchOut * sizeof(__half));
    }
    return result;
  }

  cudaError_t run(cudaStream_t stream, const void* a, const void* b, const void* time,
                  void* out) const override {
    if (a == nullptr || b == nullptr || time == nullptr || out == nullptr) return cudaErrorInvalidValue;
    const Images im{static_cast<const __half*>(a), static_cast<const __half*>(b),
                    static_cast<const __half*>(time), h_, w_};
    const Blocks blocks{
        {out_[0].as<__half>(), out_[1].as<__half>(), out_[2].as<__half>(), out_[3].as<__half>()}};
    cudaError_t result = block<0>(stream, im, blocks);
    if (result == cudaSuccess) result = block<1>(stream, im, blocks);
    if (result == cudaSuccess) result = block<2>(stream, im, blocks);
    if (result == cudaSuccess) result = block<3>(stream, im, blocks);
    if (result != cudaSuccess) return result;
    ifnet_output<<<blocks_for(w_ * h_), kThreads, 0, stream>>>(im, blocks, static_cast<__half*>(out));
    return cudaGetLastError();
  }

 private:
  static int blocks_for(int elements) { return (elements + kThreads - 1) / kThreads; }

  template <int K>
  cudaError_t block(cudaStream_t stream, const Images& im, const Blocks& blocks) const {
    constexpr int d = RifeNativeNet::kIfnetScales[K];
    auto* x = input_.as<__half>();
    ifnet_input<K><<<blocks_for((w_ / d) * (h_ / d)), kThreads, 0, stream>>>(im, blocks, x);
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) return result;
    return net_->block(stream, K, x, out_[K].as<void>(), workspace_.as<void>(), nullptr, nullptr);
  }

  std::unique_ptr<RifeNativeNet> net_;
  int h_;
  int w_;
  rife_native::Buffer workspace_, input_;
  std::array<rife_native::Buffer, 4> out_;
};

}  // namespace

std::unique_ptr<GmfssIfnetNative> GmfssIfnetNative::create(const std::array<RifeNativePiece, 4>& pieces,
                                                           int height, int width, cudaStream_t stream,
                                                           std::string& error) {
  auto net = RifeNativeNet::create_ifnet(pieces, height, width, stream, error);
  if (!net) return nullptr;
  auto ifnet = std::make_unique<Ifnet>(std::move(net), height, width);
  if (const cudaError_t result = ifnet->allocate(); result != cudaSuccess) {
    error = cudaGetErrorString(result);
    return nullptr;
  }
  return ifnet;
}

}  // namespace kirari::gpu::cuda
