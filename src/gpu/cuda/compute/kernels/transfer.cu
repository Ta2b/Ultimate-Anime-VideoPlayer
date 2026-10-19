// Fonctions de transfert (PQ, HLG, sRGB, BT.1886) et conversion de primaires, pixel par pixel.
// Écrit sur kernels/spec/transfer.h, dans l'ordre de kernels/ref/transfer.cpp ; les constantes
// d'écran (gamma et luma HLG, a et b de BT.1886, matrice en float) viennent de l'hôte, évaluées comme
// dans la référence. powf, logf et expf de CUDA s'écartent de quelques ulp de la bibliothèque C :
// tolerance::kTransferRelative les couvre. Tests : tests/unit/cuda_compute/transfer_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"
#include "kernels/spec/transfer.h"

namespace ajp::gpu::cuda::dev {
namespace {

namespace k = ajp::kernels;

// Borne une entrée à son domaine ; NaN donne la borne basse, comme fmaxf.
__device__ __forceinline__ float bounded(float v, float lo, float hi) {
  if (!(v > lo)) return lo;
  return v < hi ? v : hi;
}

__device__ __forceinline__ float clamp01(float v) {
  return bounded(v, 0.0F, 1.0F);
}

__device__ float decode(float v, const TransferParams& p) {
  switch (p.decode) {
    case Curve::Linear:
      return v;
    case Curve::Srgb: {
      const float s = clamp01(v);
      if (s <= k::srgb::kEncodedKnee) return s / k::srgb::kSlope;
      return powf((s + k::srgb::kOffset) / (1.0F + k::srgb::kOffset), k::srgb::kGamma);
    }
    case Curve::Bt1886:
      return p.bt1886_a * powf(fmaxf(clamp01(v) + p.bt1886_b, 0.0F), k::bt1886::kGamma);
    case Curve::Pq: {
      const float e = powf(clamp01(v), 1.0F / k::pq::kM2);
      const float y = powf(fmaxf(e - k::pq::kC1, 0.0F) / (k::pq::kC2 - k::pq::kC3 * e), 1.0F / k::pq::kM1);
      return y * k::pq::kPeakNits;
    }
    case Curve::Hlg: {
      const float s = clamp01(v);
      if (s <= 0.5F) return s * s / 3.0F;
      return (expf((s - k::hlg::kC) / k::hlg::kA) + k::hlg::kB) / 12.0F;
    }
  }
  return v;
}

__device__ float encode(float v, const TransferParams& p) {
  switch (p.encode) {
    case Curve::Linear:
      return v;
    case Curve::Srgb: {
      const float l = clamp01(v);
      if (l <= k::srgb::kLinearKnee) return l * k::srgb::kSlope;
      return (1.0F + k::srgb::kOffset) * powf(l, 1.0F / k::srgb::kGamma) - k::srgb::kOffset;
    }
    case Curve::Bt1886: {
      const float l = bounded(v, p.bt1886_black, p.bt1886_white);
      return powf(l / p.bt1886_a, 1.0F / k::bt1886::kGamma) - p.bt1886_b;
    }
    case Curve::Pq: {
      const float ym = powf(bounded(v, 0.0F, k::pq::kPeakNits) / k::pq::kPeakNits, k::pq::kM1);
      return powf((k::pq::kC1 + k::pq::kC2 * ym) / (1.0F + k::pq::kC3 * ym), k::pq::kM2);
    }
    case Curve::Hlg: {
      const float e = clamp01(v);
      if (e <= k::hlg::kKnee) return sqrtf(3.0F * e);
      return k::hlg::kA * logf(12.0F * e - k::hlg::kB) + k::hlg::kC;
    }
  }
  return v;
}

// OOTF HLG : Fd = Lw Ys^(γ − 1) E, Ys luma BT.2020 de la scène ; Ys ≤ 0 s'affiche noire.
__device__ float3 hlg_ootf(float3 s, const TransferParams& p) {
  const float ys = p.hlg_kr * s.x + (1.0F - p.hlg_kr - p.hlg_kb) * s.y + p.hlg_kb * s.z;
  const float gain = ys > 0.0F ? p.hlg_peak * powf(ys, p.hlg_gamma - 1.0F) : 0.0F;
  return make_float3(gain * s.x, gain * s.y, gain * s.z);
}

template <class S, class D>
__global__ void k_transfer(Tensor src, TransferParams p, Tensor dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= src.width || y >= src.height) return;
  const std::size_t plane = static_cast<std::size_t>(src.width) * static_cast<std::size_t>(src.height);
  const std::size_t i = at(x, y, src.width);
  const S* in = reinterpret_cast<const S*>(src.ptr);
  float3 c = make_float3(decode(to_float(in[i]), p), decode(to_float(in[plane + i]), p),
                         decode(to_float(in[2 * plane + i]), p));
  if (p.hlg_ootf) c = hlg_ootf(c, p);
  c = make_float3(c.x * p.gain, c.y * p.gain, c.z * p.gain);
  if (p.convert) {
    const float* m = p.matrix;
    c = make_float3(m[0] * c.x + m[1] * c.y + m[2] * c.z, m[3] * c.x + m[4] * c.y + m[5] * c.z,
                    m[6] * c.x + m[7] * c.y + m[8] * c.z);
  }
  D* out = reinterpret_cast<D*>(dst.ptr);
  out[i] = from_float<D>(encode(c.x, p));
  out[plane + i] = from_float<D>(encode(c.y, p));
  out[2 * plane + i] = from_float<D>(encode(c.z, p));
}

template <class S>
void launch(Tensor src, const TransferParams& p, Tensor dst, cudaStream_t stream) {
  const dim3 grid = grid2d(src.width, src.height);
  if (dst.elem == Elem::F16) {
    k_transfer<S, __half><<<grid, block2d(), 0, stream>>>(src, p, dst);
  } else {
    k_transfer<S, float><<<grid, block2d(), 0, stream>>>(src, p, dst);
  }
}

}  // namespace

cudaError_t apply_transfer(Tensor src, TransferParams params, Tensor dst, cudaStream_t stream) {
  if (src.elem == Elem::F16) {
    launch<__half>(src, params, dst, stream);
  } else {
    launch<float>(src, params, dst, stream);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
