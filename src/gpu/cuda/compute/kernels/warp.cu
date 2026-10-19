// Warp arrière par flot et mélange de deux images à l'instant t, et plus grand déplacement.
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:963-1013 (k_warp_blend),
// 1025-1041 (k_flow_max) et 1051-1096 (k_warp_blend_mid), réécrits sur kernels/spec/warp.h :
// mêmes formules, dans l'ordre de kernels/ref/warp.cpp, sans contraction FMA.
// Tests : tests/unit/cuda_compute/warp_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda::dev {
namespace {

// Déplacements vers a et vers b, en pixels de la fenêtre, et poids de a.
struct Displacement {
  float ax;
  float ay;
  float bx;
  float by;
  float weight;
};

template <class M>
struct MotionPlanes {
  const M* flow;
  const M* mask;
  std::size_t plane;
  Motion m;

  __device__ float at(int c, float u, float v) const {
    return bilinear(flow + static_cast<std::size_t>(c) * plane, m.plane_width, m.x, m.y, m.width, m.height, u,
                    v);
  }
  __device__ float weight(float u, float v) const {
    return bilinear(mask, m.plane_width, m.x, m.y, m.width, m.height, u, v);
  }
};

// L'instant estimé par RIFE : mouvement et masque tels quels.
template <class M>
__device__ Displacement estimated(const MotionPlanes<M>& m, float u, float v) {
  return {m.at(0, u, v), m.at(1, u, v), m.at(2, u, v), m.at(3, u, v), m.weight(u, v)};
}

// Un autre instant depuis le mouvement du milieu : ligne droite et loi de mélange ajustée.
template <class M>
__device__ Displacement from_middle(const MotionPlanes<M>& m, float u, float v, float t,
                                    const BlendLaw& law) {
  const float a0x = m.at(0, u, v);
  const float a0y = m.at(1, u, v);
  const float a1x = m.at(2, u, v);
  const float a1y = m.at(3, u, v);
  const float u2 = u - ((1.0F - t) * a0x + t * a1x);
  const float v2 = v - ((1.0F - t) * a0y + t * a1y);
  const float tx = m.at(2, u2, v2) - m.at(0, u2, v2);
  const float ty = m.at(3, u2, v2) - m.at(1, u2, v2);
  const float mm = fminf(fmaxf(m.weight(u2, v2), law.epsilon), 1.0F - law.epsilon);
  const float z = law.a * logf(mm / (1.0F - mm)) + law.time_term + law.c;
  return {-t * tx, -t * ty, (1.0F - t) * tx, (1.0F - t) * ty, 1.0F / (1.0F + expf(-z))};
}

template <class T, class M>
__global__ void k_warp_blend(Tensor a, Tensor b, Motion motion, float t, BlendLaw law, Tensor dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int w = dst.width;
  const int h = dst.height;
  if (x >= w || y >= h) return;
  const MotionPlanes<M> planes{
      reinterpret_cast<const M*>(motion.flow), reinterpret_cast<const M*>(motion.mask),
      static_cast<std::size_t>(motion.plane_width) * static_cast<std::size_t>(motion.plane_height), motion};
  // Où ce pixel tombe dans la grille du mouvement, centres alignés.
  const float sx = static_cast<float>(w) / static_cast<float>(motion.width);
  const float sy = static_cast<float>(h) / static_cast<float>(motion.height);
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  const float u = (fx + 0.5F) / sx - 0.5F;
  const float v = (fy + 0.5F) / sy - 0.5F;
  const Displacement d = t == 0.5F ? estimated(planes, u, v) : from_middle(planes, u, v, t, law);
  const std::size_t plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  const T* pa = reinterpret_cast<const T*>(a.ptr);
  const T* pb = reinterpret_cast<const T*>(b.ptr);
  T* out = reinterpret_cast<T*>(dst.ptr);
  for (int c = 0; c < 3; ++c) {
    const std::size_t offset = static_cast<std::size_t>(c) * plane;
    const float va = bilinear(pa + offset, w, 0, 0, w, h, fx + sx * d.ax, fy + sy * d.ay);
    const float vb = bilinear(pb + offset, w, 0, 0, w, h, fx + sx * d.bx, fy + sy * d.by);
    out[offset + at(x, y, w)] = from_float<T>(va * d.weight + vb * (1.0F - d.weight));
  }
}

// Une lecture atomique par warp : les blocs de 32 × 8 ont des warps pleins, que chaque fil
// atteint (pas de retour anticipé avant le mélange).
template <class M>
__global__ void k_flow_max(Motion motion, std::uint32_t* result) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  float m = 0.0F;
  if (x < motion.width && y < motion.height) {
    const M* flow = reinterpret_cast<const M*>(motion.flow);
    const std::size_t plane =
        static_cast<std::size_t>(motion.plane_width) * static_cast<std::size_t>(motion.plane_height);
    const std::size_t i = at(motion.x + x, motion.y + y, motion.plane_width);
    for (int c = 0; c < 4; ++c) m = fmaxf(m, fabsf(to_float(flow[static_cast<std::size_t>(c) * plane + i])));
  }
  for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFU, m, offset));
  // Un float positif se compare comme ses bits.
  if ((threadIdx.x & 31U) == 0 && m > 0.0F) atomicMax(result, __float_as_uint(m));
}

template <class T>
void launch_warp(Tensor a, Tensor b, Motion motion, float t, BlendLaw law, Tensor dst, cudaStream_t stream) {
  const dim3 grid = grid2d(dst.width, dst.height);
  if (motion.elem == Elem::F16) {
    k_warp_blend<T, __half><<<grid, block2d(), 0, stream>>>(a, b, motion, t, law, dst);
  } else {
    k_warp_blend<T, float><<<grid, block2d(), 0, stream>>>(a, b, motion, t, law, dst);
  }
}

}  // namespace

cudaError_t warp_blend(Tensor a, Tensor b, Motion motion, float t, BlendLaw law, Tensor dst,
                       cudaStream_t stream) {
  if (dst.elem == Elem::F16) {
    launch_warp<__half>(a, b, motion, t, law, dst, stream);
  } else {
    launch_warp<float>(a, b, motion, t, law, dst, stream);
  }
  return cudaGetLastError();
}

cudaError_t flow_max(Motion motion, std::uint32_t* result, cudaStream_t stream) {
  const dim3 grid = grid2d(motion.width, motion.height);
  if (motion.elem == Elem::F16) {
    k_flow_max<__half><<<grid, block2d(), 0, stream>>>(motion, result);
  } else {
    k_flow_max<float><<<grid, block2d(), 0, stream>>>(motion, result);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
