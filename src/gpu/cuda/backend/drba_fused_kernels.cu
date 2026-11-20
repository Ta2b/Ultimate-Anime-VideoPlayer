// Noyaux de DistilDRBA fusionné (voir drba_fused_kernels.h et kernels/spec/drba_fused.h).
#include <cmath>
#include <cstddef>
#include <cuda_fp16.h>
#include <limits>

#include "gpu/cuda/backend/drba_fused_kernels.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kTimeChannel = 95;
constexpr int kFirstTimeChannel = 57;

__device__ __forceinline__ float rounded(float v) {
  return __half2float(__float2half_rn(v));
}

// Le côté choisi du graphe : vrai si tau < 1 (image 0), faux sinon (image 2). tau = 2 × plan 6 de pair01 ;
// le Clip du graphe, ((1 - tau) × 4096) borné à [0, 1], vaut 0 ou 1 pour un tau FP16 (jamais tau = 1 envoyé).
__device__ __forceinline__ bool before_side(float tau) {
  return fminf(fmaxf((1.0F - tau) * 4096.0F, 0.0F), 1.0F) >= 0.5F;
}

// Sortie 52 canaux d'un bloc ; Z = DepthToSpace(CRD, 2)(Y) de zh × zw.
struct SourceZ {
  const __half* y;
  int zh, zw;
  int pitch = 0;
};

// DepthToSpace lu à la volée : Z[c][zy][zx] = Y[4c + 2 (zy & 1) + (zx & 1)][zy >> 1][zx >> 1].
__device__ __forceinline__ float read_z(const SourceZ& z, int c, int zy, int zx) {
  const auto channel = static_cast<std::size_t>(4 * c + 2 * (zy & 1) + (zx & 1));
  if (z.pitch != 0) {
    const auto pixel = static_cast<std::size_t>(zy >> 1) * (z.zw / 2) + (zx >> 1);
    return __half2float(z.y[pixel * z.pitch + channel]);
  }
  return __half2float(z.y[(channel * static_cast<std::size_t>(z.zh / 2) + static_cast<std::size_t>(zy >> 1)) *
                              static_cast<std::size_t>(z.zw / 2) +
                          static_cast<std::size_t>(zx >> 1)]);
}

template <int Width, int Height>
struct TileZ {
  const float* values;
  int zh, zw, left, top;
};

template <int Width, int Height>
__device__ __forceinline__ float read_z(const TileZ<Width, Height>& z, int c, int zy, int zx) {
  return z.values[(c * Height + zy - z.top) * Width + zx - z.left];
}

// Resize bilinéaire demi-pixel de facteur s de Z (canaux c0 .. c0 + N - 1) au pixel (px, py) du plan agrandi,
// arrondi FP16 comme la sortie du Cast qui suit le Resize FP32 du graphe.
template <int N, class Z>
__device__ __forceinline__ void enlarge(const Z& z, int c0, int s, int px, int py, float (&r)[N]) {
  const float inv = 1.0F / static_cast<float>(s);
  float sy = (static_cast<float>(py) + 0.5F) * inv - 0.5F;
  float sx = (static_cast<float>(px) + 0.5F) * inv - 0.5F;
  sy = fminf(fmaxf(sy, 0.0F), static_cast<float>(z.zh - 1));
  sx = fminf(fmaxf(sx, 0.0F), static_cast<float>(z.zw - 1));
  const int y0 = static_cast<int>(sy);
  const int x0 = static_cast<int>(sx);
  const int y1 = min(y0 + 1, z.zh - 1);
  const int x1 = min(x0 + 1, z.zw - 1);
  const float wy = sy - static_cast<float>(y0);
  const float wx = sx - static_cast<float>(x0);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float a = read_z(z, c0 + i, y0, x0);
    const float b = read_z(z, c0 + i, y0, x1);
    const float c = read_z(z, c0 + i, y1, x0);
    const float e = read_z(z, c0 + i, y1, x1);
    r[i] = rounded((a * (1.0F - wx) + b * wx) * (1.0F - wy) + (c * (1.0F - wx) + e * wx) * wy);
  }
}

// GridSample (bilinéaire, bords répétés, coins alignés) : indices et poids depuis la position en pixels. La
// grille normalisée du graphe, base + flot × pas ramenée par (g + 1) / 2 × (n - 1), vaut exactement x + flot.
// Le graphe FP16 livré la calcule en deux sommes FP16, d'un pas de 0,31 px à 1280 de large et de 0,47 px à
// 1920 (A03-18, A46-04) ; ici en FP32, comme le réseau d'origine. Le fusionné s'écarte donc exprès du graphe
// FP16 tant que celui-ci n'est pas réexporté avec sa grille en FP32 (tools/rife/convert_rife_fp16.py).
struct Tap {
  int i00, i01, i10, i11;  // indices dans un plan h × w
  float wx, wy;
};

__device__ __forceinline__ Tap tap_of(int x, int y, float flow_x, float flow_y, int w, int h) {
  float ix = __fadd_rn(static_cast<float>(x), flow_x);
  float iy = __fadd_rn(static_cast<float>(y), flow_y);
  ix = fminf(fmaxf(ix, 0.0F), static_cast<float>(w - 1));
  iy = fminf(fmaxf(iy, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(ix);
  const int y0 = static_cast<int>(iy);
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  Tap t;
  t.i00 = y0 * w + x0;
  t.i01 = y0 * w + x1;
  t.i10 = y1 * w + x0;
  t.i11 = y1 * w + x1;
  t.wx = ix - static_cast<float>(x0);
  t.wy = iy - static_cast<float>(y0);
  return t;
}

__device__ __forceinline__ float sample(const __half* plane, const Tap& t) {
  const float a = __half2float(plane[t.i00]);
  const float b = __half2float(plane[t.i01]);
  const float d = __half2float(plane[t.i10]);
  const float e = __half2float(plane[t.i11]);
  return (a * (1.0F - t.wx) + b * t.wx) * (1.0F - t.wy) + (d * (1.0F - t.wx) + e * t.wx) * t.wy;
}

// Les trois images : plan c (0..8) = img0 pair01[0..2], img1 pair01[3..5], img2 pair12[3..5].
__device__ __forceinline__ const __half* image_plane(const DrbaPreParams& p, int c) {
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const int image = c / 3;
  const int k = c % 3;
  const auto* pair01 = static_cast<const __half*>(p.pair01);
  const auto* pair12 = static_cast<const __half*>(p.pair12);
  if (image == 0) return pair01 + static_cast<std::size_t>(k) * plane;
  if (image == 1) return pair01 + static_cast<std::size_t>(3 + k) * plane;
  return pair12 + static_cast<std::size_t>(3 + k) * plane;
}

// Une section de pre : 9 plans d'images moyennés par s, puis un plan du temps.
struct Section {
  int s;
  int oh, ow;
  std::size_t images;  // éléments
  std::size_t plane;   // éléments d'un plan réduit
  __half* x;
  int time_channel;
};

template <bool ScalarTime>
__global__ void __launch_bounds__(256) pre_kernel(DrbaPreParams p) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t full_plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  Section sections[3];                                   // NOLINT(modernize-avoid-c-arrays)
  const int reduce[3]{p.reduce2, p.reduce1, p.reduce0};  // NOLINT(modernize-avoid-c-arrays)
  __half* targets[3]{static_cast<__half*>(p.x2), static_cast<__half*>(p.x1), static_cast<__half*>(p.x0)};
  const int time_channels[3]{p.compact_x2 ? kFirstTimeChannel : kTimeChannel, kTimeChannel,
                             kFirstTimeChannel};  // NOLINT(modernize-avoid-c-arrays)
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    sections[k].s = reduce[k];
    sections[k].oh = p.height / reduce[k];
    sections[k].ow = p.width / reduce[k];
    sections[k].plane = static_cast<std::size_t>(sections[k].oh) * static_cast<std::size_t>(sections[k].ow);
    sections[k].images = 9U * sections[k].plane;
    sections[k].x = targets[k];
    sections[k].time_channel = time_channels[k];
  }
  std::size_t j = i;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    if (p.times_only) break;
    if (j < sections[k].images) {
      const Section& sec = sections[k];
      const int c = static_cast<int>(j / sec.plane);
      const std::size_t r = j % sec.plane;
      const int oy = static_cast<int>(r / static_cast<std::size_t>(sec.ow));
      const int ox = static_cast<int>(r % static_cast<std::size_t>(sec.ow));
      // Resize d'échelle 1/s (demi-pixel, linéaire) = moyenne 2 × 2 des pixels s o + s/2 - 1 et s o + s/2.
      const int base = sec.s / 2 - 1;
      const __half* src = image_plane(p, c);
      const std::size_t i00 =
          static_cast<std::size_t>(sec.s * oy + base) * static_cast<std::size_t>(p.width) +
          static_cast<std::size_t>(sec.s * ox + base);
      const float v = (__half2float(src[i00]) + __half2float(src[i00 + 1]) +
                       __half2float(src[i00 + static_cast<std::size_t>(p.width)]) +
                       __half2float(src[i00 + static_cast<std::size_t>(p.width) + 1])) *
                      0.25F;
      sec.x[static_cast<std::size_t>(c) * sec.plane + r] = __float2half_rn(v);
      return;
    }
    j -= sections[k].images;
  }
  // Plans du temps : u × 2, un pixel sur s de pair01[6].
  const auto* u = static_cast<const __half*>(p.pair01) + 6 * full_plane;
#pragma unroll
  for (int k = ScalarTime ? 2 : 0; k < 3; ++k) {
    if (j < sections[k].plane) {
      const Section& sec = sections[k];
      std::size_t at = 0;
      if constexpr (!ScalarTime) {
        const int oy = static_cast<int>(j / static_cast<std::size_t>(sec.ow));
        const int ox = static_cast<int>(j % static_cast<std::size_t>(sec.ow));
        at = static_cast<std::size_t>(sec.s * oy) * static_cast<std::size_t>(p.width) +
             static_cast<std::size_t>(sec.s * ox);
      }
      sec.x[static_cast<std::size_t>(sec.time_channel) * sec.plane + j] =
          __float2half_rn(2.0F * __half2float(u[at]));
      return;
    }
    j -= sections[k].plane;
  }
}

__global__ void __launch_bounds__(256) post_kernel(DrbaPostParams p) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int h1 = p.enc_height / 2, w1 = p.enc_width / 2;  // X_1 : l'encodage au demi
  const int h0 = p.enc_height / 4, w0 = p.enc_width / 4;  // X_0 : au quart
  const std::size_t plane1 = static_cast<std::size_t>(h1) * static_cast<std::size_t>(w1);
  const std::size_t plane0 = static_cast<std::size_t>(h0) * static_cast<std::size_t>(w0);
  const std::size_t enc_plane =
      static_cast<std::size_t>(p.enc_height) * static_cast<std::size_t>(p.enc_width);
  std::size_t j = i;
  int d;
  int ow;
  std::size_t plane;
  __half* out;
  if (j < 48U * plane1) {
    d = 2, ow = w1, plane = plane1, out = static_cast<__half*>(p.x1) + 9U * plane1;
  } else if (j < 48U * plane1 + 48U * plane0) {
    j -= 48U * plane1, d = 4, ow = w0, plane = plane0, out = static_cast<__half*>(p.x0) + 9U * plane0;
  } else {
    return;
  }
  const int c = static_cast<int>(j / plane);
  const std::size_t r = j % plane;
  const int oy = static_cast<int>(r / static_cast<std::size_t>(ow));
  const int ox = static_cast<int>(r % static_cast<std::size_t>(ow));
  const int base = d / 2 - 1;
  const __half* src = static_cast<const __half*>(p.x2) + (9U + static_cast<std::size_t>(c)) * enc_plane;
  const std::size_t i00 = static_cast<std::size_t>(d * oy + base) * static_cast<std::size_t>(p.enc_width) +
                          static_cast<std::size_t>(d * ox + base);
  const float v = (__half2float(src[i00]) + __half2float(src[i00 + 1]) +
                   __half2float(src[i00 + static_cast<std::size_t>(p.enc_width)]) +
                   __half2float(src[i00 + static_cast<std::size_t>(p.enc_width) + 1])) *
                  0.25F;
  out[static_cast<std::size_t>(c) * plane + r] = __float2half_rn(v);
}

constexpr int kGlueWidth = 32, kGlueHeight = 4, kGlueThreads = kGlueWidth * kGlueHeight;
constexpr int kPackedX2 = 112, kFirstDynamicX2 = 56, kPackedGroup = 8;
constexpr int kSharedStride = kGlueThreads + 1;

// La lecture NCHW est contiguë par warp, la réécriture HWC8 groupée par 16 octets.
__device__ __forceinline__ void store_packed(__half* packed, const __half* values, int pixel, int width,
                                             int group, int first) {
  const int x = static_cast<int>(blockIdx.x) * kGlueWidth + pixel % kGlueWidth;
  const int y = static_cast<int>(blockIdx.y) * kGlueHeight + pixel / kGlueWidth;
  auto* out = packed + (y * width + x) * kPackedX2 + first + group * kPackedGroup;
  const auto pair = [&](int pair) {
    return static_cast<unsigned>(__half_as_ushort(values[(2 * pair) * kSharedStride])) |
           (static_cast<unsigned>(__half_as_ushort(values[(2 * pair + 1) * kSharedStride])) << 16U);
  };
  *reinterpret_cast<uint4*>(out) = make_uint4(pair(0), pair(1), pair(2), pair(3));
}

__global__ void __launch_bounds__(kGlueThreads)
    pack_x2_kernel(const __half* x, __half* packed, int width, int height) {
  const int thread = static_cast<int>(threadIdx.x);
  const int ox = static_cast<int>(blockIdx.x) * kGlueWidth + thread % kGlueWidth;
  const int oy = static_cast<int>(blockIdx.y) * kGlueHeight + thread / kGlueWidth;
  const int pixel = oy * width + ox, plane = width * height;
  __shared__ __half values[kFirstDynamicX2 * kSharedStride];
#pragma unroll 1
  for (int c = 0; c < kFirstDynamicX2; ++c) values[c * kSharedStride + thread] = x[c * plane + pixel];
  __syncthreads();
  constexpr int groups = kFirstDynamicX2 / kPackedGroup;
#pragma unroll 1
  for (int i = thread; i < groups * kGlueThreads; i += kGlueThreads) {
    const int p = i / groups, group = i % groups;
    store_packed(packed, values + group * kPackedGroup * kSharedStride + p, p, width, group, 0);
  }
}

template <bool Packed>
__global__ void __launch_bounds__(kGlueThreads, Packed ? 5 : 0) glue_kernel(DrbaGlueParams p) {
  const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if constexpr (!Packed) {
    if (ox >= p.width || oy >= p.height) return;
  }
  __shared__ __half tile[Packed ? (kPackedX2 - kFirstDynamicX2) * kSharedStride : 1];
  const int lane = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t o =
      static_cast<std::size_t>(oy) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(ox);
  auto* x = static_cast<__half*>(p.x);
  const auto store = [&](int channel, __half value) {
    if constexpr (Packed)
      tile[(channel - kFirstDynamicX2) * kSharedStride + lane] = value;
    else
      x[static_cast<std::size_t>(channel) * plane + o] = value;
  };
  // Z du bloc précédent : à la moitié de la résolution de X_k.
  const SourceZ z{static_cast<const __half*>(p.y), p.height / 2, p.width / 2};
  const auto* pair01 = static_cast<const __half*>(p.pair01);
  const auto full_plane = static_cast<std::size_t>(p.full_height) * static_cast<std::size_t>(p.full_width);
  const float tau = 2.0F * __half2float(pair01[6 * full_plane]);
  const bool before = before_side(tau);

  // Flot (canaux 105-108) : Resize ×2 arrondi FP16, puis × 2 (exact).
  float flow[4];
  {
    float r[4];
    enlarge<4>(z, 0, 2, ox, oy, r);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      flow[k] = rounded(r[k] * 2.0F);
      store(105 + k, __float2half_rn(flow[k]));
    }
  }
  {
    float r[9];
    enlarge<9>(z, 4, 2, ox, oy, r);
#pragma unroll
    for (int k = 0; k < 9; ++k) store(96 + k, __float2half_rn(r[k]));
  }
  // Warps : n = 0 image 1 et f1 avec le flot 0-1 ; n = 1 image et f du côté choisi avec le flot 2-3.
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const Tap t = tap_of(ox, oy, flow[2 * n], flow[2 * n + 1], p.width, p.height);
    const int image = n == 0 ? 3 : (before ? 0 : 6);
    const int encoding = n == 0 ? 25 : (before ? 9 : 41);
#pragma unroll
    for (int k = 0; k < 3; ++k) store(57 + 3 * n + k, __float2half_rn(sample(x + (image + k) * plane, t)));
    for (int k = 0; k < 16; ++k)
      store(63 + 16 * n + k, __float2half_rn(sample(x + (encoding + k) * plane, t)));
  }
  if constexpr (Packed) {
    store(56, x[56 * plane + o]);
    store(kTimeChannel, p.scalar_time ? __float2half_rn(tau)
                                      : x[(p.compact_time ? kFirstTimeChannel : kTimeChannel) * plane + o]);
#pragma unroll
    for (int c = 109; c < kPackedX2; ++c) store(c, __ushort_as_half(0));
    __syncthreads();
    auto* packed = static_cast<__half*>(p.packed_x2);
    constexpr int groups = (kPackedX2 - kFirstDynamicX2) / kPackedGroup;
    for (int i = lane; i < groups * kGlueThreads; i += kGlueThreads) {
      const int j = i / groups, group = i % groups;
      store_packed(packed, tile + group * kPackedGroup * kSharedStride + j, j, p.width, group,
                   kFirstDynamicX2);
    }
  } else if (p.scalar_time) {
    store(kTimeChannel, __float2half_rn(tau));
  }
}

constexpr int kFinishWidth = 32, kFinishHeight = 8;

// Le halo bilinéaire tient dans (taille du bloc / S + 2) par axe. La lecture FP16 vers FP32
// est commune à la tuile, et chaque pixel conserve son Resize et ses frontières d'arrondi.
template <int S>
__global__ void __launch_bounds__(256) finish_kernel(DrbaFinishParams p) {
  constexpr int tw = kFinishWidth / (S != 0 ? S : 4) + 2;
  constexpr int th = kFinishHeight / (S != 0 ? S : 4) + 2;
  constexpr int cells = tw * th;
  const int scale = S != 0 ? S : p.upscale;
  const SourceZ z{static_cast<const __half*>(p.y), p.height / scale, p.width / scale, p.y_pitch};
  __shared__ float inputs[S != 0 ? 5 * cells : 1];
  const int left = static_cast<int>(blockIdx.x) * (tw - 2) - 1;
  const int top = static_cast<int>(blockIdx.y) * (th - 2) - 1;
  if constexpr (S != 0) {
    static_assert(5 * cells <= kFinishWidth * kFinishHeight);
    const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    if (thread < 5 * cells) {
      const int c = thread / cells, i = thread % cells;
      const int zy = max(0, min(z.zh - 1, top + i / tw));
      const int zx = max(0, min(z.zw - 1, left + i % tw));
      inputs[thread] = read_z(z, c, zy, zx);
    }
    __syncthreads();
  }
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t o =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const auto* pair01 = static_cast<const __half*>(p.pair01);
  const auto* pair12 = static_cast<const __half*>(p.pair12);
  const bool before = before_side(2.0F * __half2float(pair01[6 * plane]));
  float r[5];
  if constexpr (S != 0)
    enlarge<5>(TileZ<tw, th>{inputs, z.zh, z.zw, left, top}, 0, S, x, y, r);
  else
    enlarge<5>(z, 0, scale, x, y, r);
  float flow[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) flow[k] = rounded(r[k] * static_cast<float>(scale));
  const float sigmoid = rounded(1.0F / (1.0F + expf(-r[4])));
  if (p.flow != nullptr) {
    // Variante mouvement : flot et masque, le warp et le mélange se font après SPAN (kernels/spec/warp.h).
    auto* flow_out = static_cast<__half*>(p.flow);
#pragma unroll
    for (int k = 0; k < 4; ++k) flow_out[k * plane + o] = __float2half_rn(flow[k]);
    static_cast<__half*>(p.mask)[o] = __float2half_rn(sigmoid);
    return;
  }
  const float complement = rounded(1.0F - sigmoid);
  float warped[2][3];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const Tap t = tap_of(x, y, flow[2 * n], flow[2 * n + 1], p.width, p.height);
    const __half* src = n == 0 ? pair01 + 3 * plane : (before ? pair01 : pair12 + 3 * plane);
#pragma unroll
    for (int k = 0; k < 3; ++k) warped[n][k] = rounded(sample(src + k * plane, t));
  }
  auto* out = static_cast<__half*>(p.out);
#pragma unroll
  for (int k = 0; k < 3; ++k)
    out[k * plane + o] =
        __float2half_rn(rounded(warped[0][k] * sigmoid) + rounded(warped[1][k] * complement));
}

dim3 grid_for(int w, int h, dim3 block) {
  return dim3((static_cast<unsigned>(w) + block.x - 1) / block.x,
              (static_cast<unsigned>(h) + block.y - 1) / block.y);
}

unsigned blocks_for(std::size_t elements) {
  return static_cast<unsigned>((elements + 255U) / 256U);
}

}  // namespace

cudaError_t launch_drba_pre(const DrbaPreParams& params, cudaStream_t stream) {
  std::size_t elements = 0;
  const int reduces[]{params.reduce0, params.reduce1, params.reduce2};  // NOLINT(modernize-avoid-c-arrays)
  for (int k = 0; k < 3; ++k) {
    const int s = reduces[k];
    const std::size_t plane =
        static_cast<std::size_t>(params.height / s) * static_cast<std::size_t>(params.width / s);
    elements += ((params.times_only ? 0U : 9U) + (k == 0 || !params.scalar_time ? 1U : 0U)) * plane;
  }
  const auto blocks = blocks_for(elements);
  if (params.scalar_time)
    pre_kernel<true><<<blocks, 256, 0, stream>>>(params);
  else
    pre_kernel<false><<<blocks, 256, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_drba_post(const DrbaPostParams& params, cudaStream_t stream) {
  const std::size_t enc =
      static_cast<std::size_t>(params.enc_height) * static_cast<std::size_t>(params.enc_width);
  post_kernel<<<blocks_for(48U * (enc / 4U + enc / 16U)), 256, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_drba_glue(const DrbaGlueParams& params, cudaStream_t stream) {
  const dim3 block(kGlueWidth, kGlueHeight);
  const dim3 grid = grid_for(params.width, params.height, block);
  if (params.packed_x2 != nullptr) {
    if (params.width <= 0 || params.height <= 0 || params.width % kGlueWidth != 0 ||
        params.height % kGlueHeight != 0 ||
        params.height > std::numeric_limits<int>::max() / params.width / kPackedX2)
      return cudaErrorInvalidValue;
    glue_kernel<true><<<grid, block, 0, stream>>>(params);
  } else {
    glue_kernel<false><<<grid, block, 0, stream>>>(params);
  }
  return cudaGetLastError();
}

cudaError_t launch_drba_pack_x2(const void* x2, void* packed, int height, int width, cudaStream_t stream) {
  if (x2 == nullptr || packed == nullptr || width <= 0 || height <= 0 || width % kGlueWidth != 0 ||
      height % kGlueHeight != 0 || height > std::numeric_limits<int>::max() / width / kPackedX2)
    return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(width / kGlueWidth), static_cast<unsigned>(height / kGlueHeight));
  pack_x2_kernel<<<grid, kGlueThreads, 0, stream>>>(static_cast<const __half*>(x2),
                                                    static_cast<__half*>(packed), width, height);
  return cudaGetLastError();
}

cudaError_t launch_drba_finish(const DrbaFinishParams& params, cudaStream_t stream) {
  const dim3 block(kFinishWidth, kFinishHeight);
  const dim3 grid = grid_for(params.width, params.height, block);
  if (params.shared_resize && params.upscale == 4)
    finish_kernel<4><<<grid, block, 0, stream>>>(params);
  else if (params.shared_resize && params.upscale == 8)
    finish_kernel<8><<<grid, block, 0, stream>>>(params);
  else
    finish_kernel<0><<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
