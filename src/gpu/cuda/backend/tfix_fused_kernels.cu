// TemporalFix fusionné (kernels/spec/tfix_fused.h) : l'encodeur fusionné (moyenne 2 × 2, pyramide) et les
// entrées des blocs (X0, puis la colle de X1 et X2). La tête de confiance et le mélange :
// tfix_blend_kernels.cu. Référence de chaque pas : tools/temporalfix/fused_reference.py.
#include "gpu/cuda/backend/tfix_fused.cuh"

namespace kirari::gpu::cuda {
namespace {

using tfix::Axis;
using tfix::half_ptr;

constexpr int kTile = 16;

__constant__ int kSupportFrame[kernels::kTfixSupports] = {0, 1, 2, 4, 5, 6};
__constant__ float kTime[kernels::kTfixSupports] = {-1.0F,           -0.66650390625F, -0.333251953125F,
                                                    0.333251953125F, 0.66650390625F,  1.0F};

__global__ void pool_kernel(const __half* rgb, __half* image, int height, int width) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) return;
  const int w2 = 2 * width;
  const long long plane2 = 4LL * height * width;
  const long long plane = static_cast<long long>(height) * width;
  for (int c = 0; c < kernels::kTfixImage; ++c) {
    const __half* p = rgb + c * plane2 + static_cast<long long>(2 * y) * w2 + 2 * x;
    const float sum = __half2float(p[0]) + __half2float(p[1]) + __half2float(p[w2]) + __half2float(p[w2 + 1]);
    image[c * plane + static_cast<long long>(y) * width + x] = __float2half_rn(0.25F * sum);
  }
}

// Le flou 5 × 5 (bords à zéro) d'un plan aux deux lignes y0, y0 + dy et deux colonnes x0, x0 + dx, puis le
// mélange bilinéaire des quatre : la lecture d'un point de niveau.
__device__ float blurred_sample(const __half* plane, int height, int width, const Axis& ax, const Axis& ay) {
  constexpr float kBinomial[5] = {1.0F / 16, 4.0F / 16, 6.0F / 16, 4.0F / 16, 1.0F / 16};
  float rows[6][2];
#pragma unroll
  for (int r = 0; r < 6; ++r) {
    const int y = ay.i0 - 2 + r;
    float patch[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      const int x = ax.i0 - 2 + c;
      patch[c] = (y >= 0 && y < height && x >= 0 && x < width) ? tfix::at(plane, y, x, width) : 0.0F;
    }
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      float sum = 0.0F;
#pragma unroll
      for (int k = 0; k < 5; ++k) sum += kBinomial[k] * patch[d + k];
      rows[r][d] = sum;
    }
  }
  float v[2][2];
#pragma unroll
  for (int d = 0; d < 2; ++d) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float sum = 0.0F;
#pragma unroll
      for (int k = 0; k < 5; ++k) sum += kBinomial[k] * rows[d + k][e];
      v[d][e] = sum;
    }
  }
  const int dx = ax.i1 - ax.i0;
  const int dy = ay.i1 - ay.i0;
  return warp::lerp2(v[0][0], v[0][dx], v[dy][0], v[dy][dx], ax, ay);
}

// Un fil par point d'un niveau (blockIdx.z), les sept canaux.
__global__ void pyramid_kernel(const __half* image, const __half* traits, __half* packed, int height,
                               int width, unsigned long long off0, unsigned long long off1,
                               unsigned long long off2) {
  const int level = static_cast<int>(blockIdx.z);
  const int s = 8 >> level;
  const int lh = (height + s - 1) / s;
  const int lw = (width + s - 1) / s;
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= lw || y >= lh) return;
  const Axis ax = tfix::resize_axis(x, width, lw);
  const Axis ay = tfix::resize_axis(y, height, lh);
  const unsigned long long offset = level == 0 ? off0 : (level == 1 ? off1 : off2);
  __half* out = packed + offset;
  const long long plane = static_cast<long long>(height) * width;
  const long long lplane = static_cast<long long>(lh) * lw;
  for (int c = 0; c < kernels::kTfixPyramid; ++c) {
    const __half* src =
        c < kernels::kTfixImage ? image + c * plane : traits + (c - kernels::kTfixImage) * plane;
    out[c * lplane + static_cast<long long>(y) * lw + x] =
        __float2half_rn(blurred_sample(src, height, width, ax, ay));
  }
}

// X0 : un fil par point du niveau 8 et par appui (blockIdx.z).
__global__ void first_kernel(TfixShape s) {
  const int k = static_cast<int>(blockIdx.z);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= s.lw[0] || y >= s.lh[0]) return;
  const long long plane = static_cast<long long>(s.lh[0]) * s.lw[0];
  const long long at = static_cast<long long>(y) * s.lw[0] + x;
  const __half* center = tfix::pyramid(s, kernels::kTfixCenter, 0);
  const __half* support = tfix::pyramid(s, kSupportFrame[k], 0);
  const tfix::View out = tfix::view(s, s.x[0], s.xld[0], plane, k);
  for (int c = 0; c < kernels::kTfixImage; ++c) {
    out(c, at) = center[c * plane + at];
    out(3 + c, at) = support[c * plane + at];
  }
  for (int c = 0; c < kernels::kTfixTraits; ++c) {
    out(6 + c, at) = center[(3 + c) * plane + at];
    out(10 + c, at) = support[(3 + c) * plane + at];
  }
  out(14, at) = __float2half_rn(kTime[k]);
  tfix::zero_pad(s, out, kernels::kTfixX0, s.xld[0], at);
}

// La sortie du bloc d'avant (son Resize vers son niveau, puis le Resize ×2 de la colle) au point de niveau
// (ay, ax) du niveau de départ : canal c de Y, lu aux quatre voisins.
struct Previous {
  tfix::View y;     // Y_{l-1} de cet appui, 11 canaux yh × yw
  tfix::View flow;  // X_{l-1} de cet appui (flot d'entrée du bloc d'avant : canaux 24 et 25)
  bool has_flow;    // faux au niveau 1 : le bloc 0 n'a pas de flot d'entrée
  int yh, yw;       // Y_{l-1}
  int sh, sw;       // le niveau de départ
};

// La valeur au point de départ (qy, qx) du canal c : Y agrandi, plus le flot d'entrée pour c < 2.
__device__ float previous_at(const Previous& p, int c, int qy, int qx) {
  const Axis bx = tfix::resize_axis(qx, p.yw, p.sw);
  const Axis by = tfix::resize_axis(qy, p.yh, p.sh);
  float v = tfix::resized(p.y, c, p.yw, bx, by);
  if (c < 2 && p.has_flow) v += p.flow.get(24 + c, static_cast<long long>(qy) * p.sw + qx);
  return v;
}

__device__ float previous_resized(const Previous& p, int c, const Axis& ax, const Axis& ay) {
  return warp::lerp2(previous_at(p, c, ay.i0, ax.i0), previous_at(p, c, ay.i0, ax.i1),
                     previous_at(p, c, ay.i1, ax.i0), previous_at(p, c, ay.i1, ax.i1), ax, ay);
}

// X_level (1 ou 2) : un fil par point du niveau et par appui (blockIdx.z).
__global__ void glue_kernel(TfixShape s, int level) {
  const int k = static_cast<int>(blockIdx.z);
  const int dh = s.lh[level];
  const int dw = s.lw[level];
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= dw || y >= dh) return;
  Previous p;
  p.yh = s.yh[level - 1];
  p.yw = s.yw[level - 1];
  p.sh = s.lh[level - 1];
  p.sw = s.lw[level - 1];
  const long long src_plane = static_cast<long long>(p.sh) * p.sw;
  p.y = tfix::view(s, s.y[level - 1], s.yld, static_cast<long long>(p.yh) * p.yw, k);
  // Le flot d'entrée du bloc 1 (X1 : 26 canaux) ; le bloc 0 n'en a pas.
  p.has_flow = level != 1;
  p.flow = tfix::view(s, s.x[1], s.xld[1], src_plane, k);
  const Axis ax = tfix::resize_axis(x, p.sw, dw);
  const Axis ay = tfix::resize_axis(y, p.sh, dh);

  const long long plane = static_cast<long long>(dh) * dw;
  const long long at = static_cast<long long>(y) * dw + x;
  const tfix::View out = tfix::view(s, s.x[level], s.xld[level], plane, k);
  // Le flot agrandi ×2 (Resize puis Mul 2 en FP16, comme le graphe), puis l'appui déplacé par lui.
  const __half fx = __float2half_rn(2.0F * previous_resized(p, 0, ax, ay));
  const __half fy = __float2half_rn(2.0F * previous_resized(p, 1, ax, ay));
  out(24, at) = fx;
  out(25, at) = fy;
  out(15, at) = __float2half_rn(previous_resized(p, 2, ax, ay));
  for (int c = 0; c < 8; ++c) out(16 + c, at) = __float2half_rn(previous_resized(p, 3 + c, ax, ay));

  const __half* center = tfix::pyramid(s, kernels::kTfixCenter, level);
  const __half* support = tfix::pyramid(s, kSupportFrame[k], level);
  const tfix::ZeroTap tap = tfix::zero_tap(static_cast<float>(x) + __half2float(fx),
                                           static_cast<float>(y) + __half2float(fy), dh, dw);
  for (int c = 0; c < kernels::kTfixPyramid; ++c) {
    const int to_center = c < kernels::kTfixImage ? c : 3 + c;      // 0-2 puis 6-9
    const int to_warped = c < kernels::kTfixImage ? 3 + c : 7 + c;  // 3-5 puis 10-13
    out(to_center, at) = center[c * plane + at];
    out(to_warped, at) = __float2half_rn(tfix::zero_sample(support + c * plane, tap));
  }
  out(14, at) = __float2half_rn(kTime[k]);
  tfix::zero_pad(s, out, kernels::kTfixX, s.xld[level], at);
}

dim3 grid_for(int width, int height, int z) {
  return {static_cast<unsigned>((width + kTile - 1) / kTile),
          static_cast<unsigned>((height + kTile - 1) / kTile), static_cast<unsigned>(z)};
}

}  // namespace

TfixShape tfix_shape(int height, int width, bool half, void* zone, bool native) {
  const kernels::TfixPacked packed = kernels::tfix_packed(height, width, half);
  const kernels::TfixLayout layout = kernels::tfix_layout(height, width, half, native);
  TfixShape s;
  s.height = height;
  s.width = width;
  s.half = half;
  s.image = packed.image;
  s.traits = packed.traits;
  for (int l = 0; l < kernels::kTfixLevels; ++l) {
    const auto i = static_cast<std::size_t>(l);
    s.pyramid[l] = packed.pyramid[i];
    s.x[l] = layout.x[i];
    s.y[l] = layout.y[i];
  }
  for (int l = 0; l < kernels::kTfixLevels; ++l) {
    const kernels::TfixDims level = kernels::tfix_level(l, height, width);
    const kernels::TfixDims y = kernels::tfix_block_y(l, height, width);
    s.lh[l] = level.height;
    s.lw[l] = level.width;
    s.yh[l] = y.height;
    s.yw[l] = y.width;
  }
  s.native = native;
  for (int b = 0; b < kernels::kTfixLevels; ++b)
    s.xld[b] = kernels::tfix_in_ld(b == 0 ? kernels::kTfixX0 : kernels::kTfixX, native);
  s.yld = kernels::tfix_out_ld(kernels::kTfixY, native);
  s.conf_ld = kernels::tfix_in_ld(kernels::kTfixConf, native);
  s.conf_out_ld = kernels::tfix_out_ld(1, native);
  s.conf_in = layout.conf_in;
  s.mask_half = layout.mask_half;
  s.conf_out = layout.conf_out;
  s.net_out = layout.net_out;
  s.zone = zone;
  s.table = reinterpret_cast<const TfixTable*>(static_cast<std::byte*>(zone) + layout.table * 2U);
  return s;
}

cudaError_t launch_tfix_pool(const void* rgb, void* packed, int height, int width, cudaStream_t stream) {
  const kernels::TfixPacked p = kernels::tfix_packed(height, width, true);
  pool_kernel<<<grid_for(width, height, 1), dim3(kTile, kTile), 0, stream>>>(
      static_cast<const __half*>(rgb), static_cast<__half*>(packed) + p.image, height, width);
  return cudaGetLastError();
}

cudaError_t launch_tfix_pyramid(const void* image, void* packed, int height, int width, bool half,
                                cudaStream_t stream) {
  const kernels::TfixPacked p = kernels::tfix_packed(height, width, half);
  auto* out = static_cast<__half*>(packed);
  const __half* source = half ? out + p.image : static_cast<const __half*>(image);
  const kernels::TfixDims l2 = kernels::tfix_level(2, height, width);
  pyramid_kernel<<<grid_for(l2.width, l2.height, kernels::kTfixLevels), dim3(kTile, kTile), 0, stream>>>(
      source, out + p.traits, out, height, width, p.pyramid[0], p.pyramid[1], p.pyramid[2]);
  return cudaGetLastError();
}

cudaError_t launch_tfix_first(const TfixShape& shape, cudaStream_t stream) {
  const kernels::TfixDims l = kernels::tfix_level(0, shape.height, shape.width);
  first_kernel<<<grid_for(l.width, l.height, kernels::kTfixSupports), dim3(kTile, kTile), 0, stream>>>(shape);
  return cudaGetLastError();
}

cudaError_t launch_tfix_glue(const TfixShape& shape, int level, cudaStream_t stream) {
  if (level < 1 || level > 2) return cudaErrorInvalidValue;
  const kernels::TfixDims l = kernels::tfix_level(level, shape.height, shape.width);
  glue_kernel<<<grid_for(l.width, l.height, kernels::kTfixSupports), dim3(kTile, kTile), 0, stream>>>(shape,
                                                                                                      level);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
