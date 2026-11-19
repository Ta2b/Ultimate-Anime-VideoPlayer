// TemporalFix fusionné (kernels/spec/tfix_fused.h) : après le bloc 2, l'entrée de la tête de confiance et le
// mélange final. Le flot du bloc 2 est agrandi jusqu'à l'image et chaque appui déplacé par lui deux fois,
// sans tampon pleine taille entre les deux : une fois par tuile en mémoire partagée pour la tête (son flou
// réduit à demi-résolution), une fois au pixel pour le mélange. Référence :
// tools/temporalfix/fused_reference.py.
#include "gpu/cuda/backend/tfix_fused.cuh"

namespace kirari::gpu::cuda {
namespace {

using tfix::Axis;
using tfix::half_ptr;

__constant__ int kFrame[kernels::kTfixSupports] = {0, 1, 2, 4, 5, 6};
__constant__ float kTimes[kernels::kTfixSupports] = {-1.0F,           -0.66650390625F, -0.333251953125F,
                                                     0.333251953125F, 0.66650390625F,  1.0F};

// Le bloc 2 d'un appui au point (qy, qx) du niveau 2 : son flot de sortie (flot d'entrée + Y2 agrandi) et
// son masque.
struct Last {
  const __half* y;     // Y2 de l'appui
  const __half* flow;  // X2 de l'appui, canaux 24 et 25
  int yh, yw, h2, w2;
};

__device__ Last last_of(const TfixShape& s, int k) {
  Last l;
  l.yh = s.yh[2];
  l.yw = s.yw[2];
  l.h2 = s.lh[2];
  l.w2 = s.lw[2];
  const long long plane = static_cast<long long>(l.h2) * l.w2;
  l.y = half_ptr(s.zone, s.y[2]) + static_cast<long long>(k) * kernels::kTfixY * l.yh * l.yw;
  l.flow = half_ptr(s.zone, s.x[2]) + (k * kernels::kTfixX + 24) * plane;
  return l;
}

// Flot x, flot y, masque au point (qy, qx) du niveau 2.
__device__ void last_at(const Last& l, int qy, int qx, float (&v)[3]) {
  const Axis bx = tfix::resize_axis(qx, l.yw, l.w2);
  const Axis by = tfix::resize_axis(qy, l.yh, l.h2);
  const long long yplane = static_cast<long long>(l.yh) * l.yw;
  const long long plane = static_cast<long long>(l.h2) * l.w2;
  for (int c = 0; c < 3; ++c) v[c] = tfix::resized(l.y + c * yplane, l.yw, bx, by);
  v[0] += tfix::at(l.flow, qy, qx, l.w2);
  v[1] += tfix::at(l.flow + plane, qy, qx, l.w2);
}

// La tête de confiance : une tuile de 16 × 16 points du niveau 2 par bloc, un appui par blockIdx.z.
constexpr int kConfTile = 16;
constexpr int kLowSide = kConfTile + 4;       // niveau 2 : la tuile, deux de plus de chaque côté
constexpr int kFullSide = 2 * kConfTile + 4;  // image : 2 × la tuile, deux de plus de chaque côté (flou)

__global__ void __launch_bounds__(kConfTile* kConfTile) conf_kernel(TfixShape s) {
  __shared__ float low[kLowSide][kLowSide][3];
  __shared__ float full[kFullSide][kFullSide][6];  // flot x, y, masque, appui déplacé (3)
  const int k = static_cast<int>(blockIdx.z);
  const Last l = last_of(s, k);
  const int tx0 = static_cast<int>(blockIdx.x) * kConfTile;
  const int ty0 = static_cast<int>(blockIdx.y) * kConfTile;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int threads = static_cast<int>(blockDim.x * blockDim.y);

  // 1. Le flot et le masque du bloc 2 sur la région du niveau 2 que lit la tuile.
  for (int i = tid; i < kLowSide * kLowSide; i += threads) {
    const int qy = ty0 - 2 + i / kLowSide;
    const int qx = tx0 - 2 + i % kLowSide;
    if (qy < 0 || qy >= l.h2 || qx < 0 || qx >= l.w2) continue;
    float v[3];
    last_at(l, qy, qx, v);
    for (int c = 0; c < 3; ++c) low[i / kLowSide][i % kLowSide][c] = v[c];
  }
  __syncthreads();

  // 2. À pleine taille : le flot agrandi ×2 (FP16 comme le graphe), le masque agrandi, l'appui déplacé ;
  // zéro hors de l'image (le bord du flou).
  const __half* image = tfix::net_image(s, kFrame[k]);
  const long long plane = static_cast<long long>(s.height) * s.width;
  for (int i = tid; i < kFullSide * kFullSide; i += threads) {
    const int py = 2 * ty0 - 2 + i / kFullSide;
    const int px = 2 * tx0 - 2 + i % kFullSide;
    float* cell = full[i / kFullSide][i % kFullSide];
    if (py < 0 || py >= s.height || px < 0 || px >= s.width) {
      for (int c = 0; c < 6; ++c) cell[c] = 0.0F;
      continue;
    }
    const Axis ax = tfix::resize_axis(px, l.w2, s.width);
    const Axis ay = tfix::resize_axis(py, l.h2, s.height);
    // Les voisins sont dans la région relevée (bornés au niveau, qui la contient).
    const int r0 = ay.i0 - (ty0 - 2);
    const int r1 = ay.i1 - (ty0 - 2);
    const int c0 = ax.i0 - (tx0 - 2);
    const int c1 = ax.i1 - (tx0 - 2);
    float v[3];
    for (int c = 0; c < 3; ++c)
      v[c] = warp::lerp2(low[r0][c0][c], low[r0][c1][c], low[r1][c0][c], low[r1][c1][c], ax, ay);
    const float fx = warp::rounded(2.0F * v[0]);
    const float fy = warp::rounded(2.0F * v[1]);
    cell[0] = fx;
    cell[1] = fy;
    cell[2] = warp::rounded(v[2]);
    const tfix::ZeroTap tap =
        tfix::zero_tap(static_cast<float>(px) + fx, static_cast<float>(py) + fy, s.height, s.width);
    for (int c = 0; c < kernels::kTfixImage; ++c)
      cell[3 + c] = warp::rounded(tfix::zero_sample(image + c * plane, tap));
  }
  __syncthreads();

  // 3. Un fil par point du niveau 2 : le flou 5 × 5 réduit ×2 des six plans, puis la Concat de la tête.
  const int x = tx0 + static_cast<int>(threadIdx.x);
  const int y = ty0 + static_cast<int>(threadIdx.y);
  if (x >= l.w2 || y >= l.h2) return;
  float down[6] = {};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      const float w = tfix::half_kernel(r) * tfix::half_kernel(c);
      const float* cell = full[2 * static_cast<int>(threadIdx.y) + r][2 * static_cast<int>(threadIdx.x) + c];
      for (int ch = 0; ch < 6; ++ch) down[ch] += w * cell[ch];
    }
  }
  const float hx = warp::rounded(0.5F * warp::rounded(down[0]));
  const float hy = warp::rounded(0.5F * warp::rounded(down[1]));
  const long long lplane = static_cast<long long>(l.h2) * l.w2;
  const long long at = static_cast<long long>(y) * l.w2 + x;
  const __half* center = tfix::pyramid(s, kernels::kTfixCenter, 2);
  const __half* support = tfix::pyramid(s, kFrame[k], 2);
  __half* out = half_ptr(s.zone, s.conf_in) + static_cast<long long>(k) * kernels::kTfixConf * lplane;
  for (int c = 0; c < kernels::kTfixImage; ++c) {
    const float ci = __half2float(center[c * lplane + at]);
    const float wi = warp::rounded(down[3 + c]);
    out[c * lplane + at] = center[c * lplane + at];
    out[(3 + c) * lplane + at] = __float2half_rn(wi);
    out[(6 + c) * lplane + at] = __float2half_rn(fabsf(wi - ci));
  }
  const tfix::ZeroTap tap =
      tfix::zero_tap(static_cast<float>(x) + hx, static_cast<float>(y) + hy, l.h2, l.w2);
  for (int c = 0; c < kernels::kTfixTraits; ++c) {
    const long long from = (kernels::kTfixImage + c) * lplane;
    const float cf = __half2float(center[from + at]);
    const float wf = warp::rounded(tfix::zero_sample(support + from, tap));
    out[(9 + c) * lplane + at] = center[from + at];
    out[(13 + c) * lplane + at] = __float2half_rn(wf);
    out[(17 + c) * lplane + at] = __float2half_rn(fabsf(wf - cf));
  }
  out[21 * lplane + at] = __float2half_rn(hx);
  out[22 * lplane + at] = __float2half_rn(hy);
  out[23 * lplane + at] = __float2half_rn(sqrtf(hx * hx + hy * hy));
  out[24 * lplane + at] = __float2half_rn(kTimes[k]);
  half_ptr(s.zone, s.mask_half)[k * lplane + at] = __float2half_rn(down[2]);
}

// Le mélange : une tuile de 32 × 8 pixels du réseau par bloc ; le flot du bloc 2 et la confiance (sortie de
// la tête plus le masque moyenné) des six appuis relevés d'abord sur la région du niveau 2 qu'elle lit.
constexpr int kBlendW = 32;
constexpr int kBlendH = 8;
constexpr int kBlendLowW = kBlendW / 2 + 3;
constexpr int kBlendLowH = kBlendH / 2 + 3;

// net_out : le tampon du réseau en demi-taille ; nul, la sortie de la table.
__global__ void __launch_bounds__(kBlendW* kBlendH) blend_kernel(TfixShape s, __half* net_out) {
  __shared__ float low[kernels::kTfixSupports][kBlendLowH][kBlendLowW][3];
  const int px0 = static_cast<int>(blockIdx.x) * kBlendW;
  const int py0 = static_cast<int>(blockIdx.y) * kBlendH;
  const int qx0 = px0 / 2 - 1;
  const int qy0 = py0 / 2 - 1;
  const int h2 = s.lh[2];
  const int w2 = s.lw[2];
  const long long lplane = static_cast<long long>(h2) * w2;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int threads = static_cast<int>(blockDim.x * blockDim.y);
  const __half* conf = half_ptr(s.zone, s.conf_out);
  const __half* mask = half_ptr(s.zone, s.mask_half);
  for (int i = tid; i < kernels::kTfixSupports * kBlendLowH * kBlendLowW; i += threads) {
    const int k = i / (kBlendLowH * kBlendLowW);
    const int r = (i / kBlendLowW) % kBlendLowH;
    const int c = i % kBlendLowW;
    const int qy = qy0 + r;
    const int qx = qx0 + c;
    if (qy < 0 || qy >= h2 || qx < 0 || qx >= w2) continue;
    float v[3];
    last_at(last_of(s, k), qy, qx, v);
    const long long at = k * lplane + static_cast<long long>(qy) * w2 + qx;
    low[k][r][c][0] = v[0];
    low[k][r][c][1] = v[1];
    low[k][r][c][2] = __half2float(conf[at]) + __half2float(mask[at]);
  }
  __syncthreads();
  const int x = px0 + static_cast<int>(threadIdx.x);
  const int y = py0 + static_cast<int>(threadIdx.y);
  if (x >= s.width || y >= s.height) return;
  const Axis ax = tfix::resize_axis(x, w2, s.width);
  const Axis ay = tfix::resize_axis(y, h2, s.height);
  const int r0 = ay.i0 - qy0;
  const int r1 = ay.i1 - qy0;
  const int c0 = ax.i0 - qx0;
  const int c1 = ax.i1 - qx0;
  const long long plane = static_cast<long long>(s.height) * s.width;
  const long long at = static_cast<long long>(y) * s.width + x;
  const __half* center = tfix::net_image(s, kernels::kTfixCenter);
  __half* out = net_out != nullptr ? net_out : half_ptr(s.table->out);
  float num[3] = {};
  float sum_a = 0.0F;
  float sum_s = 0.0F;
  for (int k = 0; k < kernels::kTfixSupports; ++k) {
    float v[3];
    for (int c = 0; c < 3; ++c)
      v[c] = warp::lerp2(low[k][r0][c0][c], low[k][r0][c1][c], low[k][r1][c0][c], low[k][r1][c1][c], ax, ay);
    const float fx = warp::rounded(2.0F * v[0]);
    const float fy = warp::rounded(2.0F * v[1]);
    const tfix::ZeroTap tap =
        tfix::zero_tap(static_cast<float>(x) + fx, static_cast<float>(y) + fy, s.height, s.width);
    const float w = tfix::sigmoid(v[2]) * tap.valid;
    const float sk = tfix::sigmoid((w - 0.60009765625F) * 12.0F);
    const float a = w * sk;
    const __half* image = tfix::net_image(s, kFrame[k]);
    for (int c = 0; c < kernels::kTfixImage; ++c)
      num[c] += a * warp::rounded(tfix::zero_sample(image + c * plane, tap));
    sum_a += a;
    sum_s += sk;
  }
  const float den = fmaxf(sum_a + 1.0F, 1.0132789611816406e-06F);
  const float g = tfix::sigmoid((sum_s - 0.5F) * 4.0F);
  for (int c = 0; c < kernels::kTfixImage; ++c) {
    const float i = __half2float(center[c * plane + at]);
    out[c * plane + at] = __float2half_rn(i + g * ((i + num[c]) / den - i));
  }
}

// Demi-taille : l'image de travail du milieu plus l'écart du réseau à son image moyennée, agrandi ×2
// (Resize linéaire demi-pixel, échelle 2).
__global__ void half_out_kernel(TfixShape s) {
  const int wide = 2 * s.width;
  const int tall = 2 * s.height;
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= wide || y >= tall) return;
  const Axis ax = warp::coarse_axis(x, 0.5F, s.width);
  const Axis ay = warp::coarse_axis(y, 0.5F, s.height);
  const long long plane = static_cast<long long>(s.height) * s.width;
  const long long big = static_cast<long long>(tall) * wide;
  const __half* net = half_ptr(s.zone, s.net_out);
  const __half* pooled = tfix::net_image(s, kernels::kTfixCenter);
  const __half* rgb = half_ptr(s.table->rgb[kernels::kTfixCenter]);
  __half* out = half_ptr(s.table->out);
  for (int c = 0; c < kernels::kTfixImage; ++c) {
    const auto delta = [&](int yy, int xx) {
      const long long i = c * plane + static_cast<long long>(yy) * s.width + xx;
      return __half2float(net[i]) - __half2float(pooled[i]);
    };
    const float d = warp::lerp2(delta(ay.i0, ax.i0), delta(ay.i0, ax.i1), delta(ay.i1, ax.i0),
                                delta(ay.i1, ax.i1), ax, ay);
    const long long at = c * big + static_cast<long long>(y) * wide + x;
    out[at] = __float2half_rn(__half2float(rgb[at]) + d);
  }
}

dim3 grid_for(int width, int height, int tile_w, int tile_h, int z) {
  return {static_cast<unsigned>((width + tile_w - 1) / tile_w),
          static_cast<unsigned>((height + tile_h - 1) / tile_h), static_cast<unsigned>(z)};
}

}  // namespace

cudaError_t launch_tfix_conf(const TfixShape& shape, cudaStream_t stream) {
  conf_kernel<<<grid_for(shape.lw[2], shape.lh[2], kConfTile, kConfTile, kernels::kTfixSupports),
                dim3(kConfTile, kConfTile), 0, stream>>>(shape);
  return cudaGetLastError();
}

cudaError_t launch_tfix_blend(const TfixShape& shape, cudaStream_t stream) {
  // La sortie : l'image de travail, ou en demi-taille le tampon du réseau, reporté ensuite.
  __half* out = shape.half ? static_cast<__half*>(shape.zone) + shape.net_out : nullptr;
  blend_kernel<<<grid_for(shape.width, shape.height, kBlendW, kBlendH, 1), dim3(kBlendW, kBlendH), 0,
                 stream>>>(shape, out);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess || !shape.half) return e;
  half_out_kernel<<<grid_for(2 * shape.width, 2 * shape.height, 16, 16, 1), dim3(16, 16), 0, stream>>>(shape);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
