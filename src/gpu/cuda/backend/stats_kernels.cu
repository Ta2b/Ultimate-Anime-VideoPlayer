// Portage CUDA de frame_stats.comp (stats_kernels.h) : mêmes points, mêmes cases, mêmes sommes ;
// seules les sommes flottantes s'additionnent dans un autre ordre.
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/stats_kernels.h"
#include "kernels/spec/color.h"
#include "kernels/spec/frame_stats.h"
#include "kernels/spec/transfer.h"

namespace ajp::gpu::cuda {
namespace {

namespace fs = kernels::frame_stats;

constexpr int kThreads = 256;
constexpr unsigned kGw = fs::kGridW;
constexpr unsigned kGh = fs::kGridH;
constexpr unsigned kLumaBins = fs::kLumaBins;
constexpr unsigned kLinearBins = fs::kLinearBins;
constexpr unsigned kZoneSide = fs::kZones;
constexpr unsigned kZoneCount = kZoneSide * kZoneSide;
// Places dans FrameCounts, en mots de 4 octets.
constexpr unsigned kAtLuma = 0;
constexpr unsigned kAtMaxrgb = kAtLuma + kLumaBins;
constexpr unsigned kAtI = kAtMaxrgb + kLinearBins;
constexpr unsigned kAtZoneSum = kAtI + kLinearBins;
constexpr unsigned kAtZoneCount = kAtZoneSum + kZoneCount;
constexpr unsigned kAtPoints = kAtZoneCount + kZoneCount;
constexpr unsigned kAtSums = kAtPoints + 4;
constexpr unsigned kAtFrame = kAtSums + 5;
constexpr unsigned kSums = 5 + kZoneCount;
static_assert((kAtFrame + 4) * 4 == sizeof(kernels::FrameCounts));

constexpr float kPqM1 = kernels::pq::kM1;
constexpr float kPqM2 = kernels::pq::kM2;
constexpr float kPqC1 = kernels::pq::kC1;
constexpr float kPqC2 = kernels::pq::kC2;
constexpr float kPqC3 = kernels::pq::kC3;
constexpr float kPqPeak = kernels::pq::kPeakNits;
constexpr float kHlgA = kernels::hlg::kA;
constexpr float kHlgB = kernels::hlg::kB;
constexpr float kHlgC = kernels::hlg::kC;
constexpr float kHlgGamma = kernels::hlg::kGammaAt1000;
constexpr float kBt1886Gamma = kernels::bt1886::kGamma;
constexpr float kKr709 = kernels::coefficients(kernels::Matrix::Bt709).kr;
constexpr float kKb709 = kernels::coefficients(kernels::Matrix::Bt709).kb;
constexpr float kKr2020 = kernels::coefficients(kernels::Matrix::Bt2020Ncl).kr;
constexpr float kKb2020 = kernels::coefficients(kernels::Matrix::Bt2020Ncl).kb;
#define AJP_M(r, c) static_cast<float>(kernels::kBt709ToBt2020.at(r, c))
constexpr float kM00 = AJP_M(0, 0), kM01 = AJP_M(0, 1), kM02 = AJP_M(0, 2);
constexpr float kM10 = AJP_M(1, 0), kM11 = AJP_M(1, 1), kM12 = AJP_M(1, 2);
constexpr float kM20 = AJP_M(2, 0), kM21 = AJP_M(2, 1), kM22 = AJP_M(2, 2);
#undef AJP_M

struct Rgb {
  float r, g, b;
};

__device__ float unit(float v) {
  return v > 0.0F ? fminf(v, 1.0F) : 0.0F;  // NaN compris
}

__device__ float sample(const FrameStatsParams& p, unsigned i) {
  if (p.fp16) return __half2float(static_cast<const __half*>(p.rgb)[i]);
  return static_cast<const float*>(p.rgb)[i];
}

__device__ Rgb encoded(const FrameStatsParams& p, unsigned gx, unsigned gy) {
  const unsigned w = static_cast<unsigned>(p.width);
  const unsigned h = static_cast<unsigned>(p.height);
  const unsigned sx = min((2U * gx + 1U) * w / (2U * kGw), w - 1U);
  const unsigned sy = min((2U * gy + 1U) * h / (2U * kGh), h - 1U);
  const unsigned plane = w * h;
  const unsigned i = sy * w + sx;
  return {unit(sample(p, i)), unit(sample(p, plane + i)), unit(sample(p, 2U * plane + i))};
}

__device__ float luma_of(Rgb e) {
  return __fadd_rn(__fadd_rn(__fmul_rn(0.2126F, e.r), __fmul_rn(0.7152F, e.g)), __fmul_rn(0.0722F, e.b));
}

__device__ float pq_eotf(float signal) {
  const float p = powf(fminf(fmaxf(signal, 0.0F), 1.0F), 1.0F / kPqM2);
  return powf(fmaxf(p - kPqC1, 0.0F) / (kPqC2 - kPqC3 * p), 1.0F / kPqM1) * kPqPeak;
}

__device__ float pq_inverse_eotf(float nits) {
  const float ym = powf(fminf(fmaxf(nits, 0.0F), kPqPeak) / kPqPeak, kPqM1);
  return powf((kPqC1 + kPqC2 * ym) / (1.0F + kPqC3 * ym), kPqM2);
}

__device__ float hlg_inverse_oetf(float v) {
  if (v <= 0.5F) return v * v / 3.0F;
  return (expf((v - kHlgC) / kHlgA) + kHlgB) / 12.0F;
}

// frame_nits : le signal en cd/m², dans les primaires de la source.
__device__ Rgb nits_of(Rgb e, int transfer) {
  if (transfer == 1) return {pq_eotf(e.r), pq_eotf(e.g), pq_eotf(e.b)};
  if (transfer == 2) {
    const Rgb s{hlg_inverse_oetf(e.r), hlg_inverse_oetf(e.g), hlg_inverse_oetf(e.b)};
    const float ys = kKr2020 * s.r + (1.0F - kKr2020 - kKb2020) * s.g + kKb2020 * s.b;
    const float gain = ys > 0.0F ? fs::kHlgPeakNits * powf(ys, kHlgGamma - 1.0F) : 0.0F;
    return {gain * s.r, gain * s.g, gain * s.b};
  }
  return {powf(e.r, kBt1886Gamma) * fs::kSdrWhiteNits, powf(e.g, kBt1886Gamma) * fs::kSdrWhiteNits,
          powf(e.b, kBt1886Gamma) * fs::kSdrWhiteNits};
}

__device__ float maxrgb_of(Rgb n) {
  return fmaxf(fmaxf(n.r, n.g), n.b);
}

__device__ float ictcp_i(Rgb n) {
  const float l = (1688.0F * n.r + 2146.0F * n.g + 262.0F * n.b) / 4096.0F;
  const float m = (683.0F * n.r + 2951.0F * n.g + 462.0F * n.b) / 4096.0F;
  return 0.5F * pq_inverse_eotf(l) + 0.5F * pq_inverse_eotf(m);
}

__device__ unsigned bin_of(float v, unsigned bins) {
  return min(static_cast<unsigned>(unit(v) * static_cast<float>(bins)), bins - 1U);
}

__device__ bool is_skin(Rgb e) {
  if (!(e.r > e.g && e.g > e.b && e.r >= fs::kSkinMinRed)) return false;
  const float span = e.r - e.b;
  const float hue = (e.g - e.b) / span;
  const float sat = span / e.r;
  return hue >= fs::kSkinHueLo && hue <= fs::kSkinHueHi && sat >= fs::kSkinSatLo && sat <= fs::kSkinSatHi;
}

// Le point (gx, gy) du cadre [x0, x1) × [y0, y1) : clair, compté dans tallies[1], et nettement plus clair
// que ses huit voisins (bornés au cadre), dans tallies[2] ; les mêmes valeurs, dans le même ordre, que
// lorsque chaque voisin se recalculait.
__device__ void count_bright_at(const float* grid, unsigned gx, unsigned gy, unsigned x0, unsigned y0,
                                unsigned x1, unsigned y1, unsigned* tallies) {
  const float maxrgb = grid[gy * kGw + gx];
  if (!(maxrgb > fs::kBrightNits)) return;
  atomicAdd(&tallies[1], 1U);
  float around = 0.0F;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      const int nx = min(max(static_cast<int>(gx) + dx, static_cast<int>(x0)), static_cast<int>(x1) - 1);
      const int ny = min(max(static_cast<int>(gy) + dy, static_cast<int>(y0)), static_cast<int>(y1) - 1);
      around += grid[static_cast<unsigned>(ny) * kGw + static_cast<unsigned>(nx)];
    }
  }
  if (maxrgb >= fs::kIsolatedRatio * (around / 8.0F)) atomicAdd(&tallies[2], 1U);
}

// Le travail de la grille : un point par fil, sur kBlocks blocs, au lieu d'un seul bloc qui la parcourait
// seul (un SM sur toute la carte, A32). Deux passes : la première trace le cadre (lumas maximales par
// colonne et par ligne) et garde le maxRGB de chaque point ; la seconde compte et somme dans le cadre.
constexpr unsigned kPoints = kGw * kGh;
constexpr unsigned kBlocks = (kPoints + kThreads - 1U) / kThreads;

// La mémoire de travail, après les comptes (FrameCounts) dans le même tampon d'appareil.
struct Scratch {
  unsigned column_max[kGw];
  unsigned row_max[kGh];
  float maxrgb_grid[kPoints];  // le maxRGB en cd/m² de chaque point (C01-04)
};
constexpr std::size_t kScratchAt = (sizeof(kernels::FrameCounts) + 255U) / 256U * 256U;

__device__ Scratch& scratch_of(const FrameStatsParams& p) {
  return *reinterpret_cast<Scratch*>(static_cast<std::byte*>(p.counts) + kScratchAt);
}

// Première passe : luma de chaque colonne et ligne (un float positif se compare comme ses bits), et le
// maxRGB de chaque point, que la recherche des points clairs isolés relit avec ses huit voisins.
__global__ void __launch_bounds__(kThreads) frame_stats_trace(FrameStatsParams p) {
  const unsigned q = blockIdx.x * kThreads + threadIdx.x;
  if (q >= kPoints) return;
  Scratch& scratch = scratch_of(p);
  const unsigned gx = q % kGw;
  const unsigned gy = q / kGw;
  const Rgb e = encoded(p, gx, gy);
  const unsigned bits = __float_as_uint(luma_of(e));
  atomicMax(&scratch.column_max[gx], bits);
  atomicMax(&scratch.row_max[gy], bits);
  scratch.maxrgb_grid[q] = maxrgb_of(nits_of(e, p.transfer));
}

// Le cadre sans les bandes noires, d'après les lumas maximales : chaque bloc le recalcule (deux cents
// lectures), le premier l'écrit.
__device__ void frame_bounds(const Scratch& scratch, unsigned* frame) {
  unsigned x0 = 0U, x1 = kGw, y0 = 0U, y1 = kGh;
  while (x0 < kGw && __uint_as_float(scratch.column_max[x0]) < fs::kBarLuma) ++x0;
  while (x1 > x0 && __uint_as_float(scratch.column_max[x1 - 1U]) < fs::kBarLuma) --x1;
  while (y0 < kGh && __uint_as_float(scratch.row_max[y0]) < fs::kBarLuma) ++y0;
  while (y1 > y0 && __uint_as_float(scratch.row_max[y1 - 1U]) < fs::kBarLuma) --y1;
  if (x1 - x0 < kGw / 4U || y1 - y0 < kGh / 4U) {
    x0 = 0U, x1 = kGw, y0 = 0U, y1 = kGh;
  }
  frame[0] = x0, frame[1] = x1, frame[2] = y0, frame[3] = y1;
}

// Seconde passe : le point de rang q du cadre pour chaque fil ; histogrammes et comptes du bloc en mémoire
// partagée, puis ajoutés aux comptes de l'image (remis à zéro avant la première passe).
__global__ void __launch_bounds__(kThreads) frame_stats_count(FrameStatsParams p) {
  __shared__ unsigned frame[4];
  __shared__ unsigned hist_luma[kLumaBins];
  __shared__ unsigned hist_maxrgb[kLinearBins];
  __shared__ unsigned hist_i[kLinearBins];
  __shared__ unsigned zone_count[kZoneCount];
  __shared__ unsigned tallies[4];
  __shared__ float reduce[kThreads];
  const Scratch& scratch = scratch_of(p);
  const unsigned tid = threadIdx.x;
  for (unsigned i = tid; i < kLumaBins; i += kThreads) hist_luma[i] = 0U;
  for (unsigned i = tid; i < kLinearBins; i += kThreads) {
    hist_maxrgb[i] = 0U;
    hist_i[i] = 0U;
  }
  if (tid < kZoneCount) zone_count[tid] = 0U;
  if (tid < 4U) tallies[tid] = 0U;
  if (tid == 0U) frame_bounds(scratch, frame);
  __syncthreads();
  const unsigned x0 = frame[0], x1 = frame[1], y0 = frame[2], y1 = frame[3];
  const unsigned fw = x1 - x0, fh = y1 - y0;
  const bool sdr = p.transfer == 0;
  const float kr = sdr ? kKr709 : kKr2020;
  const float kb = sdr ? kKb709 : kKb2020;
  float sums[kSums];
#pragma unroll
  for (unsigned s = 0U; s < kSums; ++s) sums[s] = 0.0F;
  const unsigned q = blockIdx.x * kThreads + tid;
  if (q < fw * fh) {
    const unsigned gx = x0 + q % fw;
    const unsigned gy = y0 + q / fw;
    const Rgb e = encoded(p, gx, gy);
    const float luma = luma_of(e);
    const Rgb n = nits_of(e, p.transfer);
    const float maxrgb = maxrgb_of(n);
    const Rgb n2020 = sdr ? Rgb{kM00 * n.r + kM01 * n.g + kM02 * n.b, kM10 * n.r + kM11 * n.g + kM12 * n.b,
                                kM20 * n.r + kM21 * n.g + kM22 * n.b}
                          : n;
    atomicAdd(&hist_luma[bin_of(luma, kLumaBins)], 1U);
    const float code = p.transfer == 1 ? fmaxf(fmaxf(e.r, e.g), e.b) : pq_inverse_eotf(maxrgb);
    atomicAdd(&hist_maxrgb[bin_of(code, kLinearBins)], 1U);
    atomicAdd(&hist_i[bin_of(ictcp_i(n2020), kLinearBins)], 1U);
    const unsigned zone = ((gy - y0) * kZoneSide / fh) * kZoneSide + (gx - x0) * kZoneSide / fw;
    atomicAdd(&zone_count[zone], 1U);
    // Indice variable : la somme de la zone passe par une boucle déroulée (registres, pas de pile).
#pragma unroll
    for (unsigned z = 0U; z < kZoneCount; ++z) {
      if (z == zone) sums[5U + z] += luma;
    }
    atomicAdd(&tallies[0], 1U);
    if (is_skin(e)) atomicAdd(&tallies[3], 1U);
    sums[0] += luma;
    sums[1] += fmaxf(0.0F, e.b - fmaxf(e.r, e.g));
    sums[2] += fmaxf(fmaxf(e.r, e.g), e.b) - fminf(fminf(e.r, e.g), e.b);
    sums[3] += kr * n.r + (1.0F - kr - kb) * n.g + kb * n.b;
    sums[4] += maxrgb;
    count_bright_at(scratch.maxrgb_grid, gx, gy, x0, y0, x1, y1, tallies);
  }
  // Somme de chaque compte sur le bloc, par arbre en mémoire partagée, puis sur l'image.
  auto* out = static_cast<unsigned*>(p.counts);
  auto* out_sums = reinterpret_cast<float*>(out);
#pragma unroll
  for (unsigned s = 0U; s < kSums; ++s) {
    reduce[tid] = sums[s];
    __syncthreads();
    for (unsigned step = kThreads / 2U; step > 0U; step >>= 1U) {
      if (tid < step) reduce[tid] += reduce[tid + step];
      __syncthreads();
    }
    if (tid == 0U && reduce[0] != 0.0F)
      atomicAdd(&out_sums[s < 5U ? kAtSums + s : kAtZoneSum + (s - 5U)], reduce[0]);
    __syncthreads();
  }
  const auto add = [](unsigned* to, unsigned v) {
    if (v != 0U) atomicAdd(to, v);
  };
  for (unsigned i = tid; i < kLumaBins; i += kThreads) add(&out[kAtLuma + i], hist_luma[i]);
  for (unsigned i = tid; i < kLinearBins; i += kThreads) {
    add(&out[kAtMaxrgb + i], hist_maxrgb[i]);
    add(&out[kAtI + i], hist_i[i]);
  }
  if (tid < kZoneCount) add(&out[kAtZoneCount + tid], zone_count[tid]);
  if (tid < 4U) add(&out[kAtPoints + tid], tallies[tid]);
  if (blockIdx.x == 0U && tid < 4U) out[kAtFrame + tid] = frame[tid];
}

}  // namespace

std::size_t frame_stats_device_bytes() {
  return kScratchAt + sizeof(Scratch);
}

cudaError_t launch_frame_stats(const FrameStatsParams& params, cudaStream_t stream) {
  // Comptes, sommes et lumas maximales repartent de zéro (la grille des maxRGB est réécrite en entier).
  const std::size_t zeroed = kScratchAt + offsetof(Scratch, maxrgb_grid);
  if (cudaError_t e = cudaMemsetAsync(params.counts, 0, zeroed, stream); e != cudaSuccess) return e;
  frame_stats_trace<<<kBlocks, kThreads, 0, stream>>>(params);
  frame_stats_count<<<kBlocks, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
