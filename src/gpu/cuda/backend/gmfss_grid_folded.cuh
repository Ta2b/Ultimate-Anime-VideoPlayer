// Les deux dernières convolutions du GridNet sont linéaires. Après PixelShuffle(2), chaque phase RGB
// se calcule directement par une convolution 4 × 4 au niveau 0. On supprime l'arrondi intermédiaire
// en FP16 et on range le filtre combiné en FP16. La somme des produits reste en FP32. Les quatre bords
// prennent des filtres distincts au bord de l'image haute résolution, avec leurs biais propres.
#pragma once

#include <array>
#include <vector>

#include "gpu/cuda/backend/gmfss_grid_rgb.cuh"
#include "prepare/gmfss_grid.h"

namespace kirari::gpu::cuda::gmfss_grid_folded {
namespace rn = rife_native;

constexpr int kC = 64, kTaps = 16, kSteps = 4;
constexpr int kChannels = 3, kPhases = 4, kBorders = 4;
constexpr int kChannelSlice = 32, kPhasePair = 2;
constexpr int kWeightPhase = kTaps * kSteps * 32;

// Les poids des deux convolutions proviennent du même ONNX que le plan de repli. Les sommes de la
// composition se font en FP64 avant le rangement des fragments B du MMA m16n8k16, puis en FP32
// pour les bords. Le zéro au-delà du PixelShuffle exclut aussi le biais de la première convolution.
struct Packed {
  std::vector<uint2> interior;
  std::vector<float> edges;
  std::array<float, kPhases * kBorders * kChannels> bias{};
};

inline Packed pack(const prepare::GmfssGridConv& up, const prepare::GmfssGridConv& rgb) {
  const auto row = [](int phase, int border, int channel) {
    return (phase * kBorders + border) * kChannels + channel;
  };
  const auto input = [](int out, int tap, int in) { return (out * 9 + tap) * kC + in; };
  std::vector<double> combined(kPhases * kBorders * kChannels * kTaps * kC, 0.);
  Packed result;
  for (int phase = 0; phase < kPhases; ++phase) {
    const int py = phase / 2, px = phase % 2;
    for (int border = 0; border < kBorders; ++border) {
      for (int c = 0; c < kChannels; ++c) {
        const auto at = row(phase, border, c);
        double bias = rgb.bias[c];
        for (int dy = -1; dy <= 1; ++dy) {
          if ((border & 1) != 0 && dy == 2 * py - 1) continue;
          const int y = py + dy + 2;
          for (int dx = -1; dx <= 1; ++dx) {
            if ((border & 2) != 0 && dx == 2 * px - 1) continue;
            const int x = px + dx + 2, subphase = 2 * (y % 2) + x % 2;
            const int tap = (dy + 1) * 3 + dx + 1;
            for (int feature = 0; feature < kC; ++feature) {
              const int shuffled = feature * kPhases + subphase;
              const double gain = rgb.weights[input(c, tap, feature)];
              bias += gain * up.bias[shuffled];
              for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                  const int out_tap = (y / 2 + ky - py) * 4 + x / 2 + kx - px;
                  double* target = combined.data() + (at * kTaps + out_tap) * kC;
                  const float* source = up.weights.data() + input(shuffled, ky * 3 + kx, 0);
                  for (int ch = 0; ch < kC; ++ch) target[ch] += gain * source[ch];
                }
              }
            }
          }
        }
        result.bias[at] = static_cast<float>(bias);
      }
    }
  }
  result.edges.assign(combined.begin(), combined.end());
  result.interior.reserve(kPhases * kWeightPhase);
  for (int phase = 0; phase < kPhases; ++phase) {
    for (int tap = 0; tap < kTaps; ++tap) {
      for (int step = 0; step < kSteps; ++step) {
        for (int lane = 0; lane < 32; ++lane) {
          const int out = lane / 4, first = step * 16 + 2 * (lane % 4);
          const auto half_bits = [&](int channel) {
            const double v =
                out < kChannels ? combined[(row(phase, 0, out) * kTaps + tap) * kC + channel] : 0.;
            return static_cast<unsigned>(__half_as_ushort(__float2half_rn(static_cast<float>(v))));
          };
          result.interior.push_back(make_uint2(half_bits(first) | (half_bits(first + 1) << 16),
                                               half_bits(first + 8) | (half_bits(first + 9) << 16)));
        }
      }
    }
  }
  return result;
}

struct Params {
  const __half* input;
  __half* output;
  const uint2* weights;
  const float* bias;
  const float* edges;
  int height, width;
};

__global__ void __launch_bounds__(rn::kThreads, 2) conv(Params p) {
  constexpr int tile_rows = rn::kWarps, load_rows = tile_rows + 4;
  constexpr int load_cols = rn::kTW + 4, pitch = kChannelSlice + 8;
  constexpr int fragments = 2, kSub = kChannelSlice / 16;
  constexpr auto precision = rn::Acc::kF32;
  extern __shared__ __align__(16) unsigned char bytes[];
  auto* storage = reinterpret_cast<__half*>(bytes);
  const int thread = static_cast<int>(threadIdx.x), warp = thread / 32, lane = thread % 32;
  const int x0 = static_cast<int>(blockIdx.x) * rn::kTW, y0 = static_cast<int>(blockIdx.y) * tile_rows;
  const int phase0 = static_cast<int>(blockIdx.z) * kPhasePair;
  const rn::Params source{.in = p.input, .in_ld = kC, .ih = p.height, .iw = p.width};
  rn::Frag<precision> acc[kPhasePair][fragments];
#pragma unroll
  for (int phase = 0; phase < kPhasePair; ++phase)
#pragma unroll
    for (int f = 0; f < fragments; ++f) acc[phase][f].clear();

  const auto base =
      rn::smem_u32(storage) + static_cast<unsigned>(((lane & 15) * pitch + (lane >> 4) * 8) * 2);
#pragma unroll 1
  for (int slice = 0; slice < kC / kChannelSlice; ++slice) {
    if (slice != 0) __syncthreads();
    rn::load_nhwc<kChannelSlice, 1, load_rows, load_cols>(storage, source, slice * kChannelSlice, y0 - 2,
                                                          x0 - 2);
    rn::cp_async_wait_all();
    __syncthreads();
#pragma unroll
    for (int part = 0; part < kPhasePair; ++part) {
      const int phase = phase0 + part;
      const int phase_y = phase >> 1, phase_x = phase & 1;
      const uint2* w = p.weights + phase * kWeightPhase + slice * kSub * 32 + lane;
#pragma unroll 1
      for (int tap = 0; tap < kTaps; ++tap) {
        const int dy = tap / 4 + phase_y, dx = tap % 4 + phase_x;
#pragma unroll
        for (int sub = 0; sub < kSub; ++sub) {
          const int weight_at = (tap * kSteps + sub) * 32;
          const uint2 weight = __ldg(w + weight_at);
#pragma unroll
          for (int f = 0; f < fragments; ++f) {
            const int row = warp + dy, col = 16 * (f & 1) + dx;
            const auto addr = base + static_cast<unsigned>(((row * load_cols + col) * pitch + sub * 16) * 2);
            std::uint32_t a[4];
            rn::ldmatrix4(a, addr);
            acc[part][f].mma(a, weight.x, weight.y);
          }
        }
      }
    }
  }
  const int c = 2 * (lane & 3);
  if (c >= 3) return;
  const int full_width = 2 * p.width;
  const long out_plane = static_cast<long>(2 * p.height) * full_width;
#pragma unroll
  for (int part = 0; part < kPhasePair; ++part) {
    const int phase = phase0 + part;
    const float bias0 = __ldg(p.bias + phase * kBorders * kChannels + c);
    const float bias1 = c == 0 ? __ldg(p.bias + phase * kBorders * kChannels + 1) : 0.F;
#pragma unroll
    for (int f = 0; f < fragments; ++f) {
      const int y = y0 + warp, x_base = x0 + 16 * (f & 1) + (lane >> 2);
      if (y >= p.height) continue;
      const long out_row = static_cast<long>(2 * y + (phase >> 1)) * full_width;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = x_base + 8 * h;
        if (x >= p.width) continue;
        const long at = out_row + 2 * x + (phase & 1);
        const float2 v = acc[part][f].pair(h);
        p.output[c * out_plane + at] = gmfss_grid_rgb::clamp_half(__float2half_rn(v.x + bias0));
        if (c == 0) p.output[out_plane + at] = gmfss_grid_rgb::clamp_half(__float2half_rn(v.y + bias1));
      }
    }
  }
}

// L'intérieur résulte de la composition sans couper la première convolution aux bords.
// Ici les échantillons au-delà de l'image haute résolution ont déjà été exclus des coefficients.
__global__ void edges(Params p) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int point = static_cast<int>(blockIdx.x) * 4 + static_cast<int>(threadIdx.x) / 32;
  const int w = 2 * p.width, h = 2 * p.height;
  if (point >= 2 * w + 2 * (h - 2)) return;
  const bool horizontal = point < 2 * w;
  const int x = horizontal ? point % w : ((point - 2 * w) & 1) * (w - 1);
  const int y = horizontal ? (point / w) * (h - 1) : (point - 2 * w) / 2 + 1;
  const int py = y & 1, px = x & 1, phase = 2 * py + px;
  const int border = (y == 0 || y + 1 == h ? 1 : 0) | (x == 0 || x + 1 == w ? 2 : 0);
  float acc[3]{};
#pragma unroll 1
  for (int tap = 0; tap < kTaps; ++tap) {
    const int yy = y / 2 + tap / 4 - 2 + py, xx = x / 2 + tap % 4 - 2 + px;
    if (yy < 0 || yy >= p.height || xx < 0 || xx >= p.width) continue;
    const auto* a = p.input + (static_cast<long>(yy) * p.width + xx) * kC;
    const auto* b = p.edges + ((phase * kBorders + border) * kChannels * kTaps + tap) * kC;
    const float a0 = __half2float(a[lane]), a1 = __half2float(a[lane + 32]);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const auto* weight = b + c * kTaps * kC;
      acc[c] = fmaf(a0, weight[lane], acc[c]);
      acc[c] = fmaf(a1, weight[lane + 32], acc[c]);
    }
  }
#pragma unroll
  for (int step = 16; step > 0; step /= 2)
#pragma unroll
    for (int c = 0; c < 3; ++c) acc[c] += __shfl_down_sync(0xffffffffU, acc[c], step);
  if (lane == 0) {
    const long plane = static_cast<long>(h) * w, at = static_cast<long>(y) * w + x;
#pragma unroll
    for (int c = 0; c < 3; ++c)
      p.output[c * plane + at] = gmfss_grid_rgb::clamp_half(
          __float2half_rn(acc[c] + p.bias[(phase * kBorders + border) * kChannels + c]));
  }
}

inline void enqueue(const Params& p, cudaStream_t stream) {
  constexpr int tile_rows = rn::kWarps;
  constexpr int smem = (tile_rows + 4) * (rn::kTW + 4) * (kChannelSlice + 8) * 2;
  static_assert(smem <= 48 * 1024);
  const dim3 grid((p.width + rn::kTW - 1) / rn::kTW, (p.height + tile_rows - 1) / tile_rows,
                  kPhases / kPhasePair);
  conv<<<grid, rn::kThreads, smem, stream>>>(p);
  const int boundary = 4 * p.width + 4 * p.height - 4;
  edges<<<(boundary + 3) / 4, 128, 0, stream>>>(p);
}
}  // namespace kirari::gpu::cuda::gmfss_grid_folded
