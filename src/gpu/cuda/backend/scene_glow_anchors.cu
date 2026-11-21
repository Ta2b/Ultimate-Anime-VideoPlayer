#include <cuda_fp16.h>

#include "gpu/cuda/backend/scene_glow_kernels.h"
#include "kernels/spec/scene_glow.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kSide = kernels::kGlowAnchorBlock;
constexpr int kThreads = kSide * kSide;
constexpr int kRings = 5;
constexpr int kRays = 8;

__device__ float channel(GlowSource source, std::size_t index) {
  if (source.fp16) return __half2float(static_cast<const __half*>(source.rgb)[index]);
  return static_cast<const float*>(source.rgb)[index];
}

__device__ float hdr_code(float r, float g, float b) {
  const float lum = (0.2126F * powf(r, 2.4F) + 0.7152F * powf(g, 2.4F) + 0.0722F * powf(b, 2.4F));
  const float q = powf(lum * (203.0F / 10'000.0F), 0.1593017578125F);
  return powf((0.8359375F + 18.8515625F * q) / (1.0F + 18.6875F * q), 78.84375F);
}

__global__ void glow_reduce(GlowSource source, GlowAggregate* tiles) {
  __shared__ float4 sums[kThreads];
  __shared__ float best[kThreads];
  __shared__ int first[kThreads];
  const int thread = static_cast<int>(threadIdx.x);
  const int want_x = static_cast<int>(blockIdx.x) * kSide + thread % kSide;
  const int want_y = static_cast<int>(blockIdx.y) * kSide + thread / kSide;
  // Les moyennes des dernières tuiles prolongent la dernière ligne/colonne. Le centre reste dans l'image.
  const int x = min(want_x, source.width - 1), y = min(want_y, source.height - 1);
  const std::size_t plane = static_cast<std::size_t>(source.width) * source.height;
  const std::size_t index = static_cast<std::size_t>(y) * source.width + x;
  const float r = fminf(fmaxf(channel(source, index), 0.0F), 1.0F);
  const float g = fminf(fmaxf(channel(source, index + plane), 0.0F), 1.0F);
  const float b = fminf(fmaxf(channel(source, index + plane * 2), 0.0F), 1.0F);
  const float pq = hdr_code(r, g, b);
  sums[thread] = make_float4(r, g, b, pq);
  best[thread] =
      want_x < source.width && want_y < source.height && fminf(fminf(r, g), b) >= 0.90F ? pq : -1.0F;
  first[thread] = thread;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (thread < stride) {
      const int next = thread + stride;
      sums[thread].x += sums[next].x;
      sums[thread].y += sums[next].y;
      sums[thread].z += sums[next].z;
      sums[thread].w += sums[next].w;
      if (best[next] > best[thread] || (best[next] == best[thread] && first[next] < first[thread])) {
        best[thread] = best[next];
        first[thread] = first[next];
      }
    }
    __syncthreads();
  }
  if (thread != 0) return;
  constexpr float m = 1.0F / kThreads;
  const auto avg = sums[0];
  const int origin_x = static_cast<int>(blockIdx.x) * kSide;
  const int origin_y = static_cast<int>(blockIdx.y) * kSide;
  tiles[blockIdx.y * gridDim.x + blockIdx.x] = {avg.x * m,
                                                avg.y * m,
                                                avg.z * m,
                                                avg.w * m,
                                                best[0],
                                                origin_x + first[0] % kSide,
                                                origin_y + first[0] / kSide};
}

struct Ring {
  float luma;
  float2 hue;
  float high;
  float gap;
  bool inside;
};

__device__ float mix(float a, float b, float s) {
  return a * (1.0F - s) + b * s;
}

__device__ float interpolate(float a, float b, float c, float d, float s, float t) {
  return mix(mix(a, b, s), mix(c, d, s), t);
}

__device__ Ring read_ring(const GlowAggregate* tiles, int cols, int rows, int x, int y, int w, int h) {
  const float u = fminf(fmaxf((x + 0.5F) / kSide - 0.5F, 0.0F), cols - 1.0F);
  const float v = fminf(fmaxf((y + 0.5F) / kSide - 0.5F, 0.0F), rows - 1.0F);
  const int x0 = static_cast<int>(u), y0 = static_cast<int>(v);
  const int x1 = min(x0 + 1, cols - 1), y1 = min(y0 + 1, rows - 1);
  const float s = u - x0, t = v - y0;
  const auto a = tiles[y0 * cols + x0], b = tiles[y0 * cols + x1];
  const auto c = tiles[y1 * cols + x0], d = tiles[y1 * cols + x1];
  const float red = interpolate(a.red, b.red, c.red, d.red, s, t);
  const float green = interpolate(a.green, b.green, c.green, d.green, s, t);
  const float blue = interpolate(a.blue, b.blue, c.blue, d.blue, s, t);
  const float high = fmaxf(fmaxf(red, green), blue), low = fminf(fminf(red, green), blue);
  return {interpolate(a.pq, b.pq, c.pq, d.pq, s, t), make_float2(red - green, blue - green), high,
          2.0F * high + low - (red + green + blue), x >= 0 && x < w && y >= 0 && y < h};
}

__device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

__device__ float magnitude(float2 a) {
  return sqrtf(dot(a, a));
}

// Rayons directeurs arrondis au pixel comme probe_glow.py, sur les cinq anneaux distincts.
__device__ __constant__ int kRadii[kRings] = {16, 32, 48, 64, 96};
__device__ __constant__ int kDiagonals[kRings] = {11, 23, 34, 45, 68};
__device__ __constant__ int kX[kRays] = {1, 1, 0, -1, -1, -1, 0, 1};
__device__ __constant__ int kY[kRays] = {0, 1, 1, 1, 0, -1, -1, -1};

__device__ Ring ring(const GlowAggregate* tiles, int cols, int rows, GlowSource source,
                     const GlowAggregate& centre, int depth, int direction) {
  const int step = (direction & 1) ? kDiagonals[depth] : kRadii[depth];
  return read_ring(tiles, cols, rows, centre.x + kX[direction] * step, centre.y + kY[direction] * step,
                   source.width, source.height);
}

struct Arc {
  Ring near[kRays];
  unsigned inside = 0;
  unsigned support = 0;
};

__device__ void sample_arc(const GlowAggregate& centre, int direction, Arc& arc, const Ring& a, const Ring& b,
                           const Ring& c) {
  const unsigned bit = 1U << direction;
  arc.near[direction] = a;
  if (!a.inside || !b.inside || !c.inside) return;
  arc.inside |= bit;
  const float dy = centre.centre_pq - a.luma;
  if (dy < 0.005F || a.luma - b.luma < 0.007F || b.luma - c.luma < 0.003F ||
      dy > 0.67F * fmaxf(centre.centre_pq - c.luma, 0.02F) || a.high < 0.52F || b.high < 0.34F)
    return;
  const float ma = magnitude(a.hue), mb = magnitude(b.hue);
  if (ma < 0.14F || mb < 0.12F || dot(a.hue, b.hue) < 0.94F * ma * mb) return;
  const float2 da = make_float2(a.hue.x - c.hue.x, a.hue.y - c.hue.y);
  const float2 db = make_float2(b.hue.x - c.hue.x, b.hue.y - c.hue.y);
  const float mda = magnitude(da), mdb = magnitude(db);
  if (mda < 0.035F || mda - mdb < 0.006F || dot(da, db) < 0.91F * mda * mdb) return;
  arc.support |= bit;
}

__device__ GpuAnchor score(Arc& arc, unsigned edge_ok, GlowAggregate at) {
  const int visible = __popc(arc.inside);
  if (visible < 3) return {at.x, at.y, 0, 0, 0};
  if (visible < kRays) arc.support &= edge_ok;
  unsigned rays = 0;
  for (int d = 0; d < kRays; ++d) {
    const int last = (d + kRays - 1) % kRays;
    const unsigned span = (1U << d) | (1U << last);
    const float2 a = arc.near[d].hue, b = arc.near[last].hue;
    if ((arc.support & span) == span && dot(a, b) >= 0.96F * magnitude(a) * magnitude(b)) rays |= span;
  }
  const float share = static_cast<float>(__popc(rays)) / visible;
  if (share < (visible == kRays ? 0.625F : 0.4F)) return {at.x, at.y, 0, 0, 0};
  float2 hue{};
  for (int d = 0; d < kRays; ++d) {
    if (!(rays & (1U << d))) continue;
    hue.x += arc.near[d].hue.x;
    hue.y += arc.near[d].hue.y;
  }
  const float denominator = fmaxf(magnitude(hue), 1.0e-6F);
  return {at.x, at.y, hue.x / denominator, hue.y / denominator, share};
}

__global__ void glow_crowns(GlowSource source, const GlowAggregate* tiles, GpuAnchor* anchors, int cols,
                            int rows) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= cols * rows) return;
  const GlowAggregate centre = tiles[index];
  if (centre.centre_pq < 0.0F) {
    anchors[index] = {centre.x, centre.y, 0, 0, 0};
    return;
  }
  Arc small, large;
  unsigned small_edge = 0, large_edge = 0;
  for (int d = 0; d < kRays; ++d) {
    const Ring a = ring(tiles, cols, rows, source, centre, 0, d);
    const Ring b = ring(tiles, cols, rows, source, centre, 1, d);
    const Ring c = ring(tiles, cols, rows, source, centre, 2, d);
    const Ring e = ring(tiles, cols, rows, source, centre, 3, d);
    const Ring f = ring(tiles, cols, rows, source, centre, 4, d);
    sample_arc(centre, d, small, a, b, c);
    sample_arc(centre, d, large, b, e, f);
    if (b.high >= 0.93F * a.high && c.gap - a.gap >= 0.010F) small_edge |= 1U << d;
    if (e.high >= 0.93F * b.high && f.gap - b.gap >= 0.010F) large_edge |= 1U << d;
  }
  const auto a = score(small, small_edge, centre), b = score(large, large_edge, centre);
  anchors[index] = b.score > a.score ? b : a;
}

}  // namespace

cudaError_t launch_glow_reduce(GlowSource source, GlowAggregate* aggregates, cudaStream_t stream) {
  if (!source.rgb || source.width < 1 || source.height < 1 || source.width > 4096 || source.height > 4096 ||
      !aggregates)
    return cudaErrorInvalidValue;
  const int cols = glow_div_up(source.width, kSide), rows = glow_div_up(source.height, kSide);
  glow_reduce<<<dim3(cols, rows), kThreads, 0, stream>>>(source, aggregates);
  return cudaGetLastError();
}

cudaError_t launch_glow_crowns(GlowSource source, const GlowAggregate* aggregates, GpuAnchor* anchors,
                               cudaStream_t stream) {
  if (source.width < 1 || source.height < 1 || source.width > 4096 || source.height > 4096 || !aggregates ||
      !anchors)
    return cudaErrorInvalidValue;
  const int cols = glow_div_up(source.width, kSide), rows = glow_div_up(source.height, kSide);
  constexpr int kBlock = 128;
  glow_crowns<<<glow_div_up(cols * rows, kBlock), kBlock, 0, stream>>>(source, aggregates, anchors, cols,
                                                                       rows);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
