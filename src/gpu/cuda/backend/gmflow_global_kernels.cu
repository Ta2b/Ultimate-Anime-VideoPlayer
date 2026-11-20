#include <cstdint>
#include <limits>

#include "gpu/cuda/backend/gmflow_attention_kernels.cuh"
#include "gpu/cuda/backend/gmflow_global_kernels.h"

namespace kirari::gpu::cuda {
namespace {

namespace ga = gmflow_attention;
namespace rn = rife_native;

struct Params {
  const __half* query;
  const __half* keys;
  const float2* values;  // Null pour l’appariement : les valeurs sont les coordonnées.
  float2* out;
  int width;
  int tokens;
};

constexpr float kScale = ga::kLog2e / 11.313708498984761F;  // 1/√128, en base 2.

// Charge une tuile de jetons, zéros au-delà de la grille. Même pas partagé que scores de l'attention.
__device__ __forceinline__ void load_tile(__half* shared, const __half* input, int begin, int tokens) {
  const int chunk = static_cast<int>(threadIdx.x) & 15;
  for (int row = static_cast<int>(threadIdx.x) >> 4; row < ga::kRows; row += ga::kThreads / 16) {
    const bool valid = begin + row < tokens;
    const auto token = static_cast<std::int64_t>(valid ? begin + row : 0);
    rn::cp_async16(rn::smem_u32(shared + row * ga::kLds + chunk * 8), input + token * ga::kD + chunk * 8,
                   valid);
  }
  ga::cp_async_commit();
  ga::cp_async_wait<0>();
}

struct Row {
  float max_score = -INFINITY;
  float denominator = 0.F;
  float x = 0.F;
  float y = 0.F;
};

__device__ __forceinline__ float group_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffU, v, 2));
}

__device__ __forceinline__ float group_sum(float v) {
  v += __shfl_xor_sync(0xffffffffU, v, 1);
  return v + __shfl_xor_sync(0xffffffffU, v, 2);
}

// Quatre fils d'une ligne gardent leurs parts FP32 du dénominateur et des deux numérateurs.
__device__ __forceinline__ void accumulate(Row& row, const rn::Frag<ga::Acc::kF32> (&scores)[8],
                                           const float2* values, int left, int h, int lane) {
  const int t = lane & 3;
  float s[16];
  float top = -INFINITY;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const float2 v = scores[nt].pair(h);
    const int key = 8 * nt + 2 * t;
    s[2 * nt] = key < left ? v.x * kScale : -INFINITY;
    s[2 * nt + 1] = key + 1 < left ? v.y * kScale : -INFINITY;
    top = fmaxf(top, fmaxf(s[2 * nt], s[2 * nt + 1]));
  }
  const float next = fmaxf(row.max_score, group_max(top));
  const float alpha = exp2f(row.max_score - next);
  row.max_score = next;
  row.denominator *= alpha;
  row.x *= alpha;
  row.y *= alpha;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float weight = exp2f(s[i] - next);
    const float2 v = values[8 * (i / 2) + 2 * t + i % 2];
    row.denominator += weight;
    row.x += weight * v.x;
    row.y += weight * v.y;
  }
}

__device__ __forceinline__ float2 coords(int token, int width) {
  const int y = token / width;
  return make_float2(static_cast<float>(token - y * width), static_cast<float>(y));
}

template <bool kPropagation>
__global__ void __launch_bounds__(ga::kThreads) global_kernel(Params p) {
  __shared__ __align__(16) __half shared[ga::kRows * ga::kLds];
  __shared__ float2 values[ga::kRows];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int direction = static_cast<int>(blockIdx.y);
  const int first = static_cast<int>(blockIdx.x) * ga::kRows;
  const auto plane = static_cast<std::int64_t>(p.tokens) * ga::kD;
  const __half* q = p.query + direction * plane;
  const __half* k = p.keys + (kPropagation ? direction : direction ^ 1) * plane;
  load_tile(shared, q, first, p.tokens);
  __syncthreads();
  std::uint32_t query[ga::kSteps][4];
  const auto base = rn::smem_u32(shared) +
                    static_cast<std::uint32_t>(((warp * 16 + (lane & 15)) * ga::kLds + (lane >> 4) * 8) * 2);
#pragma unroll
  for (int step = 0; step < ga::kSteps; ++step)
    rn::ldmatrix4(query[step], base + static_cast<std::uint32_t>(step * 32));
  __syncthreads();

  Row row[2];
  for (int from = 0; from < p.tokens; from += ga::kRows) {
    load_tile(shared, k, from, p.tokens);
    if (tid < ga::kRows) {
      const int key = from + tid;
      float2 v = make_float2(0.F, 0.F);
      if (key < p.tokens) {
        if constexpr (kPropagation)
          v = p.values[static_cast<std::int64_t>(direction) * p.tokens + key];
        else
          v = coords(key, p.width);
      }
      values[tid] = v;
    }
    __syncthreads();
    rn::Frag<ga::Acc::kF32> scores[8];
    ga::scores(scores, query, shared, lane);
    const int left = p.tokens - from;
    accumulate(row[0], scores, values, left, 0, lane);
    accumulate(row[1], scores, values, left, 1, lane);
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float denom = group_sum(row[h].denominator);
    const float nx = group_sum(row[h].x);
    const float ny = group_sum(row[h].y);
    const int token = first + 16 * warp + (lane >> 2) + 8 * h;
    if ((lane & 3) != 0 || token >= p.tokens) continue;
    float2 flow = make_float2(nx / denom, ny / denom);
    if constexpr (!kPropagation) {
      const float2 origin = coords(token, p.width);
      flow.x -= origin.x;
      flow.y -= origin.y;
    }
    p.out[static_cast<std::int64_t>(direction) * p.tokens + token] = flow;
  }
}

bool aligned(const void* p, std::size_t bytes) {
  return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

template <bool kPropagation>
cudaError_t launch(cudaStream_t stream, const void* query, const void* keys, const void* flow, int height,
                   int width, void* out) {
  const auto tokens = static_cast<std::int64_t>(height) * width;
  if (height < 1 || width < 1 || tokens > std::numeric_limits<int>::max() - ga::kRows ||
      !aligned(query, 16) || !aligned(keys, 16) || !aligned(out, alignof(float2)) ||
      (kPropagation && (!aligned(flow, alignof(float2)) || flow == out)) || query == out || keys == out)
    return cudaErrorInvalidValue;
  const Params params{.query = static_cast<const __half*>(query),
                      .keys = static_cast<const __half*>(keys),
                      .values = static_cast<const float2*>(flow),
                      .out = static_cast<float2*>(out),
                      .width = width,
                      .tokens = static_cast<int>(tokens)};
  const auto blocks = static_cast<unsigned>((params.tokens + ga::kRows - 1) / ga::kRows);
  global_kernel<kPropagation><<<dim3(blocks, 2), ga::kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_gmflow_global_match(cudaStream_t stream, const void* tokens, int height, int width,
                                       void* flow) {
  return launch<false>(stream, tokens, tokens, nullptr, height, width, flow);
}

cudaError_t launch_gmflow_global_propagate(cudaStream_t stream, const void* query, const void* keys,
                                           const void* flow, int height, int width, void* out) {
  return launch<true>(stream, query, keys, flow, height, width, out);
}

}  // namespace kirari::gpu::cuda
