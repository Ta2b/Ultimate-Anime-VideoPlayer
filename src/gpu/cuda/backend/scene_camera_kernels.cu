// Noyaux de l'affinage de la caméra de la carte (scene_camera_kernels.h), calqués sur
// kernels/ref/scene_camera.cpp.
#include <cstddef>

#include "gpu/cuda/backend/scene_camera_kernels.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::gpu::cuda {
namespace {

namespace scene = kernels::scene;

constexpr int kSide = 16;  // fils par côté d'un bloc
constexpr int kThreads = kSide * kSide;
constexpr int kWarps = kThreads / 32;
constexpr int kTile = kernels::kSceneTileFlowPixels;
static_assert(kTile == kSide, "un fil par pixel de la tuile");

__device__ int clamp_index(int v, int high) {
  return v < 0 ? 0 : (v > high ? high : v);
}

__global__ void scene_box_rows(SceneBoxParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const unsigned char* row = p.luma + static_cast<std::size_t>(y) * p.pitch;
  float sum = 0.0F;
  for (int k = -p.radius; k <= p.radius; ++k)
    sum = __fadd_rn(sum, static_cast<float>(row[clamp_index(x + k, p.width - 1)]));
  p.rows[static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x)] = sum;
}

__global__ void scene_box_cols(SceneBoxParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  float sum = 0.0F;
  for (int k = -p.radius; k <= p.radius; ++k) {
    sum = __fadd_rn(sum, p.rows[static_cast<std::size_t>(clamp_index(y + k, p.height - 1)) *
                                    static_cast<std::size_t>(p.width) +
                                static_cast<std::size_t>(x)]);
  }
  const float area = 1.0F / static_cast<float>((2 * p.radius + 1) * (2 * p.radius + 1));
  p.out[static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x)] =
      __fmul_rn(sum, area);
}

__device__ void model_of(const SceneRefineState* state, float* p) {
  for (int k = 0; k < 6; ++k) p[k] = static_cast<float>(state->model[k]);
}

// Le pixel (i, j) d'un pas : dans une tuile du décor, sur la grille de kRefineStride.
struct StepPixel {
  int i = 0;
  int j = 0;
  bool used = false;
};
__device__ StepPixel step_pixel(const SceneRefineParams& p) {
  StepPixel out;
  out.i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * scene::kRefineStride;
  out.j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) * scene::kRefineStride;
  out.used = out.i < p.width && out.j < p.height &&
             p.decor[static_cast<std::size_t>(out.j / kTile) * static_cast<std::size_t>(p.cols) +
                     static_cast<std::size_t>(out.i / kTile)] != 0;
  return out;
}

__global__ void scene_refine_histogram(SceneRefineParams p) {
  if (p.state->stopped) return;  // le même pour tout le bloc
  __shared__ unsigned histogram[scene::kRefineBins];
  __shared__ unsigned count;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int k = tid; k < scene::kRefineBins; k += kThreads) histogram[k] = 0U;
  if (tid == 0) count = 0U;
  __syncthreads();
  float model[6];
  model_of(p.state, model);
  const kernels::SceneCameraFrame frame = kernels::scene_camera_frame(p.width, p.height);
  const StepPixel px = step_pixel(p);
  if (px.used) {
    const kernels::SceneModelAt m = kernels::scene_model_at(model, frame, px.i, px.j);
    const kernels::SceneWarpTerm t =
        kernels::scene_warp_term(p.a, p.b, p.width, p.height, px.i, px.j, m.dx, m.dy);
    if (t.valid) {
      atomicAdd(&histogram[kernels::scene_refine_bin(fabsf(t.e))], 1U);
      atomicAdd(&count, 1U);
    }
  }
  __syncthreads();
  for (int k = tid; k < scene::kRefineBins; k += kThreads) {
    if (histogram[k] != 0U) atomicAdd(&p.state->histogram[k], histogram[k]);
  }
  if (tid == 0 && count != 0U) atomicAdd(&p.state->count, count);
}

__global__ void scene_refine_accumulate(SceneRefineParams p) {
  if (p.state->stopped) return;
  __shared__ float median;
  __shared__ float partial[kWarps][scene::kRefineSums];
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  if (tid == 0) median = kernels::scene_refine_median(p.state->histogram, p.state->count);
  __syncthreads();
  float model[6];
  model_of(p.state, model);
  const kernels::SceneCameraFrame frame = kernels::scene_camera_frame(p.width, p.height);
  float terms[scene::kRefineSums] = {};
  const StepPixel px = step_pixel(p);
  if (px.used) {
    const kernels::SceneModelAt m = kernels::scene_model_at(model, frame, px.i, px.j);
    const kernels::SceneWarpTerm t =
        kernels::scene_warp_term(p.a, p.b, p.width, p.height, px.i, px.j, m.dx, m.dy);
    if (t.valid) kernels::scene_refine_terms(t, m, kernels::scene_huber_weight(t.e, median), terms);
  }
  const int lane = tid % 32;
  const int warp = tid / 32;
  for (int k = 0; k < scene::kRefineSums; ++k) {
    float v = terms[k];
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xFFFFFFFFU, v, offset);
    if (lane == 0) partial[warp][k] = v;
  }
  __syncthreads();
  if (tid < scene::kRefineSums) {
    float sum = 0.0F;
    for (int w = 0; w < kWarps; ++w) sum += partial[w][tid];
    if (sum != 0.0F) atomicAdd(&p.state->sums[tid], static_cast<double>(sum));
  }
}

// Un fil : le pas, puis les sommes et l'histogramme remis à zéro pour le suivant.
__global__ void scene_refine_update(SceneRefineState* state) {
  if (state->stopped) return;
  if (kernels::scene_refine_solve(state->sums, state->model)) {
    ++state->steps;
  } else {
    state->stopped = 1;
  }
  for (int k = 0; k < scene::kRefineSums; ++k) state->sums[k] = 0.0;
  for (int k = 0; k < scene::kRefineBins; ++k) state->histogram[k] = 0U;
  state->count = 0U;
}

// Une tuile par bloc, un fil par pixel ; toutes les tuiles, décor ou non (l'écart après la caméra).
__global__ void scene_check(SceneCheckParams p) {
  const std::size_t tile = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(p.cols) +
                           static_cast<std::size_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  constexpr int kSums = 8;
  __shared__ float partial[kWarps][kSums];
  __shared__ float shift[2];  // le décalage du pas en cours, celui du résultat que tient le fil 0
  kernels::SceneTileCheck result;
  if (tid == 0) shift[0] = shift[1] = 0.0F;
  float model[6];
  model_of(p.state, model);
  const kernels::SceneCameraFrame frame = kernels::scene_camera_frame(p.width, p.height);
  const float slack = kernels::scene_moving_slack(p.height);
  const int i = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
  const int j = static_cast<int>(blockIdx.y) * kTile + static_cast<int>(threadIdx.y);
  const int lane = tid % 32;
  const int warp = tid / 32;
  __syncthreads();
  for (int step = 0; step < scene::kCheckSteps; ++step) {
    float v[kSums] = {};
    if (i < p.width && j < p.height) {
      const kernels::SceneModelAt m = kernels::scene_model_at(model, frame, i, j);
      const kernels::SceneWarpTerm t =
          kernels::scene_warp_term(p.a, p.b, p.width, p.height, i, j, m.dx + shift[0], m.dy + shift[1]);
      if (t.valid) {
        v[0] = t.gx * t.gx;
        v[1] = t.gx * t.gy;
        v[2] = t.gy * t.gy;
        v[3] = t.gx * t.e;
        v[4] = t.gy * t.e;
        v[5] = 1.0F;
        v[6] = fabsf(t.e);
        v[7] = kernels::scene_pixel_moving(t, slack) ? 1.0F : 0.0F;
      }
    }
    for (int k = 0; k < kSums; ++k) {
      float s = v[k];
      for (int offset = 16; offset > 0; offset /= 2) s += __shfl_down_sync(0xFFFFFFFFU, s, offset);
      if (lane == 0) partial[warp][k] = s;
    }
    __syncthreads();
    if (tid == 0) {
      kernels::SceneCheckSums sums;
      for (int w = 0; w < kWarps; ++w) {
        sums.xx += partial[w][0];
        sums.xy += partial[w][1];
        sums.yy += partial[w][2];
        sums.bx += partial[w][3];
        sums.by += partial[w][4];
        sums.pixels += partial[w][5];
        sums.error += partial[w][6];
        sums.moving += partial[w][7];
      }
      if (step == 0) kernels::scene_check_seen(sums, result);
      kernels::scene_check_step(sums, result);
      shift[0] = result.sx;
      shift[1] = result.sy;
    }
    __syncthreads();
  }
  if (tid == 0) p.out[tile] = result;
}

dim3 cover(int width, int height) {
  return dim3(static_cast<unsigned>((width + kSide - 1) / kSide),
              static_cast<unsigned>((height + kSide - 1) / kSide));
}

}  // namespace

cudaError_t launch_scene_box(const SceneBoxParams& params, cudaStream_t stream) {
  if (params.width <= 0 || params.height <= 0 || params.radius < 0) return cudaErrorInvalidValue;
  const dim3 block(kSide, kSide);
  scene_box_rows<<<cover(params.width, params.height), block, 0, stream>>>(params);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return status;
  scene_box_cols<<<cover(params.width, params.height), block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_scene_refine_step(const SceneRefineParams& params, cudaStream_t stream) {
  if (params.width <= 0 || params.height <= 0 || params.cols <= 0) return cudaErrorInvalidValue;
  const int stride = scene::kRefineStride;
  const dim3 grid = cover((params.width + stride - 1) / stride, (params.height + stride - 1) / stride);
  const dim3 block(kSide, kSide);
  scene_refine_histogram<<<grid, block, 0, stream>>>(params);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return status;
  scene_refine_accumulate<<<grid, block, 0, stream>>>(params);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return status;
  scene_refine_update<<<1, 1, 0, stream>>>(params.state);
  return cudaGetLastError();
}

cudaError_t launch_scene_check(const SceneCheckParams& params, cudaStream_t stream) {
  if (params.cols <= 0 || params.rows <= 0) return cudaErrorInvalidValue;
  scene_check<<<dim3(static_cast<unsigned>(params.cols), static_cast<unsigned>(params.rows)),
                dim3(kSide, kSide), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
