// Le banc du réglage du RIFE natif (rife_native_layer.h) : chaque forme d'une couche est lancée sur une
// entrée variée, sa sortie comparée à celle de la forme de référence, puis chronométrée en graphe.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#include "gpu/cuda/backend/rife_native_layer.h"

namespace kirari::gpu::cuda::rife_native {
namespace {

// Des valeurs entre -1 et 1, d'un hachage de l'indice : chaque pixel et chaque canal diffèrent.
__global__ void fill_kernel(__half* p, long n) {
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<long>(gridDim.x) * blockDim.x) {
    auto x = static_cast<std::uint32_t>(i) * 2654435761U;
    x ^= x >> 15U;
    x *= 2246822519U;
    x ^= x >> 13U;
    p[i] = __float2half(static_cast<float>(x & 0xFFFFU) / 32768.F - 1.F);
  }
}

// stats[0] : le plus grand |want|, stats[1] : le plus grand |got - want| (infini si une valeur n'est pas
// finie), en bits de flottant : positifs, ils se rangent comme des entiers.
__global__ void compare_kernel(const __half* want, const __half* got, long n, unsigned* stats) {
  float peak = 0.F;
  float diff = 0.F;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<long>(gridDim.x) * blockDim.x) {
    const float a = __half2float(want[i]);
    const float d = fabsf(__half2float(got[i]) - a);
    peak = fmaxf(peak, fabsf(a));
    diff = isfinite(d) ? fmaxf(diff, d) : INFINITY;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    peak = fmaxf(peak, __shfl_xor_sync(0xFFFFFFFFU, peak, offset));
    diff = fmaxf(diff, __shfl_xor_sync(0xFFFFFFFFU, diff, offset));
  }
  if ((threadIdx.x & 31U) == 0) {
    atomicMax(&stats[0], __float_as_uint(peak));
    atomicMax(&stats[1], __float_as_uint(diff));
  }
}

unsigned blocks_for(long n) {
  return static_cast<unsigned>(std::clamp((n + 255) / 256, 1L, 1024L));
}

}  // namespace

bool Bench::failed(const char* what, cudaError_t result, std::string& error) {
  static_cast<void>(cudaGetLastError());  // une erreur collante reste, pour l'appelant
  error = std::string("réglage du RIFE natif (") + what + ") : " + cudaGetErrorString(result);
  return false;
}

bool Bench::reserve(const Layer& layer, std::string& error) {
  const auto grow = [&](Buffer& buffer, std::int64_t& have, std::int64_t want) {
    if (want <= have) return cudaSuccess;
    have = want;
    return buffer.allocate(static_cast<std::size_t>(want));
  };
  const std::int64_t had = in_bytes_;
  cudaError_t result = grow(in_, in_bytes_, layer.in_bytes());
  if (result == cudaSuccess && in_bytes_ != had) {
    const long n = static_cast<long>(in_bytes_ / 2);
    fill_kernel<<<blocks_for(n), 256, 0, stream_>>>(in_.as<__half>(), n);
    result = cudaGetLastError();
  }
  if (result == cudaSuccess && layer.out_bytes() > out_bytes_) {
    out_bytes_ = layer.out_bytes();
    result = out_.allocate(static_cast<std::size_t>(out_bytes_));
    if (result == cudaSuccess) result = ref_.allocate(static_cast<std::size_t>(out_bytes_));
  }
  if (result == cudaSuccess)
    result = grow(partial_, partial_bytes_, std::max<std::int64_t>(layer.partial_bytes(), 256));
  if (result == cudaSuccess && stats_.as<void>() == nullptr) result = stats_.allocate(2 * sizeof(unsigned));
  return result == cudaSuccess || failed("mémoire", result, error);
}

// Les sorties partent de NaN (0xFFFF en FP16) : une valeur qu'une forme n'écrit pas ne passe pas la
// comparaison.
cudaError_t Bench::poison(const Buffer& out, const Layer& layer) {
  return cudaMemsetAsync(out.as<void>(), 0xFF, static_cast<std::size_t>(layer.out_bytes()), stream_);
}

bool Bench::reference(const Layer& layer, const void* w, std::string& error) {
  if (!reserve(layer, error)) return false;
  cudaError_t result = poison(ref_, layer);
  if (result == cudaSuccess) {
    enqueue(layer, stream_,
            params(layer, in_.as<const __half>(), ref_.as<__half>(), partial_.as<float>(), w));
    result = cudaGetLastError();
  }
  return result == cudaSuccess || failed("forme de référence", result, error);
}

std::optional<bool> Bench::differs(const Layer& layer, std::string& error) {
  const long n = static_cast<long>(layer.out_bytes() / 2);
  unsigned stats[2] = {0, 0};
  cudaError_t result = cudaMemsetAsync(stats_.as<void>(), 0, sizeof(stats), stream_);
  if (result == cudaSuccess) {
    compare_kernel<<<blocks_for(n), 256, 0, stream_>>>(ref_.as<const __half>(), out_.as<const __half>(), n,
                                                       stats_.as<unsigned>());
    result = cudaGetLastError();
  }
  if (result == cudaSuccess)
    result = cudaMemcpyAsync(stats, stats_.as<void>(), sizeof(stats), cudaMemcpyDeviceToHost, stream_);
  if (result == cudaSuccess) result = cudaStreamSynchronize(stream_);
  if (result != cudaSuccess) {
    failed("comparaison", result, error);
    return std::nullopt;
  }
  // Un défaut de rangement ou de lecture s'écarte de l'ordre du pic. L'ordre des sommes, d'un arrondi f16 par
  // terme au plus en f16, d'un par prise en mixte, du seul arrondi de la sortie en f32 : la marge suit.
  const float margin = layer.sums == RifeNativeSums::kF16     ? 16.F
                       : layer.sums == RifeNativeSums::kMixed ? 32.F
                                                              : 128.F;
  const float peak = std::max(std::bit_cast<float>(stats[0]), 1e-3F);
  return !(std::bit_cast<float>(stats[1]) <= peak / margin);
}

std::optional<Bench::Timing> Bench::time(const Layer& layer, const void* w, std::string& error) {
  constexpr int kBatch = 3;
  if (!reserve(layer, error)) return std::nullopt;
  const Params p = params(layer, in_.as<const __half>(), out_.as<__half>(), partial_.as<float>(), w);
  if (const cudaError_t result = poison(out_, layer); result != cudaSuccess) {
    failed("sortie", result, error);
    return std::nullopt;
  }
  enqueue(layer, stream_, p);
  if (cudaGetLastError() != cudaSuccess) {
    if (cudaPeekAtLastError() == cudaSuccess) return Timing{};  // ressources : la forme ne se lance pas
    failed("lancement", cudaPeekAtLastError(), error);
    return std::nullopt;
  }
  Timing timing;
  const std::optional<bool> off = differs(layer, error);
  if (!off) return std::nullopt;
  timing.differs = *off;
  cudaGraph_t graph = nullptr;
  cudaError_t result = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
  if (result == cudaSuccess) {
    for (int i = 0; i < kBatch; ++i) enqueue(layer, stream_, p);
    result = cudaStreamEndCapture(stream_, &graph);
  }
  cudaGraphExec_t exec = nullptr;
  if (result == cudaSuccess) result = cudaGraphInstantiate(&exec, graph, 0);
  if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
  for (int batch = 0; batch <= 3 && result == cudaSuccess; ++batch) {
    result = cudaEventRecord(start_.get(), stream_);
    if (result == cudaSuccess) result = cudaGraphLaunch(exec, stream_);
    if (result == cudaSuccess) result = cudaEventRecord(stop_.get(), stream_);
    if (result == cudaSuccess) result = cudaEventSynchronize(stop_.get());
    float ms = 0.F;
    if (result == cudaSuccess) result = cudaEventElapsedTime(&ms, start_.get(), stop_.get());
    if (batch > 0) timing.ms = std::min(timing.ms, ms / kBatch);  // le lot 0 chauffe
  }
  if (exec != nullptr) static_cast<void>(cudaGraphExecDestroy(exec));
  if (result != cudaSuccess) {
    failed("chrono", result, error);
    return std::nullopt;
  }
  return timing;
}

}  // namespace kirari::gpu::cuda::rife_native
