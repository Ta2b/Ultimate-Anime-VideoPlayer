// La chaîne du RIFE fusionné pour l'essai du natif (RifeFusedEngine::try_native, rife_native_mix.cpp) : la
// sortie de la chaîne entière sur la paire de test, le temps d'une passe de lecture de plusieurs aiguillages
// en tours intercalés, l'aiguillage de chaque morceau vers un réseau natif ou vers TensorRT, et un morceau
// seul (rife_native_trial.h).
#include <algorithm>
#include <cstddef>
#include <cuda_fp16.h>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_fused_engine.h"

namespace kirari::gpu::cuda {
namespace {

// La mesure du temps, comme celle du SPAN (span_timing.h) : des tours intercalés, chacun d'une passe de
// chauffe puis de deux passes entre deux événements ; la médiane des tours.
constexpr int kTimeWarmup = 1;
constexpr int kTimeRuns = 2;
constexpr int kTimeRounds = 11;

// Un événement CUDA rendu quoi qu'il arrive.
struct Event {
  cudaEvent_t event = nullptr;
  Event() = default;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (event != nullptr) cudaEventDestroy(event);
  }
};

double median(std::vector<double> of) {
  std::ranges::sort(of);
  return of.empty() ? 0.0 : of[of.size() / 2];
}

}  // namespace

void RifeFusedEngine::forget_passes() {
  graphs_.clear();
  static_cast<void>(graphs_.take_notice());
  kept_.reset();
  hint_ = Hint::None;
  warm_ready_ = false;
  warm_next_ = false;
}

std::expected<std::vector<float>, Refusal> RifeFusedEngine::trial_output(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, const Tensor& input,
    std::span<const Tensor> outputs, std::size_t elements) {
  forget_passes();
  // Trois passes : directe, capture du graphe, rejeu ; la dernière est comparée. Avant elle, X du dernier
  // bloc part de NaN quand son réseau le refait à la volée (il n'y écrit que le flot, que lit la fin) : un
  // pixel qu'il oublierait rend la sortie non finie, au lieu de garder le flot de la passe d'avant.
  const int last = plan_.blocks - 1;
  for (int pass = 0; pass < 3; ++pass) {
    if (pass == 2 && glued(last)) {
      kernels::FusedDims x = kernels::rife_fused_x(plan_, last, height_, width_);
      x.channels = pitch(x.channels);
      auto* base = static_cast<std::byte*>(shared->data()) + scratch_offset_;
      if (auto ok = cuda_check(cudaMemsetAsync(base + layout_.x[static_cast<std::size_t>(last)], 0xFF,
                                               static_cast<std::size_t>(x.elements()) * 2U, stream),
                               "RIFE natif : X du dernier bloc à NaN");
          !ok)
        return std::unexpected(std::move(ok.error()));
    }
    const Tensor inputs[]{input};
    if (auto ok = enqueue(stream, shared, inputs, outputs); !ok)
      return std::unexpected(std::move(ok.error()));
    if (auto ok = cuda_check(cudaStreamSynchronize(stream), "RIFE natif : essai"); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  std::vector<__half> half(elements);
  if (auto ok =
          cuda_check(cudaMemcpy(half.data(), fused_address(outputs[0]), elements * 2, cudaMemcpyDeviceToHost),
                     "RIFE natif : copie de l'essai");
      !ok)
    return std::unexpected(std::move(ok.error()));
  std::vector<float> out(elements);
  std::ranges::transform(half, out.begin(), [](__half v) { return __half2float(v); });
  return out;
}

std::expected<std::vector<double>, Refusal> RifeFusedEngine::trial_times(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, const Tensor& input,
    std::span<const Tensor> outputs, std::span<const std::vector<int>> routes) {
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return std::unexpected(cuda_refusal(error, std::format("RIFE natif : mesure du temps ({})", step)));
  };
  Event start;
  Event stop;
  for (Event* event : {&start, &stop})
    if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess)
      return failed(e, "cudaEventCreate");
  const Tensor inputs[]{input};
  // Une passe de lecture ×2 : la B d'avant devient A, seule la nouvelle B s'encode. Chaque aiguillage a ses
  // graphes (il est dans leur clé), rejoués comme en lecture une fois capturés.
  const auto pass = [&]() -> std::expected<void, Refusal> {
    next_pair_follows();
    return enqueue(stream, shared, inputs, outputs);
  };
  // Le lot de l'aiguillage v. warm : passes de chauffe (au premier tour, un premier lancement puis une
  // capture par emplacement de A).
  const auto batch = [&](std::size_t v, int warm) -> std::expected<double, Refusal> {
    set_route(routes[v]);
    std::expected<void, Refusal> ok;
    for (int i = 0; i < warm && ok; ++i) ok = pass();
    cudaError_t e = ok ? cudaEventRecord(start.event, stream) : cudaSuccess;
    for (int i = 0; i < kTimeRuns && ok && e == cudaSuccess; ++i) ok = pass();
    if (ok && e == cudaSuccess) e = cudaEventRecord(stop.event, stream);
    if (ok && e == cudaSuccess) e = cudaEventSynchronize(stop.event);
    float ms = 0.F;
    if (ok && e == cudaSuccess) e = cudaEventElapsedTime(&ms, start.event, stop.event);
    if (!ok) return std::unexpected(std::move(ok.error()));
    if (e != cudaSuccess) return failed(e, "événement");
    if (!(ms > 0.F)) return refuse(RefusalCode::VendorCallFailed, "RIFE natif : lot mesuré à 0 ms");
    return static_cast<double>(ms) / kTimeRuns;
  };
  std::vector<std::vector<double>> rounds(routes.size());
  for (int round = 0; round < kTimeRounds; ++round)
    for (std::size_t i = 0; i < routes.size(); ++i) {
      const std::size_t v = round % 2 == 0 ? i : routes.size() - 1 - i;  // dans l'ordre, puis l'inverse
      auto ms = batch(v, round == 0 ? 4 : kTimeWarmup);
      if (!ms) return std::unexpected(std::move(ms.error()));
      rounds[v].push_back(*ms);
    }
  std::vector<double> times(rounds.size());
  std::ranges::transform(rounds, times.begin(), median);
  return times;
}

void RifeFusedEngine::set_route(std::vector<int> route) {
  route_ = std::move(route);
  native_on_ = false;
  native_workspace_ = 0;
  route_code_ = 0;
  for (std::size_t p = 0; p < route_.size(); ++p) {
    if (route_[p] < 0) continue;
    native_on_ = true;
    native_workspace_ =
        std::max(native_workspace_, natives_[static_cast<std::size_t>(route_[p])]->workspace_bytes());
    route_code_ |= static_cast<std::uintptr_t>(route_[p] + 1) << (3U * p);
  }
}

std::expected<void, Refusal> RifeFusedEngine::run_piece(cudaStream_t stream,
                                                        const std::shared_ptr<SharedActivations>& arena,
                                                        const Tensor& input, int piece,
                                                        const RifeNativeNet* net) {
  if (piece < 0 || piece > plan_.blocks)
    return refuse(RefusalCode::ShapeInvalid, "RIFE natif : morceau hors de la chaîne");
  auto* base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  if (net != nullptr) arena->used(this, net->workspace_bytes());
  if (piece == 0) {
    if (net != nullptr)
      return cuda_check(net->encode(stream, fused_address(input), base + layout_.encode, arena->data()),
                        "RIFE natif : encodeur");
    const Tensor image = fused_tensor(input.handle, input.offset, 3, height_, width_);
    const Tensor encoded =
        fused_tensor(reinterpret_cast<std::uintptr_t>(base + layout_.encode), 0, 4, height_, width_);
    return encode_->enqueue_plain(stream, arena, std::span(&image, 1), std::span(&encoded, 1));
  }
  // Un bloc dont un réseau natif sait calculer l'entrée à la volée se mesure colle comprise, quelle que soit
  // l'option (ce réseau sans elle, use_glue, compris) : sans elle, TensorRT et un réseau qui ne la refait pas
  // passeraient pour plus rapides.
  const int b = piece - 1;
  const auto [encode_a, encode_b] = trial_encodes();
  const void* const in = fused_address(input);
  const bool with_glue = std::ranges::any_of(natives_, [b](const auto& n) { return n->can_glue(b); });
  if (with_glue && (net == nullptr || !net->glues(b))) {
    if (auto ok = launch_glue(stream, in, b, base, encode_a, encode_b); !ok) return ok;
  }
  return run_block(stream, arena, in, b, base, encode_a, encode_b, net);
}

}  // namespace kirari::gpu::cuda
