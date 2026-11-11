#include "gpu/cuda/backend/fused_graphs.h"

#include <format>
#include <utility>

#include "core/log.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"

namespace ajp::gpu::cuda {

FusedGraphs::FusedGraphs(std::size_t capacity, std::string who)
    : book_(capacity, kCaptureAttempts), who_(std::move(who)) {}

FusedGraphs::~FusedGraphs() {
  for (std::size_t i = 0; i < execs_.size(); ++i) destroy(i);
  // Le bilan de la séance (B33-03) : combien de passages ont rejoué un graphe, et combien sont restés en
  // lancement direct faute de place ou après un refus.
  const GraphCounters& c = book_.counters();
  if (c.replays + c.overflow + c.capture_failures + c.launch_failures == 0) return;
  try {
    log_channel("tensorrt")
        ->info(
            "{} : graphes CUDA, {} rejeux, {} captures, {} refusées, {} rejeux refusés, {} lancements "
            "directs "
            "faute de place",
            who_, c.replays, c.captures, c.capture_failures, c.launch_failures, c.overflow);
  } catch (...) {  // NOLINT(bugprone-empty-catch) : un destructeur ne lève pas, le journal même a échoué
  }
}

void FusedGraphs::destroy(std::size_t index) {
  if (index >= execs_.size() || execs_[index] == nullptr) return;
  cuda_warn(cudaGraphExecDestroy(execs_[index]), "cudaGraphExecDestroy");
  execs_[index] = nullptr;
}

void FusedGraphs::clear() {
  for (std::size_t i = 0; i < execs_.size(); ++i) destroy(i);
  execs_.clear();
  book_.clear();
}

std::expected<bool, Refusal> FusedGraphs::instantiate(std::size_t index, cudaError_t error,
                                                      cudaGraph_t captured,
                                                      const std::expected<void, Refusal>& queued) {
  if (error == cudaSuccess && queued && captured != nullptr)
    error = cudaGraphInstantiate(&execs_[index], captured, 0);
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (error == cudaSuccess && queued && captured != nullptr) {
    book_.captured(true);
    return true;
  }
  execs_[index] = nullptr;
  if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
    return std::unexpected(lost_refusal(lost, std::format("{} : capture du graphe", who_)));
  book_.captured(false, !queued                ? std::string_view(queued.error().detail)
                        : error != cudaSuccess ? std::string_view(cudaGetErrorString(error))
                                               : std::string_view("graphe vide"));
  return false;
}

std::expected<bool, Refusal> FusedGraphs::replay(cudaStream_t stream, std::size_t index) {
  const cudaError_t launched = cudaGraphLaunch(execs_[index], stream);
  if (launched == cudaSuccess) return true;
  if (sticky(launched))
    return std::unexpected(cuda_refusal(launched, std::format("{} : cudaGraphLaunch", who_)));
  destroy(index);
  book_.launch_failed(index);
  if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
    return std::unexpected(lost_refusal(lost, std::format("{} : cudaGraphLaunch", who_)));
  return false;
}

}  // namespace ajp::gpu::cuda
