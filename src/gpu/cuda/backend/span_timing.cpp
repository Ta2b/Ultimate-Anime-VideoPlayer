#include "gpu/cuda/backend/span_timing.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/trt_engine.h"

namespace ajp::gpu::cuda {
namespace {

// Un événement et un graphe rendus quoi qu'il arrive.
struct Event {
  cudaEvent_t event = nullptr;
  Event() = default;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (event != nullptr) cudaEventDestroy(event);
  }
};

struct Graph {
  cudaGraphExec_t exec = nullptr;
  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  ~Graph() {
    if (exec != nullptr) cudaGraphExecDestroy(exec);
  }
};

std::unexpected<Refusal> failed(cudaError_t error, const char* call) {
  if (const cudaError_t lost = clear_error(); lost != cudaSuccess && lost != error)
    return std::unexpected(lost_refusal(lost, call));
  return std::unexpected(cuda_refusal(error, std::format("mesure du temps du SPAN ({})", call)));
}

// Le corps du réseau FP8 capturé en graphe, comme SpanFp8Engine le rejoue en lecture. Une capture refusée
// laisse exec nul : les passes se lancent alors en direct, comme la lecture le ferait aussi.
void capture_body(SpanFp8Net& net, const void* input, cudaStream_t stream, Graph& graph) {
  std::string why;
  cudaGraph_t captured = nullptr;
  cudaError_t error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  bool queued = false;
  if (error == cudaSuccess) {
    queued = net.launch_body(stream, input, why);
    error = cudaStreamEndCapture(stream, &captured);
  }
  if (error == cudaSuccess && queued && captured != nullptr)
    error = cudaGraphInstantiate(&graph.exec, captured, 0);
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (error != cudaSuccess || !queued) {
    if (graph.exec != nullptr) cudaGraphExecDestroy(graph.exec);
    graph.exec = nullptr;
    static_cast<void>(clear_error());
  }
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<SpanTimes, Refusal> time_span_set(const SpanTimingSet& set, cudaStream_t stream,
                                                Progress& progress, float from, float to) {
  if (set.fp8 == nullptr || set.input == nullptr || set.output_fp8 == nullptr)
    return refuse(RefusalCode::ShapeInvalid, "mesure du temps du SPAN : réseau FP8 absent");
  std::array<Graph, 2> graphs;
  capture_body(*set.fp8, set.input, stream, graphs[0]);
  if (set.fp8_acc16 != nullptr) capture_body(*set.fp8_acc16, set.input, stream, graphs[1]);

  const Shape in_shape{.dims = {1, 3, set.height, set.width}};
  const Shape out_shape{.dims = {1, 3, std::int64_t{set.height} * 2, std::int64_t{set.width} * 2}};
  const std::array trt_in{Tensor{
      .type = TensorType::F16, .shape = in_shape, .handle = reinterpret_cast<std::uintptr_t>(set.input)}};
  const std::array trt_out{Tensor{.type = TensorType::F16,
                                  .shape = out_shape,
                                  .handle = reinterpret_cast<std::uintptr_t>(set.output_trt)}};

  // Une passe de la variante : 0 FP8, 1 FP8 f16, 2 TensorRT FP16.
  std::string why;
  const auto pass = [&](int variant) -> std::expected<void, Refusal> {
    if (variant == 2) return set.trt->enqueue(stream, nullptr, trt_in, trt_out);
    SpanFp8Net& net = variant == 0 ? *set.fp8 : *set.fp8_acc16;
    const Graph& graph = graphs[static_cast<std::size_t>(variant)];
    if (graph.exec != nullptr) {
      if (const cudaError_t e = cudaGraphLaunch(graph.exec, stream); e != cudaSuccess)
        return failed(e, "cudaGraphLaunch");
    } else if (!net.launch_body(stream, set.input, why)) {
      return refuse(RefusalCode::VendorCallFailed, "mesure du temps du SPAN FP8 : " + why);
    }
    if (!net.launch_tail(stream, set.output_fp8, SpanFp8Ring{}, why))
      return refuse(RefusalCode::VendorCallFailed, "mesure du temps du SPAN FP8 : " + why);
    return {};
  };

  Event start;
  Event stop;
  for (Event* event : {&start, &stop})
    if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess)
      return failed(e, "cudaEventCreate");
  // Un lot : chauffe, puis kTimingRuns passes entre deux événements ; le temps d'une passe.
  const auto batch = [&](int variant) -> std::expected<double, Refusal> {
    for (int i = 0; i < kTimingWarmup; ++i)
      if (auto ok = pass(variant); !ok) return std::unexpected(std::move(ok.error()));
    if (const cudaError_t e = cudaEventRecord(start.event, stream); e != cudaSuccess)
      return failed(e, "cudaEventRecord");
    for (int i = 0; i < kTimingRuns; ++i)
      if (auto ok = pass(variant); !ok) return std::unexpected(std::move(ok.error()));
    if (const cudaError_t e = cudaEventRecord(stop.event, stream); e != cudaSuccess)
      return failed(e, "cudaEventRecord");
    if (const cudaError_t e = cudaEventSynchronize(stop.event); e != cudaSuccess)
      return failed(e, "cudaEventSynchronize");
    float ms = 0.0F;
    if (const cudaError_t e = cudaEventElapsedTime(&ms, start.event, stop.event); e != cudaSuccess)
      return failed(e, "cudaEventElapsedTime");
    if (!(ms > 0.0F))
      return refuse(RefusalCode::VendorCallFailed, "mesure du temps du SPAN : lot mesuré à 0 ms");
    return static_cast<double>(ms) / kTimingRuns;
  };

  std::vector<int> variants{0};
  if (set.fp8_acc16 != nullptr) variants.push_back(1);
  if (set.trt != nullptr) variants.push_back(2);
  std::array<std::vector<double>, 3> times;
  const auto batches = static_cast<float>(kTimingRounds * static_cast<int>(variants.size()));
  float done = 0.0F;
  for (int round = 0; round < kTimingRounds; ++round) {
    // Un tour sur deux à rebours : aucune variante ne passe toujours la première après une pause.
    if (round > 0) std::ranges::reverse(variants);
    for (const int variant : variants) {
      if (progress.stop.stop_requested())
        return refuse(RefusalCode::Cancelled, "dérivation du SPAN FP8 interrompue");
      auto ms = batch(variant);
      if (!ms) return std::unexpected(std::move(ms.error()));
      times[static_cast<std::size_t>(variant)].push_back(*ms);
      done += 1.0F;
      if (progress.report) progress.report(from + (to - from) * done / batches, "mesure du temps");
    }
  }
  const auto median = [](std::vector<double>& of) {
    if (of.empty()) return 0.0;
    std::ranges::sort(of);
    return of[of.size() / 2];
  };
  return SpanTimes{.fp8_ms = median(times[0]), .fp8_acc16_ms = median(times[1]), .trt_ms = median(times[2])};
}

}  // namespace ajp::gpu::cuda
