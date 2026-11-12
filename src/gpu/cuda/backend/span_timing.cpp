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

// Les passes et les lots d'un jeu de variantes, sur un flux.
class Timer {
 public:
  Timer(const SpanTimingSet& set, cudaStream_t stream)
      : set_(set),
        stream_(stream),
        trt_in_{Tensor{.type = TensorType::F16,
                       .shape = Shape{.dims = {1, 3, set.height, set.width}},
                       .handle = reinterpret_cast<std::uintptr_t>(set.input)}},
        trt_out_{
            Tensor{.type = TensorType::F16,
                   .shape = Shape{.dims = {1, 3, std::int64_t{set.height} * 2, std::int64_t{set.width} * 2}},
                   .handle = reinterpret_cast<std::uintptr_t>(set.output_trt)}} {
    capture_body(*set.fp8, set.input, stream, graphs_[0]);
    if (set.fp8_acc16 != nullptr) capture_body(*set.fp8_acc16, set.input, stream, graphs_[1]);
  }

  std::expected<void, Refusal> create_events() {
    for (Event* event : {&start_, &stop_})
      if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess)
        return failed(e, "cudaEventCreate");
    return {};
  }

  // Un lot : chauffe, puis kTimingRuns passes entre deux événements ; le temps d'une passe.
  std::expected<double, Refusal> batch(int variant) {
    for (int i = 0; i < kTimingWarmup; ++i)
      if (auto ok = pass(variant); !ok) return std::unexpected(std::move(ok.error()));
    if (const cudaError_t e = cudaEventRecord(start_.event, stream_); e != cudaSuccess)
      return failed(e, "cudaEventRecord");
    for (int i = 0; i < kTimingRuns; ++i)
      if (auto ok = pass(variant); !ok) return std::unexpected(std::move(ok.error()));
    if (const cudaError_t e = cudaEventRecord(stop_.event, stream_); e != cudaSuccess)
      return failed(e, "cudaEventRecord");
    if (const cudaError_t e = cudaEventSynchronize(stop_.event); e != cudaSuccess)
      return failed(e, "cudaEventSynchronize");
    float ms = 0.0F;
    if (const cudaError_t e = cudaEventElapsedTime(&ms, start_.event, stop_.event); e != cudaSuccess)
      return failed(e, "cudaEventElapsedTime");
    if (!(ms > 0.0F))
      return refuse(RefusalCode::VendorCallFailed, "mesure du temps du SPAN : lot mesuré à 0 ms");
    return static_cast<double>(ms) / kTimingRuns;
  }

 private:
  // Une passe de la variante : 0 FP8, 1 FP8 f16, 2 TensorRT FP16.
  std::expected<void, Refusal> pass(int variant) {
    if (variant == 2) return set_.trt->enqueue(stream_, nullptr, trt_in_, trt_out_);
    SpanFp8Net& net = variant == 0 ? *set_.fp8 : *set_.fp8_acc16;
    const Graph& graph = graphs_[static_cast<std::size_t>(variant)];
    if (graph.exec != nullptr) {
      if (const cudaError_t e = cudaGraphLaunch(graph.exec, stream_); e != cudaSuccess)
        return failed(e, "cudaGraphLaunch");
    } else if (!net.launch_body(stream_, set_.input, why_)) {
      return refuse(RefusalCode::VendorCallFailed, "mesure du temps du SPAN FP8 : " + why_);
    }
    if (!net.launch_tail(stream_, set_.output_fp8, SpanFp8Ring{}, why_))
      return refuse(RefusalCode::VendorCallFailed, "mesure du temps du SPAN FP8 : " + why_);
    return {};
  }

  const SpanTimingSet& set_;
  cudaStream_t stream_;
  std::array<Tensor, 1> trt_in_;
  std::array<Tensor, 1> trt_out_;
  std::array<Graph, 2> graphs_;
  Event start_;
  Event stop_;
  std::string why_;
};

double median(std::vector<double>& of) {
  if (of.empty()) return 0.0;
  std::ranges::sort(of);
  return of[of.size() / 2];
}

}  // namespace

std::expected<SpanTimes, Refusal> time_span_set(const SpanTimingSet& set, cudaStream_t stream,
                                                Progress& progress, float from, float to) {
  if (set.fp8 == nullptr || set.input == nullptr || set.output_fp8 == nullptr)
    return refuse(RefusalCode::ShapeInvalid, "mesure du temps du SPAN : réseau FP8 absent");
  Timer timer(set, stream);
  if (auto ok = timer.create_events(); !ok) return std::unexpected(std::move(ok.error()));

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
      auto ms = timer.batch(variant);
      if (!ms) return std::unexpected(std::move(ms.error()));
      times[static_cast<std::size_t>(variant)].push_back(*ms);
      done += 1.0F;
      if (progress.report) progress.report(from + (to - from) * done / batches, "mesure du temps");
    }
  }
  return SpanTimes{.fp8_ms = median(times[0]), .fp8_acc16_ms = median(times[1]), .trt_ms = median(times[2])};
}

}  // namespace ajp::gpu::cuda
