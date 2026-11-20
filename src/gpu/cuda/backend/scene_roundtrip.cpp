#include "gpu/cuda/backend/scene_roundtrip.h"

#include <cstddef>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/scene_roundtrip_kernels.h"

namespace kirari::gpu::cuda {

std::expected<std::unique_ptr<SceneRoundtrip>, Refusal> SceneRoundtrip::open(int width, int height,
                                                                             int divisor) {
  auto reverse = NvofSession::open(width, height, false);
  if (!reverse) return std::unexpected(std::move(reverse.error()));
  std::unique_ptr<SceneRoundtrip> out(new SceneRoundtrip());
  out->reverse_ = std::move(*reverse);
  out->shape_ = {width, height};
  out->divisor_ = divisor;
  if (auto allocated = out->allocate(); !allocated) return std::unexpected(std::move(allocated.error()));
  return out;
}

std::expected<void, Refusal> SceneRoundtrip::allocate() {
  const auto bytes = tile_count() * sizeof(kernels::SceneRoundtripTile);
  if (auto ok = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host_), bytes), "cudaMallocHost"); !ok)
    return ok;
  if (auto ok = cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_), bytes), "cudaMalloc"); !ok)
    return ok;
  if (auto ok =
          cuda_check(cudaMalloc(reinterpret_cast<void**>(&forward_), shape_.words() * sizeof(std::int16_t)),
                     "cudaMalloc");
      !ok)
    return ok;
  if (auto ok =
          cuda_check(cudaEventCreateWithFlags(&inputs_ready_, cudaEventDisableTiming), "cudaEventCreate");
      !ok)
    return ok;
  if (auto ok = cuda_check(cudaEventCreate(&begin_), "cudaEventCreate"); !ok) return ok;
  return cuda_check(cudaEventCreate(&done_), "cudaEventCreate");
}

SceneRoundtrip::~SceneRoundtrip() {
  reverse_.reset();  // le propriétaire a déjà vidé la file aller, qui lit et écrit ces tampons
  if (forward_ != nullptr) static_cast<void>(cudaFree(forward_));
  if (device_ != nullptr) static_cast<void>(cudaFree(device_));
  if (host_ != nullptr) static_cast<void>(cudaFreeHost(host_));
  for (const cudaEvent_t event : {inputs_ready_, begin_, done_}) {
    if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
  }
}

std::size_t SceneRoundtrip::tile_count() const {
  return static_cast<std::size_t>(kernels::scene_flow_tiles(shape_.width)) *
         static_cast<std::size_t>(kernels::scene_flow_tiles(shape_.height));
}

std::expected<void, Refusal> SceneRoundtrip::enqueue(const NvofSession& forward, int a, int b) {
  const cudaStream_t input = forward.stream();
  const cudaStream_t output = reverse_->stream();
  // Les événements de la scène couvrent la lecture des deux tampons. Ensuite le retour possède ses lumas.
  for (const auto& [from, into] : {std::pair{a, 0}, std::pair{b, 1}}) {
    const auto src = forward.luma(from);
    const auto dst = reverse_->luma(into);
    if (auto ok =
            cuda_check(cudaMemcpy2DAsync(
                           dst.data, dst.pitch, src.data, src.pitch, static_cast<std::size_t>(shape_.width),
                           static_cast<std::size_t>(shape_.height), cudaMemcpyDeviceToDevice, input),
                       "cudaMemcpy2DAsync (lumas NVOF)");
        !ok)
      return ok;
  }
  const auto vectors = forward.flow();
  const std::size_t pitch = shape_.row_words() * sizeof(std::int16_t);
  if (auto ok = cuda_check(
          cudaMemcpy2DAsync(forward_, pitch, vectors.data, vectors.pitch, pitch,
                            static_cast<std::size_t>(shape_.rows()), cudaMemcpyDeviceToDevice, input),
          "cudaMemcpy2DAsync (aller NVOF)");
      !ok)
    return ok;
  if (auto ok = cuda_check(cudaEventRecord(inputs_ready_, input), "cudaEventRecord"); !ok) return ok;
  if (auto ok = cuda_check(cudaStreamWaitEvent(output, inputs_ready_, 0), "cudaStreamWaitEvent"); !ok)
    return ok;
  if (auto ok = cuda_check(cudaEventRecord(begin_, output), "cudaEventRecord"); !ok) return ok;
  if (auto ok = reverse_->execute(1, 0, true); !ok) return ok;
  const auto plane = [](NvofSession::Pitched view) {
    return SceneRoundtripParams::Plane{.pixels = view.data, .pitch = view.pitch};
  };
  const SceneRoundtripParams params{.luma_a = plane(reverse_->luma(0)),
                                    .luma_b = plane(reverse_->luma(1)),
                                    .flow_a = {.pixels = forward_, .pitch = pitch},
                                    .flow_b = plane(reverse_->flow()),
                                    .width = shape_.width,
                                    .height = shape_.height,
                                    .out = device_};
  if (auto ok = cuda_check(launch_scene_roundtrip(params, output), "scene_roundtrip<<<>>>"); !ok) return ok;
  if (auto ok = cuda_check(
          cudaMemcpyAsync(host_, device_, tile_count() * sizeof(*host_), cudaMemcpyDeviceToHost, output),
          "cudaMemcpyAsync (grille NVOF)");
      !ok)
    return ok;
  return cuda_check(cudaEventRecord(done_, output), "cudaEventRecord");
}

bool SceneRoundtrip::start(const NvofSession& forward, int a, int b, std::uint64_t pair) {
  poll();
  if (failure_) return false;
  if (!slot_.accepts(pair)) {
    ++busy_;
    return false;
  }
  if (auto queued = enqueue(forward, a, b); !queued) {
    failure_ = std::move(queued.error());
  } else {
    slot_.start(pair);
    ++started_;
#ifdef _WIN32
    static_cast<void>(cudaStreamQuery(reverse_->stream()));  // pousse le lot WDDM comme la file aller
#endif
  }
  return true;
}

void SceneRoundtrip::poll() {
  if (failure_ || slot_.pending() == 0) return;
  const cudaError_t done = cudaEventQuery(done_);
  if (done == cudaErrorNotReady) return;
  if (auto ok = cuda_check(done, "cudaEventQuery"); !ok) {
    failure_ = std::move(ok.error());
    return;
  }
  completed_ = kernels::SceneRoundtripTiles{.ticket = slot_.take(),
                                            .flow_width = shape_.width,
                                            .flow_height = shape_.height,
                                            .divisor = divisor_,
                                            .tiles = {host_, host_ + tile_count()}};
  std::size_t visible = 0;
  for (const auto& tile : completed_->tiles) {
    visible += static_cast<std::size_t>(tile.visible_a) + tile.visible_b;
    stats_.open += static_cast<std::size_t>(tile.open_a) + tile.open_b;
    photo_ += static_cast<std::size_t>(tile.photo_a) + tile.photo_b;
  }
  stats_.checked += visible;
  stats_.outside += shape_.words() - visible;  // deux sens, un mot par projection à compter
  float cost = 0.0F;
  if (cudaEventElapsedTime(&cost, begin_, done_) == cudaSuccess) measured_ms_ += static_cast<double>(cost);
  ++read_;
}

std::expected<std::optional<kernels::SceneRoundtripTiles>, Refusal> SceneRoundtrip::take(std::uint64_t pair) {
  poll();
  if (completed_ && completed_->ticket == pair) return std::exchange(completed_, {});
  if (!failure_ && pair != 0 && slot_.pending() == pair) return std::optional<kernels::SceneRoundtripTiles>{};
  return refuse(RefusalCode::SequenceOrder, "scene_roundtrip_take : retour indisponible pour cette paire");
}

std::string SceneRoundtrip::finish() {
  if (!failure_ && slot_.pending() != 0) {
    if (auto ok = cuda_check(cudaStreamSynchronize(reverse_->stream()), "cudaStreamSynchronize"); !ok)
      failure_ = std::move(ok.error());
    poll();
  }
  return std::format(
      "NVOF aller-retour : grille {}×{}, {} lancés, {} relus, {} sautés (retour occupé), "
      "{} fermetures ouvertes sur {} contrôlées ({} avec écart de luma), {} hors champ, retour et relecture "
      "{:.3f} ms{}",
      shape_.cols(), shape_.rows(), started_, read_, busy_, stats_.open, stats_.checked, photo_,
      stats_.outside, read_ > 0 ? measured_ms_ / static_cast<double>(read_) : 0.0,
      failure_ ? " ; arrêté : " + failure_->detail : "");
}

}  // namespace kirari::gpu::cuda
