// Voies de calcul du backend TensorRT (gpu/lanes.h) : une file CUDA non bloquante par voie, et
// les dépendances entre elles d'après les ressources de chaque opération (LaneHazards). Chaque
// opération d'une voie y laisse un événement, posé à l'opération suivante de la voie (sa fin est
// alors enfilée) ou quand une autre voie doit l'attendre. kLaneMarks événements par voie servent
// à tour de rôle : un rang ancien se lit sur un événement plus récent de la même file, attente
// plus forte, jamais plus faible.
#include <chrono>
#include <thread>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/cuda/backend/trt_engine.h"

namespace ajp::gpu::cuda {
namespace {

// L'interrogation avant l'attente bloquante : une mesure enfilée une paire en avance est
// presque toujours finie avant.
constexpr auto kQueryBeforeWait = std::chrono::microseconds(200);

}  // namespace

std::expected<void, Refusal> TensorRtBackend::create_lanes() {
  // 8 Go et plus : trois voies ; sinon toutes passent par la file de l'agrandissement.
  lanes_ = properties_.totalGlobalMem >= kLaneMemory ? static_cast<int>(kLanes) : 1;
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    if (auto ok = cuda_check(cudaStreamCreateWithFlags(&streams_[lane], cudaStreamNonBlocking),
                             "cudaStreamCreateWithFlags");
        !ok) {
      return ok;
    }
    for (cudaEvent_t& event : lane_events_[lane]) {
      if (auto ok = cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
          !ok) {
        return ok;
      }
    }
  }
  lane_ = Lane::Upscale;
  stream_ = streams_[lane_index(lane_)];
  return {};
}

void TensorRtBackend::destroy_lanes() {
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    if (streams_[lane] != nullptr) cudaStreamSynchronize(streams_[lane]);
  }
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    for (cudaEvent_t event : lane_events_[lane]) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    if (streams_[lane] != nullptr) cudaStreamDestroy(streams_[lane]);
    streams_[lane] = nullptr;
  }
  stream_ = nullptr;
}

void TensorRtBackend::use_lane(Lane lane) {
  lane_ = lanes_ > 1 ? lane : Lane::Upscale;
  if (streams_[lane_index(lane_)] != nullptr) stream_ = streams_[lane_index(lane_)];
}

const std::shared_ptr<SharedActivations>& TensorRtBackend::arena() {
  const std::size_t zone = lanes_ > 1 && lane_ != Lane::Upscale ? 1 : 0;
  if (!arenas_[zone]) arenas_[zone] = std::make_shared<SharedActivations>();
  return arenas_[zone];
}

std::expected<void, Refusal> TensorRtBackend::record(std::size_t lane) {
  const std::uint64_t mark = std::exchange(unrecorded_[lane], 0);
  if (mark == 0) return {};
  return cuda_check(cudaEventRecord(lane_events_[lane][mark % kLaneMarks], streams_[lane]),
                    "cudaEventRecord");
}

std::expected<void, Refusal> TensorRtBackend::apply(const LaneHazards::Op& op) {
  const std::size_t self = lane_index(lane_);
  if (auto ok = record(self); !ok) return ok;  // la fin de l'opération précédente de cette voie
  for (const LaneHazards::Wait& wait : op.waits) {
    const std::size_t other = lane_index(wait.lane);
    // La dernière opération de l'autre voie n'a pas encore son événement : posé maintenant, après
    // ce qu'elle a enfilé.
    if (unrecorded_[other] != 0 && wait.mark >= unrecorded_[other]) {
      if (auto ok = record(other); !ok) return ok;
    }
    if (auto ok = cuda_check(cudaStreamWaitEvent(stream_, lane_events_[other][wait.mark % kLaneMarks], 0),
                             "cudaStreamWaitEvent");
        !ok) {
      return ok;
    }
  }
  unrecorded_[self] = op.mark;
  return {};
}

std::expected<void, Refusal> TensorRtBackend::order(std::initializer_list<std::uintptr_t> reads,
                                                    std::initializer_list<std::uintptr_t> writes) {
  return order(std::span(reads.begin(), reads.size()), std::span(writes.begin(), writes.size()));
}

std::expected<void, Refusal> TensorRtBackend::order(std::span<const std::uintptr_t> reads,
                                                    std::span<const std::uintptr_t> writes) {
  if (lanes_ == 1) return {};
  return apply(hazards_.access(lane_, reads, writes));
}

std::expected<void, Refusal> TensorRtBackend::join() {
  if (lanes_ == 1) return {};
  return apply(hazards_.join(lane_));
}

std::expected<void, Refusal> TensorRtBackend::wait_event(cudaEvent_t event) {
  const auto until = std::chrono::steady_clock::now() + kQueryBeforeWait;
  while (true) {
    const cudaError_t state = cudaEventQuery(event);
    if (state == cudaSuccess) return {};
    if (state != cudaErrorNotReady) return cuda_check(state, "cudaEventQuery");
    if (std::chrono::steady_clock::now() >= until) break;
    std::this_thread::yield();
  }
  return cuda_check(cudaEventSynchronize(event), "cudaEventSynchronize");
}

}  // namespace ajp::gpu::cuda
