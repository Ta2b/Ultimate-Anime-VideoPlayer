// L'affinage de la caméra de la carte dans le backend TensorRT (SceneFlow::refine et refined, scene_flow.h ;
// kernels/spec/scene_camera.h). Sur la file de NVOF, après la paire elle-même : les lumas de la paire,
// floutées, les pas de Gauss-Newton et la vérification par tuile, puis « relâché » reposé sur les deux
// tampons (une paire qui les réécrirait attend l'affinage) et la copie vers la mémoire hôte épinglée. Aucune
// attente hôte, aucun retour à l'hôte entre les pas.
#include <array>
#include <cstddef>
#include <cstring>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/scene_camera_kernels.h"
#include "gpu/cuda/backend/scene_flow.h"

namespace kirari::gpu::cuda {
namespace {

static_assert(offsetof(SceneRefineState, model) == 0);

std::size_t tile_count(int cols, int rows) {
  return static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows);
}

template <class T>
std::expected<void, Refusal> pinned(T*& out, std::size_t count) {
  return cuda_check(cudaMallocHost(reinterpret_cast<void**>(&out), count * sizeof(T)), "cudaMallocHost");
}

template <class T>
std::expected<void, Refusal> device(T*& out, std::size_t count) {
  return cuda_check(cudaMalloc(reinterpret_cast<void**>(&out), count * sizeof(T)), "cudaMalloc");
}

}  // namespace

std::expected<void, Refusal> SceneFlow::allocate_refine() {
  const std::size_t tiles = tile_count(cols_, rows_);
  const std::size_t pixels =
      static_cast<std::size_t>(session_->width()) * static_cast<std::size_t>(session_->height());
  for (float*& plane : blurred_) {
    if (auto ok = device(plane, pixels); !ok) return ok;
  }
  if (auto ok = device(refine_state_, 1); !ok) return ok;
  if (auto ok = device(refine_decor_, tiles); !ok) return ok;
  if (auto ok = device(refine_checks_, tiles); !ok) return ok;
  for (Refine& refine : refines_) {
    if (auto ok = pinned(refine.model, 6); !ok) return ok;
    if (auto ok = pinned(refine.decor, tiles); !ok) return ok;
    if (auto ok = pinned(refine.state, 1); !ok) return ok;
    if (auto ok = pinned(refine.checks, tiles); !ok) return ok;
    if (refine.done == nullptr) {
      if (auto ok =
              cuda_check(cudaEventCreateWithFlags(&refine.done, cudaEventDisableTiming), "cudaEventCreate");
          !ok)
        return ok;
    }
  }
  return {};
}

void SceneFlow::release_refine() {
  for (float*& plane : blurred_) {
    if (plane != nullptr) static_cast<void>(cudaFree(plane));
    plane = nullptr;
  }
  for (void* block : {static_cast<void*>(refine_state_), static_cast<void*>(refine_decor_),
                      static_cast<void*>(refine_checks_)}) {
    if (block != nullptr) static_cast<void>(cudaFree(block));
  }
  refine_state_ = nullptr;
  refine_decor_ = nullptr;
  refine_checks_ = nullptr;
  for (Refine& refine : refines_) {
    for (void* block : {static_cast<void*>(refine.model), static_cast<void*>(refine.decor),
                        static_cast<void*>(refine.state), static_cast<void*>(refine.checks)}) {
      if (block != nullptr) static_cast<void>(cudaFreeHost(block));
    }
    refine.model = nullptr;
    refine.decor = nullptr;
    refine.state = nullptr;
    refine.checks = nullptr;
    refine.ticket = 0;
    refine.posted = false;
  }
}

std::expected<void, Refusal> SceneFlow::enqueue_refine(cudaStream_t own, const Lumas& lumas) {
  const int width = session_->width();
  const int height = session_->height();
  const std::array<NvofSession::Pitched, 2> source{session_->luma(lumas.a), session_->luma(lumas.b)};
  const SceneRefineParams step{.a = blurred_[0],
                               .b = blurred_[1],
                               .width = width,
                               .height = height,
                               .cols = cols_,
                               .decor = refine_decor_,
                               .state = refine_state_};
  for (const int radius : {kernels::scene::kRefineCoarseRadius, kernels::scene::kRefineFineRadius}) {
    for (std::size_t k = 0; k < 2; ++k) {
      const SceneBoxParams box{.luma = static_cast<const unsigned char*>(source[k].data),
                               .pitch = source[k].pitch,
                               .rows = blurred_[2],
                               .out = blurred_[k],
                               .width = width,
                               .height = height,
                               .radius = radius};
      if (auto ok = cuda_check(launch_scene_box(box, own), "scene_box<<<>>>"); !ok) return ok;
    }
    for (int i = 0; i < kernels::scene::kRefineSteps; ++i) {
      if (auto ok = cuda_check(launch_scene_refine_step(step, own), "scene_refine<<<>>>"); !ok) return ok;
    }
  }
  const SceneCheckParams check{.a = blurred_[0],
                               .b = blurred_[1],
                               .width = width,
                               .height = height,
                               .cols = cols_,
                               .rows = rows_,
                               .state = refine_state_,
                               .out = refine_checks_};
  return cuda_check(launch_scene_check(check, own), "scene_check<<<>>>");
}

std::expected<std::uint64_t, Refusal> SceneFlow::refine(std::uint64_t flow_ticket,
                                                        const kernels::SceneCameraRequest& request,
                                                        GpuTiming& timing) {
  const Launched& pair = launched_[flow_ticket % launched_.size()];
  // Trop ancienne (une autre grille depuis, prepare), ou ses lumas réécrites par une paire plus récente :
  // rien à affiner.
  if (flow_ticket == 0 || pair.ticket != flow_ticket ||
      writes_[static_cast<std::size_t>(pair.lumas.a)] != pair.writes[0] ||
      writes_[static_cast<std::size_t>(pair.lumas.b)] != pair.writes[1])
    return std::uint64_t{0};
  const std::size_t tiles = tile_count(cols_, rows_);
  if (request.decor.size() != tiles) {
    return refuse(RefusalCode::ShapeInvalid, std::format("scene_camera_start : {} tuiles de décor pour {}",
                                                         request.decor.size(), tiles));
  }
  const std::uint64_t ticket = refine_tickets_ + 1;
  Refine& slot = refines_[ticket % refines_.size()];
  if (slot.posted) {
    const cudaError_t state = cudaEventQuery(slot.done);
    if (state == cudaErrorNotReady) return std::uint64_t{0};  // la relecture d'avant est encore en vol
    if (auto ok = cuda_check(state, "cudaEventQuery"); !ok) return std::unexpected(std::move(ok.error()));
  }
  std::memcpy(slot.model, request.model.data(), sizeof(double) * 6);
  std::memcpy(slot.decor, request.decor.data(), tiles);
  cudaStream_t own = session_->stream();
  // La file de NVOF a déjà passé la paire (ses tuiles sont relues) : les lumas sont là.
  timing.begin(own);
  std::expected<void, Refusal> ready =
      cuda_check(cudaMemsetAsync(refine_state_, 0, sizeof(SceneRefineState), own), "cudaMemsetAsync");
  if (ready) {
    // Le modèle est en tête de l'état (static_assert ci-dessus).
    ready = cuda_check(
        cudaMemcpyAsync(refine_state_, slot.model, sizeof(double) * 6, cudaMemcpyHostToDevice, own),
        "cudaMemcpyAsync");
  }
  if (ready) {
    ready = cuda_check(cudaMemcpyAsync(refine_decor_, slot.decor, tiles, cudaMemcpyHostToDevice, own),
                       "cudaMemcpyAsync");
  }
  if (ready) ready = enqueue_refine(own, pair.lumas);
  timing.end(own);
  // Les deux lumas sont lues : une paire qui réécrirait leurs tampons attend l'affinage.
  for (const int luma : {pair.lumas.a, pair.lumas.b}) {
    if (ready)
      ready = cuda_check(cudaEventRecord(released_[static_cast<std::size_t>(luma)], own), "cudaEventRecord");
  }
  if (ready) {
    ready = cuda_check(
        cudaMemcpyAsync(slot.state, refine_state_, sizeof(SceneRefineState), cudaMemcpyDeviceToHost, own),
        "cudaMemcpyAsync");
  }
  if (ready) {
    ready = cuda_check(cudaMemcpyAsync(slot.checks, refine_checks_, tiles * sizeof(kernels::SceneTileCheck),
                                       cudaMemcpyDeviceToHost, own),
                       "cudaMemcpyAsync");
  }
  if (ready) ready = cuda_check(cudaEventRecord(slot.done, own), "cudaEventRecord");
  if (!ready) return std::unexpected(std::move(ready.error()));
#ifdef _WIN32
  static_cast<void>(cudaStreamQuery(own));  // WDDM : la file poussée tout de suite (SceneFlow::run)
#endif
  slot.posted = true;
  slot.ticket = ticket;
  refine_tickets_ = ticket;
  return ticket;
}

std::expected<std::optional<kernels::SceneCameraRefined>, Refusal> SceneFlow::refined(std::uint64_t ticket) {
  Refine& slot = refines_[ticket % refines_.size()];
  if (ticket == 0 || slot.ticket != ticket || slot.state == nullptr)
    return refuse(RefusalCode::SequenceOrder, std::format("scene_camera_take : ticket {} périmé", ticket));
  const cudaError_t state = cudaEventQuery(slot.done);
  if (state == cudaErrorNotReady) return std::optional<kernels::SceneCameraRefined>{};
  if (auto ok = cuda_check(state, "cudaEventQuery"); !ok) return std::unexpected(std::move(ok.error()));
  slot.ticket = 0;
  kernels::SceneCameraRefined out;
  std::memcpy(out.model.data(), slot.state->model, sizeof(double) * 6);
  out.steps = slot.state->steps;
  out.tiles.resize(tile_count(cols_, rows_));
  std::memcpy(out.tiles.data(), slot.checks, out.tiles.size() * sizeof(kernels::SceneTileCheck));
  return std::optional(std::move(out));
}

}  // namespace kirari::gpu::cuda
