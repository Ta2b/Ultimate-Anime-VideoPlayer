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
constexpr std::array kRefineRadii{kernels::scene::kRefineCoarseRadius, kernels::scene::kRefineFineRadius};

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
  for (RefineBlur& blur : refine_blur_) {
    if (auto ok = device(blur.radius[0], pixels); !ok) return ok;
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
  for (RefineBlur& blur : refine_blur_) {
    if (auto ok = device(blur.radius[1], pixels); !ok) {
      if (ok.error().code == RefusalCode::DeviceLost) return ok;
      // Sans place pour les deux rayons, recalculer les flous dans les premiers plans.
      static_cast<void>(cudaGetLastError());
      for (RefineBlur& cache : refine_blur_) {
        if (cache.radius[1] != nullptr) static_cast<void>(cudaFree(cache.radius[1]));
        cache.radius[1] = nullptr;
      }
      return device(refine_rows_, pixels);
    }
  }
  return {};
}

void SceneFlow::release_refine() {
  for (RefineBlur& blur : refine_blur_) {
    for (float* plane : blur.radius)
      if (plane != nullptr) static_cast<void>(cudaFree(plane));
    blur = {};
  }
  if (refine_rows_ != nullptr) static_cast<void>(cudaFree(refine_rows_));
  refine_rows_ = nullptr;
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
  const std::array slots{lumas.a, lumas.b};
  const bool reuse = refine_blur_[0].radius[1] != nullptr;
  const auto write_of = [&](int slot) { return writes_[static_cast<std::size_t>(slot)]; };
  const auto matches = [&](const RefineBlur& blur, int slot) {
    return blur.luma == slot && blur.write == write_of(slot);
  };
  if (reuse && (matches(refine_blur_[1], lumas.a) || matches(refine_blur_[0], lumas.b)))
    std::swap(refine_blur_[0], refine_blur_[1]);
  if (reuse) {
    for (std::size_t k = 0; k < slots.size(); ++k) {
      RefineBlur& blur = refine_blur_[k];
      if (blur.ready && matches(blur, slots[k])) continue;
      blur.ready = false;
      const auto luma = session_->luma(slots[k]);
      const SceneBoxPair boxes{.luma = static_cast<const unsigned char*>(luma.data),
                               .pitch = luma.pitch,
                               .width = width,
                               .height = height,
                               .coarse = blur.radius[0],
                               .fine = blur.radius[1]};
      if (auto ok = cuda_check(launch_scene_box_pair(boxes, own), "scene_box_pair<<<>>>"); !ok) return ok;
      blur.luma = slots[k];
      blur.write = write_of(slots[k]);
      blur.ready = true;
    }
  }
  for (std::size_t r = 0; r < kRefineRadii.size(); ++r) {
    const std::size_t plane = reuse ? r : 0;
    if (!reuse) {
      for (std::size_t k = 0; k < slots.size(); ++k) {
        const auto luma = session_->luma(slots[k]);
        const SceneBoxParams box{.luma = static_cast<const unsigned char*>(luma.data),
                                 .pitch = luma.pitch,
                                 .rows = refine_rows_,
                                 .out = refine_blur_[k].radius[0],
                                 .width = width,
                                 .height = height,
                                 .radius = kRefineRadii[r]};
        if (auto ok = cuda_check(launch_scene_box(box, own), "scene_box<<<>>>"); !ok) return ok;
      }
    }
    const SceneRefineParams step{.a = refine_blur_[0].radius[plane],
                                 .b = refine_blur_[1].radius[plane],
                                 .width = width,
                                 .height = height,
                                 .cols = cols_,
                                 .decor = refine_decor_,
                                 .state = refine_state_};
    for (int i = 0; i < kernels::scene::kRefineSteps; ++i) {
      if (auto ok = cuda_check(launch_scene_refine_step(step, own), "scene_refine<<<>>>"); !ok) return ok;
    }
  }
  const std::size_t fine = reuse ? kRefineRadii.size() - 1 : 0;
  const SceneCheckParams check{.a = refine_blur_[0].radius[fine],
                               .b = refine_blur_[1].radius[fine],
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
  // Sur la file de NVOF, derrière le flux de la paire (ses tuiles relues ou non) : les lumas sont là.
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
