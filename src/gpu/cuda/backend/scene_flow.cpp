// La carte de la scène dans le backend TensorRT (scene_flow.h).
#include "gpu/cuda/backend/scene_flow.h"

#include <cstring>
#include <format>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/scene_flow_kernels.h"

namespace kirari::gpu::cuda {
namespace {

std::size_t tile_count(int cols, int rows) {
  return static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows);
}

}  // namespace

SceneFlow::~SceneFlow() {
  session_.reset();  // attend la file de NVOF
  release_grid();
  if (lumas_ready_ != nullptr) static_cast<void>(cudaEventDestroy(lumas_ready_));
  for (cudaEvent_t event : released_) {
    if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
  }
  for (const Read& read : reads_) {
    if (read.done != nullptr) static_cast<void>(cudaEventDestroy(read.done));
  }
  for (const Refine& refine : refines_) {
    if (refine.done != nullptr) static_cast<void>(cudaEventDestroy(refine.done));
  }
}

void SceneFlow::release_grid() {
  if (device_ != nullptr) static_cast<void>(cudaFree(device_));
  device_ = nullptr;
  for (Read& read : reads_) {
    if (read.host != nullptr) static_cast<void>(cudaFreeHost(read.host));
    read.host = nullptr;
    read.ticket = 0;
  }
  release_refine();
  cols_ = 0;
  rows_ = 0;
}

std::expected<bool, Refusal> SceneFlow::prepare(int flow_width, int flow_height, int divisor) {
  if (refused_) return std::unexpected(*refused_);
  if (session_ && session_->width() == flow_width && session_->height() == flow_height && divisor_ == divisor)
    return false;
  // Une autre taille : l'ancienne session finit son travail, puis tout est refait.
  session_.reset();
  released_posted_ = {};
  last_.reset();
  launched_ = {};
  writes_ = {};
  release_grid();
  auto opened = NvofSession::open(flow_width, flow_height);
  if (!opened) {
    // La bibliothèque, le GPU ou la taille ne changeront pas en cours de lecture : on ne réessaie pas.
    refused_ = opened.error();
    return std::unexpected(std::move(opened.error()));
  }
  session_ = std::move(*opened);
  divisor_ = divisor;
  cols_ = kernels::scene_flow_tiles(flow_width);
  rows_ = kernels::scene_flow_tiles(flow_height);
  if (auto ok = allocate_grid(); !ok) {
    // À moitié prête, la session ne servirait pas : tout est rendu, et la carte s'arrête là.
    session_.reset();
    release_grid();
    refused_ = ok.error();
    return std::unexpected(std::move(ok.error()));
  }
  return true;
}

std::expected<void, Refusal> SceneFlow::allocate_grid() {
  const std::size_t bytes = tile_count(cols_, rows_) * sizeof(kernels::SceneTileSums);
  if (auto ok = cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_), bytes), "cudaMalloc"); !ok)
    return ok;
  for (Read& read : reads_) {
    if (auto ok = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&read.host), bytes), "cudaMallocHost");
        !ok)
      return ok;
    if (read.done == nullptr) {
      if (auto ok =
              cuda_check(cudaEventCreateWithFlags(&read.done, cudaEventDisableTiming), "cudaEventCreate");
          !ok)
        return ok;
    }
  }
  for (cudaEvent_t* event : {&lumas_ready_, &released_[0], &released_[1], &released_[2]}) {
    if (*event != nullptr) continue;
    if (auto ok = cuda_check(cudaEventCreateWithFlags(event, cudaEventDisableTiming), "cudaEventCreate"); !ok)
      return ok;
  }
  return allocate_refine();
}

std::expected<std::optional<SceneFlow::Lumas>, Refusal> SceneFlow::plan(bool follows) const {
  static_assert(NvofSession::kLumas == 3);
  Lumas lumas;
  if (last_) {
    const int third = 3 - last_->a - last_->b;  // le tampon que la dernière paire n'a pas lu
    lumas = follows ? Lumas{.a = last_->b, .b = third, .keep_a = true} : Lumas{.a = third, .b = last_->a};
  }
  for (const int slot : {lumas.a, lumas.b}) {
    if (slot == lumas.a && lumas.keep_a) continue;
    const auto i = static_cast<std::size_t>(slot);
    if (!released_posted_[i]) continue;
    const cudaError_t state = cudaEventQuery(released_[i]);
    if (state == cudaErrorNotReady) return std::optional<Lumas>{};
    if (auto ok = cuda_check(state, "cudaEventQuery"); !ok) return std::unexpected(std::move(ok.error()));
  }
  return std::optional(lumas);
}

std::expected<void, Refusal> SceneFlow::before_lumas(cudaStream_t lane, const Lumas& lumas) {
  for (const int slot : {lumas.a, lumas.b}) {
    if (slot == lumas.a && lumas.keep_a) continue;
    const auto i = static_cast<std::size_t>(slot);
    if (!released_posted_[i]) continue;
    if (auto ok = cuda_check(cudaStreamWaitEvent(lane, released_[i], 0), "cudaStreamWaitEvent"); !ok)
      return ok;
  }
  return {};
}

std::expected<std::uint64_t, Refusal> SceneFlow::run(cudaStream_t lane, const Lumas& lumas, bool restart,
                                                     GpuTiming& flow_timing, GpuTiming& tiles_timing) {
  cudaStream_t own = session_->stream();
  if (auto ok = cuda_check(cudaEventRecord(lumas_ready_, lane), "cudaEventRecord"); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = cuda_check(cudaStreamWaitEvent(own, lumas_ready_, 0), "cudaStreamWaitEvent"); !ok)
    return std::unexpected(std::move(ok.error()));
  flow_timing.begin(own);
  std::expected<void, Refusal> ready = session_->execute(lumas.a, lumas.b, restart);
  flow_timing.end(own);
  tiles_timing.begin(own);
  if (ready) {
    const NvofSession::Pitched a = session_->luma(lumas.a);
    const NvofSession::Pitched b = session_->luma(lumas.b);
    const NvofSession::Pitched flow = session_->flow();
    const NvofSession::Pitched cost = session_->cost();
    const SceneTilesParams params{.a = static_cast<const unsigned char*>(a.data),
                                  .a_pitch = a.pitch,
                                  .b = static_cast<const unsigned char*>(b.data),
                                  .b_pitch = b.pitch,
                                  .width = session_->width(),
                                  .height = session_->height(),
                                  .flow = static_cast<const short2*>(flow.data),
                                  .flow_pitch = flow.pitch,
                                  .cost = static_cast<const unsigned char*>(cost.data),
                                  .cost_pitch = cost.pitch,
                                  .cols = cols_,
                                  .rows = rows_,
                                  .out = device_};
    ready = cuda_check(launch_scene_tiles(params, own), "scene_tiles<<<>>>");
  }
  tiles_timing.end(own);
  // Les deux lumas sont lues (flux et réduction) : la voie peut réécrire leurs tampons.
  for (const int slot : {lumas.a, lumas.b}) {
    if (ready)
      ready = cuda_check(cudaEventRecord(released_[static_cast<std::size_t>(slot)], own), "cudaEventRecord");
  }
  if (!ready) return std::unexpected(std::move(ready.error()));
  released_posted_[static_cast<std::size_t>(lumas.a)] = true;
  released_posted_[static_cast<std::size_t>(lumas.b)] = true;
  last_ = lumas;
  const std::uint64_t ticket = tickets_ + 1;
  // Les tampons que la voie vient d'écrire : un affinage de cette paire ne vaut que tant qu'ils restent.
  if (!lumas.keep_a) ++writes_[static_cast<std::size_t>(lumas.a)];
  ++writes_[static_cast<std::size_t>(lumas.b)];
  launched_[ticket % launched_.size()] = Launched{
      .ticket = ticket,
      .lumas = lumas,
      .writes = {writes_[static_cast<std::size_t>(lumas.a)], writes_[static_cast<std::size_t>(lumas.b)]}};
  Read& read = reads_[ticket % reads_.size()];
  const std::size_t bytes = tile_count(cols_, rows_) * sizeof(kernels::SceneTileSums);
  ready =
      cuda_check(cudaMemcpyAsync(read.host, device_, bytes, cudaMemcpyDeviceToHost, own), "cudaMemcpyAsync");
  if (ready) ready = cuda_check(cudaEventRecord(read.done, own), "cudaEventRecord");
  if (!ready) return std::unexpected(std::move(ready.error()));
#ifdef _WIN32
  // WDDM garde les lancements dans un lot logiciel : la file de NVOF est poussée tout de suite, comme la voie
  // à chaque étape (TensorRtBackend::push_launches). « Pas prête » est la réponse normale.
  static_cast<void>(cudaStreamQuery(own));
#endif
  tickets_ = ticket;
  read.ticket = ticket;
  return ticket;
}

std::expected<std::optional<kernels::SceneFlowTiles>, Refusal> SceneFlow::take(std::uint64_t ticket) {
  Read& read = reads_[ticket % reads_.size()];
  if (ticket == 0 || read.ticket != ticket || read.host == nullptr)
    return refuse(RefusalCode::SequenceOrder, std::format("scene_flow_take : ticket {} périmé", ticket));
  const cudaError_t state = cudaEventQuery(read.done);
  if (state == cudaErrorNotReady) return std::optional<kernels::SceneFlowTiles>{};
  if (auto ok = cuda_check(state, "cudaEventQuery"); !ok) return std::unexpected(std::move(ok.error()));
  read.ticket = 0;
  kernels::SceneFlowTiles tiles{.divisor = divisor_,
                                .flow_width = session_->width(),
                                .flow_height = session_->height(),
                                .cols = cols_,
                                .rows = rows_,
                                .tiles = {}};
  tiles.tiles.resize(tile_count(cols_, rows_));
  std::memcpy(tiles.tiles.data(), read.host, tiles.tiles.size() * sizeof(kernels::SceneTileSums));
  return std::optional(std::move(tiles));
}

}  // namespace kirari::gpu::cuda
