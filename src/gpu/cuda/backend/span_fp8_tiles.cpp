// Les tuiles qui bougent du SPAN FP16 natif (kernels/spec/span_tiles.h, span_fp8_engine.h) : la demande du
// lecteur, le décalage de la sortie d'avant et l'envoi des masques de chaque couche puis du cœur. Les masques
// passent par une mémoire hôte épinglée tournante : la copie lit l'hôte quand le flux y arrive, un
// emplacement n'est réécrit qu'une fois sa copie faite (son événement).
#include <algorithm>
#include <cstring>
#include <format>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/span_fp8_engine.h"

namespace kirari::gpu::cuda {

kernels::SpanTileGrid SpanFp8Engine::span_tile_grid() const {
  const kernels::SpanTileGrid grid = net_ ? net_->tile_grid() : kernels::SpanTileGrid{};
  return grid.usable() ? grid : kernels::SpanTileGrid{};
}

bool SpanFp8Engine::next_span_tiles(std::span<const std::uint8_t> core, int shift_x, int shift_y,
                                    const Tensor* previous) {
  tile_job_.reset();
  const kernels::SpanTileGrid grid = span_tile_grid();
  if (!grid.accepts_shift(shift_x, shift_y) || core.size() != static_cast<std::size_t>(grid.count()))
    return false;
  if ((shift_x != 0 || shift_y != 0) && previous == nullptr) return false;
  if (previous != nullptr && (previous->type != TensorType::F16 || !(previous->shape == output_)))
    return false;
  // Appelé par le lecteur sans le filet du backend (guarded_backend.h) : rien ne sort du module.
  try {
    tile_job_ = TileJob{.core = std::vector<std::uint8_t>(core.begin(), core.end()),
                        .shift_x = shift_x,
                        .shift_y = shift_y,
                        .previous = previous != nullptr ? previous->handle + previous->offset : 0,
                        .source = previous != nullptr ? *previous : Tensor{},
                        .reads = previous != nullptr};
  } catch (...) {
    tile_job_.reset();
    return false;
  }
  return true;
}

std::expected<bool, Refusal> SpanFp8Engine::start_tiles(cudaStream_t stream, std::uintptr_t output) {
  std::optional<TileJob> job = std::exchange(tile_job_, std::nullopt);
  if (!job) return false;
  const kernels::SpanTileGrid grid = span_tile_grid();
  // La sortie d'avant dans la sortie même : rien à copier, mais pas de décalage possible sur place.
  const bool in_place = job->previous == 0 || job->previous == output;
  if (in_place && (job->shift_x != 0 || job->shift_y != 0)) return false;
  const auto failed = [](cudaError_t error, const char* step) {
    return std::unexpected(cuda_refusal(error, std::format("SPAN tuiles : {}", step)));
  };
  if (tile_masks_ == nullptr && !allocate_tiles()) return false;
  if (!in_place) {
    // NOLINTBEGIN(performance-no-int-to-ptr) : un tenseur porte son adresse CUDA en entier
    const void* from = reinterpret_cast<const void*>(job->previous);
    void* to = reinterpret_cast<void*>(output);
    // NOLINTEND(performance-no-int-to-ptr)
    if (job->shift_x == 0 && job->shift_y == 0) {
      const std::size_t bytes =
          static_cast<std::size_t>(output_.dims[1] * output_.dims[2] * output_.dims[3]) * 2U;
      if (cudaError_t e = cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToDevice, stream);
          e != cudaSuccess)
        return failed(e, "copie de la sortie d'avant");
    } else if (std::string why; !net_->shift_output(stream, from, to, job->shift_x * grid.scale,
                                                    job->shift_y * grid.scale, why)) {
      return refuse(RefusalCode::VendorCallFailed, "SPAN tuiles : " + why);
    }
  }
  if (job->core != uploaded_core_) {
    // Les appels du même moteur sont ordonnés entre voies, y compris l'envoi et la lecture des masques.
    const std::size_t slot = static_cast<std::size_t>(tile_stage_);
    if (cudaError_t e = cudaEventSynchronize(tile_sent_[slot]); e != cudaSuccess)
      return failed(e, "attente d'un envoi de masques");
    std::uint32_t* host = tile_staging_[slot];
    net_->pack_tile_layers(job->core, host);
    constexpr std::size_t kMaskWords = kSpanBodyMaskWords + kernels::kSpanTileMaskWords;
    kernels::pack_span_tiles(job->core, std::span(host + kSpanBodyMaskWords, kernels::kSpanTileMaskWords));
    const std::size_t bytes = kMaskWords * sizeof(std::uint32_t);
    uploaded_core_.clear();
    if (cudaError_t e = cudaMemcpyAsync(tile_masks_, host, bytes, cudaMemcpyHostToDevice, stream);
        e != cudaSuccess)
      return failed(e, "envoi des masques");
    if (cudaError_t e = cudaEventRecord(tile_sent_[slot], stream); e != cudaSuccess) {
      // La copie a pu démarrer sans événement : attendre avant de réutiliser le tampon épinglé.
      static_cast<void>(cudaStreamSynchronize(stream));
      return failed(e, "cudaEventRecord");
    }
    uploaded_core_ = std::move(job->core);
    tile_stage_ = (tile_stage_ + 1) % kTileStages;
  }
  net_->set_tile_masks(tile_masks_, tile_masks_ + kSpanBodyMaskWords, true);
  return true;
}

bool SpanFp8Engine::allocate_tiles() {
  const std::size_t bytes = (kSpanBodyMaskWords + kernels::kSpanTileMaskWords) * sizeof(std::uint32_t);
  void* device = nullptr;
  bool ok = cudaMalloc(&device, bytes) == cudaSuccess;
  tile_masks_ = ok ? static_cast<std::uint32_t*>(device) : nullptr;
  for (std::size_t i = 0; ok && i < tile_staging_.size(); ++i) {
    void* host = nullptr;
    ok = cudaMallocHost(&host, bytes) == cudaSuccess;
    tile_staging_[i] = ok ? static_cast<std::uint32_t*>(host) : nullptr;
    if (ok) std::fill_n(tile_staging_[i], bytes / sizeof(std::uint32_t), 0U);
    ok = ok && cudaEventCreateWithFlags(&tile_sent_[i], cudaEventDisableTiming) == cudaSuccess;
  }
  if (ok) return true;
  static_cast<void>(cudaGetLastError());
  release_tiles();
  return false;
}

void SpanFp8Engine::end_tiles() {
  if (net_) net_->set_tile_masks(nullptr, nullptr);
}

void SpanFp8Engine::release_tiles() {
  end_tiles();
  for (cudaEvent_t& event : tile_sent_) {
    if (event != nullptr) {
      static_cast<void>(cudaEventSynchronize(event));
      cudaEventDestroy(event);
    }
    event = nullptr;
  }
  for (std::uint32_t*& host : tile_staging_) {
    if (host != nullptr) cudaFreeHost(host);
    host = nullptr;
  }
  if (tile_masks_ != nullptr) cudaFree(tile_masks_);
  tile_masks_ = nullptr;
  uploaded_core_.clear();
}

}  // namespace kirari::gpu::cuda
