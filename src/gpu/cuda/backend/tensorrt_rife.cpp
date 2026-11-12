// RIFE dans le backend TensorRT de la tranche verticale : entrée remplie (rife_input, puis son
// instant seul par rife_time, les paires suivantes sans le maillage par rife_pair), recadrage de
// la sortie (rife_crop, du contrat) et mesures d'une paire (scores_start, scores_take).
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <format>
#include <span>
#include <string>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/ref/half.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::cuda {
namespace {

bool fp16_planes(const Tensor& tensor, std::int64_t channels) {
  return tensor.type == TensorType::F16 && tensor.handle != 0 && tensor.shape.dims[0] == 1 &&
         tensor.shape.dims[1] == channels && tensor.shape.dims[2] > 0 && tensor.shape.dims[3] > 0;
}

int width_of(const Tensor& tensor) {
  return static_cast<int>(tensor.shape.dims[3]);
}
int height_of(const Tensor& tensor) {
  return static_cast<int>(tensor.shape.dims[2]);
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::rife_input(const Tensor& a, const Tensor& b, float t,
                                                         const Tensor& input) {
  return fill_rife(a, b, t, input, true, "rife_input");
}

std::expected<void, Refusal> TensorRtBackend::rife_pair(const Tensor& a, const Tensor& b, float t,
                                                        const Tensor& input) {
  return fill_rife(a, b, t, input, false, "rife_pair");
}

std::expected<void, Refusal> TensorRtBackend::fill_rife(const Tensor& a, const Tensor& b, float t,
                                                        const Tensor& input, bool mesh, const char* op) {
  if (!fp16_planes(a, 3) || !fp16_planes(b, 3) || a.shape.dims != b.shape.dims) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("{} : deux images RGB FP16 de même taille attendues", op));
  }
  const std::int64_t planes = input.shape.dims[1];
  if (!kernels::is_rife_input_planes(planes) || !fp16_planes(input, planes) ||
      !kernels::rife_fits(width_of(a), height_of(a), width_of(input), height_of(input))) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("{} : tenseur FP16 1×(11 ou 7)×H×W attendu, H et W multiples de {} "
                              "et au moins {}×{}",
                              op, kernels::kRifeAlign, height_of(a), width_of(a)));
  }
  // La fenêtre se lit sur le tenseur : le multiple de remplissage dépend du modèle (spec/rife.h).
  const kernels::RifeGeometry g =
      kernels::rife_window(width_of(a), height_of(a), width_of(input), height_of(input));
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("{} : instant {} hors de ]0, 1[", op, t));
  }
  if (auto ok = bind(); !ok) return ok;
  const RifeInputParams params{.a = address(a),
                               .b = address(b),
                               .width = g.width,
                               .height = g.height,
                               .padded_width = g.padded_width,
                               .padded_height = g.padded_height,
                               .left = g.left,
                               .top = g.top,
                               .black = 0.0F,  // la chaîne ingère en plage limitée : le noir vaut 0
                               .time = t,
                               .input = address(input),
                               .planes = static_cast<int>(planes),
                               .mesh = mesh};
  if (auto ok = order({a.handle, b.handle}, {input.handle}); !ok) return ok;
  // Le plan de l'instant est réécrit en entier : plus rien à reposer pour cette entrée.
  const std::uintptr_t key = input.handle + input.offset;
  std::erase_if(stale_times_, [key](const StaleTime& stale) { return stale.key == key; });
  // La fenêtre de l'image dans cette entrée, que le warp final de RIFE fusionné ne quitte pas (C03-03).
  std::erase_if(rife_windows_, [key](const RifeWindow& known) { return known.key == key; });
  if (rife_windows_.size() >= kStaleTimes) rife_windows_.erase(rife_windows_.begin());
  rife_windows_.push_back(
      {.key = key, .window = {.x = g.left, .y = g.top, .width = g.width, .height = g.height}});
  GpuTiming& timing = stage("entrée de RIFE");
  timing.begin(stream_);
  auto launched =
      cuda_check(launch_rife_input(params, stream_), mesh ? "rife_input<<<>>>" : "rife_pair<<<>>>");
  timing.end(stream_);
  return launched;
}

void TensorRtBackend::note_rife_reader(const Tensor& input, bool compact) {
  const std::uintptr_t key = input.handle + input.offset;
  const auto found = std::ranges::find(compact_rife_inputs_, key);
  if (compact && found == compact_rife_inputs_.end()) {
    if (compact_rife_inputs_.size() >= 8) compact_rife_inputs_.erase(compact_rife_inputs_.begin());
    compact_rife_inputs_.push_back(key);
  } else if (!compact && found != compact_rife_inputs_.end()) {
    compact_rife_inputs_.erase(found);
  }
}

void TensorRtBackend::forget_rife_input(std::uintptr_t key) {
  std::erase(compact_rife_inputs_, key);
  std::erase_if(rife_windows_, [key](const RifeWindow& known) { return known.key == key; });
  std::erase_if(stale_times_, [key](const StaleTime& stale) { return stale.key == key; });
}

std::expected<void, Refusal> TensorRtBackend::restore_rife_times(std::span<const Tensor> inputs) {
  for (const Tensor& input : inputs) {
    const std::uintptr_t key = input.handle + input.offset;
    const auto found = std::ranges::find(stale_times_, key, &StaleTime::key);
    if (found == stale_times_.end()) continue;
    const StaleTime stale = *found;
    stale_times_.erase(found);
    const CUresult set =
        cuMemsetD16Async(static_cast<CUdeviceptr>(stale.plane), stale.half, stale.count, stream_);
    if (set != CUDA_SUCCESS)
      return refuse(RefusalCode::VendorCallFailed,
                    std::format("cuMemsetD16Async : erreur {}", static_cast<int>(set)));
  }
  return {};
}

std::expected<void, Refusal> TensorRtBackend::rife_time(const Tensor& input, float t) {
  const std::int64_t planes = input.shape.dims[1];
  if (!kernels::is_rife_input_planes(planes) || !fp16_planes(input, planes)) {
    return refuse(RefusalCode::ShapeInvalid, "rife_time : tenseur FP16 1×(11 ou 7)×H×W attendu");
  }
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("rife_time : instant {} hors de ]0, 1[", t));
  }
  if (auto ok = bind(); !ok) return ok;
  if (auto ok = order({input.handle}, {input.handle}); !ok) return ok;
  // Seul le plan de l'instant change : les images et le maillage restent ceux de rife_input.
  // Une mise à valeur 16 bits du plan, sans noyau, au même arrondi que __float2half_rn.
  const std::size_t plane =
      static_cast<std::size_t>(width_of(input)) * static_cast<std::size_t>(height_of(input));
  const CUdeviceptr time_plane =
      input.handle + input.offset + std::size_t{kernels::kRifeChannelTime} * plane * sizeof(std::uint16_t);
  // RIFE fusionné ne lit que l'élément 0 du plan (étape 5.2) : un demi-flottant au lieu d'un plan entier.
  const bool compact =
      std::ranges::find(compact_rife_inputs_, input.handle + input.offset) != compact_rife_inputs_.end();
  const std::uint16_t half = kernels::ref::float_to_half(t);
  GpuTiming& timing = stage("entrée de RIFE");
  timing.begin(stream_);
  const CUresult set = cuMemsetD16Async(time_plane, half, compact ? 1 : plane, stream_);
  timing.end(stream_);
  if (set != CUDA_SUCCESS)
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("cuMemsetD16Async : erreur {}", static_cast<int>(set)));
  // Compact, le reste du plan garde l'instant d'avant : un moteur entier qui lirait cette entrée ensuite le
  // fera reposer (restore_rife_times, B03-04).
  const std::uintptr_t key = input.handle + input.offset;
  std::erase_if(stale_times_, [key](const StaleTime& stale) { return stale.key == key; });
  if (compact) {
    if (stale_times_.size() >= kStaleTimes) {
      // Le plus ancien n'est pas oublié avec son plan à moitié écrit : reposé tout de suite, sur cette voie,
      // pour un moteur entier qui lirait encore cette entrée.
      const StaleTime oldest = stale_times_.front();
      stale_times_.erase(stale_times_.begin());
      const std::array writes{oldest.handle};
      if (auto ok = order(std::span<const std::uintptr_t>{}, writes); !ok) return ok;
      if (const CUresult reset =
              cuMemsetD16Async(static_cast<CUdeviceptr>(oldest.plane), oldest.half, oldest.count, stream_);
          reset != CUDA_SUCCESS)
        return refuse(RefusalCode::VendorCallFailed,
                      std::format("cuMemsetD16Async : erreur {}", static_cast<int>(reset)));
    }
    stale_times_.push_back(
        {.key = key, .handle = input.handle, .plane = time_plane, .count = plane, .half = half});
  }
  return {};
}

std::expected<void, Refusal> TensorRtBackend::rife_crop(const Tensor& padded, const Tensor& out) {
  if (!fp16_planes(padded, 3) || !fp16_planes(out, 3)) {
    return refuse(RefusalCode::ShapeInvalid, "rife_crop : tenseurs RGB FP16 attendus");
  }
  if (!kernels::rife_fits(width_of(out), height_of(out), width_of(padded), height_of(padded))) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("rife_crop : sortie de RIFE {}×{} trop petite ou hors des multiples de {} pour {}×{}",
                    height_of(padded), width_of(padded), kernels::kRifeAlign, height_of(out), width_of(out)));
  }
  const kernels::RifeGeometry g =
      kernels::rife_window(width_of(out), height_of(out), width_of(padded), height_of(padded));
  if (auto ok = bind(); !ok) return ok;
  const RifeCropParams params{.padded = address(padded),
                              .padded_width = g.padded_width,
                              .padded_height = g.padded_height,
                              .left = g.left,
                              .top = g.top,
                              .width = g.width,
                              .height = g.height,
                              .out = address(out)};
  if (auto ok = order({padded.handle}, {out.handle}); !ok) return ok;
  GpuTiming& timing = stage("recadrage de RIFE");
  timing.begin(stream_);
  auto launched = cuda_check(launch_rife_crop(params, stream_), "rife_crop<<<>>>");
  timing.end(stream_);
  return launched;
}

std::expected<PairScores, Refusal> TensorRtBackend::scores_now(const Tensor& a, const Tensor& b) {
  return scores_start(a, b).and_then([&](std::uint64_t ticket) { return scores_take(ticket); });
}

std::expected<std::uint64_t, Refusal> TensorRtBackend::scores_start(const Tensor& a, const Tensor& b) {
  if (!fp16_planes(a, 3) || !fp16_planes(b, 3) || a.shape.dims != b.shape.dims) {
    return refuse(RefusalCode::ShapeInvalid, "scores_start : deux images RGB FP16 de même taille attendues");
  }
  const std::uint64_t ticket = pair_tickets_ + 1;
  PairRead& slot = pair_reads_[ticket % pair_reads_.size()];
  std::expected<void, Refusal> ready = bind();
  if (ready) ready = reserve_readback(slot, sizeof(PairSums));
  // Le flux ordonne les emplacements : un ticket périmé n'est plus relu, rien n'est à attendre.
  if (ready) ready = order({a.handle, b.handle}, {reinterpret_cast<std::uintptr_t>(slot.device)});
  if (ready)
    ready = cuda_check(cudaMemsetAsync(slot.device, 0, sizeof(PairSums), stream_), "cudaMemsetAsync");
  if (ready) {
    const PairDiffParams params{.a = address(a),
                                .b = address(b),
                                .width = width_of(a),
                                .height = height_of(a),
                                .levels = pair_levels_,  // la plage de la dernière source ingérée
                                .kr = pair_k_.kr,        // et sa matrice
                                .kb = pair_k_.kb,
                                .total = &slot.device->total,
                                .worst = &slot.device->worst};
    GpuTiming& timing = stage("mesures de la paire");
    timing.begin(stream_);
    ready = cuda_check(launch_pair_diff(params, stream_), "pair_diff<<<>>>");
    timing.end(stream_);
  }
  if (ready) {
    ready =
        cuda_check(cudaMemcpyAsync(slot.host, slot.device, sizeof(PairSums), cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync");
  }
  if (ready) ready = cuda_check(cudaEventRecord(slot.done, stream_), "cudaEventRecord");
  if (!ready) return std::unexpected(std::move(ready.error()));
  const kernels::RifeGeometry g = kernels::rife_geometry(width_of(a), height_of(a));
  pair_tickets_ = ticket;
  slot.ticket = ticket;
  slot.area = static_cast<double>(g.padded_width) * static_cast<double>(g.padded_height);
  slot.levels = static_cast<double>(pair_levels_);
  return ticket;
}

std::expected<PairScores, Refusal> TensorRtBackend::scores_take(std::uint64_t ticket) {
  PairRead& slot = pair_reads_[ticket % pair_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket)
    return refuse(RefusalCode::SequenceOrder, std::format("scores_take : ticket {} périmé", ticket));
  // Attente hôte voulue : la décision de lancer RIFE dépend de cette paire (kernels/spec/detect.h).
  // Elle ne couvre que la mesure, pas ce que le flux a reçu depuis.
  if (auto ok = wait_event(slot.done); !ok) return std::unexpected(std::move(ok.error()));
  slot.ticket = 0;
  // Σ |ΔY| × (pas 8 bits / 255) : l'écart en codes rapporté à la crête (kernels/spec/detect.h), sur l'aire
  // remplie de RIFE. Le pas est celui de la plage de la source, comme pour le pire bloc : 219 en limitée,
  // 255 en pleine, où ΔY = 1 couvre toute la crête (C03-02).
  const double scene = slot.host->total * (slot.levels / 255.0) / slot.area;
  return PairScores{
      .scene_score = scene, .worst_block = slot.host->worst, .identical = slot.host->total == 0.0};
}

}  // namespace ajp::gpu::cuda
