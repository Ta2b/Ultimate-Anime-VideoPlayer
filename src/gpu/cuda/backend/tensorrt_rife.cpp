// RIFE dans le backend TensorRT de la tranche verticale : entrée remplie (rife_input, puis son
// instant seul par rife_time), recadrage de la sortie (rife_crop, du contrat) et mesures d'une
// paire (scores_start, scores_take).
#include <cstring>
#include <format>
#include <string>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

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
  if (!fp16_planes(a, 3) || !fp16_planes(b, 3) || a.shape.dims != b.shape.dims) {
    return refuse(RefusalCode::ShapeInvalid, "rife_input : deux images RGB FP16 de même taille attendues");
  }
  const std::int64_t planes = input.shape.dims[1];
  if (!kernels::is_rife_input_planes(planes) || !fp16_planes(input, planes) ||
      !kernels::rife_fits(width_of(a), height_of(a), width_of(input), height_of(input))) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("rife_input : tenseur FP16 1×(11 ou 7)×H×W attendu, H et W multiples de {} "
                              "et au moins {}×{}",
                              kernels::kRifeAlign, height_of(a), width_of(a)));
  }
  // La fenêtre se lit sur le tenseur : le multiple de remplissage dépend du modèle (spec/rife.h).
  const kernels::RifeGeometry g =
      kernels::rife_window(width_of(a), height_of(a), width_of(input), height_of(input));
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("rife_input : instant {} hors de ]0, 1[", t));
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
                               .planes = static_cast<int>(planes)};
  GpuTiming& timing = stage("entrée de RIFE");
  timing.begin(stream_);
  auto launched = cuda_check(launch_rife_input(params, stream_), "rife_input<<<>>>");
  timing.end(stream_);
  return launched;
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
  const RifeInputParams params{.padded_width = width_of(input),
                               .padded_height = height_of(input),
                               .time = t,
                               .input = address(input),
                               .planes = static_cast<int>(planes),
                               .frames = false};
  GpuTiming& timing = stage("entrée de RIFE");
  timing.begin(stream_);
  auto launched = cuda_check(launch_rife_input(params, stream_), "rife_input<<<>>>");
  timing.end(stream_);
  return launched;
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
  if (ready && slot.device == nullptr) {
    ready = cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device), sizeof(PairSums)), "cudaMalloc");
    if (ready) {
      ready = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), sizeof(PairSums)),
                         "cudaMallocHost");
    }
    if (ready)
      ready = cuda_check(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming), "cudaEventCreate");
  }
  // Le flux ordonne les emplacements : un ticket périmé n'est plus relu, rien n'est à attendre.
  if (ready)
    ready = cuda_check(cudaMemsetAsync(slot.device, 0, sizeof(PairSums), stream_), "cudaMemsetAsync");
  if (ready) {
    const PairDiffParams params{.a = address(a),
                                .b = address(b),
                                .width = width_of(a),
                                .height = height_of(a),
                                .levels = 219.0F,  // plage limitée, comme l'ingestion de la chaîne
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
  return ticket;
}

std::expected<PairScores, Refusal> TensorRtBackend::scores_take(std::uint64_t ticket) {
  PairRead& slot = pair_reads_[ticket % pair_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket)
    return refuse(RefusalCode::SequenceOrder, std::format("scores_take : ticket {} périmé", ticket));
  // Attente hôte voulue : la décision de lancer RIFE dépend de cette paire (kernels/spec/detect.h).
  // Elle ne couvre que la mesure, pas ce que le flux a reçu depuis.
  if (auto ok = cuda_check(cudaEventSynchronize(slot.done), "cudaEventSynchronize"); !ok)
    return std::unexpected(std::move(ok.error()));
  slot.ticket = 0;
  // Σ |ΔY| × (pas 8 bits / 255) : l'écart en codes rapporté à la crête, sur l'aire remplie de RIFE.
  const double scene = slot.host->total * (219.0 / 255.0) / slot.area;
  return PairScores{
      .scene_score = scene, .worst_block = slot.host->worst, .identical = slot.host->total == 0.0};
}

}  // namespace ajp::gpu::cuda
