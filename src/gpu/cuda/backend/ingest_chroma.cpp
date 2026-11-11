// Ingestion du backend TensorRT : les tables du suréchantillonnage de la chroma (B03-02). Mêmes poids que
// l'ingestion Vulkan et que kernels/ref/color.cpp (kernels::ref::compute_weights, décalages de
// chroma_shift_up), copiés sur l'appareil une fois par taille, disposition, siting et filtre.
#include <format>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"
#include "gpu/cuda/backend/slice_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/ref/resample.h"

namespace ajp::gpu::cuda {
namespace {

// Une table copiée sur l'appareil : début de chaque sortie et poids, à la suite. Dans le pool, sur la file de
// la voie ; cudaMemcpyAsync depuis la mémoire pageable de l'hôte a recopié les tables à son retour, la file
// est tout de même attendue avant de rendre la main (une fois par changement de taille).
std::expected<void, Refusal> upload_table(const kernels::ref::Weights& table, int*& start, float*& weights,
                                          cudaStream_t stream) {
  const std::size_t starts = table.start.size() * sizeof(int);
  const std::size_t taps = table.w.size() * sizeof(float);
  if (auto ok =
          cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&start), starts, stream), "cudaMallocAsync");
      !ok)
    return ok;
  if (auto ok =
          cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&weights), taps, stream), "cudaMallocAsync");
      !ok)
    return ok;
  if (auto ok = cuda_check(cudaMemcpyAsync(start, table.start.data(), starts, cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync");
      !ok)
    return ok;
  if (auto ok = cuda_check(cudaMemcpyAsync(weights, table.w.data(), taps, cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync");
      !ok)
    return ok;
  return cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

}  // namespace

void TensorRtBackend::release_chroma_tables(bool behind_lanes) {
  // L'ingestion qui lit ces tables peut être en vol sur une autre voie : en lecture, rendues derrière toutes
  // les voies ; à la destruction, cudaFree attend la fin de tout ce que l'appareil exécute.
  for (void* p : {static_cast<void*>(chroma_.x_start), static_cast<void*>(chroma_.x_weights),
                  static_cast<void*>(chroma_.y_start), static_cast<void*>(chroma_.y_weights),
                  static_cast<void*>(chroma_.rows)}) {
    if (p == nullptr) continue;
    if (behind_lanes) {
      free_after_lanes(p);
    } else {
      cuda_warn(cudaFree(p), "cudaFree");
    }
  }
  chroma_ = ChromaTables{};
}

std::expected<void, Refusal> TensorRtBackend::chroma_tables(IngestParams& params, kernels::Siting siting,
                                                            kernels::Filter filter) {
  if (params.chroma_layout == kChroma444) return {};  // lue au point, sans table
  const bool same = chroma_.width == params.width && chroma_.height == params.height &&
                    chroma_.layout == params.chroma_layout && chroma_.siting == siting &&
                    chroma_.filter == filter;
  if (!same) {
    release_chroma_tables(true);
    const kernels::ChromaShift shift = kernels::chroma_shift_up(siting);
    const bool vertical = params.chroma_layout == kChroma420;
    const auto wx = kernels::ref::compute_weights(params.chroma_width, params.width, shift.x, filter);
    if (!wx)
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("ingestion : table de chroma {}", wx.error().detail));
    if (auto ok = upload_table(*wx, chroma_.x_start, chroma_.x_weights, stream_); !ok) {
      release_chroma_tables(true);
      return ok;
    }
    chroma_.x_taps = wx->taps;
    if (vertical) {
      const auto wy = kernels::ref::compute_weights(params.chroma_height, params.height, shift.y, filter);
      if (!wy) {
        release_chroma_tables(true);
        return refuse(RefusalCode::ShapeInvalid,
                      std::format("ingestion : table de chroma {}", wy.error().detail));
      }
      if (auto ok = upload_table(*wy, chroma_.y_start, chroma_.y_weights, stream_); !ok) {
        release_chroma_tables(true);
        return ok;
      }
      chroma_.y_taps = wy->taps;
      if (auto ok = cuda_check(
              cudaMallocAsync(reinterpret_cast<void**>(&chroma_.rows), chroma_rows_bytes(params), stream_),
              "cudaMallocAsync");
          !ok) {
        release_chroma_tables(true);
        return ok;
      }
    }
    chroma_.width = params.width;
    chroma_.height = params.height;
    chroma_.layout = params.chroma_layout;
    chroma_.siting = siting;
    chroma_.filter = filter;
  }
  params.x_start = chroma_.x_start;
  params.x_weights = chroma_.x_weights;
  params.x_taps = chroma_.x_taps;
  params.y_start = chroma_.y_start;
  params.y_weights = chroma_.y_weights;
  params.y_taps = chroma_.y_taps;
  params.chroma_rows = chroma_.rows;
  return {};
}

}  // namespace ajp::gpu::cuda
