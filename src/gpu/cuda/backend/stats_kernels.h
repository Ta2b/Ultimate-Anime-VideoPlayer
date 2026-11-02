// Noyau CUDA de l'analyse d'une image pour le grade (kernels/spec/frame_stats.h) : le portage de
// frame_stats.comp, un seul bloc de 256 fils sur la grille de points d'un tenseur RGB planaire, qui
// écrit kernels::FrameCounts dans counts (remis à zéro par le noyau). Référence :
// kernels/ref/frame_stats.cpp.
#pragma once

#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct FrameStatsParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width
  bool fp16 = true;           // sinon FP32
  int width = 0;
  int height = 0;
  int transfer = 0;        // kernels::FrameTransfer : 0 SDR, 1 PQ, 2 HLG
  void* counts = nullptr;  // kernels::FrameCounts
};

cudaError_t launch_frame_stats(const FrameStatsParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
