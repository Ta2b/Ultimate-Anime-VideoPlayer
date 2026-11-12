// Noyaux CUDA de l'analyse d'une image pour le grade (kernels/spec/frame_stats.h) : le portage de
// frame_stats.comp, un point de la grille par fil sur plusieurs blocs (une passe pour le cadre, une pour
// les comptes), qui écrit kernels::FrameCounts au début de counts (remis à zéro au lancement). Référence :
// kernels/ref/frame_stats.cpp.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct FrameStatsParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width
  bool fp16 = true;           // sinon FP32
  int width = 0;
  int height = 0;
  int transfer = 0;        // kernels::FrameTransfer : 0 SDR, 1 PQ, 2 HLG
  void* counts = nullptr;  // kernels::FrameCounts, puis la mémoire de travail : frame_stats_device_bytes()
};

// La taille du tampon d'appareil de counts : les comptes et la mémoire de travail des deux passes.
std::size_t frame_stats_device_bytes();

cudaError_t launch_frame_stats(const FrameStatsParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
