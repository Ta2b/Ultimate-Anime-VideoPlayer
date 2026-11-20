#pragma once

#include <array>
#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

struct GmfssImageNormalization {
  float mean[3]{};
  float deviation[3]{1.F, 1.F, 1.F};
};

// height et width sont les dimensions de fusion. Frames porte les six plans FP16 à 2h × 2w.
// half_images reçoit deux images à h × w, gmflow_images leur version à h × w ou h/2 × w/2.
cudaError_t launch_gmfss_pair_flow_input(cudaStream_t stream, const void* frames, int height, int width,
                                         bool half_flow, void* gmflow_images,
                                         const std::array<void*, 2>& half_images,
                                         const GmfssImageNormalization* normalization = nullptr);

// Flot GMFlow [2 directions][2 axes][h][w] en FP32 aux dimensions de GMFlow.
// Pour chaque direction, flows reçoit les deux premiers plans FP16 à h × w, h/2 × w/2 et h/4 × w/4.
// metric_input reçoit les 14 plans FP16 à h × w attendus par le MetricNet.
cudaError_t launch_gmfss_pair_flow_output(cudaStream_t stream, const float* flow, int height, int width,
                                          bool half_flow, const std::array<const void*, 2>& half_images,
                                          const std::array<void*, 6>& flows, void* metric_input);

}  // namespace kirari::gpu::cuda
