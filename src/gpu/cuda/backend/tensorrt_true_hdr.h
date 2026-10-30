// État de RTX HDR (TrueHDR) dans le backend TensorRT de la tranche verticale : l'effet du SDK, le
// pont entre le flux non bloquant du backend et le flux 0 où tourne le SDK (true_hdr.h), et les deux
// tampons linéaires de l'effet, alloués une fois à la taille de la sortie de SPAN.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>

#include "gpu/cuda/backend/tensorrt_backend.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::gpu::cuda {

// GpuStream de true_hdr.h autour du flux du backend et de deux événements créés une fois.
class LegacyBridge final : public vendor_hdr::GpuStream {
 public:
  explicit LegacyBridge(cudaStream_t stream) : stream_(stream) {}
  ~LegacyBridge() override;
  LegacyBridge(const LegacyBridge&) = delete;
  LegacyBridge& operator=(const LegacyBridge&) = delete;
  std::expected<void, Refusal> create();
  vendor_hdr::Done hand_to_legacy() override;
  vendor_hdr::Done take_from_legacy() override;

 private:
  cudaStream_t stream_;
  cudaEvent_t input_ready_ = nullptr;
  cudaEvent_t output_ready_ = nullptr;
};

struct TensorRtBackend::TrueHdrState {
  std::unique_ptr<vendor_hdr::TrueHdr> effect;
  std::unique_ptr<LegacyBridge> bridge;
  int width = 0;
  int height = 0;
  void* rgba8 = nullptr;  // entrée de Maxine
  std::size_t rgba8_pitch = 0;
  void* rgb10 = nullptr;  // sortie RGB10A2
  std::size_t rgb10_pitch = 0;
  std::uint32_t frames = 0;               // graine du tramage
  Lane lane = Lane::Upscale;              // la voie de l'ouverture : le pont tient sa file
  vendor_hdr::InputPreparation prepared;  // celle de la dernière image, reprise par la rampe
  // Luminance moyenne de l'image source (mean_luma) : la somme sur le GPU et la dernière moyenne en mémoire
  // hôte projetée, lue par le thread principal (une image de retard, aucune copie ni attente).
  float* luma_accum = nullptr;
  float* luma_host = nullptr;
  float* luma_device = nullptr;  // l'adresse GPU de luma_host

  TrueHdrState() = default;
  TrueHdrState(const TrueHdrState&) = delete;
  TrueHdrState& operator=(const TrueHdrState&) = delete;
  ~TrueHdrState();
};

}  // namespace ajp::gpu::cuda
