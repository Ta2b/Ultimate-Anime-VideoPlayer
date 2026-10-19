// Backend TensorRT (gpu/backend.h, ABI 2) : moteurs construits par trtexec (prepare), exécutés
// sur un flux CUDA non bloquant du contexte primaire de la RTX du rendu, et les noyaux de la
// chaîne que la tranche verticale demande : ingestion des images NVDEC et écriture du ring
// importé de Vulkan.
//
// Chargé à l'exécution (ajp_backend_tensorrt.so) : le lecteur démarre sans CUDA ni TensorRT.
// Les noyaux et l'enveloppe TensorRT sont provisoires (slice_kernels.h, trt_engine.h) : ils
// laissent la place au socle CUDA (pc/cuda-compute) et à l'enveloppe TensorRT (pc/tensorrt).
#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/cuda/backend/gpu_timing.h"
#include "gpu/cuda/backend/ring_slot.h"
#include "gpu/kernel_ops.h"

namespace nvinfer1 {
class IRuntime;
}

namespace ajp::gpu::cuda {

class TrtLog;

class TensorRtBackend final : public Backend, public KernelOps {
 public:
  // trtexec : le constructeur de moteurs livré avec le backend.
  static std::expected<std::unique_ptr<TensorRtBackend>, Refusal> create(const BackendInit& init,
                                                                         std::filesystem::path trtexec);
  ~TensorRtBackend() override;
  TensorRtBackend(const TensorRtBackend&) = delete;
  TensorRtBackend& operator=(const TensorRtBackend&) = delete;

  // Backend
  BackendCaps capabilities() const override;
  DeviceMatch match(const DeviceUuid& render_device) const override;
  std::expected<Artifact, Refusal> build(const ModelSpec& model, const ShapeSet& shapes,
                                         Progress& progress) override;
  std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact& artifact) override;
  std::expected<Tensor, Refusal> import(const ExternalImage& image) override;
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                     const Shape& shape) override;
  KernelOps* kernels() override { return this; }
  std::expected<void, Refusal> enqueue(Engine& engine, std::span<const Tensor> inputs,
                                       std::span<Tensor> outputs) override;
  std::expected<Fence, Refusal> signal() override;
  std::expected<void, Refusal> wait(const Fence& fence) override;

  // KernelOps : ingest et write_ring ; les autres noyaux de la chaîne sont refusés ici.
  std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                      const Tensor& rgb) override;
  std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter, const Tensor& out) override;
  std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) override;
  std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) override;
  std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion, float t,
                                          const Tensor& out) override;
  std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                          const RingWrite& sync) override;
  std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) override;
  std::optional<PairScores> previous_scores() override { return std::nullopt; }
  // Extensions de la tranche verticale (tensorrt_rife.cpp) : entrée de RIFE et mesures immédiates.
  std::expected<void, Refusal> rife_input(const Tensor& a, const Tensor& b, float t,
                                          const Tensor& input) override;
  std::expected<PairScores, Refusal> scores_now(const Tensor& a, const Tensor& b) override;
  // RTX HDR (tensorrt_true_hdr.cpp).
  std::expected<void, Refusal> open_true_hdr(const vendor_hdr::InstalledSdk& sdk,
                                             const vendor_hdr::TrueHdrConfig& config) override;
  std::expected<void, Refusal> write_ring_true_hdr(const Tensor& rgb,
                                                   const vendor_hdr::InputPreparation& prepared,
                                                   const RingSlotDesc& slot, const RingWrite& sync) override;
  std::expected<void, Refusal> set_true_hdr(const vendor_hdr::TrueHdrParams& params) override;
  std::optional<DeviceMemory> device_memory() override;
  std::expected<float, Refusal> flow_max_now(const Motion& motion) override;

 private:
  // Une image décodée tenue jusqu'à la fin de son ingestion côté GPU : FFmpeg la réécrit dès
  // qu'elle revient à son pool.
  struct Held {
    std::shared_ptr<const FrameData> data;
    cudaEvent_t done = nullptr;
  };
  static constexpr std::size_t kHeldFrames = 8;
  static constexpr std::size_t kFences = 16;

  TensorRtBackend() = default;
  std::expected<void, Refusal> bind() const;  // le device courant est propre à chaque thread
  void log(BackendLogLevel level, std::string_view message) const;
  std::expected<void, Refusal> hold(const DecodedFrame& frame);
  std::expected<ImportedSlot*, Refusal> slot(const RingSlotDesc& desc);
  // Écriture d'un slot : protocole vérifié, slot importé, attente de wait_value enfilée.
  std::expected<ImportedSlot*, Refusal> begin_write(const Tensor& rgb, const RingSlotDesc& desc,
                                                    const RingWrite& sync);
  void release_retired();

  int device_ = 0;
  cudaDeviceProp properties_{};
  DeviceUuid uuid_{};
  int runtime_version_ = 0;
  int driver_version_ = 0;
  std::filesystem::path cache_dir_;
  std::filesystem::path trtexec_;
  BackendLog log_;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<TrtLog> trt_log_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::array<Held, kHeldFrames> held_{};
  std::array<cudaEvent_t, kFences> fences_{};
  std::uint64_t fence_count_ = 0;
  std::unique_ptr<GpuTiming> span_timing_;  // le moteur seul
  // Profil GPU par étape (ingestion, mesures, RIFE, chaque moteur, RTX HDR, écriture), résumé au
  // journal à la destruction du backend.
  std::map<std::string, std::unique_ptr<GpuTiming>, std::less<>> stages_;
  GpuTiming& stage(std::string_view name);
  std::unique_ptr<GpuTiming> chain_timing_;           // de l'ingestion à la fin du moteur
  std::vector<std::unique_ptr<ImportedSlot>> slots_;  // génération courante, par indice
  std::uint32_t generation_ = 0;
  // Imports d'une génération remplacée, relâchés quand retired_done_ est passé côté GPU.
  std::vector<std::unique_ptr<ImportedSlot>> retired_;
  cudaEvent_t retired_done_ = nullptr;
  // Mesures d'une paire : sommes côté GPU, relues dans une mémoire hôte épinglée.
  struct PairSums {
    double total = 0.0;
    unsigned int worst = 0;
  };
  PairSums* pair_device_ = nullptr;
  PairSums* pair_host_ = nullptr;
  unsigned int* flow_device_ = nullptr;  // flow_max_now : bits du maximum, sur l'appareil et l'hôte
  unsigned int* flow_host_ = nullptr;
  // RTX HDR : l'effet, son pont de flux, et ses deux tampons linéaires (RGBA8 tramé, RGB10A2).
  struct TrueHdrState;
  std::unique_ptr<TrueHdrState> true_hdr_;
};

}  // namespace ajp::gpu::cuda
