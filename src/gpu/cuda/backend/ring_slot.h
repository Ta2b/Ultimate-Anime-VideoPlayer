// Un slot du ring (VkImage RGBA16 et son sémaphore timeline, exportés par render) importé
// dans CUDA : une surface d'écriture et le sémaphore, à la manière de l'interop CUDA de mpv
// (video/out/hwdec/hwdec_cuda_vk.c).
//
// Les descripteurs du slot appartiennent au ring : l'import en prend une copie (dup), car
// CUDA garde le descripteur qu'on lui donne.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <memory>

#include "core/ring.h"
#include "core/types.h"

namespace ajp::gpu::cuda {

class ImportedSlot {
 public:
  // Refus : format autre que RGBA16 UNORM ou handle autre qu'un descripteur (FormatUnsupported),
  // taille nulle (ShapeInvalid), import refusé par CUDA (VendorCallFailed).
  static std::expected<std::unique_ptr<ImportedSlot>, Refusal> import(const RingSlotDesc& slot);
  ~ImportedSlot();
  ImportedSlot(const ImportedSlot&) = delete;
  ImportedSlot& operator=(const ImportedSlot&) = delete;

  const RingSlotDesc& desc() const { return desc_; }
  cudaSurfaceObject_t surface() const { return surface_; }
  // Attend wait_value puis, après les noyaux enfilés entre-temps, signale signal_value.
  std::expected<void, Refusal> wait(std::uint64_t value, cudaStream_t stream) const;
  std::expected<void, Refusal> signal(std::uint64_t value, cudaStream_t stream) const;

 private:
  explicit ImportedSlot(const RingSlotDesc& slot) : desc_(slot) {}

  RingSlotDesc desc_;
  cudaExternalMemory_t memory_ = nullptr;
  cudaMipmappedArray_t mipmaps_ = nullptr;
  cudaSurfaceObject_t surface_ = 0;
  cudaExternalSemaphore_t semaphore_ = nullptr;
};

}  // namespace ajp::gpu::cuda
