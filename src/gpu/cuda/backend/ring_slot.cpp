#include "gpu/cuda/backend/ring_slot.h"

#include <format>

#include "gpu/cuda/backend/cuda_status.h"

#ifndef _WIN32
#include <unistd.h>
#endif

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

#ifdef _WIN32
// Sous Windows, CUDA ne prend pas le HANDLE NT : celui du ring sert tel quel, fermé par le ring.
constexpr HandleKind kKind = HandleKind::OpaqueWin32;
#else
constexpr HandleKind kKind = HandleKind::OpaqueFd;
// Copie d'un descripteur exporté, que CUDA prendra en charge ; -1 si dup échoue.
int duplicate(const ExportedHandle& handle) {
  return dup(static_cast<int>(handle.value));
}
#endif

}  // namespace

std::expected<std::unique_ptr<ImportedSlot>, Refusal> ImportedSlot::import(const RingSlotDesc& slot) {
  if (slot.format != RingFormat::Rgba16Unorm) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("slot {} : seul RGBA16 UNORM est importé", slot.index));
  }
  if (slot.memory.kind != kKind || slot.semaphore_handle.kind != kKind) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("slot {} : poignées exportées du système attendues", slot.index));
  }
  if (slot.width <= 0 || slot.height <= 0 || slot.memory_size == 0) {
    return refuse(RefusalCode::ShapeInvalid, std::format("slot {} : taille nulle", slot.index));
  }
  std::unique_ptr<ImportedSlot> imported(new ImportedSlot(slot));

  cudaExternalMemoryHandleDesc memory{};
  memory.size = slot.memory_size;
#ifdef _WIN32
  memory.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  memory.handle.win32.handle = reinterpret_cast<void*>(slot.memory.value);
  if (const cudaError_t error = cudaImportExternalMemory(&imported->memory_, &memory); error != cudaSuccess)
    return std::unexpected(cuda_refusal(error, "cudaImportExternalMemory"));
#else
  memory.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memory.handle.fd = duplicate(slot.memory);
  if (memory.handle.fd < 0)
    return refuse(RefusalCode::IoFailed, std::format("slot {} : dup de la mémoire", slot.index));
  if (const cudaError_t error = cudaImportExternalMemory(&imported->memory_, &memory); error != cudaSuccess) {
    close(memory.handle.fd);  // pris par CUDA seulement en cas de succès
    return std::unexpected(cuda_refusal(error, "cudaImportExternalMemory"));
  }
#endif

  cudaExternalMemoryMipmappedArrayDesc array{};
  array.offset = slot.memory_offset;
  array.formatDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
  array.extent = cudaExtent{.width = static_cast<std::size_t>(slot.width),
                            .height = static_cast<std::size_t>(slot.height),
                            .depth = 0};
  array.flags = cudaArraySurfaceLoadStore;
  array.numLevels = 1;
  if (auto ok = cuda_check(
          cudaExternalMemoryGetMappedMipmappedArray(&imported->mipmaps_, imported->memory_, &array),
          "cudaExternalMemoryGetMappedMipmappedArray");
      !ok) {
    return std::unexpected(ok.error());
  }
  cudaArray_t level = nullptr;
  if (auto ok =
          cuda_check(cudaGetMipmappedArrayLevel(&level, imported->mipmaps_, 0), "cudaGetMipmappedArrayLevel");
      !ok) {
    return std::unexpected(ok.error());
  }
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = level;
  if (auto ok =
          cuda_check(cudaCreateSurfaceObject(&imported->surface_, &resource), "cudaCreateSurfaceObject");
      !ok) {
    return std::unexpected(ok.error());
  }

  cudaExternalSemaphoreHandleDesc semaphore{};
#ifdef _WIN32
  semaphore.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  semaphore.handle.win32.handle = reinterpret_cast<void*>(slot.semaphore_handle.value);
  if (const cudaError_t error = cudaImportExternalSemaphore(&imported->semaphore_, &semaphore);
      error != cudaSuccess)
    return std::unexpected(cuda_refusal(error, "cudaImportExternalSemaphore"));
#else
  semaphore.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  semaphore.handle.fd = duplicate(slot.semaphore_handle);
  if (semaphore.handle.fd < 0) {
    return refuse(RefusalCode::IoFailed, std::format("slot {} : dup du sémaphore", slot.index));
  }
  if (const cudaError_t error = cudaImportExternalSemaphore(&imported->semaphore_, &semaphore);
      error != cudaSuccess) {
    close(semaphore.handle.fd);
    return std::unexpected(cuda_refusal(error, "cudaImportExternalSemaphore"));
  }
#endif
  return imported;
}

ImportedSlot::~ImportedSlot() {
  if (semaphore_ != nullptr) cudaDestroyExternalSemaphore(semaphore_);
  if (surface_ != 0) cudaDestroySurfaceObject(surface_);
  if (mipmaps_ != nullptr) cudaFreeMipmappedArray(mipmaps_);
  if (memory_ != nullptr) cudaDestroyExternalMemory(memory_);
}

std::expected<void, Refusal> ImportedSlot::wait(std::uint64_t value, cudaStream_t stream) const {
  cudaExternalSemaphoreWaitParams params{};
  params.params.fence.value = value;
  return cuda_check(cudaWaitExternalSemaphoresAsync(&semaphore_, &params, 1, stream),
                    "cudaWaitExternalSemaphoresAsync");
}

std::expected<void, Refusal> ImportedSlot::signal(std::uint64_t value, cudaStream_t stream) const {
  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = value;
  return cuda_check(cudaSignalExternalSemaphoresAsync(&semaphore_, &params, 1, stream),
                    "cudaSignalExternalSemaphoresAsync");
}

}  // namespace ajp::gpu::cuda
