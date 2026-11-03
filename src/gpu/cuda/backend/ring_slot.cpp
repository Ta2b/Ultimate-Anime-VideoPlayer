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
  const bool ten_bits = slot.format == RingFormat::Rgb10a2Unorm;
  if (slot.format != RingFormat::Rgba16Unorm && !ten_bits) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("slot {} : seuls RGBA16 et RGB10A2 UNORM sont importés", slot.index));
  }
  if (slot.memory.kind != kKind || slot.semaphore_handle.kind != kKind) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("slot {} : poignées exportées du système attendues", slot.index));
  }
  if (slot.width <= 0 || slot.height <= 0 || slot.memory_size == 0) {
    return refuse(RefusalCode::ShapeInvalid, std::format("slot {} : taille nulle", slot.index));
  }
  std::unique_ptr<ImportedSlot> imported(new ImportedSlot(slot));
  imported->packing_ = ten_bits ? RingPacking::Rgb10a2 : RingPacking::Rgba16;

  // Mémoire et image mappée, avec ou sans le drapeau « dédiée ». Les images du ring viennent de libplacebo,
  // souvent dédiées : CUDA veut alors le drapeau. Une image sous-allouée (décalage non nul, ou allocation
  // partagée) l'accepte à l'import et refuse le mappage (cudaErrorInvalidValue) : tout est refait sans.
  cudaExternalMemoryMipmappedArrayDesc array{};
  array.offset = slot.memory_offset;
  // RGB10A2 (étape 5.5a) : le format 1010102 de CUDA ; les noyaux n'y font que des écritures brutes de
  // 32 bits (store_ring), un tableau de 32 bits non signés sert de repli si le pilote refuse ce format.
  array.formatDesc =
      ten_bits ? cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102)
               : cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
  array.extent = cudaExtent{.width = static_cast<std::size_t>(slot.width),
                            .height = static_cast<std::size_t>(slot.height),
                            .depth = 0};
  array.flags = cudaArraySurfaceLoadStore;
  array.numLevels = 1;
  const auto map = [&](unsigned int flags) -> std::expected<void, Refusal> {
    cudaExternalMemoryHandleDesc memory{};
    memory.size = slot.memory_size;
    memory.flags = flags;
#ifdef _WIN32
    memory.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memory.handle.win32.handle = reinterpret_cast<void*>(slot.memory.value);
    if (const cudaError_t e = cudaImportExternalMemory(&imported->memory_, &memory); e != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      imported->memory_ = nullptr;
      return std::unexpected(cuda_refusal(e, "cudaImportExternalMemory"));
    }
#else
    memory.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memory.handle.fd = duplicate(slot.memory);
    if (memory.handle.fd < 0)
      return refuse(RefusalCode::IoFailed, std::format("slot {} : dup de la mémoire", slot.index));
    if (const cudaError_t e = cudaImportExternalMemory(&imported->memory_, &memory); e != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      close(memory.handle.fd);  // pris par CUDA seulement en cas de succès
      imported->memory_ = nullptr;
      return std::unexpected(cuda_refusal(e, "cudaImportExternalMemory"));
    }
#endif
    if (const cudaError_t e =
            cudaExternalMemoryGetMappedMipmappedArray(&imported->mipmaps_, imported->memory_, &array);
        e != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      imported->mipmaps_ = nullptr;
      cudaDestroyExternalMemory(imported->memory_);  // ferme aussi le descripteur pris à l'import
      imported->memory_ = nullptr;
      return std::unexpected(cuda_refusal(e, "cudaExternalMemoryGetMappedMipmappedArray"));
    }
    return {};
  };
  const auto map_any = [&]() -> std::expected<void, Refusal> {
    if (slot.memory_offset == 0 && map(cudaExternalMemoryDedicated)) return {};
    return map(0);
  };
  if (auto ok = map_any(); !ok) {
    if (!ten_bits) return std::unexpected(ok.error());
    array.formatDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    if (auto again = map_any(); !again) return std::unexpected(again.error());
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
  if (auto ok = cuda_check(cudaSignalExternalSemaphoresAsync(&semaphore_, &params, 1, stream),
                           "cudaSignalExternalSemaphoresAsync");
      !ok)
    return ok;
  // Sous Windows (WDDM), CUDA garde les lancements dans un lot logiciel jusqu'au prochain appel qui force
  // l'envoi : le signal partirait avec l'image suivante, une image trop tard pour l'écran qui la jette.
  // L'interrogation du flux pousse le lot vers le GPU sans attendre ; sans effet sous Linux. Sa réponse
  // (prêt ou pas encore) ne dit rien d'utile ici, seule une vraie erreur compte.
  const cudaError_t pushed = cudaStreamQuery(stream);
  if (pushed == cudaErrorNotReady) return {};
  return cuda_check(pushed, "cudaStreamQuery");
}

}  // namespace ajp::gpu::cuda
