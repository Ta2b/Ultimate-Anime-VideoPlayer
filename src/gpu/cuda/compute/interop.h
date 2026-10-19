// Import des mémoires et sémaphores que Vulkan exporte, tel que render exportera les slots du
// ring : cudaImportExternalMemory et cudaImportExternalSemaphore, les équivalents du runtime de
// cuImportExternalMemory et cuImportExternalSemaphore sur le contexte primaire.
//
// Côté Vulkan : VkExportMemoryAllocateInfo et VkExternalMemoryImageCreateInfo (ou ...Buffer...)
// en OPAQUE_FD (Linux) ou OPAQUE_WIN32 (Windows), allocation dédiée de préférence ; sémaphore
// timeline exporté de même. L'image change de famille de files par VK_QUEUE_FAMILY_EXTERNAL
// autour de chaque écriture CUDA.
//
// Propriété des poignées : un import réussi d'un descripteur de fichier le donne au pilote
// (l'appelant ne le ferme plus) ; un HANDLE Win32 reste à l'appelant. Un import refusé laisse
// la poignée à l'appelant.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "gpu/cuda/compute/image.h"

namespace ajp::gpu::cuda {

struct ExternalHandle {
  enum class Kind : std::uint8_t { OpaqueFd, OpaqueWin32 };
  Kind kind = Kind::OpaqueFd;
  int fd = -1;            // vkGetMemoryFdKHR, vkGetSemaphoreFdKHR
  void* win32 = nullptr;  // vkGetMemoryWin32HandleKHR, vkGetSemaphoreWin32HandleKHR
};

// Une allocation Vulkan importée. size : VkMemoryAllocateInfo::allocationSize ; dedicated :
// allouée avec VkMemoryDedicatedAllocateInfo (les deux doivent dire vrai).
class ExternalMemory {
 public:
  static Result<ExternalMemory> import(const ExternalHandle& handle, std::size_t size, bool dedicated);

  ExternalMemory() = default;
  ExternalMemory(const ExternalMemory&) = delete;
  ExternalMemory& operator=(const ExternalMemory&) = delete;
  ExternalMemory(ExternalMemory&& other) noexcept;
  ExternalMemory& operator=(ExternalMemory&& other) noexcept;
  ~ExternalMemory();

  cudaExternalMemory_t get() const { return memory_; }
  std::size_t size() const { return size_; }

  // Un tampon Vulkan (VkBuffer lié à offset) vu en mémoire linéaire de l'appareil ; le pointeur
  // vit autant que cet objet. Un seul tampon par import. Refus : plage hors de l'allocation.
  Result<void*> map_buffer(std::size_t offset, std::size_t bytes);

 private:
  void reset();
  cudaExternalMemory_t memory_ = nullptr;
  std::size_t size_ = 0;
  void* mapped_ = nullptr;
};

// Une image Vulkan VK_FORMAT_R16G16B16A16_UNORM (tiling optimal, une couche, un niveau) importée
// en tableau CUDA, écrite par une surface : le slot RGBA16 du ring.
class ImportedImage {
 public:
  // offset : celui de vkBindImageMemory. Refus : dimensions hors de [1, kMaxDimension], appels
  // CUDA refusés.
  static Result<ImportedImage> import(const ExternalHandle& handle, std::size_t size, bool dedicated,
                                      std::size_t offset, int width, int height);

  ImportedImage(const ImportedImage&) = delete;
  ImportedImage& operator=(const ImportedImage&) = delete;
  ImportedImage(ImportedImage&& other) noexcept;
  ImportedImage& operator=(ImportedImage&& other) noexcept;
  ~ImportedImage();

  cudaSurfaceObject_t surface() const { return surface_; }
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  ImportedImage() = default;
  void reset();
  ExternalMemory memory_;
  cudaMipmappedArray_t mipmap_ = nullptr;
  cudaSurfaceObject_t surface_ = 0;
  int width_ = 0;
  int height_ = 0;
};

// Un sémaphore Vulkan importé : timeline (le protocole du ring, valeurs croissantes) ou binaire.
class ExternalSemaphore {
 public:
  static Result<ExternalSemaphore> import(const ExternalHandle& handle, bool timeline);

  ExternalSemaphore() = default;
  ExternalSemaphore(const ExternalSemaphore&) = delete;
  ExternalSemaphore& operator=(const ExternalSemaphore&) = delete;
  ExternalSemaphore(ExternalSemaphore&& other) noexcept;
  ExternalSemaphore& operator=(ExternalSemaphore&& other) noexcept;
  ~ExternalSemaphore();

  // Le flux attend que le sémaphore atteigne value (timeline) ou soit signalé (binaire, value
  // ignorée) ; puis signal le porte à value après le travail déjà soumis.
  Result<> wait(std::uint64_t value, cudaStream_t stream) const;
  Result<> signal(std::uint64_t value, cudaStream_t stream) const;

 private:
  cudaExternalSemaphore_t semaphore_ = nullptr;
  bool timeline_ = false;
};

// RGB → RGBA16 UNORM, alpha opaque (kernels::ref::to_rgba16), dans l'image importée de même
// taille.
Result<> write_rgba16(const Tensor& rgb, const ImportedImage& image, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
