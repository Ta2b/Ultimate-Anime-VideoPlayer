#include "gpu/cuda/compute/interop.h"

#include <format>
#include <utility>

#include "gpu/cuda/compute/kernels/launch.h"
#include "kernels/spec/resample.h"

namespace ajp::gpu::cuda {
namespace {

cudaExternalMemoryHandleType memory_type(ExternalHandle::Kind kind) {
  return kind == ExternalHandle::Kind::OpaqueFd ? cudaExternalMemoryHandleTypeOpaqueFd
                                                : cudaExternalMemoryHandleTypeOpaqueWin32;
}

cudaExternalSemaphoreHandleType semaphore_type(ExternalHandle::Kind kind, bool timeline) {
  if (kind == ExternalHandle::Kind::OpaqueFd) {
    return timeline ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
                    : cudaExternalSemaphoreHandleTypeOpaqueFd;
  }
  return timeline ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
                  : cudaExternalSemaphoreHandleTypeOpaqueWin32;
}

Result<> check_handle(const ExternalHandle& handle) {
  const bool fd = handle.kind == ExternalHandle::Kind::OpaqueFd;
  if ((fd && handle.fd < 0) || (!fd && handle.win32 == nullptr)) {
    return refuse(RefusalCode::InputMismatch, "poignée externe vide");
  }
  return {};
}

}  // namespace

Result<ExternalMemory> ExternalMemory::import(const ExternalHandle& handle, std::size_t size,
                                              bool dedicated) {
  if (auto ok = check_handle(handle); !ok) return std::unexpected(ok.error());
  if (size == 0) return refuse(RefusalCode::InputMismatch, "mémoire externe de 0 octet");
  cudaExternalMemoryHandleDesc desc{.type = memory_type(handle.kind),
                                    .handle = {},
                                    .size = size,
                                    .flags = dedicated ? cudaExternalMemoryDedicated : 0U,
                                    .reserved = {}};
  if (handle.kind == ExternalHandle::Kind::OpaqueFd) {
    desc.handle.fd = handle.fd;
  } else {
    desc.handle.win32.handle = handle.win32;
  }
  ExternalMemory out;
  if (auto ok = cuda_check(cudaImportExternalMemory(&out.memory_, &desc), "cudaImportExternalMemory"); !ok) {
    return std::unexpected(ok.error());
  }
  out.size_ = size;
  return out;
}

ExternalMemory::ExternalMemory(ExternalMemory&& other) noexcept
    : memory_(std::exchange(other.memory_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      mapped_(std::exchange(other.mapped_, nullptr)) {}

ExternalMemory& ExternalMemory::operator=(ExternalMemory&& other) noexcept {
  if (this != &other) {
    reset();
    memory_ = std::exchange(other.memory_, nullptr);
    size_ = std::exchange(other.size_, 0);
    mapped_ = std::exchange(other.mapped_, nullptr);
  }
  return *this;
}

ExternalMemory::~ExternalMemory() {
  reset();
}

void ExternalMemory::reset() {
  // Le tampon projeté se libère avant la mémoire importée (documentation de
  // cudaExternalMemoryGetMappedBuffer).
  if (mapped_ != nullptr) cudaFree(mapped_);
  if (memory_ != nullptr) cudaDestroyExternalMemory(memory_);
  mapped_ = nullptr;
  memory_ = nullptr;
}

Result<void*> ExternalMemory::map_buffer(std::size_t offset, std::size_t bytes) {
  if (mapped_ != nullptr) return refuse(RefusalCode::InputMismatch, "mémoire externe déjà projetée");
  if (bytes == 0 || offset > size_ || bytes > size_ - offset) {
    return refuse(
        RefusalCode::InputMismatch,
        std::format("tampon externe de {} octets en {} hors de l'allocation de {}", bytes, offset, size_));
  }
  cudaExternalMemoryBufferDesc desc{};
  desc.offset = offset;
  desc.size = bytes;
  if (auto ok = cuda_check(cudaExternalMemoryGetMappedBuffer(&mapped_, memory_, &desc),
                           "cudaExternalMemoryGetMappedBuffer");
      !ok) {
    return std::unexpected(ok.error());
  }
  return mapped_;
}

Result<ImportedImage> ImportedImage::import(const ExternalHandle& handle, std::size_t size, bool dedicated,
                                            std::size_t offset, int width, int height) {
  if (!kernels::valid_plane_dimension(width) || !kernels::valid_plane_dimension(height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("image importée {}x{} hors de [1, {}]", width, height, kernels::kMaxDimension));
  }
  auto memory = ExternalMemory::import(handle, size, dedicated);
  if (!memory) return std::unexpected(memory.error());
  ImportedImage out;
  out.memory_ = std::move(*memory);
  out.width_ = width;
  out.height_ = height;

  cudaExternalMemoryMipmappedArrayDesc desc{};
  desc.offset = offset;
  desc.formatDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
  desc.extent = {
      .width = static_cast<std::size_t>(width), .height = static_cast<std::size_t>(height), .depth = 0};
  desc.flags = cudaArraySurfaceLoadStore;
  desc.numLevels = 1;
  if (auto ok = cuda_check(cudaExternalMemoryGetMappedMipmappedArray(&out.mipmap_, out.memory_.get(), &desc),
                           "cudaExternalMemoryGetMappedMipmappedArray");
      !ok) {
    return std::unexpected(ok.error());
  }
  cudaArray_t level = nullptr;
  if (auto ok = cuda_check(cudaGetMipmappedArrayLevel(&level, out.mipmap_, 0), "cudaGetMipmappedArrayLevel");
      !ok) {
    return std::unexpected(ok.error());
  }
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = level;
  if (auto ok = cuda_check(cudaCreateSurfaceObject(&out.surface_, &resource), "cudaCreateSurfaceObject");
      !ok) {
    return std::unexpected(ok.error());
  }
  return out;
}

ImportedImage::ImportedImage(ImportedImage&& other) noexcept
    : memory_(std::move(other.memory_)),
      mipmap_(std::exchange(other.mipmap_, nullptr)),
      surface_(std::exchange(other.surface_, 0)),
      width_(std::exchange(other.width_, 0)),
      height_(std::exchange(other.height_, 0)) {}

ImportedImage& ImportedImage::operator=(ImportedImage&& other) noexcept {
  if (this != &other) {
    reset();
    memory_ = std::move(other.memory_);
    mipmap_ = std::exchange(other.mipmap_, nullptr);
    surface_ = std::exchange(other.surface_, 0);
    width_ = std::exchange(other.width_, 0);
    height_ = std::exchange(other.height_, 0);
  }
  return *this;
}

ImportedImage::~ImportedImage() {
  reset();
}

void ImportedImage::reset() {
  if (surface_ != 0) cudaDestroySurfaceObject(surface_);
  if (mipmap_ != nullptr) cudaFreeMipmappedArray(mipmap_);
  surface_ = 0;
  mipmap_ = nullptr;
  memory_ = ExternalMemory();
}

Result<ExternalSemaphore> ExternalSemaphore::import(const ExternalHandle& handle, bool timeline) {
  if (auto ok = check_handle(handle); !ok) return std::unexpected(ok.error());
  cudaExternalSemaphoreHandleDesc desc{
      .type = semaphore_type(handle.kind, timeline), .handle = {}, .flags = 0, .reserved = {}};
  if (handle.kind == ExternalHandle::Kind::OpaqueFd) {
    desc.handle.fd = handle.fd;
  } else {
    desc.handle.win32.handle = handle.win32;
  }
  ExternalSemaphore out;
  if (auto ok =
          cuda_check(cudaImportExternalSemaphore(&out.semaphore_, &desc), "cudaImportExternalSemaphore");
      !ok) {
    return std::unexpected(ok.error());
  }
  out.timeline_ = timeline;
  return out;
}

ExternalSemaphore::ExternalSemaphore(ExternalSemaphore&& other) noexcept
    : semaphore_(std::exchange(other.semaphore_, nullptr)), timeline_(other.timeline_) {}

ExternalSemaphore& ExternalSemaphore::operator=(ExternalSemaphore&& other) noexcept {
  if (this != &other) {
    if (semaphore_ != nullptr) cudaDestroyExternalSemaphore(semaphore_);
    semaphore_ = std::exchange(other.semaphore_, nullptr);
    timeline_ = other.timeline_;
  }
  return *this;
}

ExternalSemaphore::~ExternalSemaphore() {
  if (semaphore_ != nullptr) cudaDestroyExternalSemaphore(semaphore_);
}

Result<> ExternalSemaphore::wait(std::uint64_t value, cudaStream_t stream) const {
  cudaExternalSemaphoreWaitParams params{};
  params.params.fence.value = timeline_ ? value : 0;
  return cuda_check(cudaWaitExternalSemaphoresAsync(&semaphore_, &params, 1, stream),
                    "cudaWaitExternalSemaphoresAsync");
}

Result<> ExternalSemaphore::signal(std::uint64_t value, cudaStream_t stream) const {
  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = timeline_ ? value : 0;
  return cuda_check(cudaSignalExternalSemaphoresAsync(&semaphore_, &params, 1, stream),
                    "cudaSignalExternalSemaphoresAsync");
}

Result<> write_rgba16(const Tensor& rgb, const ImportedImage& image, cudaStream_t stream) {
  if (auto ok = check_tensor(rgb, image.width(), image.height(), 3, "RGBA16 vers l'image importée"); !ok)
    return ok;
  return cuda_check(dev::rgba16_to_surface(rgb, image.surface(), stream), "lancement de rgba16_to_surface");
}

}  // namespace ajp::gpu::cuda
