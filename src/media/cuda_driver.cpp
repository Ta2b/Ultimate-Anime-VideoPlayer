// Types et signatures du pilote CUDA d'après nv-codec-headers n13.0.19.0
// (include/ffnvcodec/dynlink_cuda.h, dynlink_loader.h), que FFmpeg charge de
// la même façon. Aucun en-tête CUDA n'est nécessaire à la construction.
#include "media/cuda_driver.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace ajp::media {
namespace {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctxOpaque;
struct CUstreamOpaque;
struct CUeventOpaque;
struct CUextMemoryOpaque;
struct CUarrayOpaque;
using CUcontext = CUctxOpaque*;
using CUstream = CUstreamOpaque*;
using CUevent = CUeventOpaque*;
using CUexternalMemory = CUextMemoryOpaque*;
using CUarray = CUarrayOpaque*;
struct CUuuid {
  std::array<char, 16> bytes;
};
// CUDA_EXTERNAL_MEMORY_HANDLE_DESC et CUDA_EXTERNAL_MEMORY_BUFFER_DESC, à l'identique.
struct ExternalMemoryHandleDesc {
  int type = 0;
  union {
    int fd;
    struct {
      void* handle;
      const void* name;
    } win32;
  } handle{};
  unsigned long long size = 0;
  unsigned flags = 0;
  std::array<unsigned, 16> reserved{};
};
struct ExternalMemoryBufferDesc {
  unsigned long long offset = 0;
  unsigned long long size = 0;
  unsigned flags = 0;
  std::array<unsigned, 16> reserved{};
};
// CUDA_MEMCPY2D, à l'identique : les types de mémoire sont des enum (unsigned), suivis d'un
// pointeur, donc du même bourrage.
struct Memcpy2D {
  std::size_t src_x_bytes = 0;
  std::size_t src_y = 0;
  unsigned src_memory_type = 0;
  const void* src_host = nullptr;
  CUdeviceptr src_device = 0;
  CUarray src_array = nullptr;
  std::size_t src_pitch = 0;
  std::size_t dst_x_bytes = 0;
  std::size_t dst_y = 0;
  unsigned dst_memory_type = 0;
  void* dst_host = nullptr;
  CUdeviceptr dst_device = 0;
  CUarray dst_array = nullptr;
  std::size_t dst_pitch = 0;
  std::size_t width_bytes = 0;
  std::size_t height = 0;
};

constexpr CUresult kCudaSuccess = 0;
constexpr unsigned kMemoryTypeDevice = 2;    // CU_MEMORYTYPE_DEVICE
constexpr unsigned kStreamNonBlocking = 1;   // CU_STREAM_NON_BLOCKING
constexpr unsigned kEventDisableTiming = 2;  // CU_EVENT_DISABLE_TIMING
constexpr int kHandleOpaqueFd = 1;           // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
constexpr int kHandleOpaqueWin32 = 2;        // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
constexpr unsigned kMemoryDedicated = 1;     // CUDA_EXTERNAL_MEMORY_DEDICATED

#if defined(_WIN32)
constexpr const char* kLibrary = "nvcuda.dll";
void* open_library() {
  return reinterpret_cast<void*>(LoadLibraryA(kLibrary));
}
void* find_symbol(void* library, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
}
void close_library(void* library) {
  FreeLibrary(static_cast<HMODULE>(library));
}
#else
constexpr const char* kLibrary = "libcuda.so.1";
void* open_library() {
  return dlopen(kLibrary, RTLD_NOW | RTLD_LOCAL);
}
void* find_symbol(void* library, const char* name) {
  return dlsym(library, name);
}
void close_library(void* library) {
  dlclose(library);
}
#endif

std::unexpected<Refusal> refuse_cuda(std::string_view where, CUresult result) {
  return std::unexpected(
      Refusal{RefusalCode::HwDecodeUnavailable, std::format("CUDA {} : erreur {}", where, result)});
}

}  // namespace

struct CudaDriver::Api {
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuDeviceGetCount)(int*) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuDeviceGetUuid)(CUuuid*, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
  CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;
  CUresult (*cuCtxPopCurrent)(CUcontext*) = nullptr;
  CUresult (*cuStreamCreate)(CUstream*, unsigned) = nullptr;
  CUresult (*cuStreamDestroy)(CUstream) = nullptr;
  CUresult (*cuEventCreate)(CUevent*, unsigned) = nullptr;
  CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*cuEventDestroy)(CUevent) = nullptr;
  CUresult (*cuImportExternalMemory)(CUexternalMemory*, const ExternalMemoryHandleDesc*) = nullptr;
  CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory,
                                              const ExternalMemoryBufferDesc*) = nullptr;
  CUresult (*cuDestroyExternalMemory)(CUexternalMemory) = nullptr;
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
  CUresult (*cuMemcpy2DAsync)(const Memcpy2D*, CUstream) = nullptr;
  CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
};

namespace {

template <class Fn>
bool bind(void* library, const char* name, Fn& slot) {
  void* symbol = find_symbol(library, name);
  if (symbol == nullptr) return false;
  // Pointeur d'objet vers pointeur de fonction : conversion admise par POSIX
  // (dlsym) et par Windows (GetProcAddress).
  slot = reinterpret_cast<Fn>(symbol);
  return true;
}

// Rend le contexte courant le temps d'un appel, comme FFmpeg autour de NVDEC.
class ContextScope {
 public:
  ContextScope(const CudaDriver::Api& api, std::uintptr_t context) : api_(api) {
    ok_ = api_.cuCtxPushCurrent(std::bit_cast<CUcontext>(context)) == kCudaSuccess;
  }
  ~ContextScope() {
    CUcontext previous = nullptr;
    if (ok_) api_.cuCtxPopCurrent(&previous);
  }
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
  bool ok() const { return ok_; }

 private:
  const CudaDriver::Api& api_;
  bool ok_ = false;
};

}  // namespace

CudaDriver::CudaDriver(std::unique_ptr<Api> api) : api_(std::move(api)) {}

// Pas de dlclose (ni FreeLibrary) : voir l'en-tête.
CudaDriver::~CudaDriver() = default;

std::expected<std::shared_ptr<CudaDriver>, Refusal> CudaDriver::load() {
  void* library = open_library();
  if (library == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::HwDecodeUnavailable, std::format("{} introuvable", kLibrary)});
  }
  auto api = std::make_unique<Api>();
  const bool bound = bind(library, "cuInit", api->cuInit) &&
                     bind(library, "cuDeviceGetCount", api->cuDeviceGetCount) &&
                     bind(library, "cuDeviceGet", api->cuDeviceGet) &&
                     bind(library, "cuDeviceGetUuid", api->cuDeviceGetUuid) &&
                     bind(library, "cuDevicePrimaryCtxRetain", api->cuDevicePrimaryCtxRetain) &&
                     bind(library, "cuDevicePrimaryCtxRelease_v2", api->cuDevicePrimaryCtxRelease) &&
                     bind(library, "cuCtxPushCurrent_v2", api->cuCtxPushCurrent) &&
                     bind(library, "cuCtxPopCurrent_v2", api->cuCtxPopCurrent) &&
                     bind(library, "cuStreamCreate", api->cuStreamCreate) &&
                     bind(library, "cuStreamDestroy_v2", api->cuStreamDestroy) &&
                     bind(library, "cuEventCreate", api->cuEventCreate) &&
                     bind(library, "cuEventRecord", api->cuEventRecord) &&
                     bind(library, "cuEventDestroy_v2", api->cuEventDestroy) &&
                     bind(library, "cuImportExternalMemory", api->cuImportExternalMemory) &&
                     bind(library, "cuExternalMemoryGetMappedBuffer", api->cuExternalMemoryGetMappedBuffer) &&
                     bind(library, "cuDestroyExternalMemory", api->cuDestroyExternalMemory) &&
                     bind(library, "cuMemFree_v2", api->cuMemFree) &&
                     bind(library, "cuMemcpy2DAsync_v2", api->cuMemcpy2DAsync) &&
                     bind(library, "cuStreamSynchronize", api->cuStreamSynchronize);
  if (!bound) {
    close_library(library);
    return std::unexpected(
        Refusal{RefusalCode::HwDecodeUnavailable, std::format("{} : symbole absent", kLibrary)});
  }
  // Dès cet appel, la bibliothèque reste chargée, même si cuInit échoue : il a
  // pu lancer ses threads avant de rendre l'erreur.
  if (const CUresult result = api->cuInit(0); result != kCudaSuccess) return refuse_cuda("cuInit", result);
  return std::shared_ptr<CudaDriver>(new CudaDriver(std::move(api)));
}

std::expected<std::vector<CudaUuid>, Refusal> CudaDriver::uuids() const {
  int count = 0;
  if (const CUresult result = api_->cuDeviceGetCount(&count); result != kCudaSuccess) {
    return refuse_cuda("cuDeviceGetCount", result);
  }
  std::vector<CudaUuid> found(static_cast<std::size_t>(count));
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = 0;
    if (const CUresult result = api_->cuDeviceGet(&device, ordinal); result != kCudaSuccess) {
      return refuse_cuda("cuDeviceGet", result);
    }
    CUuuid uuid{};
    if (const CUresult result = api_->cuDeviceGetUuid(&uuid, device); result != kCudaSuccess) {
      return refuse_cuda("cuDeviceGetUuid", result);
    }
    std::memcpy(found[static_cast<std::size_t>(ordinal)].data(), uuid.bytes.data(), uuid.bytes.size());
  }
  return found;
}

std::expected<int, Refusal> CudaDriver::ordinal_of(const CudaUuid& uuid) const {
  auto all = uuids();
  if (!all) return std::unexpected(all.error());
  const auto it = std::ranges::find(*all, uuid);
  if (it != all->end()) return static_cast<int>(it - all->begin());
  return std::unexpected(
      Refusal{RefusalCode::HwDecodeUnavailable,
              std::format("aucun appareil CUDA n'a l'UUID du rendu ({} appareils)", all->size())});
}

std::expected<std::uintptr_t, Refusal> CudaDriver::primary_context(int ordinal) const {
  CUdevice device = 0;
  if (const CUresult result = api_->cuDeviceGet(&device, ordinal); result != kCudaSuccess) {
    return refuse_cuda("cuDeviceGet", result);
  }
  CUcontext context = nullptr;
  if (const CUresult result = api_->cuDevicePrimaryCtxRetain(&context, device); result != kCudaSuccess) {
    return refuse_cuda("cuDevicePrimaryCtxRetain", result);
  }
  if (const CUresult result = api_->cuDevicePrimaryCtxRelease(device); result != kCudaSuccess) {
    return refuse_cuda("cuDevicePrimaryCtxRelease", result);
  }
  return std::bit_cast<std::uintptr_t>(context);
}

std::expected<PrimaryContext, Refusal> CudaDriver::retain_primary(int ordinal) const {
  CUdevice device = 0;
  if (const CUresult result = api_->cuDeviceGet(&device, ordinal); result != kCudaSuccess) {
    return refuse_cuda("cuDeviceGet", result);
  }
  CUcontext context = nullptr;
  if (const CUresult result = api_->cuDevicePrimaryCtxRetain(&context, device); result != kCudaSuccess) {
    return refuse_cuda("cuDevicePrimaryCtxRetain", result);
  }
  std::shared_ptr<const CudaDriver> driver = shared_from_this();
  std::shared_ptr<const void> owner(
      context, [driver, device](CUctxOpaque*) { driver->api_->cuDevicePrimaryCtxRelease(device); });
  return PrimaryContext{.context = std::bit_cast<std::uintptr_t>(context), .owner = std::move(owner)};
}

std::expected<ImportedMemory, Refusal> CudaDriver::import_memory(std::uintptr_t context,
                                                                 std::uintptr_t handle, std::size_t size,
                                                                 std::size_t offset, std::size_t bytes,
                                                                 bool dedicated) const {
  const ContextScope scope(*api_, context);
  if (!scope.ok()) return refuse_cuda("cuCtxPushCurrent", -1);
  ExternalMemoryHandleDesc desc{};
  desc.size = size;
  if (dedicated) desc.flags = kMemoryDedicated;
#if defined(_WIN32)
  desc.type = kHandleOpaqueWin32;
  desc.handle.win32.handle = std::bit_cast<void*>(handle);
#else
  // CUDA prend le descripteur importé ; Vulkan garde le sien, fermé avec le tampon.
  desc.type = kHandleOpaqueFd;
  desc.handle.fd = dup(static_cast<int>(handle));
  if (desc.handle.fd < 0) return std::unexpected(Refusal{RefusalCode::IoFailed, "dup du descripteur Vulkan"});
#endif
  CUexternalMemory memory = nullptr;
  if (const CUresult result = api_->cuImportExternalMemory(&memory, &desc); result != kCudaSuccess) {
#if !defined(_WIN32)
    close(desc.handle.fd);  // pris par CUDA seulement en cas de succès
#endif
    return refuse_cuda("cuImportExternalMemory", result);
  }
  const ExternalMemoryBufferDesc buffer{.offset = offset, .size = bytes};
  CUdeviceptr pointer = 0;
  if (const CUresult result = api_->cuExternalMemoryGetMappedBuffer(&pointer, memory, &buffer);
      result != kCudaSuccess) {
    api_->cuDestroyExternalMemory(memory);
    return refuse_cuda("cuExternalMemoryGetMappedBuffer", result);
  }
  // Le tampon mappé se libère avant la mémoire importée, dans le contexte, par le dernier
  // propriétaire, depuis n'importe quel thread.
  std::shared_ptr<const CudaDriver> driver = shared_from_this();
  std::shared_ptr<const void> owner(memory, [driver, context, pointer](CUextMemoryOpaque* m) {
    const ContextScope destroy_scope(*driver->api_, context);
    driver->api_->cuMemFree(pointer);
    driver->api_->cuDestroyExternalMemory(m);
  });
  return ImportedMemory{.pointer = static_cast<std::uintptr_t>(pointer), .owner = std::move(owner)};
}

std::expected<std::uintptr_t, Refusal> CudaDriver::create_stream(std::uintptr_t context) const {
  const ContextScope scope(*api_, context);
  if (!scope.ok()) return refuse_cuda("cuCtxPushCurrent", -1);
  CUstream stream = nullptr;
  if (const CUresult result = api_->cuStreamCreate(&stream, kStreamNonBlocking); result != kCudaSuccess) {
    return refuse_cuda("cuStreamCreate", result);
  }
  return std::bit_cast<std::uintptr_t>(stream);
}

void CudaDriver::destroy_stream(std::uintptr_t context, std::uintptr_t stream) const {
  const ContextScope scope(*api_, context);
  api_->cuStreamDestroy(std::bit_cast<CUstream>(stream));
}

std::expected<ReadyEvent, Refusal> CudaDriver::record(std::uintptr_t context, std::uintptr_t stream) const {
  const ContextScope scope(*api_, context);
  if (!scope.ok()) return refuse_cuda("cuCtxPushCurrent", -1);
  CUevent event = nullptr;
  if (const CUresult result = api_->cuEventCreate(&event, kEventDisableTiming); result != kCudaSuccess) {
    return refuse_cuda("cuEventCreate", result);
  }
  if (const CUresult result = api_->cuEventRecord(event, std::bit_cast<CUstream>(stream));
      result != kCudaSuccess) {
    api_->cuEventDestroy(event);
    return refuse_cuda("cuEventRecord", result);
  }
  // L'événement vit tant qu'une copie de l'image le tient ; le pilote aussi. Il
  // est détruit par le thread qui relâche la dernière copie (gpu), dans son
  // contexte, comme FFmpeg le fait pour les siens.
  std::shared_ptr<const CudaDriver> driver = shared_from_this();
  std::shared_ptr<const void> owner(event, [driver, context](CUeventOpaque* e) {
    const ContextScope destroy_scope(*driver->api_, context);
    driver->api_->cuEventDestroy(e);
  });
  return ReadyEvent{.handle = std::bit_cast<std::uintptr_t>(event), .owner = std::move(owner)};
}

std::expected<void, Refusal> CudaDriver::copy_2d(std::uintptr_t context, std::uintptr_t stream,
                                                 std::uintptr_t dst, std::size_t dst_pitch,
                                                 std::uintptr_t src, std::size_t src_pitch,
                                                 std::size_t width_bytes, std::size_t height) const {
  const ContextScope scope(*api_, context);
  if (!scope.ok()) return refuse_cuda("cuCtxPushCurrent", -1);
  Memcpy2D copy{};
  copy.src_memory_type = kMemoryTypeDevice;
  copy.src_device = static_cast<CUdeviceptr>(src);
  copy.src_pitch = src_pitch;
  copy.dst_memory_type = kMemoryTypeDevice;
  copy.dst_device = static_cast<CUdeviceptr>(dst);
  copy.dst_pitch = dst_pitch;
  copy.width_bytes = width_bytes;
  copy.height = height;
  if (const CUresult result = api_->cuMemcpy2DAsync(&copy, std::bit_cast<CUstream>(stream));
      result != kCudaSuccess) {
    return refuse_cuda("cuMemcpy2DAsync", result);
  }
  return {};
}

std::expected<void, Refusal> CudaDriver::synchronize(std::uintptr_t context, std::uintptr_t stream) const {
  const ContextScope scope(*api_, context);
  if (!scope.ok()) return refuse_cuda("cuCtxPushCurrent", -1);
  if (const CUresult result = api_->cuStreamSynchronize(std::bit_cast<CUstream>(stream));
      result != kCudaSuccess)
    return refuse_cuda("cuStreamSynchronize", result);
  return {};
}

}  // namespace ajp::media
