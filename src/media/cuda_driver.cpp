// Types et signatures du pilote CUDA d'après nv-codec-headers n13.0.19.0
// (include/ffnvcodec/dynlink_cuda.h, dynlink_loader.h), que FFmpeg charge de
// la même façon. Aucun en-tête CUDA n'est nécessaire à la construction.
#include "media/cuda_driver.h"

#include <bit>
#include <cstring>
#include <format>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::media {
namespace {

using CUresult = int;
using CUdevice = int;
struct CUctxOpaque;
struct CUstreamOpaque;
struct CUeventOpaque;
using CUcontext = CUctxOpaque*;
using CUstream = CUstreamOpaque*;
using CUevent = CUeventOpaque*;
struct CUuuid {
  std::array<char, 16> bytes;
};

constexpr CUresult kCudaSuccess = 0;
constexpr unsigned kStreamNonBlocking = 1;   // CU_STREAM_NON_BLOCKING
constexpr unsigned kEventDisableTiming = 2;  // CU_EVENT_DISABLE_TIMING

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
  CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;
  CUresult (*cuCtxPopCurrent)(CUcontext*) = nullptr;
  CUresult (*cuStreamCreate)(CUstream*, unsigned) = nullptr;
  CUresult (*cuStreamDestroy)(CUstream) = nullptr;
  CUresult (*cuEventCreate)(CUevent*, unsigned) = nullptr;
  CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*cuEventDestroy)(CUevent) = nullptr;
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

CudaDriver::CudaDriver(void* library, std::unique_ptr<Api> api) : library_(library), api_(std::move(api)) {}

CudaDriver::~CudaDriver() {
  close_library(library_);
}

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
                     bind(library, "cuCtxPushCurrent_v2", api->cuCtxPushCurrent) &&
                     bind(library, "cuCtxPopCurrent_v2", api->cuCtxPopCurrent) &&
                     bind(library, "cuStreamCreate", api->cuStreamCreate) &&
                     bind(library, "cuStreamDestroy_v2", api->cuStreamDestroy) &&
                     bind(library, "cuEventCreate", api->cuEventCreate) &&
                     bind(library, "cuEventRecord", api->cuEventRecord) &&
                     bind(library, "cuEventDestroy_v2", api->cuEventDestroy);
  if (!bound) {
    close_library(library);
    return std::unexpected(
        Refusal{RefusalCode::HwDecodeUnavailable, std::format("{} : symbole absent", kLibrary)});
  }
  if (const CUresult result = api->cuInit(0); result != kCudaSuccess) {
    close_library(library);
    return refuse_cuda("cuInit", result);
  }
  return std::shared_ptr<CudaDriver>(new CudaDriver(library, std::move(api)));
}

std::expected<int, Refusal> CudaDriver::ordinal_of(const CudaUuid& uuid) const {
  int count = 0;
  if (const CUresult result = api_->cuDeviceGetCount(&count); result != kCudaSuccess) {
    return refuse_cuda("cuDeviceGetCount", result);
  }
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = 0;
    CUuuid found{};
    if (api_->cuDeviceGet(&device, ordinal) != kCudaSuccess) continue;
    if (api_->cuDeviceGetUuid(&found, device) != kCudaSuccess) continue;
    if (std::memcmp(found.bytes.data(), uuid.data(), uuid.size()) == 0) return ordinal;
  }
  return std::unexpected(
      Refusal{RefusalCode::HwDecodeUnavailable,
              std::format("aucun appareil CUDA n'a l'UUID du rendu ({} appareils)", count)});
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

}  // namespace ajp::media
