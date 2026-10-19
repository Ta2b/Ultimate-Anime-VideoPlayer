#include "cuda_driver.h"

#include <dlfcn.h>
#include <format>

#include <catch2/catch_test_macros.hpp>

namespace ajp::vendor_hdr::test {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using Handle = void*;  // CUcontext, CUstream, CUevent

constexpr unsigned kStreamNonBlocking = 0x1;   // CU_STREAM_NON_BLOCKING
constexpr unsigned kEventDisableTiming = 0x2;  // CU_EVENT_DISABLE_TIMING
const Handle kLegacyStream = nullptr;          // flux 0 hérité (API pilote, sans _ptds)

struct CudaApi {
  void* library = nullptr;
  CUresult (*init)(unsigned) = nullptr;
  CUresult (*device_count)(int*) = nullptr;
  CUresult (*device_get)(CUdevice*, int) = nullptr;
  CUresult (*device_name)(char*, int, CUdevice) = nullptr;
  CUresult (*retain)(Handle*, CUdevice) = nullptr;
  CUresult (*release)(CUdevice) = nullptr;
  CUresult (*set_current)(Handle) = nullptr;
  CUresult (*synchronize)() = nullptr;
  CUresult (*error_string)(CUresult, const char**) = nullptr;
  CUresult (*alloc)(CUdeviceptr*, std::size_t) = nullptr;
  CUresult (*free)(CUdeviceptr) = nullptr;
  CUresult (*htod)(CUdeviceptr, const void*, std::size_t) = nullptr;
  CUresult (*dtoh)(void*, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*htod_async)(CUdeviceptr, const void*, std::size_t, Handle) = nullptr;
  CUresult (*dtoh_async)(void*, CUdeviceptr, std::size_t, Handle) = nullptr;
  CUresult (*stream_create)(Handle*, unsigned) = nullptr;
  CUresult (*stream_destroy)(Handle) = nullptr;
  CUresult (*stream_synchronize)(Handle) = nullptr;
  CUresult (*stream_wait_event)(Handle, Handle, unsigned) = nullptr;
  CUresult (*event_create)(Handle*, unsigned) = nullptr;
  CUresult (*event_destroy)(Handle) = nullptr;
  CUresult (*event_record)(Handle, Handle) = nullptr;

  ~CudaApi() {
    if (library != nullptr) dlclose(library);
  }
};

namespace {

template <class Fn>
bool bind(void* library, const char* name, Fn*& slot) {
  slot = reinterpret_cast<Fn*>(dlsym(library, name));
  return slot != nullptr;
}

std::unique_ptr<CudaApi> load_api(std::string& why) {
  auto api = std::make_unique<CudaApi>();
  api->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (api->library == nullptr) {
    why = "pilote CUDA absent (libcuda.so.1)";
    return nullptr;
  }
  void* lib = api->library;
  const bool all =
      bind(lib, "cuInit", api->init) && bind(lib, "cuDeviceGetCount", api->device_count) &&
      bind(lib, "cuDeviceGet", api->device_get) && bind(lib, "cuDeviceGetName", api->device_name) &&
      bind(lib, "cuDevicePrimaryCtxRetain", api->retain) &&
      bind(lib, "cuDevicePrimaryCtxRelease_v2", api->release) &&
      bind(lib, "cuCtxSetCurrent", api->set_current) && bind(lib, "cuCtxSynchronize", api->synchronize) &&
      bind(lib, "cuGetErrorString", api->error_string) && bind(lib, "cuMemAlloc_v2", api->alloc) &&
      bind(lib, "cuMemFree_v2", api->free) && bind(lib, "cuMemcpyHtoD_v2", api->htod) &&
      bind(lib, "cuMemcpyDtoH_v2", api->dtoh) && bind(lib, "cuMemcpyHtoDAsync_v2", api->htod_async) &&
      bind(lib, "cuMemcpyDtoHAsync_v2", api->dtoh_async) && bind(lib, "cuStreamCreate", api->stream_create) &&
      bind(lib, "cuStreamDestroy_v2", api->stream_destroy) &&
      bind(lib, "cuStreamSynchronize", api->stream_synchronize) &&
      bind(lib, "cuStreamWaitEvent", api->stream_wait_event) &&
      bind(lib, "cuEventCreate", api->event_create) && bind(lib, "cuEventDestroy_v2", api->event_destroy) &&
      bind(lib, "cuEventRecord", api->event_record);
  if (!all) {
    why = "libcuda.so.1 sans les fonctions attendues";
    return nullptr;
  }
  return api;
}

Handle handle(std::uintptr_t value) {
  return reinterpret_cast<Handle>(value);  // NOLINT(performance-no-int-to-ptr)
}

std::uintptr_t value(Handle h) {
  return reinterpret_cast<std::uintptr_t>(h);
}

}  // namespace

Cuda::Cuda(std::unique_ptr<CudaApi> api, int device, std::uintptr_t context)
    : api_(std::move(api)), device_(device), context_(context) {}

std::unique_ptr<Cuda> Cuda::open(std::string& why) {
  auto api = load_api(why);
  if (!api) return nullptr;
  int count = 0;
  if (api->init(0) != 0 || api->device_count(&count) != 0 || count == 0) {
    why = "aucun appareil CUDA";
    return nullptr;
  }
  CUdevice device = 0;
  Handle context = nullptr;
  if (api->device_get(&device, 0) != 0 || api->retain(&context, device) != 0 ||
      api->set_current(context) != 0) {
    why = "contexte CUDA impossible sur l'appareil 0";
    return nullptr;
  }
  return std::unique_ptr<Cuda>(new Cuda(std::move(api), device, value(context)));
}

Cuda::~Cuda() {
  api_->synchronize();
  api_->release(device_);
}

std::string Cuda::device_name() const {
  std::string name(256, '\0');
  check(api_->device_name(name.data(), static_cast<int>(name.size()), device_), "cuDeviceGetName");
  name.resize(name.find('\0'));
  return name;
}

void Cuda::check(int result, const char* call) const {
  if (result == 0) return;
  const char* text = nullptr;
  api_->error_string(result, &text);
  FAIL(std::format("{} : {} ({})", call, text != nullptr ? text : "erreur CUDA", result));
}

void Cuda::synchronize() const {
  check(api_->synchronize(), "cuCtxSynchronize");
}

DeviceImage::DeviceImage(const Cuda& cuda, int width, int height, TrueHdrPixel pixel, std::size_t pitch)
    : cuda_(cuda),
      width_(width),
      height_(height),
      pitch_(pitch != 0 ? pitch : static_cast<std::size_t>(width) * 4),
      pixel_(pixel) {
  CUdeviceptr device = 0;
  cuda_.check(cuda_.api().alloc(&device, pitch_ * static_cast<std::size_t>(height)), "cuMemAlloc");
  device_ = device;
}

DeviceImage::~DeviceImage() {
  cuda_.api().free(device_);
}

TrueHdrImage DeviceImage::view() const {
  return TrueHdrImage{
      .device = device_, .pitch = pitch_, .width = width_, .height = height_, .pixel = pixel_};
}

void DeviceImage::upload(const std::vector<std::uint32_t>& pixels) const {
  const auto width = static_cast<std::size_t>(width_);
  const auto height = static_cast<std::size_t>(height_);
  REQUIRE(pixels.size() == width * height);
  for (std::size_t y = 0; y < height; ++y) {
    cuda_.check(cuda_.api().htod(device_ + (y * pitch_), pixels.data() + (y * width), width * 4),
                "cuMemcpyHtoD");
  }
}

std::vector<std::uint32_t> DeviceImage::download() const {
  const auto width = static_cast<std::size_t>(width_);
  const auto height = static_cast<std::size_t>(height_);
  std::vector<std::uint32_t> pixels(width * height);
  for (std::size_t y = 0; y < height; ++y) {
    cuda_.check(cuda_.api().dtoh(pixels.data() + (y * width), device_ + (y * pitch_), width * 4),
                "cuMemcpyDtoH");
  }
  return pixels;
}

CudaGpuStream::CudaGpuStream(const Cuda& cuda) : cuda_(cuda) {
  Handle stream = nullptr;
  Handle input = nullptr;
  Handle output = nullptr;
  cuda_.check(cuda_.api().stream_create(&stream, kStreamNonBlocking), "cuStreamCreate");
  cuda_.check(cuda_.api().event_create(&input, kEventDisableTiming), "cuEventCreate");
  cuda_.check(cuda_.api().event_create(&output, kEventDisableTiming), "cuEventCreate");
  stream_ = value(stream);
  input_ready_ = value(input);
  output_ready_ = value(output);
}

CudaGpuStream::~CudaGpuStream() {
  cuda_.api().stream_synchronize(handle(stream_));
  cuda_.api().event_destroy(handle(input_ready_));
  cuda_.api().event_destroy(handle(output_ready_));
  cuda_.api().stream_destroy(handle(stream_));
}

Done CudaGpuStream::call(int result, const char* what) const {
  if (result == 0) return {};
  return refuse(RefusalCode::VendorCallFailed, std::format("{} a échoué ({})", what, result));
}

Done CudaGpuStream::hand_to_legacy() {
  if (Done recorded = call(cuda_.api().event_record(handle(input_ready_), handle(stream_)), "cuEventRecord");
      !recorded)
    return recorded;
  return call(cuda_.api().stream_wait_event(kLegacyStream, handle(input_ready_), 0), "cuStreamWaitEvent");
}

Done CudaGpuStream::take_from_legacy() {
  if (Done recorded = call(cuda_.api().event_record(handle(output_ready_), kLegacyStream), "cuEventRecord");
      !recorded)
    return recorded;
  return call(cuda_.api().stream_wait_event(handle(stream_), handle(output_ready_), 0), "cuStreamWaitEvent");
}

void CudaGpuStream::upload(const DeviceImage& image, const std::vector<std::uint32_t>& pixels) const {
  const TrueHdrImage view = image.view();
  REQUIRE(view.pitch == static_cast<std::size_t>(view.width) * 4);
  cuda_.check(cuda_.api().htod_async(view.device, pixels.data(), pixels.size() * 4, handle(stream_)),
              "cuMemcpyHtoDAsync");
  // Mémoire hôte paginable : la copie est finie au retour, mais seulement dans l'ordre du flux.
}

std::vector<std::uint32_t> CudaGpuStream::download(const DeviceImage& image) const {
  const TrueHdrImage view = image.view();
  REQUIRE(view.pitch == static_cast<std::size_t>(view.width) * 4);
  std::vector<std::uint32_t> pixels(static_cast<std::size_t>(view.width) *
                                    static_cast<std::size_t>(view.height));
  cuda_.check(cuda_.api().dtoh_async(pixels.data(), view.device, pixels.size() * 4, handle(stream_)),
              "cuMemcpyDtoHAsync");
  cuda_.check(cuda_.api().stream_synchronize(handle(stream_)), "cuStreamSynchronize");
  return pixels;
}

}  // namespace ajp::vendor_hdr::test
