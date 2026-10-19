#include "gpu/cuda/backend/tensorrt_backend.h"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <algorithm>
#include <cuda.h>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/tensorrt_true_hdr.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "prepare/builder.h"
#include "prepare/identity.h"

namespace ajp::gpu::cuda {

// Journal de TensorRT vers BackendInit::log.
class TrtLog final : public nvinfer1::ILogger {
 public:
  explicit TrtLog(BackendLog log) : log_(std::move(log)) {}
  void log(Severity severity, const char* message) noexcept override {
    if (!log_ || severity == Severity::kVERBOSE) return;
    const BackendLogLevel level = severity <= Severity::kERROR     ? BackendLogLevel::Error
                                  : severity == Severity::kWARNING ? BackendLogLevel::Warn
                                                                   : BackendLogLevel::Debug;
    try {
      log_(level, std::format("TensorRT : {}", message));
    } catch (...) {  // NOLINT(bugprone-empty-catch) : rien ne doit sortir vers TensorRT
    }
  }

 private:
  BackendLog log_;
};

namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Le contexte primaire est partagé avec NVDEC (FFmpeg, AV_CUDA_USE_PRIMARY_CONTEXT), qui le
// veut en attente bloquante (CU_CTX_SCHED_BLOCKING_SYNC) et refuse un contexte déjà actif avec
// d'autres drapeaux. Le runtime l'active avec les siens (cudaInitDevice ignore ce drapeau) :
// on les pose par l'API pilote avant toute activation.
std::expected<void, Refusal> share_primary_context(int ordinal) {
  if (const CUresult r = cuInit(0); r != CUDA_SUCCESS) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("cuInit : erreur {}", static_cast<int>(r)));
  }
  CUdevice device = 0;
  unsigned flags = 0;
  int active = 0;
  if (cuDeviceGet(&device, ordinal) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxGetState(device, &flags, &active) != CUDA_SUCCESS) {
    return refuse(RefusalCode::VendorCallFailed, "cuDevicePrimaryCtxGetState");
  }
  if (flags == CU_CTX_SCHED_BLOCKING_SYNC) return {};
  if (active != 0) {
    return refuse(
        RefusalCode::DeviceMismatch,
        std::format("contexte CUDA primaire déjà actif avec les drapeaux {:#x} : NVDEC le refuserait",
                    flags));
  }
  if (const CUresult r = cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_BLOCKING_SYNC); r != CUDA_SUCCESS) {
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("cuDevicePrimaryCtxSetFlags : erreur {}", static_cast<int>(r)));
  }
  return {};
}

DeviceUuid uuid_of(const cudaDeviceProp& properties) {
  DeviceUuid uuid{};
  for (std::size_t i = 0; i < uuid.size(); ++i) uuid[i] = static_cast<std::uint8_t>(properties.uuid.bytes[i]);
  return uuid;
}

// Tenseur alloué par cudaMalloc, libéré à sa destruction.
class CudaTensorBuffer final : public TensorBuffer {
 public:
  CudaTensorBuffer(int device, void* memory, Tensor tensor)
      : device_(device), memory_(memory), tensor_(tensor) {}
  ~CudaTensorBuffer() override {
    cudaSetDevice(device_);
    cudaFree(memory_);
  }
  CudaTensorBuffer(const CudaTensorBuffer&) = delete;
  CudaTensorBuffer& operator=(const CudaTensorBuffer&) = delete;
  const Tensor& tensor() const override { return tensor_; }

 private:
  int device_;
  void* memory_;
  Tensor tensor_;
};

std::size_t element_size(TensorType type) {
  switch (type) {
    case TensorType::F32:
      return 4;
    case TensorType::F16:
      return 2;
    case TensorType::U8:
    case TensorType::I8:
      return 1;
  }
  return 0;
}

}  // namespace

std::expected<std::unique_ptr<TensorRtBackend>, Refusal> TensorRtBackend::create(
    const BackendInit& init, std::filesystem::path trtexec) {
  std::unique_ptr<TensorRtBackend> backend(new TensorRtBackend());
  backend->log_ = init.log;
  backend->cache_dir_ = init.cache_dir;
  backend->trtexec_ = std::move(trtexec);
  int count = 0;
  if (auto ok = cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount"); !ok) {
    return refuse(RefusalCode::LibraryUnavailable, ok.error().detail);
  }
  // L'appareil du rendu, par UUID ; sans rendu (outils, tests), le premier.
  std::optional<int> chosen;
  for (int i = 0; i < count && !chosen; ++i) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, i) != cudaSuccess) continue;
    if (init.render == nullptr || uuid_of(properties) == init.render->uuid) {
      chosen = i;
      backend->properties_ = properties;
    }
  }
  if (!chosen) {
    return refuse(RefusalCode::DeviceMismatch, "aucun appareil CUDA n'a l'UUID de l'appareil de rendu");
  }
  backend->device_ = *chosen;
  backend->uuid_ = uuid_of(backend->properties_);
  if (auto ok = share_primary_context(backend->device_); !ok) return std::unexpected(ok.error());
  if (auto ok = backend->bind(); !ok) return std::unexpected(ok.error());
  cudaRuntimeGetVersion(&backend->runtime_version_);
  cudaDriverGetVersion(&backend->driver_version_);
  if (auto ok = cuda_check(cudaStreamCreateWithFlags(&backend->stream_, cudaStreamNonBlocking),
                           "cudaStreamCreateWithFlags");
      !ok) {
    return std::unexpected(ok.error());
  }
  for (Held& held : backend->held_) {
    if (auto ok = cuda_check(cudaEventCreateWithFlags(&held.done, cudaEventDisableTiming), "cudaEventCreate");
        !ok) {
      return std::unexpected(ok.error());
    }
  }
  if (auto ok = cuda_check(cudaEventCreateWithFlags(&backend->retired_done_, cudaEventDisableTiming),
                           "cudaEventCreate");
      !ok) {
    return std::unexpected(ok.error());
  }
  for (cudaEvent_t& fence : backend->fences_) {
    if (auto ok = cuda_check(cudaEventCreateWithFlags(&fence, cudaEventDisableTiming), "cudaEventCreate");
        !ok) {
      return std::unexpected(ok.error());
    }
  }
  const GpuTiming::Report report = [log = init.log](std::string_view text) {
    if (log) log(BackendLogLevel::Info, text);
  };
  backend->span_timing_ = std::make_unique<GpuTiming>("temps GPU du moteur", report);
  backend->chain_timing_ = std::make_unique<GpuTiming>("temps GPU ingestion + moteur", report);
  backend->trt_log_ = std::make_unique<TrtLog>(init.log);
  backend->runtime_.reset(nvinfer1::createInferRuntime(*backend->trt_log_));
  if (!backend->runtime_)
    return refuse(RefusalCode::LibraryUnavailable, "nvinfer1::createInferRuntime a échoué");
  backend->log(BackendLogLevel::Info,
               std::format("TensorRT {} sur {} (sm {}.{})", getInferLibVersion(), backend->properties_.name,
                           backend->properties_.major, backend->properties_.minor));
  return backend;
}

TensorRtBackend::~TensorRtBackend() {
  if (stream_ == nullptr) return;
  cudaSetDevice(device_);
  cudaStreamSynchronize(stream_);
  for (auto& [name, timing] : stages_) {
    if (const std::string line = timing->summary(); !line.empty())
      log(BackendLogLevel::Info, "profil GPU · " + line);
  }
  stages_.clear();
  slots_.clear();
  retired_.clear();
  span_timing_.reset();
  chain_timing_.reset();
  if (retired_done_ != nullptr) cudaEventDestroy(retired_done_);
  true_hdr_.reset();  // avant le flux : l'effet et ses tampons appartiennent au contexte du backend
  if (pair_device_ != nullptr) cudaFree(pair_device_);
  if (pair_host_ != nullptr) cudaFreeHost(pair_host_);
  if (flow_device_ != nullptr) cudaFree(flow_device_);
  if (flow_host_ != nullptr) cudaFreeHost(flow_host_);
  runtime_.reset();
  for (Held& held : held_) {
    held.data.reset();
    if (held.done != nullptr) cudaEventDestroy(held.done);
  }
  for (cudaEvent_t fence : fences_) {
    if (fence != nullptr) cudaEventDestroy(fence);
  }
  cudaStreamDestroy(stream_);
}

std::expected<void, Refusal> TensorRtBackend::bind() const {
  return cuda_check(cudaSetDevice(device_), "cudaSetDevice");
}

void TensorRtBackend::log(BackendLogLevel level, std::string_view message) const {
  if (log_) log_(level, message);
}

BackendCaps TensorRtBackend::capabilities() const {
  return BackendCaps{
      .name = "tensorrt",
      .device_name = properties_.name,
      .driver_version = std::format("CUDA {}.{}", driver_version_ / 1000, driver_version_ % 1000 / 10),
      .needs_host_sync = false,
      .fp16 = true,
      .int8 = false,
      .matrix_units = true,
      .import_formats = {PixelFormat::Nv12, PixelFormat::P010, PixelFormat::P016}};
}

DeviceMatch TensorRtBackend::match(const DeviceUuid& render_device) const {
  return render_device == uuid_ ? DeviceMatch::Same : DeviceMatch::Different;
}

std::expected<Artifact, Refusal> TensorRtBackend::build(const ModelSpec& model, const ShapeSet& shapes,
                                                        Progress& progress) {
  const prepare::TrtexecConfig config{
      .trtexec = trtexec_,
      .cache_dir = cache_dir_,
      .runtime =
          [&] {
            prepare::TrtRuntime runtime = prepare::decode_infer_lib_version(getInferLibVersion());
            runtime.headers_build = NV_TENSORRT_BUILD;
            runtime.cuda_runtime = runtime_version_;
            runtime.cuda_driver = driver_version_;
            return runtime;
          }(),
      .device = {.name = properties_.name,
                 .sm_major = properties_.major,
                 .sm_minor = properties_.minor,
                 .uuid = uuid_},
  };
  // Un moteur publié doit se désérialiser ici : sinon le constructeur le refait.
  const prepare::EngineCheck check =
      [this](const std::filesystem::path& engine) -> std::expected<void, Refusal> {
    if (auto ok = bind(); !ok) return ok;
    auto loaded = TrtEngine::load(*runtime_, engine);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    return {};
  };
  return prepare::TrtexecBuilder(config, check).build(model, shapes, progress);
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load(const Artifact& artifact) {
  if (auto ok = bind(); !ok) return std::unexpected(ok.error());
  auto engine = TrtEngine::load(*runtime_, artifact.path);
  if (!engine) return std::unexpected(std::move(engine.error()));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<Tensor, Refusal> TensorRtBackend::import(const ExternalImage& image) {
  return refuse(
      RefusalCode::FormatUnsupported,
      std::format("import d'une image {}x{} : pas encore dans ce backend", image.width, image.height));
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> TensorRtBackend::alloc_tensor(TensorType type,
                                                                                    const Shape& shape) {
  if (type != TensorType::F16 && type != TensorType::F32) {
    return refuse(RefusalCode::FormatUnsupported, "tenseur : F16 ou F32 seulement");
  }
  std::size_t bytes = element_size(type);
  for (const std::int64_t d : shape.dims) {
    if (d <= 0 || static_cast<std::uint64_t>(d) > (std::uint64_t{1} << 40) / bytes) {
      return refuse(RefusalCode::ShapeInvalid, "tenseur : dimension nulle, négative ou trop grande");
    }
    bytes *= static_cast<std::size_t>(d);
  }
  if (auto ok = bind(); !ok) return std::unexpected(ok.error());
  void* memory = nullptr;
  if (auto ok = cuda_check(cudaMalloc(&memory, bytes), "cudaMalloc"); !ok) return std::unexpected(ok.error());
  const Tensor tensor{.type = type, .shape = shape, .handle = reinterpret_cast<std::uintptr_t>(memory)};
  return std::make_unique<CudaTensorBuffer>(device_, memory, tensor);
}

GpuTiming& TensorRtBackend::stage(std::string_view name) {
  auto found = stages_.find(name);
  if (found == stages_.end())
    found = stages_.emplace(std::string(name), std::make_unique<GpuTiming>(std::string(name), nullptr)).first;
  return *found->second;
}

std::expected<void, Refusal> TensorRtBackend::enqueue(Engine& engine, std::span<const Tensor> inputs,
                                                      std::span<Tensor> outputs) {
  auto* trt = dynamic_cast<TrtEngine*>(&engine);
  if (trt == nullptr) return refuse(RefusalCode::DeviceMismatch, "enqueue : moteur d'un autre backend");
  if (auto ok = bind(); !ok) return ok;
  const Shape shape = trt->input_shape(0);
  GpuTiming& timing = stage(std::format("moteur {}×{}×{}", shape.dims[1], shape.dims[2], shape.dims[3]));
  span_timing_->begin(stream_);
  timing.begin(stream_);
  auto queued = trt->enqueue(stream_, inputs, outputs);
  timing.end(stream_);
  span_timing_->end(stream_);
  chain_timing_->end(stream_);
  return queued;
}

std::expected<Fence, Refusal> TensorRtBackend::signal() {
  cudaEvent_t event = fences_[fence_count_ % kFences];
  ++fence_count_;
  if (auto ok =
          bind().and_then([&] { return cuda_check(cudaEventRecord(event, stream_), "cudaEventRecord"); });
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  return Fence{.handle = reinterpret_cast<std::uintptr_t>(event), .value = fence_count_};
}

std::expected<void, Refusal> TensorRtBackend::wait(const Fence& fence) {
  auto* event = reinterpret_cast<cudaEvent_t>(fence.handle);  // NOLINT(performance-no-int-to-ptr)
  return bind().and_then(
      [&] { return cuda_check(cudaStreamWaitEvent(stream_, event, 0), "cudaStreamWaitEvent"); });
}

}  // namespace ajp::gpu::cuda
