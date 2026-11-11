#include "gpu/cuda/backend/tensorrt_backend.h"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cuda.h>
#include <format>
#include <memory>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"
#include "gpu/cuda/backend/drba_fused_engine.h"
#include "gpu/cuda/backend/rife_fused_engine.h"
#include "gpu/cuda/backend/shared_activations.h"
#include "gpu/cuda/backend/span_fp8_engine.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "prepare/builder.h"
#include "prepare/drba_split.h"
#include "prepare/identity.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace ajp::gpu::cuda {
namespace {
#ifdef _WIN32
// Ordonnancement GPU matériel (HAGS) : HwSchMode 2 dans le registre. Sans lui, WDDM garde les lancements
// dans un lot logiciel que seule une interrogation du flux pousse (push_launches).
std::string hags_state() {
  DWORD mode = 0;
  DWORD size = sizeof(mode);
  const LSTATUS status =
      RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", L"HwSchMode",
                   RRF_RT_REG_DWORD, nullptr, &mode, &size);
  if (status != ERROR_SUCCESS) return "ordonnancement GPU matériel (HAGS) : état illisible";
  return mode == 2 ? "ordonnancement GPU matériel (HAGS) : actif"
                   : "ordonnancement GPU matériel (HAGS) : inactif, envoi WDDM forcé à chaque étape";
}
#endif
}  // namespace

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
  // Le mode d'ordonnancement seul : un autre bit (carte hôte, mémoire locale) ne gêne pas NVDEC.
  constexpr unsigned kSchedMask = 0x7;
  if ((flags & kSchedMask) == CU_CTX_SCHED_BLOCKING_SYNC) return {};
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

// Tenseur alloué dans le pool de l'appareil, rendu à sa destruction : en file derrière toutes les voies
// tant que le backend vit (sans synchroniser l'appareil), par cudaFree ensuite.
class CudaTensorBuffer final : public TensorBuffer {
 public:
  CudaTensorBuffer(int device, void* memory, Tensor tensor,
                   std::weak_ptr<TensorRtBackend::TensorReleaser> releaser)
      : device_(device), memory_(memory), tensor_(tensor), releaser_(std::move(releaser)) {}
  ~CudaTensorBuffer() override {
    cuda_warn(cudaSetDevice(device_), "cudaSetDevice");
    if (const auto owner = releaser_.lock(); owner && owner->backend != nullptr) {
      owner->backend->free_after_lanes(memory_);
      return;
    }
    cudaFree(memory_);
  }
  CudaTensorBuffer(const CudaTensorBuffer&) = delete;
  CudaTensorBuffer& operator=(const CudaTensorBuffer&) = delete;
  const Tensor& tensor() const override { return tensor_; }

 private:
  int device_;
  void* memory_;
  Tensor tensor_;
  std::weak_ptr<TensorRtBackend::TensorReleaser> releaser_;
};

}  // namespace

std::expected<std::unique_ptr<TensorRtBackend>, Refusal> TensorRtBackend::create(
    const BackendInit& init, std::filesystem::path trtexec) {
  std::unique_ptr<TensorRtBackend> backend(new TensorRtBackend());
  backend->log_ = init.log;
  backend->cache_dir_ = init.cache_dir;
  backend->trtexec_ = std::move(trtexec);
  if (!backend->trtexec_.empty()) {
    // Son empreinte d'ouverture, revérifiée avant chaque construction (A05-05).
    if (auto sha = prepare::sha256_file_known(backend->trtexec_)) backend->trtexec_sha256_ = std::move(*sha);
    if (backend->trtexec_ == std::filesystem::path(AJP_TRTEXEC_PATH)) {
      backend->log(BackendLogLevel::Warn,
                   std::format("trtexec du paquet absent ou d'empreinte inattendue : celui du poste de "
                               "développement ({}) construit les moteurs",
                               core::to_utf8(backend->trtexec_)));
    }
  }
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
  cuda_warn(cudaRuntimeGetVersion(&backend->runtime_version_), "cudaRuntimeGetVersion");
  cuda_warn(cudaDriverGetVersion(&backend->driver_version_), "cudaDriverGetVersion");
  if (auto ok = backend->create_lanes(); !ok) return std::unexpected(ok.error());
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
  backend->role_timing_ = {std::make_unique<GpuTiming>("agrandissement", report),
                           std::make_unique<GpuTiming>("interpolation", report),
                           std::make_unique<GpuTiming>("mesure", nullptr)};
  backend->chain_span_ = std::make_unique<GpuSpan>("source jusqu'à son SPAN");
  backend->trt_log_ = std::make_unique<TrtLog>(init.log);
  backend->runtime_.reset(nvinfer1::createInferRuntime(*backend->trt_log_));
  if (!backend->runtime_)
    return refuse(RefusalCode::LibraryUnavailable, "nvinfer1::createInferRuntime a échoué");
  backend->log(BackendLogLevel::Info,
               std::format("TensorRT {} sur {} (sm {}.{}), {} voie{} de calcul", getInferLibVersion(),
                           backend->properties_.name, backend->properties_.major, backend->properties_.minor,
                           backend->lanes_, backend->lanes_ > 1 ? "s" : ""));
#ifdef _WIN32
  backend->log(BackendLogLevel::Info, hags_state());
#endif
  return backend;
}

TensorRtBackend::~TensorRtBackend() {
  releaser_->backend = nullptr;  // un tenseur détruit après le backend se libère par cudaFree
  // Rien de créé : create_lanes a échoué avant la première file. Une file créée suffit à tout passer en
  // revue (événements et files déjà créés d'une création interrompue au milieu) ; stream_ n'est posé qu'à
  // la fin de create_lanes.
  if (std::ranges::none_of(streams_, [](cudaStream_t lane) { return lane != nullptr; })) return;
  cudaSetDevice(device_);
  destroy_textures();
  if (const int used = std::popcount(used_lanes_); used > 1 && !stages_.empty()) {
    log(BackendLogLevel::Info,
        std::format("profil GPU · {} voies en parallèle : chaque temps court sur la file de son étape, "
                    "recouvrement avec les autres voies compris (--one-lane : coût propre de chaque moteur)",
                    used));
  }
  for (auto& [name, timing] : stages_) {
    if (const std::string line = timing->summary(); !line.empty())
      log(BackendLogLevel::Info, "profil GPU · " + line);
  }
  for (const auto& timing : role_timing_) {
    if (const std::string line = timing ? timing->summary() : std::string(); !line.empty())
      log(BackendLogLevel::Info, "profil GPU · rôle " + line);
  }
  if (chain_span_) {
    if (const std::string line = chain_span_->summary(); !line.empty())
      log(BackendLogLevel::Info, "profil GPU · " + line);
  }
  stages_.clear();
  slots_.clear();
  retired_.clear();
  for (auto& timing : role_timing_) timing.reset();
  chain_span_.reset();
  if (retired_done_ != nullptr) cudaEventDestroy(retired_done_);
  for (const PairRead& read : pair_reads_) {
    if (read.device != nullptr) cudaFree(read.device);
    if (read.host != nullptr) cudaFreeHost(read.host);
    if (read.done != nullptr) cudaEventDestroy(read.done);
  }
  for (const FlowRead& read : flow_reads_) {
    if (read.device != nullptr) cudaFree(read.device);
    if (read.host != nullptr) cudaFreeHost(read.host);
    if (read.done != nullptr) cudaEventDestroy(read.done);
  }
  for (const StatsRead& read : stats_reads_) {
    if (read.device != nullptr) cudaFree(read.device);
    if (read.host != nullptr) cudaFreeHost(read.host);
    if (read.done != nullptr) cudaEventDestroy(read.done);
  }
  if (ingest_rows_ != nullptr) cudaFree(ingest_rows_);
  if (deband_out_ != nullptr) cudaFree(deband_out_);
  uploads_.reset();  // attend ses envois et les ingestions qui les lisent
  release_yuv_tables();
  release_chroma_tables();
  for (const MappedFences& block : mapped_fences_) cudaHostUnregister(block.host);  // pas rendus par l'étage
  mapped_fences_.clear();
  runtime_.reset();
  for (Held& held : held_) {
    held.data.reset();
    if (held.done != nullptr) cudaEventDestroy(held.done);
  }
  for (cudaEvent_t fence : fences_) {
    if (fence != nullptr) cudaEventDestroy(fence);
  }
  // Les zones d'activation, la plus grosse allocation du pool, partent avant sa purge : détruites après le
  // corps du destructeur (membres), elles revenaient au pool déjà purgé et y restaient jusqu'à la fin du
  // processus. Les moteurs qui les tenaient sont déjà détruits (ChainEngines) ; une zone qu'un moteur
  // tiendrait encore se libère seule par cudaFree, sans rappeler ce backend.
  for (std::shared_ptr<SharedActivations>& zone : arenas_) {
    if (!zone) continue;
    zone->set_pool({}, {});
    zone.reset();
  }
  destroy_lanes();  // et rend la réserve CUDA au pilote
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
      .int8 = true,  // graphes Q/DQ (SPAN INT8, span_int8.h) : Turing et après
      .fp8 = fp8_allowed(properties_.major, properties_.minor),  // Ada et Hopper ; RTX 50 sans mesure, non
      .span_fp8_cutlass = SpanFp8Net::supported_device(properties_.major, properties_.minor),
      .gpu_arch = std::format("sm_{}{}", properties_.major, properties_.minor),
      .span_fp8_kernels = SpanFp8Net::kKernelsVersion,
      .rife_fused = true,
      .drba_fused = true,
      .matrix_units = true,
      .lanes = lanes_,
      .ring_targets = true,
      .ring_rgb10a2 = true,
      .yuv_targets = true};
}

DeviceMatch TensorRtBackend::match(const DeviceUuid& render_device) const {
  return render_device == uuid_ ? DeviceMatch::Same : DeviceMatch::Different;
}

prepare::TrtexecConfig TensorRtBackend::trtexec_config() const {
  return prepare::TrtexecConfig{
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
      .trtexec_sha256 = trtexec_sha256_,
  };
}

std::expected<Artifact, Refusal> TensorRtBackend::build(const ModelSpec& model, const ShapeSet& shapes,
                                                        Progress& progress) {
  // Une variante du SPAN en paquet de poids FP8 : pas de trtexec, les noyaux CUTLASS la lisent telle quelle.
  if (is_span_fp8_package(model.onnx_path)) return build_span_fp8(model, shapes);
  // RIFE coupé en six morceaux : six constructions, un seul artefact.
  if (prepare::is_rife_fused_manifest(model.onnx_path)) return build_rife_fused(model, shapes, progress);
  // DistilDRBA coupé en quatre morceaux : quatre constructions, un seul artefact.
  if (prepare::is_drba_fused_manifest(model.onnx_path)) return build_drba_fused(model, shapes, progress);
  // Un moteur publié doit se désérialiser ici : sinon le constructeur le refait.
  const prepare::EngineCheck check =
      [this](const std::filesystem::path& engine) -> std::expected<void, Refusal> {
    if (auto ok = bind(); !ok) return ok;
    auto loaded = TrtEngine::load(*runtime_, engine);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    return {};
  };
  return prepare::TrtexecBuilder(trtexec_config(), check).build(model, shapes, progress);
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load(const Artifact& artifact) {
  if (auto ok = bind(); !ok) return std::unexpected(ok.error());
  if (span_fp8_artifact_shape(artifact.cache_key)) return load_span_fp8(artifact);
  if (rife_fused_artifact_key_parse(artifact.cache_key)) return load_rife_fused(artifact);
  if (drba_fused_artifact_key_parse(artifact.cache_key)) return load_drba_fused(artifact);
  // Les activations dans la zone de la voie choisie avant le chargement (use_lane) ; un moteur
  // lancé sur l'autre voie y passe à son lancement.
  auto engine = TrtEngine::load(*runtime_, artifact.path, arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> TensorRtBackend::alloc_tensor(TensorType type,
                                                                                    const Shape& shape) {
  if (type != TensorType::F16 && type != TensorType::F32) {
    return refuse(RefusalCode::FormatUnsupported, "tenseur : F16 ou F32 seulement");
  }
  // La borne commune à Vulkan (gpu/tensor.h, P2.28).
  const auto bytes = static_cast<std::size_t>(dense_tensor_bytes(type, shape));
  if (bytes == 0)
    return refuse(RefusalCode::ShapeInvalid,
                  "tenseur : dimension hors de [1, 2^24] ou plus de 2^32 éléments");
  if (auto ok = bind(); !ok) return std::unexpected(ok.error());
  void* memory = nullptr;
  // Dans le pool, sur la file courante, puis cette file attendue : le tenseur est aussitôt valable pour
  // toutes les voies. Une allocation se fait à la préparation, rarement pendant la lecture.
  if (auto ok = cuda_check(cudaMallocAsync(&memory, bytes, stream_), "cudaMallocAsync"); !ok)
    return std::unexpected(ok.error());
  if (auto ok = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); !ok) {
    cudaFree(memory);
    return std::unexpected(ok.error());
  }
  const Tensor tensor{.type = type, .shape = shape, .handle = reinterpret_cast<std::uintptr_t>(memory)};
  return std::make_unique<CudaTensorBuffer>(device_, memory, tensor, releaser_);
}

void TensorRtBackend::free_after_lanes(void* memory) {
  if (memory == nullptr) return;
  hazards_.forget(reinterpret_cast<std::uintptr_t>(memory));  // libérée derrière toutes les voies (A21-09)
  // Une allocation future à la même adresse n'hérite pas du drapeau compact de RIFE (B03-04).
  forget_rife_input(reinterpret_cast<std::uintptr_t>(memory));
  cudaStream_t free_on = stream_;
  bool ordered = free_on != nullptr;
  for (cudaStream_t lane : streams_) {
    if (!ordered) break;
    if (lane == nullptr || lane == free_on) continue;
    cudaEvent_t mark = nullptr;
    ordered = cudaEventCreateWithFlags(&mark, cudaEventDisableTiming) == cudaSuccess &&
              cudaEventRecord(mark, lane) == cudaSuccess &&
              cudaStreamWaitEvent(free_on, mark, 0) == cudaSuccess;
    if (mark != nullptr) cudaEventDestroy(mark);  // détruit une fois atteint, l'attente reste posée
  }
  if (ordered && cudaFreeAsync(memory, free_on) == cudaSuccess) return;
  static_cast<void>(clear_error());
  cuda_warn(cudaFree(memory), "cudaFree");  // repli : la libération synchrone d'avant
}

void TensorRtBackend::tune_pool() const {
  cudaMemPool_t pool = nullptr;
  if (cudaDeviceGetDefaultMemPool(&pool, device_) != cudaSuccess) {
    static_cast<void>(clear_error());
    return;
  }
  std::uint64_t keep = UINT64_MAX;
  int internal = 0;
  if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep) != cudaSuccess ||
      cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies, &internal) != cudaSuccess)
    static_cast<void>(clear_error());
}

GpuTiming& TensorRtBackend::stage(std::string_view name) {
  push_launches();
  auto found = stages_.find(name);
  if (found == stages_.end())
    found = stages_.emplace(std::string(name), std::make_unique<GpuTiming>(std::string(name), nullptr)).first;
  return *found->second;
}

std::expected<void, Refusal> TensorRtBackend::enqueue(Engine& engine, std::span<const Tensor> inputs,
                                                      std::span<Tensor> outputs) {
  auto* trt = dynamic_cast<TrtEngine*>(&engine);
  // Étape 5.2 : qui lit l'entrée de RIFE en dernier décide de ce que rife_time y écrit. Chaque entrée : la
  // deuxième de DistilDRBA (pair12) efface aussi un drapeau posé sur son adresse (B03-04).
  const bool fused_rife = dynamic_cast<RifeFusedEngine*>(&engine) != nullptr;
  for (const Tensor& input : inputs) note_rife_reader(input, fused_rife);
  if (trt == nullptr) {
    if (auto* rife = dynamic_cast<RifeFusedEngine*>(&engine))
      return enqueue_rife_fused(*rife, inputs, outputs);
    if (auto* drba = dynamic_cast<DrbaFusedEngine*>(&engine))
      return enqueue_drba_fused(*drba, inputs, outputs);
    auto* fused = dynamic_cast<SpanFp8Engine*>(&engine);
    if (fused == nullptr) return refuse(RefusalCode::DeviceMismatch, "enqueue : moteur d'un autre backend");
    return enqueue_span_fp8(*fused, inputs, outputs);
  }
  if (auto ok = bind(); !ok) return ok;
  // Le moteur lit ses entrées, écrit ses sorties, sa zone d'activation et son contexte.
  const std::shared_ptr<SharedActivations>& zone = arena();
  std::vector<std::uintptr_t> reads;
  std::vector<std::uintptr_t> writes{reinterpret_cast<std::uintptr_t>(zone.get()),
                                     reinterpret_cast<std::uintptr_t>(trt)};
  for (const Tensor& tensor : inputs) reads.push_back(tensor.handle);
  for (const Tensor& tensor : outputs) writes.push_back(tensor.handle);
  if (auto ok = order(reads, writes); !ok) return ok;
  // Un moteur entier lit le plan de l'instant en entier : reposé s'il n'a été écrit qu'à l'élément 0.
  if (auto ok = restore_rife_times(inputs); !ok) return ok;
  const Shape shape = trt->input_shape(0);
  GpuTiming& timing = stage(std::format("moteur {}×{}×{}", shape.dims[1], shape.dims[2], shape.dims[3]));
  // Un moteur qui agrandit (sortie plus haute que l'entrée) est le SPAN de sa source ; les autres
  // interpolent.
  const bool upscales =
      !inputs.empty() && !outputs.empty() && outputs[0].shape.dims[2] > inputs[0].shape.dims[2];
  GpuTiming& by_role = role(upscales ? Role::Upscale : Role::Interpolation);
  by_role.begin(stream_);
  timing.begin(stream_);
  auto queued = trt->enqueue(stream_, zone, inputs, outputs);
  timing.end(stream_);
  by_role.end(stream_);
  if (upscales) chain_span_->finish(inputs[0].handle, stream_);
  if (std::string notice = trt->take_notice(); !notice.empty()) {
    log(BackendLogLevel::Warn,
        std::format("moteur {}×{}×{} : {}", shape.dims[1], shape.dims[2], shape.dims[3], notice));
  }
  return queued;
}

std::expected<Fence, Refusal> TensorRtBackend::signal() {
  cudaEvent_t event = fences_[fence_count_ % kFences];
  ++fence_count_;
  // La barrière suit le travail de toutes les voies.
  if (auto ok = bind().and_then([&] { return join(); }).and_then([&] {
        return cuda_check(cudaEventRecord(event, stream_), "cudaEventRecord");
      });
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  return Fence{.handle = reinterpret_cast<std::uintptr_t>(event), .value = fence_count_};
}

std::expected<void, Refusal> TensorRtBackend::wait(const Fence& fence) {
  auto* event = reinterpret_cast<cudaEvent_t>(fence.handle);  // NOLINT(performance-no-int-to-ptr)
  if (auto ok = bind(); !ok) return ok;
  // Toutes les voies attendent : le travail enfilé après, sur n'importe laquelle, suit la barrière.
  for (std::size_t lane = 0; lane < static_cast<std::size_t>(lanes_); ++lane) {
    const cudaStream_t stream = lanes_ > 1 ? streams_[lane] : stream_;
    if (auto ok = cuda_check(cudaStreamWaitEvent(stream, event, 0), "cudaStreamWaitEvent"); !ok) return ok;
  }
  return {};
}

}  // namespace ajp::gpu::cuda

namespace ajp::gpu::cuda {

std::vector<GpuStageSummary> TensorRtBackend::gpu_profile() {
  std::vector<GpuStageSummary> out;
  out.reserve(stages_.size() + role_timing_.size());
  for (auto& [name, timing] : stages_) out.push_back(timing->profile());
  for (const auto& timing : role_timing_) {
    if (!timing) continue;
    GpuStageSummary s = timing->profile();
    s.name = "rôle " + s.name;
    out.push_back(std::move(s));
  }
  if (chain_span_) out.push_back(chain_span_->profile());
  std::erase_if(out, [](const GpuStageSummary& s) { return s.count == 0; });
  return out;
}

void TensorRtBackend::push_launches() {
#ifdef _WIN32
  // WDDM garde les lancements dans un lot logiciel jusqu'au prochain appel qui force l'envoi : chaque
  // passage d'une étape à la suivante pousse le lot, pas seulement la fin de l'image. L'état « pas
  // prêt » est la réponse normale ; une vraie erreur reparaîtra au prochain appel CUDA.
  if (stream_ != nullptr) static_cast<void>(cudaStreamQuery(stream_));
#endif
}

}  // namespace ajp::gpu::cuda
