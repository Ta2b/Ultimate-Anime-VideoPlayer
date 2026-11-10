#include "gpu/vulkan/backend/vulkan_backend.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <limits>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/utf8_path.h"
#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/drba_fused_engine.h"
#include "gpu/vulkan/backend/host_tensor.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_gate.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/backend/rife_fused_engine.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/buffer.h"
#include "prepare/drba_split.h"
#include "prepare/rife_split.h"

namespace ajp::gpu::vk {
namespace detail {

std::unexpected<Refusal> refuse_backend(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

std::size_t element_bytes(TensorType type) {
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

std::expected<std::uint64_t, Refusal> tensor_bytes(TensorType type, const Shape& shape) {
  if (tensor_element_bytes(type) == 0)
    return refuse_backend(RefusalCode::FormatUnsupported, "type de tenseur inconnu");
  const std::uint64_t bytes = dense_tensor_bytes(type, shape);  // la borne commune (gpu/tensor.h)
  if (bytes == 0)
    return refuse_backend(
        RefusalCode::ShapeInvalid,
        std::format("forme {}×{}×{}×{} : dimension hors de [1, 2^24] ou plus de 2^32 éléments", shape.dims[0],
                    shape.dims[1], shape.dims[2], shape.dims[3]));
  return bytes;
}

std::string file_stem(std::string_view id) {
  std::string out;
  for (const char c : id) {
    const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                      c == '-' || c == '_' || c == '.';
    out.push_back(safe ? c : '_');
  }
  return out.empty() ? std::string{"model"} : out;
}

}  // namespace detail

namespace {

using detail::file_stem;
using detail::refuse_backend;

std::uintptr_t handle_value(VkSemaphore semaphore) {
  return static_cast<std::uintptr_t>(detail::to_bits(semaphore));
}

// Tenseur alloué par alloc_tensor : son tampon, libéré à la destruction, une fois fini ce que la
// file a reçu. Un étage qui s'arrête rend ses tenseurs alors que ses dernières soumissions
// tournent encore ; un bloc vide retourne aussitôt à l'appareil (MemoryAllocator::release), et le
// GPU lirait une mémoire rendue.
class VulkanTensorBuffer final : public TensorBuffer {
 public:
  VulkanTensorBuffer(VkContext& context, Buffer buffer, TensorType type, const Shape& shape)
      : context_(context),
        buffer_(std::move(buffer)),
        tensor_{.type = type, .shape = shape, .handle = detail::handle_bits(buffer_.handle()), .offset = 0} {}
  VulkanTensorBuffer(const VulkanTensorBuffer&) = delete;
  VulkanTensorBuffer& operator=(const VulkanTensorBuffer&) = delete;
  VulkanTensorBuffer(VulkanTensorBuffer&&) = delete;
  VulkanTensorBuffer& operator=(VulkanTensorBuffer&&) = delete;
  // GPU bloqué : le tampon et sa mémoire restent en place (Buffer et MemoryAllocator le voient).
  ~VulkanTensorBuffer() override { context_.settle(); }
  const Tensor& tensor() const override { return tensor_; }

 private:
  VkContext& context_;
  Buffer buffer_;
  Tensor tensor_;
};

std::expected<net::BuildRequest, Refusal> request_of(const ModelSpec& model, const ShapeSet& shapes) {
  if (model.precision != TensorType::F16 && model.precision != TensorType::F32) {
    return refuse_backend(RefusalCode::FormatUnsupported,
                          std::format("{} : précision entière non prise en charge", model.id));
  }
  return net::BuildRequest{.onnx = model.onnx_path,
                           .model_id = model.id,
                           .model_sha256 = model.sha256,
                           .input_shape = {shapes.opt.dims.begin(), shapes.opt.dims.end()},
                           .fp16 = model.precision == TensorType::F16};
}

}  // namespace

std::expected<AdoptedDevice, Refusal> adopted_device(const GpuDevice& render) {
  if (render.api != GpuApi::Vulkan || !render.vulkan) {
    return refuse_backend(RefusalCode::DeviceMismatch, "appareil de rendu sans description Vulkan");
  }
  const VulkanDevice& v = *render.vulkan;
  AdoptedDevice adopted{
      .get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(v.get_instance_proc_addr),
      .instance = static_cast<VkInstance>(v.instance),
      .physical_device = static_cast<VkPhysicalDevice>(v.physical_device),
      .device = static_cast<VkDevice>(v.device),
      .api_version = v.api_version,
      .queue_family = v.compute.family,
      .queue_index = v.compute.index,
      .features = static_cast<const VkPhysicalDeviceFeatures2*>(v.enabled_features),
      .extensions = v.device_extensions};
  if (v.lock_queue != nullptr && v.unlock_queue != nullptr) {
    adopted.lock_queue = [lock = v.lock_queue, context = v.queue_context, q = v.compute] {
      lock(context, q.family, q.index);
    };
    adopted.unlock_queue = [unlock = v.unlock_queue, context = v.queue_context, q = v.compute] {
      unlock(context, q.family, q.index);
    };
  }
  // Les voies en plus (V7) : les files libres qui suivent celle du backend, que le créateur de
  // l'appareil garantit (compute_lanes) ; aucune sinon, une file non créée n'existe pas.
  for (std::uint32_t lane = 1; lane <= std::min(v.compute_lanes, 2U); ++lane)
    adopted.lane_queue_indices.push_back(v.compute.index + lane);
  return adopted;
}

VulkanBackend::VulkanBackend(const BackendInit& init, BackendOptions options)
    : log_(init.log), cache_dir_(init.cache_dir), options_(std::move(options)) {
  // Le lecteur le demande par BackendInit (--profile-dispatches), les tests par BackendOptions (A45-08).
  options_.profile_dispatches = options_.profile_dispatches || init.profile_dispatches;
  // Les temporaires d'une écriture coupée (lecteur tué pendant une construction) ne s'accumulent pas.
  if (cache_dir_.empty()) return;
  if (const int purged = net::purge_stale_temporaries(cache_dir_ / "vulkan"); purged > 0)
    log(BackendLogLevel::Info, std::format("cache Vulkan : {} fichiers temporaires anciens retirés", purged));
}

VulkanBackend::~VulkanBackend() {
  kernels_.reset();                                           // attend la file, rend images et vues
  if (queue_ && context_) context_->settle(queue_->lasts());  // la dernière soumission de chaque voie
  if (pipelines_ && !pipeline_cache_file_.empty()) {
    const std::vector<std::byte> bytes = seal_pipeline_cache(pipelines_->serialize());
    // Un cache perdu se refait.
    if (!bytes.empty() && !net::write_file(pipeline_cache_file_, bytes))
      log(BackendLogLevel::Warn, "cache de pipelines non écrit");
  }
}

std::expected<std::unique_ptr<VulkanBackend>, Refusal> VulkanBackend::create(const BackendInit& init,
                                                                             BackendOptions options) {
  std::unique_ptr<VulkanBackend> backend(new VulkanBackend(init, std::move(options)));
  if (auto ok = backend->open(init); !ok) return std::unexpected(ok.error());
  return backend;
}

std::expected<void, Refusal> VulkanBackend::open(const BackendInit& init) {
  if (init.render != nullptr) {
    auto adopted = adopted_device(*init.render);
    if (!adopted) return std::unexpected(adopted.error());
    auto context = VkContext::adopt(std::move(*adopted));
    if (!context) return std::unexpected(context.error());
    context_ = std::move(*context);
  } else {
    auto context = VkContext::create({.validation = options_.validation,
                                      .validation_sink = options_.validation_sink,
                                      .cooperative_matrix = true,
                                      .alias_lanes = options_.alias_lanes});
    if (!context) return std::unexpected(context.error());
    context_ = std::move(*context);
  }
  auto memory = MemoryAllocator::create(*context_, device_limits(*context_));
  if (!memory) return std::unexpected(memory.error());
  memory_ = std::move(*memory);
  // Le cache de pipelines du pilote revient d'un lancement à l'autre : le pilote écarte lui-même
  // des octets qui ne sont pas les siens (autre appareil ou autre version).
  if (!cache_dir_.empty()) {
    const DeviceCaps& caps = context_->caps();
    pipeline_cache_file_ =
        cache_dir_ / "vulkan" / std::format("pipelines-{:04x}-{:04x}.bin", caps.vendor_id, caps.device_id);
  }
  // Un fichier abîmé (tronqué, autre appareil, somme fausse) ne passe plus au pilote : cache vide, fichier
  // effacé (P2.17). Certains pilotes refusaient de démarrer sur ces octets.
  std::vector<std::byte> saved;
  if (!pipeline_cache_file_.empty()) {
    if (auto bytes = net::read_file(pipeline_cache_file_)) {
      const DeviceCaps& caps = context_->caps();
      const std::span<const std::byte> driver = open_pipeline_cache(*bytes, caps.vendor_id, caps.device_id);
      if (driver.empty()) {
        log(BackendLogLevel::Warn, "cache de pipelines illisible : effacé");
        std::error_code removed;
        std::filesystem::remove(pipeline_cache_file_, removed);
      } else {
        saved.assign(driver.begin(), driver.end());
      }
    }
  }
  auto pipelines = PipelineCache::create(*context_, saved);
  if (!pipelines) return std::unexpected(pipelines.error());
  pipelines_ = std::move(*pipelines);
  if (options_.runtime) {
    auto runtime = options_.runtime(context_->caps());
    if (!runtime) return std::unexpected(runtime.error());
    runtime_ = std::move(*runtime);
    host_views_ = std::make_shared<HostViews>();
    // GPU intégré seulement : sur une carte, la mémoire hôte se lit par le bus, et le runtime
    // copie mieux vers la mémoire de la carte que ses noyaux ne la liraient de loin.
    if (context_->caps().type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      host_alignment_ = host_import_alignment(*context_);
  }
  // Plusieurs voies sans runtime seulement : ses étapes s'ordonnent sur la timeline principale.
  auto queue = BackendQueue::create(*context_, !runtime_);
  if (!queue) return std::unexpected(queue.error());
  BackendQueue& made = queue_.emplace(std::move(*queue));
  profile_ = profile_device(*context_);
  kernels_ = std::make_unique<VulkanKernelOps>(*context_, *memory_, *pipelines_, made);
  if (runtime_) {
    log(BackendLogLevel::Info,
        std::format("réseaux par {} {} sur {}{}", runtime_->name(), runtime_->version(),
                    runtime_->device_name(),
                    host_alignment_ != 0 ? ", tenseurs dans la mémoire hôte du runtime, importée"
                                         : ", tenseurs par l'hôte"));
  }
  log(BackendLogLevel::Info,
      std::format(
          "backend Vulkan : {}, famille {}, unités matricielles {}{}{}", context_->caps().device_name,
          name(profile_.family), profile_.target.matrix_units ? "utilisées" : "non utilisées",
          !profile_.matrix_hardware ? ""
          : !profile_.matrix_allowed
              ? std::format(" (présentes, exclues : {})",
                            matrix_units_excluded(profile_.family, profile_.driver).value_or("?"))
          : profile_.matrix_unlisted ? " (famille inconnue : vérifiées par l'autotuner, jamais par défaut)"
                                     : "",
          profile_.matrix_shapes.empty() ? ""
                                         : std::format(" ; tuiles du pilote : {}", profile_.matrix_shapes)));
  return {};
}

void VulkanBackend::log(BackendLogLevel level, std::string_view message) const {
  if (log_) log_(level, message);
}

BackendCaps VulkanBackend::capabilities() const {
  const DeviceCaps& caps = context_->caps();
  if (runtime_) {
    // Le réseau passe par le runtime, attendu sur l'hôte ; la chaîne reste sur Vulkan.
    return {.name = std::string(runtime_->name()),
            .device_name = std::format("{} (chaîne Vulkan : {})", runtime_->device_name(), caps.device_name),
            .driver_version = runtime_->version(),
            .needs_host_sync = true,
            .fp16 = true,
            .int8 = false,
            .matrix_units = false};
  }
  return {.name = "vulkan",
          .device_name = caps.device_name,
          .driver_version = caps.driver_info,
          .needs_host_sync = false,
          .fp16 = profile_.target.f16_storage,
          .int8 = false,
          // Les noyaux de colle lisent et écrivent des tenseurs FP16.
          .rife_fused = profile_.target.f16_storage,
          .drba_fused = profile_.target.f16_storage,
          .matrix_units = profile_.target.matrix_units,
          .lanes = queue_ ? queue_->lanes() : 1};
}

DeviceMatch VulkanBackend::match(const DeviceUuid& render_device) const {
  return render_device == context_->uuid() ? DeviceMatch::Same : DeviceMatch::Different;
}

std::filesystem::path VulkanBackend::artifact_path(const ModelSpec& model, const ShapeSet& shapes) const {
  auto request = request_of(model, shapes);
  const std::uint64_t key = request ? net::artifact_key(*request, profile_) : 0;
  return cache_dir_ / "vulkan" / std::format("{}-{:016x}.ajpvk", file_stem(model.id), key);
}

std::expected<Artifact, Refusal> VulkanBackend::build(const ModelSpec& model, const ShapeSet& shapes,
                                                      Progress& progress) {
  if (cache_dir_.empty())
    return refuse_backend(RefusalCode::CacheIo, "aucun dossier de cache pour les artefacts");
  if (progress.stop.stop_requested()) return refuse_backend(RefusalCode::Cancelled, "construction annulée");
  if (runtime_) return build_runtime(model, shapes, progress);
  if (prepare::is_rife_fused_manifest(model.onnx_path)) return build_rife_fused(model, shapes, progress);
  if (prepare::is_drba_fused_manifest(model.onnx_path)) return build_drba_fused(model, shapes, progress);
  auto request = request_of(model, shapes);
  if (!request) return std::unexpected(request.error());
  const std::uint64_t key = net::artifact_key(*request, profile_);
  Artifact artifact{.cache_key = std::format("{:016x}", key), .path = artifact_path(model, shapes)};
  // Un artefact déjà construit et toujours valable pour cet appareil et ces shaders sert tel quel.
  // Rien ne s'abaisse ici : load le fait, une fois, pour ranger les poids.
  if (auto bytes = net::read_file(artifact.path)) {
    if (auto data = net::deserialize(*bytes); data && net::check_artifact(*data, profile_)) return artifact;
  }
  const auto started = std::chrono::steady_clock::now();  // A45-12 : la durée se lit au journal
  std::optional<net::Autotuner> tuner;
  if (options_.autotune) {
    tuner.emplace(*context_, *memory_, *pipelines_, profile_, cache_dir_ / "vulkan" / "autotune.tsv",
                  options_.tuner);
  }
  auto data =
      net::build_network(*request, profile_, tuner ? &*tuner : nullptr, progress.report, progress.stop);
  if (!data) return std::unexpected(data.error());
  if (tuner) {
    for (const std::string& line : tuner->decisions()) log(BackendLogLevel::Debug, model.id + " : " + line);
    if (auto ok = gate(model, *data); !ok) return std::unexpected(ok.error());
  }
  if (auto ok = net::write_file(artifact.path, net::serialize(*data)); !ok)
    return std::unexpected(ok.error());
  log(BackendLogLevel::Info,
      std::format("{} construit en {:.1f} s : {} nœuds, {} dispatchs{}", model.id,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                  data->graph.nodes.size(), data->dispatches.size(),
                  tuner ? std::format(", {} couches mesurées, {} variantes écartées (sortie discordante), "
                                      "{} dès leur sondage, {} choix revérifiés sous ce pilote ({}, {})",
                                      tuner->measured(), tuner->rejected(), tuner->pruned(),
                                      tuner->reverified(), name(profile_.family), profile_.driver_build)
                        : ""));
  return artifact;
}

std::expected<void, Refusal> VulkanBackend::gate(const ModelSpec& model, net::ArtifactData& data) {
  // Les noyaux de référence partout, quand les choix de l'autotuner ne peuvent pas être vérifiés.
  const auto to_reference = [&](std::string_view why) -> std::expected<void, Refusal> {
    auto reference = net::reference_choices(data.graph, profile_.target);
    if (!reference) return std::unexpected(reference.error());
    auto program = net::lower(data.graph, *reference, profile_.target);
    if (!program) return std::unexpected(program.error());
    log(BackendLogLevel::Warn, std::format("{} : {}, noyaux de référence partout", model.id, why));
    data.choices = std::move(*reference);
    data.dispatches = std::move(program->dispatches);
    data.shaders = net::shader_hashes(data.dispatches);
    return {};
  };
  auto gated = net::gate_network(*context_, *memory_, *pipelines_, data.graph, data.choices, profile_.target);
  if (!gated) return to_reference(std::format("porte de bout en bout non jouée ({})", gated.error().detail));
  if (!gated->unjudged.empty()) {
    log(BackendLogLevel::Warn,
        std::format("{} : porte jugée sans le calcul en float ({}), contre les noyaux de "
                    "référence",
                    model.id, gated->unjudged));
  }
  if (gated->fallback.empty()) {
    log(BackendLogLevel::Info,
        gated->against_float
            ? std::format(
                  "{} : porte de bout en bout passée ({:.1f} dB contre le calcul en float, {:.1f} dB aux "
                  "noyaux de référence)",
                  model.id, gated->db, gated->floor_db)
            : std::format("{} : porte de bout en bout passée (écart {:.4f})", model.id, gated->drift));
    return {};
  }
  auto program = net::lower(data.graph, gated->choices, profile_.target);
  if (!program) return to_reference(std::format("repli de la porte refusé ({})", program.error().detail));
  log(BackendLogLevel::Warn,
      std::format("{} : sortie du réseau trop loin de la référence, {}", model.id, gated->fallback));
  data.choices = std::move(gated->choices);
  data.dispatches = std::move(program->dispatches);
  data.shaders = net::shader_hashes(data.dispatches);
  return {};
}

std::expected<std::unique_ptr<Engine>, Refusal> VulkanBackend::load(const Artifact& artifact) {
  if (runtime_) return load_runtime(artifact);
  if (rife_fused_artifact_key_parse(artifact.cache_key)) return load_rife_fused(artifact);
  if (drba_fused_artifact_key_parse(artifact.cache_key)) return load_drba_fused(artifact);
  auto bytes = net::read_file(artifact.path);
  if (!bytes) return std::unexpected(bytes.error());
  auto data = net::deserialize(*bytes);
  if (!data) return std::unexpected(data.error());
  if (auto ok = net::check_artifact(*data, profile_); !ok) return std::unexpected(ok.error());
  auto program = net::lower(data->graph, data->choices, profile_.target);
  if (!program) return std::unexpected(program.error());
  // Mêmes graphe, choix, appareil et shaders, mais un abaissement qui a changé sans changer de
  // version : le programme d'aujourd'hui sert, et l'artefact est réécrit pour lui.
  if (std::vector<net::ShaderHash> shaders = net::shader_hashes(program->dispatches);
      program->dispatches != data->dispatches || shaders != data->shaders) {
    log(BackendLogLevel::Warn, std::format("{} : programme refait autrement qu'à sa construction, artefact "
                                           "réécrit",
                                           artifact.path.filename().string()));
    data->dispatches = program->dispatches;
    data->shaders = std::move(shaders);
    // Un programme que la porte n'a jamais vu ne sert pas sans elle (A45-13) : elle rejoue comme à la
    // construction, et son repli éventuel (noyaux de référence) donne le programme chargé.
    if (options_.autotune) {
      if (auto ok = gate(ModelSpec{.id = core::to_utf8(artifact.path.stem())}, *data); !ok)
        return std::unexpected(ok.error());
      if (data->dispatches != program->dispatches) {
        program = net::lower(data->graph, data->choices, profile_.target);
        if (!program) return std::unexpected(program.error());
      }
    }
    if (auto ok = net::write_file(artifact.path, net::serialize(*data)); !ok)
      log(BackendLogLevel::Warn, ok.error().detail);
  }
  net::NetEngineOptions engine_options{.profile_dispatches = options_.profile_dispatches};
  // Le journal est copié, pas le backend : le moteur écrit son tableau à sa destruction.
  if (options_.profile_dispatches && log_) {
    engine_options.log = [log = log_](std::string_view text) { log(BackendLogLevel::Info, text); };
  }
  auto engine = net::NetEngine::create(*context_, *memory_, *pipelines_, std::move(data->graph),
                                       std::move(*program), std::move(engine_options));
  if (!engine) return std::unexpected(engine.error());
  return std::make_unique<VulkanEngine>(std::move(*engine));
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> VulkanBackend::alloc_tensor(TensorType type,
                                                                                  const Shape& shape) {
  if (runtime_) return alloc_runtime_tensor(type, shape);
  auto dense = detail::tensor_bytes(type, shape);
  if (!dense) return std::unexpected(dense.error());
  // Arrondi à 4 octets : les noyaux lisent et écrivent des mots de 32 bits.
  const std::uint64_t bytes = (*dense + 3) / 4 * 4;
  auto buffer = Buffer::create(*context_, *memory_, bytes, MemoryUse::Device);
  if (!buffer) return std::unexpected(buffer.error());
  return std::make_unique<VulkanTensorBuffer>(*context_, std::move(*buffer), type, shape);
}

std::expected<void, Refusal> VulkanBackend::enqueue(Engine& engine, std::span<const Tensor> inputs,
                                                    std::span<Tensor> outputs) {
  if (runtime_) return enqueue_runtime(engine, inputs, outputs);
  if (auto* rife = dynamic_cast<RifeFusedEngine*>(&engine)) {
    if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
    return rife->enqueue(*queue_, inputs, outputs);
  }
  if (auto* drba = dynamic_cast<DrbaFusedEngine*>(&engine)) {
    if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
    return drba->enqueue(*queue_, inputs, outputs);
  }
  auto* vulkan = dynamic_cast<VulkanEngine*>(&engine);
  if (vulkan == nullptr) return refuse_backend(RefusalCode::InputMismatch, "moteur d'un autre backend");
  net::NetEngine& net = vulkan->net();
  const ir::Graph& graph = net.graph();
  if (inputs.size() != graph.inputs.size() || outputs.size() != graph.outputs.size()) {
    return refuse_backend(RefusalCode::InputMismatch,
                          std::format("{} entrées et {} sorties, {} et {} attendues", inputs.size(),
                                      outputs.size(), graph.inputs.size(), graph.outputs.size()));
  }
  // Chaque tenseur a le type et la forme NCHW de son entrée ou de sa sortie dans le graphe.
  const std::span<BufferView> views = vulkan->io_views();
  std::size_t at = 0;
  const auto check = [&](const Tensor& t, ir::TensorId id) -> std::expected<void, Refusal> {
    const ir::Tensor& want = graph.tensors[id];
    const TensorType type = want.type == ir::DataType::F16 ? TensorType::F16 : TensorType::F32;
    const bool shape_ok = want.dims.size() == 4 && std::ranges::equal(want.dims, t.shape.dims);
    if (t.type != type || !shape_ok || t.handle == 0) {
      return refuse_backend(RefusalCode::InputMismatch,
                            std::format("{} : tenseur {} {}, {} {} attendu", want.name,
                                        t.type == TensorType::F16 ? "F16" : "F32",
                                        ir::format_dims(ir::Dims(t.shape.dims.begin(), t.shape.dims.end())),
                                        type == TensorType::F16 ? "F16" : "F32", ir::format_dims(want.dims)));
    }
    views[at++] = {
        .buffer = detail::to_buffer(t.handle), .offset = t.offset, .size = ir::storage_bytes(want)};
    return {};
  };
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    if (auto ok = check(inputs[k], graph.inputs[k]); !ok) return ok;
  }
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    if (auto ok = check(outputs[k], graph.outputs[k]); !ok) return ok;
  }
  auto recording = net.recording(views.first(inputs.size()), views.subspan(inputs.size(), outputs.size()));
  if (!recording) return std::unexpected(recording.error());
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  return queue_->submit(**recording);
}

std::expected<Fence, Refusal> VulkanBackend::signal() {
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  // Toutes les voies jointes : la barrière suit tout ce qui a été soumis.
  auto last = queue_->joined();
  if (!last) return std::unexpected(last.error());
  if (auto ok = queue_->flush(); !ok) return std::unexpected(ok.error());  // attendu ailleurs
  return Fence{.handle = handle_value(last->semaphore), .value = last->value};
}

std::expected<void, Refusal> VulkanBackend::wait(const Fence& fence) {
  if (fence.handle == 0) return {};
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  queue_->wait({.semaphore = detail::to_semaphore(fence.handle), .value = fence.value});
  return {};
}

}  // namespace ajp::gpu::vk
