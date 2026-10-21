// Le backend Vulkan dont les réseaux passent par un autre runtime (net_runtime.h) : construction
// et chargement par le runtime, tenseurs en mémoire hôte visible (celle du runtime, importée, sur
// un GPU intégré ; ou partagée, en interop), enqueue mis en file sur la timeline importée par le
// runtime (interop : ni attente ni retour à l'hôte), sinon différé au fil du runtime
// (run_deferred : l'attente des entrées et le signal de l'étape hors du fil du GPU), sinon
// synchrone sur ce fil après l'attente de la file.
#include <format>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/shared_tensor.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/buffer.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace ajp::gpu::vk {
namespace {

using detail::refuse_backend;

class RuntimeEngine final : public Engine {
 public:
  explicit RuntimeEngine(std::unique_ptr<net::RuntimeNet> net, net::IoNames names)
      : net_(std::move(net)), names_(std::move(names)) {}
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }
  net::RuntimeNet& net() { return *net_; }

 private:
  std::unique_ptr<net::RuntimeNet> net_;
  net::IoNames names_;
};

// Tampon hôte visible, inscrit dans les projections du backend tant qu'il vit.
class HostTensorBuffer final : public TensorBuffer {
 public:
  HostTensorBuffer(VkContext& context, Buffer buffer, TensorType type, const Shape& shape,
                   std::shared_ptr<HostViews> views)
      : context_(context),
        buffer_(std::move(buffer)),
        tensor_{.type = type, .shape = shape, .handle = detail::handle_bits(buffer_.handle()), .offset = 0},
        views_(std::move(views)) {
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.data = buffer_.mapped(), .bytes = buffer_.size()};
  }
  HostTensorBuffer(const HostTensorBuffer&) = delete;
  HostTensorBuffer& operator=(const HostTensorBuffer&) = delete;
  HostTensorBuffer(HostTensorBuffer&&) = delete;
  HostTensorBuffer& operator=(HostTensorBuffer&&) = delete;
  // Ce que la file a reçu finit avant la libération (VulkanTensorBuffer).
  ~HostTensorBuffer() override {
    {
      const std::scoped_lock lock(views_->mutex);
      views_->views.erase(tensor_.handle);
    }
    context_.settle();
  }
  const Tensor& tensor() const override { return tensor_; }

 private:
  VkContext& context_;
  Buffer buffer_;
  Tensor tensor_;
  std::shared_ptr<HostViews> views_;
};

}  // namespace

std::expected<Artifact, Refusal> VulkanBackend::build_runtime(const ModelSpec& model, const ShapeSet& shapes,
                                                              Progress& progress) {
  if (model.precision != TensorType::F16 && model.precision != TensorType::F32) {
    return refuse_backend(RefusalCode::FormatUnsupported,
                          std::format("{} : précision entière non prise en charge", model.id));
  }
  auto names = net::onnx_io_names(model.onnx_path);
  if (!names) return std::unexpected(names.error());
  // Une entrée : la forme demandée ; plusieurs : celles du fichier, statiques (derived_model).
  std::vector<net::InputShape> inputs;
  if (names->inputs.size() == 1)
    inputs.push_back({.name = names->inputs[0], .dims = {shapes.opt.dims.begin(), shapes.opt.dims.end()}});
  std::string key_text = std::format("{}|{}|{}|{}|{}|{}", model.id, model.sha256,
                                     model.precision == TensorType::F16 ? "f16" : "f32", runtime_->version(),
                                     runtime_->device_name(), runtime_->name());
  if (shared_memory_) key_text += "|interop";  // programme compilé pour la mémoire de l'appareil
  for (const auto& input : inputs) {
    key_text += '|' + input.name;
    for (const std::int64_t d : input.dims) key_text += std::format(",{}", d);
  }
  const std::uint64_t key = net::fnv1a(std::as_bytes(std::span(key_text)));
  Artifact artifact{
      .cache_key = std::format("{:016x}", key),
      .path = cache_dir_ / std::string(runtime_->name()) /
              std::format("{}-{:016x}{}", detail::file_stem(model.id), key, runtime_->extension())};
  // Un artefact déjà compilé, avec ses noms, sert tel quel : la clé couvre runtime et appareil.
  std::error_code ec;
  if (std::filesystem::exists(artifact.path, ec) && net::read_io_names(artifact.path)) return artifact;
  // Construit dans un dossier de travail propre à cet appel, puis publié fichier par fichier
  // (net::publish_artifact) : un arrêt en cours de route, ou le lecteur et un export qui
  // compilent le même réseau, ne laissent jamais un artefact tronqué sous le nom final.
  std::random_device random;
  const std::filesystem::path work =
      artifact.path.parent_path() / std::format(".construction-{}-{}", random(), random());
  std::filesystem::create_directories(work, ec);
  if (ec) return refuse_backend(RefusalCode::CacheIo, std::format("{} : {}", work.string(), ec.message()));
  const std::filesystem::path staged = work / artifact.path.filename();
  auto built = runtime_->build(model, inputs, staged, progress);
  if (built) built = net::write_io_names(staged, names->inputs, names->outputs);
  if (built) built = net::publish_artifact(staged, artifact.path);
  std::filesystem::remove_all(work, ec);
  if (!built) return std::unexpected(built.error());
  log(BackendLogLevel::Info, std::format("{} compilé par {}", model.id, runtime_->name()));
  return artifact;
}

std::expected<std::unique_ptr<Engine>, Refusal> VulkanBackend::load_runtime(const Artifact& artifact) {
  auto names = net::read_io_names(artifact.path);
  if (!names) return std::unexpected(names.error());
  auto net = runtime_->load(artifact.path);
  if (!net) {
    // Un artefact que le runtime ne relit plus (disque abîmé, pilote ou runtime changés sans
    // changer la clé) est effacé : la prochaine préparation le reconstruit au lieu d'échouer
    // ici à chaque lancement.
    net::discard_artifact(artifact.path);
    log(BackendLogLevel::Warn, std::format("{} illisible ({}) : effacé pour être reconstruit",
                                           artifact.path.filename().string(), net.error().detail));
    return std::unexpected(net.error());
  }
  return std::make_unique<RuntimeEngine>(std::move(*net), std::move(*names));
}

void VulkanBackend::import_timeline(BackendQueue& queue) {
  auto fd = semaphore_fd(*context_, queue.semaphore());
  if (!fd) {
    log(BackendLogLevel::Warn,
        std::format("timeline non exportée ({}) : attente par l'hôte", fd.error().detail));
    return;
  }
  auto imported = runtime_->import_semaphore(*fd);
  if (!imported) {
#if !defined(_WIN32)
    ::close(*fd);  // le runtime ne le prend qu'en cas de succès
#endif
    log(BackendLogLevel::Warn, std::format("{} n'importe pas la timeline ({}) : attente par l'hôte",
                                           runtime_->name(), imported.error().detail));
    return;
  }
  imported_timeline_ = std::move(*imported);
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> VulkanBackend::alloc_runtime_tensor(
    TensorType type, const Shape& shape) {
  auto dense = detail::tensor_bytes(type, shape);
  if (!dense) return std::unexpected(dense.error());
  if (shared_memory_)
    return shared_tensor(*context_, *runtime_, type, shape, (*dense + 3) / 4 * 4, host_views_);
  // GPU intégré : la mémoire hôte du runtime (USM d'OpenVINO), importée par Vulkan, que le réseau
  // lit et écrit sans copie. Au premier refus, la mémoire hôte du backend pour la suite.
  if (host_alignment_ != 0) {
    auto imported = host_imported_tensor(*context_, *runtime_, type, shape, (*dense + 3) / 4 * 4,
                                         host_alignment_, host_views_);
    if (imported) return imported;
    host_alignment_ = 0;
    log(BackendLogLevel::Info, std::format("tenseurs de {} par la mémoire du backend ({})", runtime_->name(),
                                           imported.error().detail));
  }
  // En cache côté hôte : le runtime y lit les entrées et y écrit les sorties ; les noyaux de la
  // chaîne y accèdent comme à tout tampon de stockage.
  auto buffer = Buffer::create(*context_, *memory_, (*dense + 3) / 4 * 4, MemoryUse::Readback);
  if (!buffer) return std::unexpected(buffer.error());
  return std::make_unique<HostTensorBuffer>(*context_, std::move(*buffer), type, shape, host_views_);
}

std::expected<void, Refusal> VulkanBackend::enqueue_runtime(Engine& engine, std::span<const Tensor> inputs,
                                                            std::span<Tensor> outputs) {
  auto* runtime_engine = dynamic_cast<RuntimeEngine*>(&engine);
  if (runtime_engine == nullptr)
    return refuse_backend(RefusalCode::InputMismatch, "moteur d'un autre backend");
  if (inputs.size() != runtime_engine->input_names().size() ||
      outputs.size() != runtime_engine->output_names().size()) {
    return refuse_backend(
        RefusalCode::InputMismatch,
        std::format("{} entrées et {} sorties, {} et {} attendues", inputs.size(), outputs.size(),
                    runtime_engine->input_names().size(), runtime_engine->output_names().size()));
  }
  std::vector<net::HostBinding> bindings;
  bindings.reserve(inputs.size() + outputs.size());
  {
    const std::scoped_lock lock(host_views_->mutex);
    const auto bind = [&](const Tensor& t) -> std::expected<void, Refusal> {
      const auto found = host_views_->views.find(t.handle);
      auto bytes = detail::tensor_bytes(t.type, t.shape);
      if (!bytes) return std::unexpected(bytes.error());
      if (found == host_views_->views.end() ||
          (found->second.data == nullptr && found->second.device == nullptr) ||
          t.offset + *bytes > found->second.bytes) {
        return refuse_backend(RefusalCode::InputMismatch,
                              std::format("tenseur hors des tampons alloués pour {}", runtime_->name()));
      }
      const HostViews::View& v = found->second;
      bindings.push_back(
          {.data = v.data != nullptr ? v.data + t.offset : nullptr,
           .bytes = *bytes,
           .type = t.type,
           .shape = t.shape,
           .device = v.device != nullptr ? static_cast<std::byte*>(v.device) + t.offset : nullptr});
      return {};
    };
    for (const Tensor& t : inputs) {
      if (auto ok = bind(t); !ok) return ok;
    }
    for (const Tensor& t : outputs) {
      if (auto ok = bind(t); !ok) return ok;
    }
  }
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  {
    const std::scoped_lock lock(runtime_mutex_);
    if (runtime_failure_) return std::unexpected(*runtime_failure_);
  }
  const std::span<const net::HostBinding> all(bindings);
  const auto ins = all.first(inputs.size());
  const auto outs = all.subspan(inputs.size());
  // Ordonné côté GPU : le réseau attend la dernière soumission (partie avec le lot) et signale
  // l'étape que la suivante attend. Le lot part d'abord : le runtime n'attend que du soumis.
  std::optional<BackendQueue::Step> step;
  if (imported_timeline_) {
    if (auto ok = queue_->flush(); !ok) return std::unexpected(ok.error());
    step = queue_->external_step();
  }
  if (step) {
    auto ordered =
        runtime_engine->net().run_ordered(ins, outs, *imported_timeline_, step->wait, step->signal);
    if (ordered) return {};
    if (ordered.error().code != RefusalCode::FeatureUnavailable) {
      // Rien ne signalera l'étape : l'hôte le fait, une fois la précédente atteinte, sans quoi
      // la file attendrait sans fin.
      if (auto ok = context_->wait_on_host(queue_->at(step->wait)); !ok) return std::unexpected(ok.error());
      if (auto ok = queue_->signal_on_host(step->signal); !ok) return std::unexpected(ok.error());
      return ordered;
    }
  }
  // Par l'hôte, sans bloquer ce fil : le lot part, puis le runtime attend sur son propre fil que
  // les noyaux qui ont écrit les entrées soient finis (leurs écritures visibles de l'hôte par la
  // barrière de fin de chaque enregistrement), calcule, et signale l'étape ; ce fil enregistre
  // déjà la suite, que le GPU attend sur la timeline. Les sorties écrites par le runtime le sont
  // des soumissions suivantes, qui commencent après le signal.
  if (!step) {
    if (auto ok = queue_->flush(); !ok) return std::unexpected(ok.error());
    step = queue_->host_step();
    auto deferred = runtime_engine->net().run_deferred(
        ins, outs, [this, wait = step->wait] { return queue_->wait_elsewhere(wait); },
        [this, &queue = *queue_, signal = step->signal](std::expected<void, Refusal> ran) {
          finish_host_step(queue, signal, std::move(ran));
        });
    if (deferred) return {};
    if (deferred.error().code != RefusalCode::FeatureUnavailable) {
      // Rien n'a été lancé : l'étape se signale ici, sans quoi la file attendrait sans fin.
      if (auto ok = queue_->signal_on_host(step->signal); !ok) return std::unexpected(ok.error());
      return deferred;
    }
  }
  // Par l'hôte, sur ce fil : l'attente des entrées, le calcul, le signal de l'étape.
  if (auto ok = context_->wait_on_host(queue_->at(step->wait)); !ok) return std::unexpected(ok.error());
  auto ran = runtime_engine->net().run(ins, outs);
  if (auto ok = queue_->signal_on_host(step->signal); !ok) return std::unexpected(ok.error());
  return ran;
}

void VulkanBackend::finish_host_step(BackendQueue& queue, std::uint64_t signal,
                                     std::expected<void, Refusal> ran) {
  if (!ran) {
    log(BackendLogLevel::Error, std::format("réseau par {} : {}", runtime_->name(), ran.error().detail));
    const std::scoped_lock lock(runtime_mutex_);
    if (!runtime_failure_) runtime_failure_ = std::move(ran.error());
  }
  if (auto ok = queue.signal_on_host(signal); !ok) {
    log(BackendLogLevel::Error, std::format("signal de l'étape {} : {}", signal, ok.error().detail));
  }
}

}  // namespace ajp::gpu::vk
