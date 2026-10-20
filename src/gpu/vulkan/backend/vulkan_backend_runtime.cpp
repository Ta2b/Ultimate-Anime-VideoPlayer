// Le backend Vulkan dont les réseaux passent par un autre runtime (net_runtime.h) : construction
// et chargement par le runtime, tenseurs en mémoire hôte visible, enqueue synchrone après
// l'attente de la file.
#include <format>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/buffer.h"

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
  HostTensorBuffer(Buffer buffer, TensorType type, const Shape& shape, std::shared_ptr<HostViews> views)
      : buffer_(std::move(buffer)),
        tensor_{.type = type, .shape = shape, .handle = detail::handle_bits(buffer_.handle()), .offset = 0},
        views_(std::move(views)) {
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.data = buffer_.mapped(), .bytes = buffer_.size()};
  }
  HostTensorBuffer(const HostTensorBuffer&) = delete;
  HostTensorBuffer& operator=(const HostTensorBuffer&) = delete;
  HostTensorBuffer(HostTensorBuffer&&) = delete;
  HostTensorBuffer& operator=(HostTensorBuffer&&) = delete;
  ~HostTensorBuffer() override {
    const std::scoped_lock lock(views_->mutex);
    views_->views.erase(tensor_.handle);
  }
  const Tensor& tensor() const override { return tensor_; }

 private:
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
  std::filesystem::create_directories(artifact.path.parent_path(), ec);
  if (ec) {
    return refuse_backend(RefusalCode::CacheIo,
                          std::format("{} : {}", artifact.path.parent_path().string(), ec.message()));
  }
  if (auto ok = runtime_->build(model, inputs, artifact.path, progress); !ok)
    return std::unexpected(ok.error());
  if (auto ok = net::write_io_names(artifact.path, names->inputs, names->outputs); !ok)
    return std::unexpected(ok.error());
  log(BackendLogLevel::Info, std::format("{} compilé par {}", model.id, runtime_->name()));
  return artifact;
}

std::expected<std::unique_ptr<Engine>, Refusal> VulkanBackend::load_runtime(const Artifact& artifact) {
  auto names = net::read_io_names(artifact.path);
  if (!names) return std::unexpected(names.error());
  auto net = runtime_->load(artifact.path);
  if (!net) return std::unexpected(net.error());
  return std::make_unique<RuntimeEngine>(std::move(*net), std::move(*names));
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> VulkanBackend::alloc_runtime_tensor(
    TensorType type, const Shape& shape) {
  auto dense = detail::tensor_bytes(type, shape);
  if (!dense) return std::unexpected(dense.error());
  // En cache côté hôte : le runtime y lit les entrées et y écrit les sorties ; les noyaux de la
  // chaîne y accèdent comme à tout tampon de stockage.
  auto buffer = Buffer::create(*context_, *memory_, (*dense + 3) / 4 * 4, MemoryUse::Readback);
  if (!buffer) return std::unexpected(buffer.error());
  return std::make_unique<HostTensorBuffer>(std::move(*buffer), type, shape, host_views_);
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
      if (found == host_views_->views.end() || found->second.data == nullptr ||
          t.offset + *bytes > found->second.bytes) {
        return refuse_backend(RefusalCode::InputMismatch,
                              std::format("tenseur hors des tampons alloués pour {}", runtime_->name()));
      }
      bindings.push_back(
          {.data = found->second.data + t.offset, .bytes = *bytes, .type = t.type, .shape = t.shape});
      return {};
    };
    for (const Tensor& t : inputs) {
      if (auto ok = bind(t); !ok) return ok;
    }
    for (const Tensor& t : outputs) {
      if (auto ok = bind(t); !ok) return ok;
    }
  }
  // Les noyaux qui ont écrit les entrées sont finis, leurs écritures visibles de l'hôte
  // (barrière de fin de chaque enregistrement) ; les sorties écrites ici le sont des soumissions
  // suivantes, qui commencent après le retour.
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  if (auto ok = context_->wait_on_host(queue_->last()); !ok) return std::unexpected(ok.error());
  const std::span<const net::HostBinding> all(bindings);
  return runtime_engine->net().run(all.first(inputs.size()), all.subspan(inputs.size()));
}

}  // namespace ajp::gpu::vk
