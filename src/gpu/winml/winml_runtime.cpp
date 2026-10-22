#include "gpu/winml/winml_runtime.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/log.h"
#include "gpu/ir/onnx_io_type.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/vulkan/backend/deferred_steps.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/winml/winml_detail.h"
#include "gpu/winml/winml_library.h"

namespace ajp::gpu::winml_rt {
namespace {

using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorAmd = 0x1002;
// Le nom sous lequel la bibliothèque du fournisseur s'enregistre auprès d'ONNX Runtime.
constexpr const char* kRegistration = "ajp-migraphx";

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

ONNXTensorElementDataType element_of(TensorType type) {
  return type == TensorType::F32 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

std::optional<TensorType> type_of(ONNXTensorElementDataType type) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return TensorType::F32;
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) return TensorType::F16;
  return std::nullopt;
}

std::size_t element_bytes(TensorType type) {
  return type == TensorType::F32 ? 4U : 2U;
}

// Le journal d'ONNX Runtime dans celui du lecteur (param : le canal) : ses avertissements (nœuds
// laissés au CPU compris) et ses erreurs.
void ORT_API_CALL forward_log(void* param, OrtLoggingLevel severity, const char* /*category*/,
                              const char* /*logid*/, const char* /*code_location*/, const char* message) {
  auto* channel = static_cast<spdlog::logger*>(param);
  const std::string_view text(message != nullptr ? message : "");
  if (severity >= ORT_LOGGING_LEVEL_ERROR) {
    channel->error("{}", text);
  } else if (severity == ORT_LOGGING_LEVEL_WARNING) {
    channel->warn("{}", text);
  }
}

// L'environnement d'ONNX Runtime, unique dans le processus, avec la bibliothèque de MIGraphX
// enregistrée : partagé par les runtimes ouverts en même temps (un second enregistrement sous le
// même nom serait refusé), rendu avec le dernier. Le canal du journal vit plus longtemps que lui.
struct Shared {
  std::shared_ptr<spdlog::logger> channel;
  Ort::Env env{nullptr};
};

std::shared_ptr<Shared> shared_env(const Provider& provider) {
  static std::mutex mutex;
  static std::weak_ptr<Shared> current;  // sous mutex
  const std::scoped_lock lock(mutex);
  if (std::shared_ptr<Shared> alive = current.lock()) return alive;
  auto shared = std::make_shared<Shared>();
  shared->channel = log_channel("onnxruntime");
  shared->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ajp", &forward_log, shared->channel.get());
  shared->env.RegisterExecutionProviderLibrary(kRegistration, provider.library.native());
  current = shared;
  return shared;
}

// MIGraphX lit ses réglages dans l'environnement du processus à la création de chaque session :
// le dossier où il garde le programme compilé, et le calcul en half. Une session à la fois.
std::mutex& session_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Sous Windows, _wputenv_s change la copie de la CRT (getenv) et celle du système
// (GetEnvironmentVariable) : le fournisseur peut lire l'une ou l'autre.
void set_variable(const char* name, const std::filesystem::path& value) {
#ifdef _WIN32
  const std::filesystem::path wide_name(name);
  _wputenv_s(wide_name.c_str(), value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

// Écrit <artefact>.onnx, le modèle tel que la session l'ouvre, et dit si elle doit l'ouvrir (faux
// sans rien écrire quand l'original l'est déjà) : l'interface dans le type des tenseurs de la
// chaîne (ir/onnx_io_type.h), sauf pour un graphe qui ne se réécrit pas (sous-graphes, conversion
// par l'hôte à chaque image), et les entrées à la forme de la construction, que MIGraphX compile
// statiques. Comme pour Core ML (coreml_runtime.cpp), sans le dépliage 5-D, que MIGraphX n'exige
// pas.
std::expected<bool, Refusal> derive_onnx(const ModelSpec& model, std::span<const InputShape> inputs,
                                         const std::filesystem::path& artifact) {
  auto bytes = vk::net::read_file(model.onnx_path);
  if (!bytes) return std::unexpected(bytes.error());
  std::vector<std::byte> current = std::move(*bytes);
  bool derived = false;
  const std::int32_t io = model.precision == TensorType::F16 ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  auto typed = ir::onnx::with_io_type(current, io);
  if (!typed && typed.error().code != RefusalCode::FeatureUnavailable) return std::unexpected(typed.error());
  if (typed && !typed->empty()) {
    current = std::move(*typed);
    derived = true;
  }
  std::vector<ir::onnx::InputDims> fixed;
  fixed.reserve(inputs.size());
  for (const InputShape& input : inputs) fixed.push_back({.name = input.name, .dims = input.dims});
  auto shaped = ir::onnx::with_input_dims(current, fixed);
  if (!shaped) return std::unexpected(shaped.error());
  if (!shaped->empty()) {
    current = std::move(*shaped);
    derived = true;
  }
  if (!derived) return false;
  if (auto ok = vk::net::write_file(onnx_of(artifact), current); !ok) return std::unexpected(ok.error());
  return true;
}

std::size_t files_in(const std::filesystem::path& dir) {
  std::error_code error;
  std::size_t count = 0;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error))
    count += it->is_regular_file(error) ? 1U : 0U;
  return count;
}

class WinmlNet final : public vk::net::RuntimeNet {
 public:
  WinmlNet(std::shared_ptr<Shared> shared, Ort::Session session, vk::net::IoNames names,
           std::vector<TensorType> in_types, std::vector<TensorType> out_types)
      : shared_(std::move(shared)),
        session_(std::move(session)),
        binding_(session_),
        names_(std::move(names)),
        in_types_(std::move(in_types)),
        out_types_(std::move(out_types)),
        in_scratch_(in_types_.size()),
        out_scratch_(out_types_.size()),
        bound_(in_types_.size() + out_types_.size()) {}
  WinmlNet(const WinmlNet&) = delete;
  WinmlNet& operator=(const WinmlNet&) = delete;
  WinmlNet(WinmlNet&&) = delete;
  WinmlNet& operator=(WinmlNet&&) = delete;
  ~WinmlNet() override = default;
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "MIGraphX (Windows ML) : nombre d'entrées ou de sorties");
    const std::scoped_lock lock(session_mutex_);
    return infer(inputs, outputs);
  }

  // Sur le fil du réseau (DeferredSteps), dans l'ordre des appels : le fil qui nourrit le GPU
  // n'attend pas MIGraphX.
  std::expected<void, Refusal> run_deferred(std::span<const HostBinding> inputs,
                                            std::span<const HostBinding> outputs,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "MIGraphX (Windows ML) : nombre d'entrées ou de sorties");
    return steps_.push(inputs, outputs, std::move(ready), std::move(done));
  }

 private:
  struct Bound {
    std::byte* data = nullptr;
    std::size_t bytes = 0;
    TensorType type = TensorType::F16;
    Shape shape;
  };

  // La mémoire du lecteur telle quelle quand le réseau prend le type de ses tenseurs (interface
  // réécrite à la construction) : ONNX Runtime la copie vers le GPU et l'y recopie. Sinon une
  // copie convertie, dans les deux sens.
  std::expected<void, Refusal> infer(std::span<const HostBinding> inputs,
                                     std::span<const HostBinding> outputs) {
    try {
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const HostBinding& b = inputs[i];
        const Bound target = staged(b, in_types_[i], in_scratch_[i]);
        if (in_types_[i] != b.type) {
          const std::size_t count = b.bytes / element_bytes(b.type);
          if (auto ok = vk::net::copy_converted(b.data, b.type, target.data, in_types_[i], count); !ok)
            return ok;
        }
        bind(i, target, [&](const Ort::Value& v) { binding_.BindInput(names_.inputs[i].c_str(), v); });
      }
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Bound target = staged(outputs[i], out_types_[i], out_scratch_[i]);
        bind(inputs.size() + i, target,
             [&](const Ort::Value& v) { binding_.BindOutput(names_.outputs[i].c_str(), v); });
      }
      session_.Run(Ort::RunOptions{nullptr}, binding_);
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const HostBinding& b = outputs[i];
        if (out_types_[i] == b.type) continue;
        const std::size_t count = b.bytes / element_bytes(b.type);
        if (auto ok = vk::net::copy_converted(out_scratch_[i].data(), out_types_[i], b.data, b.type, count);
            !ok)
          return ok;
      }
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable,
                    std::format("MIGraphX (Windows ML), exécution : {}", e.what()));
    }
  }

  static Bound staged(const HostBinding& b, TensorType wanted, std::vector<std::byte>& scratch) {
    if (wanted == b.type) return {.data = b.data, .bytes = b.bytes, .type = wanted, .shape = b.shape};
    scratch.resize(b.bytes / element_bytes(b.type) * element_bytes(wanted));
    return {.data = scratch.data(), .bytes = scratch.size(), .type = wanted, .shape = b.shape};
  }

  // Lié une fois tant que la chaîne donne la même mémoire : ni tenseur ni liaison refaits à
  // chaque image.
  template <class Attach>
  void bind(std::size_t slot, const Bound& target, Attach&& attach) {
    Bound& last = bound_[slot];
    if (last.data == target.data && last.bytes == target.bytes && last.type == target.type &&
        last.shape == target.shape)
      return;
    Ort::Value value = Ort::Value::CreateTensor(memory_, target.data, target.bytes, target.shape.dims.data(),
                                                target.shape.dims.size(), element_of(target.type));
    attach(value);
    last = target;
  }

  std::shared_ptr<Shared> shared_;  // l'environnement vit plus longtemps que la session
  Ort::Session session_;
  Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::IoBinding binding_;
  vk::net::IoNames names_;
  std::vector<TensorType> in_types_;
  std::vector<TensorType> out_types_;
  std::vector<std::vector<std::byte>> in_scratch_;
  std::vector<std::vector<std::byte>> out_scratch_;
  std::vector<Bound> bound_;  // entrées puis sorties
  std::mutex session_mutex_;  // session_, binding_, bound_ et les copies : le fil des étapes ou celui de run
  // Dernier membre : arrêté le premier, ses étapes en file faites sur une session encore là.
  vk::net::DeferredSteps steps_{
      "migraphx", "MIGraphX (Windows ML)",
      [this](std::span<const HostBinding> inputs, std::span<const HostBinding> outputs) {
        const std::scoped_lock lock(session_mutex_);
        return infer(inputs, outputs);
      }};
};

// La mémoire épinglée du fournisseur (son allocateur HOST_ACCESSIBLE : hipHostMalloc), qu'un GPU
// intégré importe dans Vulkan (Runtime::allocate_host) : la chaîne y écrit les entrées du réseau et
// y lit ses sorties, et HIP les copie par DMA. D'une mémoire ordinaire, il les recopie d'abord par
// le CPU dans la sienne, à chaque image. Liée comme mémoire du CPU, comme avant : ONNX Runtime suit
// le même chemin, HIP reconnaît ses pages.
class PinnedMemory final : public vk::net::HostMemory {
 public:
  PinnedMemory(std::shared_ptr<Shared> shared, OrtAllocator* allocator, void* raw, std::byte* data)
      : shared_(std::move(shared)), allocator_(allocator), raw_(raw), data_(data) {}
  PinnedMemory(const PinnedMemory&) = delete;
  PinnedMemory& operator=(const PinnedMemory&) = delete;
  PinnedMemory(PinnedMemory&&) = delete;
  PinnedMemory& operator=(PinnedMemory&&) = delete;
  ~PinnedMemory() override { allocator_->Free(allocator_, raw_); }
  std::byte* data() const override { return data_; }

 private:
  std::shared_ptr<Shared> shared_;  // l'allocateur appartient à l'environnement
  OrtAllocator* allocator_;
  void* raw_;
  std::byte* data_;  // raw_ aligné
};

class WinmlRuntime final : public vk::net::Runtime {
 public:
  WinmlRuntime(std::shared_ptr<Shared> shared, Ort::ConstEpDevice device, std::string device_name,
               std::string ep_version)
      : shared_(std::move(shared)),
        device_(device),
        device_name_(std::move(device_name)),
        ep_version_(std::move(ep_version)) {}
  std::string_view name() const override { return "migraphx"; }
  std::string device_name() const override {
    return std::format("{} (MIGraphX de Windows ML)", device_name_);
  }
  // ONNX Runtime et le fournisseur, que Windows met à jour : un autre programme compilé.
  std::string version() const override {
    return std::format("onnxruntime {} + MIGraphX {}", ort_version(), ep_version_);
  }
  std::string_view extension() const override { return ".winml"; }

  // Sans allocateur épinglé, refus FeatureUnavailable : la mémoire du backend pour tous les
  // tenseurs, comme avant. Une allocation refusée : ce tenseur seul (FormatUnsupported).
  std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_host(
      std::uint64_t bytes, std::uint64_t alignment, TensorType /*type*/, const Shape& /*shape*/) override {
    try {
      const OrtApi& api = Ort::GetApi();
      const OrtMemoryInfo* info = api.EpDevice_MemoryInfo(device_, OrtDeviceMemoryType_HOST_ACCESSIBLE);
      if (info == nullptr)
        return refuse(RefusalCode::FeatureUnavailable, "MIGraphX (Windows ML) : pas de mémoire épinglée");
      OrtAllocator* allocator = nullptr;
      Ort::ThrowOnError(api.GetSharedAllocator(shared_->env, info, &allocator));
      if (allocator == nullptr)
        return refuse(RefusalCode::FeatureUnavailable,
                      "MIGraphX (Windows ML) : pas d'allocateur de mémoire épinglée");
      // HIP rend des pages entières, déjà alignées ; sinon une marge pour aligner l'adresse.
      for (const std::uint64_t margin : {std::uint64_t{0}, alignment}) {
        void* raw = allocator->Alloc(allocator, static_cast<std::size_t>(bytes + margin));
        if (raw == nullptr)
          return refuse(
              RefusalCode::FormatUnsupported,
              std::format("MIGraphX (Windows ML) : {} o de mémoire épinglée refusés", bytes + margin));
        const auto address = reinterpret_cast<std::uintptr_t>(raw);
        const std::uintptr_t aligned = (address + alignment - 1) / alignment * alignment;
        if (aligned - address <= margin)
          return std::make_unique<PinnedMemory>(shared_, allocator, raw,
                                                reinterpret_cast<std::byte*>(aligned));
        allocator->Free(allocator, raw);
      }
      return refuse(RefusalCode::FormatUnsupported, "MIGraphX (Windows ML) : mémoire épinglée non alignée");
    } catch (const std::exception& e) {
      return refuse(RefusalCode::FeatureUnavailable,
                    std::format("MIGraphX (Windows ML), mémoire épinglée : {}", e.what()));
    }
  }

  std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                     const std::filesystem::path& path, Progress& progress) override {
    auto dims = vk::net::symbolic_overrides(model.onnx_path, inputs);
    if (!dims) return std::unexpected(dims.error());
    Manifest manifest{.onnx = model.onnx_path,
                      .dims = std::move(*dims),
                      .io = false,
                      .fp16 = model.precision == TensorType::F16};
    auto io = derive_onnx(model, inputs, path);
    if (!io) return std::unexpected(io.error());
    manifest.io = *io;
    if (progress.report) progress.report(0.1F, "compilation MIGraphX (Windows ML)");
    const auto start = std::chrono::steady_clock::now();
    // Ouvrir la session compile le programme et le range dans le cache de l'artefact.
    if (auto session = open(manifest, path); !session) return std::unexpected(session.error());
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    if (auto ok = write_manifest(path, manifest); !ok) return ok;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::size_t cached = files_in(cache_of(path));
    log_channel("winml")->info("{} : compilé par MIGraphX en {:.1f} s, {} fichier(s) en cache{}", model.id,
                               seconds, cached, cached == 0 ? " (le prochain chargement recompilera)" : "");
    if (progress.report) progress.report(1.0F, std::format("programme MIGraphX compilé ({:.0f} s)", seconds));
    return {};
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    auto manifest = read_manifest(path);
    if (!manifest) return std::unexpected(manifest.error());
    auto session = open(*manifest, path);
    if (!session) return std::unexpected(session.error());
    try {
      // Le type de chaque entrée et sortie, par son nom.
      const auto types =
          [&](std::size_t count, auto name_at, auto info_at,
              const std::vector<std::string>& wanted) -> std::expected<std::vector<TensorType>, Refusal> {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<TensorType> out(wanted.size(), TensorType::F16);
        for (std::size_t i = 0; i < count; ++i) {
          const std::string name = name_at(i, allocator).get();
          const auto at = std::ranges::find(wanted, name);
          if (at == wanted.end()) continue;
          const auto type = type_of(info_at(i).GetTensorTypeAndShapeInfo().GetElementType());
          if (!type)
            return refuse(RefusalCode::FormatUnsupported,
                          std::format("MIGraphX (Windows ML) : {} n'est pas flottant", name));
          out[static_cast<std::size_t>(at - wanted.begin())] = *type;
        }
        return out;
      };
      Ort::Session& s = *session;
      auto in_types = types(
          s.GetInputCount(), [&](std::size_t i, auto& a) { return s.GetInputNameAllocated(i, a); },
          [&](std::size_t i) { return s.GetInputTypeInfo(i); }, names->inputs);
      if (!in_types) return std::unexpected(in_types.error());
      auto out_types = types(
          s.GetOutputCount(), [&](std::size_t i, auto& a) { return s.GetOutputNameAllocated(i, a); },
          [&](std::size_t i) { return s.GetOutputTypeInfo(i); }, names->outputs);
      if (!out_types) return std::unexpected(out_types.error());
      return std::make_unique<WinmlNet>(shared_, std::move(*session), std::move(*names), std::move(*in_types),
                                        std::move(*out_types));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo,
                    std::format("MIGraphX (Windows ML), {} : {}", path.string(), e.what()));
    }
  }

 private:
  // Le calcul est sur le GPU : les fils de l'hôte d'ONNX Runtime dorment au lieu d'attendre en
  // tournant, ce qui laisse sur un APU l'énergie au GPU.
  Ort::SessionOptions session_options(const Manifest& manifest) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    for (const auto& [symbol, value] : manifest.dims)
      Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, symbol.c_str(), value));
    options.AppendExecutionProvider_V2(shared_->env, {device_},
                                       std::unordered_map<std::string, std::string>{});
    return options;
  }

  std::expected<Ort::Session, Refusal> open(const Manifest& manifest, const std::filesystem::path& artifact) {
    const std::filesystem::path onnx = manifest.io ? onnx_of(artifact) : manifest.onnx;
    try {
      std::error_code error;
      std::filesystem::create_directories(cache_of(artifact), error);
      const std::scoped_lock lock(session_mutex());
      set_variable("ORT_MIGRAPHX_MODEL_CACHE_PATH", cache_of(artifact));
      set_variable("ORT_MIGRAPHX_FP16_ENABLE", manifest.fp16 ? "1" : "0");
      return Ort::Session(shared_->env, onnx.c_str(), session_options(manifest));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed,
                    std::format("MIGraphX (Windows ML), {} : {}", onnx.string(), e.what()));
    }
  }

  std::shared_ptr<Shared> shared_;
  Ort::ConstEpDevice device_;  // tenu par l'environnement, que shared_ garde
  std::string device_name_;
  std::string ep_version_;
};

}  // namespace

std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  if (caps.vendor_id != kVendorAmd) {
    return refuse(
        RefusalCode::DeviceMismatch,
        std::format("MIGraphX de Windows ML sert les GPU AMD ; {} : backend Vulkan seul", caps.device_name));
  }
  if (auto ok = load_libraries(); !ok) return std::unexpected(ok.error());
  auto provider = find_migraphx();
  if (!provider) return std::unexpected(provider.error());
  const auto log = log_channel("winml");
  if (!provider->ready) {
    // Windows l'installe une fois (Windows Update), le lecteur attend.
    log->info("MIGraphX {} : installation par Windows ML", provider->version);
    provider = ensure_migraphx({}, {});
    if (!provider) return std::unexpected(provider.error());
  }
  try {
    std::shared_ptr<Shared> shared = shared_env(*provider);
    std::vector<Ort::ConstEpDevice> devices;
    std::vector<DeviceId> ids;
    for (const Ort::ConstEpDevice& d : shared->env.GetEpDevices()) {
      if (std::strcmp(d.EpName(), kMigraphxProvider) != 0 || d.Device().Type() != OrtHardwareDeviceType_GPU)
        continue;
      devices.push_back(d);
      ids.push_back({.vendor = d.Device().VendorId(), .device = d.Device().DeviceId()});
    }
    const std::optional<std::size_t> chosen =
        pick_device(ids, {.vendor = caps.vendor_id, .device = caps.device_id});
    if (!chosen) {
      return refuse(
          RefusalCode::DeviceUnavailable,
          std::format("MIGraphX {} de Windows ML ne voit aucun GPU AMD (pilote AMD 25.10.13.09 ou après "
                      "requis) ; {} : backend Vulkan seul",
                      provider->version, caps.device_name));
    }
    const DeviceId& id = ids[*chosen];
    log->info("MIGraphX {} par Windows ML (ONNX Runtime {}), appareil {:04x}:{:04x}{}", provider->version,
              ort_version(), id.vendor, id.device,
              id.device == caps.device_id ? "" : std::format(" (chaîne Vulkan sur {:04x})", caps.device_id));
    return std::make_shared<WinmlRuntime>(std::move(shared), devices[*chosen], caps.device_name,
                                          provider->version);
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("ONNX Runtime de Windows ML : {}", e.what()));
  }
}

}  // namespace ajp::gpu::winml_rt
