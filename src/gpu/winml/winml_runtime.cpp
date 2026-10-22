#include "gpu/winml/winml_runtime.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
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
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/log.h"
#include "gpu/ir/onnx_io_type.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/vulkan/backend/deferred_steps.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_measure.h"
#include "gpu/winml/winml_detail.h"
#include "gpu/winml/winml_library.h"

namespace ajp::gpu::winml_rt {
namespace {

using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorAmd = 0x1002;
// Les noms sous lesquels les bibliothèques des fournisseurs s'enregistrent auprès d'ONNX Runtime.
constexpr const char* kRegistration = "ajp-migraphx";
constexpr const char* kVitisAiRegistration = "ajp-vitisai";
// L'attente d'une installation de VitisAI par Windows ML : le NPU est un plus, le GPU suffit ; au
// lancement suivant, l'installation reprend.
constexpr std::chrono::minutes kNpuInstallWait{3};
constexpr const char* kGpuLabel = "MIGraphX (Windows ML)";
constexpr std::string_view kNpuLabel = "VitisAI (NPU)";

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
// enregistrée, et celle de VitisAI quand le NPU sert : partagé par les runtimes ouverts en même
// temps (un second enregistrement sous le même nom serait refusé), rendu avec le dernier. Le canal
// du journal vit plus longtemps que lui.
struct Shared {
  std::shared_ptr<spdlog::logger> channel;
  Ort::Env env{nullptr};
  std::mutex mutex;      // vitisai
  bool vitisai = false;  // la bibliothèque de VitisAI enregistrée
};

// Enregistre VitisAI une fois dans l'environnement. Lève Ort::Exception.
void register_vitisai(Shared& shared, const Provider& provider) {
  const std::scoped_lock lock(shared.mutex);
  if (shared.vitisai) return;
  shared.env.RegisterExecutionProviderLibrary(kVitisAiRegistration, provider.library.native());
  shared.vitisai = true;
}

// Le fournisseur prêt, installé par Windows au besoin en au plus limit : au-delà, l'installation
// est annulée (Cancelled).
std::expected<Provider, Refusal> ensure_within(Ep ep, std::chrono::seconds limit) {
  std::stop_source deadline;
  std::jthread watchdog([&deadline, limit](const std::stop_token& done) {
    std::mutex mutex;
    std::condition_variable_any wake;
    std::unique_lock lock(mutex);
    static_cast<void>(wake.wait_for(lock, done, limit, [] { return false; }));
    if (!done.stop_requested()) deadline.request_stop();
  });
  return ensure_provider(ep, {}, deadline.get_token());
}

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

// Écrit <artefact>.npu.onnx, le modèle pour le NPU : tout en FLOAT (ir::onnx::with_float32), que
// le compilateur de VitisAI calcule en BF16, l'interface remise dans le type des tenseurs de la
// chaîne (des Cast autour d'un graphe FLOAT), les entrées à la forme de la construction.
std::expected<void, Refusal> derive_npu_onnx(const ModelSpec& model, std::span<const InputShape> inputs,
                                             const std::filesystem::path& artifact) {
  auto bytes = vk::net::read_file(model.onnx_path);
  if (!bytes) return std::unexpected(bytes.error());
  std::vector<std::byte> current = std::move(*bytes);
  auto f32 = ir::onnx::with_float32(current);
  if (!f32) return std::unexpected(f32.error());
  if (!f32->empty()) current = std::move(*f32);
  const std::int32_t io = model.precision == TensorType::F16 ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  auto typed = ir::onnx::with_io_type(current, io);
  if (!typed) return std::unexpected(typed.error());
  if (!typed->empty()) current = std::move(*typed);
  std::vector<ir::onnx::InputDims> fixed;
  fixed.reserve(inputs.size());
  for (const InputShape& input : inputs) fixed.push_back({.name = input.name, .dims = input.dims});
  auto shaped = ir::onnx::with_input_dims(current, fixed);
  if (!shaped) return std::unexpected(shaped.error());
  if (!shaped->empty()) current = std::move(*shaped);
  return vk::net::write_file(npu_onnx_of(artifact), current);
}

// Un réseau mesuré sur l'image de test : la médiane d'une inférence et sa première sortie.
struct Measure {
  double ms = 0.0;
  std::vector<float> first_out;
};

// La session d'un réseau à une entrée sur l'image de test (vk::net::probe_image) à la forme de
// input, comme la chaîne la lie : la mémoire de l'hôte, liée une fois. Une première inférence dit
// la forme des sorties. Lève Ort::Exception.
std::expected<Measure, Refusal> measure(Ort::Session& session, const vk::net::IoNames& names,
                                        const InputShape& input) {
  const Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const auto in_type = type_of(session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType());
  if (!in_type) return refuse(RefusalCode::FormatUnsupported, "entrée qui n'est pas flottante");
  const std::vector<float> image = vk::net::probe_image(input.dims);
  std::vector<std::byte> in_bytes(image.size() * element_bytes(*in_type));
  if (auto ok = vk::net::copy_converted(reinterpret_cast<const std::byte*>(image.data()), TensorType::F32,
                                        in_bytes.data(), *in_type, image.size());
      !ok)
    return std::unexpected(ok.error());
  Ort::Value in_value = Ort::Value::CreateTensor(cpu, in_bytes.data(), in_bytes.size(), input.dims.data(),
                                                 input.dims.size(), element_of(*in_type));
  const char* in_name = names.inputs[0].c_str();
  std::vector<const char*> out_names;
  for (const std::string& name : names.outputs) out_names.push_back(name.c_str());
  std::vector<Ort::Value> first =
      session.Run(Ort::RunOptions{nullptr}, &in_name, &in_value, 1, out_names.data(), out_names.size());
  Ort::IoBinding binding(session);
  binding.BindInput(in_name, in_value);
  std::vector<std::vector<std::byte>> outputs(first.size());
  std::vector<TensorType> out_types(first.size(), TensorType::F16);
  for (std::size_t i = 0; i < first.size(); ++i) {
    const Ort::TensorTypeAndShapeInfo info = first[i].GetTensorTypeAndShapeInfo();
    const auto type = type_of(info.GetElementType());
    if (!type) return refuse(RefusalCode::FormatUnsupported, "sortie qui n'est pas flottante");
    out_types[i] = *type;
    const std::vector<std::int64_t> shape = info.GetShape();
    outputs[i].resize(info.GetElementCount() * element_bytes(*type));
    binding.BindOutput(out_names[i], Ort::Value::CreateTensor(cpu, outputs[i].data(), outputs[i].size(),
                                                              shape.data(), shape.size(), element_of(*type)));
  }
  for (int i = 1; i < vk::net::kUnitWarmupRuns; ++i) session.Run(Ort::RunOptions{nullptr}, binding);
  std::array<double, vk::net::kUnitTimedRuns> times{};
  for (double& t : times) {
    const auto start = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, binding);
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  Measure measured{.ms = vk::net::median(times)};
  if (!outputs.empty()) {
    const std::size_t count = outputs[0].size() / element_bytes(out_types[0]);
    measured.first_out.resize(count);
    if (auto ok = vk::net::copy_converted(outputs[0].data(), out_types[0],
                                          reinterpret_cast<std::byte*>(measured.first_out.data()),
                                          TensorType::F32, count);
        !ok)
      return std::unexpected(ok.error());
  }
  return measured;
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
  // label : le fournisseur, dans les refus (« MIGraphX (Windows ML) », « VitisAI (NPU) »).
  WinmlNet(std::shared_ptr<Shared> shared, std::string label, Ort::Session session, vk::net::IoNames names,
           std::vector<TensorType> in_types, std::vector<TensorType> out_types)
      : shared_(std::move(shared)),
        label_(std::move(label)),
        session_(std::move(session)),
        binding_(session_),
        names_(std::move(names)),
        in_types_(std::move(in_types)),
        out_types_(std::move(out_types)),
        in_scratch_(in_types_.size()),
        out_scratch_(out_types_.size()),
        bound_(in_types_.size() + out_types_.size()),
        steps_(label_ == kNpuLabel ? "vitisai" : "migraphx", label_,
               [this](std::span<const HostBinding> inputs, std::span<const HostBinding> outputs) {
                 const std::scoped_lock lock(session_mutex_);
                 return infer(inputs, outputs);
               }) {}
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
      return refuse(RefusalCode::InputMismatch, std::format("{} : nombre d'entrées ou de sorties", label_));
    const std::scoped_lock lock(session_mutex_);
    return infer(inputs, outputs);
  }

  // Sur le fil du réseau (DeferredSteps), dans l'ordre des appels : le fil qui nourrit le GPU
  // n'attend pas le fournisseur.
  std::expected<void, Refusal> run_deferred(std::span<const HostBinding> inputs,
                                            std::span<const HostBinding> outputs,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, std::format("{} : nombre d'entrées ou de sorties", label_));
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
      return refuse(RefusalCode::DeviceUnavailable, std::format("{}, exécution : {}", label_, e.what()));
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
  std::string label_;
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
  vk::net::DeferredSteps steps_;
};

// Le réseau chargé : le type de chaque entrée et sortie, par son nom. Lève Ort::Exception.
std::expected<std::unique_ptr<WinmlNet>, Refusal> make_net(std::shared_ptr<Shared> shared, std::string label,
                                                           Ort::Session session, vk::net::IoNames names) {
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
        return refuse(RefusalCode::FormatUnsupported, std::format("{} : {} n'est pas flottant", label, name));
      out[static_cast<std::size_t>(at - wanted.begin())] = *type;
    }
    return out;
  };
  Ort::Session& s = session;
  auto in_types = types(
      s.GetInputCount(), [&](std::size_t i, auto& a) { return s.GetInputNameAllocated(i, a); },
      [&](std::size_t i) { return s.GetInputTypeInfo(i); }, names.inputs);
  if (!in_types) return std::unexpected(in_types.error());
  auto out_types = types(
      s.GetOutputCount(), [&](std::size_t i, auto& a) { return s.GetOutputNameAllocated(i, a); },
      [&](std::size_t i) { return s.GetOutputTypeInfo(i); }, names.outputs);
  if (!out_types) return std::unexpected(out_types.error());
  return std::make_unique<WinmlNet>(std::move(shared), std::move(label), std::move(session), std::move(names),
                                    std::move(*in_types), std::move(*out_types));
}

// Le NPU d'un Ryzen AI, par VitisAI.
struct Npu {
  Ort::ConstEpDevice device;  // tenu par l'environnement
  std::string version;        // celle du paquet de VitisAI
};

// Le NPU par VitisAI si Windows ML le propose (installé au besoin, en au plus kNpuInstallWait) et
// voit le NPU ; sinon rien, et les réseaux restent sur le GPU.
std::optional<Npu> find_npu(Shared& shared, spdlog::logger& log) {
  auto provider = find_provider(Ep::VitisAi);
  if (!provider) {
    log.info("NPU : {} ; réseaux sur le GPU", provider.error().detail);
    return std::nullopt;
  }
  if (!provider->ready) {
    log.info("VitisAI {} : installation par Windows ML", provider->version);
    provider = ensure_within(Ep::VitisAi, kNpuInstallWait);
    if (!provider) {
      log.info("VitisAI : {} ; réseaux sur le GPU", provider.error().detail);
      return std::nullopt;
    }
  }
  try {
    register_vitisai(shared, *provider);
    for (const Ort::ConstEpDevice& d : shared.env.GetEpDevices()) {
      if (std::strcmp(d.EpName(), kVitisAiProvider) != 0 || d.Device().Type() != OrtHardwareDeviceType_NPU)
        continue;
      log.info(
          "VitisAI {} par Windows ML, NPU {:04x}:{:04x} : les réseaux sans GridSample y sont mesurés "
          "contre le GPU",
          provider->version, d.Device().VendorId(), d.Device().DeviceId());
      return Npu{.device = d, .version = provider->version};
    }
    log.info("VitisAI {} ne voit aucun NPU ; réseaux sur le GPU", provider->version);
  } catch (const std::exception& e) {
    log.info("VitisAI : {} ; réseaux sur le GPU", e.what());
  }
  return std::nullopt;
}

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
               std::string ep_version, std::optional<Npu> npu)
      : shared_(std::move(shared)),
        device_(device),
        device_name_(std::move(device_name)),
        ep_version_(std::move(ep_version)),
        npu_(std::move(npu)) {}
  std::string_view name() const override { return "migraphx"; }
  // Entre dans la clé des artefacts : un NPU qui apparaît (VitisAI installé) remesure les réseaux.
  std::string device_name() const override {
    return std::format("{} (MIGraphX de Windows ML){}", device_name_, npu_ ? " + NPU (VitisAI)" : "");
  }
  // ONNX Runtime et les fournisseurs, que Windows met à jour : un autre programme compilé.
  std::string version() const override {
    return std::format("onnxruntime {} + MIGraphX {}{}", ort_version(), ep_version_,
                       npu_ ? std::format(" + VitisAI {}", npu_->version) : "");
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
          return refuse(RefusalCode::FormatUnsupported,
                        std::format("{} : {} o de mémoire épinglée refusés", kGpuLabel, bytes + margin));
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
    auto session = open(manifest, path);
    if (!session) return std::unexpected(session.error());
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::size_t cached = files_in(cache_of(path));
    log_channel("winml")->info("{} : compilé par MIGraphX en {:.1f} s, {} fichier(s) en cache{}", model.id,
                               seconds, cached, cached == 0 ? " (le prochain chargement recompilera)" : "");
    if (npu_ && inputs.size() == 1) {
      auto warps = vk::net::onnx_uses_op(model.onnx_path, "GridSample");
      if (!warps) return std::unexpected(warps.error());
      // RIFE, GMFSS et DRBA déforment par GridSample : ils restent sur le GPU.
      if (!*warps) manifest.npu = try_npu(model, inputs, path, manifest, *session, progress);
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    }
    if (auto ok = write_manifest(path, manifest); !ok) return ok;
    if (progress.report) {
      progress.report(1.0F, std::format("programme MIGraphX compilé ({:.0f} s){}", seconds,
                                        manifest.npu ? ", exécuté sur le NPU" : ""));
    }
    return {};
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    auto manifest = read_manifest(path);
    if (!manifest) return std::unexpected(manifest.error());
    // Mesuré plus juste et pas plus lent sur le NPU : il y reste, le GPU garde la chaîne et
    // l'interpolation. Sans NPU maintenant, la clé de l'artefact a changé avec lui.
    const bool on_npu = manifest->npu && npu_;
    auto session = on_npu ? load_npu(*manifest, path) : open(*manifest, path);
    if (!session) return std::unexpected(session.error());
    const std::string label(on_npu ? kNpuLabel : kGpuLabel);
    try {
      auto net = make_net(shared_, label, std::move(*session), std::move(*names));
      if (!net) return std::unexpected(net.error());
      return std::unique_ptr<vk::net::RuntimeNet>(std::move(*net));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("{}, {} : {}", label, path.string(), e.what()));
    }
  }

 private:
  // Le réseau sur le NPU s'il y tient (vk::net::choose_unit, le GPU en référence) : compilé par
  // VitisAI, aussi juste que MIGraphX sur l'image de test, et pas plus lent que kUnitTolerance fois
  // lui. Sinon faux, ses fichiers effacés : le GPU le garde.
  bool try_npu(const ModelSpec& model, std::span<const InputShape> inputs, const std::filesystem::path& path,
               const Manifest& manifest, Ort::Session& gpu, Progress& progress) {
    const auto say = [&](float fraction, const std::string& text) {
      if (progress.report) progress.report(fraction, text);
    };
    const auto drop = [&] {
      std::error_code error;
      std::filesystem::remove(npu_onnx_of(path), error);
      std::filesystem::remove_all(npu_cache_of(path), error);
    };
    const auto refused = [&](const std::string& why) {
      const std::string text = std::format("{} : NPU écarté ({}), GPU", model.id, why);
      log_channel("winml")->info("{}", text);
      say(0.95F, text);
      drop();
      return false;
    };
    try {
      auto names = vk::net::onnx_io_names(model.onnx_path);
      if (!names) return refused(names.error().detail);
      if (auto ok = derive_npu_onnx(model, inputs, path); !ok) return refused(ok.error().detail);
      say(0.40F, "compilation VitisAI (NPU)");
      double npu_seconds = 0.0;
      // La session du NPU fermée avant que ses fichiers soient effacés.
      auto measured = [&]() -> std::expected<std::array<Measure, 2>, Refusal> {
        const auto start = std::chrono::steady_clock::now();
        Ort::Session npu = open_npu(manifest, path);
        npu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
        say(0.75F, "mesure du GPU et du NPU");
        auto on_gpu = measure(gpu, *names, inputs[0]);
        if (!on_gpu) return std::unexpected(on_gpu.error());
        auto on_npu = measure(npu, *names, inputs[0]);
        if (!on_npu) return std::unexpected(on_npu.error());
        return std::array<Measure, 2>{std::move(*on_gpu), std::move(*on_npu)};
      }();
      if (!measured) return refused(measured.error().detail);
      const auto& [on_gpu, on_npu] = *measured;
      const double error = vk::net::mean_error(on_gpu.first_out, on_npu.first_out);
      const std::array<std::optional<vk::net::UnitMeasure>, 2> units{
          vk::net::UnitMeasure{.ms = on_gpu.ms}, vk::net::UnitMeasure{.ms = on_npu.ms, .error = error}};
      const bool kept = vk::net::choose_unit(units, 1) == 1U;
      const std::string text =
          std::format("{} : NPU {:.1f} ms (compilé en {:.0f} s), GPU {:.1f} ms, écart {:.4f} : {}", model.id,
                      on_npu.ms, npu_seconds, on_gpu.ms, error, kept ? "NPU" : "GPU");
      log_channel("winml")->info("{}", text);
      say(0.95F, text);
      if (!kept) drop();
      return kept;
    } catch (const std::exception& e) {
      return refused(e.what());
    }
  }

  // La session du NPU : VitisAI compile <artefact>.npu.onnx (en BF16) et range le programme dans
  // <artefact>.vaicache, que le chargement suivant reprend. Lève Ort::Exception.
  Ort::Session open_npu(const Manifest& manifest, const std::filesystem::path& artifact) {
    std::error_code error;
    std::filesystem::create_directories(npu_cache_of(artifact), error);
    const std::u8string cache = npu_cache_of(artifact).u8string();
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    for (const auto& [symbol, value] : manifest.dims)
      Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, symbol.c_str(), value));
    options.AppendExecutionProvider_V2(
        shared_->env, {npu_->device},
        std::unordered_map<std::string, std::string>{{"cache_dir", std::string(cache.begin(), cache.end())},
                                                     {"cache_key", "ajp"},
                                                     {"enable_cache_file_io_in_mem", "0"}});
    return Ort::Session(shared_->env, npu_onnx_of(artifact).c_str(), options);
  }

  std::expected<Ort::Session, Refusal> load_npu(const Manifest& manifest,
                                                const std::filesystem::path& artifact) {
    try {
      return open_npu(manifest, artifact);
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed,
                    std::format("{}, {} : {}", kNpuLabel, npu_onnx_of(artifact).string(), e.what()));
    }
  }

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
  std::optional<Npu> npu_;
};

}  // namespace

std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  if (caps.vendor_id != kVendorAmd) {
    return refuse(
        RefusalCode::DeviceMismatch,
        std::format("MIGraphX de Windows ML sert les GPU AMD ; {} : backend Vulkan seul", caps.device_name));
  }
  if (auto ok = load_libraries(); !ok) return std::unexpected(ok.error());
  auto provider = find_provider(Ep::Migraphx);
  if (!provider) return std::unexpected(provider.error());
  const auto log = log_channel("winml");
  if (!provider->ready) {
    // Windows l'installe une fois (Windows Update), le lecteur attend.
    log->info("MIGraphX {} : installation par Windows ML", provider->version);
    provider = ensure_provider(Ep::Migraphx, {}, {});
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
    std::optional<Npu> npu = find_npu(*shared, *log);
    return std::make_shared<WinmlRuntime>(std::move(shared), devices[*chosen], caps.device_name,
                                          provider->version, std::move(npu));
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("ONNX Runtime de Windows ML : {}", e.what()));
  }
}

}  // namespace ajp::gpu::winml_rt
