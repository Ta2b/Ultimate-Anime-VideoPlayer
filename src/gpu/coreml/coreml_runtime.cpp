#include "gpu/coreml/coreml_runtime.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/log.h"
#include "gpu/coreml/coreml_compiled.h"
#include "gpu/coreml/coreml_detail.h"
#include "gpu/coreml/coreml_direct.h"
#include "gpu/coreml/coreml_steps.h"
#include "gpu/coreml/coreml_vt.h"
#include "gpu/ir/onnx_io_type.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/ir/onnx_unshuffle.h"
#include "gpu/ir/protobuf.h"
#include "gpu/system_interpolator.h"
#include "gpu/vulkan/backend/net_build.h"

namespace ajp::gpu::coreml_rt {

std::string_view units_option(Units units) {
  switch (units) {
    case Units::CpuAndGpu:
      return "CPUAndGPU";
    case Units::CpuAndNeuralEngine:
      return "CPUAndNeuralEngine";
    case Units::All:
      break;
  }
  return "ALL";
}

std::string_view units_name(Units units) {
  switch (units) {
    case Units::CpuAndGpu:
      return "gpu";
    case Units::CpuAndNeuralEngine:
      return "ane";
    case Units::All:
      break;
  }
  return "all";
}

std::optional<Units> units_from(std::string_view name) {
  for (const Units u : {Units::All, Units::CpuAndGpu, Units::CpuAndNeuralEngine}) {
    if (units_name(u) == name) return u;
  }
  return std::nullopt;
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

std::filesystem::path cache_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".mlcache";
  return p;
}

std::filesystem::path onnx_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".onnx";
  return p;
}

std::filesystem::path direct_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".mlmodelc";
  return p;
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

namespace {

using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorApple = 0x106B;

// Le journal d'ONNX Runtime dans celui du lecteur (param : le canal) : ses avertissements et
// erreurs, et le bilan que le fournisseur CoreML écrit en information à l'ouverture d'une session
// (nœuds pris, en combien de morceaux ; ceux qu'il laisse tournent sur le CPU).
void ORT_API_CALL forward_log(void* param, OrtLoggingLevel severity, const char* /*category*/,
                              const char* /*logid*/, const char* /*code_location*/, const char* message) {
  auto* channel = static_cast<spdlog::logger*>(param);
  const std::string_view text(message != nullptr ? message : "");
  if (severity >= ORT_LOGGING_LEVEL_ERROR) {
    channel->error("{}", text);
  } else if (severity == ORT_LOGGING_LEVEL_WARNING) {
    channel->warn("{}", text);
  } else if (text.contains("number of nodes supported by CoreML")) {
    channel->info("{}", text);
  }
}

// FP16 du programme ML, sur les unités données (choose_units). Le modèle compilé est gardé dans
// le cache de l'artefact : le chargement ne recompile pas.
Ort::SessionOptions session_options(const Manifest& manifest, const std::filesystem::path& artifact,
                                    Units units) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);  // le bilan du fournisseur (forward_log)
  for (const auto& [symbol, value] : manifest.dims)
    Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, symbol.c_str(), value));
  const std::unordered_map<std::string, std::string> coreml{
      {"ModelFormat", "MLProgram"},
      {"MLComputeUnits", std::string(units_option(units))},
      {"RequireStaticInputShapes", "0"},
      {"SpecializationStrategy", "FastPrediction"},
      {"AllowLowPrecisionAccumulationOnGPU", "1"},
      {"ModelCacheDirectory", cache_of(artifact).string()},
  };
  options.AppendExecutionProvider("CoreML", coreml);
  return options;
}

// Le fichier ONNX que la session ouvre : la copie à l'interface réécrite, ou l'original.
std::filesystem::path session_onnx(const Manifest& manifest, const std::filesystem::path& artifact) {
  return manifest.io ? onnx_of(artifact) : manifest.onnx;
}

// Écrit <artefact>.onnx, le modèle tel que la session l'ouvre, et dit si elle doit l'ouvrir (faux
// sans rien écrire quand l'original l'est déjà) :
// - la clé du cache de Core ML dans ses métadonnées (COREML_CACHE_KEY) : sans elle, ONNX Runtime
//   retrouve le modèle compilé par le chemin du fichier ONNX, qui change entre la construction
//   (dossier de travail) et le chargement, et recompilait le réseau au premier lancement ;
// - l'interface dans le type des tenseurs de la chaîne (ir/onnx_io_type.h), sauf pour un graphe
//   qui ne se réécrit pas (sous-graphes) : la conversion par l'hôte reste alors, à chaque image
//   (CoremlNet::run) ;
// - les entrées à la forme de la construction. Une dimension libre sans nom (les exports RIFE de
//   vs-mlrt écrivent -1) échappe aux surcharges par nom, et le fournisseur CoreML laisse alors au
//   CPU chaque nœud qui demande une forme connue : Reshape, DepthToSpace CRD, Resize qui réduit
//   (RIFE 4.22 lite : 28 nœuds sur 200 confiés à Core ML, en 4 morceaux) ;
// - pour un lot de 1, le dépliage de pixels 6-D des SPAN en 5-D (ir/onnx_unshuffle.h) : le
//   fournisseur refuse les rangs au-delà de 5, et ses trois nœuds passaient par le CPU.
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
  const bool batch_of_one = !inputs.empty() && std::ranges::all_of(inputs, [](const InputShape& input) {
    return !input.dims.empty() && input.dims[0] == 1;
  });
  if (batch_of_one) {
    auto unbatched = ir::onnx::with_unshuffle_5d(current);
    if (!unbatched) return std::unexpected(unbatched.error());
    if (!unbatched->empty()) {
      current = std::move(*unbatched);
      derived = true;
    }
  }
  // Le cache est propre à l'artefact (cache_of) : une clé fixe y suffit.
  auto keyed = ir::onnx::with_metadata(current, "COREML_CACHE_KEY", "ajp");
  if (!keyed) return std::unexpected(keyed.error());
  if (!keyed->empty()) {
    current = std::move(*keyed);
    derived = true;
  }
  if (!derived) return false;
  if (auto ok = vk::net::write_file(onnx_of(artifact), current); !ok) return std::unexpected(ok.error());
  return true;
}

// La version de la marque d'un interpolateur du système (gpu/system_interpolator.h), ou rien pour
// un réseau : les seuls champs producer_name et producer_version du ModelProto, sans décoder le
// graphe (poids externes compris, que le reste de la construction lit à sa façon).
std::expected<std::optional<std::string>, Refusal> interpolator_mark(const std::filesystem::path& onnx) {
  constexpr std::uint32_t kProducerName = 2;
  constexpr std::uint32_t kProducerVersion = 3;
  auto bytes = vk::net::read_file(onnx);
  if (!bytes) return std::unexpected(bytes.error());
  std::optional<std::string> producer;
  std::string version;
  ir::proto::Reader reader(*bytes);
  for (auto field = reader.next(); field && *field; field = reader.next()) {
    const ir::proto::Field& f = **field;
    if (f.number != kProducerName && f.number != kProducerVersion) continue;
    auto text = ir::proto::as_string(f);
    if (!text) return std::nullopt;
    if (f.number == kProducerName) {
      producer = std::move(*text);
    } else {
      version = std::move(*text);
    }
  }
  if (producer != kSystemInterpolatorProducer) return std::nullopt;
  return std::optional<std::string>(std::move(version));
}

// Ce qui calcule le réseau sur des liaisons déjà dans ses types : ONNX Runtime, ou Core ML
// directement (coreml_direct.h). Appelé sous le verrou de CoremlNet ; lève les exceptions
// d'ONNX Runtime.
class Executor {
 public:
  virtual ~Executor() = default;
  virtual std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                           std::span<const HostBinding> outputs) = 0;
};

// ONNX Runtime et son fournisseur CoreML : il enveloppe la mémoire des entrées dans ses
// MLMultiArray, et recopie chaque sortie que Core ML lui rend dans celle de la chaîne.
class OrtExecutor final : public Executor {
 public:
  OrtExecutor(std::shared_ptr<Ort::Env> env, Ort::Session session, const vk::net::IoNames& names)
      : env_(std::move(env)),
        session_(std::move(session)),
        binding_(session_),
        names_(names),
        bound_(names_.inputs.size() + names_.outputs.size()) {}

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    for (std::size_t i = 0; i < inputs.size(); ++i)
      bind(i, inputs[i], [&](const Ort::Value& v) { binding_.BindInput(names_.inputs[i].c_str(), v); });
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      bind(inputs.size() + i, outputs[i],
           [&](const Ort::Value& v) { binding_.BindOutput(names_.outputs[i].c_str(), v); });
    }
    session_.Run(Ort::RunOptions{nullptr}, binding_);
    return {};
  }

 private:
  // Lié une fois tant que la chaîne donne la même mémoire : ni tenseur ni liaison refaits à
  // chaque image.
  template <class Attach>
  void bind(std::size_t slot, const HostBinding& target, Attach&& attach) {
    HostBinding& last = bound_[slot];
    if (last.data == target.data && last.bytes == target.bytes && last.type == target.type &&
        last.shape == target.shape)
      return;
    Ort::Value value = Ort::Value::CreateTensor(memory_, target.data, target.bytes, target.shape.dims.data(),
                                                target.shape.dims.size(), element_of(target.type));
    attach(value);
    last = target;
  }

  std::shared_ptr<Ort::Env> env_;  // vit plus longtemps que la session
  Ort::Session session_;
  Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::IoBinding binding_;  // garde les tenseurs liés (bound_) d'un appel à l'autre
  vk::net::IoNames names_;
  std::vector<HostBinding> bound_;  // entrées puis sorties
};

// Core ML lui-même, sur le modèle qu'ONNX Runtime a compilé : les sorties écrites en place.
class DirectExecutor final : public Executor {
 public:
  explicit DirectExecutor(std::unique_ptr<DirectModel> model) : model_(std::move(model)) {}
  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    return model_->predict(inputs, outputs);
  }

 private:
  std::unique_ptr<DirectModel> model_;
};

class CoremlNet final : public vk::net::RuntimeNet {
 public:
  CoremlNet(std::unique_ptr<Executor> executor, vk::net::IoNames names, std::vector<TensorType> in_types,
            std::vector<TensorType> out_types)
      : executor_(std::move(executor)),
        names_(std::move(names)),
        in_types_(std::move(in_types)),
        out_types_(std::move(out_types)),
        in_scratch_(in_types_.size()),
        out_scratch_(out_types_.size()),
        staged_(in_types_.size() + out_types_.size()) {}
  CoremlNet(const CoremlNet&) = delete;
  CoremlNet& operator=(const CoremlNet&) = delete;
  CoremlNet(CoremlNet&&) = delete;
  CoremlNet& operator=(CoremlNet&&) = delete;
  ~CoremlNet() override = default;
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "Core ML : nombre d'entrées ou de sorties");
    const std::scoped_lock lock(session_mutex_);
    return infer(inputs, outputs);
  }

  // Sur le fil du réseau (DeferredSteps), dans l'ordre des appels.
  std::expected<void, Refusal> run_deferred(std::span<const HostBinding> inputs,
                                            std::span<const HostBinding> outputs,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "Core ML : nombre d'entrées ou de sorties");
    return steps_.push(inputs, outputs, std::move(ready), std::move(done));
  }

 private:
  // La mémoire du lecteur telle quelle : le réseau prend le type de ses tenseurs (interface
  // réécrite à la construction, derive_onnx), et Core ML y lit et y écrit. Sinon (artefact plus
  // ancien, graphe qui ne se réécrit pas) une copie convertie, dans les deux sens.
  std::expected<void, Refusal> infer(std::span<const HostBinding> inputs,
                                     std::span<const HostBinding> outputs) {
    try {
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const HostBinding& b = inputs[i];
        staged_[i] = staged(b, in_types_[i], in_scratch_[i]);
        if (in_types_[i] != b.type) {
          const std::size_t count = b.bytes / element_bytes(b.type);
          if (auto ok = vk::net::copy_converted(b.data, b.type, staged_[i].data, in_types_[i], count); !ok)
            return ok;
        }
      }
      for (std::size_t i = 0; i < outputs.size(); ++i)
        staged_[inputs.size() + i] = staged(outputs[i], out_types_[i], out_scratch_[i]);
      const std::span<const HostBinding> all(staged_);
      if (auto ok = executor_->run(all.first(inputs.size()), all.subspan(inputs.size())); !ok) return ok;
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
      return refuse(RefusalCode::DeviceUnavailable, std::format("Core ML, exécution : {}", e.what()));
    }
  }

  // La mémoire que le réseau lit ou écrit pour b : la sienne, ou la copie dans le type du réseau.
  static HostBinding staged(const HostBinding& b, TensorType wanted, std::vector<std::byte>& scratch) {
    if (wanted == b.type) return b;
    scratch.resize(b.bytes / element_bytes(b.type) * element_bytes(wanted));
    return {.data = scratch.data(), .bytes = scratch.size(), .type = wanted, .shape = b.shape};
  }

  std::unique_ptr<Executor> executor_;
  vk::net::IoNames names_;
  std::vector<TensorType> in_types_;
  std::vector<TensorType> out_types_;
  std::vector<std::vector<std::byte>> in_scratch_;
  std::vector<std::vector<std::byte>> out_scratch_;
  std::vector<HostBinding> staged_;  // entrées puis sorties, dans les types du réseau
  std::mutex session_mutex_;         // executor_, staged_ et les copies : le fil des étapes ou celui de run
  // Dernier membre : arrêté le premier, ses étapes en file faites sur un réseau encore là.
  DeferredSteps steps_{"", [this](std::span<const HostBinding> inputs, std::span<const HostBinding> outputs) {
                         const std::scoped_lock lock(session_mutex_);
                         return infer(inputs, outputs);
                       }};
};

std::expected<void, Refusal> write_manifest(const std::filesystem::path& path, const Manifest& manifest) {
  std::ofstream out(path, std::ios::trunc);
  out << "onnx " << manifest.onnx.string() << '\n';
  for (const auto& [symbol, value] : manifest.dims) out << "dim " << value << ' ' << symbol << '\n';
  if (manifest.units) out << "units " << units_name(*manifest.units) << '\n';
  if (manifest.io) out << "io rewritten\n";
  if (manifest.direct) out << "direct\n";
  if (manifest.surfaces) out << "surfaces\n";
  if (manifest.videotoolbox)
    out << "videotoolbox " << manifest.videotoolbox->width << ' ' << manifest.videotoolbox->height << '\n';
  if (!out) return refuse(RefusalCode::CacheIo, std::format("{} : écriture", path.string()));
  return {};
}

std::expected<Manifest, Refusal> read_manifest(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) return refuse(RefusalCode::CacheIo, std::format("{} : lecture", path.string()));
  Manifest manifest;
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("onnx ")) {
      manifest.onnx = line.substr(5);
    } else if (line.starts_with("dim ")) {
      std::istringstream fields(line.substr(4));
      std::int64_t value = 0;
      std::string symbol;
      fields >> value;
      std::getline(fields >> std::ws, symbol);
      if (!fields.eof() && fields.fail())
        return refuse(RefusalCode::CacheIo, std::format("{} : ligne « {} »", path.string(), line));
      manifest.dims.emplace_back(std::move(symbol), value);
    } else if (line.starts_with("units ")) {
      manifest.units = units_from(line.substr(6));
    } else if (line.starts_with("io ")) {
      manifest.io = true;
    } else if (line == "direct") {
      manifest.direct = true;
    } else if (line == "surfaces") {
      manifest.surfaces = true;
    } else if (line.starts_with("videotoolbox ")) {
      std::istringstream fields(line.substr(13));
      VtFrames frames;
      fields >> frames.width >> frames.height;
      if (fields.fail() || frames.width <= 0 || frames.height <= 0)
        return refuse(RefusalCode::CacheIo, std::format("{} : ligne « {} »", path.string(), line));
      manifest.videotoolbox = frames;
    }
  }
  if (manifest.onnx.empty())
    return refuse(RefusalCode::CacheIo, std::format("{} : sans ONNX", path.string()));
  return manifest;
}

// Le chemin direct (coreml_direct.h) pour cet artefact : le seul modèle compilé du cache copié à
// côté (direct_of), retenu si Core ML le charge et que sa sortie retrouve celle de la session sur
// l'image de test. Sinon ONNX Runtime à chaque image, comme avant, et la raison au journal. Ses
// IOSurface de même, sur un passage de plus ; sans elles, la mémoire des tenseurs telle quelle.
struct DirectChoice {
  bool direct = false;
  bool surfaces = false;
};

DirectChoice adopt_direct(Ort::Session& session, std::span<const InputShape> inputs,
                          const std::filesystem::path& artifact, Units units,
                          const std::shared_ptr<Surfaces>& surfaces) {
  const std::filesystem::path target = direct_of(artifact);
  const auto reject = [&](std::string_view why) {
    log_channel("coreml")->info("{} : Core ML direct écarté ({}) ; ONNX Runtime à chaque image",
                                artifact.filename().string(), why);
    std::error_code error;
    std::filesystem::remove_all(target, error);
    return DirectChoice{};
  };
  const std::optional<std::filesystem::path> compiled = single_compiled(cache_of(artifact));
  if (!compiled) return reject("pas un modèle compilé unique : réseau coupé en morceaux");
  std::error_code error;
  std::filesystem::remove_all(target, error);
  std::filesystem::copy(*compiled, target, std::filesystem::copy_options::recursive, error);
  if (error) return reject(std::format("copie de {} : {}", compiled->string(), error.message()));
  auto measured = direct_error(
      session, inputs,
      [&](std::span<const std::string> in, std::span<const std::string> out) {
        return open_direct(target, units, in, out, surfaces);
      },
      surfaces);
  if (!measured) return reject(measured.error().detail);
  if (measured->mean > kDirectMaxMeanError || measured->max > kDirectMaxError)
    return reject(std::format("écart à ONNX Runtime de {:.3g} en moyenne, {:.3g} au pire", measured->mean,
                              measured->max));
  if (!measured->on_surfaces) {
    log_channel("coreml")->info("{} : Core ML direct sans IOSurface ({})", artifact.filename().string(),
                                measured->surfaces_skipped);
    return {.direct = true};
  }
  if (measured->surface_mean > kDirectMaxMeanError || measured->surface_max > kDirectMaxError) {
    log_channel("coreml")->info(
        "{} : Core ML direct sans IOSurface (écart de {:.3g} en moyenne, {:.3g} au pire)",
        artifact.filename().string(), measured->surface_mean, measured->surface_max);
    return {.direct = true};
  }
  return {.direct = true, .surfaces = true};
}

class CoremlRuntime final : public vk::net::Runtime {
 public:
  CoremlRuntime(std::shared_ptr<Ort::Env> env, std::string device)
      : env_(std::move(env)), device_(std::move(device)) {}
  std::string_view name() const override { return "coreml"; }
  std::string device_name() const override { return std::format("{} (Core ML)", device_); }
  std::string version() const override { return Ort::GetVersionString(); }
  // Entrées à la forme de la construction et dépliage 5-D (derive_onnx) : les artefacts d'avant
  // laissaient au CPU une part des réseaux. Puis Core ML direct (adopt_direct) et la clé du cache
  // de Core ML : les artefacts d'avant n'ont pas le modèle compilé à côté, et se reconstruisent.
  // Puis les IOSurface, vérifiées à la construction.
  std::string_view recipe() const override { return "fixed-inputs-5d-direct-surfaces"; }
  std::string_view extension() const override { return ".coreml"; }

  // Les tenseurs de la chaîne en IOSurface (coreml_direct.h), que Vulkan importe : Core ML direct
  // les donne tels quels au GPU et au Neural Engine.
  std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_host(std::uint64_t bytes,
                                                                             std::uint64_t alignment,
                                                                             TensorType type,
                                                                             const Shape& shape) override {
    return allocate_surface(surfaces_, bytes, alignment, type, shape);
  }

  std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                     const std::filesystem::path& path, Progress& progress) override {
    auto mark = interpolator_mark(model.onnx_path);
    if (!mark) return std::unexpected(mark.error());
    if (*mark) return build_interpolator(model, **mark, inputs, path, progress);
    auto dims = vk::net::symbolic_overrides(model.onnx_path, inputs);
    if (!dims) return std::unexpected(dims.error());
    Manifest manifest{.onnx = model.onnx_path, .dims = std::move(*dims), .units = std::nullopt, .io = false};
    auto io = derive_onnx(model, inputs, path);
    if (!io) return std::unexpected(io.error());
    manifest.io = *io;
    if (progress.report) progress.report(0.05F, "compilation Core ML");
    auto units = choose_units(manifest, inputs, progress, [&](Units u) { return open(manifest, path, u); });
    if (!units) return std::unexpected(units.error());
    manifest.units = *units;
    // Ouvrir la session compile le programme ML (déjà fait par la mesure, alors relu du cache)
    // et remplit le cache de l'artefact.
    auto session = open(manifest, path, *units);
    if (!session) return std::unexpected(session.error());
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    if (progress.report) progress.report(0.9F, "vérification de Core ML direct");
    const DirectChoice direct = adopt_direct(*session, inputs, path, *units, surfaces_);
    manifest.direct = direct.direct;
    manifest.surfaces = direct.surfaces;
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    if (auto ok = write_manifest(path, manifest); !ok) return ok;
    if (progress.report) {
      progress.report(1.0F, std::format("modèle Core ML compilé ({}{}{})", units_name(*units),
                                        manifest.io ? ", interface réécrite" : "",
                                        manifest.surfaces ? ", exécuté en direct sur IOSurface"
                                        : manifest.direct ? ", exécuté en direct"
                                                          : ""));
    }
    return {};
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    auto manifest = read_manifest(path);
    if (!manifest) return std::unexpected(manifest.error());
    if (manifest->videotoolbox)
      return vt_net(std::move(*names), manifest->videotoolbox->width, manifest->videotoolbox->height,
                    surfaces_);
    auto units = units_of(*manifest);
    if (!units) return std::unexpected(units.error());
    if (manifest->direct) {
      auto direct = open_direct(direct_of(path), *units, names->inputs, names->outputs,
                                manifest->surfaces ? surfaces_ : nullptr);
      if (direct) {
        std::vector<TensorType> in_types((*direct)->input_types().begin(), (*direct)->input_types().end());
        std::vector<TensorType> out_types((*direct)->output_types().begin(), (*direct)->output_types().end());
        return std::make_unique<CoremlNet>(std::make_unique<DirectExecutor>(std::move(*direct)),
                                           std::move(*names), std::move(in_types), std::move(out_types));
      }
      log_channel("coreml")->warn("{} : {} ; ONNX Runtime à la place", path.filename().string(),
                                  direct.error().detail);
    }
    auto session = open(*manifest, path, *units);
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
                          std::format("Core ML : {} n'est pas flottant", name));
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
      auto executor = std::make_unique<OrtExecutor>(env_, std::move(*session), *names);
      return std::make_unique<CoremlNet>(std::move(executor), std::move(*names), std::move(*in_types),
                                         std::move(*out_types));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("Core ML, {} : {}", path.string(), e.what()));
    }
  }

 private:
  // L'interpolation d'Apple à la place du graphe : essayée à la taille des plans remplis de la
  // chaîne, son cadre gardé dans le manifeste. Refus FeatureUnavailable : une autre version de la
  // marque, un Mac qui ne l'offre pas ; la chaîne reprend alors RIFE.
  std::expected<void, Refusal> build_interpolator(const ModelSpec& model, std::string_view version,
                                                  std::span<const InputShape> inputs,
                                                  const std::filesystem::path& path, Progress& progress) {
    if (version != kVideoToolboxFrameRate)
      return refuse(RefusalCode::FeatureUnavailable,
                    std::format("interpolateur du système inconnu ({})", version));
    if (inputs.size() != 1 || inputs[0].dims.size() != 4 || inputs[0].dims[1] != 7 ||
        inputs[0].dims[2] <= 0 || inputs[0].dims[3] <= 0)
      return refuse(RefusalCode::InputMismatch, "interpolation d'Apple : entrée 1 × 7 × H × W attendue");
    const VtFrames frames{.width = static_cast<int>(inputs[0].dims[3]),
                          .height = static_cast<int>(inputs[0].dims[2])};
    if (progress.report) progress.report(0.1F, "interpolation d'Apple : essai de VideoToolbox");
    auto probe = vt_probe(frames.width, frames.height);
    if (!probe) return std::unexpected(probe.error());
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    const Manifest manifest{.onnx = model.onnx_path, .videotoolbox = frames};
    if (auto ok = write_manifest(path, manifest); !ok) return ok;
    if (progress.report) progress.report(1.0F, std::format("interpolation d'Apple prête ({})", *probe));
    return {};
  }

  std::expected<Ort::Session, Refusal> open(const Manifest& manifest, const std::filesystem::path& artifact,
                                            Units units) {
    const std::filesystem::path onnx = session_onnx(manifest, artifact);
    try {
      std::error_code error;
      std::filesystem::create_directories(cache_of(artifact), error);
      return Ort::Session(*env_, onnx.c_str(), session_options(manifest, artifact, units));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed, std::format("Core ML, {} : {}", onnx.string(), e.what()));
    }
  }

  std::shared_ptr<Ort::Env> env_;
  std::string device_;
  std::shared_ptr<Surfaces> surfaces_ = make_surfaces();
};

}  // namespace

std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  if (caps.vendor_id != kVendorApple) {
    return refuse(RefusalCode::DeviceMismatch,
                  std::format("Core ML sert les puces Apple ; {} : backend Vulkan seul", caps.device_name));
  }
  try {
    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    if (std::ranges::find(providers, "CoreMLExecutionProvider") == providers.end())
      return refuse(RefusalCode::LibraryUnavailable, "ONNX Runtime sans fournisseur CoreML");
    // Le canal vit autant que l'environnement, qui garde son adresse.
    auto channel = log_channel("onnxruntime");
    spdlog::logger* sink = channel.get();
    std::shared_ptr<Ort::Env> env(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ajp", &forward_log, sink),
                                  [channel = std::move(channel)](Ort::Env* e) { delete e; });
    return std::make_shared<CoremlRuntime>(std::move(env), caps.device_name);
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("ONNX Runtime : {}", e.what()));
  }
}

}  // namespace ajp::gpu::coreml_rt
