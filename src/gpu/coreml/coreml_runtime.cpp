#include "gpu/coreml/coreml_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ajp::gpu::coreml_rt {
namespace {

using vk::net::DimOverride;
using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorApple = 0x106B;

// Mesure d'une session : inférences de chauffe, puis médiane des suivantes.
constexpr int kWarmupRuns = 2;
constexpr std::size_t kTimedRuns = 5;
// Le Neural Engine garde un réseau s'il ne coûte pas plus que ce facteur du meilleur temps : il
// laisse le GPU, partagé avec la chaîne MoltenVK, RIFE et le rendu, à tout le reste.
constexpr double kAneTolerance = 1.25;
// Écart moyen permis entre deux répartitions sur la même entrée (images entre 0 et 1).
constexpr double kMaxMeanError = 0.01;

// Les unités de calcul que Core ML reçoit pour un réseau entier.
enum class Units : std::uint8_t { All, CpuAndGpu, CpuAndNeuralEngine };

std::string_view units_option(Units units) {
  switch (units) {
    case Units::CpuAndGpu:
      return "MLComputeUnitsCPUAndGPU";
    case Units::CpuAndNeuralEngine:
      return "MLComputeUnitsCPUAndNeuralEngine";
    case Units::All:
      break;
  }
  return "MLComputeUnitsAll";
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

// Le manifeste de l'artefact : le fichier ONNX, une ligne par dimension symbolique fixée, puis
// les unités retenues par la mesure (absentes d'un artefact plus ancien).
struct Manifest {
  std::filesystem::path onnx;
  std::vector<DimOverride> dims;
  std::optional<Units> units;
};

std::filesystem::path cache_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".mlcache";
  return p;
}

// FP16 du programme ML, sur les unités données (choose_units). Le modèle compilé est gardé dans
// le cache de l'artefact : le chargement ne recompile pas.
Ort::SessionOptions session_options(const Manifest& manifest, const std::filesystem::path& artifact,
                                    Units units) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
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

// Des entrées fixes et reproductibles entre 0 et 1, aux formes fixées par build (sinon celles du
// modèle), les mêmes pour chaque répartition mesurée.
struct Probe {
  std::vector<std::string> inputs;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<TensorType> types;
  std::vector<std::vector<std::byte>> data;
  std::vector<std::string> outputs;
};

std::expected<Probe, Refusal> make_probe(Ort::Session& session, std::span<const InputShape> fixed) {
  Probe probe;
  Ort::AllocatorWithDefaultOptions allocator;
  for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
    std::string name = session.GetInputNameAllocated(i, allocator).get();
    const Ort::TypeInfo info = session.GetInputTypeInfo(i);
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    const std::optional<TensorType> type = type_of(tensor.GetElementType());
    if (!type)
      return refuse(RefusalCode::FormatUnsupported, std::format("Core ML : {} n'est pas flottant", name));
    std::vector<std::int64_t> shape = tensor.GetShape();
    if (const auto at = std::ranges::find(fixed, name, &InputShape::name); at != fixed.end())
      shape = at->dims;
    std::size_t count = 1;
    for (const std::int64_t d : shape) {
      if (d <= 0)
        return refuse(RefusalCode::ShapeInvalid, std::format("Core ML : {}, dimension libre", name));
      count *= static_cast<std::size_t>(d);
    }
    std::vector<float> values(count);
    for (std::size_t k = 0; k < count; ++k)
      values[k] = static_cast<float>((k * 2654435761U) % 1024U) / 1023.0F;
    std::vector<std::byte> bytes(count * element_bytes(*type));
    if (auto ok = vk::net::copy_converted(reinterpret_cast<const std::byte*>(values.data()), TensorType::F32,
                                          bytes.data(), *type, count);
        !ok)
      return std::unexpected(ok.error());
    probe.inputs.push_back(std::move(name));
    probe.shapes.push_back(std::move(shape));
    probe.types.push_back(*type);
    probe.data.push_back(std::move(bytes));
  }
  for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
    probe.outputs.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
  return probe;
}

struct Timing {
  double ms = 0.0;               // médiane d'une inférence
  std::vector<float> first_out;  // première sortie, pour comparer deux répartitions
};

// Lève les exceptions d'ONNX Runtime.
std::expected<Timing, Refusal> measure(Ort::Session& session, Probe& probe) {
  const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> values;
  std::vector<const char*> in_names;
  for (std::size_t i = 0; i < probe.inputs.size(); ++i) {
    values.push_back(Ort::Value::CreateTensor(memory, probe.data[i].data(), probe.data[i].size(),
                                              probe.shapes[i].data(), probe.shapes[i].size(),
                                              element_of(probe.types[i])));
    in_names.push_back(probe.inputs[i].c_str());
  }
  std::vector<const char*> out_names;
  for (const std::string& name : probe.outputs) out_names.push_back(name.c_str());
  const auto run = [&] {
    return session.Run(Ort::RunOptions{nullptr}, in_names.data(), values.data(), values.size(),
                       out_names.data(), out_names.size());
  };
  for (int i = 0; i < kWarmupRuns; ++i) static_cast<void>(run());
  std::array<double, kTimedRuns> times{};
  std::vector<Ort::Value> outputs;
  for (double& t : times) {
    const auto start = std::chrono::steady_clock::now();
    outputs = run();
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  std::ranges::nth_element(times, times.begin() + static_cast<std::ptrdiff_t>(kTimedRuns / 2));
  Timing timing{.ms = times[kTimedRuns / 2], .first_out = {}};
  if (outputs.empty()) return timing;
  const auto info = outputs.front().GetTensorTypeAndShapeInfo();
  const std::optional<TensorType> type = type_of(info.GetElementType());
  if (!type) return timing;
  timing.first_out.resize(info.GetElementCount());
  if (auto ok = vk::net::copy_converted(static_cast<const std::byte*>(outputs.front().GetTensorRawData()),
                                        *type, reinterpret_cast<std::byte*>(timing.first_out.data()),
                                        TensorType::F32, timing.first_out.size());
      !ok)
    return std::unexpected(ok.error());
  return timing;
}

// Sans sortie flottante à comparer : nul.
double mean_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  if (a.empty()) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
  return sum / static_cast<double>(a.size());
}

class CoremlNet final : public vk::net::RuntimeNet {
 public:
  CoremlNet(std::shared_ptr<Ort::Env> env, Ort::Session session, vk::net::IoNames names,
            std::vector<TensorType> in_types, std::vector<TensorType> out_types)
      : env_(std::move(env)),
        session_(std::move(session)),
        names_(std::move(names)),
        in_types_(std::move(in_types)),
        out_types_(std::move(out_types)),
        in_scratch_(in_types_.size()),
        out_scratch_(out_types_.size()) {}
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "Core ML : nombre d'entrées ou de sorties");
    try {
      const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      Ort::IoBinding binding(session_);
      // La mémoire du lecteur telle quelle quand le réseau prend son type ; sinon une copie
      // convertie (un ONNX en FP32 : Core ML calcule en FP16 de toute façon).
      const auto bind = [&](const HostBinding& b, TensorType wanted, std::vector<std::byte>& scratch) {
        std::byte* data = b.data;
        std::size_t bytes = b.bytes;
        if (wanted != b.type) {
          bytes = b.bytes / (b.type == TensorType::F32 ? 4U : 2U) * (wanted == TensorType::F32 ? 4U : 2U);
          scratch.resize(bytes);
          data = scratch.data();
        }
        return Ort::Value::CreateTensor(memory, data, bytes, b.shape.dims.data(), b.shape.dims.size(),
                                        element_of(wanted));
      };
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const HostBinding& b = inputs[i];
        Ort::Value value = bind(b, in_types_[i], in_scratch_[i]);
        if (in_types_[i] != b.type) {
          const std::size_t count = b.bytes / (b.type == TensorType::F32 ? 4U : 2U);
          if (auto ok = vk::net::copy_converted(b.data, b.type, in_scratch_[i].data(), in_types_[i], count);
              !ok)
            return ok;
        }
        binding.BindInput(names_.inputs[i].c_str(), value);
      }
      for (std::size_t i = 0; i < outputs.size(); ++i)
        binding.BindOutput(names_.outputs[i].c_str(), bind(outputs[i], out_types_[i], out_scratch_[i]));
      session_.Run(Ort::RunOptions{nullptr}, binding);
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const HostBinding& b = outputs[i];
        if (out_types_[i] == b.type) continue;
        const std::size_t count = b.bytes / (b.type == TensorType::F32 ? 4U : 2U);
        if (auto ok = vk::net::copy_converted(out_scratch_[i].data(), out_types_[i], b.data, b.type, count);
            !ok)
          return ok;
      }
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("Core ML, exécution : {}", e.what()));
    }
  }

 private:
  std::shared_ptr<Ort::Env> env_;  // vit plus longtemps que la session
  Ort::Session session_;
  vk::net::IoNames names_;
  std::vector<TensorType> in_types_;
  std::vector<TensorType> out_types_;
  std::vector<std::vector<std::byte>> in_scratch_;
  std::vector<std::vector<std::byte>> out_scratch_;
};

std::expected<void, Refusal> write_manifest(const std::filesystem::path& path, const Manifest& manifest) {
  std::ofstream out(path, std::ios::trunc);
  out << "onnx " << manifest.onnx.string() << '\n';
  for (const auto& [symbol, value] : manifest.dims) out << "dim " << value << ' ' << symbol << '\n';
  if (manifest.units) out << "units " << units_name(*manifest.units) << '\n';
  if (!out) return refuse(RefusalCode::CacheIo, std::format("{} : écriture", path.string()));
  return {};
}

// Les unités d'un artefact : celles de la mesure ; sans elle (un artefact plus ancien, une
// mesure impossible), le GPU pour un réseau qui déforme (GridSample, que le Neural Engine
// n'exécute pas), sinon toutes.
std::expected<Units, Refusal> units_of(const Manifest& manifest) {
  if (manifest.units) return *manifest.units;
  auto warps = vk::net::onnx_uses_op(manifest.onnx, "GridSample");
  if (!warps) return std::unexpected(warps.error());
  return *warps ? Units::CpuAndGpu : Units::All;
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
    }
  }
  if (manifest.onnx.empty())
    return refuse(RefusalCode::CacheIo, std::format("{} : sans ONNX", path.string()));
  return manifest;
}

class CoremlRuntime final : public vk::net::Runtime {
 public:
  CoremlRuntime(std::shared_ptr<Ort::Env> env, std::string device)
      : env_(std::move(env)), device_(std::move(device)) {}
  std::string_view name() const override { return "coreml"; }
  std::string device_name() const override { return std::format("{} (Core ML)", device_); }
  std::string version() const override { return Ort::GetVersionString(); }
  std::string_view extension() const override { return ".coreml"; }

  std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                     const std::filesystem::path& path, Progress& progress) override {
    auto dims = vk::net::symbolic_overrides(model.onnx_path, inputs);
    if (!dims) return std::unexpected(dims.error());
    Manifest manifest{.onnx = model.onnx_path, .dims = std::move(*dims), .units = std::nullopt};
    if (progress.report) progress.report(0.05F, "compilation Core ML");
    auto units = choose_units(manifest, inputs, path, progress);
    if (!units) return std::unexpected(units.error());
    manifest.units = *units;
    // Ouvrir la session compile le programme ML (déjà fait par la mesure, alors relu du cache)
    // et remplit le cache de l'artefact.
    if (auto session = open(manifest, path, *units); !session) return std::unexpected(session.error());
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    if (auto ok = write_manifest(path, manifest); !ok) return ok;
    if (progress.report)
      progress.report(1.0F, std::format("modèle Core ML compilé ({})", units_name(*units)));
    return {};
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    auto manifest = read_manifest(path);
    if (!manifest) return std::unexpected(manifest.error());
    auto units = units_of(*manifest);
    if (!units) return std::unexpected(units.error());
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
      return std::make_unique<CoremlNet>(env_, std::move(*session), std::move(*names), std::move(*in_types),
                                         std::move(*out_types));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("Core ML, {} : {}", path.string(), e.what()));
    }
  }

 private:
  std::expected<Ort::Session, Refusal> open(const Manifest& manifest, const std::filesystem::path& artifact,
                                            Units units) {
    try {
      std::error_code error;
      std::filesystem::create_directories(cache_of(artifact), error);
      return Ort::Session(*env_, manifest.onnx.c_str(), session_options(manifest, artifact, units));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed,
                    std::format("Core ML, {} : {}", manifest.onnx.string(), e.what()));
    }
  }

  // Chaque répartition sur les mêmes entrées, le GPU d'abord, qui sert de référence : une
  // répartition dont la sortie s'en écarte est écartée. Le Neural Engine s'il reste dans
  // kAneTolerance du meilleur temps, sinon la plus rapide. Un réseau qui déforme reste au GPU ;
  // rien de mesurable : les unités par défaut (units_of).
  std::expected<Units, Refusal> choose_units(const Manifest& manifest, std::span<const InputShape> inputs,
                                             const std::filesystem::path& path, Progress& progress) {
    auto fallback = units_of(manifest);
    if (!fallback || *fallback == Units::CpuAndGpu) return fallback;
    struct Tried {
      Units units;
      double ms;
    };
    std::vector<Tried> kept;
    std::optional<std::vector<float>> reference;
    constexpr std::array kOrder{Units::CpuAndGpu, Units::All, Units::CpuAndNeuralEngine};
    for (std::size_t i = 0; i < kOrder.size(); ++i) {
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
      if (progress.report)
        progress.report(0.1F + 0.25F * static_cast<float>(i),
                        std::format("mesure Core ML ({})", units_name(kOrder[i])));
      auto session = open(manifest, path, kOrder[i]);
      if (!session) continue;
      try {
        auto probe = make_probe(*session, inputs);
        if (!probe) return fallback;
        auto timing = measure(*session, *probe);
        if (!timing) continue;
        if (!reference) {
          reference = std::move(timing->first_out);
        } else if (mean_error(*reference, timing->first_out) > kMaxMeanError) {
          continue;
        }
        kept.push_back({.units = kOrder[i], .ms = timing->ms});
      } catch (const std::exception&) {
        continue;  // une répartition que Core ML refuse à l'exécution
      }
    }
    if (kept.empty()) return fallback;
    const auto fastest = std::ranges::min_element(kept, {}, &Tried::ms);
    const auto ane = std::ranges::find(kept, Units::CpuAndNeuralEngine, &Tried::units);
    if (ane != kept.end() && ane->ms <= kAneTolerance * fastest->ms) return ane->units;
    return fastest->units;
  }

  std::shared_ptr<Ort::Env> env_;
  std::string device_;
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
    auto env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ajp");
    return std::make_shared<CoremlRuntime>(std::move(env), caps.device_name);
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("ONNX Runtime : {}", e.what()));
  }
}

}  // namespace ajp::gpu::coreml_rt
