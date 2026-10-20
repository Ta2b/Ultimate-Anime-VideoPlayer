#include "gpu/coreml/coreml_runtime.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <optional>
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

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

// Le manifeste de l'artefact : le fichier ONNX, puis une ligne par dimension symbolique fixée.
struct Manifest {
  std::filesystem::path onnx;
  std::vector<DimOverride> dims;
};

std::filesystem::path cache_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".mlcache";
  return p;
}

// FP16 du programme ML ; Core ML répartit les couches entre le Neural Engine, le GPU et le CPU.
// Le modèle compilé est gardé dans le cache de l'artefact : le chargement ne recompile pas.
Ort::SessionOptions session_options(const Manifest& manifest, const std::filesystem::path& artifact) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  for (const auto& [symbol, value] : manifest.dims)
    Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, symbol.c_str(), value));
  const std::unordered_map<std::string, std::string> coreml{
      {"ModelFormat", "MLProgram"},
      {"MLComputeUnits", "MLComputeUnitsAll"},
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
    const Manifest manifest{.onnx = model.onnx_path, .dims = std::move(*dims)};
    if (progress.report) progress.report(0.1F, "compilation Core ML");
    // Ouvrir la session compile le programme ML et remplit le cache de l'artefact.
    if (auto session = open(manifest, path); !session) return std::unexpected(session.error());
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    if (auto ok = write_manifest(path, manifest); !ok) return ok;
    if (progress.report) progress.report(1.0F, "modèle Core ML compilé");
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
  std::expected<Ort::Session, Refusal> open(const Manifest& manifest, const std::filesystem::path& artifact) {
    try {
      std::error_code error;
      std::filesystem::create_directories(cache_of(artifact), error);
      return Ort::Session(*env_, manifest.onnx.c_str(), session_options(manifest, artifact));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed,
                    std::format("Core ML, {} : {}", manifest.onnx.string(), e.what()));
    }
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
