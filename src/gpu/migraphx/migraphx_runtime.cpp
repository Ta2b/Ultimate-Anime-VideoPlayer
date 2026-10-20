#include "gpu/migraphx/migraphx_runtime.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/version.h>
#include <optional>
#include <string>
#include <vector>

namespace ajp::gpu::migraphx_rt {
namespace {

using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorAmd = 0x1002;

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

std::unexpected<Refusal> refuse_exception(std::string_view what, const std::exception& e) {
  return refuse(RefusalCode::BuilderFailed, std::format("MIGraphX, {} : {}", what, e.what()));
}

std::optional<TensorType> type_of(migraphx_shape_datatype_t type) {
  if (type == migraphx_shape_float_type) return TensorType::F32;
  if (type == migraphx_shape_half_type) return TensorType::F16;
  return std::nullopt;
}

// Un programme chargé ; ses entrées dans l'ordre de l'ONNX (fichier .io de l'artefact).
class MigraphxNet final : public vk::net::RuntimeNet {
 public:
  MigraphxNet(int device, migraphx::program program, vk::net::IoNames names)
      : device_(device), program_(std::move(program)), names_(std::move(names)) {}
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "MIGraphX : nombre d'entrées ou de sorties");
    try {
      if (hipSetDevice(device_) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
      const migraphx::program_parameter_shapes shapes = program_.get_parameter_shapes();
      migraphx::program_parameters params;
      staged_.resize(inputs.size());
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::string& name = names_.inputs[i];
        const migraphx::shape shape = shapes[name.c_str()];
        const auto want = type_of(shape.type());
        if (!want) return refuse(RefusalCode::FormatUnsupported, std::format("{} : type MIGraphX", name));
        void* data = inputs[i].data;
        if (*want != inputs[i].type) {
          // Programme en float (quantize_fp16 garde ses entrées), tenseurs du lecteur en F16.
          staged_[i].resize(shape.bytes());
          if (auto ok = vk::net::copy_converted(inputs[i].data, inputs[i].type, staged_[i].data(), *want,
                                                shape.elements());
              !ok) {
            return ok;
          }
          data = staged_[i].data();
        } else if (shape.bytes() != inputs[i].bytes) {
          return refuse(RefusalCode::InputMismatch,
                        std::format("{} : {} octets, {} attendus", name, inputs[i].bytes, shape.bytes()));
        }
        params.add(name.c_str(), migraphx::argument(shape, data));
      }
      const migraphx::arguments results = program_.eval(params);
      if (results.size() != outputs.size()) {
        return refuse(RefusalCode::InputMismatch,
                      std::format("MIGraphX : {} sorties, {} attendues", results.size(), outputs.size()));
      }
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const migraphx::argument result = results[i];
        const migraphx::shape shape = result.get_shape();
        const auto type = type_of(shape.type());
        if (!type) return refuse(RefusalCode::FormatUnsupported, "sortie MIGraphX d'un type inattendu");
        const std::size_t count = shape.elements();
        if (count * (outputs[i].type == TensorType::F32 ? 4U : 2U) != outputs[i].bytes) {
          return refuse(RefusalCode::InputMismatch,
                        std::format("{} : {} éléments", names_.outputs[i], count));
        }
        if (auto ok = vk::net::copy_converted(reinterpret_cast<const std::byte*>(result.data()), *type,
                                              outputs[i].data, outputs[i].type, count);
            !ok) {
          return ok;
        }
      }
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("MIGraphX, exécution : {}", e.what()));
    }
  }

 private:
  int device_;
  migraphx::program program_;
  vk::net::IoNames names_;
  std::vector<std::vector<std::byte>> staged_;  // entrées converties, gardées d'un appel à l'autre
};

class MigraphxRuntime final : public vk::net::Runtime {
 public:
  MigraphxRuntime(int device, std::string device_name)
      : device_(device), device_name_(std::move(device_name)) {}
  std::string_view name() const override { return "migraphx"; }
  std::string device_name() const override { return device_name_; }
  std::string version() const override {
    return std::format("{}.{}.{}", MIGRAPHX_VERSION_MAJOR, MIGRAPHX_VERSION_MINOR, MIGRAPHX_VERSION_PATCH);
  }
  std::string_view extension() const override { return ".mxr"; }

  std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                     const std::filesystem::path& path, Progress& progress) override {
    try {
      if (hipSetDevice(device_) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
      if (progress.report) progress.report(0.05F, "analyse ONNX (MIGraphX)");
      migraphx::onnx_options options;
      for (const InputShape& input : inputs) {
        options.set_input_parameter_shape(input.name,
                                          std::vector<std::size_t>(input.dims.begin(), input.dims.end()));
      }
      migraphx::program program = migraphx::parse_onnx(model.onnx_path.string().c_str(), options);
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
      // Un ONNX en float demandé en F16 : MIGraphX range ses calculs en half, entrées comprises
      // converties dans le programme.
      bool float_graph = false;
      const migraphx::program_parameter_shapes shapes = program.get_parameter_shapes();
      for (const char* name : shapes.names()) float_graph |= shapes[name].type() == migraphx_shape_float_type;
      if (model.precision == TensorType::F16 && float_graph) migraphx::quantize_fp16(program);
      if (progress.report) progress.report(0.15F, "compilation MIGraphX");
      migraphx::compile_options compile;
      compile.set_offload_copy(true);
      compile.set_fast_math(true);
      // Réglage exhaustif des noyaux (ROCm 6) : quelques minutes à la première construction, puis
      // gardé dans le .mxr.
      compile.set_exhaustive_tune_flag(true);
      program.compile(migraphx::target("gpu"), compile);
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
      migraphx::save(program, path.string().c_str());
      if (progress.report) progress.report(1.0F, "programme MIGraphX écrit");
      return {};
    } catch (const std::exception& e) {
      return refuse_exception(model.id, e);
    }
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    try {
      if (hipSetDevice(device_) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
      auto names = vk::net::read_io_names(path);
      if (!names) return std::unexpected(names.error());
      migraphx::program program = migraphx::load(path.string().c_str());
      const migraphx::program_parameter_shapes shapes = program.get_parameter_shapes();
      const std::vector<const char*> known = shapes.names();
      for (const std::string& name : names->inputs) {
        if (std::ranges::none_of(known, [&](const char* k) { return name == k; }))
          return refuse(RefusalCode::CacheIo,
                        std::format("{} : entrée {} absente du programme", path.string(), name));
      }
      return std::make_unique<MigraphxNet>(device_, std::move(program), std::move(*names));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("MIGraphX, {} : {}", path.string(), e.what()));
    }
  }

 private:
  int device_;
  std::string device_name_;
};

}  // namespace

std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  if (caps.vendor_id != kVendorAmd) {
    return refuse(RefusalCode::DeviceMismatch,
                  std::format("MIGraphX sert les GPU AMD ; {} : backend Vulkan seul", caps.device_name));
  }
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0)
    return refuse(RefusalCode::DeviceUnavailable, "aucun GPU ROCm (HIP) visible");
  // Le GPU HIP de même UUID que l'appareil Vulkan, sinon le premier.
  int chosen = 0;
  for (int i = 0; i < count; ++i) {
    hipDevice_t device{};
    hipUUID uuid{};
    if (hipDeviceGet(&device, i) == hipSuccess && hipDeviceGetUuid(&uuid, device) == hipSuccess &&
        std::memcmp(uuid.bytes, caps.uuid.data(), caps.uuid.size()) == 0) {
      chosen = i;
      break;
    }
  }
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, chosen) != hipSuccess)
    return refuse(RefusalCode::DeviceUnavailable, "hipGetDeviceProperties");
  return std::make_shared<MigraphxRuntime>(chosen, std::format("{} ({})", props.name, props.gcnArchName));
}

}  // namespace ajp::gpu::migraphx_rt
