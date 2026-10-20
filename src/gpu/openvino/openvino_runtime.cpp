#include "gpu/openvino/openvino_runtime.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <map>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/version.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace ajp::gpu::openvino_rt {
namespace {

using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorIntel = 0x8086;

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

ov::element::Type element_of(TensorType type) {
  return type == TensorType::F32 ? ov::element::f32 : ov::element::f16;
}

ov::AnyMap compile_config() {
  return {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
          ov::hint::inference_precision(ov::element::f16), ov::hint::num_requests(1)};
}

class OpenvinoNet final : public vk::net::RuntimeNet {
 public:
  OpenvinoNet(ov::CompiledModel compiled, vk::net::IoNames names)
      : compiled_(std::move(compiled)),
        request_(compiled_.create_infer_request()),
        names_(std::move(names)) {}
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "OpenVINO : nombre d'entrées ou de sorties");
    try {
      // Les tenseurs enveloppent la mémoire du lecteur, sans copie côté hôte.
      const auto wrap = [](const HostBinding& b) {
        ov::Shape shape;
        for (const std::int64_t d : b.shape.dims) shape.push_back(static_cast<std::size_t>(d));
        return ov::Tensor(element_of(b.type), shape, b.data);
      };
      for (std::size_t i = 0; i < inputs.size(); ++i)
        request_.set_tensor(compiled_.input(names_.inputs[i]), wrap(inputs[i]));
      for (std::size_t i = 0; i < outputs.size(); ++i)
        request_.set_tensor(compiled_.output(names_.outputs[i]), wrap(outputs[i]));
      request_.infer();
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("OpenVINO, exécution : {}", e.what()));
    }
  }

 private:
  ov::CompiledModel compiled_;
  ov::InferRequest request_;
  vk::net::IoNames names_;
};

class OpenvinoRuntime final : public vk::net::Runtime {
 public:
  OpenvinoRuntime(std::shared_ptr<ov::Core> core, std::string device, std::string full_name)
      : core_(std::move(core)), device_(std::move(device)), full_name_(std::move(full_name)) {}
  std::string_view name() const override { return "openvino"; }
  std::string device_name() const override { return std::format("{} ({})", full_name_, device_); }
  std::string version() const override {
    const ov::Version v = ov::get_openvino_version();
    return v.buildNumber != nullptr ? std::string(v.buildNumber) : std::string("?");
  }
  std::string_view extension() const override { return ".ovblob"; }

  std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                     const std::filesystem::path& path, Progress& progress) override {
    try {
      if (progress.report) progress.report(0.05F, "lecture ONNX (OpenVINO)");
      std::shared_ptr<ov::Model> graph = core_->read_model(model.onnx_path.string());
      if (!inputs.empty()) {
        std::map<std::string, ov::PartialShape> shapes;
        for (const InputShape& input : inputs) {
          std::vector<ov::Dimension> dims;
          for (const std::int64_t d : input.dims) dims.emplace_back(d);
          shapes[input.name] = ov::PartialShape(dims);
        }
        graph->reshape(shapes);
      }
      // Entrées et sorties dans le type des tenseurs du lecteur : la conversion se fait sur le GPU.
      ov::preprocess::PrePostProcessor ppp(graph);
      const ov::element::Type io = element_of(model.precision);
      for (std::size_t i = 0; i < graph->inputs().size(); ++i) ppp.input(i).tensor().set_element_type(io);
      for (std::size_t i = 0; i < graph->outputs().size(); ++i) ppp.output(i).tensor().set_element_type(io);
      graph = ppp.build();
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
      if (progress.report) progress.report(0.15F, "compilation OpenVINO");
      ov::CompiledModel compiled = core_->compile_model(graph, device_, compile_config());
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      compiled.export_model(out);
      if (!out) return refuse(RefusalCode::CacheIo, std::format("{} : écriture", path.string()));
      if (progress.report) progress.report(1.0F, "modèle OpenVINO écrit");
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed, std::format("OpenVINO, {} : {}", model.id, e.what()));
    }
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    try {
      std::ifstream in(path, std::ios::binary);
      if (!in) return refuse(RefusalCode::CacheIo, std::format("{} : lecture", path.string()));
      ov::CompiledModel compiled = core_->import_model(in, device_, compile_config());
      return std::make_unique<OpenvinoNet>(std::move(compiled), std::move(*names));
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("OpenVINO, {} : {}", path.string(), e.what()));
    }
  }

 private:
  std::shared_ptr<ov::Core> core_;
  std::string device_;
  std::string full_name_;
};

}  // namespace

std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  if (caps.vendor_id != kVendorIntel) {
    return refuse(RefusalCode::DeviceMismatch,
                  std::format("OpenVINO sert les GPU Intel ; {} : backend Vulkan seul", caps.device_name));
  }
  try {
    auto core = std::make_shared<ov::Core>();
    // Le GPU OpenVINO de même UUID que l'appareil Vulkan (« GPU.0 » l'intégré, « GPU.1 » une
    // carte Arc), sinon le premier.
    std::string chosen;
    for (const std::string& device : core->get_available_devices()) {
      if (!device.starts_with("GPU")) continue;
      if (chosen.empty()) chosen = device;
      try {
        const ov::device::UUID uuid = core->get_property(device, ov::device::uuid);
        if (std::memcmp(uuid.uuid.data(), caps.uuid.data(), caps.uuid.size()) == 0) {
          chosen = device;
          break;
        }
      } catch (const std::exception&) {
        // propriété absente de ce pilote : on garde le premier GPU
      }
    }
    if (chosen.empty())
      return refuse(RefusalCode::DeviceUnavailable, "aucun GPU OpenVINO (pilote compute Intel)");
    const std::string full = core->get_property(chosen, ov::device::full_name);
    return std::make_shared<OpenvinoRuntime>(std::move(core), chosen, full);
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("OpenVINO : {}", e.what()));
  }
}

}  // namespace ajp::gpu::openvino_rt
