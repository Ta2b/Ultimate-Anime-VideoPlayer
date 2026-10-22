#include "gpu/openvino/openvino_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <fstream>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/version.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu/openvino/openvino_import.h"
#include "gpu/openvino/openvino_net.h"
#include "gpu/vulkan/backend/net_measure.h"
#include "gpu/vulkan/compute/caps.h"

namespace ajp::gpu::openvino_rt {
namespace {

using vk::net::InputShape;

constexpr std::uint32_t kVendorIntel = 0x8086;

// Les réglages que l'appareil connaît (supported_properties), les autres écartés : un pilote ou
// un NPU plus ancien refuserait la compilation entière pour une propriété inconnue.
ov::AnyMap supported_only(ov::Core& core, const std::string& device, ov::AnyMap wanted) {
  const std::vector<ov::PropertyName> supported = core.get_property(device, ov::supported_properties);
  for (auto it = wanted.begin(); it != wanted.end();) {
    const bool known =
        std::ranges::any_of(supported, [&](const ov::PropertyName& p) { return p == it->first; });
    it = known ? std::next(it) : wanted.erase(it);
  }
  return wanted;
}

// GPU (Arc, Xe intégré) : latence d'une requête à la fois, en f16 ; la file du GPU en priorité
// ordinaire, celle de l'affichage : en priorité haute, chaque inférence passait devant le rendu
// des images déjà prêtes, qui manquait alors des rafraîchissements pendant tout le calcul (le
// GPU alterne les contextes de même priorité par tranches de temps) ; sans ralenti ; les tâches
// hôte du greffon sur les cœurs performants d'un processeur hybride (Core Ultra).
ov::AnyMap gpu_config(ov::Core& core, const std::string& device) {
  return supported_only(core, device,
                        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                         ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                         ov::hint::inference_precision(ov::element::f16), ov::hint::num_requests(1),
                         ov::intel_gpu::hint::queue_priority(ov::hint::Priority::MEDIUM),
                         ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::HIGH),
                         ov::intel_gpu::hint::host_task_priority(ov::hint::Priority::HIGH)});
}

// NPU (Core Ultra) : latence, f16, et la fréquence haute du NPU (turbo) quand le pilote l'offre.
ov::AnyMap npu_config(ov::Core& core, const std::string& device) {
  return supported_only(core, device,
                        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                         ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                         ov::hint::inference_precision(ov::element::f16), ov::hint::num_requests(1),
                         ov::intel_npu::turbo(true)});
}

// L'appareil d'un artefact, rangé à côté de lui : « GPU.0 », « NPU ». Absent (artefact d'une
// version qui ne le rangeait pas) : le GPU.
std::filesystem::path device_path(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".device";
  return p;
}

// L'image de test de l'entrée (vk::net::probe_image, décalée de shift), la même pour le GPU et le
// NPU.
ov::Tensor probe_input(const ov::Output<const ov::Node>& input, int shift) {
  ov::Tensor tensor(input.get_element_type(), input.get_shape());
  std::vector<std::int64_t> dims;
  dims.reserve(input.get_shape().size());
  for (const std::size_t d : input.get_shape()) dims.push_back(static_cast<std::int64_t>(d));
  const std::vector<float> values = vk::net::probe_image(dims, shift);
  const TensorType type = input.get_element_type() == ov::element::f32 ? TensorType::F32 : TensorType::F16;
  if (auto ok = vk::net::copy_converted(reinterpret_cast<const std::byte*>(values.data()), TensorType::F32,
                                        static_cast<std::byte*>(tensor.data()), type, values.size());
      !ok)
    throw std::runtime_error(ok.error().detail);
  return tensor;
}

// Le tenseur par valeur : la poignée se partage, et data() const est déprécié depuis OpenVINO 2025.3.
// Lève std::runtime_error, comme OpenVINO : try_npu les rattrape.
std::vector<float> as_floats(ov::Tensor tensor) {
  std::vector<float> out(tensor.get_size());
  const TensorType type = tensor.get_element_type() == ov::element::f32 ? TensorType::F32 : TensorType::F16;
  if (auto ok =
          vk::net::copy_converted(static_cast<const std::byte*>(tensor.data()), type,
                                  reinterpret_cast<std::byte*>(out.data()), TensorType::F32, out.size());
      !ok)
    throw std::runtime_error(ok.error().detail);
  return out;
}

struct Measure {
  double ms = 0.0;               // médiane d'une inférence
  std::vector<float> first_out;  // première sortie, pour comparer deux appareils
};

// Chaque entrée a son décalage de l'image de test : les deux images d'une paire diffèrent.
Measure measure(ov::CompiledModel& compiled) {
  ov::InferRequest request = compiled.create_infer_request();
  int shift = 0;
  for (const ov::Output<const ov::Node>& input : compiled.inputs())
    request.set_tensor(input, probe_input(input, shift++));
  for (int i = 0; i < vk::net::kUnitWarmupRuns; ++i) request.infer();
  std::array<double, vk::net::kUnitTimedRuns> times{};
  for (double& t : times) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  return {.ms = vk::net::median(times),
          .first_out = as_floats(request.get_tensor(compiled.outputs().front()))};
}

// Le NPU de la puce, s'il y en a un et que son pilote répond.
struct Npu {
  std::string device;  // « NPU »
  std::string label;   // « Intel(R) AI Boost, pilote 1.10 »
};

class OpenvinoRuntime final : public vk::net::Runtime {
 public:
  OpenvinoRuntime(std::shared_ptr<ov::Core> core, std::string gpu, std::string full_name,
                  std::optional<Npu> npu)
      : core_(std::move(core)), gpu_(std::move(gpu)), full_name_(std::move(full_name)), npu_(std::move(npu)) {
    try {
      const ov::RemoteContext context = core_->get_default_context(gpu_);
      usm_ = std::make_shared<UsmPool>(context);
      // La mémoire de la chaîne ne s'importe que sur une carte : sur un GPU intégré, la mémoire hôte
      // du runtime (USM) est déjà partagée, sans copie.
      if (core_->get_property(gpu_, ov::device::type) == ov::device::Type::DISCRETE)
        imports_ = DeviceImports::open(context);
    } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch) : contexte du GPU indisponible,
      // les tenseurs passent par la mémoire du backend
    }
  }
  std::string_view name() const override { return "openvino"; }
  // Entre dans la clé des artefacts : un NPU qui apparaît (pilote installé) ou change de pilote
  // fait reconstruire, et la mesure choisit à nouveau. En mémoire d'appareil le NPU ne sert pas
  // (il ne lit pas la mémoire de la carte) : la clé le dit.
  std::string device_name() const override {
    const bool npu = npu_ && !device_memory_.load();
    return std::format("{} ({}){}", full_name_, gpu_, npu ? std::format(" + NPU {}", npu_->label) : "");
  }
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
          dims.reserve(input.dims.size());
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
      if (progress.report) progress.report(0.15F, "compilation OpenVINO (GPU)");
      ov::CompiledModel compiled = core_->compile_model(graph, gpu_, gpu_config(*core_, gpu_));
      std::string device = gpu_;
      // En mémoire d'appareil, le réseau reste sur le GPU : le NPU ne lit pas cette mémoire.
      if (npu_ && !device_memory_.load()) {
        const std::string npu_device = npu_->device;
        auto warps = vk::net::onnx_uses_op(model.onnx_path, "GridSample");
        if (!warps) return std::unexpected(warps.error());
        // RIFE, GMFSS et DRBA déforment par GridSample, que le NPU ne fait pas : ils restent sur le GPU.
        if (!*warps) {
          if (auto chosen = try_npu(model, graph, compiled, progress)) {
            compiled = std::move(*chosen);
            device = npu_device;
          }
        }
      }
      if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      compiled.export_model(out);
      if (!out) return refuse(RefusalCode::CacheIo, std::format("{} : écriture", path.string()));
      std::ofstream tag(device_path(path), std::ios::trunc);
      tag << device << '\n';
      if (!tag) return refuse(RefusalCode::CacheIo, std::format("{} : écriture", device_path(path).string()));
      if (progress.report) progress.report(1.0F, std::format("modèle OpenVINO écrit ({})", device));
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::BuilderFailed, std::format("OpenVINO, {} : {}", model.id, e.what()));
    }
  }

  // Une mémoire USM hôte du GPU (alignée sur la page par le pilote, que l'appelant vérifie).
  // Interop sur une carte Arc : la chaîne exporte sa mémoire d'appareil (dma-buf sous Linux, HANDLE
  // NT sous Windows) et OpenVINO la lit et l'écrit en place. Les réseaux lient chaque tenseur selon
  // ce qu'il est (openvino_net.cpp), donc un import refusé laisse les autres à la mémoire hôte.
  bool imports_memory() const override { return imports_ != nullptr; }
  vk::net::MemoryHandle memory_handle() const override {
#if defined(_WIN32)
    return vk::net::MemoryHandle::OpaqueWin32;
#else
    return vk::net::MemoryHandle::DmaBuf;
#endif
  }
  bool mixes_memory() const override { return true; }
  void use_device_memory(bool on) override { device_memory_.store(on); }
  std::expected<std::unique_ptr<vk::net::ImportedMemory>, Refusal> import_memory(
      vk::net::HandleValue handle, std::uint64_t /*bytes*/, TensorType type, const Shape& shape) override {
    if (!imports_) return refuse(RefusalCode::FeatureUnavailable, "OpenVINO : pas d'import sur ce GPU");
    return imports_->import(handle, type, shape);
  }

  std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_host(
      std::uint64_t bytes, std::uint64_t /*alignment*/, TensorType /*type*/,
      const Shape& /*shape*/) override {
    if (!usm_) return refuse(RefusalCode::FeatureUnavailable, "OpenVINO : contexte du GPU indisponible");
    try {
      auto [tensor, data] = usm_->allocate(bytes);
      return std::make_unique<UsmMemory>(usm_, std::move(tensor), data);
    } catch (const std::exception& e) {
      return refuse(RefusalCode::FeatureUnavailable, std::format("OpenVINO, mémoire USM : {}", e.what()));
    }
  }

  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    try {
      std::string device = gpu_;
      if (std::ifstream tag(device_path(path)); tag) std::getline(tag, device);
      const bool on_npu = npu_ && device == npu_->device && !device_memory_.load();
      if (!on_npu) device = gpu_;  // un GPU renuméroté garde son artefact (la clé couvre le modèle)
      std::ifstream in(path, std::ios::binary);
      if (!in) return refuse(RefusalCode::CacheIo, std::format("{} : lecture", path.string()));
      ov::CompiledModel compiled =
          core_->import_model(in, device, on_npu ? npu_config(*core_, device) : gpu_config(*core_, device));
      return make_net(std::move(compiled), std::move(*names), on_npu ? nullptr : usm_,
                      on_npu ? nullptr : imports_);
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("OpenVINO, {} : {}", path.string(), e.what()));
    }
  }

 private:
  // Le réseau compilé pour le NPU s'il y tient (vk::net::choose_unit, le GPU en référence) :
  // compilé sans erreur, aussi juste que le GPU sur la même entrée, et pas plus lent que
  // kUnitTolerance fois le GPU. Sinon rien : le GPU le garde.
  std::optional<ov::CompiledModel> try_npu(const ModelSpec& model, const std::shared_ptr<ov::Model>& graph,
                                           ov::CompiledModel& on_gpu, Progress& progress) {
    const auto say = [&](float fraction, const std::string& text) {
      if (progress.report) progress.report(fraction, text);
    };
    if (!npu_) return std::nullopt;
    const std::string device = npu_->device;
    try {
      say(0.40F, "compilation OpenVINO (NPU)");
      ov::CompiledModel on_npu = core_->compile_model(graph, device, npu_config(*core_, device));
      if (progress.stop.stop_requested()) return std::nullopt;
      say(0.75F, "mesure du GPU et du NPU");
      const Measure gpu = measure(on_gpu);
      const Measure npu = measure(on_npu);
      const double error = vk::net::mean_error(gpu.first_out, npu.first_out);
      const std::array<std::optional<vk::net::UnitMeasure>, 2> units{
          vk::net::UnitMeasure{.ms = gpu.ms}, vk::net::UnitMeasure{.ms = npu.ms, .error = error}};
      const bool kept = vk::net::choose_unit(units, 1) == 1U;
      say(0.95F, std::format("{} : NPU {:.1f} ms, GPU {:.1f} ms, écart {:.4f} : {}", model.id, npu.ms, gpu.ms,
                             error, kept ? "NPU" : "GPU"));
      if (!kept) return std::nullopt;
      return on_npu;
    } catch (const std::exception& e) {
      say(0.95F, std::format("{} : NPU refusé ({}), GPU", model.id, e.what()));
      return std::nullopt;
    }
  }

  std::shared_ptr<ov::Core> core_;
  std::string gpu_;
  std::string full_name_;
  std::optional<Npu> npu_;
  std::shared_ptr<UsmPool> usm_;             // nul sans contexte GPU
  std::shared_ptr<DeviceImports> imports_;   // nul hors carte dédiée
  std::atomic<bool> device_memory_ = false;  // le backend lie la mémoire de la chaîne (interop)
};

// Le NPU de la puce (Core Ultra), s'il y en a un : son nom et la version de son pilote, qui
// entrent dans la clé des artefacts.
std::optional<Npu> find_npu(ov::Core& core) {
  for (const std::string& device : core.get_available_devices()) {
    if (!device.starts_with("NPU")) continue;
    try {
      std::string label = core.get_property(device, ov::device::full_name);
      try {
        const std::uint32_t driver = core.get_property(device, ov::intel_npu::driver_version);
        label += std::format(", pilote {}", driver);
      } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch) : propriété absente de ce
        // greffon, le nom suffit
      }
      return Npu{.device = device, .label = std::move(label)};
    } catch (const std::exception&) {
      return std::nullopt;  // NPU listé mais sans pilote qui réponde
    }
  }
  return std::nullopt;
}

}  // namespace

std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  if (caps.vendor_id != kVendorIntel) {
    return refuse(RefusalCode::DeviceMismatch,
                  std::format("OpenVINO sert les GPU Intel ; {} : backend Vulkan seul", caps.device_name));
  }
  try {
    auto core = std::make_shared<ov::Core>();
    // Le GPU OpenVINO de l'appareil Vulkan (« GPU.0 » l'intégré, « GPU.1 » une carte Arc ;
    // vk::match_device : même UUID, sinon même bus PCI), sinon le premier.
    std::vector<std::string> gpus;
    std::vector<vk::DeviceIdentity> seen;
    for (const std::string& device : core->get_available_devices()) {
      if (!device.starts_with("GPU")) continue;
      vk::DeviceIdentity& identity = seen.emplace_back();
      gpus.push_back(device);
      try {
        identity.uuid = core->get_property(device, ov::device::uuid).uuid;
      } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch) : propriété absente de ce
        // pilote, le bus PCI peut encore servir
      }
      try {
        const ov::device::PCIInfo pci = core->get_property(device, ov::device::pci_info);
        identity.pci = vk::PciAddress{
            .domain = pci.domain, .bus = pci.bus, .device = pci.device, .function = pci.function};
      } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch) : propriété absente de ce
        // pilote, l'UUID seul
      }
    }
    if (gpus.empty())
      return refuse(RefusalCode::DeviceUnavailable, "aucun GPU OpenVINO (pilote compute Intel)");
    const std::string chosen = gpus[vk::match_device(seen, caps).value_or(0)];
    const std::string full = core->get_property(chosen, ov::device::full_name);
    std::optional<Npu> npu = find_npu(*core);
    return std::make_shared<OpenvinoRuntime>(std::move(core), chosen, full, std::move(npu));
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("OpenVINO : {}", e.what()));
  }
}

}  // namespace ajp::gpu::openvino_rt
