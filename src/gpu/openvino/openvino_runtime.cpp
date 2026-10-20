#include "gpu/openvino/openvino_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/version.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ajp::gpu::openvino_rt {
namespace {

using vk::net::HostBinding;
using vk::net::InputShape;

constexpr std::uint32_t kVendorIntel = 0x8086;

// Mesure d'un modèle compilé : inférences de chauffe, puis médiane des suivantes.
constexpr int kWarmupRuns = 2;
constexpr std::size_t kTimedRuns = 5;
// Le NPU garde un réseau s'il ne coûte pas plus que ce facteur du temps du GPU : il laisse le
// GPU, partagé avec la chaîne Vulkan, RIFE, le décodage et le rendu, à tout le reste.
constexpr double kNpuTolerance = 1.25;
// Écart moyen permis entre les sorties du NPU et du GPU sur la même entrée (images entre 0 et 1) :
// au-delà, le compilateur du NPU a perdu en précision et le GPU garde le réseau.
constexpr double kNpuMaxMeanError = 0.01;

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

ov::element::Type element_of(TensorType type) {
  return type == TensorType::F32 ? ov::element::f32 : ov::element::f16;
}

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
// haute et sans ralenti, pour ne pas passer derrière le rendu du lecteur sur le même GPU ; les
// tâches hôte du greffon sur les cœurs performants d'un processeur hybride (Core Ultra).
ov::AnyMap gpu_config(ov::Core& core, const std::string& device) {
  return supported_only(core, device,
                        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                         ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                         ov::hint::inference_precision(ov::element::f16), ov::hint::num_requests(1),
                         ov::intel_gpu::hint::queue_priority(ov::hint::Priority::HIGH),
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

// Une entrée fixe et reproductible entre 0 et 1, la même pour le GPU et le NPU.
ov::Tensor probe_input(const ov::Output<const ov::Node>& input) {
  ov::Tensor tensor(input.get_element_type(), input.get_shape());
  std::vector<float> values(tensor.get_size());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>((i * 2654435761U) % 1024U) / 1023.0F;
  const TensorType type = input.get_element_type() == ov::element::f32 ? TensorType::F32 : TensorType::F16;
  static_cast<void>(vk::net::copy_converted(reinterpret_cast<const std::byte*>(values.data()),
                                            TensorType::F32, static_cast<std::byte*>(tensor.data()), type,
                                            values.size()));
  return tensor;
}

std::vector<float> as_floats(const ov::Tensor& tensor) {
  std::vector<float> out(tensor.get_size());
  const TensorType type = tensor.get_element_type() == ov::element::f32 ? TensorType::F32 : TensorType::F16;
  static_cast<void>(vk::net::copy_converted(static_cast<const std::byte*>(tensor.data()), type,
                                            reinterpret_cast<std::byte*>(out.data()), TensorType::F32,
                                            out.size()));
  return out;
}

struct Measure {
  double ms = 0.0;               // médiane d'une inférence
  std::vector<float> first_out;  // première sortie, pour comparer deux appareils
};

Measure measure(ov::CompiledModel& compiled) {
  ov::InferRequest request = compiled.create_infer_request();
  for (const ov::Output<const ov::Node>& input : compiled.inputs())
    request.set_tensor(input, probe_input(input));
  for (int i = 0; i < kWarmupRuns; ++i) request.infer();
  std::array<double, kTimedRuns> times{};
  for (double& t : times) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  std::ranges::nth_element(times, times.begin() + kTimedRuns / 2);
  return {.ms = times[kTimedRuns / 2],
          .first_out = as_floats(request.get_tensor(compiled.outputs().front()))};
}

double mean_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return INFINITY;
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
  return sum / static_cast<double>(a.size());
}

// Les mémoires USM hôte du contexte GPU (Runtime::allocate_host), que Vulkan importe sur un GPU
// intégré : le greffon GPU les lit et les écrit sans copie. Les réseaux les reconnaissent à leur
// adresse et les lui donnent pour ce qu'elles sont (USM_USER_BUFFER), pas comme mémoire hôte
// ordinaire, qu'il recopierait.
class UsmPool {
 public:
  explicit UsmPool(ov::RemoteContext context) : context_(std::move(context)) {}

  // Lève les exceptions d'OpenVINO (pilote sans USM).
  std::pair<ov::RemoteTensor, std::byte*> allocate(std::uint64_t bytes) {
    ov::RemoteTensor tensor = context_.create_tensor(
        ov::element::u8, {static_cast<std::size_t>(bytes)},
        {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
    auto* data = static_cast<std::byte*>(
        tensor.get_params().at(ov::intel_gpu::mem_handle.name()).as<ov::intel_gpu::gpu_handle_param>());
    const std::scoped_lock lock(mutex_);
    blocks_[data] = bytes;
    return {std::move(tensor), data};
  }
  void forget(const std::byte* data) {
    const std::scoped_lock lock(mutex_);
    blocks_.erase(data);
  }
  // L'adresse est-elle dans un bloc USM vivant (les tenseurs de la chaîne y ont un décalage) ?
  bool owns(const std::byte* data, std::uint64_t bytes) const {
    const std::scoped_lock lock(mutex_);
    auto it = blocks_.upper_bound(data);
    if (it == blocks_.begin()) return false;
    --it;
    return data + bytes <= it->first + it->second;
  }
  ov::RemoteTensor wrap(ov::element::Type type, const ov::Shape& shape, std::byte* data) {
    return context_.create_tensor(
        type, shape,
        {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
         {ov::intel_gpu::mem_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(data)}});
  }

 private:
  ov::RemoteContext context_;
  mutable std::mutex mutex_;
  std::map<const std::byte*, std::uint64_t> blocks_;  // début, taille
};

class UsmMemory final : public vk::net::HostMemory {
 public:
  UsmMemory(std::shared_ptr<UsmPool> pool, ov::RemoteTensor tensor, std::byte* data)
      : pool_(std::move(pool)), tensor_(std::move(tensor)), data_(data) {}
  UsmMemory(const UsmMemory&) = delete;
  UsmMemory& operator=(const UsmMemory&) = delete;
  UsmMemory(UsmMemory&&) = delete;
  UsmMemory& operator=(UsmMemory&&) = delete;
  ~UsmMemory() override { pool_->forget(data_); }
  std::byte* data() const override { return data_; }

 private:
  std::shared_ptr<UsmPool> pool_;
  ov::RemoteTensor tensor_;  // possède la mémoire
  std::byte* data_;
};

class OpenvinoNet final : public vk::net::RuntimeNet {
 public:
  // usm : les mémoires USM du GPU, pour un réseau compilé sur ce GPU ; nul sur le NPU, qui lit
  // toute mémoire hôte de la même façon.
  OpenvinoNet(ov::CompiledModel compiled, vk::net::IoNames names, std::shared_ptr<UsmPool> usm)
      : compiled_(std::move(compiled)),
        request_(compiled_.create_infer_request()),
        names_(std::move(names)),
        usm_(std::move(usm)),
        bound_(names_.inputs.size() + names_.outputs.size()) {}
  OpenvinoNet(const OpenvinoNet&) = delete;
  OpenvinoNet& operator=(const OpenvinoNet&) = delete;
  OpenvinoNet(OpenvinoNet&&) = delete;
  OpenvinoNet& operator=(OpenvinoNet&&) = delete;
  // Les étapes encore en file sont faites avant de rendre la requête : chacune signale la
  // timeline que la chaîne attend.
  ~OpenvinoNet() override {
    if (!worker_.joinable()) return;
    worker_.request_stop();
    wake_.notify_all();
    worker_.join();
  }
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "OpenVINO : nombre d'entrées ou de sorties");
    const std::scoped_lock lock(request_mutex_);
    return infer(inputs, outputs);
  }

  // Sur un fil propre au réseau, dans l'ordre des appels : ready (l'attente des entrées sur la
  // timeline Vulkan), l'inférence, done (le signal de l'étape). Le fil du lecteur n'attend
  // plus le GPU Intel, et le NPU calcule pendant que le GPU fait la chaîne. Une seule requête :
  // la timeline ordonne les étapes, et les entrées de la suivante ne sont écrites qu'après le
  // signal de la précédente ; une seconde requête l'attendrait de la même façon.
  std::expected<void, Refusal> run_deferred(std::span<const HostBinding> inputs,
                                            std::span<const HostBinding> outputs,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "OpenVINO : nombre d'entrées ou de sorties");
    if (!ready || !done) return refuse(RefusalCode::InputMismatch, "OpenVINO : étape sans rappels");
    {
      const std::scoped_lock lock(jobs_mutex_);
      jobs_.push_back(Job{.inputs = {inputs.begin(), inputs.end()},
                          .outputs = {outputs.begin(), outputs.end()},
                          .ready = std::move(ready),
                          .done = std::move(done)});
      if (!worker_.joinable()) worker_ = std::jthread([this](const std::stop_token& stop) { serve(stop); });
    }
    wake_.notify_one();
    return {};
  }

 private:
  struct Bound {
    std::byte* data = nullptr;
    TensorType type = TensorType::F16;
    Shape shape;
  };
  struct Job {
    std::vector<HostBinding> inputs;
    std::vector<HostBinding> outputs;
    std::function<std::expected<void, Refusal>()> ready;
    std::function<void(std::expected<void, Refusal>)> done;
  };

  std::expected<void, Refusal> infer(std::span<const HostBinding> inputs,
                                     std::span<const HostBinding> outputs) {
    try {
      for (std::size_t i = 0; i < inputs.size(); ++i) bind(i, compiled_.input(names_.inputs[i]), inputs[i]);
      for (std::size_t i = 0; i < outputs.size(); ++i)
        bind(inputs.size() + i, compiled_.output(names_.outputs[i]), outputs[i]);
      request_.infer();
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("OpenVINO, exécution : {}", e.what()));
    }
  }

  // Le fil des étapes : chaque étape en file est faite, même à l'arrêt (sa timeline l'attend).
  void serve(const std::stop_token& stop) {
    while (true) {
      std::optional<Job> job;
      {
        std::unique_lock lock(jobs_mutex_);
        wake_.wait(lock, [&] { return !jobs_.empty() || stop.stop_requested(); });
        if (jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      std::expected<void, Refusal> result = job->ready();
      if (result) {
        const std::scoped_lock lock(request_mutex_);
        result = infer(job->inputs, job->outputs);
      }
      job->done(std::move(result));
    }
  }

  // Les tenseurs enveloppent la mémoire du lecteur, sans copie côté hôte, et restent liés tant
  // que la chaîne donne les mêmes : le greffon ne revérifie pas la mémoire à chaque image.
  void bind(std::size_t slot, const ov::Output<const ov::Node>& port, const HostBinding& b) {
    Bound& last = bound_[slot];
    if (last.data == b.data && last.type == b.type && last.shape == b.shape) return;
    ov::Shape shape;
    for (const std::int64_t d : b.shape.dims) shape.push_back(static_cast<std::size_t>(d));
    if (usm_ && usm_->owns(b.data, b.bytes)) {
      request_.set_tensor(port, usm_->wrap(element_of(b.type), shape, b.data));
    } else {
      request_.set_tensor(port, ov::Tensor(element_of(b.type), shape, b.data));
    }
    last = {.data = b.data, .type = b.type, .shape = b.shape};
  }

  ov::CompiledModel compiled_;
  ov::InferRequest request_;
  vk::net::IoNames names_;
  std::shared_ptr<UsmPool> usm_;
  std::vector<Bound> bound_;
  std::mutex request_mutex_;  // request_ et bound_ : le fil des étapes ou celui de run
  std::mutex jobs_mutex_;
  std::condition_variable wake_;
  std::deque<Job> jobs_;
  std::jthread worker_;
};

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
      usm_ = std::make_shared<UsmPool>(core_->get_default_context(gpu_));
    } catch (const std::exception&) {
      // contexte du GPU indisponible : les tenseurs passent par la mémoire du backend
    }
  }
  std::string_view name() const override { return "openvino"; }
  // Entre dans la clé des artefacts : un NPU qui apparaît (pilote installé) ou change de pilote
  // fait reconstruire, et la mesure choisit à nouveau.
  std::string device_name() const override {
    return std::format("{} ({}){}", full_name_, gpu_, npu_ ? std::format(" + NPU {}", npu_->label) : "");
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
      if (npu_) {
        auto warps = vk::net::onnx_uses_op(model.onnx_path, "GridSample");
        if (!warps) return std::unexpected(warps.error());
        // RIFE, GMFSS et DRBA déforment par GridSample, que le NPU ne fait pas : ils restent sur le GPU.
        if (!*warps) {
          if (auto chosen = try_npu(model, graph, compiled, progress)) {
            compiled = std::move(*chosen);
            device = npu_->device;
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
  std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_host(
      std::uint64_t bytes, std::uint64_t /*alignment*/) override {
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
      const bool on_npu = npu_ && device == npu_->device;
      if (!on_npu) device = gpu_;  // un GPU renuméroté garde son artefact (la clé couvre le modèle)
      std::ifstream in(path, std::ios::binary);
      if (!in) return refuse(RefusalCode::CacheIo, std::format("{} : lecture", path.string()));
      ov::CompiledModel compiled =
          core_->import_model(in, device, on_npu ? npu_config(*core_, device) : gpu_config(*core_, device));
      return std::make_unique<OpenvinoNet>(std::move(compiled), std::move(*names), on_npu ? nullptr : usm_);
    } catch (const std::exception& e) {
      return refuse(RefusalCode::CacheIo, std::format("OpenVINO, {} : {}", path.string(), e.what()));
    }
  }

 private:
  // Le réseau compilé pour le NPU s'il y tient : compilé sans erreur, aussi juste que le GPU sur
  // la même entrée, et pas plus lent que kNpuTolerance fois le GPU. Sinon rien : le GPU le garde.
  std::optional<ov::CompiledModel> try_npu(const ModelSpec& model, const std::shared_ptr<ov::Model>& graph,
                                           ov::CompiledModel& on_gpu, Progress& progress) {
    const auto say = [&](float fraction, const std::string& text) {
      if (progress.report) progress.report(fraction, text);
    };
    try {
      say(0.40F, "compilation OpenVINO (NPU)");
      ov::CompiledModel on_npu = core_->compile_model(graph, npu_->device, npu_config(*core_, npu_->device));
      if (progress.stop.stop_requested()) return std::nullopt;
      say(0.75F, "mesure du GPU et du NPU");
      const Measure gpu = measure(on_gpu);
      const Measure npu = measure(on_npu);
      const double error = mean_error(gpu.first_out, npu.first_out);
      const bool kept = error <= kNpuMaxMeanError && npu.ms <= gpu.ms * kNpuTolerance;
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
  std::shared_ptr<UsmPool> usm_;  // nul sans contexte GPU
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
      } catch (const std::exception&) {
        // propriété absente de ce greffon : le nom suffit
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
    std::optional<Npu> npu = find_npu(*core);
    return std::make_shared<OpenvinoRuntime>(std::move(core), chosen, full, std::move(npu));
  } catch (const std::exception& e) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("OpenVINO : {}", e.what()));
  }
}

}  // namespace ajp::gpu::openvino_rt
