#include "gpu/openvino/openvino_net.h"

#include <exception>
#include <format>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "core/inbox.h"
#include "core/thread.h"

namespace ajp::gpu::openvino_rt {
namespace {

using vk::net::HostBinding;

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
      if (failed_) return std::unexpected(*failed_);
      if (!worker_.joinable()) {
        // Le fil des étapes, au premier appel ; un refus du fil lui-même (exception échappée) est
        // gardé pour l'appel suivant.
        auto worker = start_thread(
            "openvino", ThreadReport([this](const Refusal& refusal) { note_failure(refusal); }),
            [this](const std::stop_token& stop, const ThreadReport& /*report*/) { serve(stop); });
        if (!worker) return std::unexpected(std::move(worker.error()));
        worker_ = std::move(*worker);
      }
    }
    if (!jobs_.try_push(Job{.inputs = {inputs.begin(), inputs.end()},
                            .outputs = {outputs.begin(), outputs.end()},
                            .ready = std::move(ready),
                            .done = std::move(done)})) {
      // Rien n'est lancé : l'appelant fait l'étape lui-même, et signale la timeline.
      return refuse(RefusalCode::ThreadUnavailable,
                    std::format("OpenVINO : {} étapes en file, le fil n'en reprend pas", kStepBound));
    }
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

  void note_failure(const Refusal& refusal) {
    const std::scoped_lock lock(jobs_mutex_);
    if (!failed_) failed_ = refusal;
  }

  // Le fil des étapes : chaque étape en file est faite, même à l'arrêt (sa timeline l'attend), et
  // chacune appelle done, quoi qu'il arrive : la file attendrait sans fin un signal perdu.
  void serve(const std::stop_token& stop) {
    while (std::optional<Inbox<Job>::Message> message = jobs_.wait(stop)) {
      Job& job = std::get<Job>(*message);
      std::expected<void, Refusal> result;
      try {
        result = job.ready();
        if (result) {
          const std::scoped_lock lock(request_mutex_);
          result = infer(job.inputs, job.outputs);
        }
      } catch (...) {
        result = std::unexpected(refusal_from_exception(RefusalCode::DeviceUnavailable, "OpenVINO, étape"));
      }
      job.done(std::move(result));
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
  // Étapes en file pour le fil ; au-delà, l'appelant fait l'étape lui-même.
  static constexpr std::size_t kStepBound = 16;

  std::mutex request_mutex_;  // request_ et bound_ : le fil des étapes ou celui de run
  std::mutex jobs_mutex_;     // failed_ et le lancement du fil
  Inbox<Job> jobs_{{kStepBound}};
  std::optional<Refusal> failed_;  // le fil des étapes a cédé : plus aucune étape différée
  std::jthread worker_;
};

}  // namespace

std::unique_ptr<vk::net::RuntimeNet> make_net(ov::CompiledModel compiled, vk::net::IoNames names,
                                              std::shared_ptr<UsmPool> usm) {
  return std::make_unique<OpenvinoNet>(std::move(compiled), std::move(names), std::move(usm));
}

}  // namespace ajp::gpu::openvino_rt
