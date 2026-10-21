// Le fil des étapes différées d'un réseau Core ML (RuntimeNet::run_deferred), partagé par les
// réseaux d'ONNX Runtime (CoremlNet) et l'interpolation d'Apple (coreml_vt.h) : dans l'ordre des
// appels, ready (l'attente des entrées sur la timeline Vulkan), le calcul, done (le signal de
// l'étape). Le fil du lecteur n'attend plus le réseau : le Neural Engine ou le bloc média calcule
// pendant que le GPU fait le reste de la chaîne.
#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "core/inbox.h"
#include "core/types.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

class DeferredSteps {
 public:
  using Compute = std::function<std::expected<void, Refusal>(std::span<const vk::net::HostBinding> inputs,
                                                             std::span<const vk::net::HostBinding> outputs)>;
  // compute calcule une étape sur le fil (« coreml » suivi de label), lancé à la première.
  DeferredSteps(std::string label, Compute compute)
      : label_(std::move(label)), compute_(std::move(compute)) {}
  DeferredSteps(const DeferredSteps&) = delete;
  DeferredSteps& operator=(const DeferredSteps&) = delete;
  DeferredSteps(DeferredSteps&&) = delete;
  DeferredSteps& operator=(DeferredSteps&&) = delete;
  // Les étapes encore en file sont faites avant le retour : chacune signale la timeline que la
  // chaîne attend.
  ~DeferredSteps();

  // Met l'étape en file. Refus : le fil a cédé ou ne démarre pas, ou kStepBound étapes déjà en
  // file (ThreadUnavailable) ; rien n'est alors lancé, l'appelant fait l'étape lui-même.
  std::expected<void, Refusal> push(std::span<const vk::net::HostBinding> inputs,
                                    std::span<const vk::net::HostBinding> outputs,
                                    std::function<std::expected<void, Refusal>()> ready,
                                    std::function<void(std::expected<void, Refusal>)> done);

 private:
  struct Job {
    std::vector<vk::net::HostBinding> inputs;
    std::vector<vk::net::HostBinding> outputs;
    std::function<std::expected<void, Refusal>()> ready;
    std::function<void(std::expected<void, Refusal>)> done;
  };
  // Étapes en file pour le fil ; au-delà, l'appelant fait l'étape lui-même.
  static constexpr std::size_t kStepBound = 16;

  void note_failure(const Refusal& refusal);
  // Chaque étape en file est faite, même à l'arrêt (sa timeline l'attend), et chacune appelle
  // done, quoi qu'il arrive : la file attendrait sans fin un signal perdu.
  void serve(const std::stop_token& stop);

  std::string label_;
  Compute compute_;
  std::mutex mutex_;  // failed_ et le lancement du fil
  Inbox<Job> jobs_{{kStepBound}};
  std::optional<Refusal> failed_;  // le fil a cédé : plus aucune étape différée
  std::jthread worker_;
};

}  // namespace ajp::gpu::coreml_rt
