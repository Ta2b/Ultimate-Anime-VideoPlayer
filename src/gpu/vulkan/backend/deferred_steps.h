// Le fil des étapes différées d'un réseau confié à un runtime de l'hôte (RuntimeNet::run_deferred) :
// dans l'ordre des appels, ready (l'attente des entrées sur la timeline Vulkan), le calcul, done
// (le signal de l'étape). Le fil du lecteur n'attend plus le réseau : le runtime calcule pendant
// que le GPU fait le reste de la chaîne. Partagé par les réseaux d'ONNX Runtime de Core ML
// (gpu/coreml) et de Windows ML (gpu/winml), et par l'interpolation d'Apple.
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

namespace ajp::gpu::vk::net {

class DeferredSteps {
 public:
  using Compute = std::function<std::expected<void, Refusal>(std::span<const HostBinding> inputs,
                                                             std::span<const HostBinding> outputs)>;
  // compute calcule une étape sur le fil thread (son nom), lancé à la première ; runtime nomme le
  // runtime dans les refus (« Core ML »).
  DeferredSteps(std::string thread, std::string runtime, Compute compute)
      : thread_(std::move(thread)), runtime_(std::move(runtime)), compute_(std::move(compute)) {}
  DeferredSteps(const DeferredSteps&) = delete;
  DeferredSteps& operator=(const DeferredSteps&) = delete;
  DeferredSteps(DeferredSteps&&) = delete;
  DeferredSteps& operator=(DeferredSteps&&) = delete;
  // Les étapes encore en file sont faites avant le retour : chacune signale la timeline que la
  // chaîne attend.
  ~DeferredSteps();

  // Met l'étape en file. Refus : le fil a cédé ou ne démarre pas, ou kStepBound étapes déjà en
  // file (ThreadUnavailable) ; rien n'est alors lancé, l'appelant fait l'étape lui-même.
  std::expected<void, Refusal> push(std::span<const HostBinding> inputs, std::span<const HostBinding> outputs,
                                    std::function<std::expected<void, Refusal>()> ready,
                                    std::function<void(std::expected<void, Refusal>)> done);

  // Étapes en file pour le fil ; au-delà, l'appelant fait l'étape lui-même.
  static constexpr std::size_t kStepBound = 16;

 private:
  struct Job {
    std::vector<HostBinding> inputs;
    std::vector<HostBinding> outputs;
    std::function<std::expected<void, Refusal>()> ready;
    std::function<void(std::expected<void, Refusal>)> done;
  };

  void note_failure(const Refusal& refusal);
  // Chaque étape en file est faite, même à l'arrêt (sa timeline l'attend), et chacune appelle
  // done, quoi qu'il arrive : la file attendrait sans fin un signal perdu.
  void serve(const std::stop_token& stop);

  std::string thread_;
  std::string runtime_;
  Compute compute_;
  std::mutex mutex_;  // failed_ et le lancement du fil
  Inbox<Job> jobs_{{kStepBound}};
  std::optional<Refusal> failed_;  // le fil a cédé : plus aucune étape différée
  std::jthread worker_;
};

}  // namespace ajp::gpu::vk::net
