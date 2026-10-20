// File du backend : toutes les soumissions du backend (noyaux de la chaîne, moteurs) s'y
// enchaînent dans l'ordre des appels, côté GPU, par un sémaphore timeline qui compte les
// soumissions. signal() rend le point de la dernière soumission ; wait() fait attendre la
// suivante, côté GPU, sur un point d'une autre file. Aucune attente hôte.
//
// Les soumissions s'accumulent en un lot (VkContext::enqueue) : un seul vkQueueSubmit par image,
// au premier signal d'un sémaphore d'ailleurs (slot du ring, image décodée), à flush() (avant
// d'exporter last() vers une autre file) ou à toute attente hôte.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/recording.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/sync.h"

namespace ajp::gpu::vk {

class BackendQueue {
 public:
  // exports : types de descripteurs vers lesquels la timeline s'exporte (interop d'un runtime).
  static std::expected<BackendQueue, Refusal> create(VkContext& context,
                                                     VkExternalSemaphoreHandleTypeFlags exports = 0);

  // L'enregistrement après tout ce qui a été soumis avant, et après les attentes demandées par
  // wait() depuis ; extra_waits et extra_signals s'y ajoutent (sémaphores d'un slot du ring,
  // d'une image décodée).
  Done submit(Recording& recording, std::span<const TimelinePoint> extra_waits = {},
              std::span<const TimelinePoint> extra_signals = {});

  // Le point atteint quand la dernière soumission s'achève (valeur 0 avant la première).
  TimelinePoint last() const { return timeline_.at(value_); }
  // Le point d'une soumission passée, de valeur value.
  TimelinePoint at(std::uint64_t value) const { return timeline_.at(value); }
  // La prochaine soumission attendra ce point, côté GPU.
  void wait(const TimelinePoint& point) { pending_.push_back(point); }
  // Soumet le lot : last() devient attendable par une autre file ou une autre API.
  Done flush() { return context_->flush(); }

  // Une étape d'une autre API sur la timeline (un réseau MIGraphX par HIP) : elle attend wait
  // (last() à cet instant) et signale signal, que la prochaine soumission attend. Aucune s'il
  // reste des attentes de wait(), sur des sémaphores que l'autre API ne voit pas.
  struct Step {
    std::uint64_t wait = 0;
    std::uint64_t signal = 0;
  };
  std::optional<Step> external_step();
  // Signal d'une étape par l'hôte, quand l'autre API ne l'a pas mise en file.
  Done signal_on_host(std::uint64_t value) { return timeline_.signal(value); }
  VkSemaphore semaphore() const { return timeline_.handle(); }
  // Valeur atteinte à cet instant : les soumissions jusqu'à elle sont finies (constat, sans
  // attente).
  std::expected<std::uint64_t, Refusal> completed() const { return timeline_.value(); }

 private:
  BackendQueue(VkContext& context, TimelineSemaphore timeline)
      : context_(&context), timeline_(std::move(timeline)) {}

  VkContext* context_;
  TimelineSemaphore timeline_;
  std::uint64_t value_ = 0;
  std::vector<TimelinePoint> pending_;
  std::vector<TimelinePoint> waits_;  // réutilisés d'une soumission à l'autre
  std::vector<TimelinePoint> signals_;
};

}  // namespace ajp::gpu::vk
