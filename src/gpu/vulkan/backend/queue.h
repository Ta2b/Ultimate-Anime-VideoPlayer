// File du backend : toutes les soumissions du backend (noyaux de la chaîne, moteurs) s'y
// enchaînent dans l'ordre des appels, côté GPU, par un sémaphore timeline qui compte les
// soumissions. signal() rend le point de la dernière soumission ; wait() fait attendre la
// suivante, côté GPU, sur un point d'une autre file. Aucune attente hôte.
//
// Les soumissions s'accumulent en un lot (VkContext::enqueue) : un seul vkQueueSubmit par image,
// au premier signal d'un sémaphore d'ailleurs (slot du ring, image décodée), à flush() (avant
// d'exporter last() vers une autre file) ou à toute attente hôte.
//
// Voies (gpu/lanes.h, V7) : quand l'appareil a plusieurs files dans la famille, chaque voie a sa
// timeline et sa file (l'agrandissement sur la principale, l'interpolation et la mesure sur les
// autres). use_lane choisit la voie des soumissions suivantes ; l'ordre entre voies vient de ce que
// chaque enregistrement lit et écrit (Recording::reads, writes) par LaneHazards. Sans files en
// plus, ou lanes éteintes (runtime, --one-lane) : une seule voie, comme avant.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/types.h"
#include "gpu/lanes.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/recording.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/sync.h"

namespace ajp::gpu::vk {

class BackendQueue {
 public:
  // exports : types de descripteurs vers lesquels la timeline s'exporte (interop d'un runtime).
  // lanes : voies sur plusieurs files quand le contexte en a (sinon une seule).
  static std::expected<BackendQueue, Refusal> create(VkContext& context,
                                                     VkExternalSemaphoreHandleTypeFlags exports = 0,
                                                     bool lanes = false);

  // Voies qui se recouvrent vraiment (BackendCaps::lanes) : 1, 2 ou 3.
  int lanes() const { return files_; }
  // La voie des soumissions suivantes ; sans voies, rien ne change.
  void use_lane(Lane lane) { current_ = files_ > 1 ? lane_index(lane) : kMain; }

  // L'enregistrement après tout ce qui a été soumis avant sur sa voie, après ce qu'il lit ou écrit
  // sur les autres, et après les attentes demandées par wait() depuis ; extra_waits et
  // extra_signals s'y ajoutent (sémaphores d'un slot du ring, d'une image décodée).
  Done submit(Recording& recording, std::span<const TimelinePoint> extra_waits = {},
              std::span<const TimelinePoint> extra_signals = {});

  // Le point atteint quand la dernière soumission de la voie courante s'achève (valeur 0 avant la
  // première).
  TimelinePoint last() const { return lanes_[current_].timeline.at(lanes_[current_].value); }
  // La dernière soumission de chaque voie (fin de vie d'une ressource : VkContext::settle).
  std::array<TimelinePoint, kLanes> lasts() const {
    std::array<TimelinePoint, kLanes> points{};
    for (std::size_t lane = 0; lane < kLanes; ++lane)
      points[lane] = lanes_[lane].timeline.at(lanes_[lane].value);
    return points;
  }
  // Un point qui suit tout ce qui a été soumis, sur toutes les voies (Backend::signal) : une
  // jointure sans commande sur la voie courante quand d'autres voies ont du travail après elle.
  std::expected<TimelinePoint, Refusal> joined();
  // Le ticket de la dernière soumission, toutes voies confondues : croissant d'une soumission à
  // l'autre, il date un usage (un tampon libre après lui) ; sans voies, la valeur de last().
  std::uint64_t ticket() const { return files_ == 1 ? lanes_[kMain].value : tickets_; }
  // Le point qui marque la fin de la soumission du ticket ; une soumission déjà finie rend un point
  // atteint.
  TimelinePoint at(std::uint64_t ticket) const;
  // La prochaine soumission de chaque voie attendra ce point, côté GPU.
  void wait(const TimelinePoint& point);
  // Soumet le lot : last() devient attendable par une autre file ou une autre API.
  Done flush() { return context_->flush(); }

  // Une étape d'une autre API sur la timeline (un réseau MIGraphX par HIP) : elle attend wait
  // (last() à cet instant) et signale signal, que la prochaine soumission attend. Aucune s'il
  // reste des attentes de wait(), sur des sémaphores que l'autre API ne voit pas. Une seule
  // voie avec un runtime.
  struct Step {
    std::uint64_t wait = 0;
    std::uint64_t signal = 0;
  };
  std::optional<Step> external_step();
  // Une étape faite par l'hôte (un runtime sans interop, sur son propre fil) : elle attend wait
  // par wait_elsewhere et signale signal par signal_on_host. Les attentes de wait() restent à
  // la prochaine soumission : l'étape ne lit que du travail déjà soumis, que flush() a fait
  // partir.
  Step host_step();
  // Signal d'une étape par l'hôte, quand l'autre API ne l'a pas mise en file. Appelable depuis
  // un autre fil que celui du contexte.
  Done signal_on_host(std::uint64_t value) { return lanes_[kMain].timeline.signal(value); }
  // Attente de value depuis un autre fil que celui du contexte : ni lot (l'appelant a fait
  // flush() avant), ni compteur host_waits. Refus VulkanCallFailed au délai dépassé.
  Done wait_elsewhere(std::uint64_t value, std::chrono::nanoseconds timeout = std::chrono::seconds(10)) const;
  VkSemaphore semaphore() const { return lanes_[kMain].timeline.handle(); }
  // Valeur atteinte à cet instant : les soumissions jusqu'à elle sont finies (constat, sans
  // attente).
  // En tickets : toutes les soumissions jusqu'à lui sont finies.
  std::expected<std::uint64_t, Refusal> completed();

 private:
  static constexpr std::size_t kMain = lane_index(Lane::Upscale);  // la timeline des runtimes
  static constexpr std::size_t kMarks = 1024;  // rangs gardés par voie (LaneHazards → valeur)
  // Une voie : sa timeline, sa file (indice de VkContext::queue_lanes), ses attentes en suspens et
  // la valeur de ses derniers rangs.
  struct LaneState {
    TimelineSemaphore timeline;
    std::size_t file = 0;
    std::uint64_t value = 0;
    std::vector<TimelinePoint> pending;
    std::array<std::uint64_t, kMarks> marks{};   // rang gardé à chaque place
    std::array<std::uint64_t, kMarks> values{};  // sa valeur
    std::uint64_t last_mark_value = 0;           // valeur du dernier rang soumis
  };
  BackendQueue(VkContext& context, std::vector<LaneState> lanes, int files)
      : context_(&context), lanes_(std::move(lanes)), files_(files) {}
  // Point de la timeline de lane qui couvre son rang mark : la valeur gardée, ou une plus récente
  // (attente plus forte, jamais plus faible).
  TimelinePoint mark_point(std::size_t lane, std::uint64_t mark) const;
  void keep_mark(std::size_t lane, std::uint64_t mark);
  void add_waits(const LaneHazards::Op& op);

  VkContext* context_;
  std::vector<LaneState> lanes_;  // kLanes voies ; sans voies, seule kMain sert
  int files_ = 1;
  std::size_t current_ = kMain;
  LaneHazards hazards_;
  // Les tickets pas encore vus finis (voies seulement), dans l'ordre.
  struct Ticket {
    std::uint64_t ticket = 0;
    std::size_t lane = 0;
    std::uint64_t value = 0;
  };
  std::deque<Ticket> open_;
  std::uint64_t tickets_ = 0;
  void issued(std::size_t lane);
  std::vector<TimelinePoint> waits_;  // réutilisés d'une soumission à l'autre
  std::vector<TimelinePoint> signals_;
  std::vector<std::uintptr_t> reads_;
};

}  // namespace ajp::gpu::vk
