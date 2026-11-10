// Le retour de présentation aux niveaux Timing et Wait (chantier 6.3, docs/adr/0011) : un identifiant à
// chaque présentation, la date demandée au pilote (VK_EXT_present_timing) et relue sans attente au tour
// suivant, ou, au niveau Wait, la fin de chaque présentation sondée (vkWaitForPresent2KHR, délai 0). Les
// dates relues nourrissent core::VblankClock ; le registre (core/present_ledger.h) apparie chaque retour à
// sa présentation. Tout se passe sur main, entre pl_swapchain_start_frame et la soumission, seul moment où
// la VkSwapchainKHR de libplacebo est sûre. Rien n'est créé au niveau Open : la présentation d'aujourd'hui.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>
#include <spdlog/logger.h>

#include "app/play/placebo_wayland.h"
#include "app/play/present_level.h"
#include "core/present_ledger.h"
#include "core/vblank_clock.h"

namespace ajp::app::play {

// Le dernier retour lu, pour --stats.
struct FeedbackRead {
  std::uint64_t id = 0;  // 0 : rien relu à ce tour
  std::int64_t target_us = 0;
  std::int64_t visible_us = 0;
  std::int64_t vsync_k = -1;  // numéro du vblank sur la grille de VblankClock ; −1 sans grille
};

class PresentFeedback {
 public:
  // wayland : la source du retour au niveau Wayland (placebo_wayland.h) ; aux niveaux Timing, le repli.
  PresentFeedback(pl_vulkan vk, PresentLevel level, const PresentCaps& caps,
                  std::shared_ptr<spdlog::logger> channel, std::unique_ptr<WaylandPresentation> wayland = {});
  ~PresentFeedback();
  PresentFeedback(const PresentFeedback&) = delete;
  PresentFeedback& operator=(const PresentFeedback&) = delete;

  // Entre start_frame et la soumission : relit ce qui est revenu, puis prépare la présentation qui part
  // (target_us : date visée en heure hôte, 0 pour aucune). Le pointeur vaut jusqu'à submitted().
  const pl_vulkan_present* prepare(pl_swapchain swapchain, std::int64_t now_us, std::int64_t target_us);
  // Le résultat de vkQueuePresentKHR (pl_vulkan_swapchain_submit).
  void submitted(VkResult result);

  // Le niveau effectif : Timing retombe (Wayland s'il est ouvert, sinon Wait) sans domaine de temps lisible
  // ou quand rien ne revient (present_level.h : feedback_silent).
  PresentLevel level() const { return level_; }
  std::uint64_t last_id() const { return last_id_; }
  const FeedbackRead& last_read() const { return read_; }
  const core::VblankClock& clock() const { return clock_; }
  // Les propriétés de timing de la swapchain (6.10) : durée d'un rafraîchissement et intervalle (UINT64_MAX
  // en VRR), en ns ; 0 tant qu'inconnues.
  std::uint64_t refresh_duration_ns() const { return refresh_duration_; }
  std::uint64_t refresh_interval_ns() const { return refresh_interval_; }
  // Les dates relues sont celles de l'affichage (Timing ou Wayland), pas d'un simple sondage (Wait).
  bool dated() const {
    return level_ == PresentLevel::Timing || level_ == PresentLevel::TimingAbsolute ||
           level_ == PresentLevel::Wayland;
  }
  // Fréquence fixe dite par la swapchain (intervalle = durée) ou par le compositeur (rafraîchissement non
  // nul) ; faux en VRR ou tant que rien n'est dit.
  bool fixed_refresh() const { return refresh_state() == RefreshState::Fixed; }
  // L'état de la fréquence (6.10) : present timing seulement ; wp_presentation ne distingue pas un
  // rafraîchissement inconnu d'un VRR, il dit seulement Fixed quand il en donne un.
  RefreshState refresh_state() const {
    if (level_ == PresentLevel::Wayland)
      return refresh_duration_ != 0 ? RefreshState::Fixed : RefreshState::Unknown;
    if (level_ != PresentLevel::Timing && level_ != PresentLevel::TimingAbsolute)
      return RefreshState::Unknown;
    return vrr_state(refresh_duration_, refresh_interval_);
  }
  // Vrai une fois après chaque changement de refresh_state() (relu au compteur de propriétés).
  bool take_refresh_change() { return std::exchange(refresh_changed_, false); }
  // <stats>.screen.tsv (0.7) : chaque retour wp_presentation (niveau Wayland) y est écrit, une ligne par
  // image.
  void record_stamps(const std::filesystem::path& file);

 private:
  struct Api;
  void new_swapchain(VkSwapchainKHR swapchain, std::int64_t now_us);
  void read_properties(VkSwapchainKHR swapchain);
  void choose_domain(VkSwapchainKHR swapchain);
  void calibrate(VkSwapchainKHR swapchain, std::int64_t now_us);
  void read_back(VkSwapchainKHR swapchain, std::int64_t now_us);
  void poll_wait(VkSwapchainKHR swapchain, std::int64_t now_us);
  void accept(std::uint64_t id, std::int64_t visible_us, bool dated);
  void read_wayland();
  void fall_back(std::string_view why);
  void drop_if_silent();

  pl_vulkan vk_;
  PresentLevel level_;
  PresentCaps caps_;
  std::shared_ptr<spdlog::logger> channel_;
  std::unique_ptr<Api> api_;
  std::unique_ptr<WaylandPresentation> wayland_;
  std::vector<WaylandDone> wayland_done_;
  std::ofstream stamps_;
  std::uint64_t discarded_ = 0;  // écartées par le compositeur (niveau Wayland)
  core::PresentLedger ledger_;
  core::TimeDomainMap domain_;
  core::VblankClock clock_;
  FeedbackRead read_;
  std::uint64_t recreations_ = UINT64_MAX;
  std::uint64_t properties_counter_ = UINT64_MAX;
  std::uint64_t domains_counter_ = UINT64_MAX;
  std::uint64_t domain_id_ = 0;
  int domain_kind_ = 0;        // VkTimeDomainKHR du domaine retenu
  bool local_domain_ = false;  // domaine propre à la swapchain, recalé par horodatages calibrés
  std::uint32_t stage_ = 0;
  std::uint64_t refresh_duration_ = 0;
  std::uint64_t refresh_interval_ = 0;
  bool refresh_changed_ = false;
  std::uint64_t last_id_ = 0;
  std::uint64_t queue_full_ = 0;
  pl_vulkan_present present_{};
  struct TimingStorage;
  std::unique_ptr<TimingStorage> timing_;
};

}  // namespace ajp::app::play
