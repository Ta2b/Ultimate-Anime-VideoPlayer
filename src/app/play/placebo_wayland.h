// Le retour de présentation par wp_presentation (chantier 6.4, niveau Wayland) : quand Vulkan ne date pas
// les images, le compositeur dit quand chacune est apparue. Une file d'événements à nous, sur la connexion
// de SDL ; une demande de retour (wp_presentation.feedback) juste avant chaque présentation, que le commit
// du pilote dans vkQueuePresentKHR emporte ; une relecture non bloquante sur main. Exige l'horloge
// CLOCK_MONOTONIC du compositeur, celle de host_now_us (comme mpv). Rien sans Wayland, sans wp_presentation
// ou sans wayland-client à la compilation (AJP_WITH_WAYLAND_PRESENTATION).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/logger.h>

struct SDL_Window;

namespace ajp::app::play {

struct WaylandDone {
  std::uint64_t id = 0;
  std::int64_t visible_us = 0;   // 0 : image écartée par le compositeur (discarded)
  std::uint32_t refresh_ns = 0;  // durée d'un rafraîchissement dite par le compositeur ; 0 : inconnue ou VRR
  std::uint64_t msc = 0;         // compteur des rafraîchissements de la sortie (0 : inconnu)
  std::uint32_t flags = 0;       // wp_presentation_feedback.kind (vsync, horloge matérielle, zéro copie…)
};

// L'horodatage d'une image vue (0.7) : ce que dit wp_presentation_feedback.presented, en clair.
struct ScreenStamp {
  std::uint64_t seq = 0;  // la présentation (PresentFeedback)
  std::int64_t presented_us = 0;
  std::uint32_t refresh_ns = 0;
  std::uint64_t msc = 0;
  std::uint32_t flags = 0;
  friend bool operator==(const ScreenStamp&, const ScreenStamp&) = default;
};
// Les arguments de presented (secondes sur 64 bits en deux moitiés, nanosecondes, MSC en deux moitiés).
ScreenStamp stamp_of(std::uint64_t seq, std::uint32_t sec_hi, std::uint32_t sec_lo, std::uint32_t nsec,
                     std::uint32_t refresh_ns, std::uint32_t seq_hi, std::uint32_t seq_lo,
                     std::uint32_t flags);
// <stats>.screen.tsv : l'en-tête, puis une ligne par image vue.
std::string_view screen_stamp_header();
std::string screen_stamp_line(const ScreenStamp& stamp);

class WaylandPresentation {
 public:
  // Rien hors Wayland, sans wp_presentation, ou à une autre horloge que CLOCK_MONOTONIC.
  static std::unique_ptr<WaylandPresentation> open(SDL_Window* window,
                                                   const std::shared_ptr<spdlog::logger>& channel);
  ~WaylandPresentation();
  WaylandPresentation(const WaylandPresentation&) = delete;
  WaylandPresentation& operator=(const WaylandPresentation&) = delete;

  // Juste avant la présentation `id` : la demande de retour, emportée par le commit qui suit.
  void request(std::uint64_t id);
  // Sans attendre : les retours arrivés depuis la dernière relecture.
  void read(std::vector<WaylandDone>& out);

  struct State;

 private:
  explicit WaylandPresentation(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace ajp::app::play
