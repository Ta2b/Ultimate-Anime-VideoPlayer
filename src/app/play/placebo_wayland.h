// Le retour de présentation par wp_presentation (chantier 6.4, niveau Wayland) : quand Vulkan ne date pas
// les images, le compositeur dit quand chacune est apparue. Une file d'événements à nous, sur la connexion
// de SDL ; une demande de retour (wp_presentation.feedback) juste avant chaque présentation, que le commit
// du pilote dans vkQueuePresentKHR emporte ; une relecture non bloquante sur main. Exige l'horloge
// CLOCK_MONOTONIC du compositeur, celle de host_now_us (comme mpv). Rien sans Wayland, sans wp_presentation
// ou sans wayland-client à la compilation (AJP_WITH_WAYLAND_PRESENTATION).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <spdlog/logger.h>

struct SDL_Window;

namespace ajp::app::play {

struct WaylandDone {
  std::uint64_t id = 0;
  std::int64_t visible_us = 0;   // 0 : image écartée par le compositeur (discarded)
  std::uint32_t refresh_ns = 0;  // durée d'un rafraîchissement dite par le compositeur ; 0 : inconnue ou VRR
};

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
