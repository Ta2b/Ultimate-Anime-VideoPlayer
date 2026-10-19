// L'interface du lecteur : le contexte Dear ImGui, les événements SDL3 vers ImGui (en pixels de
// la sortie, sans le backend SDL d'ImGui), la police du système à l'échelle de l'écran, le thème,
// et l'image de l'interface, recomposée seulement quand Controls le demande.
//
// La souris sur l'image (hors de la barre) : simple clic pause, double clic plein écran, clic
// droit pause, molette volume. Les touches restent au lecteur (input.h), sauf pendant une saisie
// dans l'interface.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "app/play/controls.h"
#include "app/play/controls_draw.h"
#include "app/play/input.h"
#include "app/play/links.h"
#include "app/play/settings_panel.h"
#include "core/types.h"

struct SDL_Window;
union SDL_Event;
struct ImGuiContext;

namespace ajp::app::play {

class PlaceboScreen;

class PlayerUi {
 public:
  // Refus : contexte ImGui impossible (LibraryUnavailable).
  static std::expected<std::unique_ptr<PlayerUi>, Refusal> create(SDL_Window* window, std::string title);
  ~PlayerUi();
  PlayerUi(const PlayerUi&) = delete;
  PlayerUi& operator=(const PlayerUi&) = delete;

  // Un événement SDL. ImGui le voit ; la souris montre la barre. Les gestes sur l'image vont dans
  // actions. true : l'interface prend l'événement (saisie en cours), le lecteur ne le traite pas.
  bool handle(const SDL_Event& event, std::int64_t now_us, std::vector<Input>& actions);
  // Recompose l'interface si Controls le demande (au plus ~60 Hz) ; rend ce qu'il faut dessiner.
  // Les clics de l'interface, et le simple clic sur l'image une fois son délai passé, vont dans
  // actions.
  UiFrame frame(std::int64_t now_us, const ControlsView& view, std::vector<Input>& actions);
  // L'image de la préparation des moteurs (avant la lecture), recomposée à chaque appel.
  const ImDrawData* preparation(std::int64_t now_us, const PrepareView& view);

  Controls& controls() { return controls_; }
  // Les pistes que les menus de la barre proposent, et celles choisies.
  void set_tracks(TrackMenu tracks);
  // Les réglages en cours et ce que le menu des paramètres propose.
  void set_settings(const PlaySettings& running, SettingsChoices choices);
  void set_running(const PlaySettings& running);
  void toggle_settings();
  void close_settings();
  bool settings_open() const { return panel_.is_open(); }
  // Rend les textures de l'interface à l'écran avant qu'il ne soit détruit.
  void release(PlaceboScreen& screen);

 private:
  PlayerUi(SDL_Window* window, std::string title);
  void apply_scale();
  bool over_ui() const;

  SDL_Window* window_;
  ImGuiContext* context_ = nullptr;
  Controls controls_;
  UiLook look_;
  SettingsPanel panel_;
  PlaySettings running_;
  SettingsChoices choices_;
  bool input_pending_ = false;
  struct Click {
    int button = 0;
    int count = 1;
    std::int64_t at_us = 0;
  };
  std::vector<Click> clicks_;  // en attente de l'image ImGui qui dira s'ils touchent l'interface
  float wheel_ = 0.0F;         // crans de molette pas encore rendus en pas de volume
  bool cursor_hidden_ = false;
  bool text_input_ = false;
  std::int64_t last_frame_us_ = 0;
  const ImDrawData* draw_ = nullptr;  // dernière image recomposée, si quelque chose y est visible
  int bottom_ = 0;                    // hauteur de la barre dans cette image
  int top_ = 0;                       // hauteur du titre
};

}  // namespace ajp::app::play
