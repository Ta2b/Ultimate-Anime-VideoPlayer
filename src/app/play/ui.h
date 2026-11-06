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
#include <thread>
#include <vector>

#include "app/play/controls.h"
#include "app/play/controls_draw.h"
#include "app/play/first_run.h"
#include "app/play/home_view.h"
#include "app/play/input.h"
#include "app/play/links.h"
#include "app/play/settings_panel.h"
#include "core/types.h"

struct SDL_Window;
union SDL_Event;
struct ImGuiContext;

namespace ajp::app::play {

class PlaceboScreen;

// Une police du système, lue en mémoire : son fichier et son numéro dans une collection (.ttc).
struct FontFile {
  std::string path;
  int index = 0;
  std::vector<unsigned char> data;
};

// Les polices de l'interface, cherchées (fontconfig) et lues sur un thread pendant que l'écran
// s'ouvre : à froid, FcInit bâtit son cache (300 ms pour une centaine de polices) et la police
// japonaise pèse souvent 17 Mo. Sans fontconfig, aucune : la police vectorielle d'ImGui.
class FontSearch {
 public:
  static FontSearch start();
  // Les polices trouvées, la principale d'abord ; attend la fin de la recherche. Sans thread
  // (refusé par le système), la recherche se fait ici.
  std::vector<FontFile> take();

 private:
  std::unique_ptr<std::vector<FontFile>> found_ = std::make_unique<std::vector<FontFile>>();
  std::jthread thread_;
};

class PlayerUi {
 public:
  // Refus : contexte ImGui impossible (LibraryUnavailable).
  static std::expected<std::unique_ptr<PlayerUi>, Refusal> create(SDL_Window* window, std::string title,
                                                                  FontSearch& fonts);
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
  // L'image de l'écran de bienvenue (first_run.h), recomposée à chaque appel.
  const ImDrawData* welcome(std::int64_t now_us, FirstRunView& view);
  // L'image de l'écran d'accueil (home_view.h) et du menu des paramètres s'il est ouvert ;
  // recomposée à chaque appel. Les réglages changés au menu vont dans actions.
  const ImDrawData* home(std::int64_t now_us, HomeView& view, std::vector<Input>& actions);

  Controls& controls() { return controls_; }
  // Les pistes que les menus de la barre proposent, et celles choisies.
  void set_tracks(TrackMenu tracks);
  // Les réglages en cours et ce que le menu des paramètres propose.
  void set_settings(const PlaySettings& running, SettingsChoices choices);
  // L'état du VRR relu en cours de lecture (plein écran) : l'avertissement du mode auto le suit.
  void set_display_vrr(bool capable, bool armed, bool fullscreen) {
    choices_.display_vrr = capable;
    choices_.display_vrr_armed = armed;
    choices_.window_fullscreen = fullscreen;
  }
  void set_running(const PlaySettings& running);
  void set_profiles(ProfilesView profiles);
  void set_black_status(std::string status);  // ligne du vrai noir dans l'onglet Grade
  void toggle_settings();
  void open_settings();
  // Le nom de la vidéo, en haut de l'image pendant la lecture.
  void set_title(std::string title);
  void close_settings();
  bool settings_open() const { return panel_.is_open(); }
  // Rend les textures de l'interface à l'écran avant qu'il ne soit détruit.
  void release(PlaceboScreen& screen);

 private:
  PlayerUi(SDL_Window* window, std::string title);
  void apply_scale();
  bool over_ui() const;
  void begin_frame(std::int64_t now_us);
  void follow_text_input();

  SDL_Window* window_;
  std::vector<FontFile> fonts_;  // lues par l'atlas d'ImGui tant que le contexte vit
  ImGuiContext* context_ = nullptr;
  Controls controls_;
  UiLook look_;
  SettingsPanel panel_;
  PlaySettings running_;
  SettingsChoices choices_;
  bool input_pending_ = false;
  static constexpr int kSettleFrames = 2;
  int settle_ = 0;  // images encore à recomposer après la dernière entrée
  // Un élément survolé attend son infobulle (immobilité de la souris puis délai d'ImGui) : en pause,
  // l'interface se recompose jusqu'à kTooltipWindowUs après la dernière entrée.
  static constexpr std::int64_t kTooltipWindowUs = 300'000;
  bool hover_waiting_ = false;
  std::int64_t last_input_us_ = 0;
  std::int64_t last_busy_frame_us_ = 0;  // la dernière image demandée par une tâche du panneau
  struct Click {
    int button = 0;
    int count = 1;
    std::int64_t at_us = 0;
  };
  std::vector<Click> clicks_;  // en attente de l'image ImGui qui dira s'ils touchent l'interface
  float wheel_ = 0.0F;         // crans de molette pas encore rendus en pas de volume
  bool cursor_hidden_ = false;
  bool text_input_ = false;
  bool home_ = false;  // l'écran d'accueil est à l'écran : le menu y prend la molette
  std::int64_t last_frame_us_ = 0;
  const ImDrawData* draw_ = nullptr;  // dernière image recomposée, si quelque chose y est visible
  int bottom_ = 0;                    // hauteur de la barre dans cette image
  int top_ = 0;                       // hauteur du titre
};

}  // namespace ajp::app::play
