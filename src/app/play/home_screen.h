// L'écran d'accueil : « ajp » lancé sans fichier, et rendu à la fin d'une vidéo ouverte depuis lui.
// La fenêtre montre le menu des paramètres, tous ses onglets, et « Ouvrir une vidéo »
// (SDL_ShowOpenFileDialog) ; un fichier glissé sur la fenêtre lance aussi la lecture, comme celui que
// le Finder ouvre avec AJP.app sous macOS (SDL le rend en SDL_EVENT_DROP_FILE). Les réglages changés
// ici sont gardés comme pendant la lecture : play.toml, user.toml (raccourcis), settings.toml (moteurs).
#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "app/play/chain.h"
#include "app/play/engine_install.h"
#include "app/play/input.h"
#include "app/play/options.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"
#include "app/play/player_engines.h"
#include "app/play/settings.h"
#include "app/play/ui.h"
#include "core/display.h"
#include "core/settings.h"
#include "core/types.h"

union SDL_Event;

namespace ajp::app::play {

// La source supposée tant qu'aucune vidéo n'est ouverte, la plus courante : un épisode 1080p à
// 23,976 images par seconde. Elle choisit la règle que le menu montre.
inline constexpr core::SourceInfo kHomeSource{.width = 1920, .height = 1080, .fps = 24000.0 / 1001.0};

struct HomeScreen {
  PlaceboScreen& screen;
  PlayerUi& ui;
  const PlayOptions& options;
  const PlayerPaths& paths;
  const EngineCatalog& catalog;
  std::filesystem::path settings_path;  // play.toml
};

// Ce que l'accueil lit et garde.
struct HomeState {
  PlaySettings& saved;      // play.toml, réécrit quand un réglage change
  core::Settings& engines;  // settings.toml
  std::size_t& next_key;
  SnapshotPlan& snapshot;
  std::optional<PlayFile> start;  // la vidéo choisie par PlayFile : instant et pause
};

// Ce qu'est un chemin glissé, d'après son extension seule, sans toucher au disque (A24-15). Pur.
DropKind drop_kind(const std::filesystem::path& path);
// Le fichier d'un SDL_EVENT_DROP_FILE et ce qu'il est (Dropped, input.h) ; rien pour un autre événement ou un
// dossier. Lit le disque une fois (dossier ou fichier) et appelle raise_for_forwarded.
std::optional<Dropped> dropped(const SDL_Event& event);
// La fenêtre passe devant quand l'événement est un fichier transmis par un autre lancement
// (single_instance.h).
void raise_for_forwarded(const SDL_Event& event);
// La vidéo d'un SDL_EVENT_DROP_FILE ; rien pour le reste.
std::optional<std::filesystem::path> dropped_file(const SDL_Event& event);

// Les vidéos déposées sur la fenêtre, ou transmises par un second lancement, pendant une attente qui n'en
// ouvre aucune (assistant, installation des moteurs, calibration) : elles sont gardées et rendues à la file
// de SDL à la fin, d'où la boucle suivante les lit comme un dépôt. Sans cela, l'événement était consommé et
// perdu alors que le second lancement avait répondu « ok » et s'était fermé (B29-04). La fenêtre passe devant
// dans tous les cas.
class DeferredDrops {
 public:
  DeferredDrops() = default;
  ~DeferredDrops();
  DeferredDrops(const DeferredDrops&) = delete;
  DeferredDrops& operator=(const DeferredDrops&) = delete;
  // Vrai si l'événement est un dépôt (il ne va pas plus loin) ; une vidéo est gardée, le reste est écarté.
  bool keep(const SDL_Event& event);
  // Les chemins gardés, le plus ancien d'abord. Pour les tests.
  const std::vector<std::string>& files() const { return files_; }
  // Rend les fichiers à la file de SDL maintenant ; la liste se vide.
  void release();

 private:
  std::vector<std::string> files_;
};

// L'accueil jusqu'à une vidéo choisie ; rien si l'utilisateur quitte. problem : pourquoi la vidéo
// précédente n'a pas pu être lue, vide sinon.
std::expected<std::optional<std::filesystem::path>, Refusal> run_home(const HomeScreen& h, HomeState& state,
                                                                      const std::string& problem);

}  // namespace ajp::app::play
