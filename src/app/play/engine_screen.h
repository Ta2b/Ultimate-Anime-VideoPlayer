// L'écran des moteurs (first_run.h) dans la fenêtre du lecteur : au premier lancement, avec
// --first-run, ou rouvert pendant la lecture depuis l'onglet Moteurs ; puis les moteurs cochés
// téléchargés, settings.toml écrit et le profil de démarrage gardé. Et la bibliothèque du
// backend qui en découle, le moteur préféré des réseaux essayé en premier (engine_switch.h).
#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <vector>

#include "app/play/backend_library.h"
#include "app/play/chain.h"
#include "app/play/engine_install.h"
#include "app/play/input.h"
#include "app/play/options.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"
#include "app/play/player_engines.h"
#include "app/play/ui.h"
#include "core/engines.h"
#include "core/settings.h"
#include "gpu/backend.h"

namespace ajp::app::play {

struct EngineScreen {
  PlaceboScreen& screen;
  PlayerUi& ui;
  const PlayOptions& options;
  const PlayerPaths& paths;
  const EngineCatalog& catalog;
  std::filesystem::path engines_path;  // settings.toml
  bool restart_wizard =
      false;  // rouvert depuis les paramètres : l'assistant en entier, depuis le premier écran
};

// L'écran jusqu'à « Commencer », puis les téléchargements, settings.toml et user.toml. revisit :
// rouvert pendant la lecture (« Annuler » n'y change rien, le profil actif reste, et le moteur des
// réseaux choisi à l'onglet Moteurs aussi tant qu'aucun moteur n'est ajouté). false : l'utilisateur
// a fermé la fenêtre.
std::expected<bool, Refusal> run_engine_screen(const EngineScreen& s, core::Settings& engines,
                                               ProfileState& profiles, bool revisit, std::size_t& next_key,
                                               SnapshotPlan& snapshot);

// Les bibliothèques de backend du poste, par préférence : les téléchargées d'abord, puis celles
// livrées ou de l'arbre de construction (présentes ou non).
std::vector<std::filesystem::path> backend_candidates(const EngineCatalog& catalog, const PlayerPaths& paths);

// Les moteurs des réseaux dont la bibliothèque est sur le poste, pour l'onglet Moteurs.
std::vector<core::EngineId> installed_switchable(const EngineCatalog& catalog, const PlayerPaths& paths);

// La bibliothèque pour les moteurs choisis, celle du moteur préféré essayée en premier ; Vulkan
// reste le recours (BackendLibrary::choose).
std::expected<std::shared_ptr<BackendLibrary>, Refusal> open_backend(const EngineCatalog& catalog,
                                                                     const PlayerPaths& paths,
                                                                     const core::Settings& engines,
                                                                     const gpu::BackendInit& probe);

// Ce que l'onglet Moteurs a demandé, entre deux ouvertures de la chaîne : le moteur des réseaux
// changé, ou l'écran des moteurs rouvert ; puis la bibliothèque rouverte (l'ancienne reste si
// aucune ne s'ouvre) et p.notice dit le résultat. false : l'utilisateur a fermé la fenêtre.
std::expected<bool, Refusal> handle_engine_change(Player& p, const EngineChange& request);

// La ligne de l'onglet quand le moteur préféré a refusé ce GPU et qu'un autre le remplace ; vide sinon.
std::string fallback_note(const core::Settings& engines, const std::filesystem::path& library);

}  // namespace ajp::app::play
