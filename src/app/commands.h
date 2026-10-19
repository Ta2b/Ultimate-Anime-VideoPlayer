// Exécution des commandes de la ligne de commande. main() analyse argv, trouve les
// dossiers une fois, puis appelle run() ; tout le reste reçoit ce contexte en paramètre.
//
// play, export et bench vérifient leurs fichiers puis refusent CommandUnavailable tant
// que render, present et gpu ne sont pas branchés ; engines et import-sdk sont complets.
#pragma once

#include <expected>
#include <functional>
#include <ostream>
#include <stop_token>

#include "app/cli.h"
#include "app/data_paths.h"
#include "app/download.h"
#include "app/engine_chooser.h"
#include "app/install.h"
#include "app/library_loader.h"
#include "app/system_report.h"
#include "core/engines.h"
#include "core/settings.h"
#include "core/types.h"

namespace ajp::app {

struct AppContext {
  core::Os os = host_os();
  DataPaths paths;
  const LibraryLoader* loader = nullptr;  // sondes du système ; jamais nul
  std::ostream* out = nullptr;            // résultats et progression
  std::ostream* err = nullptr;            // refus
  DownloadOptions download;               // https seulement, délais par défaut
  std::stop_token stop;
};

// Codes de sortie du processus.
inline constexpr int kExitOk = 0;
inline constexpr int kExitRefused = 1;  // la commande a été refusée en cours d'exécution
inline constexpr int kExitUsage = 2;    // ligne de commande refusée

// Exécute la commande ; le refus remonte tel quel.
std::expected<void, Refusal> run_command(const Invocation& invocation, const AppContext& context);
// run_command, puis « ajp : <Code> : <détail> » sur err en cas de refus. Rend le code de sortie.
int run(const Invocation& invocation, const AppContext& context);

// Texte d'un refus, tel que la ligne de commande l'affiche.
std::string refusal_text(const Refusal& refusal);

// --- engines (commands_engines.cpp) ---

// L'écran « Moteurs » pour ce système : détection, blocages des runtimes, réglages, index,
// modèles manquants, place libre.
std::expected<EngineChooser, Refusal> open_chooser(const AppContext& context, const SystemReport& report);

// Exécute le plan validé : télécharge et installe paquets et modèles, retire les moteurs
// décochés, puis écrit les réglages. Après un refus en cours de route, ce qui s'est
// installé est quand même noté dans settings.toml ; un moteur bloqué (installé en mode
// avancé) est noté sans devenir préféré.
std::expected<core::Settings, Refusal> apply_plan(
    const ChooserPlan& plan, const core::SystemInfo& system, std::span<const EngineBlocker> blockers,
    const AppContext& context, const Downloader& downloader,
    const std::function<void(const InstallProgress&)>& on_progress);

std::expected<void, Refusal> list_engines(const AppContext& context);
std::expected<void, Refusal> install_engines(const EnginesArgs& args, const AppContext& context);

}  // namespace ajp::app
