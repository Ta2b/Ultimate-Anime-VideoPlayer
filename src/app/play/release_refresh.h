// Le catalogue des moteurs au lancement (A18-02) : engines-nvidia.lock se relit sur un fil à part, parti
// avant l'ouverture de la fenêtre, et le thread de la fenêtre ne l'attend que s'il n'a aucune copie gardée
// (premier lancement : l'assistant doit pouvoir proposer TensorRT), en pompant ses événements.
#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>

#include "app/play/engine_install.h"
#include "app/play/engine_online.h"
#include "app/play/options.h"
#include "app/play/settings.h"

namespace ajp::app::play {

struct PlayerPaths;

struct ReleaseRefresh {
  std::filesystem::path cache;              // engines-nvidia.lock du dossier des données
  ReleaseIndexCopy before;                  // la copie gardée, lue avant que le fil ne la remplace
  std::shared_ptr<std::atomic<bool>> done;  // nul : aucun fil (hors ligne voulu, copie fraîche)
  std::jthread thread;                      // arrêté et attendu à la fin du lancement (curl coupé)
};

// Lit la copie gardée, puis lance la relecture en ligne si elle est due et permise (mises à jour non
// refusées, pas de dossier de données à part).
ReleaseRefresh start_release_refresh(const PlayOptions& options, const PlayerPaths& paths,
                                     const PlaySettings& saved);

// Les moteurs téléchargeables : ceux de l'engines.lock livré, et ceux joints à la release après elle. Depuis
// le thread de la fenêtre ouverte : attend le fil (fenêtre pompée) seulement sans copie gardée.
EngineCatalog engine_catalog(const PlayerPaths& paths, const PlaySettings& saved, ReleaseRefresh& refresh);

}  // namespace ajp::app::play
