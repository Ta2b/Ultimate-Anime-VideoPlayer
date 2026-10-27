// Une ouverture de la chaîne (player_run.cpp) : moteurs, RTX HDR, threads, session et boucle de
// main, jusqu'à la fin du fichier, q, ou un réglage qui demande de la rouvrir. Interne à
// player.cpp, qui enchaîne les ouvertures.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>

#include "app/play/engine_switch.h"
#include "app/play/export_job.h"
#include "app/play/player_engines.h"
#include "app/play/settings.h"
#include "core/types.h"

namespace ajp::app::play {

// Fin d'une ouverture de la chaîne : la lecture est finie (rien), ou le menu demande de la rouvrir
// avec ces réglages à cette position.
struct Reopen {
  PlaySettings settings;
  std::int64_t position_us = 0;
  bool paused = false;                             // la lecture était en pause : elle le reste
  std::optional<EngineChange> engines;             // onglet Moteurs : à faire avant de rouvrir
  std::optional<ExportJob> export_job;             // onglet Export : à faire avant de rouvrir
  std::optional<std::filesystem::path> next_file;  // fichier glissé sur la fenêtre : le lire à la place
};

// Une ouverture de la chaîne : moteurs, RTX HDR, threads, session et boucle de main, jusqu'à la
// fin du fichier, q, ou un réglage qui demande de la rouvrir. start_paused : la lecture rouverte
// attend l'utilisateur sur sa première image.
std::expected<std::optional<Reopen>, Refusal> run_chain(Player& p, PlaySettings settings,
                                                        std::int64_t start_us, bool start_paused = false);

}  // namespace ajp::app::play
