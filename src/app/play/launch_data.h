// Les données du lecteur au lancement, avant la fenêtre : les dossiers d'une version d'avant déplacés, les
// restes rangés, l'inscription suivie (main.cpp).
#pragma once

#include <filesystem>

#include "app/play/default_player.h"

namespace ajp::app::play {

// Où vivent les données lourdes et les caches, une fois déplacés ceux d'une version d'avant.
struct DataPlaces {
  std::filesystem::path heavy;
  std::filesystem::path cache;
};

// purge_updates : le dossier de l'utilisateur est rangé (faux pour --data-dir et le banc).
DataPlaces settle_data(const std::filesystem::path& data, const PlayerPlace& place, bool purge_updates);

}  // namespace ajp::app::play
