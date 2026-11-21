// Les données du lecteur au lancement, avant la fenêtre : les dossiers d'une version d'avant déplacés, les
// restes rangés, l'inscription suivie (main.cpp).
#pragma once

#include <filesystem>

#include "app/play/default_player.h"

namespace kirari::app::play {

// Où vivent les données lourdes et les caches, une fois déplacés ceux d'une version d'avant.
struct DataPlaces {
  std::filesystem::path heavy;
  std::filesystem::path cache;
};

// Détermine les deux emplacements et, en lecture normale, y reprend les données d'une version antérieure.
DataPlaces locate_launch_data(const std::filesystem::path& data, const std::filesystem::path& home,
                              const std::filesystem::path& xdg_cache, bool normal_launch);
DataPlaces settle_data(const std::filesystem::path& data, const PlayerPlace& place, bool normal_launch);

}  // namespace kirari::app::play
