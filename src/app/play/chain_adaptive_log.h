// Le journal de la scène pour l'apprentissage (adaptive-log.tsv, à côté de play.toml) : une ligne par seconde
// de lecture avec le grade adaptatif, sur toutes les routes (le plan analysé,
// chain_shots.h), pour ajuster des courbes sur les réglages faits à la main. Le fichier reste ouvert
// le temps de la chaîne ; chaque ligne dit depuis combien de temps l'utilisateur n'a rien touché, pour
// séparer les gestes de l'accord tacite.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>

#include "gpu/kernel_ops.h"

namespace ajp::app::play {

struct Chain;

// Le fichier ouvert et la date de sa dernière ligne.
struct AdaptiveLog {
  std::filesystem::path path;  // vide : pas de journal
  std::shared_ptr<std::ofstream> out;
  std::int64_t last_us = 0;
  std::uint64_t lines = 0;  // écrites depuis l'ouverture, pour la rotation
};

// Une ligne pour la seconde de lecture en cours (au plus une par seconde, rien en pause).
void adaptive_record(Chain& to, const gpu::KernelOps::SceneStats& stats, std::int64_t now_us);

}  // namespace ajp::app::play
