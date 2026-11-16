// Le crédit de temps des clés GMFSS du mélange (upscale_late.cpp, credit_gmfss) : ce qu'une image source
// ajoute au crédit.
#pragma once

#include <algorithm>
#include <cstdint>

namespace ajp::app::play {

// Les images de RIFE et SPAN qu'une paire garde toujours devant ses clés.
inline constexpr int kGmfssFloorImages = 2;

// Ce que gagne une image source de durée `pts_step_us` : les trois quarts (le reste : RIFE, ingestion,
// rendu), sans jamais prendre le temps de kGmfssFloorImages images, `image_us` le temps de service mesuré
// d'une image interpolée (0 : inconnu, la seule part des trois quarts). Sans ce plancher, un travail lourd
// (SPAN vers la 4K) laissait les clés prendre presque toute la paire : la cadence tombait sous celle de
// RIFE seul et variait d'une seconde à l'autre.
constexpr std::int64_t gmfss_earned_us(std::int64_t pts_step_us, std::int64_t image_us) {
  const std::int64_t share = pts_step_us * 3 / 4;
  const std::int64_t left = pts_step_us - image_us * kGmfssFloorImages;
  return std::max<std::int64_t>(std::min(share, left), 0);
}

}  // namespace ajp::app::play
