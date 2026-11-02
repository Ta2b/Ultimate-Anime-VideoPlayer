// Référence de l'analyse d'une image pour le grade (kernels/spec/frame_stats.h). Liée aussi au
// lecteur : la pré-analyse des plans (media/shot_probe.cpp) mesure par elle, et frame_stats_of tire
// les mesures des comptes du noyau GPU.
#pragma once

#include "kernels/image.h"
#include "kernels/spec/frame_stats.h"

namespace ajp::kernels::ref {

// Le signal R'G'B' décodé en cd/m², dans les primaires de la source.
Rgb frame_nits(const Rgb& encoded, FrameTransfer transfer);
// I d'ICtCp (BT.2100) d'une couleur linéaire en cd/m² sur les primaires BT.2020.
float ictcp_i(const Rgb& bt2020_nits);

// Les comptes sur la grille de points de rgb (au moins 2 × 2). DimensionOutOfRange sinon.
Result<FrameCounts> frame_counts(const RgbImage& rgb, FrameTransfer transfer);
// Les mesures, à partir des comptes (de la référence ou du noyau).
FrameStats frame_stats_of(const FrameCounts& counts);

}  // namespace ajp::kernels::ref
