// DistilDRBA fusionné (kernels/spec/drba_fused.h) : faut-il l'essayer pour cette lecture ? Il l'est quand le
// réglage le veut, que le moteur le fait et que DistilDRBA est le moteur du mouvement au cran Équilibre (le
// modèle tel que livré). Ce qui reste, la reconnaissance du modèle et la construction des quatre moteurs, se
// juge en les tentant (player_engines.cpp) : un refus garde le DistilDRBA entier.
#pragma once

#include <string>

#include "app/play/precision.h"
#include "app/play/settings.h"

namespace ajp::app::play {

// Vide : DistilDRBA fusionné s'essaie. Sinon la raison, pour le journal. Le cran de précision se juge à part
// (drba_tier), une fois le modèle trouvé.
std::string drba_fused_declined(const PlaySettings& settings, bool backend_does_it);

// Ce que DistilDRBA charge pour le cran demandé (A31-06, A31-07) :
//  - Qualité avec son graphe « _fp32 » : ce graphe, en moteur entier ;
//  - Performance : la chaîne fusionnée FP16 quand elle s'essaie, plus rapide et plus juste que le graphe
//    « _fp8 » entier (tools/release/publier-modeles-fp8.sh) ; ce graphe seulement sans elle, sur un moteur
//    FP8 ;
//  - tout le reste, dont un cran dont le graphe manque : le modèle livré, fusionné s'il s'essaie.
struct DrbaTier {
  PrecisionTier tier = PrecisionTier::Balanced;  // le graphe à charger
  bool fused = false;                            // la chaîne fusionnée s'essaie
  std::string why;                               // vide si le cran demandé est rendu tel quel
};
DrbaTier drba_tier(PrecisionTier asked, bool fused_possible, TierSupport graphs);

}  // namespace ajp::app::play
