// Des réponses de tête synthétiques « à la HeSuVi », sans fichier : deux enceintes virtuelles devant
// l'auditeur (et les six autres positions du mixeur), avec le retard interaural, l'ombre de la tête, un creux
// de pavillon et une pièce (premières réflexions, queue diffuse). Le casque en sort une sortie stéréo
// ouverte, bien plus nette que le crossfeed ; le fichier de l'utilisateur (hrir.h) la remplace s'il en
// choisit un.
#pragma once

#include "audio/hrir.h"

namespace ajp::audio {

struct SyntheticHrirParams {
  float amount = 1.0F;  // 0 : identité (chaque canal à son oreille) ; 1 : enceintes virtuelles pleines
  float room = 0.3F;    // 0 à 1 : niveau des réflexions et de la queue
  float head_radius_m = 0.0875F;  // retard interaural (Woodworth)
  int sample_rate = 48'000;
};

// Les réponses au taux demandé, déjà à l'échelle (pas de normalize : la somme des valeurs absolues d'une
// queue diffuse écraserait le niveau) : un signal stéréo sort au niveau de l'entrée à quelques dB près.
Hrir synthetic_hrir(const SyntheticHrirParams& params);

}  // namespace ajp::audio
