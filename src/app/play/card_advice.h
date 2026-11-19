// Les recommandations des infobulles selon la carte : la tranche de la machine (gpu_tier.h), classée par
// puissance pour kirari (docs/classement-gpu.md). Une 4070 Ti et une 1080 Ti ne reçoivent pas le même
// conseil.
#pragma once

#include <string>
#include <string_view>

#include "app/play/gpu_tier.h"

namespace kirari::app::play {

struct SettingsChoices;

struct CardAdvice {
  Tier tier = Tier::Normal;
  bool tensor_cores = false;  // une NVIDIA RTX : GMFSS et TensorRT
};
CardAdvice card_advice(const SettingsChoices& choices);

// « Recommandé : {valeur} (niveau Rapide de cette carte). », la fin d'une infobulle (split_tip).
std::string recommended_for(const CardAdvice& card, std::string_view value);

// Les valeurs conseillées, comme le préréglage Auto de la tranche les règle (auto_config.cpp).
std::string upscale_advice(const CardAdvice& card);      // « désactivé » en Économe, sinon « activé »
std::string model_advice(const CardAdvice& card);        // le modèle de la tranche
std::string work_height_advice(const CardAdvice& card);  // 540p, 720p, la source jusqu'à 1080p
std::string interpolation_level_advice(const CardAdvice& card);
std::string light_motion_advice(const CardAdvice& card);  // GMFSS et RIFE en 540p

// La phrase de recommandation d'une valeur conseillée pour la carte de choices.
std::string advice_line(const SettingsChoices& choices, std::string (*value)(const CardAdvice&));

}  // namespace kirari::app::play
