// Les infobulles du menu des paramètres, mises en forme à la manière de Dolphin : le nom du réglage en gras,
// une phrase, la vitesse et l'impact sur la qualité sur leurs lignes, la recommandation en bleu, et en ambre
// ce que le changement demande (recharger la vidéo, redémarrer le lecteur).
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ajp::app::play::panel {

// Ce qu'un réglage demande pour s'appliquer : rien (en direct), rouvrir la vidéo au même endroit, ou relancer
// le lecteur (settings.h : ChangeCost). Le pied du panneau le dit de la même manière pour tous.
enum class Apply : std::uint8_t { Live, Reload, Restart };

// Le texte d'une infobulle découpé selon ses repères (« Speed: », « Vitesse : »…), dans la langue de
// l'interface : la description d'abord, puis les champs reconnus, dans l'ordre de l'écran.
struct TipParts {
  std::string text;         // la description, sans les champs
  std::string speed;        // « environ −3 % »
  std::string quality;      // « gain mineur »
  std::string recommended;  // « activé »
  std::string after;        // ce qui suit la recommandation (« RIFE seulement. »)
};
TipParts split_tip(std::string_view text);

// L'infobulle du dernier élément, après un court délai, même grisé. title : le nom du réglage (vide : sans
// titre). body passe par split_tip.
void tip(const char* title, std::string_view body, Apply apply = Apply::Live);
// Sans titre ni repère : une phrase seule (notes, choix d'une liste).
void tip(const char* body);
inline void tip(const std::string& body) {
  tip(body.c_str());
}

}  // namespace ajp::app::play::panel
