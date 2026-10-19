// Textes français de l'écran « Moteurs » et de « ajp engines » : noms, badges,
// raisons, tailles. Partagés par la ligne de commande et, plus tard, par le dessin ImGui.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/engines.h"

namespace ajp::app {

std::string_view engine_title(core::EngineId engine);    // « TensorRT », « Maxine VFX (TrueHDR) »
std::string_view engine_summary(core::EngineId engine);  // ce qu'il apporte, en une ligne
std::string_view badge_label(core::Badge badge);         // « Recommandé », « Seulement AMD »…
std::string_view reason_text(core::Reason reason);       // pourquoi la ligne est cochée, proposée ou grisée

// Taille lisible, en unités binaires et virgule décimale : « 512 octets », « 3,4 Mio », « 1,2 Gio ».
std::string size_text(std::uint64_t bytes);

}  // namespace ajp::app
