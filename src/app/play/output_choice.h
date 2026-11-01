// Après la calibration de la machine de l'assistant, sur un écran HDR : la sortie choisie, HDR (la
// calibration du grade adaptatif s'ouvre alors) ou SDR.
#pragma once

#include <cstdint>

namespace ajp::app::play {

enum class OutputPick : std::uint8_t { None, Hdr, Sdr };

// La fenêtre du choix, au centre de l'écran ; rend le bouton pressé à cette image.
OutputPick draw_output_choice(float scale);

}  // namespace ajp::app::play
