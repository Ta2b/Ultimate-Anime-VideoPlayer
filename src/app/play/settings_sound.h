#pragma once

// L'onglet Audio du panneau des réglages (settings_sound.cpp).
#include <string>
#include <string_view>

#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

// Vrai pendant qu'on tient le curseur du délai audio.
bool sound_tab(PlaySettings& d, const SettingsChoices& choices);
// Un autre appareil choisi (audio_device différent de before) : le délai du précédent est gardé pour lui,
// celui du nouveau repris, 0 s'il n'en a pas (P2.7). system_default : la carte que le système prend ; la
// suivre ou la nommer est le même appareil, le délai reste (B23-06).
void switch_device_delay(PlaySettings& d, const std::string& before, std::string_view system_default = {});

}  // namespace ajp::app::play
