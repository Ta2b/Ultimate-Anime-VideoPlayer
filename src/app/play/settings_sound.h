#pragma once

// L'onglet Audio du panneau des réglages (settings_sound.cpp).
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

// Vrai pendant qu'on tient le curseur du délai audio.
bool sound_tab(PlaySettings& d, const SettingsChoices& choices);

}  // namespace ajp::app::play
