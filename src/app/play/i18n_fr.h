#pragma once

#include <span>

#include "app/play/i18n.h"

namespace ajp::app::play::i18n {

// Les tables françaises, une par zone de l'interface ; french_entries() les réunit.
std::span<const Entry> french_player();    // barre de lecture, pistes, notifications
std::span<const Entry> french_settings();  // pages de réglages

}  // namespace ajp::app::play::i18n
