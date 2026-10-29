#pragma once

#include <span>

#include "app/play/i18n.h"

namespace ajp::app::play::i18n {

// Les tables françaises, une par zone de l'interface ; french_entries() les réunit.
std::span<const Entry> french_player();    // barre de lecture, pistes, notifications
std::span<const Entry> french_settings();  // pages de réglages
std::span<const Entry> french_image();     // Image IA : agrandissement et fluidité
std::span<const Entry> french_system();    // Export, moteurs, application, mises à jour, configuration
std::span<const Entry> french_wizard();    // assistant de premier lancement, écran des moteurs
std::span<const Entry> french_stats();     // statistiques à l'écran, export, accueil
std::span<const Entry> french_sound();     // Audio : sortie, profils, son anime, synchronisation
std::span<const Entry>
french_messages();  // lecteur par défaut, mises à jour, benchmark, installation des moteurs

}  // namespace ajp::app::play::i18n
