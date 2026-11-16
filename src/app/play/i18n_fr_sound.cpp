#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Audio : (clé anglaise, français).
constexpr std::array kSoundPage{
    Entry{"Audio delay (ms)", "Décalage audio (ms)"},
    Entry{"Audio delay: {} ms", "Décalage audio : {} ms"},
    Entry{"{:.0f} fps: many TVs and phones refuse or stutter above 60.",
          "{:.0f} i/s : beaucoup de téléviseurs et de téléphones refusent ou saccadent au-delà de 60."},
    Entry{"Device", "Appareil"},
    Entry{"Display latency (ms)", "Latence de l'écran (ms)"},
    Entry{"Follow the system default device", "Suivre l'appareil par défaut du système"},
    Entry{"Output device", "Appareil de sortie"},
    Entry{"{} (not connected)", "{} (non connecté)"},
    Entry{"Reset to 0", "Remettre à 0"},
    Entry{"Sync", "Synchronisation"},
    Entry{"Sync playback to the display", "Caler la lecture sur l'écran"},
    Entry{"estimated", "estimée"},
    Entry{"{} channels on this device.", "{} canaux sur cet appareil."},
};

}  // namespace

std::span<const Entry> french_sound() {
  return kSoundPage;
}

}  // namespace ajp::app::play::i18n
