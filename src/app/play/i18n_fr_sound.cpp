#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Audio : (clé anglaise, français).
constexpr std::array kSoundPage{
    Entry{"Audio delay (ms)", "Délai audio (ms)"},
    Entry{"Audio delay: {} ms", "Délai audio : {} ms"},
    Entry{"{:.0f} fps: many TVs and phones refuse or stutter above 60.",
          "{:.0f} i/s : beaucoup de téléviseurs et de téléphones refusent ou saccadent au-delà de 60."},
    Entry{"Changing the device reopens the video at the same spot.",
          "Changer d'appareil rouvre la vidéo au même endroit."},
    Entry{"Device", "Appareil"},
    Entry{"Display latency (ms)", "Latence de l'écran (ms)"},
    Entry{"Follow the system default device", "Suivre l'appareil par défaut du système"},
    Entry{"On a fixed-rate display, playback speeds up or slows down by at most 0.125% so each frame fits "
          "whole refreshes: no more doubled frames. Sound follows, inaudibly.",
          "Sur un écran à fréquence fixe, la lecture accélère ou ralentit d'au plus 0,125 % pour que chaque "
          "image tienne des rafraîchissements entiers : plus d'image doublée. Le son suit, sans changement "
          "audible."},
    Entry{"Output device", "Appareil de sortie"},
    Entry{
        "Positive delays the sound. Kept for this output device and every video, until you change it. "
        "Ctrl+click the slider to type a value (up to ±5,000 ms).",
        "Positif : le son est retardé. Gardé pour cet appareil de sortie et pour toutes les vidéos, jusqu'à "
        "ce que vous le changiez. Ctrl+clic sur le curseur pour saisir une valeur (jusqu'à ±5 000 ms)."},
    Entry{"{} (not connected)", "{} (non connecté)"},
    Entry{"Reset to 0", "Remettre à 0"},
    Entry{"Sync", "Synchronisation"},
    Entry{"Sync playback to the display", "Caler la lecture sur l'écran"},
    Entry{"The time between choosing a frame and seeing it. Set it here if the picture seems early or late.",
          "Le temps entre le choix d'une image et son apparition. À régler ici si l'image paraît en avance "
          "ou en retard sur le son."},
    Entry{"estimated", "estimée"},
    Entry{"{} channels on this device.", "{} canaux sur cet appareil."},
};

}  // namespace

std::span<const Entry> french_sound() {
  return kSoundPage;
}

}  // namespace ajp::app::play::i18n
