// Le rythme de l'accueil (A24-17). Sans vidéo, rien ne bouge à l'écran tant que personne n'y touche :
// redessiner et présenter 60 fois par seconde tenait le GPU et le compositeur éveillés pour une image fixe.
// Après kHomeIdleAfterUs sans événement, l'accueil attend kHomeIdleMs entre deux images ; un événement
// (souris, clavier, fichier glissé) réveille l'attente aussitôt et rend le rythme plein. 10 images par
// seconde au repos : ce qui avance sans événement (téléchargement d'un moteur, banc, vérification de mise à
// jour) reste suivi. Pur.
#pragma once

#include <cstdint>

namespace ajp::app::play {

inline constexpr std::int32_t kHomeFrameMs = 16;
inline constexpr std::int32_t kHomeIdleMs = 100;
inline constexpr std::int64_t kHomeIdleAfterUs = 2'000'000;

// L'attente avant l'image suivante. busy : quelque chose avance sans événement et se regarde (un message
// affiché, une boîte de fichiers ouverte, une séance scriptée par --keys ou --snapshot, dont les instants
// comptent).
constexpr std::int32_t home_wait_ms(std::int64_t now_us, std::int64_t last_event_us, bool busy) {
  return !busy && now_us - last_event_us >= kHomeIdleAfterUs ? kHomeIdleMs : kHomeFrameMs;
}

}  // namespace ajp::app::play
