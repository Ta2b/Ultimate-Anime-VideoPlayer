// Touches SDL3 vers les commandes du lecteur : la carte de core/keys.h (espace, flèches…),
// plus les touches de la tranche verticale que core::Action ne porte pas encore : « f » (plein
// écran), Échap (quitter le plein écran), « q » (quitter), « m » (muet), haut et bas (volume),
// « j » (sous-titres suivants), « a » (piste audio suivante).
// L'interface (ui.h) rend les mêmes actions.
#pragma once

#include <optional>
#include <variant>

#include <SDL3/SDL_events.h>

#include "core/keys.h"
#include "core/messages.h"

namespace ajp::app::play {

struct ToggleFullscreen {};
struct LeaveFullscreen {};
struct ToggleMute {};
struct VolumeStep {
  float delta = 0.0F;
};
struct SelectAudio {
  int track = -1;  // index du flux
};
struct SelectSubtitles {
  std::optional<int> track;  // rien : sous-titres désactivés
};
struct CycleAudio {};
struct CycleSubtitles {};
using Input = std::variant<Command, ToggleFullscreen, LeaveFullscreen, ToggleMute, VolumeStep, SelectAudio,
                           SelectSubtitles, CycleAudio, CycleSubtitles>;

// L'action d'un événement SDL, ou rien. Fermer la fenêtre vaut cmd::Quit.
std::optional<Input> input_of(const SDL_Event& event, const core::KeyMap& keys);

// La touche neutre de core d'un code de touche SDL, ou rien pour une touche sans équivalent.
std::optional<core::Key> key_of(SDL_Keycode key);

}  // namespace ajp::app::play
