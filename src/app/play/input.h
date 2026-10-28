// Touches SDL3 vers les commandes du lecteur : la carte de core/keys.h (espace, flèches…),
// plus les touches de la tranche verticale que core::Action ne porte pas encore : « f » (plein
// écran), Échap (quitter le plein écran), « q » (quitter), « m » (muet), haut et bas (volume),
// « j » (sous-titres suivants), « a » (piste audio suivante), « p » (paramètres),
// Page précédente et Page suivante (chapitre suivant et précédent, comme dans mpv).
// L'interface (ui.h) rend les mêmes actions.
#pragma once

#include <optional>
#include <variant>

#include <SDL3/SDL_events.h>

#include "app/play/export_job.h"
#include "app/play/settings.h"
#include "core/engines.h"
#include "core/keys.h"
#include "core/messages.h"
#include "core/user_overlay.h"

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
// Chapitre suivant (+1) ou précédent (−1) : Page précédente et Page suivante, comme dans mpv.
struct StepChapter {
  int direction = 1;
};
struct CycleSubtitles {};
struct ToggleSettings {};
// Épisode suivant (+1) ou précédent (−1) du dossier (episodes.h) : « n » et Maj + « n ».
struct StepEpisode {
  int direction = 1;
};
// Passer le générique (opening ou ending) : « o ».
struct SkipCredits {};
// Modification d'un profil dans user.toml (nom, profil de démarrage, retour à la base).
struct EditProfile {
  core::UserEdit edit;
};
// Réglages du menu des paramètres : en direct, ou confirmés pour rouvrir la chaîne.
struct ApplySettings {
  PlaySettings settings;
  bool reopen = false;
};
// Onglet Moteurs : SPAN et RIFE sur un autre moteur installé (engine_switch.h), ou l'écran des
// moteurs rouvert ; la chaîne se rouvre ensuite au même endroit.
struct UseEngine {
  core::EngineId id = core::EngineId::Vulkan;
};
// B : bascule A/B du son, traité (casque, enceintes…) contre fidèle (aucun traitement).
struct ToggleFaithful {};
struct OpenEngineScreen {
  bool wizard = false;  // Général > Configuration > « Relancer l'assistant » : les huit écrans, pas seulement les moteurs
};
using EngineChange = std::variant<UseEngine, OpenEngineScreen>;
// Onglet Export : la lecture se ferme, l'export se fait (export_run.h), puis la lecture reprend.
struct StartExport {
  ExportJob job;
};
using Input =
    std::variant<Command, ToggleFullscreen, LeaveFullscreen, ToggleMute, VolumeStep, SelectAudio,
                 SelectSubtitles, CycleAudio, CycleSubtitles, StepChapter, StepEpisode, SkipCredits,
                 ToggleSettings, ApplySettings, EditProfile, UseEngine, OpenEngineScreen, StartExport, ToggleFaithful>;

// L'action d'un événement SDL, ou rien. Fermer la fenêtre vaut cmd::Quit.
std::optional<Input> input_of(const SDL_Event& event, const core::KeyMap& keys);

// La touche neutre de core d'un code de touche SDL, ou rien pour une touche sans équivalent.
std::optional<core::Key> key_of(SDL_Keycode key);

}  // namespace ajp::app::play
