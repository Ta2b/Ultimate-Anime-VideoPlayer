// Le menu des paramètres pendant une préparation (construction d'un moteur, préparation du SPAN FP8) : on
// peut y changer de moteur, de modèle ou de résolution sans attendre la fin. Un réglage en direct (couleurs,
// son) attend que la lecture parte, puis s'applique ; un réglage qui demande de recharger arrête la
// préparation, et la vidéo se rouvre aussitôt avec lui, au même endroit (player_run.cpp : run_chain).
#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "app/play/input.h"
#include "app/play/settings.h"

namespace kirari::app::play {

class PlayerUi;

// Ce que le menu a demandé pendant la préparation en cours.
struct LoadingChange {
  std::optional<PlaySettings> live;     // en direct : appliqué dès que la lecture part
  std::optional<PlaySettings> reopen;   // « Appliquer (recharge la vidéo) »
  std::optional<EngineChange> engines;  // un autre moteur, ou l'écran des moteurs
  std::optional<PlaySettings> restart;  // « Appliquer (relance le lecteur) »
  bool stops() const { return reopen || engines || restart; }
  // Les réglages avec lesquels rouvrir : ceux d'« Appliquer », sinon les derniers en direct, sinon `current`.
  PlaySettings settings_or(const PlaySettings& current) const;
};

// La clé du refus d'une préparation arrêtée par un réglage du menu : rien n'est retenu contre l'étage.
inline constexpr std::string_view kSettingsChangedKey = "engine.settings-changed";

// Les entrées d'un tour de préparation qui touchent au menu : p (ou le bouton « Paramètres ») l'ouvre ou le
// ferme, ses réglages vont dans `change`. Les autres entrées restent dans `actions`.
void take_loading_inputs(PlayerUi& ui, std::vector<Input>& actions, LoadingChange& change);

}  // namespace kirari::app::play
