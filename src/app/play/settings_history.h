// Annuler / rétablir du panneau des réglages. L'historique ne garde que ce que l'utilisateur change dans le
// panneau : le volume, le délai audio (touches - et =), le délai des sous-titres (z et x) et les statistiques
// (Tab) changent aussi hors de lui, et un changement de ce genre ne crée pas d'entrée ni ne revient en
// arrière avec « Annuler » (B23-01).
#pragma once

#include <vector>

#include "app/play/settings.h"

namespace ajp::app::play {

// Ce que le lecteur a changé seul entre deux tours du panneau.
struct ExternalChange {
  bool volume = false;
  bool audio_delay = false;
  bool subtitle_delay = false;
  bool stats = false;
  bool any() const { return volume || audio_delay || subtitle_delay || stats; }
};

// running a changé depuis seen (le tour précédent) : un réglage est « venu d'ailleurs » quand running n'est
// pas la valeur que le brouillon montre déjà (un changement du panneau arrive aussi dans running, par la part
// directe, et celui-là reste annulable).
ExternalChange external_changes(const PlaySettings& seen, const PlaySettings& running,
                                const PlaySettings& draft);

class SettingsHistory {
 public:
  // L'ouverture du panneau : l'état posé, sans entrée.
  void reset(const PlaySettings& settled);
  bool can_undo() const { return !undo_.empty(); }
  bool can_redo() const { return !redo_.empty(); }
  // Aucun curseur n'est tenu : draft devient l'état posé, et l'état d'avant une entrée d'annulation s'il en
  // diffère (un glissement ne compte que pour un changement).
  void settle(const PlaySettings& draft);
  // Annuler : l'état à reprendre ; draft passe dans « rétablir ». Sans entrée, draft rendu tel quel.
  PlaySettings undo(const PlaySettings& draft);
  PlaySettings redo(const PlaySettings& draft);
  // Des réglages changés hors du panneau : leur valeur de running remplace celle de tous les états gardés,
  // pour qu'aucun ne les rejoue ni ne les compte comme un changement.
  void adopt(const ExternalChange& changed, const PlaySettings& running);

 private:
  PlaySettings stable_;
  std::vector<PlaySettings> undo_;
  std::vector<PlaySettings> redo_;
};

}  // namespace ajp::app::play
