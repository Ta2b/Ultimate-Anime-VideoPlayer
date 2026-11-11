// L'assistant de premier démarrage (docs/ux-parametres.md, 4.8) : cinq écrans guidés au plus, l'écran des
// moteurs n'ayant lieu que si un moteur autre que Vulkan est possible (P3.18). Ici la logique, sans dessin :
// quels écrans, dans quel ordre, ce qui a été choisi ou sauté, le résumé final et ce qui s'applique à «
// Terminer ». Le dessin est dans first_run.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/settings.h"
#include "core/settings.h"

namespace ajp::app::play {

// Dans l'ordre de l'assistant (P3.18) : les choix légers d'abord, le lourd (moteurs) en dernier, puis Prêt
// avec « Calibrer maintenant » ou « Régler moi-même ». La valeur est celle de first_run_step (1 à 5) ; une
// valeur plus haute d'une version d'avant (1 à 8) reprend au dernier écran.
enum class WizardStep : std::uint8_t {
  Welcome = 1,  // carte, tranche, écran, langue de l'interface
  Languages,    // audio et sous-titres préférés
  Updates,      // intégration : mises à jour (Me prévenir / Jamais), lecteur par défaut
  Engines,      // moteurs à installer (TensorRT sur RTX, Core ML sur Mac)
  Summary,      // Prêt : ce qui a été choisi, la calibration proposée
};
inline constexpr int kWizardSteps = 5;

std::string_view title(WizardStep step);  // « Bienvenue », « Langues », « Intégration », « Moteur », « Prêt »

// Ce que la machine permet : les écrans sans objet sont sautés.
struct WizardFacts {
  bool engine_choice = true;     // un moteur autre que Vulkan est possible (sinon pas d'écran Moteur)
  bool updates_managed = false;  // paquet de distribution, Flatpak : mis à jour ailleurs (écran grisé, gardé)
};

// Les écrans de cette machine, dans l'ordre.
std::vector<WizardStep> wizard_steps(const WizardFacts& facts);

// L'état de l'assistant : l'écran courant, ce qui est choisi, ce qui est sauté.

struct Wizard {
  std::vector<WizardStep> steps;
  std::size_t at = 0;
  std::vector<WizardStep> skipped;
  UpdateMode update_mode = UpdateMode::Notify;
  // Première préférence de langue de l'audio (vide : selon le fichier) et des sous-titres (vide : aucun),
  // celle par défaut d'abord : audio japonais, sous-titres dans la langue du système.
  std::string audio_language = default_language_presets().front().audio;
  std::string subtitle_language = default_language_presets().front().subtitles;
  std::string interface_language;  // « en » ou « fr » ; vide : celle détectée, inchangée
  bool make_default = false;       // lecteur par défaut des vidéos, demandé à « Terminer »
  bool calibrate = true;           // « Calibrer maintenant » ; « Régler moi-même » l'éteint
  int seen = 0;                    // le plus loin atteint (rang de WizardStep), pour reprendre en route
};

// Le lecteur par défaut est-il demandé ? Non si l'écran qui le propose a été passé (A25-04).
bool make_default_requested(const Wizard& wizard);

// Reprend à l'écran noté dans first_run_step (le suivant du dernier vu), sans dépasser le résumé.
Wizard begin_wizard(const WizardFacts& facts, int last_step_seen);

WizardStep current(const Wizard& wizard);
bool is_first(const Wizard& wizard);
bool is_last(const Wizard& wizard);
void next(Wizard& wizard);      // Suivant
void previous(Wizard& wizard);  // Précédent
void skip(Wizard& wizard);      // Passer : l'écran est noté sauté, on avance
// « Régler moi-même » (P3.1) : la calibration sautée, le préréglage Auto estimé comme point de départ.
void set_up_myself(Wizard& wizard);
// « Étape 3 sur 5 » : le rang parmi les écrans de cette machine.
std::string progress_text(const Wizard& wizard);

// Le résumé final : une ligne par choix fait ou sauté, avec où le changer.
std::vector<std::string> wizard_summary(const Wizard& wizard);

// Ce que « Terminer » écrit : les mises à jour, le dernier écran vu.
core::Settings wizard_settings(core::Settings settings);
// Les réglages de lecture : HDR, langues, mises à jour choisis. Un écran sauté n'écrase rien. La calibration
// de la machine sautée : le préréglage Auto.
PlaySettings wizard_play_settings(const Wizard& wizard, PlaySettings settings);
// P3.3 : l'assistant repris en route retrouve les choix enregistrés à chaque écran (mises à jour, langues).
void resume_choices(Wizard& wizard, const PlaySettings& saved);
// « Calibrer maintenant » : la calibration de la machine après l'assistant.
bool wants_machine_calibration(const Wizard& wizard);

}  // namespace ajp::app::play
