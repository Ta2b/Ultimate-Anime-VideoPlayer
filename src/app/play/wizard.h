// L'assistant de premier démarrage (docs/ux-parametres.md, 4.8) : huit écrans guidés, tous sautables, dont
// certains n'ont pas lieu selon la machine. Ici la logique, sans dessin : quels écrans, dans quel ordre, ce
// qui a été choisi ou sauté, le résumé final et ce qui s'applique à « Terminer ». Le dessin est dans
// first_run.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/settings.h"
#include "core/settings.h"

namespace ajp::app::play {

// Dans l'ordre de l'assistant ; la valeur est celle de first_run_step (1 à 8).
enum class WizardStep : std::uint8_t {
  Welcome = 1,  // carte, tranche, écran
  Engines,      // moteurs à installer
  Chain,        // la calibration de la machine (machine_calibration.h), faite après « Terminer »
  Hdr,          // HDR / SDR
  Grade,        // retiré (la calibration du grade adaptatif) : son rang reste pris, first_run_step le garde
  Languages,    // audio et sous-titres préférés
  Updates,      // Me prévenir / Automatique / Jamais
  Summary,      // ce qui a été choisi, Terminer
};
inline constexpr int kWizardSteps = 8;

std::string_view title(WizardStep step);  // « Bienvenue », « Moteur », « Image et fluidité »…

// Ce que la machine permet : les écrans sans objet sont sautés.
struct WizardFacts {
  bool engine_choice = true;     // un moteur autre que Vulkan est possible (sinon pas d'écran Moteur)
  bool hdr_display = false;      // l'écran reçoit du HDR (sinon pas d'écran HDR)
  bool updates_managed = false;  // paquet de distribution, Flatpak : mis à jour ailleurs (écran grisé, gardé)
};

// Les écrans de cette machine, dans l'ordre.
std::vector<WizardStep> wizard_steps(const WizardFacts& facts);

enum class HdrChoice : std::uint8_t { Auto, Hdr, Sdr };
std::string_view label(HdrChoice choice);

// L'état de l'assistant : l'écran courant, ce qui est choisi, ce qui est sauté.

struct Wizard {
  std::vector<WizardStep> steps;
  std::size_t at = 0;
  std::vector<WizardStep> skipped;
  HdrChoice hdr = HdrChoice::Auto;
  std::string update_mode = "notify";
  std::string update_channel = "stable";
  // Première préférence de langue de l'audio (vide : selon le fichier) et des sous-titres (vide : aucun),
  // celle par défaut d'abord : audio japonais, sous-titres dans la langue du système.
  std::string audio_language = default_language_presets().front().audio;
  std::string subtitle_language = default_language_presets().front().subtitles;
  int seen = 0;  // le plus loin atteint (rang de WizardStep), pour reprendre en route
};

// Reprend à l'écran noté dans first_run_step (le suivant du dernier vu), sans dépasser le résumé.
Wizard begin_wizard(const WizardFacts& facts, int last_step_seen);

WizardStep current(const Wizard& wizard);
bool is_first(const Wizard& wizard);
bool is_last(const Wizard& wizard);
void next(Wizard& wizard);      // Suivant
void previous(Wizard& wizard);  // Précédent
void skip(Wizard& wizard);      // Passer : l'écran est noté sauté, on avance
// « Étape 3 sur 8 » : le rang parmi les écrans de cette machine.
std::string progress_text(const Wizard& wizard);

// Le résumé final : une ligne par choix fait ou sauté, avec où le changer.
std::vector<std::string> wizard_summary(const Wizard& wizard);

// Ce que « Terminer » écrit : les mises à jour, le dernier écran vu.
core::Settings wizard_settings(const Wizard& wizard, core::Settings settings);
// Les réglages de lecture : HDR, langues, mises à jour choisis. Un écran sauté n'écrase rien. La calibration
// de la machine sautée : le préréglage Auto.
PlaySettings wizard_play_settings(const Wizard& wizard, PlaySettings settings);
// « Terminer » lance-t-il la calibration de la machine (écran non sauté) ?
bool wants_machine_calibration(const Wizard& wizard);

}  // namespace ajp::app::play
