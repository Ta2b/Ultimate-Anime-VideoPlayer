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
  Hdr,          // RTX HDR / HDR adaptatif / SDR
  Grade,        // la calibration du grade HDR, ouverte après celle de la machine (écran HDR seulement)
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
  bool rtx_guided = false;       // Linux et carte NVIDIA RTX : RTX HDR se prépare pas à pas (4b)
};

// Les écrans de cette machine, dans l'ordre.
std::vector<WizardStep> wizard_steps(const WizardFacts& facts);

enum class HdrChoice : std::uint8_t { Auto, RtxHdr, Adaptive, Sdr };
std::string_view label(HdrChoice choice);

// L'état de l'assistant : l'écran courant, ce qui est choisi, ce qui est sauté.
// L'écran 4b (RTX HDR pas à pas, Linux et NVIDIA) : un point à vérifier et, s'il manque, le geste à faire.
struct RtxCheck {
  std::string name;
  bool done = false;
  std::string todo;
};
bool rtx_ready(const std::vector<RtxCheck>& checks);  // tous faits (et au moins un point)

struct Wizard {
  std::vector<WizardStep> steps;
  std::size_t at = 0;
  std::vector<WizardStep> skipped;
  HdrChoice hdr = HdrChoice::Auto;
  std::string update_mode = "notify";
  std::string update_channel = "stable";
  std::string audio_language = "jpn";  // première préférence de langue de l'audio ; vide : selon le fichier
  std::string subtitle_language = "fre";  // et des sous-titres ; vide : aucun
  std::vector<RtxCheck>
      rtx_checks;  // relus par « Vérifier de nouveau » (4b) ; vide : pas de guide (Windows, hors NVIDIA)
  int seen = 0;    // le plus loin atteint (rang de WizardStep), pour reprendre en route
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
// « Terminer » lance-t-il la calibration de la machine (écran non sauté) ? Et celle du grade HDR (écran non
// sauté, et un HDR choisi) ?
bool wants_machine_calibration(const Wizard& wizard);
bool wants_grade_calibration(const Wizard& wizard);

}  // namespace ajp::app::play
