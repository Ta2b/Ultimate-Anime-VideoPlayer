// L'onglet Export du menu des paramètres : la chaîne de l'export (les onglets Image et Fluidité, repris
// des réglages de la lecture), la résolution, le codec, la qualité et le fichier. « Exporter » pousse
// StartExport ; export_run.h fait le reste, la lecture fermée, puis la lecture reprend au même endroit.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "app/play/export_job.h"
#include "app/play/input.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

struct ExportDraft {
  ExportJob job;
  bool chain_set = false;         // job.chain reprise de la lecture à la première ouverture de l'onglet
  std::array<char, 1024> path{};  // le fichier, en UTF-8, en cours de saisie
  bool path_edited = false;       // tant qu'il ne l'est pas, le nom suit la taille et la fréquence
  std::string named_for;          // la taille et la fréquence du nom proposé
};

void export_tab(ExportDraft& draft, const PlaySettings& running, const SettingsChoices& choices,
                std::vector<Input>& actions);

}  // namespace ajp::app::play
