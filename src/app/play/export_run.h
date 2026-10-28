// L'export d'un fichier par la chaîne de la lecture (onglet Export du menu des paramètres) : les mêmes
// moteurs (chain_setup.h) et le même étage gpu, mais sans horloge, donc sans paire sautée pour retard :
// chaque image est faite, au moins aussi fine qu'en direct. Chaque image du ring est rendue à la taille
// de l'export et convertie en YUV par libplacebo (placebo_export.h), puis encodée par l'encodeur
// matériel du GPU et écrite en MKV avec l'audio, les sous-titres, les polices et les chapitres de la
// source (export/export_writer.h). Ni RTX HDR, ni HDR adaptatif, ni grade.
//
// La lecture est fermée pendant l'export (la mémoire vidéo va aux moteurs) et la fenêtre montre la
// progression. Échap ou q annulent (le fichier partiel est supprimé) ; fermer la fenêtre quitte.
#pragma once

#include <cstdint>
#include <string>

#include "app/play/export_job.h"
#include "app/play/player_engines.h"

namespace ajp::app::play {

enum class ExportEnd : std::uint8_t {
  Done,       // fichier écrit
  Cancelled,  // Échap ou q
  Failed,     // refusé ou arrêté en route ; notice dit pourquoi
  Quit,       // fenêtre fermée pendant l'export ou sa préparation
};

struct ExportOutcome {
  ExportEnd end = ExportEnd::Done;
  std::string notice;  // à annoncer quand la lecture reprend
};

ExportOutcome run_export(Player& p, const ExportJob& job);

}  // namespace ajp::app::play
