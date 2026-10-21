// Un export demandé par l'onglet Export du menu des paramètres : la chaîne (onglets Image et Fluidité),
// la taille, le codec, la qualité et le fichier. Les calculs purs de l'export (taille encodée,
// fréquence de sortie, nom du fichier) vivent ici, testés sans GPU ; export_run.h le réalise.
#pragma once

#include <filesystem>
#include <string>

#include "app/play/options.h"
#include "app/play/settings.h"
#include "export/encoder_choice.h"
#include "media/track.h"

namespace ajp::app::play {

// La hauteur encodée : celle de la sortie de la chaîne, celle de la source, ou une hauteur fixe.
inline constexpr int kExportChainHeight = 0;
inline constexpr int kExportSourceHeight = -1;

struct ExportJob {
  PlaySettings chain;               // la chaîne de l'export ; seuls Image et Fluidité comptent (export_chain)
  int height = kExportChainHeight;  // ou kExportSourceHeight, ou un nombre de lignes
  exporter::Codec codec = exporter::Codec::Hevc;
  exporter::Quality quality = exporter::Quality::VeryHigh;
  bool constant_rate = true;     // WriterConfig::constant_rate
  std::filesystem::path output;  // .mkv
  friend bool operator==(const ExportJob&, const ExportJob&) = default;
};

// La chaîne d'un export : celle des réglages sans RTX HDR, HDR adaptatif ni grade, sans Smooth Motion
// (une couche de l'écran) : l'image de la chaîne telle quelle, comme la lecture la donne en SDR.
PlaySettings export_chain(const PlaySettings& s);

struct ExportSize {
  int width = 0;
  int height = 0;
  friend bool operator==(const ExportSize&, const ExportSize&) = default;
};

// La sortie de la chaîne pour une source : la taille de travail (work_size), doublée par SPAN.
ExportSize chain_output(const PlaySettings& chain, int source_width, int source_height);

// La taille encodée : la sortie de la chaîne (kExportChainHeight), la source (kExportSourceHeight),
// ou height lignes au rapport de la sortie de la chaîne. Paire (4:2:0), au moins 2 × 2.
ExportSize export_size(ExportSize chain, ExportSize source, int height);

// La fréquence des images exportées : celle de la source multipliée par l'interpolation (rien de
// plus sans elle), réduite. {0, 1} si celle de la source est inconnue.
media::Rational export_rate(media::Rational source, Factor factor);

// « Épisode 01 [ajp 3840x2160 47.952fps].mkv » à côté de la source ; « (2) », « (3) »… si le nom
// est pris.
std::filesystem::path default_export_path(const std::filesystem::path& source, ExportSize size,
                                          media::Rational rate);

// « HEVC 10 bits · qualité très haute », « Sans perte (FFV1) » : pour l'interface.
std::string export_label(exporter::Codec codec, exporter::Quality quality);

}  // namespace ajp::app::play
