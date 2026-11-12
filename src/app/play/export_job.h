// Un export demandé par l'onglet Export du menu des paramètres : la chaîne (onglets Image et Fluidité),
// la taille, le codec, la qualité et le fichier. Les calculs purs de l'export (taille encodée,
// fréquence de sortie, nom du fichier) vivent ici, testés sans GPU ; export_run.h le réalise.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "app/play/options.h"
#include "app/play/settings.h"
#include "core/frame.h"
#include "export/container.h"
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
  exporter::CopyChoice copies;   // les pistes choisies à la lecture (P3.90)
  friend bool operator==(const ExportJob&, const ExportJob&) = default;
};

// La chaîne d'un export : celle des réglages sans HDR ni grade, sans Smooth Motion
// (une couche de l'écran) : l'image de la chaîne telle quelle, comme la lecture la donne en SDR. L'Auto
// devient ×2, qui ne dépend pas de l'écran (P3.87). Le grade se pose au rendu de l'export (export_grade).
PlaySettings export_chain(const PlaySettings& s);

// Le grade HDR anime à l'export, comme à la lecture : rien quand il est coupé ou que la source est déjà HDR
// (elle passe telle quelle) ; sinon un fichier HDR10 (PQ BT.2020) dans un codec 10 bits, ou, en H.264 (8
// bits, où le PQ ferait des paliers), le même grade ramené au SDR BT.709, comme sur un écran SDR.
enum class ExportGrade : std::uint8_t { None, Hdr10, Sdr };
ExportGrade export_grade(const PlaySettings& chain, bool source_hdr, exporter::Codec codec);
// Le pic d'un export HDR10 : l'écran n'est pas connu, le grade vise celui que la lecture prend quand l'écran
// ne dit pas le sien, et le fichier l'annonce (mastering et MaxCLL).
inline constexpr int kExportPeakNits = 1000;
// L'étiquette du flux d'un export gradé : HDR10 au pic kExportPeakNits, ou SDR BT.709 ; rien sans grade
// (celle de la source, ExportWriter).
std::optional<ColorInfo> export_color(ExportGrade grade);

struct ExportSize {
  int width = 0;
  int height = 0;
  friend bool operator==(const ExportSize&, const ExportSize&) = default;
};

// La sortie de la chaîne pour une source : la taille de travail (chain_work_size), doublée quand SPAN tourne.
ExportSize chain_output(const PlaySettings& chain, int source_width, int source_height);

// La taille encodée : la sortie de la chaîne (kExportChainHeight), la source (kExportSourceHeight),
// ou height lignes au rapport de la sortie de la chaîne. Paire (4:2:0), au moins 2 × 2.
ExportSize export_size(ExportSize chain, ExportSize source, int height);

// La fréquence des images exportées : celle de la source multipliée par l'interpolation (rien de
// plus sans elle), réduite. {0, 1} si celle de la source est inconnue.
media::Rational export_rate(media::Rational source, Factor factor);

// Le dossier est sur un FAT (FAT32, vfat, msdos) : un fichier n'y dépasse pas 4 Gio (P3.89).
bool on_fat32(const std::filesystem::path& dir);

// « Épisode 01 [ajp 3840x2160 47.952fps].mkv » à côté de la source ; « (2) », « (3) »… si le nom
// est pris ; un chemin vide si « (999) » l'est aussi. Le nom de la source est borné à 160 octets.
std::filesystem::path default_export_path(const std::filesystem::path& source, ExportSize size,
                                          media::Rational rate);

// « HEVC 10 bits · qualité très haute », « FFV1 4:2:0 10 bits, sans perte du codec » : pour l'interface.
std::string export_label(exporter::Codec codec, exporter::Quality quality);

}  // namespace ajp::app::play
