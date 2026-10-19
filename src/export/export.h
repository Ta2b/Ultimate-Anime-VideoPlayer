// Export : le même Plan que la lecture, sans horloge. Les images finales de render sont
// encodées (libavcodec), puis muxées (libavformat, Matroska ou MP4) avec les pistes audio et
// sous-titres, les chapitres et les pièces jointes du fichier source, et les métadonnées HDR10.
//
// run_export() tourne sans fenêtre : la ligne de commande ajp export l'appelle telle quelle ;
// le lecteur la lance sur son thread export (export_thread.h). Deux passes de mux :
//   1. les images sont encodées dans un fichier vidéo seul (MP4, qui garde dts et microsecondes),
//      à côté de la sortie, pendant que MaxCLL et MaxFALL se mesurent ;
//   2. le mux final recopie ce flux et les pistes du fichier source, horodatages recalés sur la
//      première image exportée, et écrit les niveaux de lumière mesurés : Matroska et MP4 les
//      portent dans l'en-tête du flux, écrit avant les images.
// La sortie s'écrit sous « <sortie>.part », renommée à la fin : un export refusé ou annulé ne
// laisse aucun fichier.
#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"
#include "export/encoder_settings.h"
#include "export/frame_source.h"
#include "export/hdr10.h"
#include "export/progress.h"

namespace ajp::exporter {

// Ce que le mux final recopie du fichier source.
struct StreamCopy {
  bool audio = true;
  bool subtitles = true;  // false quand render grave les sous-titres dans l'image
  bool chapters = true;
  bool attachments = true;  // polices des sous-titres ASS ; Matroska seulement
};

// Cadence nominale de sortie, en images par seconde : num / den.
struct FrameRate {
  int num = 24000;
  int den = 1001;
};

inline constexpr int kMaxFrameRate = 1000;

struct ExportJob {
  std::filesystem::path output;
  Container container = Container::Matroska;
  int width = 0;  // taille de sortie : chaque image doit l'avoir
  int height = 0;
  FrameRate frame_rate;
  EncoderSettings encoder;
  Hdr10Mastering mastering;  // max_nits : le pic visé par le grade

  std::filesystem::path copy_from;  // fichier source ; vide : aucune piste recopiée
  StreamCopy copy;
  // Métadonnées globales du conteneur (hachages du Plan, des modèles et des moteurs…).
  std::vector<std::pair<std::string, std::string>> tags;

  ExportFrameSource* source = nullptr;                     // non possédée ; obligatoire
  std::optional<std::int64_t> frames_total;                // prioritaire sur source->total_frames()
  std::function<void(const ExportProgress&)> on_progress;  // appelé sur le thread de l'export
  std::stop_token stop;                                    // annulation : refus Cancelled
};

struct ExportSummary {
  std::filesystem::path output;
  std::int64_t frames = 0;
  LightLevel light;  // MaxCLL et MaxFALL écrits dans le conteneur
  std::chrono::milliseconds elapsed{0};
  double frames_per_second = 0.0;  // débit moyen de l'encodage, mux final exclu
};

// Valide le travail, encode toutes les images de la source puis écrit la sortie.
// Refus : FieldMissing (sortie, source), FieldOutOfRange (taille, cadence, option, mastering),
// FormatUnsupported (format, codec ou piste que le conteneur ne porte pas, voie appareil pas
// encore branchée), InputMismatch (image d'une autre taille, pts inconnu ou qui ne croît pas),
// EncoderUnavailable, EncodeFailed, MuxFailed, FileUnwritable, MediaUnreadable (source),
// Cancelled (arrêt demandé), et tout refus rendu par la source.
std::expected<ExportSummary, Refusal> run_export(const ExportJob& job);

// Vérifie le travail sans rien ouvrir : réglages, taille, cadence, mastering, chemins.
std::expected<void, Refusal> validate_job(const ExportJob& job);

}  // namespace ajp::exporter
