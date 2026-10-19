// Mux final de l'export : le flux vidéo de la première passe, plus les pistes, chapitres et
// pièces jointes du fichier source, et les métadonnées HDR10 mesurées. Interne à export.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <stop_token>
#include <string>
#include <utility>

#include "core/types.h"
#include "export/encoder_settings.h"
#include "export/export.h"
#include "export/hdr10.h"

namespace ajp::exporter {

struct RemuxRequest {
  std::filesystem::path video;   // fichier vidéo seul de la première passe, horodaté depuis 0
  std::filesystem::path source;  // vide : aucune piste à recopier
  StreamCopy copy;
  // Fenêtre exportée, en microsecondes depuis le début du fichier source : la première image
  // devient l'instant 0 de la sortie, et les paquets recopiés suivent le même décalage.
  std::int64_t begin_us = 0;
  std::int64_t end_us = 0;
  Container container = Container::Matroska;
  Hdr10Mastering mastering;
  LightLevel light;
  std::span<const std::pair<std::string, std::string>> tags;
  std::filesystem::path output;
  std::stop_token stop;  // annulation : refus Cancelled entre deux paquets
};

// Refus : MediaUnreadable (entrées), FormatUnsupported (piste que le conteneur ne porte pas),
// MuxFailed, FileUnwritable, Cancelled.
std::expected<void, Refusal> remux(const RemuxRequest& request);

// Ouvre le fichier source et vérifie que le conteneur porte chaque piste à recopier, sans rien
// écrire : un refus tombe avant l'encodage plutôt qu'après. Mêmes refus que remux().
std::expected<void, Refusal> check_copy(const RemuxRequest& request);

}  // namespace ajp::exporter
