// Sortie libavformat de l'export (Matroska ou MP4) : flux, en-tête, paquets entrelacés, fin.
// Interne à export ; sert au fichier vidéo de la première passe et au mux final.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string_view>

#include "core/types.h"
#include "export/av_util.h"
#include "export/encoder_settings.h"

namespace ajp::exporter {

class MuxOutput {
 public:
  // Contexte de sortie pour path ; rien n'est écrit avant write_header(). index_first : en MP4,
  // l'index (moov) en tête du fichier, au prix d'une réécriture à la fin. Refus MuxFailed.
  static std::expected<MuxOutput, Refusal> create(const std::filesystem::path& path, Container container,
                                                  bool index_first = false);

  // Nouveau flux aux paramètres copiés de par, dans la base de temps time_base. Refus
  // FormatUnsupported quand le conteneur ne porte pas ce codec ; what nomme le flux. Une pièce
  // jointe se passe de query_codec : son codec n'est pas dans les tables du conteneur.
  std::expected<AVStream*, Refusal> add_stream(const AVCodecParameters& par, AVRational time_base,
                                               std::string_view what, bool query_codec = true);

  AVFormatContext& context() { return *ctx_; }

  // Ouvre le fichier et écrit l'en-tête. Refus FileUnwritable ou MuxFailed.
  std::expected<void, Refusal> write_header();

  // Un paquet du flux packet.stream_index, horodaté dans la base de temps time_base ; la
  // référence du paquet est prise. Refus MuxFailed.
  std::expected<void, Refusal> write(AVPacket& packet, AVRational time_base);

  // Fin du conteneur, fichier fermé. Refus MuxFailed.
  std::expected<void, Refusal> finish();

 private:
  struct Closer {
    void operator()(AVFormatContext* ctx) const;
  };

  std::unique_ptr<AVFormatContext, Closer> ctx_;
  std::filesystem::path path_;
  bool index_first_ = false;
};

// Supprime un fichier à la destruction, sauf après release() : fichiers de travail et sortie
// partielle d'un export refusé ou annulé.
class RemoveOnExit {
 public:
  explicit RemoveOnExit(std::filesystem::path path) : path_(std::move(path)) {}
  RemoveOnExit(const RemoveOnExit&) = delete;
  RemoveOnExit& operator=(const RemoveOnExit&) = delete;
  ~RemoveOnExit();
  void release() { path_.clear(); }

 private:
  std::filesystem::path path_;
};

}  // namespace ajp::exporter
