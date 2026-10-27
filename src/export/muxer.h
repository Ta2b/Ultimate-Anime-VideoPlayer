// Le fichier d'un export : un MKV, un MP4 ou un MOV (export/container.h) où la vidéo encodée vient en
// premier, suivie des pistes de la source copiées telles quelles (audio, sous-titres, polices jointes), avec
// ses chapitres et ses métadonnées. Les paquets copiés sont lus au fil de l'écriture, jusqu'à l'instant de la
// vidéo, pour que le mux les entrelace sans tout garder en mémoire. Interne au module.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "core/frame.h"
#include "core/types.h"
#include "export/av_handles.h"
#include "export/container.h"

namespace ajp::exporter {

class Muxer {
 public:
  // Ouvre la source et prépare la sortie, sans créer le fichier, dans le conteneur de son extension.
  // Refus : MediaUnreadable, FileUnwritable (extension inconnue).
  // video_stream : le flux vidéo que le lecteur a choisi (SAR, couleur et langue viennent de lui) ;
  // -1 : le meilleur selon FFmpeg.
  static std::expected<Muxer, Refusal> open(const std::filesystem::path& source,
                                            const std::filesystem::path& output, int video_stream = -1);
  Muxer(Muxer&& other) noexcept;
  Muxer& operator=(Muxer&&) = delete;
  Muxer(const Muxer&) = delete;
  Muxer& operator=(const Muxer&) = delete;
  ~Muxer();

  AVRational source_aspect() const;                    // rapport d'aspect des pixels de la vidéo source
  ColorInfo source_color() const;                      // sa couleur (av_handles.h : color_of)
  std::int64_t start_us() const { return start_us_; }  // début de la source dans le conteneur

  // La piste vidéo de l'encodeur, puis les pistes copiées, l'en-tête et le fichier. Refus :
  // FileUnwritable, EncodeFailed.
  std::expected<void, Refusal> begin(const AVCodecContext& encoder);
  // Un paquet de l'encodeur, daté dans sa base de temps : les paquets copiés jusqu'à son instant
  // passent avant lui. Le paquet est consommé. Refus : EncodeFailed.
  std::expected<void, Refusal> write_video(AVPacket& packet);
  // Les paquets copiés restants, puis la fin du fichier. Refus : EncodeFailed.
  std::expected<void, Refusal> finish();
  // Ferme le fichier sans le finir et supprime le fichier partiel : un fichier existant au même
  // chemin n'est touché qu'à la fin réussie de finish().
  void abandon();
  // Non vide : la lecture des pistes copiées s'est arrêtée sur une erreur (pas la fin du fichier),
  // l'audio et les sous-titres s'arrêtent là. L'export reste écrit.
  const std::string& copy_error() const { return copy_error_; }
  std::int64_t bytes() const;  // octets écrits jusqu'ici

 private:
  Muxer() = default;
  std::expected<void, Refusal> read_copy();  // le paquet copié suivant dans pending_, ou la fin
  std::expected<void, Refusal> write_copies(std::int64_t until_us);
  int close_output();  // < 0 : la fermeture du fichier a échoué (écriture finale, disque plein)

  std::filesystem::path output_path_;
  std::filesystem::path partial_path_;  // écrit ici, puis renommé en output_path_ à la fin
  std::string copy_error_;
  Container container_ = Container::Matroska;
  InputPtr input_;
  AVFormatContext* output_ = nullptr;
  std::vector<int> map_;  // flux de la source → flux de la sortie, -1 : non copié
  int source_video_ = -1;
  AVRational encoder_time_base_{1, 1'000'000};
  std::int64_t start_us_ = 0;
  PacketPtr pending_;  // le prochain paquet copié, lu d'avance
  std::int64_t pending_us_ = 0;
  std::int64_t written_ = 0;  // octets du fichier clos
  // Dernier dts vidéo écrit, dans la base de la piste : MP4 et MOV refusent un dts égal.
  std::int64_t last_dts_ = std::numeric_limits<std::int64_t>::min();
  bool copies_done_ = false;
  bool opened_ = false;  // le fichier existe
  bool finished_ = false;
};

}  // namespace ajp::exporter
