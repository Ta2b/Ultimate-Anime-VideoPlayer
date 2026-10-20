// Le fichier d'un export : un MKV où la vidéo encodée vient en premier, suivie des pistes de la
// source copiées telles quelles (audio, sous-titres, polices jointes), avec ses chapitres et ses
// métadonnées. Les paquets copiés sont lus au fil de l'écriture, jusqu'à l'instant de la vidéo,
// pour que le mux les entrelace sans tout garder en mémoire. Interne au module.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <vector>

#include "core/frame.h"
#include "core/types.h"
#include "export/av_handles.h"

namespace ajp::exporter {

class Muxer {
 public:
  // Ouvre la source et prépare la sortie, sans créer le fichier. Refus : MediaUnreadable.
  static std::expected<Muxer, Refusal> open(const std::filesystem::path& source,
                                            const std::filesystem::path& output);
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
  // Ferme le fichier sans le finir et le supprime.
  void abandon();
  std::int64_t bytes() const;  // octets écrits jusqu'ici

 private:
  Muxer() = default;
  std::expected<void, Refusal> read_copy();  // le paquet copié suivant dans pending_, ou la fin
  std::expected<void, Refusal> write_copies(std::int64_t until_us);
  void close_output();

  std::filesystem::path output_path_;
  InputPtr input_;
  AVFormatContext* output_ = nullptr;
  std::vector<int> map_;  // flux de la source → flux de la sortie, -1 : non copié
  int source_video_ = -1;
  AVRational encoder_time_base_{1, 1'000'000};
  std::int64_t start_us_ = 0;
  PacketPtr pending_;  // le prochain paquet copié, lu d'avance
  std::int64_t pending_us_ = 0;
  std::int64_t written_ = 0;  // octets du fichier clos
  bool copies_done_ = false;
  bool opened_ = false;  // le fichier existe
  bool finished_ = false;
};

}  // namespace ajp::exporter
