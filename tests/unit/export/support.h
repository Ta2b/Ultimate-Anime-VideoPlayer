// Outils des tests de l'export : images RGBA16 synthétiques, source d'images en mémoire,
// fichier source à pistes multiples écrit par libavformat, relecture des sorties. Tout se
// génère dans le dossier de build : aucun fichier binaire n'entre dans le dépôt.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
}

#include "export/av_util.h"
#include "export/export.h"
#include "export/frame_source.h"

namespace ajp::exporter::test {

// Dossier de travail neuf, vidé s'il existait.
std::filesystem::path work_dir(std::string_view name);

// pts de l'image index à la cadence rate, en microseconde arrondie.
std::int64_t frame_pts(int index, FrameRate rate);

struct OwnedFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint16_t> pixels;  // RGBA16 tassé : stride = width * 8
  std::int64_t pts_us = 0;
  HostFrame view() const;
};

// Image déterministe : chaque composante dépend de x, y, de l'indice et du canal.
OwnedFrame pattern_frame(int width, int height, int index, std::int64_t pts_us);
// Image unie : chaque pixel vaut (r, g, b, 65535).
OwnedFrame flat_frame(int width, int height, std::uint16_t r, std::uint16_t g, std::uint16_t b);
std::vector<OwnedFrame> pattern_frames(int width, int height, int count, FrameRate rate, int first_index = 0);

// Les images d'un vecteur, dans l'ordre ; before_next(i) est appelé avant de rendre l'image i.
class VectorSource final : public ExportFrameSource {
 public:
  explicit VectorSource(const std::vector<OwnedFrame>& frames) : frames_(frames) {}
  std::expected<std::optional<SourceFrame>, Refusal> next() override;
  std::optional<std::int64_t> total_frames() const override {
    return static_cast<std::int64_t>(frames_.size());
  }
  std::function<void(std::size_t)> before_next;

 private:
  const std::vector<OwnedFrame>& frames_;
  std::size_t index_ = 0;
};

// Travail FFV1 RGB 16 bits en Matroska, sans source à recopier.
ExportJob ffv1_job(const std::filesystem::path& output, int width, int height, ExportFrameSource& source);

// Fichier source synthétique (Matroska) : vidéo rawvideo, audio PCM 48 kHz stéréo par paquets
// de 1024 échantillons, deux sous-titres SRT, deux chapitres, une police jointe, un titre.
struct SourceLayout {
  static constexpr int kAudioPackets = 48;  // 1,024 s
  static constexpr int kAudioSamples = 1024;
  static constexpr int kSampleRate = 48000;
  static constexpr std::int64_t kSubtitle1Us = 100'000;
  static constexpr std::int64_t kSubtitle2Us = 600'000;
  static constexpr std::int64_t kSubtitleDurationUs = 400'000;
  static constexpr std::int64_t kChapterSplitUs = 200'000;
  static constexpr std::int64_t kChapterEndUs = 1'000'000;
  static constexpr std::string_view kFontName = "police.ttf";
};
void write_source_media(const std::filesystem::path& path);
std::vector<std::uint8_t> audio_payload(int packet);
std::vector<std::uint8_t> font_payload();

// Paquets d'un flux d'un fichier : pts en microsecondes, données.
struct StreamDump {
  AVMediaType type = AVMEDIA_TYPE_UNKNOWN;
  AVCodecID codec = AV_CODEC_ID_NONE;
  std::vector<std::int64_t> pts_us;
  std::vector<std::vector<std::uint8_t>> data;
};
std::vector<StreamDump> dump_streams(const std::filesystem::path& path);

// Chaque image vidéo décodée du fichier, dans l'ordre.
void decode_video(const std::filesystem::path& path, const std::function<void(const AVFrame&)>& on_frame);

}  // namespace ajp::exporter::test
