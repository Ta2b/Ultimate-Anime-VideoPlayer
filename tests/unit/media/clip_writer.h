// Petits clips de test, générés dans le dossier de build par libavformat et les
// encodeurs natifs de FFmpeg : aucun fichier binaire n'entre dans le dépôt.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

namespace ajp::media::test {

enum class VideoCodec : std::uint8_t { Mpeg4, Ffv1 };

struct AudioSpec {
  std::string language;
  std::string title;
  bool is_default = false;
};

struct SubtitleEvent {
  std::int64_t start_ms = 0;
  std::int64_t duration_ms = 0;
  std::string text;
};

struct SubtitleSpec {
  std::string language;
  std::string title;
  bool is_default = false;
  bool forced = false;
  std::vector<SubtitleEvent> events;
};

struct AttachmentSpec {
  std::string filename;
  std::string mime_type;
  std::string data;
};

struct ChapterSpec {
  std::int64_t start_ms = 0;
  std::int64_t end_ms = 0;
  std::string title;
};

struct ColorSpec {
  AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
  AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;
  AVColorSpace matrix = AVCOL_SPC_UNSPECIFIED;
  AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
  AVChromaLocation siting = AVCHROMA_LOC_UNSPECIFIED;
  bool hdr10 = false;  // mastering BT.2020 1000 / 0,0001 nits, MaxCLL 1000, MaxFALL 400
};

struct ClipSpec {
  std::string name;                    // nom du fichier dans le dossier des clips
  std::string container = "matroska";  // ou « mp4 »
  VideoCodec codec = VideoCodec::Ffv1;
  AVPixelFormat pixel_format = AV_PIX_FMT_YUV420P;
  int width = 64;
  int height = 48;
  int frames = 30;
  AVRational frame_rate{25, 1};
  int gop = 10;
  int b_frames = 0;
  int video_delay = 0;  // pts de la première image vidéo, en images ; l'audio part de 0
  ColorSpec color;
  std::vector<AudioSpec> audio;  // PCM 16 bits stéréo 48 kHz, paquets de 20 ms
  std::vector<SubtitleSpec> subtitles;
  std::vector<AttachmentSpec> attachments;
  std::vector<ChapterSpec> chapters;
};

// Écrit le clip (fichier temporaire puis renommage) et rend son chemin.
std::expected<std::filesystem::path, std::string> write_clip(const ClipSpec& spec);

// Motif sans perte (FFV1) : valeur du plan (0 Y, 1 U, 2 V) au point (x, y) de
// l'image n, sur 8 ou 10 bits.
int pattern(int plane, int x, int y, int n, int bits);

// Motif avec perte (MPEG-4) : luma uniforme propre à chaque image, chroma 128.
int flat_luma(int n);
// Image dont la luma uniforme décodée vaut `value` (à ±3 près), ou -1.
int frame_of_flat_luma(int value);

inline constexpr char kAssHeader[] =
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 64\nPlayResY: 48\n\n"
    "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, "
    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, "
    "MarginL, MarginR, MarginV, Encoding\n"
    "Style: "
    "Default,Sans,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n\n"
    "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

}  // namespace ajp::media::test
