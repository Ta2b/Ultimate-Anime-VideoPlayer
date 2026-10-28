// Pistes et chapitres d'un fichier, tels que le conteneur les décrit.
// Aucun en-tête FFmpeg ici : les modules qui décodent eux-mêmes (audio, subs)
// atteignent AVCodecParameters par CodecParameters::av().
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "core/frame.h"
#include "core/types.h"

struct AVCodecParameters;

namespace ajp::media {

enum class TrackKind : std::uint8_t { Video, Audio, Subtitle, Attachment };

struct Rational {
  int num = 0;
  int den = 1;
  friend bool operator==(const Rational&, const Rational&) = default;
};

// Copie immuable des paramètres de codec d'une piste.
class CodecParameters {
 public:
  static std::expected<std::shared_ptr<const CodecParameters>, Refusal> copy_of(
      const AVCodecParameters& source);
  ~CodecParameters();
  CodecParameters(const CodecParameters&) = delete;
  CodecParameters& operator=(const CodecParameters&) = delete;

  const AVCodecParameters& av() const { return *params_; }
  // Données privées du codec : en-tête ASS, configuration AAC, police d'une pièce jointe.
  std::span<const std::byte> extradata() const;

 private:
  explicit CodecParameters(AVCodecParameters* params) : params_(params) {}
  AVCodecParameters* params_;
};

struct TrackInfo {
  int index = -1;  // index du flux dans le conteneur
  TrackKind kind = TrackKind::Video;
  std::string codec;     // nom court FFmpeg : « mpeg4 », « aac », « ass », « ttf »
  std::string language;  // balise du conteneur (« jpn », « en-US »), vide si absente
  std::string title;
  bool is_default = false;
  bool forced = false;
  bool attached_picture = false;  // pochette : jamais choisie comme vidéo
  std::int64_t duration_us = 0;   // 0 si inconnue

  int width = 0;  // vidéo
  int height = 0;
  Rational frame_rate;     // fréquence moyenne, {0, 1} si inconnue
  Rational sample_aspect;  // forme du pixel (SAR) : celle du flux, sinon du conteneur ; {0, 1} si inconnue
  // Ce que le conteneur étiquette (codecpar) : rien si non renseigné, ou hors de ces trois valeurs.
  std::optional<ColorMatrix> color_matrix;
  bool full_range = false;

  int sample_rate = 0;  // audio
  int channels = 0;

  std::string filename;  // pièce jointe
  std::string mime_type;

  std::shared_ptr<const CodecParameters> parameters;
};

// La couleur d'une piste vidéo telle que le conteneur et le flux l'étiquettent, devinée comme pour une
// image décodée (mêmes règles) : connue avant la première image. Rien pour une piste qui n'est pas de
// la vidéo, ou dont la couleur sort du vocabulaire de core/frame.h.
std::optional<ColorInfo> track_color(const TrackInfo& track);

struct Chapter {
  std::int64_t start_us = 0;
  std::int64_t end_us = 0;
  std::string title;
};

// Pièce jointe que libass peut charger comme police : nom et données non vides,
// type MIME de police, ou à défaut extension de police (MKV mal étiquetés).
bool is_font(const TrackInfo& track);

}  // namespace ajp::media
