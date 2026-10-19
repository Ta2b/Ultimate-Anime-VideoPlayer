// Réglages de l'encodeur de l'export, validés contre une table fixe avant toute ouverture.
//
// La table nomme chaque encodeur libavcodec que l'export sait piloter, ses formats, ses modes
// de débit et les plages de ses options. Une valeur hors de ces plages est refusée ici, jamais
// corrigée ; la présence de l'encodeur dans la construction de FFmpeg se vérifie à
// l'ouverture (encoder.cpp). Aucune dépendance à FFmpeg dans cet en-tête.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::exporter {

enum class Container : std::uint8_t { Matroska, Mp4 };

// Formats d'entrée de l'encodeur, après conversion depuis RGBA16 PQ BT.2020.
enum class PixelFormat : std::uint8_t {
  Yuv420p10,  // planaire 4:2:0 10 bits, BT.2020 non constant, plage limitée
  P010,       // semi-planaire 4:2:0 10 bits (NVENC, encodeurs Vulkan), mêmes étiquettes
  Rgb16,      // RGB planaire 16 bits, pleine échelle : sans perte de l'image de render
};

enum class RateControl : std::uint8_t {
  Lossless,         // sans perte (FFV1)
  ConstantQuality,  // qualité visée : cq de NVENC, crf de SVT-AV1
  ConstantQp,       // quantificateur fixe
  Bitrate,          // débit moyen visé, plafond facultatif
};

enum class EncoderFamily : std::uint8_t { Ffv1, Nvenc, SvtAv1, Vulkan };

struct EncoderSettings {
  std::string encoder = "ffv1";  // nom libavcodec : ffv1, hevc_nvenc, av1_nvenc, libsvtav1…
  PixelFormat format = PixelFormat::Rgb16;
  RateControl rate = RateControl::Lossless;
  int quality = 0;                       // ConstantQuality ou ConstantQp ; 0 dans les autres modes
  std::int64_t bitrate_kbps = 0;         // Bitrate seulement
  std::int64_t max_bitrate_kbps = 0;     // Bitrate seulement ; 0 : pas de plafond
  std::optional<int> preset;             // NVENC 1 à 7 (p1 à p7), SVT-AV1 0 à 13 ; absent : défaut
  std::optional<int> keyframe_interval;  // images entre deux images clés ; absent : défaut
};

// Plage fermée [min, max] ; min > max : option sans objet pour cet encodeur.
struct IntRange {
  int min = 1;
  int max = 0;
  constexpr bool empty() const { return min > max; }
  constexpr bool contains(int value) const { return value >= min && value <= max; }
};

struct EncoderSpec {
  std::string_view name;
  EncoderFamily family;
  std::span<const PixelFormat> formats;
  std::span<const RateControl> rates;
  IntRange quality;  // ConstantQuality
  IntRange qp;       // ConstantQp
  IntRange preset;
  bool hardware = false;  // exige un GPU et son pilote à l'ouverture
};

// Bornes communes de l'export.
inline constexpr IntRange kSizeRange{16, 16384};
inline constexpr std::int64_t kMinBitrateKbps = 100;
inline constexpr std::int64_t kMaxBitrateKbps = 2'000'000;
inline constexpr IntRange kKeyframeRange{1, 1000};

// Toute la table, dans l'ordre de préférence.
std::span<const EncoderSpec> encoder_table();

// L'entrée de la table ; refus EncoderUnavailable pour un nom inconnu.
std::expected<const EncoderSpec*, Refusal> find_encoder_spec(std::string_view name);

// Vérifie les réglages pour une image de width × height. Refus : EncoderUnavailable (nom
// inconnu), FormatUnsupported (format ou mode de débit que l'encodeur ne prend pas),
// FieldOutOfRange (taille, taille impaire en 4:2:0, option hors plage ou sans objet).
std::expected<const EncoderSpec*, Refusal> validate_settings(const EncoderSettings& settings, int width,
                                                             int height);

std::string_view name(Container container);
std::string_view name(PixelFormat format);
std::string_view name(RateControl rate);

// Vrai pour les formats 4:2:0, qui exigent une largeur et une hauteur paires.
constexpr bool is_420(PixelFormat format) {
  return format != PixelFormat::Rgb16;
}

}  // namespace ajp::exporter
