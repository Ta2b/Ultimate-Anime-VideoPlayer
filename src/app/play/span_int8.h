// SPAN INT8 (étude 4, 6.3) : la variante d'un SPAN quantifiée par tools/span/quantize_span.py
// (nœuds Q/DQ de Model Optimizer, calibration entropy sur des images d'anime), dans le dossier
// « modeles » des données du lecteur : « <nom>-int8-<16 premiers chiffres du SHA-256 du modèle>.onnx »
// et son reçu « .toml ». Elle se construit et vit sur ce PC seulement : dérivée des poids
// d'AnimeJaNai, elle ne se publie pas.
//
// Les crans de précision (precision.h) y ajoutent deux variantes du même SPAN, au même format de reçu :
// « -fp8- » (Performance : poids et activations en FP8 E4M3, nœuds Q/DQ, tools/span/quantize_span.py
// --precision fp8) et « -fp32- » (Qualité : calcul en FP32, E/S en FP16, --precision fp32). Les garde-fous
// du FP8 ne refusent que les sorties détruites ; la perte qu'il garde se juge à l'œil, c'est le choix du
// cran Performance.
//
// La chaîne la prend à la place du SPAN choisi sous un backend qui sait l'INT8 (TensorRT) quand le
// reçu est celui de ce modèle et de ce fichier, et passe les garde-fous ci-dessous, mesurés contre
// le modèle d'origine sur les images de validation. Les mêmes valeurs par défaut que le script ;
// un reçu écrit avec des seuils plus lâches ne suffit pas.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "app/play/precision.h"

namespace ajp::app::play {

inline constexpr double kInt8MinPsnrMeanDb = 42.0;
inline constexpr double kInt8MinPsnrWorstDb = 38.0;
inline constexpr double kInt8MinSsimWorst = 0.985;
// FP8 : un SPAN simulé en FP8 naïf tombe vers 32 dB ; sous ces seuils la sortie est abîmée.
inline constexpr double kFp8MinPsnrMeanDb = 30.0;
inline constexpr double kFp8MinPsnrWorstDb = 26.0;
inline constexpr double kFp8MinSsimWorst = 0.93;

enum class SpanVariantKind : std::uint8_t { Int8, Fp8, Fp32 };

struct SpanInt8 {  // une variante quantifiée ou recalculée du SPAN, quel que soit son kind
  SpanVariantKind kind = SpanVariantKind::Int8;
  std::filesystem::path onnx;
  double psnr_mean_db = 0.0;
  double psnr_worst_db = 0.0;
  double ssim_worst = 0.0;
};

struct SpanInt8Lookup {
  std::optional<SpanInt8> variant;
  std::string note;  // pour le journal : la variante prise, ou pourquoi pas ; vide sans reçu
};

// La variante de span dans dir, si son reçu la laisse passer.
SpanInt8Lookup find_span_variant(SpanVariantKind kind, const std::filesystem::path& span,
                                 const std::filesystem::path& dir);
inline SpanInt8Lookup find_span_int8(const std::filesystem::path& span, const std::filesystem::path& dir) {
  return find_span_variant(SpanVariantKind::Int8, span, dir);
}

// Les crans de SPAN dont la variante est dans dir et passe ses garde-fous ; fp8_engine : le moteur calcule
// le FP8.
TierSupport span_tiers(const std::filesystem::path& span, const std::filesystem::path& dir, bool fp8_engine);

// « INT8 », « FP8 », « FP32 » : pour les libellés de la chaîne.
std::string_view variant_text(SpanVariantKind kind);

// « 2x_…_Performance.onnx » pour « 2x_…_Performance-int8-0123456789abcdef.onnx » (ou -fp8-, -fp32-) ; rien
// pour un autre nom. Les badges du cache nomment ainsi le modèle d'origine. kind rend laquelle des trois.
std::optional<std::string> span_int8_source(const std::string& file, SpanVariantKind* kind = nullptr);

}  // namespace ajp::app::play
