// SPAN INT8 (étude 4, 6.3) : la variante d'un SPAN quantifiée par tools/span/quantize_span.py
// (nœuds Q/DQ de Model Optimizer, calibration entropy sur des images d'anime), dans le dossier
// « modeles » des données du lecteur : « <nom>-int8-<16 premiers chiffres du SHA-256 du modèle>.onnx »
// et son reçu « .toml ». Elle se construit et vit sur ce PC seulement : dérivée des poids
// d'AnimeJaNai, elle ne se publie pas.
//
// La chaîne la prend à la place du SPAN choisi sous un backend qui sait l'INT8 (TensorRT) quand le
// reçu est celui de ce modèle et de ce fichier, et passe les garde-fous ci-dessous, mesurés contre
// le modèle d'origine sur les images de validation. Les mêmes valeurs par défaut que le script ;
// un reçu écrit avec des seuils plus lâches ne suffit pas.
#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace ajp::app::play {

inline constexpr double kInt8MinPsnrMeanDb = 42.0;
inline constexpr double kInt8MinPsnrWorstDb = 38.0;
inline constexpr double kInt8MinSsimWorst = 0.985;

struct SpanInt8 {
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
SpanInt8Lookup find_span_int8(const std::filesystem::path& span, const std::filesystem::path& dir);

// « 2x_…_Performance.onnx » pour « 2x_…_Performance-int8-0123456789abcdef.onnx » ; rien pour un
// autre nom. Les badges du cache nomment ainsi le modèle d'origine.
std::optional<std::string> span_int8_source(const std::string& file);

}  // namespace ajp::app::play
