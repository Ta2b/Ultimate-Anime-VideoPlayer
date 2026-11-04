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
// Quatrième variante, « -fp8c- » (Performance, TensorRT seulement, RTX 40) : le SPAN entier en FP8 par les
// noyaux CUTLASS du backend (docs/span-fp8-cutlass.md), un paquet de poids « .span8 » et non un ONNX, dérivé
// par tools/span/derive_span_fp8.py. Reçu aux seuils de l'INT8 (le FP8 par Q/DQ ci-dessus a des seuils
// lâches). La chaîne la prend avant « -fp8- » quand le backend l'annonce (BackendCaps::span_fp8_cutlass) et,
// si son chargement est refusé (autre GPU, taille que les noyaux ne prennent pas, erreur CUDA), passe à la
// variante suivante puis au FP16.
//
// La chaîne la prend à la place du SPAN choisi sous un backend qui sait l'INT8 (TensorRT) quand le
// reçu est celui de ce modèle et de ce fichier, et passe les garde-fous ci-dessous, mesurés contre
// le modèle d'origine sur les images de validation. Les mêmes valeurs par défaut que le script ;
// un reçu écrit avec des seuils plus lâches ne suffit pas.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/precision.h"
#include "core/types.h"

namespace ajp::app::play {

inline constexpr double kInt8MinPsnrMeanDb = 42.0;
inline constexpr double kInt8MinPsnrWorstDb = 38.0;
inline constexpr double kInt8MinSsimWorst = 0.985;
// FP8 : un SPAN simulé en FP8 naïf tombe vers 32 dB ; sous ces seuils la sortie est abîmée.
inline constexpr double kFp8MinPsnrMeanDb = 30.0;
inline constexpr double kFp8MinPsnrWorstDb = 26.0;
inline constexpr double kFp8MinSsimWorst = 0.93;

enum class SpanVariantKind : std::uint8_t { Int8, Fp8, Fp32, Fp8Cutlass };

struct SpanInt8 {  // une variante quantifiée ou recalculée du SPAN, quel que soit son kind
  SpanVariantKind kind = SpanVariantKind::Int8;
  std::filesystem::path onnx;
  double psnr_mean_db = 0.0;
  double psnr_worst_db = 0.0;
  double ssim_worst = 0.0;
  bool receipt_v2 = false;  // FP8 CUTLASS : reçu ajp-span-fp8c-v2 (seul à servir l'Auto, étape 3.7b)
};

struct SpanInt8Lookup {
  std::optional<SpanInt8> variant;
  std::string note;  // pour le journal : la variante prise, ou pourquoi pas ; vide sans reçu
  // FP8 CUTLASS avec contexte : reçu v2 de cet appareil et de ces noyaux, mais pas encore jugé à cette taille
  // (« 1280x720 ») ; une validation seule la tranche (span_derive.h, validate_span_size).
  std::string unvalidated_size;
  // FP8 CUTLASS : à refaire, pas refusé pour de bon (reçu d'une autre carte ou d'autres noyaux, ou refus
  // d'une porte plus ancienne que gpu::span_fp8::kGateVersion) ; la préparation le refait.
  bool stale = false;
};

// Ce que le moteur sait calculer, pour l'ordre des variantes (BackendCaps).
struct SpanVariantCaps {
  bool cutlass = false;  // span_fp8_cutlass
  bool fp8 = false;      // graphes Q/DQ FP8
  bool int8 = false;
};

// Les variantes de SPAN à essayer, dans l'ordre, pour un cran (étape 3.6 ; « la plus basse précision qui
// passe la porte ») : Performance prend le SPAN FP8 CUTLASS, puis le FP8 par Q/DQ, puis l'INT8 ; Équilibre
// l'INT8 ; Qualité le FP32. strict (cran choisi par un préréglage, pas à la main) : sans le FP8 par Q/DQ,
// dont les seuils sont lâches ; la chaîne y demande aussi un reçu v2 au FP8 CUTLASS.
std::vector<SpanVariantKind> span_variant_order(PrecisionTier tier, const SpanVariantCaps& caps, bool strict);

// Les listes de tailles d'un reçu v2 et son texte (validation seule, étape 3.7b). Refus FileUnreadable.
struct ReceiptSizes {
  std::string text;
  std::vector<std::string> validated;
  std::vector<std::string> refused;
};
std::expected<ReceiptSizes, Refusal> read_receipt_sizes(const std::filesystem::path& receipt);
// Les refus de taille notés aussi sous la porte qui les a pris (« refused_sizes_gate3 ») : un refus de
// refused_sizes que la porte d'aujourd'hui n'a pas noté (lecteur plus ancien, porte d'avant) se revalide.
std::string gate_refusals_key();
std::string with_gate_refusals(std::string_view receipt, const std::vector<std::string>& refused);

// Ce que la chaîne sait du SPAN FP8 CUTLASS au chargement (étape 3.7b) : BackendCaps::gpu_arch et
// span_fp8_kernels, et la taille de travail. Un reçu v2 d'une autre architecture ou d'autres noyaux est
// refusé ; une taille qu'il n'a pas jugée rend unvalidated_size au lieu de la variante.
struct SpanFp8Context {
  std::string gpu_arch;
  int kernels_version = 0;
  int width = 0;
  int height = 0;
};

// La variante de span dans dir, si son reçu la laisse passer.
SpanInt8Lookup find_span_variant(SpanVariantKind kind, const std::filesystem::path& span,
                                 const std::filesystem::path& dir, const SpanFp8Context* fp8 = nullptr);
inline SpanInt8Lookup find_span_int8(const std::filesystem::path& span, const std::filesystem::path& dir) {
  return find_span_variant(SpanVariantKind::Int8, span, dir);
}

// Les crans de SPAN dont la variante est dans dir et passe ses garde-fous ; fp8_engine : le moteur calcule
// le FP8 ; cutlass_engine : il sait aussi le SPAN entier en noyaux CUTLASS : le cran Performance est alors
// offert même sans paquet (le lecteur le prépare, span_auto.h), sauf pour le modèle b5f48 (Performance par le
// paquet « .span8 »).
TierSupport span_tiers(const std::filesystem::path& span, const std::filesystem::path& dir, bool fp8_engine,
                       bool cutlass_engine = false);

// Le fichier de la variante de span qui a ce SHA-256 (celui du modèle) : dir/<nom>-<infixe><16
// chiffres>.<extension>, « .onnx » ou « .span8 ». Son reçu est le même nom en « .toml ».
std::filesystem::path span_variant_path(SpanVariantKind kind, const std::filesystem::path& span,
                                        const std::filesystem::path& dir, std::string_view sha256);

// « INT8 », « FP8 », « FP32 », « FP8 CUTLASS » : pour les libellés de la chaîne.
std::string_view variant_text(SpanVariantKind kind);

// « 2x_…_Performance.onnx » pour « 2x_…_Performance-int8-0123456789abcdef.onnx » (ou -fp8-, -fp32-, et
// -fp8c- en « .span8 ») ; rien
// pour un autre nom. Les badges du cache nomment ainsi le modèle d'origine. kind rend laquelle des trois.
std::optional<std::string> span_int8_source(const std::string& file, SpanVariantKind* kind = nullptr);

}  // namespace ajp::app::play
