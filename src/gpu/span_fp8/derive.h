// Ce qui est commun à toute dérivation du paquet « .span8 » (docs/span-fp8-cutlass.md) : l'outil Python
// tools/span/derive_span_fp8.py et la dérivation native du lecteur (app/play/span_derive.h) disent la même
// chose des échelles, des garde-fous et du reçu. C++ pur, sans GPU.
#pragma once

#include <array>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::span_fp8 {

inline constexpr float kE4m3Max = 448.0F;  // le plus grand E4M3 fini

// Format du reçu et seuils : ceux de l'INT8 (app/play/span_int8.cpp), pas ceux, lâches, du FP8 par Q/DQ.
inline constexpr const char* kReceiptFormat = "ajp-span-fp8c-v1";
inline constexpr double kMinPsnrMeanDb = 42.0;
inline constexpr double kMinPsnrWorstDb = 38.0;
inline constexpr double kMinSsimWorst = 0.985;

// Les échelles des 25 tenseurs quantifiés d'après les maxima observés : maximum / 448. Refus
// FormatUnsupported si un maximum est nul ou non fini (corpus inutilisable, ou réseau qui déborde).
std::expected<std::array<float, kScaleCount>, Refusal> scales_from_peaks(std::span<const float> peaks);

// PSNR et SSIM d'une image de validation.
struct ImageScore {
  double psnr_db = 0.0;
  double ssim = 0.0;
};

struct Verdict {
  double psnr_mean_db = 0.0;
  double psnr_worst_db = 0.0;
  double ssim_worst = 0.0;
  bool passed = false;
};

// Moyenne et pires valeurs, et leur comparaison aux seuils. Sans image, rien ne passe.
Verdict judge(std::span<const ImageScore> scores, double min_psnr_mean = kMinPsnrMeanDb,
              double min_psnr_worst = kMinPsnrWorstDb, double min_ssim_worst = kMinSsimWorst);

// Le reçu « .toml » : mêmes clés, même ordre et même mise en forme que write_receipt de quantize_span.py
// (flottants à quatre décimales). Les champs de l'outil Python qu'une dérivation native ne mesure pas
// valent 0 ; reference et derived_by disent comment elle a été faite.
struct Receipt {
  std::string source;         // nom du fichier ONNX d'origine
  std::string source_sha256;  // du fichier ONNX d'origine
  std::string model_sha256;   // du paquet « .span8 » écrit ; vide s'il n'a pas été écrit
  std::string quantizer;
  double speedup = 0.0;
  double kernels_ms = 0.0;
  double trt_fp16_ms = 0.0;
  int calibration_images = 0;
  int calibration_crops = 0;
  int calibration_crop = 0;
  std::string corpus_sha256;
  int validation_images = 0;
  std::string validation_size;  // « 1920x1080 »
  Verdict verdict;
  // Seulement pour la dérivation native.
  std::string reference;   // d'où vient la sortie de référence (« cutlass-fp16 »)
  std::string derived_by;  // « ajp (natif) »
};

std::string receipt_text(const Receipt& receipt, std::string_view comment);

// Les tailles d'image que la dérivation prend : au plus width × height, ramenées à ce que les noyaux
// acceptent (hauteur multiple de 4, largeur multiple de 128). {0, 0} si rien n'en reste.
struct Crop {
  int width = 0;
  int height = 0;
};
Crop kernel_crop(int width, int height, int max_width = 1920, int max_height = 1088);

}  // namespace ajp::gpu::span_fp8
