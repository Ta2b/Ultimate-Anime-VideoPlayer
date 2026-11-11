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

// Format du reçu et seuils : ceux de l'INT8 (app/play/span_int8.cpp), pas ceux, lâches, du FP8 par Q/DQ.
inline constexpr const char* kReceiptFormat = "ajp-span-fp8c-v2";  // étape 3.7b
// La version de la porte (judge_hardened) : un refus d'une porte plus ancienne se refait. 2 : les cas durs
// synthétiques ne décident plus (séance de mesure).
// 3 : centiles réglés sur la RTX 4070 Ti. 4 : gain sur le SPAN FP16 de TensorRT mesuré et exigé, échelles
// avec marge de tête (B02-01, B02-02) ; un paquet d'une porte plus ancienne, même passé, se refait.
inline constexpr int kGateVersion = 4;
// Le reçu de tools/span/derive_span_fp8.py, encore lu (app/play/span_int8.cpp).
inline constexpr const char* kReceiptFormatV1 = "ajp-span-fp8c-v1";
inline constexpr double kMinPsnrMeanDb = 42.0;
inline constexpr double kMinPsnrWorstDb = 38.0;
inline constexpr double kMinSsimWorst = 0.985;
// Étape 3.7a, porte durcie : 99,9e centile de l'erreur absolue (sur [0, 1]) des images de validation, le même
// sur les pixels sombres. Réglés comme le plan le prévoyait, à 1,25 × ceux du b8f64 mesurés sur la RTX 4070
// Ti (8,1/255 et 3,6/255 à 720p, PSNR 52,0 dB) ; les valeurs de départ (4/255 et 2/255) refusaient
// tout.
inline constexpr double kErrorQuantile = 0.999;
inline constexpr double kMaxErrorP999 = 10.0 / 255.0;
inline constexpr double kMaxDarkErrorP999 = 4.5 / 255.0;
// B02-01 : le FP8 perd en qualité contre le FP16 ; il n'est gardé que s'il va au moins ce rapport plus vite
// que le SPAN FP16 de TensorRT sur la même carte, à la même taille (le seuil de
// tools/span/derive_span_fp8.py).
inline constexpr double kMinSpeedup = 1.05;
// B02-02 : les échelles valent maximum observé × kScaleHeadroom / 448. Les maxima viennent d'une dizaine
// d'images d'une seule vidéo, et le paquet sert ensuite à toutes : sans marge, une image plus lumineuse ou
// plus saturée que le corpus écrête ses activations (bandes, teintes). La marge coûte un quart de pas en bas
// de l'échelle ; la porte juge le paquet avec elle.
inline constexpr float kScaleHeadroom = 1.25F;
// Étape 3.5a : l'accumulateur f16 hors blocs ne coûte pas plus que ceci au PSNR moyen.
inline constexpr double kMaxAcc16PsnrLossDb = 0.3;

// Les échelles des 1 + 3·blocks tenseurs quantifiés d'après les maxima observés : maximum × kScaleHeadroom /
// 448 (peaks garde
// ses 25 places, les suivantes à 1). Refus FormatUnsupported si un maximum servi est nul ou non fini (corpus
// inutilisable, ou réseau qui déborde).
std::expected<std::array<float, kScaleCount>, Refusal> scales_from_peaks(std::span<const float> peaks,
                                                                         std::size_t blocks = kBlocks);

// PSNR et SSIM d'une image de validation.
struct ImageScore {
  double psnr_db = 0.0;
  double ssim = 0.0;
  double error_p999 = 0.0;       // abs_error_quantile(…, kErrorQuantile) ; 0 : non mesuré (outil Python)
  double dark_error_p999 = 0.0;  // dark_abs_error_quantile(…, kErrorQuantile)
};

struct Verdict {
  double psnr_mean_db = 0.0;
  double psnr_worst_db = 0.0;
  double ssim_worst = 0.0;
  bool passed = false;
  // Porte durcie (judge_hardened) ; 0 et 0 cas durs pour judge seul.
  double error_p999_worst = 0.0;
  double dark_error_p999_worst = 0.0;
  double hard_psnr_worst_db = 0.0;
  double hard_error_p999_worst = 0.0;  // les cas durs : relevés, sans décider
  int hard_images = 0;
};

// Moyenne et pires valeurs, et leur comparaison aux seuils. Sans image, rien ne passe.
Verdict judge(std::span<const ImageScore> scores, double min_psnr_mean = kMinPsnrMeanDb,
              double min_psnr_worst = kMinPsnrWorstDb, double min_ssim_worst = kMinSsimWorst);

// judge, puis la porte durcie (étape 3.7a) : pire 99,9e centile ≤ kMaxErrorP999 et, sur le sombre, ≤
// kMaxDarkErrorP999, sur les images de validation. Les cas durs synthétiques sont relevés (hard_*) sans
// décider : leurs seuils n'ont jamais été réglés sur une carte (séance de mesure).
Verdict judge_hardened(std::span<const ImageScore> scores, std::span<const ImageScore> hard);

// Garder l'accumulateur f16 hors blocs (étape 3.5a) : sa variante passe la porte, la variante f32 aussi, et
// le PSNR moyen perd au plus kMaxAcc16PsnrLossDb.
bool keep_acc16(const Verdict& f32, const Verdict& f16);

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
  // Version 2 (étape 3.7b) : ce qui lie le paquet à l'appareil, aux noyaux et aux tailles jugées.
  std::string gpu_arch;  // « sm_89 »
  int kernels_version = 0;
  int blocks = 0;
  int channels = 0;  // réels : 48 ou 64
  bool acc16 = false;
  std::vector<std::string> validated_sizes;  // « 1280x720 » : tailles où la porte a passé
  std::vector<std::string> refused_sizes;    // et où elle a échoué (rien n'est refait à cette taille)
  bool timed_during_playback = false;        // B02-08 : les temps pris pendant la lecture, sous sa charge
};

std::string receipt_text(const Receipt& receipt, std::string_view comment);

// La ligne TOML d'une liste de tailles : « validated_sizes = ["1280x720", "1920x1080"] ».
std::string sizes_line(std::string_view key, const std::vector<std::string>& sizes);
// Le texte d'un reçu v2 dont les lignes validated_sizes et refused_sizes sont remplacées (validation seule).
std::string with_sizes(std::string_view receipt, const std::vector<std::string>& validated,
                       const std::vector<std::string>& refused);
// « 1280x720 » : la taille de la découpe que la préparation ferait à width × height (kernel_crop).
std::string size_key(int width, int height);

// Les tailles d'image que la dérivation prend : au plus width × height, ramenées à ce que les noyaux
// acceptent (hauteur et largeur paires, étape 3.1). {0, 0} si rien n'en reste.
struct Crop {
  int width = 0;
  int height = 0;
};
Crop kernel_crop(int width, int height, int max_width = 1920, int max_height = 1088);

}  // namespace ajp::gpu::span_fp8
