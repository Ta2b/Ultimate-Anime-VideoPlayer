// Les doublures de la dérivation native du paquet « .span8 » (span_derive_test.cpp, span_speed_test.cpp) :
// un dossier « modeles » avec un faux SPAN, douze images, et un « backend » qui rend des mesures données.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../../support/span_fp8_synthetic.h"
#include "app/play/span_derive.h"
#include "app/play/span_int8.h"
#include "gpu/span_fp8/package.h"
#include "media/stills.h"
#include "prepare/sha256.h"

namespace ajp::app::play::span_test {

namespace fs = std::filesystem;
using gpu::span_fp8::Package;

// Un dossier « modeles » et un faux SPAN dedans (le reçu s'attache au SHA-256 du fichier).
struct Folder {
  // Un dossier par essai : ctest lance les cas en parallèle, un nom fixe ferait se marcher dessus.
  Folder()
      : root(fs::temp_directory_path() /
             std::format("ajp-span-derive-test-{}",
                         std::chrono::steady_clock::now().time_since_epoch().count())) {
    fs::remove_all(root);
    fs::create_directories(models());
    model = root / "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
    std::ofstream(model, std::ios::binary) << "un SPAN pour les essais";
  }
  ~Folder() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  Folder(const Folder&) = delete;
  Folder& operator=(const Folder&) = delete;

  fs::path models() const { return root / "modeles"; }
  fs::path package() const {
    return span_variant_path(SpanVariantKind::Fp8Cutlass, model, models(),
                             prepare::sha256_file(model).value());
  }
  fs::path receipt() const {
    fs::path path = package();
    path.replace_extension(".toml");
    return path;
  }
  SpanDerivePlan plan() const {
    return {.model = model, .models_dir = models(), .load_weights = [] {
              return std::expected<Package, Refusal>(gpu::span_fp8::testing::synthetic());
            }};
  }
  std::vector<std::string> files() const {
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(models()))
      names.push_back(entry.path().filename().string());
    std::ranges::sort(names);
    return names;
  }

  fs::path root;
  fs::path model;
};

// Douze images 128×4 dont le premier octet dit leur rang.
inline std::vector<media::Still> twelve() {
  std::vector<media::Still> stills;
  for (int i = 0; i < 12; ++i) {
    media::Still still;
    still.width = 128;
    still.height = 4;
    still.pts_us = 1'000'000 * i;
    still.rgb.assign(std::size_t{3} * 128 * 4, static_cast<std::uint8_t>(i));
    stills.push_back(std::move(still));
  }
  return stills;
}

inline SpanStillsSource source_of(std::vector<media::Still> stills) {
  return [stills = std::move(stills)](const std::stop_token&, const std::function<void(float)>& progress) {
    if (progress) progress(1.0F);
    return std::expected<std::vector<media::Still>, Refusal>(stills);
  };
}

// Un « backend » qui note ce qu'on lui demande et rend des mesures données.
struct FakeBackend {
  double psnr = 50.0;
  double ssim = 0.995;
  double psnr16 = 0.0;              // non nul : la variante à accumulateur f16 est mesurée à ce PSNR (3.5a)
  double ms16 = 8.9;                // et à ce temps (contre 9,2 ms pour l'accumulateur f32)
  double error_p999 = 1.0 / 255.0;  // 99,9e centile de l'erreur de chaque image (3.7a)
  double hard_psnr = 40.0;          // PSNR de chaque cas dur rendu (3.7a)
  bool refuse = false;
  double trt_ms = 0.0;           // le SPAN FP16 de TensorRT chronométré, quand la requête l'apporte (C02-01)
  float validation_peak = 0.0F;  // maximum relevé sur la validation, pour chaque tenseur (C02-04)
  mutable gpu::SpanFp8Request seen;
  SpanDeriver deriver() const {
    return [this](const gpu::SpanFp8Request& request, gpu::Progress& progress) {
      seen = request;
      if (progress.report) progress.report(0.5F, "calibration");
      if (refuse)
        return std::expected<gpu::SpanFp8Derivation, Refusal>(
            std::unexpected(Refusal{RefusalCode::GpuMemoryExhausted, "plus de mémoire vidéo"}));
      gpu::SpanFp8Derivation derivation;
      derivation.scales.fill(0.05F);
      derivation.peaks.fill(22.4F);
      derivation.scores.assign(request.validation.size(),
                               {.psnr_db = psnr, .ssim = ssim, .error_p999 = error_p999});
      derivation.hard_scores.assign(3, {.psnr_db = hard_psnr, .ssim = ssim, .error_p999 = error_p999});
      derivation.kernels_ms = 9.2;
      if (psnr16 > 0.0) {
        derivation.scores_acc16.assign(request.validation.size(), {.psnr_db = psnr16, .ssim = ssim});
        derivation.kernels_ms_acc16 = ms16;
      }
      derivation.validation_peaks.fill(validation_peak);
      if (request.fp16) derivation.trt_fp16_ms = trt_ms;
      derivation.timed_size = std::format("{}x{}", request.timing_width, request.timing_height);
      derivation.reference = "cutlass-fp16";
      derivation.gpu_arch = "sm_89";
      derivation.kernels_version = 1;
      return std::expected<gpu::SpanFp8Derivation, Refusal>(derivation);
    };
  }
};

}  // namespace ajp::app::play::span_test
