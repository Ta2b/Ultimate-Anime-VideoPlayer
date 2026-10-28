#include "gpu/span_fp8/derive.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>

namespace ajp::gpu::span_fp8 {
namespace {

std::string quoted(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  out += '"';
  return out;
}

}  // namespace

std::expected<std::array<float, kScaleCount>, Refusal> scales_from_peaks(std::span<const float> peaks) {
  if (peaks.size() != static_cast<std::size_t>(kScaleCount))
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   std::format("{} maxima reçus, {} attendus", peaks.size(), kScaleCount)});
  std::array<float, kScaleCount> scales{};
  for (std::size_t i = 0; i < peaks.size(); ++i) {
    if (!std::isfinite(peaks[i]))
      return std::unexpected(Refusal{
          RefusalCode::FormatUnsupported,
          std::format("le tenseur {} déborde en FP16 sur les images de calibration : SPAN non pris", i)});
    if (!(peaks[i] > 0.0F))
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                     std::format("le tenseur {} est nul partout : images de calibration "
                                                 "inutilisables",
                                                 i)});
    scales[i] = peaks[i] / kE4m3Max;
  }
  return scales;
}

Verdict judge(std::span<const ImageScore> scores, double min_psnr_mean, double min_psnr_worst,
              double min_ssim_worst) {
  Verdict verdict;
  if (scores.empty()) return verdict;
  double sum = 0.0;
  verdict.psnr_worst_db = scores.front().psnr_db;
  verdict.ssim_worst = scores.front().ssim;
  for (const ImageScore& score : scores) {
    sum += score.psnr_db;
    verdict.psnr_worst_db = std::min(verdict.psnr_worst_db, score.psnr_db);
    verdict.ssim_worst = std::min(verdict.ssim_worst, score.ssim);
  }
  verdict.psnr_mean_db = sum / static_cast<double>(scores.size());
  verdict.passed = verdict.psnr_mean_db >= min_psnr_mean && verdict.psnr_worst_db >= min_psnr_worst &&
                   verdict.ssim_worst >= min_ssim_worst;
  return verdict;
}

std::string receipt_text(const Receipt& r, std::string_view comment) {
  std::string out = std::format("# {}\n", comment);
  const auto text = [&](std::string_view key, std::string_view value) {
    out += std::format("{} = {}\n", key, quoted(value));
  };
  const auto real = [&](std::string_view key, double value) {
    out += std::format("{} = {:.4f}\n", key, value);
  };
  const auto whole = [&](std::string_view key, int value) { out += std::format("{} = {}\n", key, value); };
  text("format", kReceiptFormat);
  text("source", r.source);
  text("source_sha256", r.source_sha256);
  text("model_sha256", r.model_sha256);
  text("quantizer", r.quantizer);
  text("precision", "fp8");
  real("speedup", r.speedup);
  real("kernels_ms", r.kernels_ms);
  real("trt_fp16_ms", r.trt_fp16_ms);
  text("method", "max");
  whole("calibration_images", r.calibration_images);
  whole("calibration_crops", r.calibration_crops);
  whole("calibration_crop", r.calibration_crop);
  text("corpus_sha256", r.corpus_sha256);
  whole("validation_images", r.validation_images);
  text("validation_size", r.validation_size);
  real("psnr_mean_db", r.verdict.psnr_mean_db);
  real("psnr_worst_db", r.verdict.psnr_worst_db);
  real("ssim_worst", r.verdict.ssim_worst);
  out += std::format("passed = {}\n", r.verdict.passed ? "true" : "false");
  if (!r.reference.empty()) text("reference", r.reference);
  if (!r.derived_by.empty()) text("derived_by", r.derived_by);
  return out;
}

Crop kernel_crop(int width, int height, int max_width, int max_height) {
  const int h = std::min(height, max_height) / 4 * 4;
  const int w = std::min(width, max_width) / 128 * 128;
  if (h <= 0 || w <= 0) return {};
  return Crop{.width = w, .height = h};
}

}  // namespace ajp::gpu::span_fp8
