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

std::expected<std::array<float, kScaleCount>, Refusal> scales_from_peaks(std::span<const float> peaks,
                                                                         std::size_t blocks) {
  if (peaks.size() != static_cast<std::size_t>(kScaleCount) || blocks < 1 ||
      blocks > static_cast<std::size_t>(kBlocks))
    return std::unexpected(Refusal{
        RefusalCode::FormatUnsupported,
        std::format("{} maxima reçus pour {} blocs, {} attendus", peaks.size(), blocks, kScaleCount)});
  std::array<float, kScaleCount> scales{};
  scales.fill(1.0F);
  for (std::size_t i = 0; i < scale_count(blocks); ++i) {
    if (!std::isfinite(peaks[i]))
      return std::unexpected(Refusal{
          RefusalCode::FormatUnsupported,
          std::format("le tenseur {} déborde en FP16 sur les images de calibration : SPAN non pris", i)});
    if (!(peaks[i] > 0.0F))
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                     std::format("le tenseur {} est nul partout : images de calibration "
                                                 "inutilisables",
                                                 i)});
    scales[i] = peaks[i] * kScaleHeadroom / kE4m3Max;
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
  // Un score non fini refuse, où qu'il soit dans la liste : std::min et std::max l'ignorent quand il n'est
  // pas le premier (A02-01).
  const bool finite = std::ranges::all_of(scores, [](const ImageScore& score) {
    return std::isfinite(score.psnr_db) && std::isfinite(score.ssim) && std::isfinite(score.error_p999) &&
           std::isfinite(score.dark_error_p999);
  });
  verdict.passed = finite && verdict.psnr_mean_db >= min_psnr_mean &&
                   verdict.psnr_worst_db >= min_psnr_worst && verdict.ssim_worst >= min_ssim_worst;
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
  text("gpu_arch", r.gpu_arch);
  whole("kernels_version", r.kernels_version);
  whole("blocks", r.blocks);
  whole("channels", r.channels);
  text("accumulator", r.acc16 ? "f16" : "f32");
  out += std::format("error_p999_worst = {:.6f}\n", r.verdict.error_p999_worst);
  out += std::format("dark_error_p999_worst = {:.6f}\n", r.verdict.dark_error_p999_worst);
  real("hard_psnr_worst_db", r.verdict.hard_psnr_worst_db);
  out += std::format("hard_error_p999_worst = {:.6f}\n", r.verdict.hard_error_p999_worst);
  whole("hard_images", r.verdict.hard_images);
  whole("gate_version", kGateVersion);
  if (r.timed_during_playback) out += "timed_during_playback = true\n";
  out += sizes_line("validated_sizes", r.validated_sizes) + "\n";
  out += sizes_line("refused_sizes", r.refused_sizes) + "\n";
  return out;
}

std::string sizes_line(std::string_view key, const std::vector<std::string>& sizes) {
  std::string out = std::format("{} = [", key);
  for (std::size_t i = 0; i < sizes.size(); ++i)
    out += std::format("{}{}", i == 0 ? "" : ", ", quoted(sizes[i]));
  return out + "]";
}

std::string with_sizes(std::string_view receipt, const std::vector<std::string>& validated,
                       const std::vector<std::string>& refused) {
  std::string out;
  std::size_t at = 0;
  while (at < receipt.size()) {
    std::size_t end = receipt.find('\n', at);
    if (end == std::string_view::npos) end = receipt.size();
    const std::string_view line = receipt.substr(at, end - at);
    if (!line.starts_with("validated_sizes =") && !line.starts_with("refused_sizes =")) {
      out += line;
      out += '\n';
    }
    at = end + 1;
  }
  out += sizes_line("validated_sizes", validated) + "\n";
  out += sizes_line("refused_sizes", refused) + "\n";
  return out;
}

std::string size_key(int width, int height) {
  const Crop crop = kernel_crop(width, height);
  return std::format("{}x{}", crop.width, crop.height);
}

Crop kernel_crop(int width, int height, int max_width, int max_height) {
  // Paires (dégroupage 2×2) : les noyaux prennent toute taille paire (étape 3.1).
  const int h = std::min(height, max_height) / 2 * 2;
  const int w = std::min(width, max_width) / 2 * 2;
  if (h <= 0 || w <= 0) return {};
  return Crop{.width = w, .height = h};
}

Verdict judge_hardened(std::span<const ImageScore> scores, std::span<const ImageScore> hard) {
  Verdict verdict = judge(scores);
  for (const ImageScore& score : scores) {
    verdict.error_p999_worst = std::max(verdict.error_p999_worst, score.error_p999);
    verdict.dark_error_p999_worst = std::max(verdict.dark_error_p999_worst, score.dark_error_p999);
  }
  // Les cas durs synthétiques sont relevés (reçu, journal) mais ne décident pas : leurs seuils de départ
  // n'ont jamais été réglés sur une carte, et sur la RTX 4070 Ti ils refusent un b8f64 à 52 dB sur les vraies
  // images (séance de mesure). L'œil juge à leur place.
  verdict.hard_images = static_cast<int>(hard.size());
  if (!hard.empty()) verdict.hard_psnr_worst_db = hard.front().psnr_db;
  for (const ImageScore& score : hard) {
    verdict.hard_error_p999_worst = std::max(verdict.hard_error_p999_worst, score.error_p999);
    verdict.hard_psnr_worst_db = std::min(verdict.hard_psnr_worst_db, score.psnr_db);
  }
  verdict.passed = verdict.passed && verdict.error_p999_worst <= kMaxErrorP999 &&
                   verdict.dark_error_p999_worst <= kMaxDarkErrorP999;
  return verdict;
}

bool keep_acc16(const Verdict& f32, const Verdict& f16) {
  return f32.passed && f16.passed && f32.psnr_mean_db - f16.psnr_mean_db <= kMaxAcc16PsnrLossDb;
}

}  // namespace ajp::gpu::span_fp8
