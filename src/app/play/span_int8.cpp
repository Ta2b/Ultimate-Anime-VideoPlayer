#include "app/play/span_int8.h"

#include <format>
#include <fstream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#include "prepare/sha256.h"
#include "prepare/utf8.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::string_view kReceiptFormat = "ajp-span-int8-v1";
constexpr std::string_view kInfix = "-int8-";
constexpr std::size_t kShortSha = 16;

SpanInt8Lookup refused(std::string note) {
  return SpanInt8Lookup{.variant = std::nullopt, .note = std::move(note)};
}

}  // namespace

SpanInt8Lookup find_span_int8(const std::filesystem::path& span, const std::filesystem::path& dir) {
  auto sha = prepare::sha256_file_known(span);
  if (!sha) return refused(sha.error().detail);
  // Par path et non par string : un nom Unicode ne passe pas par la page de code sous Windows.
  std::filesystem::path stem = dir / span.stem();
  stem += std::format("{}{}", kInfix, sha->substr(0, kShortSha));
  std::filesystem::path receipt_path = stem;
  receipt_path += ".toml";
  std::filesystem::path onnx = stem;
  onnx += ".onnx";
  std::error_code error;
  if (!std::filesystem::exists(receipt_path, error)) return {};

  const std::string where = prepare::utf8_path(receipt_path.filename());
  std::ifstream in(receipt_path, std::ios::binary);  // pas parse_file : un chemin Unicode sous Windows
  std::ostringstream text;
  text << in.rdbuf();
  toml::parse_result parsed = toml::parse(text.str());
  if (!parsed) return refused(std::format("{} illisible : {}", where, parsed.error().description()));
  const toml::table& receipt = parsed.table();
  if (receipt["format"].value_or(std::string_view()) != kReceiptFormat)
    return refused(std::format("{} : format inconnu", where));
  if (receipt["source_sha256"].value_or(std::string_view()) != *sha)
    return refused(std::format("{} : reçu d'un autre modèle", where));
  const SpanInt8 variant{.onnx = onnx,
                         .psnr_mean_db = receipt["psnr_mean_db"].value_or(0.0),
                         .psnr_worst_db = receipt["psnr_worst_db"].value_or(0.0),
                         .ssim_worst = receipt["ssim_worst"].value_or(0.0)};
  const std::string scores = std::format("PSNR moyen {:.1f} dB, pire {:.1f} dB, pire SSIM {:.4f}",
                                         variant.psnr_mean_db, variant.psnr_worst_db, variant.ssim_worst);
  if (!receipt["passed"].value_or(false) || variant.psnr_mean_db < kInt8MinPsnrMeanDb ||
      variant.psnr_worst_db < kInt8MinPsnrWorstDb || variant.ssim_worst < kInt8MinSsimWorst) {
    return refused(std::format("{} : sous les garde-fous ({} ; il faut {:.0f} dB, {:.0f} dB et {:.3f})",
                               where, scores, kInt8MinPsnrMeanDb, kInt8MinPsnrWorstDb, kInt8MinSsimWorst));
  }
  auto model_sha = prepare::sha256_file_known(variant.onnx);
  if (!model_sha) return refused(model_sha.error().detail);
  if (receipt["model_sha256"].value_or(std::string_view()) != *model_sha)
    return refused(std::format("{} : modifié depuis son reçu", prepare::utf8_path(onnx.filename())));
  return SpanInt8Lookup{.variant = variant,
                        .note = std::format("{} ({})", prepare::utf8_path(onnx.filename()), scores)};
}

std::optional<std::string> span_int8_source(const std::string& file) {
  const std::string_view name(file);
  constexpr std::string_view kSuffix = ".onnx";
  if (!name.ends_with(kSuffix)) return std::nullopt;
  const std::string_view stem = name.substr(0, name.size() - kSuffix.size());
  if (stem.size() <= kInfix.size() + kShortSha) return std::nullopt;
  const std::string_view tail = stem.substr(stem.size() - kInfix.size() - kShortSha);
  if (!tail.starts_with(kInfix)) return std::nullopt;
  for (const char c : tail.substr(kInfix.size())) {
    if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) return std::nullopt;
  }
  return std::string(stem.substr(0, stem.size() - tail.size())) + std::string(kSuffix);
}

}  // namespace ajp::app::play
