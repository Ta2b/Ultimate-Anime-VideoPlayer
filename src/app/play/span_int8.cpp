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
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::size_t kShortSha = 16;

struct KindFacts {
  std::string_view receipt_format;
  std::string_view infix;
  std::string_view
      extension;  // du fichier de la variante : un ONNX, ou le paquet « .span8 » des noyaux CUTLASS
  double min_psnr_mean;
  double min_psnr_worst;
  double min_ssim_worst;
};

// Le FP32 se garde aux seuils de l'INT8 : il ne s'écarte du FP16 que de bruit de calcul.
constexpr KindFacts facts(SpanVariantKind kind) {
  switch (kind) {
    case SpanVariantKind::Int8:
      return {"ajp-span-int8-v1",  "-int8-",         ".onnx", kInt8MinPsnrMeanDb,
              kInt8MinPsnrWorstDb, kInt8MinSsimWorst};
    case SpanVariantKind::Fp8:
      return {"ajp-span-fp8-v1", "-fp8-", ".onnx", kFp8MinPsnrMeanDb, kFp8MinPsnrWorstDb, kFp8MinSsimWorst};
    case SpanVariantKind::Fp32:
      return {"ajp-span-fp32-v1",  "-fp32-",         ".onnx", kInt8MinPsnrMeanDb,
              kInt8MinPsnrWorstDb, kInt8MinSsimWorst};
    case SpanVariantKind::Fp8Cutlass:  // des noyaux écrits à la main : les seuils de l'INT8, pas ceux du FP8
                                       // par Q/DQ
      return {"ajp-span-fp8c-v1",  "-fp8c-",         ".span8", kInt8MinPsnrMeanDb,
              kInt8MinPsnrWorstDb, kInt8MinSsimWorst};
  }
  return facts(SpanVariantKind::Int8);
}

SpanInt8Lookup refused(std::string note) {
  return SpanInt8Lookup{.variant = std::nullopt, .note = std::move(note)};
}

}  // namespace

TierSupport span_tiers(const std::filesystem::path& span, const std::filesystem::path& dir, bool fp8_engine,
                       bool cutlass_engine) {
  return TierSupport{
      .performance =
          (cutlass_engine && find_span_variant(SpanVariantKind::Fp8Cutlass, span, dir).variant.has_value()) ||
          (fp8_engine && find_span_variant(SpanVariantKind::Fp8, span, dir).variant.has_value()),
      .quality = find_span_variant(SpanVariantKind::Fp32, span, dir).variant.has_value()};
}

std::string_view variant_text(SpanVariantKind kind) {
  switch (kind) {
    case SpanVariantKind::Int8:
      return "INT8";
    case SpanVariantKind::Fp8:
      return "FP8";
    case SpanVariantKind::Fp32:
      return "FP32";
    case SpanVariantKind::Fp8Cutlass:
      return "FP8 CUTLASS";
  }
  return "INT8";
}

SpanInt8Lookup find_span_variant(SpanVariantKind kind, const std::filesystem::path& span,
                                 const std::filesystem::path& dir) {
  const KindFacts limits = facts(kind);
  auto sha = prepare::sha256_file_known(span);
  if (!sha) return refused(sha.error().detail);
  // Par path et non par string : un nom Unicode ne passe pas par la page de code sous Windows.
  std::filesystem::path stem = dir / span.stem();
  stem += std::format("{}{}", limits.infix, sha->substr(0, kShortSha));
  std::filesystem::path receipt_path = stem;
  receipt_path += ".toml";
  std::filesystem::path onnx = stem;
  onnx += limits.extension;
  std::error_code error;
  if (!std::filesystem::exists(receipt_path, error)) return {};

  const std::string where = prepare::utf8_path(receipt_path.filename());
  std::ifstream in(receipt_path, std::ios::binary);  // pas parse_file : un chemin Unicode sous Windows
  std::ostringstream text;
  text << in.rdbuf();
  toml::parse_result parsed = toml::parse(text.str());
  if (!parsed) return refused(std::format("{} illisible : {}", where, parsed.error().description()));
  const toml::table& receipt = parsed.table();
  if (receipt["format"].value_or(std::string_view()) != limits.receipt_format)
    return refused(std::format("{} : format inconnu", where));
  if (receipt["source_sha256"].value_or(std::string_view()) != *sha)
    return refused(std::format("{} : reçu d'un autre modèle", where));
  const SpanInt8 variant{.kind = kind,
                         .onnx = onnx,
                         .psnr_mean_db = receipt["psnr_mean_db"].value_or(0.0),
                         .psnr_worst_db = receipt["psnr_worst_db"].value_or(0.0),
                         .ssim_worst = receipt["ssim_worst"].value_or(0.0)};
  const std::string scores = std::format("PSNR moyen {:.1f} dB, pire {:.1f} dB, pire SSIM {:.4f}",
                                         variant.psnr_mean_db, variant.psnr_worst_db, variant.ssim_worst);
  if (!receipt["passed"].value_or(false) || variant.psnr_mean_db < limits.min_psnr_mean ||
      variant.psnr_worst_db < limits.min_psnr_worst || variant.ssim_worst < limits.min_ssim_worst) {
    return refused(std::format("{} : sous les garde-fous ({} ; il faut {:.0f} dB, {:.0f} dB et {:.3f})",
                               where, scores, limits.min_psnr_mean, limits.min_psnr_worst,
                               limits.min_ssim_worst));
  }
  auto model_sha = prepare::sha256_file_known(variant.onnx);
  if (!model_sha) return refused(model_sha.error().detail);
  if (receipt["model_sha256"].value_or(std::string_view()) != *model_sha)
    return refused(std::format("{} : modifié depuis son reçu", prepare::utf8_path(onnx.filename())));
  return SpanInt8Lookup{.variant = variant,
                        .note = std::format("{} ({})", prepare::utf8_path(onnx.filename()), scores)};
}

std::optional<std::string> span_int8_source(const std::string& file, SpanVariantKind* kind) {
  const std::string_view name(file);
  constexpr std::string_view kSuffix = ".onnx";
  for (const SpanVariantKind candidate :
       {SpanVariantKind::Int8, SpanVariantKind::Fp8, SpanVariantKind::Fp32, SpanVariantKind::Fp8Cutlass}) {
    const std::string_view extension = facts(candidate).extension;
    if (!name.ends_with(extension)) continue;
    const std::string_view stem = name.substr(0, name.size() - extension.size());
    const std::string_view infix = facts(candidate).infix;
    if (stem.size() <= infix.size() + kShortSha) continue;
    const std::string_view tail = stem.substr(stem.size() - infix.size() - kShortSha);
    if (!tail.starts_with(infix)) continue;
    bool hex = true;
    for (const char c : tail.substr(infix.size()))
      hex = hex && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    if (!hex) continue;
    if (kind != nullptr) *kind = candidate;
    return std::string(stem.substr(0, stem.size() - tail.size())) +
           std::string(kSuffix);  // la source est un ONNX
  }
  return std::nullopt;
}

}  // namespace ajp::app::play
