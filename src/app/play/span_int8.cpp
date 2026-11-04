#include "app/play/span_int8.h"

#include <cstdint>
#include <format>
#include <fstream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#include "app/play/i18n.h"
#include "gpu/span_fp8/derive.h"
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
  // Le moteur qui sait les noyaux CUTLASS suffit : le lecteur fabrique lui-même le paquet au premier
  // démarrage avec ce cran (span_auto.h), et la lecture reste en FP16 d'ici là. Le b5f48 (48 canaux, 5 blocs)
  // est pris depuis l'étape 3.2c, rangé en 64 canaux.
  const bool cutlass_possible = cutlass_engine;
  return TierSupport{
      .performance = cutlass_possible ||
                     (fp8_engine && find_span_variant(SpanVariantKind::Fp8, span, dir).variant.has_value()),
      .quality = find_span_variant(SpanVariantKind::Fp32, span, dir).variant.has_value()};
}

std::filesystem::path span_variant_path(SpanVariantKind kind, const std::filesystem::path& span,
                                        const std::filesystem::path& dir, std::string_view sha256) {
  const KindFacts limits = facts(kind);
  // Par path et non par string : un nom Unicode ne passe pas par la page de code sous Windows.
  std::filesystem::path path = dir / span.stem();
  path += std::format("{}{}", limits.infix, sha256.substr(0, kShortSha));
  path += limits.extension;
  return path;
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
                                 const std::filesystem::path& dir, const SpanFp8Context* fp8) {
  const KindFacts limits = facts(kind);
  auto sha = prepare::sha256_file_known(span);
  if (!sha) return refused(sha.error().detail);
  const std::filesystem::path onnx = span_variant_path(kind, span, dir, *sha);
  std::filesystem::path receipt_path = onnx;
  receipt_path.replace_extension(".toml");
  std::error_code error;
  if (!std::filesystem::exists(receipt_path, error)) return {};

  const std::string where = prepare::utf8_path(receipt_path.filename());
  std::ifstream in(receipt_path, std::ios::binary);  // pas parse_file : un chemin Unicode sous Windows
  std::ostringstream text;
  text << in.rdbuf();
  toml::parse_result parsed = toml::parse(text.str());
  if (!parsed) return refused(i18n::tr_format("{} unreadable: {}", where, parsed.error().description()));
  const toml::table& receipt = parsed.table();
  const std::string_view format = receipt["format"].value_or(std::string_view());
  const bool v2 = kind == SpanVariantKind::Fp8Cutlass && format == gpu::span_fp8::kReceiptFormat;
  if (format != limits.receipt_format && !v2) return refused(i18n::tr_format("{}: unknown format", where));
  if (receipt["source_sha256"].value_or(std::string_view()) != *sha)
    return refused(i18n::tr_format("{}: receipt from another model", where));
  const SpanInt8 variant{.kind = kind,
                         .onnx = onnx,
                         .psnr_mean_db = receipt["psnr_mean_db"].value_or(0.0),
                         .psnr_worst_db = receipt["psnr_worst_db"].value_or(0.0),
                         .ssim_worst = receipt["ssim_worst"].value_or(0.0),
                         .receipt_v2 = v2};
  const std::string scores = std::format("PSNR moyen {:.1f} dB, pire {:.1f} dB, pire SSIM {:.4f}",
                                         variant.psnr_mean_db, variant.psnr_worst_db, variant.ssim_worst);
  // Un refus d'une porte plus ancienne, ou d'une autre carte ou d'autres noyaux, se refait (stale).
  const bool old_gate =
      v2 && fp8 != nullptr && receipt["gate_version"].value_or(std::int64_t{1}) < gpu::span_fp8::kGateVersion;
  const auto stale = [](std::string note) {
    SpanInt8Lookup out = refused(std::move(note));
    out.stale = true;
    return out;
  };
  if (!receipt["passed"].value_or(false) || variant.psnr_mean_db < limits.min_psnr_mean ||
      variant.psnr_worst_db < limits.min_psnr_worst || variant.ssim_worst < limits.min_ssim_worst) {
    const std::string note =
        i18n::tr_format("{}: below the quality limits ({}; {:.0f} dB, {:.0f} dB and {:.3f} are required)",
                        where, scores, limits.min_psnr_mean, limits.min_psnr_worst, limits.min_ssim_worst);
    return old_gate ? stale(note) : refused(note);
  }
  auto model_sha = prepare::sha256_file_known(variant.onnx);
  if (!model_sha) return refused(model_sha.error().detail);
  if (receipt["model_sha256"].value_or(std::string_view()) != *model_sha)
    return refused(i18n::tr_format("{}: changed since its receipt", prepare::utf8_path(onnx.filename())));
  if (v2 && fp8 != nullptr) {
    // Étape 3.7b : le paquet ne vaut que pour l'appareil, les noyaux et les tailles où la porte l'a jugé.
    const std::string_view arch = receipt["gpu_arch"].value_or(std::string_view());
    if (arch != fp8->gpu_arch)
      return stale(
          i18n::tr_format("{}: derived on {}, this GPU is {}: prepare it again", where, arch, fp8->gpu_arch));
    const auto kernels = receipt["kernels_version"].value_or(std::int64_t{0});
    if (kernels != fp8->kernels_version)
      return stale(i18n::tr_format("{}: kernels version {}, this engine has {}: prepare it again", where,
                                   kernels, fp8->kernels_version));
    const std::string size = gpu::span_fp8::size_key(fp8->width, fp8->height);
    const auto listed = [&](std::string_view key) {
      if (const toml::array* sizes = receipt[key].as_array())
        for (const toml::node& entry : *sizes)
          if (entry.value_or(std::string_view()) == size) return true;
      return false;
    };
    if (listed("refused_sizes")) {
      const std::string note = i18n::tr_format("{}: rejected at {}", where, size);
      return old_gate ? stale(note) : refused(note);
    }
    if (!listed("validated_sizes")) {
      SpanInt8Lookup pending = refused(i18n::tr_format("{}: not validated at {} yet", where, size));
      pending.unvalidated_size = size;
      return pending;
    }
  }
  return SpanInt8Lookup{.variant = variant,
                        .note = std::format("{} ({})", prepare::utf8_path(onnx.filename()), scores)};
}

std::vector<SpanVariantKind> span_variant_order(PrecisionTier tier, const SpanVariantCaps& caps,
                                                bool strict) {
  std::vector<SpanVariantKind> order;
  if (tier == PrecisionTier::Performance && caps.cutlass) order.push_back(SpanVariantKind::Fp8Cutlass);
  if (tier == PrecisionTier::Performance && caps.fp8 && !strict) order.push_back(SpanVariantKind::Fp8);
  if (tier != PrecisionTier::Quality && caps.int8) order.push_back(SpanVariantKind::Int8);
  if (tier == PrecisionTier::Quality) order.push_back(SpanVariantKind::Fp32);
  return order;
}

std::expected<ReceiptSizes, Refusal> read_receipt_sizes(const std::filesystem::path& receipt_path) {
  std::ifstream in(receipt_path, std::ios::binary);
  if (!in)
    return std::unexpected(
        Refusal{RefusalCode::FileUnreadable, std::format("{} illisible", prepare::utf8_path(receipt_path))});
  std::ostringstream text;
  text << in.rdbuf();
  ReceiptSizes out{.text = text.str(), .validated = {}, .refused = {}};
  toml::parse_result parsed = toml::parse(out.text);
  if (!parsed)
    return std::unexpected(Refusal{
        RefusalCode::FileUnreadable,
        std::format("{} illisible : {}", prepare::utf8_path(receipt_path), parsed.error().description())});
  const auto take = [&](std::string_view key, std::vector<std::string>& into) {
    if (const toml::array* sizes = parsed.table()[key].as_array())
      for (const toml::node& entry : *sizes)
        if (const auto value = entry.value<std::string>()) into.push_back(*value);
  };
  take("validated_sizes", out.validated);
  take("refused_sizes", out.refused);
  return out;
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
