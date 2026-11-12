#include "app/play/span_int8.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#include "app/play/i18n.h"
#include "gpu/span_fp8/derive.h"
#include "media/track.h"
#include "media/yuv_sample.h"
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
      return {gpu::span_fp8::kReceiptFormatV1,
              "-fp8c-",
              ".span8",
              kInt8MinPsnrMeanDb,
              kInt8MinPsnrWorstDb,
              kInt8MinSsimWorst};
  }
  return facts(SpanVariantKind::Int8);
}

SpanInt8Lookup refused(std::string note) {
  return SpanInt8Lookup{.variant = std::nullopt, .note = std::move(note)};
}

// Un reçu ou un paquet abîmé (coupure pendant l'écriture : vide, tronqué) n'est pas un jugement de la porte :
// il se refait comme un reçu périmé, sans bloquer la préparation automatique (A02-14).
SpanInt8Lookup stale(std::string note) {
  SpanInt8Lookup out = refused(std::move(note));
  out.stale = true;
  return out;
}

// Le reçu lu, avec un format ; sinon son refus, périmé (stale).
std::expected<toml::table, SpanInt8Lookup> read_receipt(const std::filesystem::path& receipt_path,
                                                        const std::string& where) {
  std::ifstream in(receipt_path, std::ios::binary);  // pas parse_file : un chemin Unicode sous Windows
  std::ostringstream text;
  text << in.rdbuf();
  toml::parse_result parsed = toml::parse(text.str());
  if (!parsed)
    return std::unexpected(stale(i18n::tr_format("{} unreadable: {}", where, parsed.error().description())));
  if (parsed.table()["format"].value_or(std::string_view()).empty())
    return std::unexpected(stale(i18n::tr_format("{} unreadable: {}", where, i18n::tr("empty receipt"))));
  return std::move(parsed).table();
}

// Le paquet ne correspond plus à son reçu : à refaire (stale), ou refusé s'il ne se lit pas. Un paquet
// disparu (cache nettoyé, coupure entre deux renommages) alors que son reçu l'admet est à refaire, pas refusé
// (B07-12).
std::optional<SpanInt8Lookup> package_differs(const std::filesystem::path& package,
                                              std::string_view receipt_sha, const std::string& where) {
  std::error_code error;
  if (!std::filesystem::exists(package, error))
    return stale(i18n::tr_format("{}: package missing, prepare it again", where));
  auto model_sha = prepare::sha256_file_known(package);
  if (!model_sha) return refused(model_sha.error().detail);
  if (receipt_sha != *model_sha)
    return stale(i18n::tr_format("{}: changed since its receipt", prepare::utf8_path(package.filename())));
  return std::nullopt;
}

// Pourquoi un reçu refuse sa variante : le gain sur le FP16 trop faible (B02-01), le 99,9e centile (la porte
// durcie, quand PSNR et SSIM passent), ou les seuils de qualité.
std::string refusal_note(const toml::table& receipt, const std::string& where, const KindFacts& limits,
                         const std::string& scores, bool scores_pass, bool v2) {
  const double speedup = receipt["speedup"].value_or(0.0);
  if (v2 && scores_pass && speedup > 0.0 && speedup < gpu::span_fp8::kMinSpeedup)
    return i18n::tr_format("{}: no faster than TensorRT FP16 ({:.2f}× as fast, {:.2f}× is required)", where,
                           speedup, gpu::span_fp8::kMinSpeedup);
  if (scores_pass && v2)
    return i18n::tr_format("{}: rejected at the 99.9th percentile ({:.1f}/255, dark {:.1f}/255; {} passes)",
                           where, 255.0 * receipt["error_p999_worst"].value_or(0.0),
                           255.0 * receipt["dark_error_p999_worst"].value_or(0.0), scores);
  return i18n::tr_format("{}: below the quality limits ({}; {:.0f} dB, {:.0f} dB and {:.3f} are required)",
                         where, scores, limits.min_psnr_mean, limits.min_psnr_worst, limits.min_ssim_worst);
}

// Étape 3.7b : un paquet admis ne vaut que pour la porte, l'appareil et les noyaux qui l'ont jugé ; sinon il
// se refait. Une porte plus ancienne n'avait ni mesuré son gain sur le FP16 ni gardé de marge à ses échelles
// (B02-01, B02-02), ou mesurait les deux temps de deux façons, gardait la variante f16 sans gain et ne voyait
// pas l'écrêtage (porte 4, chantier E01).
std::optional<SpanInt8Lookup> outdated(const toml::table& receipt, const std::string& where,
                                       const SpanFp8Context& fp8, bool old_gate) {
  if (old_gate)
    return stale(i18n::tr_format(
        "{}: prepared by an older player under an older quality and speed gate: prepare it again", where));
  const std::string_view arch = receipt["gpu_arch"].value_or(std::string_view());
  if (arch != fp8.gpu_arch)
    return stale(
        i18n::tr_format("{}: derived on {}, this GPU is {}: prepare it again", where, arch, fp8.gpu_arch));
  const auto kernels = receipt["kernels_version"].value_or(std::int64_t{0});
  if (kernels != fp8.kernels_version)
    return stale(i18n::tr_format("{}: kernels version {}, this engine has {}: prepare it again", where,
                                 kernels, fp8.kernels_version));
  return std::nullopt;
}

// Repris dans « Ready: … » et les refus du menu : dans la langue de l'interface (C07-09).
std::string variant_scores(const SpanInt8& variant) {
  return i18n::tr_format(
      "mean PSNR {} dB, worst {} dB, worst SSIM {}", std::format("{:.1f}", variant.psnr_mean_db),
      std::format("{:.1f}", variant.psnr_worst_db), std::format("{:.4f}", variant.ssim_worst));
}

// Étape 3.7b : un paquet v2 ne vaut que pour l'appareil, les noyaux et les tailles où la porte l'a jugé ;
// rien quand il vaut pour fp8.
std::optional<SpanInt8Lookup> size_judgement(const toml::table& receipt, const std::string& where,
                                             const SpanFp8Context& fp8, bool old_gate,
                                             const SpanInt8& variant) {
  if (auto old = outdated(receipt, where, fp8, old_gate)) return *old;
  const std::string size = gpu::span_fp8::size_key(fp8.width, fp8.height);
  const auto listed = [&](std::string_view key) {
    if (const toml::array* sizes = receipt[key].as_array())
      for (const toml::node& entry : *sizes)
        if (entry.value_or(std::string_view()) == size) return true;
    return false;
  };
  if (listed("refused_sizes")) {
    const std::string note = i18n::tr_format("{}: rejected at {}", where, size);
    if (listed(gate_refusals_key())) {
      SpanInt8Lookup out = refused(note);
      out.unjudged = variant;
      return out;
    }
    // Refusé par un lecteur plus ancien : la porte d'aujourd'hui rejuge cette taille.
  }
  if (!listed("validated_sizes") || listed("refused_sizes")) {
    SpanInt8Lookup pending = refused(i18n::tr_format("{}: not validated at {} yet", where, size));
    pending.unvalidated_size = size;
    pending.unjudged = variant;
    return pending;
  }
  return std::nullopt;
}

}  // namespace

bool span_fp8_cutlass_model(const std::filesystem::path& span) {
  return span.stem().u8string().find(u8"ArtCNN") == std::u8string::npos;
}

TierSupport span_tiers(const std::filesystem::path& span, const std::filesystem::path& dir, bool fp8_engine,
                       bool cutlass_engine) {
  // Le moteur qui sait les noyaux CUTLASS suffit : le lecteur fabrique lui-même le paquet au premier
  // démarrage avec ce cran (span_auto.h), et la lecture reste en FP16 d'ici là. Le b5f48 (48 canaux, 5 blocs)
  // est pris depuis l'étape 3.2c, rangé en 64 canaux ; ArtCNN jamais (span_fp8_cutlass_model).
  const bool cutlass_possible = cutlass_engine && span_fp8_cutlass_model(span);
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
  const std::expected<toml::table, SpanInt8Lookup> read = read_receipt(receipt_path, where);
  if (!read) return read.error();
  const toml::table& receipt = *read;
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
  const std::string scores = variant_scores(variant);
  // Un refus d'une porte plus ancienne, ou d'une autre carte ou d'autres noyaux, se refait (stale).
  const bool old_gate =
      v2 && fp8 != nullptr && receipt["gate_version"].value_or(std::int64_t{1}) < gpu::span_fp8::kGateVersion;
  const bool scores_pass = variant.psnr_mean_db >= limits.min_psnr_mean &&
                           variant.psnr_worst_db >= limits.min_psnr_worst &&
                           variant.ssim_worst >= limits.min_ssim_worst;
  if (!receipt["passed"].value_or(false) || !scores_pass) {
    const std::string note = refusal_note(receipt, where, limits, scores, scores_pass, v2);
    return old_gate ? stale(note) : refused(note);
  }
  // C46-06, C46-17 : un reçu qui n'est pas v2 (INT8, FP8 par Q/DQ, FP8 CUTLASS de l'outil Python) n'a que son
  // gain mesuré pour garantir qu'il vaut mieux que le FP16. Sans gain (--skip-speed-check, ou pas de trtexec
  // : speedup = 0) ou sous le seuil, la variante est refusée. Le FP32 (cran Qualité) ne promet aucune
  // vitesse.
  if (!v2 && kind != SpanVariantKind::Fp32 && receipt["speedup"].value_or(0.0) < gpu::span_fp8::kMinSpeedup) {
    const double speedup = receipt["speedup"].value_or(0.0);
    if (speedup > 0.0)
      return refused(
          i18n::tr_format("{}: no faster than TensorRT FP16 ({:.2f}× as fast, {:.2f}× is required)", where,
                          speedup, gpu::span_fp8::kMinSpeedup));
    return refused(
        i18n::tr_format("{}: gain over FP16 not measured, prepare it again with the speed check", where));
  }
  if (auto differs =
          package_differs(variant.onnx, receipt["model_sha256"].value_or(std::string_view()), where))
    return *differs;
  if (v2 && fp8 != nullptr) {
    if (auto judged = size_judgement(receipt, where, *fp8, old_gate, variant)) return *judged;
  }
  return SpanInt8Lookup{.variant = variant,
                        .note = std::format("{} ({})", prepare::utf8_path(onnx.filename()), scores)};
}

bool hdr_source(const media::TrackInfo& video) {
  const std::optional<ColorInfo> color = media::track_color(video);
  return color && media::yuv::is_hdr(*color);
}

std::vector<SpanVariantKind> span_variant_order(PrecisionTier tier, const SpanVariantCaps& caps,
                                                bool strict) {
  std::vector<SpanVariantKind> order;
  if (tier == PrecisionTier::Performance && caps.cutlass) order.push_back(SpanVariantKind::Fp8Cutlass);
  if (tier == PrecisionTier::Performance && caps.fp8 && !strict) order.push_back(SpanVariantKind::Fp8);
  // L'INT8 au cran Performance seulement : l'Équilibre, annoncé FP16, joue le FP16 même quand un reçu INT8
  // traîne d'un essai du cran Performance (A49-07).
  if (tier == PrecisionTier::Performance && caps.int8) order.push_back(SpanVariantKind::Int8);
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

std::string gate_refusals_key() {
  return std::format("refused_sizes_gate{}", gpu::span_fp8::kGateVersion);
}

std::string with_gate_refusals(std::string_view receipt, const std::vector<std::string>& refused) {
  const std::string key = gate_refusals_key();
  const std::string head = key + " =";
  std::string out;
  std::size_t at = 0;
  while (at < receipt.size()) {
    std::size_t end = receipt.find('\n', at);
    if (end == std::string_view::npos) end = receipt.size();
    const std::string_view line = receipt.substr(at, end - at);
    if (!line.starts_with(head)) {
      out += line;
      out += '\n';
    }
    at = end + 1;
  }
  return out + gpu::span_fp8::sizes_line(key, refused) + "\n";
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
