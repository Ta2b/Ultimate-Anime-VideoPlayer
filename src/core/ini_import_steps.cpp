// Étapes des chaînes INI vers les étapes v5.
// Repris, pour la sémantique seulement (le code est réécrit) :
//   animejanai-inference-rtx@118c272 src/aji_conf.cpp:120-205 : model_M_*, rife_* ;
//   mpv-animejanai-hdr AnimeJaNaiPlatform/PlatformServices.cs:345-452 (Migrate, Normalize),
//     :479-540 (validation des étapes), :604-609 (RifeFile) ;
//   manager-v2 ViewModels/InterpolationStages.cs:97-101 (ModelScale).
// Comportement fixé par tests/unit/core/ini_import_test.cpp.
#include <algorithm>
#include <format>
#include <numeric>
#include <set>

#include "core/ini_import_internal.h"

namespace ajp::core::ini {
namespace {

// « 2x_AnimeJaNai… » agrandit par 2 ; sans préfixe, par 2 (ModelScale du gestionnaire).
std::expected<Step, Refusal> upscale_step(const Keys& keys, std::string_view rest, const std::string& model) {
  std::int64_t factor = 2;
  const auto digits = static_cast<std::size_t>(
      std::ranges::find_if(model, [](char c) { return c < '0' || c > '9'; }) - model.begin());
  if (digits > 0 && digits + 1 < model.size() && (model[digits] == 'x' || model[digits] == 'X') &&
      (model[digits + 1] == '_' || model[digits + 1] == '-')) {
    factor = std::max<std::int64_t>(parse_integer(std::string_view(model).substr(0, digits)).value_or(2), 1);
  }
  if (!limits::kUpscaleFactor.contains(factor)) {
    return keys.bad(RefusalCode::FieldOutOfRange, rest,
                    std::format("agrandissement x{} hors de 1 à 8", factor));
  }
  return Step{.kind = StepKind::Upscale, .model = model, .factor = {factor, 1}};
}

std::expected<Step, Refusal> downscale_step(const Keys& keys, std::string_view rest, double height,
                                            DownscaleFilter filter) {
  const auto rounded = static_cast<std::int64_t>(height);
  if (static_cast<double>(rounded) != height || !limits::kDownscaleHeight.contains(rounded)) {
    return keys.bad(RefusalCode::FieldOutOfRange, rest, "hauteur entière de 144 à 4320 attendue");
  }
  return Step{.kind = StepKind::Downscale, .filter = filter, .height = static_cast<int>(rounded)};
}

// Code du gestionnaire vers ModelId : 414 → rife-4.14, 4141 → rife-4.14-lite, 46 → rife-4.6.
std::optional<std::string> rife_model(std::string_view code, bool ensemble) {
  if (code.size() < 2 || code.size() > 4 || !parse_integer(code)) return std::nullopt;
  std::string id = std::format("rife-{}.{}", code[0], code.substr(1, code.size() == 2 ? 1 : 2));
  if (code.size() == 4 && code.back() == '1') id += "-lite";
  if (ensemble) id += "-ensemble";
  return id;
}

// Facteur d'interpolation réduit ; std::nullopt pour 1, qui ne change rien.
std::expected<std::optional<Ratio>, Refusal> interp_factor(const Keys& keys, std::string_view num_key,
                                                           std::string_view den_key) {
  auto num = keys.integer(num_key, 1);
  if (!num) return std::unexpected(num.error());
  auto den = keys.integer(den_key, 1);
  if (!den) return std::unexpected(den.error());
  if (*num < 1 || *den < 1)
    return keys.bad(RefusalCode::FieldOutOfRange, *num < 1 ? num_key : den_key, "≥ 1 attendu");
  const std::int64_t gcd = std::gcd(*num, *den);
  const Ratio ratio{*num / gcd, *den / gcd};
  if (ratio.num == ratio.den) return std::optional<Ratio>{};
  const double value = ratio.value();
  if (value < 1.0 || value > limits::kInterpFactor.hi || !limits::kRatioTerm.contains(ratio.num) ||
      !limits::kRatioTerm.contains(ratio.den)) {
    return keys.bad(RefusalCode::FieldOutOfRange, num_key, "facteur d'interpolation hors de ]1, 8]");
  }
  return std::optional<Ratio>{ratio};
}

// Fast avant l'agrandissement suit la timeline des dessins, comme aji_rife_drawings du fork.
Pacing pacing_for(const Importer& im, Policy policy, bool before_upscale) {
  return policy == Policy::Fast && before_upscale && im.drawings ? Pacing::Smooth : Pacing::Source;
}

std::expected<Step, Refusal> rife_step(const Keys& keys, std::string_view code_key, std::string_view fallback,
                                       bool ensemble, Ratio factor, Policy policy, Pacing pacing) {
  const std::optional<std::string> model = rife_model(keys.text(code_key, fallback), ensemble);
  if (!model) return keys.bad(RefusalCode::FieldType, code_key, "code RIFE de 2 à 4 chiffres attendu");
  return Step{
      .kind = StepKind::Interpolate, .model = *model, .factor = factor, .policy = policy, .pacing = pacing};
}

std::expected<void, Refusal> unit_factor(const Keys& stage) {
  for (const std::string_view key : {"factor_numerator", "factor_denominator"}) {
    auto value = stage.integer(key, 1);
    if (!value) return std::unexpected(value.error());
    if (*value != 1)
      return stage.bad(RefusalCode::FieldOutOfRange, key, "seule l'interpolation change la cadence");
  }
  return {};
}

using StageResult = std::expected<std::optional<Step>, Refusal>;

StageResult resize_stage(const Keys& stage, Importer& im) {
  if (auto r = unit_factor(stage); !r) return std::unexpected(r.error());
  stage.find("legacy_geometry");
  const std::string method = lower(stage.text("method"));
  DownscaleFilter filter = DownscaleFilter::Spline36;
  if (method == "area") filter = DownscaleFilter::Area;
  if (method == "bilinear") im.note(stage.line_of("method"), "réduction bilinéaire remplacée par Spline36");
  if (method != "area" && method != "spline36" && method != "bilinear") {
    return stage.bad(RefusalCode::ImportUnsupported, "method", "réduction inconnue");
  }
  if ((stage.find("target_height") == nullptr) == (stage.find("target_percent") == nullptr)) {
    return stage.bad(RefusalCode::FieldMissing, "kind", "une réduction veut target_height ou target_percent");
  }
  if (stage.find("target_percent") != nullptr) {
    auto percent = stage.real("target_percent", 100.0);
    if (!percent) return std::unexpected(percent.error());
    if (*percent != 100.0) {
      return stage.bad(RefusalCode::ImportUnsupported, "target_percent",
                       "réduction en pourcentage sans équivalent en v5");
    }
    im.note(stage.line_of("target_percent"), "réduction à 100 % : sans effet, retirée");
    return std::optional<Step>{};
  }
  auto height = stage.real("target_height", 0.0);
  if (!height) return std::unexpected(height.error());
  auto step = downscale_step(stage, "target_height", *height, filter);
  if (!step) return std::unexpected(step.error());
  return std::optional<Step>{*step};
}

StageResult interpolate_stage(const Keys& stage, Importer& im, bool upscaled) {
  const std::string method = lower(stage.text("method"));
  if (method != "rife") {
    return stage.bad(RefusalCode::ImportUnsupported, "method",
                     method == "nvidia-fruc" ? "FRUC a été retiré" : "interpolation inconnue");
  }
  const std::string policy_name = lower(stage.text("policy", "quality"));
  if (policy_name != "quality" && policy_name != "fast") {
    return stage.bad(RefusalCode::FieldOutOfRange, "policy", "quality ou fast attendu");
  }
  const Policy policy = policy_name == "fast" ? Policy::Fast : Policy::Quality;
  auto ensemble = stage.flag("ensemble", false);
  if (!ensemble) return std::unexpected(ensemble.error());
  auto factor = interp_factor(stage, "factor_numerator", "factor_denominator");
  if (!factor) return std::unexpected(factor.error());
  if (!*factor) {
    im.note(stage.line_of("kind"), "interpolation de facteur 1 : sans effet, retirée");
    return std::optional<Step>{};
  }
  auto step = rife_step(stage, "model", "", *ensemble, **factor, policy, pacing_for(im, policy, !upscaled));
  if (!step) return std::unexpected(step.error());
  return std::optional<Step>{*step};
}

// Une étape ordonnée ; std::nullopt quand elle ne produit pas d'étape v5 (HDR, réduction à 100 %).
StageResult stage_step(const Keys& stage, Importer& im, bool upscaled, std::vector<HdrMode>& hdr) {
  const std::string kind = lower(stage.text("kind"));
  stage.find("id");
  stage.find("runtime");  // le moteur se choisit dans les réglages, pas dans le profil
  if (kind == "resize") return resize_stage(stage, im);
  if (kind == "interpolate") return interpolate_stage(stage, im, upscaled);
  const std::string method = lower(stage.text("method"));
  if (kind == "upscale") {
    if (method != "onnx")
      return stage.bad(RefusalCode::ImportUnsupported, "method", "agrandissement inconnu");
    if (auto r = unit_factor(stage); !r) return std::unexpected(r.error());
    const std::string model = stage.text("model");
    if (model.empty())
      return stage.bad(RefusalCode::FieldMissing, "model", "modèle d'agrandissement attendu");
    auto step = upscale_step(stage, "model", model);
    if (!step) return std::unexpected(step.error());
    return std::optional<Step>{*step};
  }
  if (kind == "hdr") {
    if (method != "mpv" && method != "nvidia-truehdr") {
      return stage.bad(RefusalCode::ImportUnsupported, "method", "conversion HDR inconnue");
    }
    hdr.push_back(method == "mpv" ? HdrMode::Native : HdrMode::TrueHdr);
    im.note(stage.line_of("kind"),
            "l'étape HDR devient le mode HDR du profil ; son réglage d'aspect se refait dans le grade");
    stage.consume("");
    return std::optional<Step>{};
  }
  return stage.bad(RefusalCode::ImportUnsupported, "kind", "étape inconnue");
}

// Réduction avant l'agrandissement d'un modèle historique ; std::nullopt sans réduction.
std::expected<std::optional<Step>, Refusal> model_downscale(const Keys& model, Importer& im, bool first) {
  auto height = model.real("resize_height_before_upscale", 0.0);
  if (!height) return std::unexpected(height.error());
  auto percent = model.real("resize_factor_before_upscale", 100.0);
  if (!percent) return std::unexpected(percent.error());
  if (*height <= 0.0 && *percent == 100.0) return std::optional<Step>{};
  // La hauteur l'emporte sur le pourcentage, comme dans Normalize.
  const std::string_view key =
      *height > 0.0 ? "resize_height_before_upscale" : "resize_factor_before_upscale";
  if (!first) {
    return model.bad(RefusalCode::ImportUnsupported, key,
                     "seule la réduction du premier modèle a un équivalent en v5");
  }
  if (*height <= 0.0) {
    return model.bad(RefusalCode::ImportUnsupported, key, "réduction en pourcentage sans équivalent en v5");
  }
  auto step = downscale_step(model, key, *height, DownscaleFilter::Spline36);
  if (!step) return std::unexpected(step.error());
  if (im.backend != "tensorrt" && im.backend != "directml") {
    im.note(model.line_of(key),
            std::format("réduction bilinéaire du moteur {} remplacée par Spline36", im.backend));
  }
  return std::optional<Step>{*step};
}

}  // namespace

std::expected<std::vector<Step>, Refusal> legacy_steps(const Keys& chain, Importer& im) {
  std::set<std::int64_t> indices;
  const std::string models_prefix = chain.name("model_");
  for (const auto& [key, value] : *chain.section) {
    if (!key.starts_with(models_prefix) || !key.ends_with("_name")) continue;
    const std::string_view middle = std::string_view(key).substr(models_prefix.size());
    if (auto index = parse_integer(middle.substr(0, middle.size() - 5)); index && *index > 0)
      indices.insert(*index);
  }

  std::optional<Step> downscale;
  std::vector<Step> upscales;
  bool temporal = false;
  for (const std::int64_t index : indices) {
    const Keys model = chain.sub(std::format("model_{}_", index));
    const std::string name = model.text("name");
    if (name.empty()) continue;  // étape sans effet, comme dans le fork
    auto resize = model_downscale(model, im, upscales.empty());
    if (!resize) return std::unexpected(resize.error());
    if (*resize) downscale = **resize;
    const std::string frames = model.text("frames", "0");
    temporal |= frames != "0" && frames != "1" && !frames.empty();
    auto step = upscale_step(model, "name", name);
    if (!step) return std::unexpected(step.error());
    upscales.push_back(std::move(*step));
  }

  std::optional<Step> rife;
  bool before = true;
  auto enabled = chain.flag("rife", false);
  if (!enabled) return std::unexpected(enabled.error());
  if (*enabled) {
    auto ensemble = chain.flag("rife_ensemble", false);
    auto rife_before = chain.flag("rife_before_upscale", true);
    auto factor = interp_factor(chain, "rife_factor_numerator", "rife_factor_denominator");
    if (!ensemble || !rife_before || !factor) {
      return std::unexpected(!ensemble      ? ensemble.error()
                             : !rife_before ? rife_before.error()
                                            : factor.error());
    }
    before = *rife_before;
    if (before && temporal && !upscales.empty()) {
      before = false;
      im.note(chain.line_of("rife_before_upscale"),
              "modèle temporel : l'interpolation passe après l'agrandissement, comme dans le lecteur NVIDIA");
    }
    if (!*factor) {
      im.note(chain.line_of("rife"), "interpolation de facteur 1 : sans effet, retirée");
    } else {
      auto step = rife_step(chain, "rife_model", "414", *ensemble, **factor, im.policy,
                            pacing_for(im, im.policy, before));
      if (!step) return std::unexpected(step.error());
      rife = *step;
    }
  }

  // La réduction du premier modèle passe avant RIFE, comme dans le lecteur NVIDIA ; v5 la veut en tête.
  std::vector<Step> steps;
  if (downscale) steps.push_back(*downscale);
  if (rife && before) steps.push_back(*rife);
  steps.insert(steps.end(), upscales.begin(), upscales.end());
  if (rife && !before) steps.push_back(*rife);
  return steps;
}

std::expected<std::vector<Step>, Refusal> stage_steps(const Keys& chain, Importer& im,
                                                      std::vector<HdrMode>& hdr) {
  auto version = chain.integer("pipeline_version", 0);
  if (!version) return std::unexpected(version.error());
  if (*version != 1)
    return chain.bad(RefusalCode::FormatVersion, "pipeline_version", "seule la version 1 existe");
  if (chain.find("stage_count") == nullptr) {
    return refuse(RefusalCode::FieldMissing,
                  std::format("ligne {} : {}", chain.line_of("pipeline_version"), chain.name("stage_count")));
  }
  auto count = chain.integer("stage_count", 0);
  if (!count) return std::unexpected(count.error());
  if (*count > static_cast<std::int64_t>(limits::kMaxSteps)) {
    return chain.bad(RefusalCode::TooManySteps, "stage_count", "16 étapes au plus");
  }
  std::vector<Step> steps;
  bool upscaled = false;
  for (std::int64_t j = 1; j <= *count; ++j) {
    const Keys stage = chain.sub(std::format("stage_{}_", j));
    auto enabled = stage.flag("enabled", true);
    if (!enabled) return std::unexpected(enabled.error());
    if (!*enabled) {
      im.note(stage.line_of("kind"),
              std::format("étape {} désactivée : v5 n'a pas d'étape inactive, retirée", j));
      stage.consume("");
      continue;
    }
    auto step = stage_step(stage, im, upscaled, hdr);
    if (!step) return std::unexpected(step.error());
    if (!*step) continue;
    if ((*step)->kind == StepKind::Downscale && !steps.empty()) {
      return stage.bad(RefusalCode::ImportUnsupported, "kind",
                       "réduction après une autre étape : v5 la veut en tête");
    }
    upscaled |= (*step)->kind == StepKind::Upscale;
    steps.push_back(std::move(**step));
  }
  // Les clés historiques recopiées par le gestionnaire v4 ne sont qu'un miroir des étapes.
  chain.consume("rife");
  chain.consume("model_");
  return steps;
}

}  // namespace ajp::core::ini
