#include "app/play/profiles.h"

#include <cstdint>
#include <format>
#include <utility>

#include "core/cadence.h"
#include "core/log.h"
#include "core/user_overlay.h"

namespace ajp::app::play {
namespace {

// Modèles des profils livrés : SPAN Sharp1 Performance (le défaut), Sharp1 Balanced (plus fin) ;
// @PERF@ et @BAL@ dans le texte ci-dessous.
constexpr std::string_view kSpanPerformance = kSpanModelId;
constexpr std::string_view kSpanBalanced = "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16";

constexpr std::string_view kBaseTemplate = R"toml(version = 5
start_profile = 1

[[profile]]
slot = 1
name = "Qualité 1440p"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "downscale", filter = "area", height = 720 },
  { kind = "interpolate", model = "rife_v4.25", factor = 3, policy = "quality" },
  { kind = "upscale", model = "@PERF@", factor = 2 },
]

[[profile]]
slot = 2
name = "Fluide 1440p ×4"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "downscale", filter = "area", height = 720 },
  { kind = "interpolate", model = "rife_v4.25", factor = 4, policy = "quality" },
  { kind = "upscale", model = "@PERF@", factor = 2 },
]

[[profile]]
slot = 3
name = "4K"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "interpolate", model = "rife_v4.25", factor = 2, policy = "quality" },
  { kind = "upscale", model = "@PERF@", factor = 2 },
]

[[profile]]
slot = 4
name = "4K sans interpolation"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "upscale", model = "@PERF@", factor = 2 },
]

[[profile]]
slot = 5
name = "1440p SDR"
hdr = "off"
[[profile.rule]]
steps = [
  { kind = "downscale", filter = "area", height = 720 },
  { kind = "interpolate", model = "rife_v4.25", factor = 3, policy = "quality" },
  { kind = "upscale", model = "@PERF@", factor = 2 },
]

[[profile]]
slot = 6
name = "Net 1440p (Balanced)"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "downscale", filter = "area", height = 720 },
  { kind = "interpolate", model = "rife_v4.25", factor = 3, policy = "quality" },
  { kind = "upscale", model = "@BAL@", factor = 2 },
]

[[profile]]
slot = 7
name = "Léger 1080p"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "downscale", filter = "area", height = 540 },
  { kind = "interpolate", model = "rife_v4.25", factor = 2, policy = "quality" },
  { kind = "upscale", model = "@PERF@", factor = 2 },
]

[[profile]]
slot = 8
name = "RIFE seul"
hdr = "truehdr"
[[profile.rule]]
steps = [
  { kind = "interpolate", model = "rife_v4.25", factor = 3, policy = "quality" },
]

[[profile]]
slot = 9
name = "Source"
hdr = "off"
[[profile.rule]]
steps = []
)toml";

std::string with_models(std::string_view text) {
  std::string out(text);
  for (const auto& [token, model] : {std::pair{std::string_view("@PERF@"), kSpanPerformance},
                                     std::pair{std::string_view("@BAL@"), kSpanBalanced}}) {
    for (std::size_t at = out.find(token); at != std::string::npos; at = out.find(token, at + model.size()))
      out.replace(at, token.size(), model);
  }
  return out;
}

std::string_view filter_label(core::DownscaleFilter filter) {
  return label(filter);
}

}  // namespace

std::string_view base_profiles_toml() {
  static const std::string base = with_models(kBaseTemplate);  // écrit une fois, lu ensuite
  return base;
}

PlaySettings apply_profile(const core::Profile& profile, const core::SourceInfo& source,
                           PlaySettings settings, std::vector<std::string>& notes) {
  settings.upscale = false;
  settings.work_height = 0;
  settings.rife_factor = 0;
  settings.rife_factor_den = 1;
  settings.true_hdr = profile.hdr == core::HdrMode::TrueHdr;
  settings.hdr_native = profile.hdr == core::HdrMode::Native;
  const core::Rule* rule = core::match_rule(profile, source);
  if (rule == nullptr) {
    notes.emplace_back("aucune règle pour cette source : image telle quelle");
    return settings;
  }
  bool upscaled = false;
  for (const core::Step& step : rule->steps) {
    switch (step.kind) {
      case core::StepKind::Downscale:
        settings.work_height = step.height;
        settings.downscale = step.filter;
        break;
      case core::StepKind::Interpolate: {
        const core::Ratio ratio = core::reduced(step.factor);
        const Factor factor{.num = static_cast<int>(ratio.num), .den = static_cast<int>(ratio.den)};
        if (ratio.num > std::int64_t{2} * limits::kRifeFactor.hi || ratio.num <= ratio.den ||
            !valid_factor(factor)) {
          notes.push_back(
              std::format("interpolation ×{}/{} : seuls ×1,5 et les facteurs entiers 2 à {} sont faits",
                          step.factor.num, step.factor.den, limits::kRifeFactor.hi));
          break;
        }
        if (upscaled) notes.emplace_back("interpolation après l'agrandissement : faite avant");
        if (settings.rife_factor >= 2) {
          notes.emplace_back("plusieurs interpolations : la dernière vaut");
        }
        settings.rife_factor = factor.num;
        settings.rife_factor_den = factor.den;
        // Auto pour tout facteur entier (le multiplicateur du profil en est le plafond d'origine) ; une
        // moitié (×1,5) n'a pas de mode auto.
        settings.rife_auto = factor.den == 1;
        break;
      }
      case core::StepKind::Upscale:
        if (step.factor.num != kSpanFactor) {
          notes.push_back(std::format("agrandissement ×{} : SPAN fait ×{}", step.factor.num, kSpanFactor));
        }
        if (upscaled) {
          notes.emplace_back("un seul agrandissement par source");
          break;
        }
        settings.upscale = true;
        settings.model = step.model + ".onnx";
        upscaled = true;
        break;
    }
  }
  return settings;
}

std::vector<core::Rule> rules_of(const PlaySettings& s, std::string_view default_model) {
  core::Rule rule{.max_pixels = core::limits::kPixels.hi, .max_fps = core::limits::kFps.hi};
  if (s.work_height > 0) {
    rule.steps.push_back(
        core::Step{.kind = core::StepKind::Downscale, .filter = s.downscale, .height = s.work_height});
  }
  if (s.rife_factor >= 2) {
    rule.steps.push_back(core::Step{.kind = core::StepKind::Interpolate,
                                    .model = std::string(kRifeModelId),
                                    .factor = {.num = s.rife_factor, .den = s.rife_factor_den},
                                    .policy = core::Policy::Quality});
  }
  if (s.upscale) {
    std::string model = s.model.empty() ? std::string(default_model) : s.model;
    if (model.ends_with(".onnx")) model.resize(model.size() - 5);
    rule.steps.push_back(core::Step{.kind = core::StepKind::Upscale,
                                    .model = std::move(model),
                                    .factor = {.num = kSpanFactor, .den = 1}});
  }
  return {std::move(rule)};
}

core::HdrMode hdr_of(const PlaySettings& s) {
  if (s.true_hdr) return core::HdrMode::TrueHdr;
  return s.hdr_native ? core::HdrMode::Native : core::HdrMode::Off;
}

std::string profile_summary(const core::Profile& profile, const core::SourceInfo& source) {
  std::vector<std::string> notes;
  const PlaySettings s = apply_profile(profile, source, PlaySettings{}, notes);
  std::string text;
  const auto add = [&](const std::string& part) { text += text.empty() ? part : " → " + part; };
  if (s.work_height > 0 && s.work_height < source.height)
    add(std::format("{}p {}", s.work_height, filter_label(s.downscale)));
  if (s.rife_factor >= 2) add(std::format("RIFE {}", factor_text(factor_of(s))));
  if (s.upscale) {
    const int height =
        (s.work_height > 0 && s.work_height < source.height ? s.work_height : source.height) * kSpanFactor;
    const bool usual = s.model.empty() || s.model == std::string(kSpanPerformance) + ".onnx";
    add(usual ? std::format("SPAN {}p", height) : std::format("SPAN {}p ({})", height, model_label(s.model)));
  }
  if (text.empty()) text = "source telle quelle";
  if (s.true_hdr) text += " · RTX HDR";
  if (s.hdr_native) text += s.grade ? " · HDR adaptatif" : " · HDR sans grade (expansion simple)";
  return text;
}

namespace {
core::KeyMap& key_slot() {
  static core::KeyMap keys = core::KeyMap::defaults();
  return keys;
}
}  // namespace

void apply_user_keys(std::string_view user_toml) {
  auto parsed = core::parse_keys(user_toml);
  if (!parsed) {
    log_channel("app")->warn("user.toml [keys] : {} {} ; raccourcis par défaut", name(parsed.error().code),
                             parsed.error().detail);
    key_slot() = core::KeyMap::defaults();
    return;
  }
  key_slot() = *parsed;
}

const core::KeyMap& configured_keys() {
  return key_slot();
}

std::expected<Profiles, Refusal> load_profiles(const std::filesystem::path& user_toml) {
  auto text = core::read_text_file_if_present(user_toml);
  if (!text) return std::unexpected(std::move(text.error()));
  const std::string user = text->value_or(std::string());
  auto layered = core::parse_layered(base_profiles_toml(), user);
  if (!layered) return std::unexpected(std::move(layered.error()));
  return Profiles{.file = std::move(layered->profile), .user_text = user};
}

std::expected<Profiles, Refusal> edit_profiles(const std::filesystem::path& user_toml,
                                               const Profiles& current,
                                               std::span<const core::UserEdit> edits) {
  auto text = core::apply_edits(base_profiles_toml(), current.user_text, edits);
  if (!text) return std::unexpected(std::move(text.error()));
  auto layered = core::parse_layered(base_profiles_toml(), *text);
  if (!layered) return std::unexpected(std::move(layered.error()));
  if (auto written = core::write_text_atomically(user_toml, *text); !written)
    return std::unexpected(std::move(written.error()));
  return Profiles{.file = std::move(layered->profile), .user_text = std::move(*text)};
}

}  // namespace ajp::app::play
