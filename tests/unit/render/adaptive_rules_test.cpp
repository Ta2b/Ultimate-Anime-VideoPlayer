// Les règles du grade adaptatif sont les mêmes des deux côtés : les valeurs par défaut des paramètres ar_*
// du hook (1-params.hook) valent les constantes de render/adaptive_rules.h, et GradeParams les porte.
#include "render/adaptive_rules.h"

#include <string>
#include <string_view>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "render/grade_params.h"
#include "render/shaders.h"

namespace rules = ajp::render::adaptive_rules;
using Catch::Approx;

namespace {

// La valeur par défaut d'un //!PARAM : la première ligne qui ne commence pas par //! après son nom.
double param_default(std::string_view hook, std::string_view name) {
  const std::string tag = "//!PARAM " + std::string(name) + "\n";
  std::size_t at = hook.find(tag);
  if (at == std::string_view::npos) return -1.0;
  at += tag.size();
  while (hook.substr(at, 3) == "//!") at = hook.find('\n', at) + 1;
  return std::stod(std::string(hook.substr(at, hook.find('\n', at) - at)));
}

}  // namespace

TEST_CASE("règles adaptatives : le shader et le lecteur partagent les mêmes seuils", "[render][grade]") {
  const ajp::render::GradeParams p;
  const std::pair<std::string_view, std::pair<double, double>> cases[] = {
      {"ar_series_ref", {rules::kSeriesRef, p.ar_series_ref}},
      {"ar_ratio_min", {rules::kRatioMin, p.ar_ratio_min}},
      {"ar_ratio_max", {rules::kRatioMax, p.ar_ratio_max}},
      {"ar_bright_cap", {rules::kBrightCap, p.ar_bright_cap}},
      {"ar_blue_lo", {rules::kBlueLo, p.ar_blue_lo}},
      {"ar_blue_hi", {rules::kBlueHi, p.ar_blue_hi}},
      {"ar_blue_gate", {rules::kBlueGate, p.ar_blue_gate}},
      {"ar_chroma_lo", {rules::kChromaLo, p.ar_chroma_lo}},
      {"ar_chroma_hi", {rules::kChromaHi, p.ar_chroma_hi}},
      {"ar_series_lo", {rules::kBrightSeriesLo, p.ar_series_lo}},
      {"ar_series_hi", {rules::kBrightSeriesHi, p.ar_series_hi}},
      {"ar_glow_lo", {rules::kGlowShareLo, p.ar_glow_lo}},
      {"ar_glow_hi", {rules::kGlowShareHi, p.ar_glow_hi}},
      {"ar_scene_lo", {rules::kGlowSceneLo, p.ar_scene_lo}},
      {"ar_scene_hi", {rules::kGlowSceneHi, p.ar_scene_hi}},
  };
  for (const auto& [name, values] : cases) {
    INFO(name);
    CHECK(param_default(ajp::render::shaders::anime_hdr_grade, name) == Approx(values.first));
    CHECK(values.second == values.first);
  }
}
