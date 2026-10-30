#include "render/looks.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>

namespace ajp::render {
namespace {

using Rgb = std::array<float, 3>;

struct Recipe {
  float contrast = 0.0F;     // courbe en S autour du gris moyen (0 : aucune)
  float saturation = 1.0F;   // facteur sur l'écart à la luminance
  float lift = 0.0F;         // noirs relevés (pastel) ou, négatif, creusés
  float gain = 1.0F;         // multiplicateur des hautes lumières
  Rgb shadow_tint{0, 0, 0};  // ajouté dans les ombres
  Rgb light_tint{0, 0, 0};   // ajouté dans les hautes lumières
  float cool_boost = 0.0F;   // saturation en plus des bleus et cyans (ciels, eau)
  float warm_boost = 0.0F;   // saturation en plus des rouges et oranges
};

constexpr std::array<BuiltinLook, 4> kLooks{{
    {.key = "maison:vif", .label = "Vivid"},
    {.key = "maison:shinkai", .label = "Cool sky"},
    {.key = "maison:pastel", .label = "Warm pastel"},
    {.key = "maison:aplats", .label = "Crisp flats"},
}};

Recipe recipe_of(std::string_view key) {
  Recipe r;
  if (key == "maison:vif") {  // couleurs et contraste poussés, hautes lumières légèrement relevées
    r.contrast = 0.40F;
    r.saturation = 1.35F;
    r.gain = 1.04F;
    r.warm_boost = 0.10F;
  } else if (key == "maison:shinkai") {  // ombres froides, ciels et eaux saturés, lumière claire
    r.contrast = 0.28F;
    r.saturation = 1.18F;
    r.gain = 1.05F;
    r.shadow_tint = {-0.015F, 0.005F, 0.04F};
    r.light_tint = {-0.005F, 0.008F, 0.02F};
    r.cool_boost = 0.45F;
  } else if (key == "maison:pastel") {  // noirs relevés, tons chauds et doux
    r.contrast = 0.10F;
    r.saturation = 1.08F;
    r.lift = 0.035F;
    r.gain = 0.98F;
    r.shadow_tint = {0.012F, 0.004F, -0.004F};
    r.light_tint = {0.02F, 0.01F, -0.015F};
    r.warm_boost = 0.18F;
  } else {  // « aplats » : contraste fort, noirs creusés, couleurs d'aplat franches
    r.contrast = 0.55F;
    r.saturation = 1.28F;
    r.lift = -0.012F;
    r.gain = 1.02F;
    r.warm_boost = 0.08F;
    r.cool_boost = 0.12F;
  }
  return r;
}

float s_curve(float x, float amount) {
  const float t = std::clamp(x, 0.0F, 1.0F);
  const float smooth = t * t * (3.0F - 2.0F * t);
  return t + (smooth - t) * amount;
}

Rgb grade(Rgb c, const Recipe& r) {
  for (float& v : c) v = std::clamp(v * r.gain + r.lift * (1.0F - v), 0.0F, 1.0F);
  const float luma = 0.2126F * c[0] + 0.7152F * c[1] + 0.0722F * c[2];
  // Teintes : les ombres reçoivent shadow_tint, les hautes lumières light_tint.
  const float dark = (1.0F - luma) * (1.0F - luma);
  const float bright = luma * luma;
  for (std::size_t i = 0; i < 3; ++i) c[i] += r.shadow_tint[i] * dark + r.light_tint[i] * bright;
  // Saturation, plus forte sur les bleus/cyans et sur les rouges/oranges selon la recette.
  const float cool = std::clamp((c[2] - 0.5F * (c[0] + c[1])) * 3.0F, 0.0F, 1.0F);
  const float warm = std::clamp((c[0] - 0.5F * (c[1] + c[2])) * 3.0F, 0.0F, 1.0F);
  const float sat = r.saturation + r.cool_boost * cool + r.warm_boost * warm;
  const float l2 = 0.2126F * c[0] + 0.7152F * c[1] + 0.0722F * c[2];
  for (float& v : c) v = l2 + (v - l2) * sat;
  for (float& v : c) v = s_curve(v, r.contrast);
  for (float& v : c) v = std::clamp(v, 0.0F, 1.0F);
  return c;
}

}  // namespace

std::span<const BuiltinLook> builtin_looks() {
  return kLooks;
}

std::expected<CubeLut, Refusal> builtin_look(std::string_view key, int size) {
  const bool known = std::ranges::any_of(kLooks, [&](const BuiltinLook& look) { return look.key == key; });
  if (!known) return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look maison inconnu"});
  size = std::clamp(size, kLutMinSize, kLutMaxSize);
  const Recipe recipe = recipe_of(key);
  CubeLut lut = identity_lut(size);
  lut.title = std::string(key);
  for (std::size_t i = 0; i < lut.rgb.size(); i += 3) {
    const Rgb out = grade({lut.rgb[i], lut.rgb[i + 1], lut.rgb[i + 2]}, recipe);
    lut.rgb[i] = out[0];
    lut.rgb[i + 1] = out[1];
    lut.rgb[i + 2] = out[2];
  }
  return lut;
}

bool look_is_pq(std::string_view name, std::string_view title) {
  const auto has = [](std::string_view text, std::string_view word) {
    std::string lowered(text);
    std::ranges::transform(lowered, lowered.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find(word) != std::string::npos;
  };
  for (const std::string_view text : {name, title})
    if (has(text, "pq") || has(text, "hdr") || has(text, "2020")) return true;
  return false;
}

}  // namespace ajp::render
