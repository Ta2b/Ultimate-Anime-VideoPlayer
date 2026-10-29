// Les trois crans de précision des réseaux (Performance, Équilibre, Qualité) : FP8, FP16, FP32.
// Un cran ne change que le fichier ONNX que la chaîne charge ; le fichier d'un cran absent, ou que
// le moteur refuse, laisse la place à celui d'Équilibre (FP16), qui est le réseau tel que livré.
#pragma once

#include <cstdint>
#include <string_view>

namespace ajp::app::play {

enum class PrecisionTier : std::uint8_t { Performance, Balanced, Quality };
inline constexpr int kPrecisionTierCount = 3;

constexpr std::string_view name(PrecisionTier tier) {  // la clé de play.toml
  switch (tier) {
    case PrecisionTier::Performance:
      return "performance";
    case PrecisionTier::Balanced:
      return "balanced";
    case PrecisionTier::Quality:
      return "quality";
  }
  return "balanced";
}

constexpr std::string_view label(PrecisionTier tier) {  // clé de traduction du menu (i18n::tr)
  switch (tier) {
    case PrecisionTier::Performance:
      return "Performance";
    case PrecisionTier::Balanced:
      return "Balance";
    case PrecisionTier::Quality:
      return "Quality";
  }
  return "Balance";
}

// La précision de calcul du cran : « FP8 », « FP16 », « FP32 ».
constexpr std::string_view precision_text(PrecisionTier tier) {
  switch (tier) {
    case PrecisionTier::Performance:
      return "FP8";
    case PrecisionTier::Balanced:
      return "FP16";
    case PrecisionTier::Quality:
      return "FP32";
  }
  return "FP16";
}

// Les crans que le poste peut rendre pour un réseau : Équilibre (FP16) l'est toujours ; Performance (FP8)
// et Qualité (FP32) demandent leur graphe et, pour le FP8, un moteur qui le calcule.
struct TierSupport {
  bool performance = false;
  bool quality = false;
  constexpr bool has(PrecisionTier tier) const {
    return tier == PrecisionTier::Balanced || (tier == PrecisionTier::Performance ? performance : quality);
  }
  // Le cran demandé s'il est rendu, Équilibre sinon.
  constexpr PrecisionTier effective(PrecisionTier tier) const {
    return has(tier) ? tier : PrecisionTier::Balanced;
  }
  friend constexpr bool operator==(const TierSupport&, const TierSupport&) = default;
};

}  // namespace ajp::app::play
