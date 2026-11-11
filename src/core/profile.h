// Ce qui reste du profil TOML v5, retiré du lecteur (ADR 0012) : les types et les plages que les réglages
// (play.toml), la cadence et les modèles partagent encore.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::core {

// Filtres de la réduction : Area et Spline36 (ceux d'AnimeJaNai), puis ceux du menu du lecteur.
enum class DownscaleFilter : std::uint8_t {
  Area,
  Spline36,
  Bilinear,
  Bicubic,
  Mitchell,
  CatmullRom,
  Hermite
};
// « truehdr » des anciens profils (RTX HDR, retiré) se lit comme Native.
enum class HdrMode : std::uint8_t { Off, Native };

struct Ratio {
  std::int64_t num = 1;
  std::int64_t den = 1;
  double value() const { return static_cast<double>(num) / static_cast<double>(den); }
  friend bool operator==(const Ratio&, const Ratio&) = default;
};

// Plages de chaque champ. Une valeur hors plage est refusée, jamais corrigée.
template <class T>
struct Range {
  T lo;
  T hi;
  constexpr bool contains(T v) const { return lo <= v && v <= hi; }
};

namespace limits {
inline constexpr Range<std::int64_t> kPixels{0, 16384LL * 16384LL};
inline constexpr Range<double> kFps{0.0, 1000.0};
inline constexpr Range<std::int64_t> kRatioTerm{1, 100000};
inline constexpr Range<double> kInterpFactor{1.0, 8.0};  // borne basse exclue
}  // namespace limits

}  // namespace ajp::core
