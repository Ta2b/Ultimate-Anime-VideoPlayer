// Profil TOML v5 : neuf profils, un profil de démarrage, des règles par
// nombre de pixels et fps. Lu en un seul endroit, profile.cpp.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::core {

enum class Policy : std::uint8_t { Quality, Fast, Refine };
enum class Pacing : std::uint8_t { Source, Smooth, KeepPoses };
enum class StepKind : std::uint8_t { Downscale, Upscale, Interpolate };
// Filtres de la réduction : Area et Spline36 (ceux d'AnimeJaNai), puis ceux du menu du lecteur.
enum class DownscaleFilter : std::uint8_t { Area, Spline36, Bilinear, Bicubic, Mitchell, CatmullRom, Hermite };
enum class HdrMode : std::uint8_t { Off, Native, TrueHdr };

using ModelId = std::string;

struct Ratio {
  std::int64_t num = 1;
  std::int64_t den = 1;
  double value() const { return static_cast<double>(num) / static_cast<double>(den); }
  friend bool operator==(const Ratio&, const Ratio&) = default;
};

struct Step {
  StepKind kind = StepKind::Upscale;
  ModelId model;                                   // Upscale, Interpolate
  Ratio factor;                                    // Upscale : entier ; Interpolate : rationnel
  Policy policy = Policy::Quality;                 // Interpolate
  Pacing pacing = Pacing::Source;                  // Interpolate
  DownscaleFilter filter = DownscaleFilter::Area;  // Downscale
  int height = 0;                                  // Downscale : hauteur cible
  friend bool operator==(const Step&, const Step&) = default;
};

// Bornes incluses. La première règle qui correspond gagne.
struct Rule {
  std::int64_t min_pixels = 0;
  std::int64_t max_pixels = 0;
  double min_fps = 0.0;
  double max_fps = 0.0;
  std::vector<Step> steps;
  friend bool operator==(const Rule&, const Rule&) = default;
};

struct Profile {
  std::string name;
  HdrMode hdr = HdrMode::Off;
  std::vector<Rule> rules;
  friend bool operator==(const Profile&, const Profile&) = default;
};

inline constexpr int kProfileVersion = 5;
inline constexpr std::size_t kProfileSlots = 9;

struct ProfileFile {
  int start_profile = 1;                                       // 1 à 9
  std::array<std::optional<Profile>, kProfileSlots> profiles;  // index = numéro - 1
  friend bool operator==(const ProfileFile&, const ProfileFile&) = default;
};

// Plages de chaque champ. Une valeur hors plage est refusée, jamais corrigée.
template <class T>
struct Range {
  T lo;
  T hi;
  constexpr bool contains(T v) const { return lo <= v && v <= hi; }
};

namespace limits {
inline constexpr Range<std::int64_t> kStartProfile{1, 9};
inline constexpr Range<std::int64_t> kPixels{0, 16384LL * 16384LL};
inline constexpr Range<double> kFps{0.0, 1000.0};
inline constexpr Range<std::int64_t> kUpscaleFactor{1, 8};
inline constexpr Range<std::int64_t> kRatioTerm{1, 100000};
inline constexpr Range<double> kInterpFactor{1.0, 8.0};  // borne basse exclue
inline constexpr Range<std::int64_t> kDownscaleHeight{144, 4320};
inline constexpr std::size_t kMaxSteps = 16;
inline constexpr double kMaxFactorProduct = 16.0;
}  // namespace limits

std::expected<ProfileFile, Refusal> parse_profile(std::string_view toml_text);
std::expected<ProfileFile, Refusal> load_profile(const std::filesystem::path& path);

// Texte TOML v5 de file, que parse_profile relit à l'identique. Chaque profil
// écrit son nom et son mode HDR ; les bornes et le rythme par défaut sont omis.
std::string format_profile(const ProfileFile& file);

}  // namespace ajp::core
