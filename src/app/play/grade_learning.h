// Apprentissage du grade adaptatif : ce que l'utilisateur corrige sur les scènes qu'il regarde.
//
// Chaque correction est une scène (sa luminosité rapportée à la série, son éclat, sa couleur) et les valeurs
// que l'utilisateur y a retenues (noirs, couleurs, gris moyen vraiment appliqués). Le grade adaptatif est un
// mélange linéaire de bouts de courbe (scène sombre, sombre avec éclat, claire, ciel bleu) pondérés par la
// scène : apprendre, c'est trouver les bouts et les seuils qui rendent le mieux toutes les corrections à la
// fois, sans s'éloigner de la calibration là où rien ne la contredit (moindres carrés régularisés).
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/settings.h"

namespace ajp::app::play {

// La luminosité de référence d'une série (Re:Zero, 0,30) : les seuils du grade adaptatif suivent la série en
// proportion (chain_hdr.cpp : adaptive_tick).
inline constexpr double kSeriesRef = 0.30;
double series_ratio(double series_luma);

// Les bouts de la courbe du grade adaptatif et ses seuils (% du blanc).
struct GradeCurve {
  double lo = 13.0;
  double hi = 54.0;
  double shadows = 0.15;  // scène sombre (les curseurs du grade)
  double shadows_bright = 0.80;
  double shadows_glow = 0.20;
  double colors = 0.10;  // vibrance de RTX HDR
  double colors_cut = 0.0;
  double colors_glow = 0.05;
  double colors_sky = 0.0;  // ciel bleu saturé, flash très coloré
  double gray = 50.0;       // gris moyen de RTX HDR
  double gray_bright = 30.0;
  double gray_glow = 40.0;
  friend bool operator==(const GradeCurve&, const GradeCurve&) = default;
};
GradeCurve curve_of(const PlaySettings& s);
void set_curve(PlaySettings& s, const GradeCurve& curve);

// Ce que la courbe rend pour une scène : t (sombre 0 à claire 1), glow (éclat) et tv (ciel), comme
// with_adaptive (chain.h).
struct SceneLook {
  double shadows = 0.0;
  double colors = 0.0;
  double gray = 0.0;
};
SceneLook look_at(const GradeCurve& curve, double t, double glow, double tv);

// Les familles de réglages qu'une correction touche (masque).
enum Family : std::uint8_t { kShadowsFamily = 1, kColorsFamily = 2, kGrayFamily = 4 };
inline constexpr std::uint8_t kAllFamilies = kShadowsFamily | kColorsFamily | kGrayFamily;
std::uint8_t touched_families(const GradeCurve& before, const GradeCurve& after);

// « Cette scène » : la valeur voulue pour la scène à l'écran, répartie sur les bouts de la courbe selon leur
// part dans la scène (le plus petit changement qui l'obtient, bornes des curseurs comprises).
void correct_scene(GradeCurve& curve, double t, double glow, double tv, Family family, double wanted);

// Une scène telle que le grade adaptatif la voit : de quoi recalculer ses poids sous d'autres seuils.
struct LearnScene {
  double x = 0.0;          // luminance lissée de la scène, rapportée à la série (luma / series_ratio)
  double glow_seen = 0.0;  // mesure lissée de la scène sombre avec éclat (0 à 1)
  double tv = 0.0;         // poids du ciel bleu ou du flash coloré (0 à 1)
  double force = -1.0;     // calibration : la scène forcée (GradeSettings::adaptive_force) ; négatif : aucune
};

struct LearnSample {
  enum class Kind : std::uint8_t {
    Correction,   // l'utilisateur a changé la courbe sur cette scène
    Calibration,  // il l'a réglée pendant la calibration (scène forcée)
    Approval,     // il l'a regardée longtemps sans rien changer
  };
  std::int64_t unix_s = 0;
  Kind kind = Kind::Correction;
  std::uint8_t touched = 0;  // les familles corrigées (Family) ; les autres valent approbation
  LearnScene scene;
  SceneLook look;  // les valeurs appliquées, après la correction
};

// Les poids d'une scène sous les seuils lo et hi : t, puis l'éclat qui ne vaut que dans le sombre.
struct SceneMix {
  double t = 0.0;
  double glow = 0.0;
  double tv = 0.0;
};
SceneMix scene_mix(const LearnScene& scene, double lo, double hi);

struct LearnReport {
  GradeCurve curve;
  int corrections = 0;  // corrections et réglages de calibration retenus
  int approvals = 0;
  double support_dark = 0.0;  // poids des corrections dans chaque genre de scène
  double support_glow = 0.0;
  double support_bright = 0.0;
  double error_before = 0.0;  // écart moyen des corrections à la courbe (part de la plage des curseurs)
  double error_after = 0.0;
};
// Les bouts et les seuils qui rendent le mieux les échantillons, régularisés vers la courbe now. Refuse
// (raison en clair) sans au moins deux corrections.
std::expected<LearnReport, std::string> learn_grade(std::span<const LearnSample> samples,
                                                    const GradeCurve& now, std::int64_t now_unix);

// adaptive-learning.tsv : une ligne par échantillon (en-tête à la création).
std::string learn_line(const LearnSample& sample);
std::string_view learn_header();
std::vector<LearnSample> parse_learn_samples(std::string_view text);
std::vector<LearnSample> read_learn_samples(const std::filesystem::path& file);
void append_learn_sample(const std::filesystem::path& file, const LearnSample& sample);

}  // namespace ajp::app::play
