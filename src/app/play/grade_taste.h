// Le goût de l'utilisateur pour le grade du contrat v2 (chantier 8.10, essai --hdr-contract-v2-curve) : ce
// que l'apprentissage retient quand la courbe par plan (render/curve.h) remplace le grade adaptatif.
//
// - gris → un décalage E par classe de scène (sombre, sombre avec éclat, claire, ciel), ±kCurveMaxE ;
// - noirs → le contraste du pied de la courbe (±kCurveMaxToe) ;
// - couleurs → la vibrance, de 1 à kCurveMaxVibrance.
//
// La table de migration (migrate) part des réglages d'avant (GradeCurve) ; les échantillons de
// adaptive-learning.tsv, faits pour l'autre grade, sont écartés (et comptés au journal), ceux du goût
// s'écrivent dans adaptive-learning-v2.tsv.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/grade_learning.h"
#include "media/shot_cutter.h"

namespace ajp::app::play {

enum class SceneClass : std::uint8_t { Dark, Glow, Bright, Sky };
inline constexpr std::size_t kSceneClasses = 4;
std::string_view name(SceneClass scene);

struct GradeTaste {
  std::array<double, kSceneClasses> e{};  // diaphragmes, par SceneClass
  double toe = 0.0;
  double vibrance = 1.0;
  double e_of(SceneClass scene) const { return e.at(static_cast<std::size_t>(scene)); }
  friend bool operator==(const GradeTaste&, const GradeTaste&) = default;
};
// Bornes du goût : E à ±kCurveMaxE, pied à ±kCurveMaxToe, vibrance de 1 à kCurveMaxVibrance.
GradeTaste clamped(GradeTaste taste);

// La table de migration des réglages d'avant :
//   E(classe) = log2(gris de la classe / gris par défaut de la classe) / 2 (gris : sombre et éclat, gris
//               clair : claire et ciel ; 50, 40, 30 par défaut) ;
//   pied      = (noirs − 0,15) / 2 ;
//   vibrance  = 1 + (couleurs − 0,10) × 0,75.
// Une courbe aux valeurs par défaut donne le goût neutre.
GradeTaste migrate(const GradeCurve& curve);

// La classe d'une scène vue par le grade adaptatif (ses poids) : ciel quand tv ≥ 0,5, sinon la plus lourde
// de sombre (1 − t)(1 − glow), éclat (1 − t) glow et claire t.
SceneClass class_of(const SceneMix& mix);
// La classe d'un plan de la pré-analyse, sur les mêmes règles (render/adaptive_rules.h) : series_mean est
// la luminosité médiane de la série (ShotStats::mean), 0 si inconnue.
SceneClass class_of(const media::ShotStats& stats, double series_mean);

struct TasteSample {
  std::int64_t unix_s = 0;
  LearnSample::Kind kind = LearnSample::Kind::Correction;
  SceneClass scene = SceneClass::Dark;
  GradeTaste taste;  // le goût appliqué, après la correction
};

struct TasteReport {
  GradeTaste taste;
  int corrections = 0;
  int approvals = 0;
  std::array<double, kSceneClasses> support{};  // poids des échantillons par classe
};
// La moyenne régularisée vers now : E de chaque classe sur les échantillons de sa classe, pied et vibrance
// sur tous ; une correction pèse 1, une approbation 0,25, now kTastePrior. Moitié du poids après 90 jours.
// Refuse sans au moins deux corrections.
inline constexpr double kTastePrior = 1.0;
std::expected<TasteReport, std::string> learn_taste(std::span<const TasteSample> samples,
                                                    const GradeTaste& now, std::int64_t now_unix);

// adaptive-learning-v2.tsv.
std::string_view taste_header();
std::string taste_line(const TasteSample& sample);
struct TasteRead {
  std::vector<TasteSample> samples;
  int rejected = 0;  // lignes qui ne se lisent pas en v2 (d'un autre format)
};
TasteRead parse_taste_samples(std::string_view text);
TasteRead read_taste_samples(const std::filesystem::path& file);
void append_taste_sample(const std::filesystem::path& file, const TasteSample& sample);
// Le fichier du goût à côté de celui de l'apprentissage d'avant.
std::filesystem::path taste_log_of(const std::filesystem::path& learning_log);

}  // namespace ajp::app::play
