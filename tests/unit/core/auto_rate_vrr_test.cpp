// L'Auto au changement de granularité (le VRR s'arme ou se coupe), au pas fin, et ses départs mémorisés
// (audit 3, C12).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/auto_rate.h"
#include "core/free_grid.h"
#include "core/vblank_clock.h"
#include "core/vsync_index.h"

using namespace ajp;
using namespace ajp::core;

namespace {

constexpr double kFps = 24000.0 / 1001.0;
constexpr std::int64_t kPeriod = 41'708;

// Paire après paire, une toutes les 41,7 ms, chacune peu coûteuse (1 ms par image), jusqu'au plafond ou
// jusqu'à `pairs` paires.
std::int64_t climb(AutoRate& a, std::int64_t t, int pairs = 20'000) {
  for (int i = 0; i < pairs && a.step() < a.ceiling(); ++i) {
    t += kPeriod;
    const int images = a.step() / a.granularity() + 1;
    a.cost(images, 1'000 * images, t);
    a.pair({.now_us = t, .made_step = a.step(), .late = 0, .period_us = kPeriod});
  }
  return t;
}

}  // namespace

TEST_CASE("C12-01 : le VRR qui s'arme garde le palier tenu (165 Hz, 23,976 i/s)", "[core][auto]") {
  const AutoLimits fixed = auto_limits(165.0, kFps, false, kAutoMaxStep);
  AutoRate a(fixed.ceiling, fixed.granularity);
  a.set_locked(fixed.locked, fixed.exact);
  climb(a, 1'000'000);
  REQUIRE(a.step() == 14);  // ×6,88 (11011/1600), arrondi à 14 demi-paliers
  REQUIRE(a.centi() == 688);
  const AutoLimits vrr = auto_limits(165.0, kFps, true, kAutoMaxStep);
  // L'ordre de LiveSettings::set_auto : set_ceiling(plafond, granularité) puis set_locked.
  a.set_ceiling(vrr.ceiling, vrr.granularity);
  a.set_locked(vrr.locked, vrr.exact);
  // Le plus haut palier fin sous l'écran (×6,875), et non ×1 : les paliers calés en demi-paliers ne
  // s'appliquent plus au pas fin.
  CHECK(a.step() == vrr.ceiling);
  CHECK(a.centi() >= 680);
  // Et retour à fréquence fixe : jamais plus haut que ce qui tenait, donc le calé ×6,88 n'est pas repris
  // (au-dessus de ×6,875), mais ni ×1 ni un palier hors de la liste.
  a.set_ceiling(fixed.ceiling, fixed.granularity);
  a.set_locked(fixed.locked, fixed.exact);
  CHECK(a.step() > fixed.granularity);
  CHECK(std::ranges::find(fixed.locked, a.step()) != fixed.locked.end());
}

TEST_CASE("C12-03 : au pas fin, la montée atteint le plafond de l'écran quand le GPU a de la marge",
          "[core][auto]") {
  for (const double hz : {60.0, 120.0, 144.0, 165.0}) {
    const AutoLimits vrr = auto_limits(hz, kFps, true, kAutoMaxStep);
    AutoRate a(vrr.ceiling, vrr.granularity);
    climb(a, 1'000'000);
    INFO("écran à " << hz << " Hz");
    // 1 ms par image pour 41,7 ms de période : le plafond tient avec 75 % de marge.
    CHECK(a.step() == vrr.ceiling);
  }
}

TEST_CASE("C12-04 : sans mesure de coût, l'Auto au pas fin monte quand même", "[core][auto]") {
  const AutoLimits vrr = auto_limits(165.0, kFps, true, kAutoMaxStep);
  AutoRate a(vrr.ceiling, vrr.granularity);
  REQUIRE(a.step() == 2 * kFineGranularity);
  std::int64_t t = 1'000'000;
  for (int i = 0; i < 4'000; ++i) {  // ~170 s, aucune paire mesurée
    t += kPeriod;
    a.pair({.now_us = t, .made_step = a.step(), .late = 0, .period_us = kPeriod});
  }
  CHECK(a.step() > 2 * kFineGranularity);
}

TEST_CASE("C12-05 : le départ mémorisé retrouve le palier calé par son facteur exact", "[core][auto]") {
  // 165 Hz fixes : calés {2, 3, 5, 7, 14} en demi-paliers, 7 étant ×3,44 (11011/3200).
  const AutoLimits fixed = auto_limits(165.0, kFps, false, kAutoMaxStep);
  REQUIRE(std::ranges::find(fixed.locked, 7) != fixed.locked.end());
  AutoRate a(fixed.ceiling, fixed.granularity);
  a.set_locked(fixed.locked, fixed.exact);
  std::int64_t t = 1'000'000;
  a.seed(344, t);
  CHECK(a.step() == 5);  // un cran calé sous ×3,44
  // Un GPU dont la prédiction (15 % de marge) ne tient que ×2,5 (t(×3) = 89 % de la période), mais qui tient
  // ×3,44 sous la période (t(×4) = 98 %) : seul le retour au palier retenu, quand le palier courant laisse
  // 10 % d'air, peut faire monter. Une division entière retenait 6, sous le calé suivant (7) : aucune montée.
  const auto cost_of = [](double images) { return (0.62 + 0.09 * images) * static_cast<double>(kPeriod); };
  for (int i = 0; i < 400; ++i) {
    t += kPeriod;
    const int images = a.step() / a.granularity() + i % 2;
    a.cost(images, static_cast<std::int64_t>(cost_of(images)), t);
    a.pair({.now_us = t, .made_step = a.step(), .late = 0, .period_us = kPeriod});
  }
  CHECK(a.info(t).predicted == 5);
  CHECK(a.step() == 7);
}

TEST_CASE("C12-06 : l'Auto au niveau du menu suit un écran de 240 Hz jusqu'à ×10", "[core][auto]") {
  const AutoLimits fixed = auto_limits(240.0, kFps, false, kAutoMaxStep);
  CHECK(fixed.ceiling == 2 * kAutoMaxStep);  // ×10 calé, et non ×5
  const AutoLimits vrr = auto_limits(240.0, kFps, true, kAutoMaxStep);
  CHECK(vrr.ceiling == auto_ceiling(240.0, kFps, kFineGranularity));
  CHECK(vrr.ceiling > 8 * kFineGranularity);
}

namespace {

// Vblanks affichés : 144 Hz, phase 0 ; horloge maître à la vitesse 1 (temps de contenu = temps hôte).
constexpr VblankGrid kGrid{.phase_us = 0.0, .period_us = 1e6 / 144.0};

// Les instants de 400 paires de 23,976 i/s, GPU rapide (pas = 1 période), ancre donnée.
std::vector<std::int64_t> instants(std::int64_t anchor) {
  FreeGrid grid;
  std::vector<std::int64_t> out;
  const auto period = static_cast<std::int64_t>(std::llround(kGrid.period_us));
  for (int i = 0; i < 400; ++i) {
    const auto pts = static_cast<std::int64_t>(std::llround(1'000'000.0 + i * 1e6 * 1001.0 / 24000.0));
    const CadenceStep step = grid.push(
        {.id = SourceId{static_cast<std::uint32_t>(i + 1)}, .pts_us = pts},
        {.min_gap_us = period, .step_us = period, .vsync_period_us = period, .vsync_origin_us = anchor});
    for (const OutputInstant& at : step.instants())
      if (at.pts_us) out.push_back(*at.pts_us);
  }
  return out;
}

// Le vblank que Session::pick donne à chaque instant, l'heure de son pts tremblant de ±jitter périodes.
int collisions(const std::vector<std::int64_t>& pts, double jitter) {
  std::mt19937 rng(12);
  std::uniform_real_distribution<double> noise(-jitter, jitter);
  VsyncPicker picker;
  std::set<std::int64_t> used;
  int same = 0;
  for (const std::int64_t p : pts) {
    const std::int64_t k = picker.pick(kGrid.index_of(static_cast<double>(p)) + noise(rng));
    if (!used.insert(k).second) ++same;  // deux images neuves pour un même vblank : l'une est sautée
  }
  return same;
}

}  // namespace

TEST_CASE("C12-02 : ancrés sur le vblank, les instants du Rapide résistent à la gigue", "[core][free]") {
  const double index = std::round(kGrid.index_of(1'000'000.0));
  const auto period = static_cast<std::int64_t>(std::llround(kGrid.period_us));
  const std::int64_t on = std::llround(kGrid.at(static_cast<std::int64_t>(index)));
  // Gigue de 0,2 période (1,4 ms à 144 Hz) entre la planification (gpu) et le choix (main) : l'ancre sur le
  // vblank (LiveSettings::vsync_grid) ne saute aucune image ; à mi-chemin, l'ancre d'avant en sautait.
  CHECK(collisions(instants(on), 0.2) == 0);
  CHECK(collisions(instants(on + period / 2), 0.2) > 0);
}

TEST_CASE("Stable à 165 Hz fixe : la paire faite au facteur exact du palier calé compte, et l'Auto monte",
          "[core][auto]") {
  // Le lecteur donne le facteur de la paire, 11011/6400 au palier calé ×1,72 (palier 3) : comparé au palier
  // tel quel, aucune paire ne comptait et le Stable restait à ×1,72 avec une prédiction à ×4.
  const AutoLimits limits = auto_limits(165.0, kFps, false, 4);
  AutoRate a;
  a.set_ceiling(limits.ceiling, limits.granularity);
  a.set_locked(limits.locked, limits.exact);
  REQUIRE(a.step() == 3);
  CHECK(a.step_of(a.factor()) == 3);
  CHECK(a.step_of(Ratio{1, 1}) == limits.granularity);  // ×1, le plancher
  CHECK(a.step_of(Ratio{11011, 6401}) == -1);
  std::int64_t t = 1'000'000;
  for (int i = 0; i < 24 * 20 && a.step() < limits.locked[3]; ++i) {  // 20 s de paires légères
    t += kPeriod;
    const int images = static_cast<int>(std::ceil(a.factor().value()));
    a.cost(images, 1'000 * images, t);
    a.pair({.now_us = t, .late = 0, .period_us = kPeriod, .made_factor = a.factor()});
    a.tick(t, true);
  }
  CHECK(a.step() == limits.locked[3]);  // ×3,44, le plafond du niveau ×4
  CHECK(a.factor() == Ratio{11011, 3200});
  CHECK(a.reason() == AutoReason::Rise);
}
