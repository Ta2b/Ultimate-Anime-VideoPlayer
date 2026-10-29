#include "core/cadence.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::SourceId;
using namespace ajp::core;

namespace {

CadenceAccumulator make(Ratio factor) {
  auto accumulator = CadenceAccumulator::create(factor);
  REQUIRE(accumulator);
  return *accumulator;
}

SourceStamp stamp(std::uint32_t id, std::optional<std::int64_t> pts, std::uint32_t epoch = 0) {
  return {.id = {id}, .pts_us = pts, .epoch = epoch};
}

std::vector<Ratio> instants_t(const CadenceStep& step) {
  std::vector<Ratio> out;
  for (const OutputInstant& instant : step.instants()) out.push_back(instant.t);
  return out;
}

// 23,976 im/s en microsecondes : 1001000/24 = 125125/3 µs par source, arrondi par défaut.
std::int64_t ntsc_pts(std::int64_t k) {
  return k * 125125 / 3;
}

}  // namespace

TEST_CASE("cadence : facteurs refusés et réduits", "[cadence]") {
  for (Ratio bad : {Ratio{1, 1}, Ratio{0, 1}, Ratio{3, 0}, Ratio{-2, 1}, Ratio{21, 2}, Ratio{11, 1},
                    Ratio{100001, 100000}, Ratio{2, 3}}) {
    auto accumulator = CadenceAccumulator::create(bad);
    REQUIRE_FALSE(accumulator);
    CHECK(accumulator.error().code == RefusalCode::FieldOutOfRange);
  }
  CHECK(make({8, 1}).factor() == Ratio{8, 1});
  CHECK(make({16, 2}).factor() == Ratio{8, 1});
  CHECK(make({6, 4}).factor() == Ratio{3, 2});
  CHECK(make({11011, 3200}).factor() == Ratio{11011, 3200});
}

TEST_CASE("cadence : reduced() rend la forme irréductible", "[cadence]") {
  CHECK(reduced({6, 4}) == Ratio{3, 2});
  CHECK(reduced({3, -6}) == Ratio{-1, 2});
  CHECK(reduced({0, 5}) == Ratio{0, 1});
  CHECK(reduced({11011, 3200}) == Ratio{11011, 3200});
}

TEST_CASE("cadence : la première source est l'origine de la grille", "[cadence]") {
  CadenceAccumulator accumulator = make({2, 1});
  const CadenceStep step = accumulator.push(stamp(7, 1000));
  CHECK(step.event == GridEvent::Start);
  CHECK(step.right == SourceId{7});
  REQUIRE(step.instants().size() == 1);
  CHECK(step.instants()[0].is_source());
  CHECK(step.instants()[0].pts_us == 1000);
}

TEST_CASE("cadence : instants de chaque paire, comme rife_acc", "[cadence]") {
  struct Case {
    Ratio factor;
    std::vector<std::vector<Ratio>> pairs;  // motif répété
  };
  const std::vector<Case> cases{
      {{2, 1}, {{{1, 2}, {2, 2}}}},
      {{3, 1}, {{{1, 3}, {2, 3}, {3, 3}}}},
      {{4, 1}, {{{1, 4}, {2, 4}, {3, 4}, {4, 4}}}},
      {{5, 2}, {{{2, 5}, {4, 5}}, {{1, 5}, {3, 5}, {5, 5}}}},
      {{5, 3}, {{{3, 5}}, {{1, 5}, {4, 5}}, {{2, 5}, {5, 5}}}},
      {{3, 2}, {{{2, 3}}, {{1, 3}, {3, 3}}}},
  };
  for (const Case& c : cases) {
    CAPTURE(c.factor.num, c.factor.den);
    CadenceAccumulator accumulator = make(c.factor);
    accumulator.push(stamp(0, 0));
    for (std::uint32_t k = 1; k <= 12; ++k) {
      const CadenceStep step = accumulator.push(stamp(k, std::int64_t{k} * 40000));
      CHECK(step.event == GridEvent::Pair);
      CHECK(step.left == SourceId{k - 1});
      CHECK(step.right == SourceId{k});
      CHECK(instants_t(step) == c.pairs[(k - 1) % c.pairs.size()]);
    }
  }
}

TEST_CASE("cadence : 3/2 fait une interpolation par paire, à t = 2/3 puis t = 1/3", "[cadence]") {
  CadenceAccumulator accumulator = make({3, 2});
  accumulator.push(stamp(0, 0));
  for (std::uint32_t k = 1; k <= 20; ++k) {
    const CadenceStep step = accumulator.push(stamp(k, std::int64_t{k} * 41708));
    int interpolations = 0;
    for (const OutputInstant& instant : step.instants()) interpolations += instant.is_source() ? 0 : 1;
    CHECK(interpolations == 1);
    CHECK(step.instants()[0].t == (k % 2 == 1 ? Ratio{2, 3} : Ratio{1, 3}));
  }
}

TEST_CASE("cadence : horodatages interpolés, arrondis au plus proche", "[cadence]") {
  CadenceAccumulator accumulator = make({3, 2});
  accumulator.push(stamp(0, 0));
  const CadenceStep first = accumulator.push(stamp(1, 41708));
  REQUIRE(first.instants().size() == 1);
  CHECK(first.instants()[0].pts_us == 27805);  // 41708 · 2/3 = 27805,33
  const CadenceStep second = accumulator.push(stamp(2, 83417));
  REQUIRE(second.instants().size() == 2);
  CHECK(second.instants()[0].pts_us == 55611);  // 41708 + 41709/3
  CHECK(second.instants()[1].pts_us == 83417);  // la source elle-même

  CadenceAccumulator half = make({2, 1});
  half.push(stamp(0, 0));
  CHECK(half.push(stamp(1, 3)).instants()[0].pts_us == 2);  // 1,5 arrondi vers le haut
}

TEST_CASE("cadence : aucune dérive sur une longue séquence, contre le calcul exact", "[cadence]") {
  const std::vector<Ratio> factors{{2, 1}, {3, 1},        {4, 1},        {5, 2},       {5, 3},
                                   {3, 2}, {11011, 3200}, {11011, 1600}, {11011, 4800}};
  for (const Ratio factor : factors) {
    CAPTURE(factor.num, factor.den);
    CadenceAccumulator accumulator = make(factor);
    const std::int64_t sources = factor.num > 100 ? 100000 : 20000;
    std::int64_t j = 0;  // rang de l'image de sortie depuis l'origine
    bool all_exact = true;
    for (std::int64_t k = 0; k < sources; ++k) {
      const CadenceStep step = accumulator.push(stamp(static_cast<std::uint32_t>(k), ntsc_pts(k)));
      for (const OutputInstant& instant : step.instants()) {
        // L'image j se place à la position exacte j·den/num de la source d'origine.
        const std::int64_t whole = j * factor.den / factor.num;
        const std::int64_t r = j * factor.den % factor.num;
        const Ratio expected_t = r == 0 ? Ratio{factor.num, factor.num} : Ratio{r, factor.num};
        const std::int64_t left = r == 0 ? whole - 1 : whole;
        const bool same_pair = j == 0 || step.left == SourceId{static_cast<std::uint32_t>(left)};
        // |pts - (pts_k + Δ·r/num)| <= 1/2, en entiers.
        const std::int64_t base = ntsc_pts(whole);
        const std::int64_t delta = ntsc_pts(whole + 1) - base;
        const std::int64_t error = *instant.pts_us * factor.num - (base * factor.num + delta * r);
        all_exact = all_exact && instant.t == expected_t && same_pair && 2 * error <= factor.num &&
                    -2 * error <= factor.num;
        ++j;
      }
    }
    CHECK(all_exact);
    CHECK(j == 1 + (sources - 1) * factor.num / factor.den);
  }
}

TEST_CASE("cadence : 11011/3200 tient 82,5 im/s sur une heure de 23,976 im/s", "[cadence]") {
  CadenceAccumulator accumulator = make({11011, 3200});
  std::int64_t j = 0;
  std::int64_t last = 0;
  for (std::int64_t k = 0; k <= 86314; ++k) {
    const CadenceStep step = accumulator.push(stamp(static_cast<std::uint32_t>(k), ntsc_pts(k)));
    for (const OutputInstant& instant : step.instants()) {
      last = *instant.pts_us;
      ++j;
    }
  }
  // L'image j tombe à j/82,5 s : 2·j·10⁶/165 µs, à l'arrondi des pts source près.
  const std::int64_t exact = (j - 1) * 2'000'000 / 165;
  CHECK(last - exact <= 1);
  CHECK(exact - last <= 1);
}

TEST_CASE("cadence : Flush et époques", "[cadence]") {
  CadenceAccumulator accumulator = make({2, 1});
  CHECK(accumulator.push(stamp(1, 0)).event == GridEvent::Start);
  CHECK(accumulator.push(stamp(2, 40000)).event == GridEvent::Pair);

  REQUIRE(accumulator.flush(1));
  SECTION("une image de l'époque passée est jetée") {
    const CadenceStep stale = accumulator.push(stamp(3, 80000, 0));
    CHECK(stale.event == GridEvent::Stale);
    CHECK(stale.instants().empty());
  }
  SECTION("la première image de la nouvelle époque est l'origine") {
    const CadenceStep origin = accumulator.push(stamp(10, 5'000'000, 1));
    CHECK(origin.event == GridEvent::NewEpoch);
    REQUIRE(origin.instants().size() == 1);
    CHECK(origin.instants()[0].pts_us == 5'000'000);
    CHECK(accumulator.push(stamp(11, 5'040'000, 1)).event == GridEvent::Pair);
    CHECK(accumulator.push(stamp(12, 5'080'000, 0)).event == GridEvent::Stale);
  }
  SECTION("une époque plus récente sans Flush vaut un Flush") {
    CHECK(accumulator.push(stamp(20, 9'000'000, 3)).event == GridEvent::NewEpoch);
    CHECK(accumulator.push(stamp(21, 9'040'000, 3)).event == GridEvent::Pair);
    CHECK(accumulator.push(stamp(22, 9'080'000, 1)).event == GridEvent::Stale);
  }
  SECTION("une époque plus ancienne est refusée, la même est permise") {
    auto older = accumulator.flush(0);
    REQUIRE_FALSE(older);
    CHECK(older.error().code == RefusalCode::SequenceOrder);
    CHECK(accumulator.flush(1));
  }
}

TEST_CASE("cadence : un pts invalide ou qui ne croît pas resynchronise", "[cadence]") {
  CadenceAccumulator accumulator = make({5, 2});
  accumulator.push(stamp(0, 0));
  CHECK(accumulator.push(stamp(1, 40000)).event == GridEvent::Pair);

  const CadenceStep invalid = accumulator.push(stamp(2, std::nullopt));
  CHECK(invalid.event == GridEvent::InvalidPts);
  REQUIRE(invalid.instants().size() == 1);
  CHECK_FALSE(invalid.instants()[0].pts_us);

  // La gauche n'a pas d'horodatage : la source suivante redevient l'origine.
  const CadenceStep after = accumulator.push(stamp(3, 120000));
  CHECK(after.event == GridEvent::InvalidPts);
  CHECK(after.instants()[0].pts_us == 120000);

  // La grille repart à den/num : le motif de 5/2 recommence.
  CHECK(instants_t(accumulator.push(stamp(4, 160000))) == std::vector<Ratio>{{2, 5}, {4, 5}});

  CHECK(accumulator.push(stamp(5, 160000)).event == GridEvent::PtsNotIncreasing);
  CHECK(accumulator.push(stamp(6, 150000)).event == GridEvent::PtsNotIncreasing);
  CHECK(instants_t(accumulator.push(stamp(7, 190000))) == std::vector<Ratio>{{2, 5}, {4, 5}});
}

TEST_CASE("cadence : stamp_of traduit kNoPts en horodatage absent, et retour", "[cadence]") {
  const ajp::DecodedFrame unknown{.id = {4}, .epoch = 2};
  CHECK(unknown.pts_us == ajp::kNoPts);  // inconnu tant que media ne l'a pas posé
  const SourceStamp none = stamp_of(unknown);
  CHECK(none.id == SourceId{4});
  CHECK(none.epoch == 2);
  CHECK_FALSE(none.pts_us);
  CHECK(stamp_of(ajp::DecodedFrame{.pts_us = 0}).pts_us == 0);  // 0 est un pts connu

  CadenceAccumulator accumulator = make({2, 1});
  accumulator.push(stamp(1, 0, 2));
  const CadenceStep step = accumulator.push(none);
  CHECK(step.event == GridEvent::InvalidPts);
  CHECK(ajp::pts_or_unknown(step.instants()[0].pts_us) == ajp::kNoPts);
  CHECK(ajp::pts_or_unknown(41'708) == 41'708);
}

TEST_CASE("cadence : resync() après un échec relance la grille à la source droite", "[cadence]") {
  CadenceAccumulator accumulator = make({3, 2});
  accumulator.push(stamp(0, 0));
  CHECK(instants_t(accumulator.push(stamp(1, 40000))) == std::vector<Ratio>{{2, 3}});
  accumulator.resync();
  // Sans resync, la paire suivante serait {1/3, 3/3}.
  CHECK(instants_t(accumulator.push(stamp(2, 80000))) == std::vector<Ratio>{{2, 3}});
}

TEST_CASE("cadence : horodatages extrêmes sans débordement", "[cadence]") {
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min() + 1;
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  CadenceAccumulator accumulator = make({2, 1});
  accumulator.push(stamp(0, kMin));
  const CadenceStep step = accumulator.push(stamp(1, kMax));
  REQUIRE(step.instants().size() == 2);
  CHECK(step.instants()[0].pts_us == 0);
  CHECK(step.instants()[1].pts_us == kMax);
}

TEST_CASE("cadence : une source à cadence variable garde la grille des paires", "[cadence]") {
  // Comme rife_acc : les instants ne dépendent que du rang de la paire, les
  // horodatages se placent dans chaque paire, quel que soit son écart.
  CadenceAccumulator accumulator = make({2, 1});
  accumulator.push(stamp(0, 0));
  const std::vector<std::pair<std::int64_t, std::vector<std::int64_t>>> pairs{
      {40000, {20000, 40000}}, {50000, {45000, 50000}}, {130000, {90000, 130000}}};
  std::uint32_t k = 1;
  for (const auto& [pts, expected] : pairs) {
    const CadenceStep step = accumulator.push(stamp(k++, pts));
    REQUIRE(step.event == GridEvent::Pair);
    REQUIRE(step.instants().size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) CHECK(step.instants()[i].pts_us == expected[i]);
  }
}

TEST_CASE("cadence : resync() sans paire en cours ne change rien", "[cadence]") {
  CadenceAccumulator accumulator = make({5, 2});
  accumulator.resync();
  CHECK(accumulator.push(stamp(0, 0)).event == GridEvent::Start);
  accumulator.resync();  // l'origine vient d'être posée : la grille part déjà d'elle
  CHECK(instants_t(accumulator.push(stamp(1, 40000))) == std::vector<Ratio>{{2, 5}, {4, 5}});
}

namespace {

// La boucle de vf_animejanai.c:1795-1855 telle quelle, en doubles, sans échec
// d'interpolation : les instants (acc, pts) d'une source reçue.
struct ForkLoop {
  using Instant = std::pair<std::int64_t, std::optional<double>>;
  std::int64_t num;
  std::int64_t den;
  std::int64_t acc = den;
  std::optional<std::optional<double>> prev;  // absent : début ; vide : pts invalide

  std::vector<Instant> push(std::optional<double> pts) {
    std::vector<Instant> out;
    if (prev && *prev && pts && *pts > **prev) {
      for (; acc <= num; acc += den) {
        const double t = static_cast<double>(acc) / static_cast<double>(num);
        out.emplace_back(acc, acc == num ? *pts : **prev + (*pts - **prev) * t);
      }
      acc -= num;
    } else {
      acc = den;
      out.emplace_back(num, pts);
    }
    prev = pts;
    return out;
  }
};

}  // namespace

TEST_CASE("cadence : même suite que la boucle du fork, pts invalides et reculs compris", "[cadence]") {
  std::mt19937_64 random(20261018);  // suite fixée : mt19937_64 est défini par la norme
  for (const Ratio factor : {Ratio{2, 1}, Ratio{4, 1}, Ratio{5, 2}, Ratio{5, 3}, Ratio{3, 2}, Ratio{8, 1},
                             Ratio{11011, 3200}, Ratio{11011, 4800}}) {
    CAPTURE(factor.num, factor.den);
    CadenceAccumulator accumulator = make(factor);
    ForkLoop fork{.num = factor.num, .den = factor.den};
    std::int64_t clock = 0;
    bool same = true;
    for (std::uint32_t k = 0; k < 5000; ++k) {
      const std::uint64_t draw = random() % 100;
      std::optional<std::int64_t> pts;  // 2 % sans horodatage
      if (draw >= 2) {
        clock += 30000 + static_cast<std::int64_t>(random() % 20000);  // cadence variable
        pts = draw < 4 ? clock - 70000 : clock;                        // 2 % de reculs
      }
      const CadenceStep step = accumulator.push(stamp(k, pts));
      const auto expected = fork.push(pts ? std::optional<double>(static_cast<double>(*pts)) : std::nullopt);
      same = same && step.instants().size() == expected.size();
      for (std::size_t i = 0; same && i < expected.size(); ++i) {
        const OutputInstant& instant = step.instants()[i];
        const auto& [acc, fork_pts] = expected[i];
        same = instant.t == Ratio{acc, factor.num} && instant.pts_us.has_value() == fork_pts.has_value() &&
               (!fork_pts || std::abs(static_cast<double>(*instant.pts_us) - *fork_pts) <= 0.5);
      }
    }
    CHECK(same);
  }
}

// Le mode auto change le facteur par petits pas : la grille garde sa phase (aucun saut de rythme) et ne
// dérive pas, même après des dizaines de minutes de changements.
TEST_CASE("cadence : retarget garde les instants équidistants, monotones et sans dérive", "[cadence][auto]") {
  constexpr std::int64_t kSourcePeriod = 41'708;  // 23,976 i/s
  const std::int64_t kSeconds = 600;
  auto accumulator = make(Ratio{.num = 67, .den = 24});
  const std::array<std::int64_t, 3> steps{67, 55, 120};  // 67 → 55 → 120 i/s environ
  std::int64_t current = steps[0];
  std::vector<std::int64_t> pts;
  std::vector<std::int64_t> step_at;  // le pas en vigueur pour chaque instant
  std::int64_t source_pts = 0;
  std::uint32_t id = 1;
  int retargets = 0;
  double ideal = 0.0;
  for (std::int64_t n = 0; source_pts < kSeconds * 1'000'000; ++n) {
    if (n > 0 && n % 120 == 0) {  // toutes les 5 s environ
      current = steps[static_cast<std::size_t>((n / 120) % 3)];
      REQUIRE(accumulator.retarget(Ratio{.num = current, .den = 24}));
      ++retargets;
    }
    const CadenceStep step = accumulator.push({.id = SourceId{id++}, .pts_us = source_pts, .epoch = 0});
    ideal += step.event == GridEvent::Pair ? static_cast<double>(current) / 24.0 : 1.0;
    for (const OutputInstant& at : step.instants()) {
      REQUIRE(at.pts_us);
      pts.push_back(*at.pts_us);
      step_at.push_back(current);
    }
    source_pts += kSourcePeriod;
  }
  REQUIRE(retargets > 100);
  for (std::size_t i = 1; i < pts.size(); ++i) {
    REQUIRE(pts[i] > pts[i - 1]);  // pts strictement croissants
    const double gap = static_cast<double>(pts[i] - pts[i - 1]) / static_cast<double>(kSourcePeriod);
    // Équidistants à un pas de grille près (1 / num de période) ; à un changement, l'écart reste entre
    // l'ancien et le nouveau pas, jamais un trou.
    const double now_gap = 24.0 / static_cast<double>(step_at[i]);
    const double before_gap = 24.0 / static_cast<double>(step_at[i - 1]);
    const double tolerance = 1.5 / static_cast<double>(std::min(step_at[i], step_at[i - 1])) + 1e-3;
    CHECK(gap >= std::min(now_gap, before_gap) - tolerance);
    CHECK(gap <= std::max(now_gap, before_gap) + tolerance);
    if (step_at[i] == step_at[i - 1]) CHECK(std::abs(gap - now_gap) <= tolerance);
  }
  // Aucune dérive : le nombre d'instants suit l'intégrale de la cadence, à un instant par changement près.
  CHECK(std::abs(static_cast<double>(pts.size()) - ideal) <= static_cast<double>(retargets) + 2.0);
  // Et le dernier instant tombe sur la fin de la lecture : la vitesse du contenu n'a pas bougé.
  CHECK(pts.back() <= source_pts);
  CHECK(pts.back() >= source_pts - 2 * kSourcePeriod);
}

TEST_CASE("cadence : retarget refuse un facteur invalide et ne change rien", "[cadence][auto]") {
  auto accumulator = make(Ratio{.num = 3, .den = 2});
  CHECK_FALSE(accumulator.retarget(Ratio{.num = 1, .den = 1}));
  CHECK(accumulator.factor().num == 3);
  CHECK(accumulator.retarget(Ratio{.num = 3, .den = 2}));
}
