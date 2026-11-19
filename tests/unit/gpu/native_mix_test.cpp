// Le choix d'un mélange de morceaux natifs (gpu/cuda/backend/native_mix.h) et les options des morceaux du
// GMFSS natif (gmfss_mix.h), sans CUDA : sur des chaînes simulées dont l'erreur est connue, le mélange retenu
// est le plus rapide qui tient le seuil, une vérification manquée recale la corrélation, rien ne tient sans
// que la base tienne, une erreur de la chaîne remonte. Puis les tolérances du banc du réglage
// (native_sums.h).
#include "gpu/cuda/backend/native_mix.h"

#include <cmath>
#include <expected>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/backend/gmfss_mix.h"
#include "gpu/cuda/backend/native_sums.h"

namespace kirari::gpu::cuda {
namespace {

using native_mix::Eval;
using native_mix::Options;

constexpr double kInf = std::numeric_limits<double>::infinity();

// Une chaîne simulée de deux morceaux, un groupe de pic 1 : effect[p][o], l'erreur quadratique d'une option
// contre la base (le natif f32) ; floor, celle de la base contre la référence. Les erreurs des morceaux
// s'ajoutent (indépendantes) ou s'alignent (le carré de la somme des racines).
struct Simulated {
  std::vector<std::vector<double>> effect;
  double floor = 1e-6;
  bool aligned = false;
  int calls = 0;
  int fail_at = -1;  // l'appel qui échoue (-1 : aucun)

  std::expected<Eval, std::string> operator()(const Options& o) {
    if (calls++ == fail_at) return std::unexpected(std::string("panne"));
    double power = 0.0, amplitude = 0.0;
    for (std::size_t p = 0; p < o.size(); ++p) {
      const double e = effect[p][static_cast<std::size_t>(o[p])];
      power += e;
      amplitude += std::sqrt(e);
    }
    const double base = aligned ? amplitude * amplitude : power;
    return Eval{.want = {floor + base}, .base = {base}};
  }
};

// Le morceau 0 a une tête (ses quatre crans), le morceau 1 n'en a pas (sans « f16 et tête f32 »).
native_mix::Problem two_pieces(double floor, double min_db) {
  return gmfss_mix::problem({{1.0, 1.1, 1.3, 1.8}, {0.5, kInf, 0.6, 0.9}}, {true, false}, {1.0}, {floor},
                            min_db);
}

std::vector<std::vector<double>> effects() {
  return {{4e-5, 1e-5, 5e-6, 0.0}, {8e-6, kInf, 2e-6, 0.0}};
}

}  // namespace

TEST_CASE("GMFSS natif : les options d'un morceau, avec ou sans tête", "[gpu][native_mix]") {
  const auto cands = gmfss_mix::candidates({true, false});
  CHECK(cands[0] ==
        std::vector<int>{gmfss_mix::kF16, gmfss_mix::kHeadF32, gmfss_mix::kMixed, gmfss_mix::kF32});
  CHECK(cands[1] == std::vector<int>{gmfss_mix::kF16, gmfss_mix::kMixed, gmfss_mix::kF32});
  const native_mix::Problem problem = two_pieces(1e-6, 45.0);
  CHECK(problem.base == Options{gmfss_mix::kF32, gmfss_mix::kF32});
  REQUIRE(problem.seeds.size() == 2);
  CHECK(problem.seeds[0] == Options{gmfss_mix::kF16, gmfss_mix::kF16});
  CHECK(problem.seeds[1] == Options{gmfss_mix::kMixed, gmfss_mix::kMixed});
  const auto sums = gmfss_mix::precisions({gmfss_mix::kHeadF32, gmfss_mix::kMixed});
  CHECK(sums[0].body == RifeNativeSums::kF16);
  CHECK(sums[0].head == RifeNativeSums::kF32);
  CHECK(sums[1].body == RifeNativeSums::kMixed);
  CHECK(gmfss_mix::mix_name({"ligne 0", "queue"}, {gmfss_mix::kF16, gmfss_mix::kHeadF32}) ==
        "ligne 0 f16, queue f16 et tête f32");
}

TEST_CASE("GMFSS natif : gardé seulement 2 % plus rapide que TensorRT", "[gpu][native_mix]") {
  CHECK(gmfss_mix::beats(10.0, 9.8));
  CHECK_FALSE(gmfss_mix::beats(10.0, 9.81));
  CHECK_FALSE(gmfss_mix::beats(10.0, 0.0));
}

TEST_CASE("Mélange natif : le plus rapide qui tient le seuil, vérifié une fois", "[gpu][native_mix]") {
  Simulated chain{.effect = effects()};
  const auto mix = native_mix::choose(two_pieces(1e-6, 45.0), chain);
  REQUIRE(mix);
  CHECK(mix->held);
  // f16 partout (1,5 ms) fait 43,1 dB ; la tête du morceau 0 en f32 (1,6 ms) 47,2 dB.
  CHECK(mix->options == Options{gmfss_mix::kHeadF32, gmfss_mix::kF16});
  CHECK(mix->db == Catch::Approx(10.0 * std::log10(1.0 / 1.9e-5)));
  CHECK(mix->predicted == Catch::Approx(mix->db));
  // Cinq effets seuls, deux graines, une vérification.
  CHECK(chain.calls == 8);
  CHECK(mix->effect_db[0][gmfss_mix::kF16] == Catch::Approx(10.0 * std::log10(1.0 / 4e-5)));
  CHECK(std::isnan(mix->effect_db[1][gmfss_mix::kHeadF32]));
  REQUIRE(mix->seed_db.size() == 2);
  CHECK(mix->seed_db[0] == Catch::Approx(10.0 * std::log10(1.0 / 4.9e-5)));
}

TEST_CASE("Mélange natif : une vérification manquée recale la corrélation", "[gpu][native_mix]") {
  Simulated chain{.effect = effects(), .aligned = true};
  const auto mix = native_mix::choose(two_pieces(1e-6, 45.0), chain);
  REQUIRE(mix);
  CHECK(mix->held);
  // Prévu indépendant, la tête f32 et le f16 font 44,3 dB alignés : manqué. Recalé, le plus rapide qui tient
  // est la tête f32 et les mixtes (1,7 ms, 46,6 dB).
  CHECK(mix->options == Options{gmfss_mix::kHeadF32, gmfss_mix::kMixed});
  CHECK(mix->db >= 45.0);
  CHECK(chain.calls == 9);
}

TEST_CASE("Mélange natif : une option non finie n'est jamais prise", "[gpu][native_mix]") {
  Simulated chain{.effect = effects()};
  chain.effect[0][gmfss_mix::kF16] = kInf;
  chain.effect[0][gmfss_mix::kHeadF32] = kInf;
  const auto mix = native_mix::choose(two_pieces(1e-6, 45.0), chain);
  REQUIRE(mix);
  CHECK(mix->held);
  CHECK(mix->options[0] == gmfss_mix::kMixed);
}

TEST_CASE("Mélange natif : rien ne tient quand la base ne tient pas", "[gpu][native_mix]") {
  Simulated chain{.effect = effects(), .floor = 1e-4};
  const auto mix = native_mix::choose(two_pieces(1e-4, 45.0), chain);
  REQUIRE(mix);
  CHECK_FALSE(mix->held);
  CHECK(mix->options == Options{gmfss_mix::kF32, gmfss_mix::kF32});
}

TEST_CASE("Mélange natif : la base retenue sans autre passe quand elle seule tient", "[gpu][native_mix]") {
  Simulated chain{.effect = {{1e-3, 1e-3, 1e-3, 0.0}, {1e-3, kInf, 1e-3, 0.0}}};
  const auto mix = native_mix::choose(two_pieces(1e-6, 45.0), chain);
  REQUIRE(mix);
  CHECK(mix->held);
  CHECK(mix->options == Options{gmfss_mix::kF32, gmfss_mix::kF32});
  CHECK(mix->db == Catch::Approx(60.0));
  CHECK(chain.calls == 7);
}

TEST_CASE("Mélange natif : une erreur de la chaîne remonte", "[gpu][native_mix]") {
  for (const int at : {0, 6, 7}) {
    Simulated chain{.effect = effects(), .fail_at = at};
    const auto mix = native_mix::choose(two_pieces(1e-6, 45.0), chain);
    REQUIRE_FALSE(mix);
    CHECK(mix.error() == "panne");
  }
}

TEST_CASE("Mélange natif : RIFE, la base à TensorRT quand aucun réseau n'est plus rapide",
          "[gpu][native_mix]") {
  const native_mix::Problem problem{.times = {{1.0, 1.2}},
                                    .cands = {{0}},
                                    .base = {0},
                                    .seeds = {},
                                    .peak = {1.0},
                                    .floor = {0.0},
                                    .min_db = 45.0};
  int calls = 0;
  const auto mix = native_mix::choose(problem, [&](const Options&) -> std::expected<Eval, std::string> {
    ++calls;
    return Eval{.want = {0.0}, .base = {0.0}};
  });
  REQUIRE(mix);
  CHECK(mix->held);
  CHECK(mix->options == Options{0});
  CHECK(mix->db == Catch::Approx(200.0));
  CHECK(calls == 0);
}

TEST_CASE("Banc du réglage natif : la tolérance suit l'ordre des sommes", "[gpu][native_mix]") {
  CHECK_FALSE(bench_differs(2.F, 2.F / 16.F, RifeNativeSums::kF16));
  CHECK(bench_differs(2.F, 2.F / 15.F, RifeNativeSums::kF16));
  CHECK_FALSE(bench_differs(2.F, 2.F / 32.F, RifeNativeSums::kMixed));
  CHECK(bench_differs(2.F, 2.F / 31.F, RifeNativeSums::kMixed));
  CHECK_FALSE(bench_differs(2.F, 2.F / 128.F, RifeNativeSums::kF32));
  CHECK(bench_differs(2.F, 2.F / 127.F, RifeNativeSums::kF32));
  // Une sortie nulle tolère l'écart d'un pic de 1e-3 ; une valeur non écrite (NaN) ou infinie ne passe pas.
  CHECK_FALSE(bench_differs(0.F, 1e-3F / 128.F, RifeNativeSums::kF32));
  CHECK(bench_differs(1.F, std::numeric_limits<float>::quiet_NaN(), RifeNativeSums::kF16));
  CHECK(bench_differs(1.F, std::numeric_limits<float>::infinity(), RifeNativeSums::kF16));
}

}  // namespace kirari::gpu::cuda
