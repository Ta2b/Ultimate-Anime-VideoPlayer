// Le look en une table (render/look_table.h) face à la pile parcourue LUT par LUT, sur le GPU.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "grade_chain_support.h"
#include "render/grade_params.h"

namespace {

using namespace ajp::render;
using namespace ajp::render::gpu_test;

// Le jeu de couleurs de la mesure A41-01 (#669) : une image SDR BT.709 (grille de 48³ codes, gamma 2,4)
// relevée à 233 nits de blanc et mise en PQ BT.2020, ici en images de kSide² pixels, la dernière complétée
// par sa dernière couleur.
constexpr int kSteps = 48;
constexpr std::size_t kColors = static_cast<std::size_t>(kSteps) * kSteps * kSteps;
constexpr std::size_t kPixels = static_cast<std::size_t>(kSide) * kSide;

std::vector<std::uint16_t> colour_set(std::size_t first) {
  std::vector<std::uint16_t> px(kPixels * 4);
  for (std::size_t i = 0; i < kPixels; ++i) {
    const std::size_t n = std::min(first + i, kColors - 1);
    const std::array<std::size_t, 3> rgb{n % kSteps, (n / kSteps) % kSteps, n / (kSteps * kSteps)};
    std::array<double, 3> lin{};
    for (std::size_t c = 0; c < 3; ++c)
      lin[c] = std::pow((static_cast<double>(rgb[c]) + 0.5) / kSteps, 2.4) * 203.0 * kAnimeLift;
    const std::array<double, 3> pq{
        nits_to_pq((0.6274040 * lin[0]) + (0.3292820 * lin[1]) + (0.0433136 * lin[2])),
        nits_to_pq((0.0690970 * lin[0]) + (0.9195400 * lin[1]) + (0.0113612 * lin[2])),
        nits_to_pq((0.0163916 * lin[0]) + (0.0880132 * lin[1]) + (0.8955950 * lin[2]))};
    for (std::size_t c = 0; c < 3; ++c)
      px[4 * i + c] = static_cast<std::uint16_t>(std::lround(pq[c] * 65535.0));
    px[4 * i + 3] = 65535;
  }
  return px;
}

struct Gaps {
  double max = 0.0;
  double p999 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  std::size_t over_one = 0;  // couleurs à plus d'un code
};

// L'écart, en codes PQ 10 bits, du look en table au look exact sur tout le jeu (le plus grand des trois
// canaux par couleur).
Gaps table_gaps(Gpu& gpu, GradeParams params, bool look_only) {
  std::vector<double> gaps;
  for (std::size_t first = 0; first < kColors; first += kPixels) {
    const std::vector<std::uint16_t> source = colour_set(first);
    params.look_table = 0.0;
    const std::vector<std::uint16_t> exact = graded_image(gpu, source, params, look_only);
    params.look_table = 1.0;
    const std::vector<std::uint16_t> table = graded_image(gpu, source, params, look_only);
    REQUIRE(gpu.grade->look_table_active());
    for (std::size_t i = 0; i < kPixels && first + i < kColors; ++i) {
      double gap = 0.0;
      for (std::size_t c = 0; c < 3; ++c)
        gap = std::max(gap, 1023.0 / 65535.0 * std::abs(double(exact[4 * i + c]) - double(table[4 * i + c])));
      gaps.push_back(gap);
    }
  }
  std::ranges::sort(gaps);
  Gaps out;
  out.max = gaps.back();
  out.p999 = gaps[gaps.size() * 999 / 1000];
  out.p99 = gaps[gaps.size() * 99 / 100];
  for (const double gap : gaps) out.mean += gap / static_cast<double>(gaps.size());
  out.over_one = static_cast<std::size_t>(std::ranges::count_if(gaps, [](double g) { return g > 1.0; }));
  return out;
}

void print(const char* what, const Gaps& g) {
  std::printf(
      "%s : écart maximal %.2f codes PQ 10 bits, 99,9 %% sous %.2f, 99 %% sous %.2f, moyen %.3f ; "
      "%zu couleurs sur %zu à plus d'un code\n",
      what, g.max, g.p999, g.p99, g.mean, g.over_one, kColors);
}

TEST_CASE("look en table : la pile livrée à 129 points, face à la pile LUT par LUT", "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile pile(0, 0, true);  // les 19 emplacements du lecteur
  REQUIRE(gpu.grade->set_looks(pile.luts));
  GradeParams params = player_params(1000.0);
  params.look_mixes = pile.mixes;
  params.look_clean = 0.0;  // le look d'un pixel seul : la table et la pile lisent la même couleur
  const Gaps look = table_gaps(gpu, params, true);
  print("look seul (pied adouci compris)", look);
  // Ce que la mesure a donné sur lavapipe (docs/hdr-anime.md) : les bornes tiennent l'approximation.
  CHECK(look.max < 10.0);
  CHECK(look.mean < 0.5);
  // Une table pour toutes ces images : construite une fois.
  CHECK(gpu.grade->look_table_builds() == 1);
}

TEST_CASE("look en table : refaite au changement d'intensité, sans relire le hook", "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile pile(0, 0, true);
  REQUIRE(gpu.grade->set_looks(pile.luts));
  GradeParams params = player_params(1000.0);
  params.look_mixes = pile.mixes;
  params.look_clean = 0.0;
  params.look_table = 1.0;
  const std::vector<std::uint16_t> source = colour_set(40000);
  const std::vector<std::uint16_t> first = graded_image(gpu, source, params, true);
  CHECK(gpu.grade->look_table_builds() == 1);
  // Le blanc des surfaces de la dernière passe ne touche pas la pile : la table reste.
  params.surface_white = 0.8;
  (void)graded_image(gpu, source, params, true);
  CHECK(gpu.grade->look_table_builds() == 1);
  // Le contraste à mi-course : la table est refaite, et suit la pile exacte.
  params.look_mixes[12] = 0.5;
  const std::vector<std::uint16_t> table = graded_image(gpu, source, params, true);
  CHECK(gpu.grade->look_table_builds() == 2);
  CHECK(table != first);
  params.look_table = 0.0;
  const std::vector<std::uint16_t> exact = graded_image(gpu, source, params, true);
  CHECK_FALSE(gpu.grade->look_table_active());
  double sum = 0.0;
  for (std::size_t i = 0; i < exact.size(); i += 4)
    for (std::size_t c = 0; c < 3; ++c) sum += std::abs(double(exact[i + c]) - double(table[i + c]));
  CHECK(sum / static_cast<double>(kPixels * 3) * 1023.0 / 65535.0 < 0.5);
}

TEST_CASE("look en table : éteinte, le hook exact au bit près", "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile pile(0, 0, true);
  REQUIRE(gpu.grade->set_looks(pile.luts));
  GradeParams params = player_params(1000.0);
  params.look_mixes = pile.mixes;
  const std::vector<std::uint16_t> source = colour_set(0);
  const std::vector<std::uint16_t> before = graded_image(gpu, source, params);
  params.look_table = 1.0;
  (void)graded_image(gpu, source, params);
  params.look_table = 0.0;
  CHECK(graded_image(gpu, source, params) == before);
  CHECK_FALSE(gpu.grade->look_table_active());
}

// La chaîne du lecteur entière (passe anime, look avec le voisinage de --clean-looks, dernière passe) :
// l'écart que l'œil verrait, à mesurer sans borne.
TEST_CASE("mesure : la chaîne entière avec le look en table", "[.mesure]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile pile(0, 0, true);
  set_player_output(gpu, pile, 1000.0);
  GradeParams params = player_params(1000.0);
  params.look_mixes = pile.mixes;
  print("chaîne entière", table_gaps(gpu, params, false));
}

}  // namespace
