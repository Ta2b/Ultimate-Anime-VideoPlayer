// La chaîne complète du grade HDR sur le GPU (audit A41-11) : la passe anime, la pile de LUT livrée et la
// dernière passe. Les niveaux écrits ici sont ceux des décisions 16 et 17 du 09/11/2026 (docs/hdr-anime.md).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "grade_chain_support.h"

namespace ajp::render {

using namespace gpu_test;

namespace {

// Un halo clair de 48 colonnes (195 nits) sur un fond sombre (3 nits) : un émetteur pour la passe anime.
std::vector<std::uint16_t> halo() {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (std::size_t y = 0; y < static_cast<std::size_t>(kSide); ++y) {
    for (std::size_t x = 0; x < static_cast<std::size_t>(kSide); ++x) {
      const double nits = x >= 104 && x < 152 ? 195.0 : 3.0;
      const auto code = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
      std::uint16_t* px = &pixels[(y * static_cast<std::size_t>(kSide) + x) * 4];
      px[0] = px[1] = px[2] = code;
      px[3] = 65535;
    }
  }
  return pixels;
}

}  // namespace

// La mesure du chantier C06, avant et après : `ajp_render_tests "[.mesure]"` écrit la courbe neutre de la
// chaîne et le niveau d'un émetteur, à lire en nits affichés.
TEST_CASE("mesure : courbe neutre de la chaîne HDR et niveau d'un émetteur", "[.mesure][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile stack;
  for (const double peak : {400.0, 1000.0, 2000.0}) {
    set_player_output(gpu, stack, peak);
    const GradeParams params = player_params(peak);
    std::printf("pic %.0f nits\n  code RVB | avant le grade | à l'écran\n", peak);
    for (const double code : {0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 160.0,
                              192.0, 224.0, 255.0})
      std::printf("  %8.0f | %14.3f | %9.3f\n", code, sdr_code_nits(code),
                  chain_level(gpu, sdr_code_nits(code), params));
    for (const double spec : {1.0, 2.0, 0.5}) {
      GradeParams p = params;
      p.an_spec = spec;
      for (const double apl : {20.0, 160.0}) {  // scène sombre, scène très claire
        p.fs_apl = apl;
        const auto line = graded(gpu, halo(), p);
        std::printf("  émetteur blanc, Éclats %3.0f %%, scène à %3.0f nits : %7.1f nits\n", 100.0 * spec, apl,
                    line[128]);
      }
    }
  }
}

}  // namespace ajp::render

namespace ajp::render {

using namespace gpu_test;
using Catch::Approx;

TEST_CASE("chaîne HDR : blanc des surfaces à la moitié du pic, gris moyen inchangé (décision 16)",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile stack;
  for (const double peak : {400.0, 1000.0, 2000.0}) {
    set_player_output(gpu, stack, peak);
    const GradeParams params = player_params(peak);
    const double white = chain_level(gpu, sdr_code_nits(255.0), params);
    const double grey = chain_level(gpu, sdr_code_nits(128.0), params);
    INFO("pic " << peak << " : blanc " << white << " nits, RVB 128 à " << grey << " nits");
    // Avant : 320, 690 et 1 310 nits (80, 69 et 65 % du pic).
    CHECK(white == Approx(surface_white_target(peak)).epsilon(0.03));
    // Le gris moyen ne bouge pas (43 à 45 nits avant).
    CHECK(grey > 41.0);
    CHECK(grey < 47.0);
    // Ce que la dernière passe annonce est ce qu'elle rend (MaxFALL, A41-02).
    REQUIRE(gpu.grade->output());
    CHECK(gpu.grade->output()->white == Approx(white).epsilon(0.02));
  }
  gpu.grade->set_kwin_match(std::nullopt);
}

TEST_CASE("chaîne HDR : le shader et son double en C++ rendent la même courbe neutre",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile stack;
  set_player_output(gpu, stack, 1000.0);
  const GradeParams params = player_params(1000.0);
  gpu.grade->set_params(params);  // la dernière passe se règle sur les paramètres posés
  REQUIRE(gpu.grade->output());
  const OutputTune tune = *gpu.grade->output();
  const LookToe toe = look_toe(203.0 * kAnimeLift);
  for (const double code : {8.0, 16.0, 24.0, 32.0, 40.0, 64.0, 128.0, 192.0, 255.0}) {
    const double lifted = sdr_code_nits(code) * kAnimeLift;
    const double model = compress_neutral(look_neutral(stack.chain(), toe, lifted), 203.0, tune.curve_peak);
    const double shader = chain_level(gpu, sdr_code_nits(code), params);
    INFO("code " << code << " : shader " << shader << " nits, modèle " << model << " nits");
    CHECK(shader == Approx(model).epsilon(0.03).margin(0.005));
  }
  gpu.grade->set_kwin_match(std::nullopt);
}

TEST_CASE("chaîne HDR : les éclats gardent leur écart, le limiteur et le curseur agissent (décision 16)",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile stack;
  constexpr double kScreen = 1000.0;
  set_player_output(gpu, stack, kScreen);
  GradeParams params = player_params(kScreen);
  // La surface du halo seule (195 nits partout) : sa référence à l'écran.
  const double surface = chain_level(gpu, 195.0, params);
  const auto emitter = [&](double spec, double apl) {
    GradeParams p = params;
    p.an_spec = spec;
    p.fs_apl = apl;
    return graded(gpu, halo(), p)[128];
  };
  const double dark = emitter(1.0, 20.0);
  const double bright = emitter(1.0, 180.0);
  const double more = emitter(2.0, 20.0);
  const double less = emitter(0.5, 20.0);
  INFO("surface " << surface << ", émetteur " << dark << " (scène claire " << bright << "), Éclats 200 % "
                  << more << ", 50 % " << less);
  // L'écart que la passe anime décide (0,53 diaphragme) arrive à l'écran : il n'en restait que 0,20.
  CHECK(std::log2(dark / surface) > 0.45);
  CHECK(std::log2(dark / surface) < 0.60);
  // Le limiteur de la scène baisse les éclats sans les ramener aux surfaces (il ne changeait rien).
  CHECK(bright < 0.92 * dark);
  CHECK(bright > 1.15 * surface);
  CHECK(bright <= kEmitterMinRoom * 0.5 * kScreen * 1.02);
  // Le curseur Éclats : à 200 %, près du pic sans le passer ; à 50 %, près des surfaces.
  CHECK(more > 1.3 * dark);
  CHECK(more <= kScreen);
  CHECK(less < 0.85 * dark);
  CHECK(less >= surface);
  // Le fond sombre du halo ne monte pas avec lui.
  GradeParams p = params;
  p.fs_apl = 20.0;
  CHECK(graded(gpu, halo(), p)[16] < 4.0);
  gpu.grade->set_kwin_match(std::nullopt);
}

TEST_CASE("chaîne HDR : vrai noir sous RVB 4, nuits gardées au-dessus (décision 17)",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile stack;
  set_player_output(gpu, stack, 1000.0);
  const GradeParams params = player_params(1000.0);
  CHECK(chain_level(gpu, sdr_code_nits(0.0), params) == 0.0);
  CHECK(chain_level(gpu, sdr_code_nits(4.0), params) < 1e-3);
  // RVB 16 sortait à 0,02 nit, RVB 32 à 0,74 nit.
  const double night16 = chain_level(gpu, sdr_code_nits(16.0), params);
  const double night32 = chain_level(gpu, sdr_code_nits(32.0), params);
  INFO("RVB 16 : " << night16 << " nit, RVB 32 : " << night32 << " nit");
  CHECK(night16 > 0.10);
  CHECK(night16 < sdr_code_nits(16.0) * kAnimeLift);
  CHECK(night32 > 1.0);
  CHECK(night32 < sdr_code_nits(32.0) * kAnimeLift);
  double before = 0.0;
  for (double code = 6.0; code <= 64.0; code += 2.0) {
    const double now = chain_level(gpu, sdr_code_nits(code), params);
    INFO("code " << code << " : " << now << " nit");
    CHECK(now > before);
    before = now;
  }
  gpu.grade->set_kwin_match(std::nullopt);
}

TEST_CASE("chaîne HDR : sans pile, la dernière passe ne compresse plus l'image (A41-04)",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  REQUIRE(gpu.grade->set_looks({}));
  gpu.grade->set_kwin_match(Grade::KwinMatch{.reference = 203.0, .peak = 1000.0});
  GradeParams params = player_params(1000.0);
  params.look_mixes.fill(0.0);
  // Le blanc sortait à 119 nits : il reste au relèvement des surfaces (233 nits).
  CHECK(chain_level(gpu, 203.0, params) == Approx(203.0 * kAnimeLift).epsilon(0.03));
  CHECK(chain_level(gpu, 40.0, params) == Approx(40.0 * kAnimeLift).epsilon(0.03));
  // La passe coupée, l'image ne dépend plus de l'alpha : les émetteurs montent dans la passe anime.
  gpu.grade->set_kwin_match(std::nullopt);
  CHECK_FALSE(gpu.grade->output());
  CHECK(graded(gpu, halo(), params)[128] > 1.3 * 195.0 * kAnimeLift);
}

TEST_CASE("sortie SDR de l'anime : le vrai noir OLED s'applique (A41-07)", "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  REQUIRE(gpu.grade->set_looks({}));
  GradeParams params = player_params(1000.0);
  sdr_tonemap(params);
  params.sdr_gamma = 2.2;
  const std::vector<std::uint16_t> source = sdr_ramp(0.25, 1.0, 1.0);  // codes de 0 à 0,25
  const std::vector<Rgb> plain = graded_codes(gpu, source, params);
  // Les seuils, en nits de l'image PQ que la dernière passe ramène au SDR : codes 0,02 et 0,06.
  params.black_lo = params.tm_white * std::pow(0.02, 2.2);
  params.black_hi = params.tm_white * std::pow(0.06, 2.2);
  const std::vector<Rgb> black = graded_codes(gpu, source, params);
  const auto code = [](std::size_t x) { return 0.25 * static_cast<double>(x + 1) / kSide; };
  for (std::size_t x = 0; x < black.size(); ++x) {
    if (code(x) < 0.015) CHECK(black[x][0] < 1e-3);
    if (code(x) > 0.09) CHECK(std::abs(black[x][0] - plain[x][0]) < 4e-3);
  }
}

// A41-01 : la pile réduite à une table s'écartait de la pile parcourue LUT par LUT. La mesure donne l'écart
// de l'ancienne table en codes PQ 10 bits, sur les couleurs que le lecteur lui présente : une image SDR
// BT.709 (grille de 48³ codes, gamma 2,4) relevée à 233 nits de blanc et mise en PQ BT.2020.
TEST_CASE("mesure : écart de la table composée à la pile exacte", "[.mesure]") {
  const Pile pile;
  constexpr std::array<std::array<double, 3>, 3> kTo2020{{{0.6274040, 0.3292820, 0.0433136},
                                                          {0.0690970, 0.9195400, 0.0113612},
                                                          {0.0163916, 0.0880132, 0.8955950}}};
  for (const int size : {65, 129}) {
    const CubeLut table = compose_luts(pile.luts, pile.mixes, size);
    std::vector<double> gaps;
    constexpr int kSteps = 48;
    for (int b = 0; b < kSteps; ++b)
      for (int g = 0; g < kSteps; ++g)
        for (int r = 0; r < kSteps; ++r) {
          std::array<double, 3> lin{};
          const std::array<int, 3> rgb{r, g, b};
          for (std::size_t ch = 0; ch < 3; ++ch)
            lin[ch] = std::pow((rgb[ch] + 0.5) / kSteps, 2.4) * 203.0 * kAnimeLift;
          std::array<float, 3> c{};
          for (std::size_t ch = 0; ch < 3; ++ch)
            c[ch] = static_cast<float>(
                pq::from_nits(kTo2020[ch][0] * lin[0] + kTo2020[ch][1] * lin[1] + kTo2020[ch][2] * lin[2]));
          const std::array<float, 3> exact = sample_chain(pile.chain(), c);
          const std::array<float, 3> approx = sample_lut(table, c);
          for (std::size_t ch = 0; ch < 3; ++ch)
            gaps.push_back(1023.0 * std::abs(static_cast<double>(exact[ch]) - approx[ch]));
        }
    std::ranges::sort(gaps);
    double sum = 0.0;
    for (const double gap : gaps) sum += gap;
    std::printf("table de %3d points : écart maximal %5.1f codes PQ 10 bits, 99,9 %% sous %.1f, moyen %.2f\n",
                size, gaps.back(), gaps[gaps.size() * 999 / 1000], sum / static_cast<double>(gaps.size()));
  }
}

TEST_CASE("look : le shader parcourt la pile LUT par LUT, sans table composée (A41-01)",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile pile;
  REQUIRE(gpu.grade->set_looks(pile.luts));
  const CubeLut table = compose_luts(pile.luts, pile.mixes);  // ce que le hook lisait avant
  // Sortie SDR neutre, pile lue comme une LUT SDR (look_pq à 0) : toutes les couleurs y passent.
  GradeParams params = sdr_neutral();
  params.look_mixes = pile.mixes;
  for (const auto& [green, blue] : {std::pair{0.8, 0.6}, std::pair{0.35, 0.9}, std::pair{1.0, 1.0}}) {
    const std::vector<std::uint16_t> source = sdr_ramp(1.0, green, blue);
    const std::vector<Rgb> out = graded_codes(gpu, source, params);
    const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
    double from_chain = 0.0;  // écart moyen du rendu à la pile parcourue, puis à l'ancienne table
    double from_table = 0.0;
    for (std::size_t x = 8; x < out.size(); ++x) {
      // Le chemin d'une LUT SDR dans le hook : l'image en linéaire (blanc de la LUT à 1, ce qui dépasse mis
      // de côté), encodée au gamma 2,4, passée par la pile, puis ramenée au code de la sortie.
      std::array<double, 3> lin{};
      for (std::size_t c = 0; c < 3; ++c)
        lin[c] =
            std::pow(source[(row + x) * 4 + c] / 65535.0, params.sdr_gamma) * params.peak / params.look_white;
      const double over = std::max({lin[0], lin[1], lin[2], 1.0});
      std::array<float, 3> in{};
      for (std::size_t c = 0; c < 3; ++c) in[c] = static_cast<float>(std::pow(lin[c] / over, 1.0 / 2.4));
      const auto code = [&](float looked) {
        const double back =
            std::pow(static_cast<double>(looked), 2.4) * over * params.look_white / params.peak;
        return std::pow(std::clamp(back, 0.0, 1.0), 1.0 / params.sdr_gamma);
      };
      const std::array<float, 3> exact = sample_chain(pile.chain(), in);
      const std::array<float, 3> approx = sample_lut(table, in);
      for (std::size_t c = 0; c < 3; ++c) {
        from_chain += std::abs(out[x][c] - code(exact[c]));
        from_table += std::abs(out[x][c] - code(approx[c]));
      }
    }
    const double count = 3.0 * static_cast<double>(out.size() - 8);
    INFO("vert " << green << ", bleu " << blue << " : écart moyen à la pile " << from_chain / count
                 << ", à la table de 65 points " << from_table / count);
    // Deux codes sur 1 023 : onze lectures trilinéaires du GPU, en demi-flottants.
    CHECK(from_chain / count < 2e-3);
    CHECK(from_chain < from_table);
  }
  // Une intensité change sans relire le hook : le //!PARAM suit, l'image aussi.
  const pl_hook* before = gpu.grade->hooks()[1];
  params.look_mixes[1] = 0.25;
  gpu.grade->set_params(params);
  CHECK(gpu.grade->hooks()[1] == before);
}

}  // namespace ajp::render
