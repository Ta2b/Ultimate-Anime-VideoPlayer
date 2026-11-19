// La forme en flux des convolutions natives (gpu/cuda/backend/conv_stream.h), sans CUDA. Le calendrier de
// l'anneau : la bande suivante se charge hors de la fenêtre lue, la sortie se range hors de ce qui reste à
// lire, chaque ligne d'entrée n'est chargée qu'une fois par colonne de bandes. Puis le noyau rejoué sur
// l'hôte, ses adresses en éléments comme en mémoire partagée (un anneau rempli de NaN au départ), ses sommes
// arrondies en FP16 dans l'ordre des noyaux : sa sortie est celle de la tuile d'une seule tranche
// (conv_kernel), au bit près, en sommes f16 et mixtes (les noyaux en flux construits), avec ou sans PReLU de
// l'entrée, résiduel et activation, sur des bords d'image, des bandes et des colonnes de bandes incomplètes.
#include "gpu/cuda/backend/conv_stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/half.h"

namespace kirari::gpu::cuda::conv_stream {
namespace {

std::size_t u(int v) {
  return static_cast<std::size_t>(v);
}

std::set<int> slots(int band, int first, int count, int th) {
  std::set<int> out;
  for (int r = first; r < first + count; ++r) out.insert(window_slot(band, r, th));
  return out;
}

bool disjoint(const std::set<int>& a, const std::set<int>& b) {
  return std::ranges::none_of(a, [&](int s) { return b.contains(s); });
}

TEST_CASE("Flux : l'anneau ne se recouvre jamais là où on le lit", "[gpu][conv_stream]") {
  for (const int rows : {1, 2}) {
    const int th = band_rows(rows);
    for (int band = 0; band < 4 * ring_rows(th); ++band) {
      const std::set<int> window = slots(band, 0, th + 2, th);
      REQUIRE(window.size() == u(th + 2));
      // Les nouvelles lignes de la bande suivante, en vol pendant les mma de celle-ci.
      const std::set<int> next = slots(band + 1, 2, th, th);
      REQUIRE(next.size() == u(th));
      REQUIRE(disjoint(window, next));
      // Les deux lignes du bord sont celles du bas de la fenêtre courante.
      REQUIRE(window_slot(band + 1, 0, th) == window_slot(band, th, th));
      REQUIRE(window_slot(band + 1, 1, th) == window_slot(band, th + 1, th));
      // La sortie se range dans les th premières lignes de la fenêtre : ni le bord de la bande suivante ni
      // ses lignes en vol ; le chargement d'après les reprend, après la barrière qui suit son écriture.
      const std::set<int> staged = slots(band, 0, th, th);
      REQUIRE(disjoint(staged, slots(band + 1, 0, th + 2, th)));
      REQUIRE(staged == slots(band + 2, 2, th, th));
    }
  }
}

TEST_CASE("Flux : mémoire partagée et sortie d'une ligne", "[gpu][conv_stream]") {
  REQUIRE(smem_bytes(64, 1) == 48960);
  REQUIRE(smem_bytes(64, 1) <= 48 * 1024);  // sans réglage de la mémoire partagée dynamique
  REQUIRE(stages_in_slot(64, 64));
  REQUIRE(stages_in_slot(64, 32));
  REQUIRE_FALSE(stages_in_slot(16, 64));
}

TEST_CASE("Flux : les colonnes de bandes couvrent chaque bande une fois", "[gpu][conv_stream]") {
  // Chaque bande une fois, quel que soit le nombre de colonnes (au banc : celui de la couche entière, sur
  // une bande de ses lignes) ; une colonne de trop n'a rien à faire.
  const auto covers = [](int tiles, int strips, int most) {
    std::vector<int> seen(u(tiles), 0);
    for (int by = 0; by < strips; ++by) {
      const Strip s = strip_of(tiles, strips, by);
      if (s.count < 0 || s.count > most) return false;
      for (int b = s.first; b < s.first + s.count; ++b) ++seen[u(b)];
    }
    return std::ranges::all_of(seen, [](int n) { return n == 1; }) &&
           strip_of(tiles, strips, strips).count == 0;
  };
  for (int tiles = 1; tiles <= 40; ++tiles) {
    for (int bands = 1; bands <= 45; ++bands) REQUIRE(covers(tiles, strips_for(tiles, bands), bands));
    for (int strips = 1; strips <= tiles; ++strips)
      REQUIRE(covers(tiles, strips, (tiles + strips - 1) / strips));
  }
  // La ligne 0 en 1080p au banc : 136 bandes des 144, quatre colonnes de 34 au lieu de 36.
  REQUIRE(strip_of(136, strips_for(144, 36), 3).count == 34);
}

TEST_CASE("Flux : les bandes à essayer", "[gpu][conv_stream]") {
  // La ligne 0 du GridNet en 1080p : 960 × 576, tuile de 64 sorties, 60 multiprocesseurs.
  REQUIRE(band_counts(30, 144, 1, 60) == std::vector<int>{36, 18, 9});
  REQUIRE(band_counts(30, 144, 2, 60) == std::vector<int>{72, 36, 18});
  // Une petite couche : deux bandes au moins, jamais plus que la couche n'en a.
  REQUIRE(band_counts(4, 3, 1, 60) == std::vector<int>{2});
  REQUIRE(band_counts(4, 1, 1, 60).empty());
}

// Le noyau rejoué : sommes comme les noyaux du RIFE natif (rife_native_kernels.cuh), un mma de 16 produits
// sommés en f32 puis ajoutés à l'accumulateur.
enum class Sums { kF16, kMixed };

float h16(float v) {
  constexpr float kMax = 65504.F;
  return kernels::ref::round_to_half(std::clamp(v, -kMax, kMax));
}

float prelu(float v, float a) {
  return v < 0.F ? v * a : v;
}

// L'élément (y, x, c) d'un tableau de lignes de w pixels à ch canaux.
std::size_t nhwc(int y, int x, int c, int w, int ch) {
  return (u(y) * u(w) + u(x)) * u(ch) + u(c);
}

// Une image de 21 × 36 : six bandes de 4 lignes, la dernière d'une ligne ; deux colonnes de 32 pixels, la
// seconde de 4.
struct Problem {
  int cin = 64, cout = 64, ct = 64, th = 4;
  int ih = 21, iw = 36;
  float in_slope = 1.F;
  bool skip = false, activation = false;
  Sums sums = Sums::kF16;
  std::vector<float> in, w, bias, residual, slope;  // in [ih][iw][cin], w [cout][9][cin]
};

Problem make(std::mt19937& rng, Problem p) {
  std::uniform_real_distribution<float> unit(-1.F, 1.F);
  const auto fill = [&](std::vector<float>& v, std::size_t n, float scale) {
    v.resize(n);
    for (float& x : v) x = h16(unit(rng) * scale);
  };
  const std::size_t pixels = u(p.ih) * u(p.iw);
  fill(p.in, pixels * u(p.cin), 1.F);
  fill(p.w, u(p.cout) * 9 * u(p.cin), 0.1F);
  fill(p.bias, u(p.cout), 0.5F);
  fill(p.residual, pixels * u(p.cout), 1.F);
  p.slope.assign(u(p.cout), 0.2F);
  return p;
}

// La somme d'un pixel et d'une sortie, prise après prise, par pas de 16 canaux : read(tap, c) l'entrée.
template <class Read>
float sum_of(const Problem& p, int n, Read read) {
  float acc = 0.F;
  for (int tap = 0; tap < 9; ++tap) {
    float part = 0.F;
    for (int c0 = 0; c0 < p.cin; c0 += 16) {
      float d = 0.F;
      for (int c = c0; c < c0 + 16; ++c) d += read(tap, c) * p.w[nhwc(n, tap, c, 9, p.cin)];
      if (p.sums == Sums::kF16)
        acc = h16(acc + d);
      else
        part = h16(part + d);
    }
    acc += part;
  }
  return acc;
}

// stage puis finish8 : biais et arrondi, puis résiduel et activation, arrondis.
float finish(const Problem& p, float acc, int y, int x, int n) {
  float v = h16(acc + p.bias[u(n)]);
  if (!p.skip && !p.activation) return v;
  if (p.skip) v += p.residual[nhwc(y, x, n, p.iw, p.cout)];
  if (p.activation) v = prelu(v, p.slope[u(n)]);
  return h16(v);
}

float input_at(const Problem& p, int y, int x, int c) {
  if (y < 0 || y >= p.ih || x < 0 || x >= p.iw) return 0.F;
  return h16(prelu(p.in[nhwc(y, x, c, p.iw, p.cin)], p.in_slope));
}

// La tuile d'une seule tranche : l'entrée lue autour de chaque pixel.
std::vector<float> tile_conv(const Problem& p) {
  std::vector<float> out(u(p.ih) * u(p.iw) * u(p.cout));
  for (int y = 0; y < p.ih; ++y)
    for (int x = 0; x < p.iw; ++x)
      for (int n = 0; n < p.cout; ++n) {
        const float acc =
            sum_of(p, n, [&](int tap, int c) { return input_at(p, y + tap / 3 - 1, x + tap % 3 - 1, c); });
        out[nhwc(y, x, n, p.iw, p.cout)] = finish(p, acc, y, x, n);
      }
  return out;
}

// Un bloc (bx, by, z) du noyau en flux sur strips colonnes de bandes, ses étapes dans l'ordre du noyau ; les
// lignes en vol s'écrivent dès leur envoi, le pire moment pour la bande en cours. loads compte les lignes
// d'entrée chargées par la colonne.
struct StreamBlock {
  const Problem& p;
  std::vector<float> ring;
  int ld, ldo;
  explicit StreamBlock(const Problem& problem)
      : p(problem),
        ring(u(ring_rows(problem.th)) * u(kCols) * u(problem.cin + 8),
             std::numeric_limits<float>::quiet_NaN()),
        ld(problem.cin + 8),
        ldo(problem.ct + 8) {}

  // L'élément c du pixel col de la ligne row de la fenêtre ; la sortie (stage) au pas ct + 8.
  std::size_t at(int band, int row, int col, int c = 0) const {
    return nhwc(window_slot(band, row, p.th), col, c, kCols, ld);
  }
  std::size_t staged(int band, int r, int col, int n) const {
    return at(band, r, 0) + u(col) * u(ldo) + u(n);
  }
  void load(int band, int first, int count, int y_top, int x0, std::vector<int>& loads) {
    for (int r = first; r < first + count; ++r) {
      const int y = y_top + band * p.th - 1 + r;
      if (y >= 0 && y < p.ih) ++loads[u(y)];
      for (int col = 0; col < kCols; ++col)
        for (int c = 0; c < p.cin; ++c) {
          const int x = x0 + col;
          const bool ok = y >= 0 && y < p.ih && x >= 0 && x < p.iw;
          ring[at(band, r, col, c)] = ok ? p.in[nhwc(y, x, c, p.iw, p.cin)] : 0.F;
        }
    }
  }
  void prelu_rows(int band, int first, int count) {
    for (int r = first; r < first + count; ++r)
      for (int col = 0; col < kCols; ++col)
        for (int c = 0; c < p.cin; ++c) {
          float& v = ring[at(band, r, col, c)];
          v = h16(prelu(v, p.in_slope));
        }
  }
  void run(int bx, int by, int strips, int z, std::vector<float>& out, std::vector<int>& loads) {
    const Strip strip = strip_of((p.ih + p.th - 1) / p.th, strips, by);
    if (strip.count <= 0) return;
    const int y_top = strip.first * p.th, tx0 = bx * kTW, n0 = z * p.ct;
    load(0, 0, p.th + 2, y_top, tx0 - 1, loads);
    for (int band = 0; band < strip.count; ++band) {
      if (band + 1 < strip.count) load(band + 1, 2, p.th, y_top, tx0 - 1, loads);
      if (p.in_slope != 1.F) prelu_rows(band, band == 0 ? 0 : 2, band == 0 ? p.th + 2 : p.th);
      // Les sommes des pixels de l'image seulement : les autres ne s'écrivent pas.
      std::vector<float> acc(u(p.th) * u(kTW) * u(p.ct));
      for (int r = 0; r < p.th && y_top + band * p.th + r < p.ih; ++r)
        for (int col = 0; col < kTW && tx0 + col < p.iw; ++col)
          for (int n = 0; n < p.ct; ++n)
            acc[nhwc(r, col, n, kTW, p.ct)] = sum_of(
                p, n0 + n, [&](int tap, int c) { return ring[at(band, r + tap / 3, col + tap % 3, c)]; });
      // stage : la ligne r de la sortie dans la ligne r de la fenêtre, au pas ct + 8.
      for (int r = 0; r < p.th; ++r)
        for (int col = 0; col < kTW; ++col)
          for (int n = 0; n < p.ct; ++n)
            ring[staged(band, r, col, n)] = h16(acc[nhwc(r, col, n, kTW, p.ct)] + p.bias[u(n0 + n)]);
      store(band, y_top + band * p.th, tx0, n0, out);
    }
  }
  void store(int band, int ty0, int tx0, int n0, std::vector<float>& out) const {
    for (int r = 0; r < p.th; ++r)
      for (int col = 0; col < kTW; ++col)
        for (int n = 0; n < p.ct; ++n) {
          const int y = ty0 + r, x = tx0 + col;
          if (y >= p.ih || x >= p.iw) continue;
          float v = ring[staged(band, r, col, n)];
          if (p.skip || p.activation) {
            if (p.skip) v += p.residual[nhwc(y, x, n0 + n, p.iw, p.cout)];
            if (p.activation) v = prelu(v, p.slope[u(n0 + n)]);
            v = h16(v);
          }
          out[nhwc(y, x, n0 + n, p.iw, p.cout)] = v;
        }
  }
};

std::vector<float> stream_conv(const Problem& p, int strips, std::vector<int>& most_loads) {
  std::vector<float> out(u(p.ih) * u(p.iw) * u(p.cout), std::numeric_limits<float>::quiet_NaN());
  most_loads.assign(u(p.ih), 0);
  for (int z = 0; z < p.cout / p.ct; ++z)
    for (int bx = 0; bx < (p.iw + kTW - 1) / kTW; ++bx)
      for (int by = 0; by < strips; ++by) {
        std::vector<int> loads(u(p.ih), 0);
        StreamBlock(p).run(bx, by, strips, z, out, loads);
        for (std::size_t y = 0; y < loads.size(); ++y) most_loads[y] = std::max(most_loads[y], loads[y]);
      }
  return out;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::ranges::equal(a, b, [](float x, float y) { return x == y && !std::isnan(x); });
}

TEST_CASE("Flux : le noyau rejoué rend la tuile d'une seule tranche au bit près", "[gpu][conv_stream]") {
  std::mt19937 rng(7919U);
  struct Case {
    Sums sums;
    int cout, ct;
    float in_slope;
    bool skip, activation;
  };
  // Les trois noyaux construits : tuile de 64 en f16, de 32 en f16 (deux tuiles) et en mixtes.
  const std::vector<Case> cases{{Sums::kF16, 64, 64, 0.25F, true, false},
                                {Sums::kF16, 64, 32, 1.F, false, true},
                                {Sums::kMixed, 32, 32, 0.25F, true, true}};
  for (const Case& c : cases) {
    const Problem p = make(rng, Problem{.cout = c.cout,
                                        .ct = c.ct,
                                        .in_slope = c.in_slope,
                                        .skip = c.skip,
                                        .activation = c.activation,
                                        .sums = c.sums});
    const std::vector<float> want = tile_conv(p);
    // Six bandes : une par colonne, trois colonnes de deux (et une de trop), deux de trois, une de six.
    for (const int strips : {6, 4, 2, 1}) {
      std::vector<int> loads;
      const std::vector<float> got = stream_conv(p, strips, loads);
      INFO("sommes " << static_cast<int>(c.sums) << ", tuile " << c.ct << ", " << strips << " colonnes");
      REQUIRE(bitwise_equal(got, want));
      REQUIRE(std::ranges::all_of(loads, [](int n) { return n == 1; }));
    }
  }
}

}  // namespace
}  // namespace kirari::gpu::cuda::conv_stream
