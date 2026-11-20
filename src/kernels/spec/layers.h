// Interpolation par calques et plaque de décor (phase 4 du plan du flux NVIDIA, docs/calques.md) : là où la
// carte de la scène (scene_flow.h) dit « décor qui suit la caméra », l'image se compose à partir des sources
// déplacées par le seul mouvement de caméra, au lieu du warp de RIFE. RIFE tourne tel quel sur toute l'image
// ; ses défauts sur le décor (traits qui ondulent, aplats qui bavent au bord d'un personnage) disparaissent,
// ses dessins restent. Aucun modèle nouveau.
//
// La composition (compose_layers, sur place dans out, RGB planaire FP16 de W × H) : pour chaque pixel
// p = (x, y) de out, en FP32 :
//   1. masque des tuiles : u = (x + 0,5) / tile_w − 0,5, v = (y + 0,5) / tile_h − 0,5, bornés à
//      [0, cols − 1] × [0, rows − 1] ; g = lecture bilinéaire des bits de grid (1 : décor) en (u, v) ;
//      m = borne(2 g − 1, 0, 1). Le masque vaut 1 au cœur d'un bloc de tuiles décor et tombe à 0 au bord
//      d'une tuile qui ne l'est pas : la transition reste dans le décor, jamais sur un dessin.
//   2. chaque source i (une ou deux) est lue en q_i = p + shift_i(p), shift_i(p) =
//      (s0 + s1 x + s2 y, s3 + s4 x + s5 y), en bilinéaire ou en Catmull-Rom borné (warp.h, RifeWarp), bornée
//      à l'image ; dedans_i : q_i dans [0, W − 1] × [0, H − 1]. Si source_grid est posé (la plaque : un
//      masque à elle, sur sa propre image), m_i = masque de source_grid en q_i comme en 1., sinon 1.
//   3. w_i = weight_i · dedans_i · m_i ; S = Σ w_i. S = 0 : out inchangé. décor = Σ w_i v_i / S.
//   4. garde : deux sources toutes deux dedans, d = |Y(v_0) − Y(v_1)|, Y la luma BT.709 du RGB ; une source
//      (la plaque), d = max sur R, G, B de |v_0 − out|, car un dessin resté dans l'image clé peut avoir la
//      luma du décor qui l'a remplacé (trait violet sur ciel bleu, traits fantômes de la v0.13.0-rc.2) ;
//      sinon d = 0. k = 1 − borne((d − guard_low) / (guard_high − guard_low), 0, 1).
//      Le décor qui ne suit pas la caméra (lumière qui change, personnage de fond, plaque périmée) garde out.
//   5. out = borne(out + m k a (décor − out), 0, 1), par canal. Pour la plaque a = m_0 ; sinon a = 1.
//      Avec une seule source, la normalisation en 3 annule m_0 ; a conserve la transition de son masque.
// Un pixel ne lit que lui-même dans out : sur place sans risque. Les sources ne sont jamais out.
//
// Les déplacements viennent du mouvement de caméra de la carte (SceneMotion::camera, pixels de travail, a
// vers b) par layer_shifts (paire : a et b vers l'instant t) et plate_shift (plaque : l'image clé vers b),
// exacts pour un mouvement affine. Référence : ref::compose_layers ; port :
// gpu/cuda/backend/layer_kernels.cu.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace kirari::kernels {

// La grille des tuiles décor : au plus kLayerMaxCols × kLayerMaxRows tuiles (une image de travail de 1080p en
// tuiles de 16 pixels en tient 120 × 68), un bit par tuile, ligne par ligne depuis le haut gauche. Passée par
// valeur au noyau : 1,5 Ko.
inline constexpr int kLayerMaxCols = 128;
inline constexpr int kLayerMaxRows = 96;
inline constexpr int kLayerGridWords = kLayerMaxCols * kLayerMaxRows / 32;

struct LayerGrid {
  int cols = 0;
  int rows = 0;
  float tile_w = 0.0F;  // côté d'une tuile, pixels de l'image que la grille couvre
  float tile_h = 0.0F;
  std::array<std::uint32_t, kLayerGridWords> bits{};

  constexpr bool fits(int c, int r) const {
    return c >= 1 && r >= 1 && c <= kLayerMaxCols && r <= kLayerMaxRows;
  }
  constexpr bool get(int col, int row) const {
    const int i = row * cols + col;
    return ((bits[static_cast<std::size_t>(i / 32)] >> (i % 32)) & 1U) != 0U;
  }
  constexpr void set(int col, int row) {
    const int i = row * cols + col;
    bits[static_cast<std::size_t>(i / 32)] |= 1U << (i % 32);
  }
};

// Un mouvement affine : le point (x, y) va en (c0 + c1 x + c2 y, c3 + c4 x + c5 y). Un déplacement (shift) a
// la même forme, le point lu étant alors p + shift(p).
using Affine = std::array<double, 6>;
inline constexpr Affine kIdentityAffine{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

// f ∘ g : g d'abord.
constexpr Affine compose(const Affine& f, const Affine& g) {
  return {f[0] + f[1] * g[0] + f[2] * g[3], f[1] * g[1] + f[2] * g[4], f[1] * g[2] + f[2] * g[5],
          f[3] + f[4] * g[0] + f[5] * g[3], f[4] * g[1] + f[5] * g[4], f[4] * g[2] + f[5] * g[5]};
}

// Déterminant de la partie linéaire ; inverse rien si elle est presque singulière (sous kMinAffineDet).
inline constexpr double kMinAffineDet = 0.25;
constexpr double determinant(const Affine& f) {
  return f[1] * f[5] - f[2] * f[4];
}
constexpr std::optional<Affine> inverse(const Affine& f) {
  const double det = determinant(f);
  if (!(det >= kMinAffineDet)) return std::nullopt;
  const double a = f[5] / det;
  const double b = -f[2] / det;
  const double c = -f[4] / det;
  const double d = f[1] / det;
  return Affine{-(a * f[0] + b * f[3]), a, b, -(c * f[0] + d * f[3]), c, d};
}

// Le mouvement moins l'identité : le déplacement qu'il fait subir au point.
constexpr Affine shift_of(const Affine& f) {
  return {f[0], f[1] - 1.0, f[2], f[3], f[4], f[5] - 1.0};
}

// Un mouvement en pixels de travail (w × h) porté à une image de out_w × out_h qui la couvre toute (agrandie
// par SPAN) : centres de pixels alignés, x_travail = (x_sortie + ½) · w / out_w − ½.
constexpr Affine to_output(const Affine& work, int w, int h, int out_w, int out_h) {
  const double sx = static_cast<double>(out_w) / static_cast<double>(w);
  const double sy = static_cast<double>(out_h) / static_cast<double>(h);
  // S : travail → sortie, x_o = sx x_w + (sx − 1) / 2.
  const Affine s{(sx - 1.0) / 2.0, sx, 0.0, (sy - 1.0) / 2.0, 0.0, sy};
  const Affine s_inv{-(sx - 1.0) / (2.0 * sx), 1.0 / sx, 0.0, -(sy - 1.0) / (2.0 * sy), 0.0, 1.0 / sy};
  return compose(s, compose(work, s_inv));
}

// Les déplacements d'une paire à l'instant t (0 < t < 1) : le décor de a a fait la fraction t du mouvement
// de caméra M (pixels de sortie), M_t = I + t (M − I). Le pixel p de l'image à t vient de M_t⁻¹ p dans a et
// de M M_t⁻¹ p dans b. Rien si M_t n'est pas inversible.
struct PairShifts {
  Affine a;
  Affine b;
};
constexpr std::optional<PairShifts> layer_shifts(const Affine& camera_out, double t) {
  Affine m_t{};
  for (std::size_t i = 0; i < 6; ++i) m_t[i] = kIdentityAffine[i] + t * (camera_out[i] - kIdentityAffine[i]);
  const std::optional<Affine> back = inverse(m_t);
  if (!back) return std::nullopt;
  return PairShifts{.a = shift_of(*back), .b = shift_of(compose(camera_out, *back))};
}

// La plaque : key_to_b (pixels de sortie) mène l'image clé à b ; le pixel p de b vient de key_to_b⁻¹ p.
constexpr std::optional<Affine> plate_shift(const Affine& key_to_b_out) {
  const std::optional<Affine> back = inverse(key_to_b_out);
  if (!back) return std::nullopt;
  return shift_of(*back);
}

// Garde (étape 4), en unités de [0, 1]. Paire : a et b déplacées viennent du même SPAN, seul le grain et le
// dessin les séparent. Plaque : la plaque (SPAN pleine précision, image clé) contre SPAN de b, que leurs
// modèles séparent aussi un peu sur les traits fins. Non mesurés : l'essai de la fin les règle.
inline constexpr float kLayerGuardLow = 4.0F / 255.0F;
inline constexpr float kLayerGuardHigh = 12.0F / 255.0F;
inline constexpr float kPlateGuardLow = 6.0F / 255.0F;
inline constexpr float kPlateGuardHigh = 18.0F / 255.0F;

struct LayerSource {
  std::array<float, 6> shift{};  // pixels de out
  float weight = 1.0F;
};

struct LayerParams {
  LayerGrid grid;  // sur out
  std::array<LayerSource, 2> sources{};
  int count = 1;  // 1 ou 2
  // Masque de la source 0 sur sa propre image (la plaque : les tuiles décor de l'image clé) ; cols nul :
  // aucun.
  LayerGrid source_grid;
  float guard_low = kLayerGuardLow;
  float guard_high = kLayerGuardHigh;
  bool cubic = false;
};

constexpr std::array<float, 6> to_float(const Affine& f) {
  return {static_cast<float>(f[0]), static_cast<float>(f[1]), static_cast<float>(f[2]),
          static_cast<float>(f[3]), static_cast<float>(f[4]), static_cast<float>(f[5])};
}

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
