// Données que l'hôte prépare pour les noyaux : tables de poids du rééchantillonnage séparable et
// paramètres d'écran des fonctions de transfert, calculés depuis kernels/spec.h. Les noyaux ne
// les recalculent pas : les tables sont faites en double, puis rangées en float, une fois.
#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "core/types.h"
#include "kernels/spec/resample.h"

namespace ajp::gpu::vk {

// Table de dst sorties sur src entrées : start[i] premier indice source (avant le miroir),
// w[i × taps + j] poids normalisés. Même contenu, au bit près, que kernels/ref compute_weights.
struct WeightTable {
  std::uint32_t taps = 0;
  std::uint32_t src = 0;
  std::uint32_t dst = 0;
  std::vector<std::int32_t> start;
  std::vector<float> w;
};

// Refus : InputMismatch (dimensions hors de [1, kMaxDimension], Area qui agrandirait),
// FieldOutOfRange (décalage non fini ou hors de [−1, 1], Area décalé).
std::expected<WeightTable, Refusal> compute_weights(int src, int dst, double shift, kernels::Filter filter);

// Écran HLG : pic Lw dans [400, 2000] cd/m², γ = 1,2 + 0,42 log10(Lw / 1000).
struct HlgDisplay {
  float peak = 1000.0F;
  float gamma = 1.2F;
};
std::expected<HlgDisplay, Refusal> hlg_display(float peak_nits);  // refus FieldOutOfRange

// Écran BT.1886 : blanc Lw > noir Lb ≥ 0, a et b précalculés (spec/transfer.h).
struct Bt1886Display {
  float white = 1.0F;
  float black = 0.0F;
  float a = 1.0F;
  float b = 0.0F;
};
std::expected<Bt1886Display, Refusal> bt1886_display(float white, float black);  // refus FieldOutOfRange

}  // namespace ajp::gpu::vk
