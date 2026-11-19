// Les poids du RIFE natif (rife_native_plan.h) : lus par prepare::RifeNativeLayer en flottants, le bêta d'un
// ResConv replié, arrondis en FP16 puis rangés dans l'ordre des fragments B du mma pour une tuile de sorties.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::rife_native {
namespace {

constexpr int pad16(int c) {
  return (c + 15) / 16 * 16;
}

// Les poids et le biais d'une couche, le facteur du résiduel replié : w[o] · s, b[o] · s ; zéros dans le
// rembourrage des canaux.
struct Folded {
  const RifeNativeConv& conv;
  float factor(int o) const { return conv.scale.empty() ? 1.F : conv.scale[static_cast<std::size_t>(o)]; }
  float weight(int o, int tap, int c) const {
    if (o >= conv.cout || c >= conv.cin) return 0.F;
    const int taps = conv.transposed ? 16 : 9;
    return conv.weights[(static_cast<std::size_t>(o) * taps + tap) * conv.cin + c] * factor(o);
  }
  float bias(int o) const {
    return o >= conv.cout ? 0.F : conv.bias[static_cast<std::size_t>(o)] * factor(o);
  }
};

}  // namespace

Dense dense_weights(const RifeNativeConv& conv, Kind kind, int cin) {
  const Folded f{conv};
  Dense d;
  d.cin = cin;
  if (kind == Kind::kPlanes4) {
    // La transposée de l'encodeur : 16 sorties j = phase · 4 + c sur une fenêtre 3 × 3 de l'entrée. La prise
    // (dy, dx) sert la phase (py, px) si dy ∈ {py - 1, py}, au poids (3 - py - 2 ty, 3 - px - 2 tx),
    // ty = dy - py + 1.
    d.cout = 16;
    d.w.resize(static_cast<std::size_t>(16) * 9 * static_cast<std::size_t>(cin));
    for (int j = 0; j < 16; ++j) {
      for (int tap = 0; tap < 9; ++tap) {
        const int py = j >> 3, px = (j >> 2) & 1;
        const int ty = tap / 3 - py, tx = tap % 3 - px;
        const bool used = ty >= 0 && ty <= 1 && tx >= 0 && tx <= 1;
        for (int c = 0; c < cin; ++c) {
          const float v = used ? f.weight(j & 3, (3 - py - 2 * ty) * 4 + (3 - px - 2 * tx), c) : 0.F;
          d.w[(static_cast<std::size_t>(j) * 9 + static_cast<std::size_t>(tap)) *
                  static_cast<std::size_t>(cin) +
              static_cast<std::size_t>(c)] = __float2half_rn(v);
        }
      }
      d.bias.push_back(f.bias(j & 3));
    }
    return d;
  }
  // Une convolution 3 × 3, ou une transposée de bloc en quatre phases 2 × 2 (prises (ty, tx) aux décalages
  // py - 1 + ty, poids (3 - py - 2 ty, 3 - px - 2 tx)).
  const bool phase = kind == Kind::kPhase;
  d.phases = phase ? 4 : 1;
  d.taps = phase ? 4 : 9;
  d.cout = pad16(conv.cout);
  const auto per_phase =
      static_cast<std::size_t>(d.cout) * static_cast<std::size_t>(d.taps) * static_cast<std::size_t>(cin);
  d.w.resize(per_phase * static_cast<std::size_t>(d.phases));
  for (int ph = 0; ph < d.phases; ++ph) {
    const int py = ph >> 1, px = ph & 1;
    __half* to = d.w.data() + per_phase * static_cast<std::size_t>(ph);
    for (int n = 0; n < d.cout; ++n)
      for (int tap = 0; tap < d.taps; ++tap) {
        const int from = phase ? (3 - py - 2 * (tap >> 1)) * 4 + (3 - px - 2 * (tap & 1)) : tap;
        for (int c = 0; c < cin; ++c)
          to[(static_cast<std::size_t>(n) * static_cast<std::size_t>(d.taps) +
              static_cast<std::size_t>(tap)) *
                 static_cast<std::size_t>(cin) +
             static_cast<std::size_t>(c)] = __float2half_rn(f.weight(n, from, c));
      }
    for (int o = 0; o < d.cout; ++o) d.bias.push_back(f.bias(o));
  }
  return d;
}

// Rangés dans l'ordre des fragments B de mma m16n8k16 : pas k (prise, tranche de 16 entrées), paire de tuiles
// n8, lane (g = lane / 4, t = lane % 4) ; 4 mots par lane : b0 et b1 de la tuile 2·np, puis de la tuile
// 2·np + 1. b0 = entrées 2t et 2t + 1 de la sortie 8·tuile + g, b1 = entrées 2t + 8 et 2t + 9 ; la plus
// petite entrée dans la moitié basse du mot.
std::vector<std::uint32_t> packed_weights(const Dense& dense, int ct) {
  const int kcs = dense.cin / 16, nps = ct / 16, taps = dense.taps, cin = dense.cin;
  const auto per_phase =
      static_cast<std::size_t>(dense.cout) * static_cast<std::size_t>(taps) * static_cast<std::size_t>(cin);
  std::vector<std::uint32_t> out;
  out.reserve(dense.w.size() / 2);
  for (int ph = 0; ph < dense.phases; ++ph)
    for (int tile = 0; tile < dense.cout / ct; ++tile) {
      const __half* w = dense.w.data() + per_phase * static_cast<std::size_t>(ph);
      const auto bits = [&](int n, int tap, int c) -> std::uint32_t {
        const std::size_t at = (static_cast<std::size_t>(tile * ct + n) * static_cast<std::size_t>(taps) +
                                static_cast<std::size_t>(tap)) *
                                   static_cast<std::size_t>(cin) +
                               static_cast<std::size_t>(c);
        return __half_as_ushort(w[at]);
      };
      for (int ks = 0; ks < taps * kcs; ++ks) {
        const int tap = ks / kcs, kc = ks % kcs;
        for (int np = 0; np < nps; ++np)
          for (int lane = 0; lane < 32; ++lane) {
            const int g = lane >> 2, t = lane & 3, k0 = kc * 16 + 2 * t;
            for (int j = 0; j < 2; ++j) {
              const int n = (2 * np + j) * 8 + g;
              out.push_back(bits(n, tap, k0) | (bits(n, tap, k0 + 1) << 16U));
              out.push_back(bits(n, tap, k0 + 8) | (bits(n, tap, k0 + 9) << 16U));
            }
          }
      }
    }
  return out;
}

}  // namespace kirari::gpu::cuda::rife_native
