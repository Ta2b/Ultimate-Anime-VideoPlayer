// Les sommes des convolutions natives (RIFE natif, docs/rife-natif.md ; GMFSS natif, docs/gmfss-natif.md) et
// ce que le banc du réglage tolère de l'une à l'autre. En-tête sans CUDA : les tests sur hôte le lisent.
#pragma once

#include <algorithm>
#include <cstdint>

namespace kirari::gpu::cuda {

// Les sommes des convolutions : f16 (le débit plein des tensor cores des GeForce), mixtes (chaque prise en
// f16, les prises entre elles en f32, au débit du f16), f32 (moitié du débit sur GeForce).
enum class RifeNativeSums : std::uint8_t { kF16, kMixed, kF32 };
// Les sommes des couches d'un morceau : la tête (la dernière couche d'un bloc de RIFE, l'encodage de son
// encodeur ; une couche qui écrit une sortie d'un réseau de GMFSS) peut sommer plus finement que les autres.
struct RifeNativePrecision {
  RifeNativeSums body = RifeNativeSums::kF16;
  RifeNativeSums head = RifeNativeSums::kF16;
};

// Le banc du réglage compare chaque forme d'une couche à la forme de référence ; l'ordre des sommes change de
// l'une à l'autre. Il tolère, en fraction du pic de la sortie de référence, ce que cet ordre explique : un
// arrondi f16 par terme au plus en f16 (1/16), un par prise en mixte (1/32), le seul arrondi de la sortie en
// f32 (1/128). Un défaut de rangement ou de lecture s'écarte de l'ordre du pic.
constexpr float bench_tolerance(RifeNativeSums sums) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return 1.F / 16.F;
    case RifeNativeSums::kMixed:
      return 1.F / 32.F;
    case RifeNativeSums::kF32:
      return 1.F / 128.F;
  }
  return 0.F;
}

// Une forme dont le plus grand écart à la référence est diff, pour un pic peak (au moins 1e-3), s'en
// écarte-t-elle trop ? Un écart non fini (NaN d'une sortie non écrite, infini) aussi.
constexpr bool bench_differs(float peak, float diff, RifeNativeSums sums) {
  return !(diff <= std::max(peak, 1e-3F) * bench_tolerance(sums));
}

}  // namespace kirari::gpu::cuda
