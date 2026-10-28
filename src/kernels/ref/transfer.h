// Référence des fonctions de transfert et de la conversion de primaires (spec/transfer.h), en
// float. Tests : tests/unit/kernels/transfer_test.cpp.
#pragma once

#include "kernels/image.h"
#include "kernels/spec/color.h"
#include "kernels/spec/transfer.h"

namespace ajp::kernels::ref {

// PQ : signal [0, 1] → cd/m² [0, 10000], et l'inverse. Entrées bornées à leur domaine.
float pq_eotf(float signal);
float pq_inverse_eotf(float nits);

// HLG : lumière de scène [0, 1] → signal [0, 1], et l'inverse. Entrées bornées à [0, 1].
float hlg_oetf(float scene);
float hlg_inverse_oetf(float signal);

// Écran HLG validé : pic Lw dans [400, 2000] cd/m², noir 0.
struct HlgDisplay {
  float peak = 1000.0F;
  float gamma = 1.2F;  // 1,2 + 0,42 log10(Lw / 1000)
};
Result<HlgDisplay> hlg_display(float peak_nits);

// OOTF HLG : lumière de scène RGB [0, 1] → lumière affichée en cd/m².
Rgb hlg_ootf(const Rgb& scene, const HlgDisplay& display);

// sRGB : signal ↔ lumière relative [0, 1]. Entrées bornées à [0, 1].
float srgb_eotf(float signal);
float srgb_inverse_eotf(float light);

// Écran BT.1886 validé : blanc Lw > noir Lb ≥ 0, constantes a et b précalculées.
struct Bt1886Display {
  float white = 1.0F;
  float black = 0.0F;
  float a = 1.0F;
  float b = 0.0F;
};
Result<Bt1886Display> bt1886_display(float white, float black);

// BT.1886 : signal [0, 1] → luminance [Lb, Lw], et l'inverse. Entrées bornées à leur domaine.
float bt1886_eotf(float signal, const Bt1886Display& display);
float bt1886_inverse_eotf(float light, const Bt1886Display& display);

// Produit matrice × RGB linéaire, la matrice rangée en float comme dans un port.
Rgb convert_primaries(const Rgb& linear, const Mat3& matrix);

}  // namespace ajp::kernels::ref
