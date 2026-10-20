// Spécification des noyaux : le contrat que chaque port (CUDA, GLSL, Metal) implémente et que
// la référence CPU de kernels/ref/ exécute en FP32 scalaire.
//
// Le contrat tient en constantes constexpr, en formules écrites et en tolérances ; les ports
// GLSL reçoivent les mêmes nombres par l'en-tête que génère glsl_spec_header() (kernels/glsl.h).
// Toute modification d'une formule ou d'une constante incrémente kSpecVersion.
#pragma once

#include "kernels/spec/color.h"         // IWYU pragma: export
#include "kernels/spec/detect.h"        // IWYU pragma: export
#include "kernels/spec/gmfss.h"         // IWYU pragma: export
#include "kernels/spec/optical_flow.h"  // IWYU pragma: export
#include "kernels/spec/poses.h"         // IWYU pragma: export
#include "kernels/spec/resample.h"      // IWYU pragma: export
#include "kernels/spec/rife.h"          // IWYU pragma: export
#include "kernels/spec/tolerance.h"     // IWYU pragma: export
#include "kernels/spec/transfer.h"      // IWYU pragma: export
#include "kernels/spec/warp.h"          // IWYU pragma: export

namespace ajp::kernels {

inline constexpr int kSpecVersion = 1;

}  // namespace ajp::kernels
