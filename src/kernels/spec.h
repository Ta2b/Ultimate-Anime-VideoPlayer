// Spécification des noyaux : le contrat que chaque port (CUDA, GLSL, Metal) implémente et que
// la référence CPU de kernels/ref/ exécute en FP32 scalaire.
//
// Le contrat tient en constantes constexpr, en formules écrites et en tolérances ; les ports
// GLSL reçoivent les mêmes nombres par l'en-tête que génère glsl_spec_header() (kernels/glsl.h).
//
// kSpecVersion est un repère de lecture, pas une clé de cache : il entre dans l'en-tête GLSL généré (donc
// dans chaque SPIR-V) et dans spec_check.comp, mais rien ne le lit pour décider d'un cache. Ce qui invalide
// un cache est le contenu : le SPIR-V embarqué pour les pipelines Vulkan, l'empreinte de src/kernels pour le
// module TensorRT (tools/release/trt-fingerprint.sh). On l'incrémente à une modification de formule ou de
// constante qu'un lecteur du contrat doit remarquer (C50-04) ; la règle n'est pas vérifiée par la CI.
//   1 : contrat d'origine.
//   2 : loi de mélange A = 1, B = 2,591 ; bloc tenu de 16 ; seuil de coupe 0,120 ; mesure de paire sur le RGB
//       ingéré à la plage de la source (pair_levels) ; Catmull-Rom borné par ses quatre points centraux.
#pragma once

#include "kernels/spec/color.h"        // IWYU pragma: export
#include "kernels/spec/detect.h"       // IWYU pragma: export
#include "kernels/spec/distil_drba.h"  // IWYU pragma: export
#include "kernels/spec/frame_stats.h"  // IWYU pragma: export
#include "kernels/spec/poses.h"        // IWYU pragma: export
#include "kernels/spec/resample.h"     // IWYU pragma: export
#include "kernels/spec/rife.h"         // IWYU pragma: export
#include "kernels/spec/tolerance.h"    // IWYU pragma: export
#include "kernels/spec/transfer.h"     // IWYU pragma: export
#include "kernels/spec/warp.h"         // IWYU pragma: export

namespace ajp::kernels {

inline constexpr int kSpecVersion = 2;

}  // namespace ajp::kernels
