// Le masque des fenêtres décalées d'une échelle du transformeur de GMFlow, lu dans l'ONNX exporté pour
// l'attention native (prepare/gmflow_attention_native.h, docs/gmflow-natif.md).
#pragma once

#include <array>
#include <string>

#include "prepare/gmflow_onnx_view.h"

namespace kirari::prepare {

// Le masque de l'échelle dont les nœuds commencent par top (« /flownet/transformer/ » au huitième,
// « /flownet/transformer_1/ » au quart) : window, les tenseurs de la fenêtre en hauteur puis en largeur ;
// shift, ceux des décalages. Nul et value rempli (le masque entre deux régions, négatif) s'il est celui de
// Swin : régions de l'image décalée (ligne · 3 + colonne, comptées au bord moins une fenêtre et au bord moins
// un décalage), découpées en fenêtres, value où deux régions diffèrent, 0 où elles sont égales. Sinon ce qui
// manque.
const char* gmflow_mask_problem(const OnnxView& g, const std::string& top,
                                const std::array<std::string, 2>& window,
                                const std::array<std::string, 2>& shift, float& value);

}  // namespace kirari::prepare
