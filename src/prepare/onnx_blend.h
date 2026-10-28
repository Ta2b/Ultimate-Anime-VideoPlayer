// Un modèle ONNX écrit de toutes pièces à l'interface de RIFE v2 (kernels/spec/rife.h) : entrée
// « input » 1 × 7 × height × width (image a en RGB, image b, instant t), sortie « output »
// 1 × 3 × height × width, en fp16, height et width libres et nommés. Son graphe fait le fondu
// a + (b − a) × t ; ses producer_name et producer_version le marquent pour un runtime qui le
// remplace par son propre interpolateur (l'interpolation d'Apple sous Core ML,
// gpu/system_interpolator.h). Un autre runtime ferait le fondu.
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace ajp::prepare {

std::vector<std::byte> blend_model(std::string_view producer_name, std::string_view producer_version);

}  // namespace ajp::prepare
