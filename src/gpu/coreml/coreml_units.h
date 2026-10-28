// Les unités de calcul que Core ML reçoit pour un réseau entier, et leurs noms : celui que le
// fournisseur CoreML d'ONNX Runtime accepte pour MLComputeUnits, celui du manifeste. Sans ONNX
// Runtime, pour que les tests les lisent.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace ajp::gpu::coreml_rt {

enum class Units : std::uint8_t { All, CpuAndGpu, CpuAndNeuralEngine };

// La valeur de MLComputeUnits : « ALL », « CPUAndGPU », « CPUAndNeuralEngine » (coreml_options.cc
// d'ONNX Runtime ; les noms d'Apple, « MLComputeUnitsAll »…, y sont refusés).
std::string_view units_option(Units units);
std::string_view units_name(Units units);  // « all », « gpu », « ane » : le manifeste
std::optional<Units> units_from(std::string_view name);

}  // namespace ajp::gpu::coreml_rt
