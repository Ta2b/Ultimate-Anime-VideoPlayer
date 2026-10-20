// Modèles ONNX de référence des tests du backend Vulkan, écrits par le constructeur de modèles
// de gpu_ir : SPAN et RIFE en petit, en float ou en float16, avec des poids déterministes.
// Partagés entre les tests du moteur de réseaux et ceux des runtimes des fabricants.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../gpu_ir/onnx_builder.h"

namespace ajp::gpu::vk::test {

// count valeurs déterministes dans [-scale, scale[.
std::vector<float> weights(std::size_t count, std::size_t seed, float scale = 0.3F);

// Poids et biais d'une Conv ou ConvTranspose, en float ou en float16 selon le modèle.
void add_weight(ir::test::ModelBuilder& m, const std::string& name, const std::vector<std::int64_t>& dims,
                bool half, std::size_t seed, float scale = 0.3F);

// SPAN en petit : conv 3x3 + PRelu, deux fois, conv 3x3 vers 12 canaux, pixel shuffle x2.
// Entrée x de 1 × 3 × 24 × 32, sortie y de 1 × 3 × 48 × 64.
ir::test::ModelBuilder span_like(bool half);

// RIFE en petit : deux images, réduction linéaire, conv, ConvTranspose vers un flot, grille par
// Transpose, GridSample bord et coins alignés, mélange par un masque sigmoïde. Entrée x de
// 1 × 6 × 16 × 24, sortie y de 1 × 3 × 16 × 24.
ir::test::ModelBuilder rife_like(bool half = false);

}  // namespace ajp::gpu::vk::test
