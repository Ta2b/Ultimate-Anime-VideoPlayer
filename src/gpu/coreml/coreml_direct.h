// Core ML sans ONNX Runtime à l'exécution : le modèle compilé (dossier .mlmodelc) qu'ONNX Runtime
// a produit pour un réseau entier, chargé et exécuté par Core ML lui-même. Les entrées enveloppent
// la mémoire de la chaîne (MLMultiArray sur ses octets) et Core ML écrit les sorties en place
// (MLPredictionOptions.outputBackings) : plus de sortie allouée à chaque image par Core ML, ni de
// copie de cette sortie par ONNX Runtime, sur un seul cœur du CPU.
//
// ONNX Runtime reste celui qui traduit le graphe ONNX en programme ML et le compile, à la
// construction (coreml_runtime.cpp) ; le chemin direct n'est retenu que si le réseau tient en un
// seul morceau confié à Core ML et que sa sortie retrouve celle d'ONNX Runtime.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "core/types.h"
#include "gpu/coreml/coreml_units.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

class DirectModel {
 public:
  virtual ~DirectModel() = default;
  // Le type de chaque entrée et sortie dans le modèle, dans l'ordre des noms d'open_direct.
  virtual std::span<const TensorType> input_types() const = 0;
  virtual std::span<const TensorType> output_types() const = 0;
  // Liaisons dans l'ordre des noms d'open_direct, chacune dans le type du modèle, NCHW dense.
  // Refus DeviceUnavailable : celui de Core ML ; InputMismatch : une sortie d'une autre taille.
  virtual std::expected<void, Refusal> predict(std::span<const vk::net::HostBinding> inputs,
                                               std::span<const vk::net::HostBinding> outputs) = 0;
};

// Charge compiled sur ces unités, avec les réglages qu'ONNX Runtime donne à Core ML (FP16 accumulé
// en basse précision sur le GPU, spécialisation pour l'inférence). Chaque nom ONNX est cherché dans
// la description du modèle, tel quel ou sous son nom de programme ML (ml_program_name). Refus
// FeatureUnavailable : hors d'un Mac, une entrée ou une sortie absente ou qui n'est pas un tableau
// flottant ; CacheIo : le modèle ne se charge pas.
std::expected<std::unique_ptr<DirectModel>, Refusal> open_direct(const std::filesystem::path& compiled,
                                                                 Units units,
                                                                 std::span<const std::string> inputs,
                                                                 std::span<const std::string> outputs);

}  // namespace ajp::gpu::coreml_rt
