// Core ML sans ONNX Runtime à l'exécution : le modèle compilé (dossier .mlmodelc) qu'ONNX Runtime
// a produit pour un réseau entier, chargé et exécuté par Core ML lui-même. Les entrées enveloppent
// la mémoire de la chaîne (MLMultiArray sur ses octets) et Core ML écrit les sorties en place
// (MLPredictionOptions.outputBackings) : plus de sortie allouée à chaque image par Core ML, ni de
// copie de cette sortie par ONNX Runtime, sur un seul cœur du CPU.
//
// ONNX Runtime reste celui qui traduit le graphe ONNX en programme ML et le compile, à la
// construction (coreml_runtime.cpp) ; le chemin direct n'est retenu que si le réseau tient en un
// seul morceau confié à Core ML et que sa sortie retrouve celle d'ONNX Runtime.
//
// Sur un Mac, les tenseurs de la chaîne sont des IOSurface (Runtime::allocate_host, que Vulkan
// importe par VK_EXT_external_memory_host) : Core ML direct les donne au GPU et au Neural Engine
// pour ce qu'elles sont (MLMultiArray sur leur tampon de pixels), qui y lisent et y écrivent sans
// passer par une mémoire à eux.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "core/types.h"
#include "gpu/coreml/coreml_units.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

// Les IOSurface des tenseurs de la chaîne, reconnues à leur adresse. Verrouillées pour le CPU, et
// pour Vulkan qui y accède par la même projection, sauf pendant une prédiction de Core ML direct.
class Surfaces;
std::shared_ptr<Surfaces> make_surfaces();

// Une IOSurface pour ce tenseur FP16 NCHW : une ligne de W éléments sans bourrage par ligne de
// l'image (N × C × H lignes), d'au moins bytes octets, à une adresse multiple d'alignment, inscrite
// dans surfaces tant qu'elle vit. Refus FormatUnsupported : pas pour ce tenseur (FP32, forme que
// le système refuse ou dont il bourre les lignes) ; FeatureUnavailable : hors d'un Mac.
std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_surface(
    const std::shared_ptr<Surfaces>& surfaces, std::uint64_t bytes, std::uint64_t alignment, TensorType type,
    const Shape& shape);

class DirectModel {
 public:
  virtual ~DirectModel() = default;
  // Le type de chaque entrée et sortie dans le modèle, dans l'ordre des noms d'open_direct.
  virtual std::span<const TensorType> input_types() const = 0;
  virtual std::span<const TensorType> output_types() const = 0;
  // Toutes les liaisons du dernier appel passées par leur IOSurface ?
  virtual bool on_surfaces() const = 0;
  // Liaisons dans l'ordre des noms d'open_direct, chacune dans le type du modèle, NCHW dense.
  // Refus DeviceUnavailable : celui de Core ML ; InputMismatch : une sortie d'une autre taille.
  virtual std::expected<void, Refusal> predict(std::span<const vk::net::HostBinding> inputs,
                                               std::span<const vk::net::HostBinding> outputs) = 0;
};

// Charge compiled sur ces unités, avec les réglages qu'ONNX Runtime donne à Core ML (FP16 accumulé
// en basse précision sur le GPU, spécialisation pour l'inférence). Chaque nom ONNX est cherché dans
// la description du modèle, tel quel ou sous son nom de programme ML (ml_program_name). Une
// liaison au début d'une IOSurface de surfaces, à sa forme, passe par elle ; nullptr : jamais.
// Refus FeatureUnavailable : hors d'un Mac, une entrée ou une sortie absente ou qui n'est pas un
// tableau flottant ; CacheIo : le modèle ne se charge pas.
std::expected<std::unique_ptr<DirectModel>, Refusal> open_direct(const std::filesystem::path& compiled,
                                                                 Units units,
                                                                 std::span<const std::string> inputs,
                                                                 std::span<const std::string> outputs,
                                                                 std::shared_ptr<Surfaces> surfaces);

}  // namespace ajp::gpu::coreml_rt
