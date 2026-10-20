// Core ML (Apple) comme runtime des réseaux du backend Vulkan (net_runtime.h), par ONNX Runtime et
// son fournisseur CoreML : programme ML (MLProgram) en FP16, réparti par Core ML entre le Neural
// Engine, le GPU et le CPU. La chaîne reste sur Vulkan (MoltenVK) ; la mémoire unifiée des puces
// Apple rend la projection hôte des tenseurs sans copie sur le bus.
//
// L'artefact est un manifeste (le fichier ONNX, les dimensions symboliques fixées, les unités
// mesurées) ; le modèle compilé par Core ML vit à côté, dans <artefact>.mlcache, que le fournisseur
// relit au chargement. L'interface du réseau est réécrite dans le type des tenseurs de la chaîne
// (<artefact>.onnx, ir/onnx_io_type.h) : Core ML lit et écrit leur mémoire telle quelle, sans
// conversion par l'hôte à chaque image.
#pragma once

#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

// Refus DeviceMismatch : appareil qui n'est pas Apple ; LibraryUnavailable : ONNX Runtime sans
// fournisseur CoreML.
std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps);

}  // namespace ajp::gpu::coreml_rt
