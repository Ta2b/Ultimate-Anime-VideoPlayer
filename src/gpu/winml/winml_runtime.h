// MIGraphX d'AMD par Windows ML comme runtime des réseaux du backend Vulkan (net_runtime.h), sous
// Windows : ONNX Runtime de Windows ML et son fournisseur MIGraphX, que Windows installe sur les GPU
// AMD RDNA 3 et plus récents (cartes et APU, Radeon 780M, 860M, 890M…). Sous Linux, MIGraphX passe
// par ROCm (gpu/migraphx) ; ROCm n'a pas MIGraphX sous Windows.
//
// La chaîne reste sur Vulkan ; les tenseurs passent par l'hôte (mémoire hôte visible du GPU), que
// le fournisseur copie vers et depuis sa mémoire. L'artefact est un manifeste (le fichier ONNX, les
// dimensions symboliques fixées) ; le programme compilé par MIGraphX vit à côté, dans
// <artefact>.mxcache, que le fournisseur relit au chargement. Moteur à l'essai : proposé, jamais
// coché d'office, à comparer avec Vulkan.
#pragma once

#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::winml_rt {

// Refus DeviceMismatch : appareil qui n'est pas AMD ; LibraryUnavailable : Windows ML absent, ou
// sans MIGraphX pour cette machine ; DeviceUnavailable : MIGraphX ne s'installe pas, ou ne voit
// aucun GPU.
std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps);

}  // namespace ajp::gpu::winml_rt
