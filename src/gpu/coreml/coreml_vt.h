// L'interpolation d'Apple derrière l'interface d'un modèle RIFE v2 (gpu/system_interpolator.h) :
// VTFrameRateConversion de VideoToolbox (macOS 15.4), le réseau de flot optique du système, sur le
// Neural Engine et le GPU de la puce. L'entrée 1 × 7 × H × W (image a, image b, instant t) passe
// dans des tampons Core Video (vt_pixels.h), la sortie 1 × 3 × H × W en revient : par le GPU
// (vt_gpu.h) quand les tenseurs sont des IOSurface de la chaîne, sinon par les cœurs du CPU.
//
// Une paire qui suit la précédente (son image a est l'image b d'avant) ou qui la répète (×3 : deux
// instants de la même paire) est soumise en séquence : VideoToolbox reprend ce qu'il a déjà calculé,
// et l'image déjà convertie sert telle quelle. Un saut (recherche, image sautée) vide son cache.
#pragma once

#include <expected>
#include <memory>
#include <string>

#include "core/types.h"
#include "gpu/coreml/coreml_direct.h"
#include "gpu/coreml/vt_pixels.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

// Le système interpole-t-il des images width × height ? Une session ouverte, une paire unie
// interpolée sur des IOSurface et vérifiée, la session refermée ; rend le format de pixels retenu
// et où se font les conversions. Refus FeatureUnavailable : avant macOS 15.4, puce ou taille
// refusées ; DeviceUnavailable : une interpolation qui échoue ou qui rend une autre couleur.
std::expected<std::string, Refusal> vt_probe(int width, int height);

// Le réseau, aux noms d'entrée et de sortie de son modèle ; surfaces : les IOSurface de la chaîne
// (coreml_direct.h), dont les conversions passent par le GPU ; nul, par le CPU. Refus : ceux de
// l'ouverture.
std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> vt_net(vk::net::IoNames names, int width,
                                                                    int height,
                                                                    std::shared_ptr<Surfaces> surfaces);

// Les conversions de ce format sur le GPU, sans VideoToolbox : vérifiées contre celles du CPU
// (vt_pixels.h) sur une paire de test width × height. Refus FeatureUnavailable : pas de GPU Metal,
// taille refusée, hors d'un Mac ; DeviceUnavailable : un écart.
std::expected<void, Refusal> vt_gpu_check(vt::Pixels format, int width, int height);

}  // namespace ajp::gpu::coreml_rt
