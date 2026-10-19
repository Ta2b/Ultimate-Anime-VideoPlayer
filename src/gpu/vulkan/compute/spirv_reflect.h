// Interface d'un shader de calcul, lue dans son SPIR-V : bindings, bloc push_constant,
// constantes de spécialisation et capacités. PipelineCache compare ses capacités à l'appareil
// avant vkCreateShaderModule, puis l'interface à la PipelineDesc de l'appelant avant de créer le
// pipeline : un shader qui attend autre chose que ce que l'hôte lie (binding absent, push de
// taille différente, constante oubliée qui garderait sa valeur par défaut) ou une capacité non
// activée (variante float16 sans stockage 16 bits) est refusé par son nom au lieu d'atteindre le
// pilote.
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/caps.h"
#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {

struct PipelineDesc;  // pipeline.h, qui inclut cet en-tête

struct ShaderInterface {
  std::string name;
  std::vector<std::uint32_t> storage_buffers;  // bindings de l'ensemble 0, croissants
  std::vector<std::uint32_t> storage_images;
  std::uint32_t push_bytes = 0;             // fin du dernier membre du bloc push_constant
  std::vector<std::uint32_t> spec_ids;      // SpecId déclarés, croissants
  std::vector<std::uint32_t> capabilities;  // OpCapability, dans l'ordre du module
};

// Refus InputMismatch : SPIR-V malformé, pas de point d'entrée GLCompute « main », ou interface
// hors de la disposition fixe (ensemble autre que 0, binding en double, uniform buffer, image
// échantillonnée, tableau de descripteurs, bloc push qu'on ne sait pas mesurer).
std::expected<ShaderInterface, Refusal> reflect(std::span<const std::uint32_t> words, std::string_view name);

// Chaque capacité du shader est activée sur l'appareil (caps.enabled, opérations de sous-groupe).
// Refus FeatureUnavailable : capacité inconnue du socle ou non activée.
Done check_capabilities(const ShaderInterface& shader, const DeviceCaps& caps);

// La description convient au shader. Refus InputMismatch : bindings, taille du push ou constantes
// de spécialisation différents de ceux du shader.
Done check_interface(const ShaderInterface& shader, const PipelineDesc& desc);

}  // namespace ajp::gpu::vk
