// Interface d'un shader de calcul, lue dans son SPIR-V : bindings, bloc push_constant,
// constantes de spécialisation et capacités. PipelineCache la compare à la PipelineDesc de
// l'appelant avant de créer le pipeline : un shader qui attend autre chose que ce que l'hôte
// lie (binding absent, push de taille différente, constante oubliée qui garderait sa valeur par
// défaut, capacité non activée) est refusé par son nom au lieu d'atteindre le pilote.
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
  std::uint32_t push_bytes = 0;              // fin du dernier membre du bloc push_constant
  std::vector<std::uint32_t> spec_ids;       // SpecId déclarés, croissants
  std::vector<std::uint32_t> capabilities;   // OpCapability, dans l'ordre du module
};

// Refus InputMismatch : SPIR-V malformé, pas de point d'entrée GLCompute « main », ou interface
// hors de la disposition fixe (ensemble autre que 0, binding en double, uniform buffer, image
// échantillonnée, tableau de descripteurs, bloc push qu'on ne sait pas mesurer).
std::expected<ShaderInterface, Refusal> reflect(std::span<const std::uint32_t> words, std::string_view name);

// La description et l'appareil conviennent au shader. Refus : InputMismatch (bindings, taille du
// push ou constantes de spécialisation différents de ceux du shader), FeatureUnavailable
// (capacité inconnue ou non activée sur l'appareil).
Done check_interface(const ShaderInterface& shader, const PipelineDesc& desc, const DeviceCaps& caps);

}  // namespace ajp::gpu::vk
