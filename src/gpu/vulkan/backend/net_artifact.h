// Artefact du moteur Vulkan : ce que build() écrit dans le dossier de cache et que load() relit.
// Format binaire versionné, petit-boutiste, terminé par une somme FNV-1a 64 de tout ce qui
// précède : l'IR optimisée (tenseurs, constantes, nœuds et attributs), le choix de noyau de
// chaque nœud, les dispatchs (shaders, constantes de spécialisation, liaisons, push, groupes)
// et le hachage du SPIR-V de chaque shader utilisé. Un artefact d'une autre version du format,
// de l'abaissement ou des shaders est refusé au chargement : il se reconstruit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_program.h"

namespace ajp::gpu::vk::net {

inline constexpr std::uint32_t kArtifactVersion = 2;  // 2 : GridSampleAttrs::grid_nchw

struct ShaderHash {
  std::string shader;
  std::uint64_t hash = 0;  // FNV-1a 64 des mots SPIR-V
  friend bool operator==(const ShaderHash&, const ShaderHash&) = default;
};

struct ArtifactData {
  std::uint32_t lowering = 0;  // kLoweringVersion à la construction
  std::string model_id;
  std::string model_sha256;
  std::string device_key;  // DeviceProfile::key
  bool fp16 = true;        // précision demandée (ModelSpec::precision)
  ir::Graph graph;
  std::vector<KernelChoice> choices;  // un par nœud
  std::vector<Dispatch> dispatches;   // le programme sans ses blobs
  std::vector<ShaderHash> shaders;
};

std::uint64_t fnv1a(std::span<const std::byte> bytes, std::uint64_t hash = 0xCBF29CE484222325ULL);
std::uint64_t spirv_hash(std::span<const std::uint32_t> words);

std::vector<std::byte> serialize(const ArtifactData& data);

// Refus : HashMismatch (somme fausse), FormatVersion (autre version du format), CacheIo (octets
// tronqués ou valeur hors de son énumération). Dans tous les cas l'artefact se reconstruit.
std::expected<ArtifactData, Refusal> deserialize(std::span<const std::byte> bytes);

}  // namespace ajp::gpu::vk::net
