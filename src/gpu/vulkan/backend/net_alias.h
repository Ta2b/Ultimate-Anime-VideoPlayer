// Tenseurs sans copie du moteur Vulkan : les entrées d'un Concat que leur producteur écrit
// directement dans leur tranche, les sorties d'un Split ou d'un Slice lues dans leur source, et
// les Reshape qui ne font que renommer la mémoire. Seules les tranches contiguës en Nchw entrent
// (axes avant celui du découpage à 1, pas de 1), de même type, à un décalage que l'appareil sait
// lier. L'abaissement n'émet aucun dispatch pour elles, le plan mémoire réserve la famille entière,
// et NetEngine lie chaque membre par une vue dans sa racine.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "gpu/ir/ir.h"
#include "gpu/ir/memory_plan.h"

namespace ajp::gpu::vk::net {

// Un alias par tenseur du graphe ; alignment : alignement des vues de tampon de l'appareil.
std::vector<ir::Alias> plan_aliases(const ir::Graph& graph, std::uint64_t alignment);

// Octets depuis le début de whole jusqu'à l'élément start de l'axe axis, si la tranche qui y
// commence est contiguë (tous les axes avant axis valent 1) et whole rangé en Nchw.
std::optional<std::uint64_t> slab_offset(const ir::Tensor& whole, std::size_t axis, std::int64_t start);

// Le tenseur child est-il rangé dans parent à ce décalage ?
inline bool aliased(const std::vector<ir::Alias>& aliases, ir::TensorId child, ir::TensorId parent,
                    std::uint64_t offset = 0) {
  return child < aliases.size() && aliases[child].parent == parent && aliases[child].offset == offset;
}

}  // namespace ajp::gpu::vk::net
