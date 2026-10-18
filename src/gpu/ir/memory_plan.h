// Plan mémoire : durée de vie de chaque activation interne et place dans une
// arène unique, avec réutilisation des tampons dont les vies ne se croisent pas.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

// Alignement des décalages : le plus grand minStorageBufferOffsetAlignment de Vulkan.
inline constexpr std::uint64_t kBufferAlignment = 256;

struct Allocation {
  TensorId tensor = kNoTensor;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;  // arrondi à l'alignement
  std::size_t first = 0;    // nœud qui écrit le tenseur
  std::size_t last = 0;     // dernier nœud qui le lit
};

struct MemoryPlan {
  std::vector<Allocation> allocations;  // activations internes, par identifiant de tenseur
  std::uint64_t arena_bytes = 0;        // taille de l'arène avec réutilisation
  std::uint64_t unshared_bytes = 0;     // somme des tampons sans réutilisation
  std::uint64_t constant_bytes = 0;     // poids et autres constantes lues par les nœuds
  std::uint64_t io_bytes = 0;           // entrées et sorties du graphe, hors arène
};

// Placement glouton du plus grand au plus petit, au plus bas décalage libre
// pendant toute la durée de vie. Les tailles suivent la disposition de chaque
// tenseur (storage_bytes). Exige des formes connues.
MemoryPlan plan_memory(const Graph& graph, std::uint64_t alignment = kBufferAlignment);

}  // namespace ajp::gpu::ir
