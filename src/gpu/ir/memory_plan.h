// Plan mémoire : durée de vie de chaque activation interne et place dans une
// arène unique, avec réutilisation des tampons dont les vies ne se croisent pas.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

// Alignement des décalages : le plus grand minStorageBufferOffsetAlignment de Vulkan.
inline constexpr std::uint64_t kBufferAlignment = 256;
inline constexpr std::uint64_t kMaxBufferAlignment = 65536;

struct Allocation {
  TensorId tensor = kNoTensor;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;  // arrondi à l'alignement
  std::size_t first = 0;    // nœud qui écrit le tenseur
  std::size_t last = 0;     // dernier nœud qui le lit
};

// Un tenseur rangé dans un autre, sans copie : tranche contiguë (entrée d'un Concat que son
// producteur écrit en place, sortie d'un Split ou d'un Slice) ou même mémoire vue autrement
// (Reshape). Le parent peut lui-même être rangé dans un autre ; la racine de la chaîne a sa propre
// place, dans l'arène ou dans une entrée ou une sortie du graphe.
struct Alias {
  TensorId parent = kNoTensor;
  std::uint64_t offset = 0;  // octets depuis le début du parent
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
// tenseur (storage_bytes). Un tenseur rangé dans un autre (aliases, un par tenseur, ou vide)
// n'a pas de place à lui : sa racine en tient lieu, réservée pendant la vie de toute sa famille.
// Refuse un graphe invalide, une forme inconnue, un alignement hors de [1, kMaxBufferAlignment]
// et des alias d'une autre taille que le graphe.
std::expected<MemoryPlan, Refusal> plan_memory(const Graph& graph, std::uint64_t alignment = kBufferAlignment,
                                               std::span<const Alias> aliases = {});

}  // namespace ajp::gpu::ir
