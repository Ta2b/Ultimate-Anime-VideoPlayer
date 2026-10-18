// Outils sur les valeurs et les graphes de l'IR : tailles, noms, voisinage, compactage.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

// Bornes de toute forme connue : au-delà, l'inférence refuse.
inline constexpr std::int64_t kMaxDim = std::int64_t{1} << 24;
inline constexpr std::uint64_t kMaxElements = std::uint64_t{1} << 32;
inline constexpr std::size_t kMaxRank = 8;

inline constexpr std::size_t kNoNode = std::numeric_limits<std::size_t>::max();

std::string_view name(DataType type);
std::size_t element_size(DataType type);

// Produit des dimensions ; les dimensions sont supposées connues et bornées.
std::size_t element_count(const Dims& dims);
std::string format_dims(const Dims& dims);

// Valeur de la forme donnée, remplie de zéros.
Value make_value(DataType type, Dims dims);
std::size_t value_size(const Value& value);  // nombre d'éléments stockés

// Octets occupés par un tenseur dans sa disposition (Nhwc16 compte le bourrage).
std::uint64_t storage_bytes(const Tensor& tensor);
std::int64_t align_up(std::int64_t value, std::int64_t alignment);

TensorId add_tensor(Graph& graph, Tensor tensor);
std::optional<TensorId> find_tensor(const Graph& graph, std::string_view name);
bool is_graph_output(const Graph& graph, TensorId id);
bool is_graph_input(const Graph& graph, TensorId id);

// Pour chaque tenseur, les nœuds qui le lisent (indices dans graph.nodes).
std::vector<std::vector<std::size_t>> consumers(const Graph& graph);
// Pour chaque tenseur, le nœud qui l'écrit, ou kNoNode.
std::vector<std::size_t> producers(const Graph& graph);

// Remplace from par to dans les entrées de tous les nœuds.
void replace_uses(Graph& graph, TensorId from, TensorId to);

// Retire les tenseurs qu'aucun nœud ni aucune interface ne référence, et renumérote.
void compact(Graph& graph);

// Vérifie les identifiants, l'unicité des producteurs et l'ordre topologique.
std::expected<void, Refusal> validate(const Graph& graph);

}  // namespace ajp::gpu::ir
