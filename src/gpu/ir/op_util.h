// Outils communs aux règles de forme et aux calculs des opérations (interne au module).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::ir::detail {

std::unexpected<Refusal> shape_error(const Node& node, std::string_view detail);
std::unexpected<Refusal> graph_error(const Node& node, std::string_view detail);
// Construction valide en ONNX, mais hors du sous-ensemble dont le sens est sûr.
std::unexpected<Refusal> unsupported_error(const Node& node, std::string_view detail);

template <class A>
std::expected<const A*, Refusal> attrs_of(const Node& node) {
  const A* attrs = std::get_if<A>(&node.attrs);
  if (attrs == nullptr) return graph_error(node, "attributs d'un autre opérateur");
  return attrs;
}

inline bool present(std::span<const Operand> in, std::size_t index) {
  return index < in.size() && in[index].dims != nullptr;
}

// Entiers ou flottants d'une entrée connue avant l'exécution (forme, axes, échelles…).
std::expected<std::vector<std::int64_t>, Refusal> const_ints(const Node& node, std::span<const Operand> in,
                                                             std::size_t index, std::string_view what);
std::expected<std::vector<float>, Refusal> const_floats(const Node& node, std::span<const Operand> in,
                                                        std::size_t index, std::string_view what);

// Axe éventuellement négatif ramené dans [0, rank).
std::expected<std::size_t, Refusal> normalize_axis(const Node& node, std::int64_t axis, std::size_t rank);

// Diffusion multidirectionnelle à la numpy.
std::expected<Dims, Refusal> broadcast_dims(const Node& node, const Dims& a, const Dims& b);

std::expected<void, Refusal> expect_float(const Node& node, const Operand& operand, std::string_view what);
// Entrée facultative absente, ou présente avec ce type.
std::expected<void, Refusal> expect_type_if_present(const Node& node, std::span<const Operand> in,
                                                    std::size_t index, DataType type, std::string_view what);
std::expected<void, Refusal> expect_rank(const Node& node, const Dims& dims, std::size_t rank,
                                         std::string_view what);

// Pas en éléments de chaque axe, en ordre ligne.
std::vector<std::size_t> row_strides(const Dims& dims);

Value make_output(const OutputInfo& info);

// Opérandes connus de valeurs déjà calculées (nullptr : entrée absente).
std::vector<Operand> operands_of(std::span<const Value* const> values);

// Parcourt les indices multiples d'une forme en ordre ligne ; fn(index, position linéaire).
template <class Fn>
void for_each_index(const Dims& dims, Fn&& fn) {
  const std::size_t count = [&] {
    std::size_t n = 1;
    for (const std::int64_t d : dims) n *= static_cast<std::size_t>(d);
    return n;
  }();
  std::vector<std::int64_t> index(dims.size(), 0);
  for (std::size_t pos = 0; pos < count; ++pos) {
    fn(std::span<const std::int64_t>(index), pos);
    for (std::size_t axis = dims.size(); axis-- > 0;) {
      if (++index[axis] < dims[axis]) break;
      index[axis] = 0;
    }
  }
}

// Sortie dont chaque élément copie in à l'indice source[k] ; kFill donne fill.
inline constexpr std::size_t kFill = static_cast<std::size_t>(-1);
Value gather_elements(const Value& in, const OutputInfo& out, std::span<const std::size_t> source,
                      float fill = 0.0F);

// Pour chaque élément de out = DepthToSpace(x) (blocs de b × b pixels, ordre mode), son indice
// dans x.
std::vector<std::size_t> depth_to_space_source(const Dims& x, const Dims& out, std::int64_t b, BlockMode mode);

}  // namespace ajp::gpu::ir::detail
