#include "gpu/ir/op_util.h"

#include <algorithm>
#include <format>

#include "gpu/ir/graph.h"

namespace ajp::gpu::ir::detail {
namespace {

std::string where(const Node& node) {
  return std::format("{} {}", name(node.op), node.name);
}

const Value* constant_input(std::span<const Operand> in, std::size_t index) {
  return present(in, index) ? in[index].value : nullptr;
}

}  // namespace

std::unexpected<Refusal> shape_error(const Node& node, std::string_view detail) {
  return std::unexpected(Refusal{RefusalCode::ShapeInvalid, std::format("{} : {}", where(node), detail)});
}

std::unexpected<Refusal> graph_error(const Node& node, std::string_view detail) {
  return std::unexpected(Refusal{RefusalCode::GraphInvalid, std::format("{} : {}", where(node), detail)});
}

std::unexpected<Refusal> unsupported_error(const Node& node, std::string_view detail) {
  return std::unexpected(Refusal{RefusalCode::OnnxUnsupported, std::format("{} : {}", where(node), detail)});
}

std::expected<std::vector<std::int64_t>, Refusal> const_ints(const Node& node, std::span<const Operand> in,
                                                             std::size_t index, std::string_view what) {
  const Value* value = constant_input(in, index);
  if (value == nullptr) return shape_error(node, std::format("{} doit être connu avant l'exécution", what));
  if (value->type != DataType::I64)
    return graph_error(node, std::format("{} doit être un tenseur d'entiers", what));
  return value->i;
}

std::expected<std::vector<float>, Refusal> const_floats(const Node& node, std::span<const Operand> in,
                                                        std::size_t index, std::string_view what) {
  const Value* value = constant_input(in, index);
  if (value == nullptr) return shape_error(node, std::format("{} doit être connu avant l'exécution", what));
  if (value->type == DataType::I64)
    return graph_error(node, std::format("{} doit être un tenseur flottant", what));
  return value->f;
}

std::expected<std::size_t, Refusal> normalize_axis(const Node& node, std::int64_t axis, std::size_t rank) {
  const auto signed_rank = static_cast<std::int64_t>(rank);
  if (axis < -signed_rank || axis >= signed_rank) {
    return graph_error(node, std::format("axe {} hors du rang {}", axis, rank));
  }
  return static_cast<std::size_t>(axis < 0 ? axis + signed_rank : axis);
}

std::expected<Dims, Refusal> broadcast_dims(const Node& node, const Dims& a, const Dims& b) {
  const std::size_t rank = std::max(a.size(), b.size());
  Dims out(rank, 1);
  for (std::size_t k = 0; k < rank; ++k) {
    const std::int64_t da = k < rank - a.size() ? 1 : a[k - (rank - a.size())];
    const std::int64_t db = k < rank - b.size() ? 1 : b[k - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      return shape_error(node, std::format("{} et {} ne se diffusent pas", format_dims(a), format_dims(b)));
    }
    out[k] = da == 1 ? db : da;
  }
  return out;
}

std::expected<void, Refusal> expect_float(const Node& node, const Operand& operand, std::string_view what) {
  if (operand.type == DataType::I64) return graph_error(node, std::format("{} doit être flottant", what));
  return {};
}

std::expected<void, Refusal> expect_type_if_present(const Node& node, std::span<const Operand> in,
                                                    std::size_t index, DataType type, std::string_view what) {
  if (!present(in, index) || in[index].type == type) return {};
  return graph_error(node,
                     std::format("{} de type {} au lieu de {}", what, name(in[index].type), name(type)));
}

std::expected<void, Refusal> expect_rank(const Node& node, const Dims& dims, std::size_t rank,
                                         std::string_view what) {
  if (dims.size() != rank) {
    return shape_error(node, std::format("{} de rang {} au lieu de {}", what, dims.size(), rank));
  }
  return {};
}

std::vector<std::size_t> row_strides(const Dims& dims) {
  std::vector<std::size_t> strides(dims.size(), 1);
  for (std::size_t axis = dims.size(); axis-- > 1;) {
    strides[axis - 1] = strides[axis] * static_cast<std::size_t>(dims[axis]);
  }
  return strides;
}

Value make_output(const OutputInfo& info) {
  return make_value(info.type, info.dims);
}

std::vector<Operand> operands_of(std::span<const Value* const> values) {
  std::vector<Operand> out;
  out.reserve(values.size());
  for (const Value* value : values) {
    out.push_back(value == nullptr ? Operand{}
                                   : Operand{.type = value->type, .dims = &value->dims, .value = value});
  }
  return out;
}

Value gather_elements(const Value& in, const OutputInfo& out, std::span<const std::size_t> source,
                      float fill) {
  Value result = make_output(out);
  if (in.type == DataType::I64) {
    for (std::size_t k = 0; k < source.size(); ++k) {
      result.i[k] = source[k] == kFill ? static_cast<std::int64_t>(fill) : in.i[source[k]];
    }
  } else {
    for (std::size_t k = 0; k < source.size(); ++k) result.f[k] = source[k] == kFill ? fill : in.f[source[k]];
  }
  return result;
}

}  // namespace ajp::gpu::ir::detail
