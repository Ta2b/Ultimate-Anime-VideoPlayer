// Opérations sur les formes : Shape, Gather, Reshape, Unsqueeze, Squeeze, Transpose.
// Reshape, Unsqueeze et Squeeze ne déplacent aucune donnée : seule la forme change.
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace ajp::gpu::ir::detail {
namespace {

// Bornes [début, fin) de Shape, ramenées dans [0, rang].
std::pair<std::size_t, std::size_t> shape_range(const ShapeAttrs& attrs, std::size_t rank) {
  const auto signed_rank = static_cast<std::int64_t>(rank);
  auto clamp = [&](std::int64_t v) {
    if (v < 0) v += signed_rank;
    return static_cast<std::size_t>(std::clamp<std::int64_t>(v, 0, signed_rank));
  };
  const std::size_t start = clamp(attrs.start);
  return {start, std::max(start, clamp(attrs.end))};
}

std::expected<Dims, Refusal> reshape_dims(const Node& node, const Dims& in, const Dims& shape,
                                          bool allowzero) {
  Dims out = shape;
  std::size_t inferred = out.size();
  std::size_t known = 1;
  for (std::size_t k = 0; k < out.size(); ++k) {
    if (out[k] == 0 && !allowzero) {
      if (k >= in.size()) return shape_error(node, "0 recopie une dimension absente de l'entrée");
      out[k] = in[k];
    }
    if (out[k] == -1) {
      if (inferred != out.size()) return shape_error(node, "plusieurs dimensions à déduire (-1)");
      inferred = k;
      continue;
    }
    if (out[k] < 0 || out[k] > kMaxDim) return shape_error(node, std::format("dimension {}", out[k]));
    known *= static_cast<std::size_t>(out[k]);
    if (known > kMaxElements) return shape_error(node, "forme trop grande");
  }
  const std::size_t total = element_count(in);
  if (inferred != out.size()) {
    if (known == 0 || total % known != 0) return shape_error(node, "dimension -1 impossible à déduire");
    out[inferred] = static_cast<std::int64_t>(total / known);
  } else if (known != total) {
    return shape_error(node, std::format("{} éléments vers {}", total, format_dims(out)));
  }
  return out;
}

std::expected<Dims, Refusal> unsqueeze_dims(const Node& node, const Dims& in, const Dims& axes) {
  const std::size_t rank = in.size() + axes.size();
  std::vector<bool> inserted(rank, false);
  for (const std::int64_t axis : axes) {
    auto a = normalize_axis(node, axis, rank);
    if (!a) return std::unexpected(a.error());
    if (inserted[*a]) return graph_error(node, "axe répété");
    inserted[*a] = true;
  }
  Dims out;
  std::size_t next = 0;
  for (std::size_t k = 0; k < rank; ++k) out.push_back(inserted[k] ? 1 : in[next++]);
  return out;
}

std::expected<Dims, Refusal> squeeze_dims(const Node& node, const Dims& in, const Dims* axes) {
  std::vector<bool> removed(in.size(), false);
  for (std::size_t k = 0; k < in.size(); ++k) removed[k] = axes == nullptr && in[k] == 1;
  if (axes != nullptr) {
    for (const std::int64_t axis : *axes) {
      auto a = normalize_axis(node, axis, in.size());
      if (!a) return std::unexpected(a.error());
      if (in[*a] != 1) return shape_error(node, std::format("l'axe {} ne vaut pas 1", axis));
      removed[*a] = true;
    }
  }
  Dims out;
  for (std::size_t k = 0; k < in.size(); ++k) {
    if (!removed[k]) out.push_back(in[k]);
  }
  return out;
}

std::expected<Dims, Refusal> permutation(const Node& node, std::size_t rank) {
  auto attrs = attrs_of<TransposeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  Dims perm = (*attrs)->perm;
  if (perm.empty()) {
    perm.resize(rank);
    std::iota(perm.rbegin(), perm.rend(), 0);
  }
  Dims sorted = perm;
  std::ranges::sort(sorted);
  Dims expected(rank);
  std::iota(expected.begin(), expected.end(), 0);
  if (sorted != expected) return graph_error(node, std::format("perm {} invalide", format_dims(perm)));
  return perm;
}

}  // namespace

Inferred infer_shape(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<ShapeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const auto [start, end] = shape_range(**attrs, in[0].dims->size());
  return std::vector<OutputInfo>{
      {.type = DataType::I64, .dims = Dims{static_cast<std::int64_t>(end - start)}}};
}

Evaluated eval_shape(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<ShapeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Dims& dims = in[0]->dims;
  const auto [start, end] = shape_range(**attrs, dims.size());
  Value out = make_output(outs[0]);
  std::copy(dims.begin() + static_cast<std::ptrdiff_t>(start),
            dims.begin() + static_cast<std::ptrdiff_t>(end), out.i.begin());
  return std::vector<Value>{std::move(out)};
}

Inferred infer_gather(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<AxisAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Dims& data = *in[0].dims;
  if (in[1].type != DataType::I64) return graph_error(node, "indices entiers attendus");
  auto axis = normalize_axis(node, (*attrs)->axis, data.size());
  if (!axis) return std::unexpected(axis.error());
  // Des indices connus d'avance sont vérifiés dès l'inférence.
  if (in[1].value != nullptr) {
    const std::int64_t extent = data[*axis];
    for (const std::int64_t raw : in[1].value->i) {
      if (raw < -extent || raw >= extent) {
        return graph_error(node, std::format("indice {} hors de [-{}, {})", raw, extent, extent));
      }
    }
  }
  Dims out(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(*axis));
  out.insert(out.end(), in[1].dims->begin(), in[1].dims->end());
  out.insert(out.end(), data.begin() + static_cast<std::ptrdiff_t>(*axis) + 1, data.end());
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_gather(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<AxisAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Value& data = *in[0];
  const Value& indices = *in[1];
  auto normalized = normalize_axis(node, (*attrs)->axis, data.dims.size());
  if (!normalized) return std::unexpected(normalized.error());
  const std::size_t axis = *normalized;
  const std::int64_t extent = data.dims[axis];
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (std::size_t k = 0; k < axis; ++k) outer *= static_cast<std::size_t>(data.dims[k]);
  for (std::size_t k = axis + 1; k < data.dims.size(); ++k) inner *= static_cast<std::size_t>(data.dims[k]);

  std::vector<std::size_t> source;
  source.reserve(element_count(outs[0].dims));
  for (std::size_t o = 0; o < outer; ++o) {
    for (const std::int64_t raw : indices.i) {
      if (raw < -extent || raw >= extent)
        return graph_error(node, std::format("indice {} hors de [-{}, {})", raw, extent, extent));
      const auto index = static_cast<std::size_t>(raw < 0 ? raw + extent : raw);
      const std::size_t base = ((o * static_cast<std::size_t>(extent)) + index) * inner;
      for (std::size_t i = 0; i < inner; ++i) source.push_back(base + i);
    }
  }
  return std::vector<Value>{gather_elements(data, outs[0], source)};
}

Inferred infer_reshape(const Node& node, std::span<const Operand> in) {
  const Dims& data = *in[0].dims;
  std::expected<Dims, Refusal> out;
  if (node.op == OpKind::Reshape) {
    auto attrs = attrs_of<ReshapeAttrs>(node);
    if (!attrs) return std::unexpected(attrs.error());
    auto shape = const_ints(node, in, 1, "la forme");
    if (!shape) return std::unexpected(shape.error());
    out = reshape_dims(node, data, *shape, (*attrs)->allowzero);
  } else if (node.op == OpKind::Unsqueeze) {
    auto axes = const_ints(node, in, 1, "les axes");
    if (!axes) return std::unexpected(axes.error());
    out = unsqueeze_dims(node, data, *axes);
  } else {
    std::expected<Dims, Refusal> axes = Dims{};
    if (present(in, 1)) axes = const_ints(node, in, 1, "les axes");
    if (!axes) return std::unexpected(axes.error());
    out = squeeze_dims(node, data, present(in, 1) ? &*axes : nullptr);
  }
  if (!out) return std::unexpected(out.error());
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(*out)}};
}

Evaluated eval_reshape(const Node& /*node*/, std::span<const Value* const> in,
                       std::span<const OutputInfo> outs) {
  Value out = *in[0];
  out.dims = outs[0].dims;
  return std::vector<Value>{std::move(out)};
}

Inferred infer_transpose(const Node& node, std::span<const Operand> in) {
  const Dims& data = *in[0].dims;
  auto perm = permutation(node, data.size());
  if (!perm) return std::unexpected(perm.error());
  Dims out(data.size());
  for (std::size_t k = 0; k < data.size(); ++k) out[k] = data[static_cast<std::size_t>((*perm)[k])];
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_transpose(const Node& node, std::span<const Value* const> in,
                         std::span<const OutputInfo> outs) {
  const Value& data = *in[0];
  auto perm = permutation(node, data.dims.size());
  if (!perm) return std::unexpected(perm.error());
  const std::vector<std::size_t> strides = row_strides(data.dims);
  std::vector<std::size_t> source(element_count(outs[0].dims));
  for_each_index(outs[0].dims, [&](std::span<const std::int64_t> index, std::size_t pos) {
    std::size_t at = 0;
    for (std::size_t k = 0; k < index.size(); ++k) {
      at += static_cast<std::size_t>(index[k]) * strides[static_cast<std::size_t>((*perm)[k])];
    }
    source[pos] = at;
  });
  return std::vector<Value>{gather_elements(data, outs[0], source)};
}

// Range : start + k × delta pour k < max(ceil((limit − start) / delta), 0). La longueur dépend des
// valeurs : les trois bornes doivent être connues dès l'inférence (pliées depuis les formes).
namespace {

std::expected<std::int64_t, Refusal> range_length(const Node& node, std::span<const Operand> in) {
  for (std::size_t k = 0; k < 3; ++k) {
    if (in[k].type != in[0].type) return graph_error(node, "bornes de types différents");
    if (element_count(*in[k].dims) != 1) return graph_error(node, "bornes scalaires attendues");
    if (in[k].value == nullptr) return unsupported_error(node, "bornes calculées pendant l'exécution");
  }
  double count = 0;
  if (in[0].type == DataType::I64) {
    const std::int64_t delta = in[2].value->i[0];
    if (delta == 0) return graph_error(node, "pas nul");
    // Au-delà de 2^53, le double ne compte plus exactement, et la différence déborderait en int64.
    constexpr std::int64_t kExactBound = std::int64_t{1} << 53;
    for (std::size_t k = 0; k < 2; ++k) {
      if (in[k].value->i[0] > kExactBound || in[k].value->i[0] < -kExactBound)
        return shape_error(node, "bornes de Range trop grandes");
    }
    if (delta > kExactBound || delta < -kExactBound) return shape_error(node, "pas de Range trop grand");
    count = std::ceil((static_cast<double>(in[1].value->i[0]) - static_cast<double>(in[0].value->i[0])) /
                      static_cast<double>(delta));
  } else {
    const float delta = in[2].value->f[0];
    if (delta == 0.0F || !std::isfinite(delta)) return graph_error(node, "pas nul ou non fini");
    count = std::ceil((static_cast<double>(in[1].value->f[0]) - in[0].value->f[0]) / delta);
  }
  if (!std::isfinite(count) || count > static_cast<double>(kMaxDim))
    return shape_error(node, "trop d'éléments");
  return std::max<std::int64_t>(0, static_cast<std::int64_t>(count));
}

}  // namespace

Inferred infer_range(const Node& node, std::span<const Operand> in) {
  auto length = range_length(node, in);
  if (!length) return std::unexpected(length.error());
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = Dims{*length}}};
}

Evaluated eval_range(const Node& /*node*/, std::span<const Value* const> in,
                     std::span<const OutputInfo> outs) {
  Value out = make_output(outs[0]);
  if (in[0]->type == DataType::I64) {
    for (std::size_t k = 0; k < out.i.size(); ++k)
      out.i[k] =
          in[0]->i[0] + (static_cast<std::int64_t>(k) * in[2]->i[0]);  // bornes ≤ 2^53 : sans débordement
  } else {
    // Comme ONNX Runtime : start + k × delta, en float.
    for (std::size_t k = 0; k < out.f.size(); ++k)
      out.f[k] = in[0]->f[0] + (static_cast<float>(k) * in[2]->f[0]);
  }
  return std::vector<Value>{std::move(out)};
}

}  // namespace ajp::gpu::ir::detail
