// Découpes et assemblages : Concat, Split, Slice, Pad. Les calculs passent par
// une table d'indices source, commune aux tenseurs flottants et entiers.
#include <algorithm>
#include <format>
#include <limits>
#include <numeric>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace ajp::gpu::ir::detail {
namespace {

// Produit des dimensions avant et après un axe.
std::pair<std::size_t, std::size_t> outer_inner(const Dims& dims, std::size_t axis) {
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (std::size_t k = 0; k < axis; ++k) outer *= static_cast<std::size_t>(dims[k]);
  for (std::size_t k = axis + 1; k < dims.size(); ++k) inner *= static_cast<std::size_t>(dims[k]);
  return {outer, inner};
}

std::expected<std::vector<std::int64_t>, Refusal> split_sizes(const Node& node, std::span<const Operand> in,
                                                              std::int64_t extent, std::size_t count) {
  auto attrs = attrs_of<SplitAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const std::int64_t num_outputs = (*attrs)->num_outputs;
  const auto parts = static_cast<std::int64_t>(count);
  std::vector<std::int64_t> sizes;
  if (present(in, 1)) {
    if (num_outputs != 0) return graph_error(node, "split et num_outputs à la fois");
    auto given = const_ints(node, in, 1, "les tailles");
    if (!given) return std::unexpected(given.error());
    sizes = std::move(*given);
  } else if (num_outputs != 0) {
    if (num_outputs != parts) return graph_error(node, "num_outputs contredit le nombre de sorties");
    // Opset 18 : parts égales arrondies au-dessus, la dernière plus petite.
    const std::int64_t chunk = (extent + parts - 1) / parts;
    for (std::int64_t k = 0; k < parts; ++k)
      sizes.push_back(std::clamp<std::int64_t>(extent - (k * chunk), 0, chunk));
  } else {
    // Avant l'opset 18 : parts strictement égales.
    if (extent % parts != 0) {
      return shape_error(node, std::format("{} ne se partage pas en {} parts égales", extent, parts));
    }
    sizes.assign(count, extent / parts);
  }
  if (sizes.size() != count) return graph_error(node, "autant de tailles que de sorties");
  std::int64_t total = 0;
  for (const std::int64_t size : sizes) {
    if (size < 0 || size > extent) return shape_error(node, std::format("taille {}", size));
    total += size;
  }
  if (total != extent) return shape_error(node, std::format("tailles de somme {} pour {}", total, extent));
  return sizes;
}

struct SliceAxis {
  std::int64_t start = 0;
  std::int64_t step = 1;
  std::int64_t length = 0;
};

// Bornes d'une découpe selon ONNX : négatifs ramenés, puis bornés selon le signe du pas.
std::expected<std::vector<SliceAxis>, Refusal> slice_plan(const Node& node, std::span<const Operand> in) {
  const Dims& dims = *in[0].dims;
  auto starts = const_ints(node, in, 1, "starts");
  auto ends = const_ints(node, in, 2, "ends");
  if (!starts) return std::unexpected(starts.error());
  if (!ends) return std::unexpected(ends.error());
  std::vector<std::int64_t> axes(starts->size());
  std::vector<std::int64_t> steps(starts->size(), 1);
  std::iota(axes.begin(), axes.end(), std::int64_t{0});
  if (present(in, 3)) {
    auto given = const_ints(node, in, 3, "axes");
    if (!given) return std::unexpected(given.error());
    axes = std::move(*given);
  }
  if (present(in, 4)) {
    auto given = const_ints(node, in, 4, "steps");
    if (!given) return std::unexpected(given.error());
    steps = std::move(*given);
  }
  if (ends->size() != starts->size() || axes.size() != starts->size() || steps.size() != starts->size()) {
    return graph_error(node, "starts, ends, axes et steps de longueurs différentes");
  }

  std::vector<SliceAxis> plan(dims.size());
  std::vector<bool> seen(dims.size(), false);
  for (std::size_t k = 0; k < dims.size(); ++k) plan[k].length = dims[k];
  for (std::size_t k = 0; k < axes.size(); ++k) {
    auto axis = normalize_axis(node, axes[k], dims.size());
    if (!axis) return std::unexpected(axis.error());
    if (seen[*axis]) return graph_error(node, "axe répété");
    seen[*axis] = true;
    const std::int64_t dim = dims[*axis];
    const std::int64_t step = steps[k];
    if (step == 0) return graph_error(node, "pas nul");
    if (dim == 0) {  // rien à découper ; les bornes d'un pas négatif n'existent pas
      plan[*axis] = {.start = 0, .step = step, .length = 0};
      continue;
    }
    std::int64_t start = (*starts)[k];
    std::int64_t end = (*ends)[k];
    // ONNX Runtime lit ends = INT_MAX comme « jusqu'au bout », même à rebours ; la
    // spécification le borne à dim - 1, soit une découpe vide. Sens incertain : refusé.
    if (step < 0 && (end == std::numeric_limits<std::int32_t>::max() ||
                     end == std::numeric_limits<std::int64_t>::max())) {
      return unsupported_error(node, "pas négatif jusqu'à INT_MAX (INT_MIN attendu)");
    }
    if (start < 0) start += dim;
    if (end < 0) end += dim;
    if (step > 0) {
      start = std::clamp<std::int64_t>(start, 0, dim);
      end = std::clamp<std::int64_t>(end, 0, dim);
    } else {
      start = std::clamp<std::int64_t>(start, 0, dim - 1);
      end = std::clamp<std::int64_t>(end, -1, dim - 1);
    }
    const std::int64_t distance = step > 0 ? end - start : start - end;
    const std::int64_t stride = step > 0 ? step : -std::max(step, -std::numeric_limits<std::int64_t>::max());
    plan[*axis] = {.start = start, .step = step, .length = distance <= 0 ? 0 : ((distance - 1) / stride) + 1};
  }
  return plan;
}

struct PadAxis {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

std::expected<std::vector<PadAxis>, Refusal> pad_plan(const Node& node, std::span<const Operand> in) {
  const Dims& dims = *in[0].dims;
  auto pads = const_ints(node, in, 1, "pads");
  if (!pads) return std::unexpected(pads.error());
  std::vector<std::int64_t> axes(dims.size());
  std::iota(axes.begin(), axes.end(), std::int64_t{0});
  if (present(in, 3)) {
    auto given = const_ints(node, in, 3, "axes");
    if (!given) return std::unexpected(given.error());
    axes = std::move(*given);
  }
  if (pads->size() != 2 * axes.size()) return graph_error(node, "pads porte un début et une fin par axe");
  std::vector<PadAxis> plan(dims.size());
  std::vector<bool> seen(dims.size(), false);
  for (std::size_t k = 0; k < axes.size(); ++k) {
    auto axis = normalize_axis(node, axes[k], dims.size());
    if (!axis) return std::unexpected(axis.error());
    if (seen[*axis]) return graph_error(node, "axe répété");
    seen[*axis] = true;
    const std::int64_t begin = (*pads)[k];
    const std::int64_t end = (*pads)[k + axes.size()];
    if (begin < -kMaxDim || begin > kMaxDim || end < -kMaxDim || end > kMaxDim) {
      return shape_error(node, "marge hors de [-2^24, 2^24]");
    }
    plan[*axis] = {.begin = begin, .end = end};
  }
  return plan;
}

// Indice source d'une coordonnée c hors de [0, dim) selon le mode ; kFill pour une constante.
std::int64_t pad_source(PadMode mode, std::int64_t c, std::int64_t dim) {
  if (c >= 0 && c < dim) return c;
  switch (mode) {
    case PadMode::Constant:
      return -1;
    case PadMode::Edge:
      return std::clamp<std::int64_t>(c, 0, dim - 1);
    case PadMode::Reflect:
      return c < 0 ? -c : (2 * (dim - 1)) - c;
  }
  return -1;
}

}  // namespace

Inferred infer_concat(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<AxisAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  Dims out = *in[0].dims;
  auto axis = normalize_axis(node, (*attrs)->axis, out.size());
  if (!axis) return std::unexpected(axis.error());
  for (std::size_t k = 1; k < in.size(); ++k) {
    if (!present(in, k)) return graph_error(node, "entrée absente");
    const Dims& dims = *in[k].dims;
    if (in[k].type != in[0].type || dims.size() != out.size())
      return graph_error(node, "types ou rangs différents");
    for (std::size_t a = 0; a < out.size(); ++a) {
      if (a != *axis && dims[a] != out[a]) return shape_error(node, "formes différentes hors de l'axe");
    }
    out[*axis] += dims[*axis];
  }
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_concat(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<AxisAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  auto normalized = normalize_axis(node, (*attrs)->axis, outs[0].dims.size());
  if (!normalized) return std::unexpected(normalized.error());
  const std::size_t axis = *normalized;
  // Les entrées se suivent dans une seule table : décalage de chacune.
  Value joined{.type = in[0]->type};
  std::vector<std::size_t> offsets;
  for (const Value* value : in) {
    offsets.push_back(value_size(joined));
    joined.f.insert(joined.f.end(), value->f.begin(), value->f.end());
    joined.i.insert(joined.i.end(), value->i.begin(), value->i.end());
  }
  const auto [outer, inner] = outer_inner(outs[0].dims, axis);
  std::vector<std::size_t> source;
  source.reserve(element_count(outs[0].dims));
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t k = 0; k < in.size(); ++k) {
      const std::size_t chunk = static_cast<std::size_t>(in[k]->dims[axis]) * inner;
      for (std::size_t e = 0; e < chunk; ++e) source.push_back(offsets[k] + (o * chunk) + e);
    }
  }
  return std::vector<Value>{gather_elements(joined, outs[0], source)};
}

Inferred infer_split(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<SplitAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const Dims& dims = *in[0].dims;
  auto axis = normalize_axis(node, (*attrs)->axis, dims.size());
  if (!axis) return std::unexpected(axis.error());
  auto sizes = split_sizes(node, in, dims[*axis], node.outputs.size());
  if (!sizes) return std::unexpected(sizes.error());
  std::vector<OutputInfo> outs;
  for (const std::int64_t size : *sizes) {
    Dims part = dims;
    part[*axis] = size;
    outs.push_back({.type = in[0].type, .dims = std::move(part)});
  }
  return outs;
}

Evaluated eval_split(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  const Value& data = *in[0];
  auto attrs = attrs_of<SplitAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  auto normalized = normalize_axis(node, (*attrs)->axis, data.dims.size());
  if (!normalized) return std::unexpected(normalized.error());
  const std::size_t axis = *normalized;
  const auto [outer, inner] = outer_inner(data.dims, axis);
  const std::size_t row = static_cast<std::size_t>(data.dims[axis]) * inner;
  std::vector<Value> result;
  std::size_t first = 0;
  for (const OutputInfo& info : outs) {
    const std::size_t chunk = static_cast<std::size_t>(info.dims[axis]) * inner;
    std::vector<std::size_t> source;
    source.reserve(outer * chunk);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t e = 0; e < chunk; ++e) source.push_back((o * row) + first + e);
    }
    result.push_back(gather_elements(data, info, source));
    first += chunk;
  }
  return result;
}

Inferred infer_slice(const Node& node, std::span<const Operand> in) {
  auto plan = slice_plan(node, in);
  if (!plan) return std::unexpected(plan.error());
  Dims out;
  for (const SliceAxis& axis : *plan) out.push_back(axis.length);
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_slice(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto plan = slice_plan(node, operands_of(in));
  if (!plan) return std::unexpected(plan.error());
  const std::vector<std::size_t> strides = row_strides(in[0]->dims);
  std::vector<std::size_t> source(element_count(outs[0].dims));
  for_each_index(outs[0].dims, [&](std::span<const std::int64_t> index, std::size_t pos) {
    std::size_t at = 0;
    for (std::size_t k = 0; k < index.size(); ++k) {
      const std::int64_t coord = (*plan)[k].start + (index[k] * (*plan)[k].step);
      at += static_cast<std::size_t>(coord) * strides[k];
    }
    source[pos] = at;
  });
  return std::vector<Value>{gather_elements(*in[0], outs[0], source)};
}

Inferred infer_pad(const Node& node, std::span<const Operand> in) {
  auto attrs = attrs_of<PadAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  if (auto ok = expect_float(node, in[0], "l'entrée"); !ok) return std::unexpected(ok.error());
  auto plan = pad_plan(node, in);
  if (!plan) return std::unexpected(plan.error());
  const Dims& dims = *in[0].dims;
  Dims out;
  for (std::size_t k = 0; k < dims.size(); ++k) {
    const PadAxis& p = (*plan)[k];
    out.push_back(dims[k] + p.begin + p.end);
    if (out.back() < 0) return shape_error(node, "marges négatives plus grandes que l'entrée");
    const bool padded = p.begin > 0 || p.end > 0;
    if (padded && (*attrs)->mode == PadMode::Reflect && (p.begin >= dims[k] || p.end >= dims[k])) {
      return shape_error(node, "reflect exige des marges plus petites que la dimension");
    }
    if (padded && dims[k] == 0 && (*attrs)->mode != PadMode::Constant) {
      return shape_error(node, "marge non constante autour d'une dimension vide");
    }
  }
  if (present(in, 2) && (in[2].type != in[0].type || element_count(*in[2].dims) != 1)) {
    return graph_error(node, "constant_value est un scalaire du type de l'entrée");
  }
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_pad(const Node& node, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto attrs = attrs_of<PadAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  auto plan = pad_plan(node, operands_of(in));
  if (!plan) return std::unexpected(plan.error());
  const PadMode mode = (*attrs)->mode;
  const Value& data = *in[0];
  const float fill = in.size() > 2 && in[2] != nullptr ? in[2]->f[0] : 0.0F;
  const std::vector<std::size_t> strides = row_strides(data.dims);
  std::vector<std::size_t> source(element_count(outs[0].dims));
  for_each_index(outs[0].dims, [&](std::span<const std::int64_t> index, std::size_t pos) {
    std::size_t at = 0;
    for (std::size_t k = 0; k < index.size(); ++k) {
      const std::int64_t c = pad_source(mode, index[k] - (*plan)[k].begin, data.dims[k]);
      if (c < 0) {
        source[pos] = kFill;
        return;
      }
      at += static_cast<std::size_t>(c) * strides[k];
    }
    source[pos] = at;
  });
  return std::vector<Value>{gather_elements(data, outs[0], source, fill)};
}

}  // namespace ajp::gpu::ir::detail
