#include "gpu/vulkan/backend/net_params.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <string_view>
#include <variant>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::vk::net {
namespace {

std::unexpected<Refusal> node_refusal(const ir::Node& node, std::string_view detail) {
  return std::unexpected(Refusal{.code = RefusalCode::ShapeInvalid,
                                 .detail = std::format("{} {} : {}", ir::name(node.op), node.name, detail)});
}

std::uint32_t u32(std::int64_t value) {
  return static_cast<std::uint32_t>(value);
}

// Marges SAME : le surplus va en fin (UPPER) ou en début (LOWER) ; op_window.cpp, same_pads.
std::int64_t same_begin(std::int64_t total, bool upper) {
  const std::int64_t small = total / 2;
  return upper ? small : total - small;
}

const ir::Tensor& input_tensor(const ir::Graph& graph, const ir::Node& node, std::size_t k) {
  return graph.tensors[node.inputs[k]];
}

template <class A>
std::expected<const A*, Refusal> attrs_of(const ir::Node& node) {
  const A* attrs = std::get_if<A>(&node.attrs);
  if (attrs == nullptr) return node_refusal(node, "attributs d'un autre opérateur");
  return attrs;
}

bool in_range(std::int64_t v, std::int64_t lo) {
  return v >= lo && v <= ir::kMaxDim;
}

}  // namespace

bool has_input(const ir::Node& node, std::size_t input) {
  return input < node.inputs.size() && node.inputs[input] != ir::kNoTensor;
}

std::expected<Desc4, Refusal> desc_of(const ir::Graph& graph, ir::TensorId id) {
  const ir::Tensor& t = graph.tensors[id];
  if (!t.known)
    return std::unexpected(Refusal{RefusalCode::ShapeInvalid, std::format("{} : forme inconnue", t.name)});
  if (t.type == ir::DataType::I64) {
    return std::unexpected(Refusal{RefusalCode::ShapeInvalid,
                                   std::format("{} : entiers I64 calculés pendant l'exécution", t.name)});
  }
  if (t.dims.size() > 4) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeInvalid, std::format("{} : rang {}, 4 au plus", t.name, t.dims.size())});
  }
  std::array<std::uint32_t, 4> d{1, 1, 1, 1};
  const std::size_t skip = 4 - t.dims.size();
  for (std::size_t k = 0; k < t.dims.size(); ++k) {
    if (t.dims[k] < 1) {
      return std::unexpected(Refusal{RefusalCode::ShapeInvalid, std::format("{} : dimension vide", t.name)});
    }
    d[skip + k] = u32(t.dims[k]);
  }
  Desc4 desc{.n = d[0], .c = d[1], .h = d[2], .w = d[3]};
  if (t.layout == ir::Layout::Nhwc16) {
    desc.order = 1;
    desc.cp = u32(ir::align_up(t.dims[1], 16));
  } else {
    desc.cp = desc.c;
  }
  return desc;
}

std::uint32_t type_code(const ir::Graph& graph, ir::TensorId id) {
  return graph.tensors[id].type == ir::DataType::F16 ? 1U : 0U;
}

std::expected<std::uint32_t, Refusal> axis4(const ir::Node& node, std::int64_t axis, std::size_t rank) {
  const auto r = static_cast<std::int64_t>(rank);
  const std::int64_t a = axis < 0 ? axis + r : axis;
  if (a < 0 || a >= r || rank > 4)
    return node_refusal(node, std::format("axe {} hors du rang {}", axis, rank));
  return u32(a + 4 - r);
}

std::expected<std::vector<std::int64_t>, Refusal> const_ints(const ir::Graph& graph, const ir::Node& node,
                                                             std::size_t input) {
  if (!has_input(node, input)) return node_refusal(node, std::format("entrée {} absente", input));
  const ir::Tensor& t = input_tensor(graph, node, input);
  if (t.kind != ir::TensorKind::Constant || t.type != ir::DataType::I64) {
    return node_refusal(node, std::format("entrée {} ({}) : entiers constants attendus", input, t.name));
  }
  return t.value.i;
}

std::expected<std::vector<float>, Refusal> const_floats(const ir::Graph& graph, const ir::Node& node,
                                                        std::size_t input) {
  if (!has_input(node, input)) return node_refusal(node, std::format("entrée {} absente", input));
  const ir::Tensor& t = input_tensor(graph, node, input);
  if (t.kind != ir::TensorKind::Constant || t.type == ir::DataType::I64) {
    return node_refusal(node, std::format("entrée {} ({}) : flottants constants attendus", input, t.name));
  }
  return t.value.f;
}

std::expected<Window, Refusal> conv_window(const ir::Graph& graph, const ir::Node& node) {
  auto attrs = attrs_of<ir::ConvAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const ir::ConvAttrs& a = **attrs;
  const ir::Dims& x = input_tensor(graph, node, ir::kConvX).dims;
  const ir::Dims& w = input_tensor(graph, node, ir::kConvW).dims;
  const ir::Dims& y = graph.tensors[node.outputs[0]].dims;
  if (x.size() != 4 || w.size() != 4 || y.size() != 4) return node_refusal(node, "rang 4 attendu");
  Window win{.kh = u32(w[2]),
             .kw = u32(w[3]),
             .sh = u32(a.strides[0]),
             .sw = u32(a.strides[1]),
             .dh = u32(a.dilations[0]),
             .dw = u32(a.dilations[1])};
  std::array<std::int64_t, 2> begin{a.pads[0], a.pads[1]};
  for (std::size_t k = 0; k < 2; ++k) {
    const std::int64_t span = ((w[k + 2] - 1) * a.dilations[k]) + 1;
    if (a.auto_pad == ir::AutoPad::SameUpper || a.auto_pad == ir::AutoPad::SameLower) {
      const std::int64_t needed = ((y[k + 2] - 1) * a.strides[k]) + span - x[k + 2];
      begin[k] = same_begin(std::max<std::int64_t>(0, needed), a.auto_pad == ir::AutoPad::SameUpper);
    } else if (a.auto_pad == ir::AutoPad::Valid) {
      begin[k] = 0;
    }
    if (!in_range(begin[k], 0)) return node_refusal(node, "marge hors de ses bornes");
  }
  win.pt = u32(begin[0]);
  win.pl = u32(begin[1]);
  return win;
}

std::expected<Window, Refusal> conv_transpose_window(const ir::Graph& graph, const ir::Node& node) {
  auto attrs = attrs_of<ir::ConvTransposeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const ir::ConvTransposeAttrs& a = **attrs;
  const ir::Dims& x = input_tensor(graph, node, 0).dims;
  const ir::Dims& w = input_tensor(graph, node, 1).dims;
  const ir::Dims& y = graph.tensors[node.outputs[0]].dims;
  if (x.size() != 4 || w.size() != 4 || y.size() != 4) return node_refusal(node, "rang 4 attendu");
  Window win{.kh = u32(w[2]),
             .kw = u32(w[3]),
             .sh = u32(a.strides[0]),
             .sw = u32(a.strides[1]),
             .dh = u32(a.dilations[0]),
             .dw = u32(a.dilations[1])};
  const bool explicit_out = !a.output_shape.empty();
  const bool same = a.auto_pad == ir::AutoPad::SameUpper || a.auto_pad == ir::AutoPad::SameLower;
  std::array<std::int64_t, 2> begin{a.pads[0], a.pads[1]};
  for (std::size_t k = 0; k < 2; ++k) {
    const std::int64_t span = ((w[k + 2] - 1) * a.dilations[k]) + 1;
    const std::int64_t full = (a.strides[k] * (x[k + 2] - 1)) + a.output_padding[k] + span;
    if (explicit_out || same) {
      begin[k] = same_begin(full - y[k + 2], a.auto_pad == ir::AutoPad::SameUpper);
    } else if (a.auto_pad == ir::AutoPad::Valid) {
      begin[k] = 0;
    }
    if (!in_range(begin[k], 0)) return node_refusal(node, "marge hors de ses bornes");
  }
  win.pt = u32(begin[0]);
  win.pl = u32(begin[1]);
  return win;
}

std::expected<Window, Refusal> pool_window(const ir::Graph& graph, const ir::Node& node) {
  auto attrs = attrs_of<ir::PoolAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const ir::PoolAttrs& a = **attrs;
  const ir::Dims& x = input_tensor(graph, node, 0).dims;
  const ir::Dims& y = graph.tensors[node.outputs[0]].dims;
  if (x.size() != 4 || y.size() != 4) return node_refusal(node, "rang 4 attendu");
  Window win{
      .kh = u32(a.kernel[0]), .kw = u32(a.kernel[1]), .sh = u32(a.strides[0]), .sw = u32(a.strides[1])};
  std::array<std::int64_t, 2> begin{a.pads[0], a.pads[1]};
  for (std::size_t k = 0; k < 2; ++k) {
    if (a.auto_pad == ir::AutoPad::SameUpper || a.auto_pad == ir::AutoPad::SameLower) {
      const std::int64_t needed = ((y[k + 2] - 1) * a.strides[k]) + a.kernel[k] - x[k + 2];
      if (needed < 0) return node_refusal(node, "auto_pad SAME avec un pas plus grand que le noyau");
      begin[k] = same_begin(needed, a.auto_pad == ir::AutoPad::SameUpper);
    } else if (a.auto_pad == ir::AutoPad::Valid) {
      begin[k] = 0;
    }
    if (!in_range(begin[k], 0)) return node_refusal(node, "marge hors de ses bornes");
  }
  win.pt = u32(begin[0]);
  win.pl = u32(begin[1]);
  return win;
}

std::expected<std::array<float, 2>, Refusal> resize_scales(const ir::Graph& graph, const ir::Node& node) {
  const ir::Dims& x = input_tensor(graph, node, 0).dims;
  const ir::Dims& y = graph.tensors[node.outputs[0]].dims;
  const std::size_t rank = x.size();
  if (rank < 2 || y.size() != rank) return node_refusal(node, "rang 2 au moins");
  auto attrs = attrs_of<ir::ResizeAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  std::vector<std::size_t> axes;
  for (const std::int64_t axis : (*attrs)->axes) {
    axes.push_back(static_cast<std::size_t>(axis < 0 ? axis + static_cast<std::int64_t>(rank) : axis));
  }
  if (axes.empty()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), std::size_t{0});
  }
  std::vector<float> scales(rank, 1.0F);
  const bool has_scales = has_input(node, 2) && !input_tensor(graph, node, 2).value.f.empty();
  if (has_scales) {
    auto given = const_floats(graph, node, 2);
    if (!given) return std::unexpected(given.error());
    if (given->size() != axes.size()) return node_refusal(node, "une échelle par axe");
    for (std::size_t k = 0; k < axes.size(); ++k) scales[axes[k]] = (*given)[k];
  } else {
    for (const std::size_t axis : axes) {
      if (axis >= rank) return node_refusal(node, "axe hors du rang");
      scales[axis] = static_cast<float>(y[axis]) / static_cast<float>(x[axis]);
    }
  }
  return std::array<float, 2>{scales[rank - 2], scales[rank - 1]};
}

std::expected<std::vector<SliceAxis>, Refusal> slice_plan(const ir::Graph& graph, const ir::Node& node) {
  const ir::Dims& dims = input_tensor(graph, node, 0).dims;
  auto starts = const_ints(graph, node, 1);
  auto ends = const_ints(graph, node, 2);
  if (!starts) return std::unexpected(starts.error());
  if (!ends) return std::unexpected(ends.error());
  std::vector<std::int64_t> axes(starts->size());
  std::iota(axes.begin(), axes.end(), std::int64_t{0});
  std::vector<std::int64_t> steps(starts->size(), 1);
  if (has_input(node, 3)) {
    auto given = const_ints(graph, node, 3);
    if (!given) return std::unexpected(given.error());
    axes = std::move(*given);
  }
  if (has_input(node, 4)) {
    auto given = const_ints(graph, node, 4);
    if (!given) return std::unexpected(given.error());
    steps = std::move(*given);
  }
  if (ends->size() != starts->size() || axes.size() != starts->size() || steps.size() != starts->size()) {
    return node_refusal(node, "starts, ends, axes et steps de longueurs différentes");
  }
  std::vector<SliceAxis> plan(dims.size());
  for (std::size_t k = 0; k < dims.size(); ++k) plan[k].length = dims[k];
  for (std::size_t k = 0; k < axes.size(); ++k) {
    const auto rank = static_cast<std::int64_t>(dims.size());
    const std::int64_t axis = axes[k] < 0 ? axes[k] + rank : axes[k];
    if (axis < 0 || axis >= rank || steps[k] == 0) return node_refusal(node, "axe ou pas invalide");
    const std::int64_t dim = dims[static_cast<std::size_t>(axis)];
    const std::int64_t step = steps[k];
    std::int64_t start = (*starts)[k];
    std::int64_t end = (*ends)[k];
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
    plan[static_cast<std::size_t>(axis)] = {
        .start = start, .step = step, .length = distance <= 0 ? 0 : ((distance - 1) / stride) + 1};
  }
  return plan;
}

std::expected<std::vector<std::int64_t>, Refusal> pad_begins(const ir::Graph& graph, const ir::Node& node) {
  const std::size_t rank = input_tensor(graph, node, 0).dims.size();
  auto pads = const_ints(graph, node, 1);
  if (!pads) return std::unexpected(pads.error());
  std::vector<std::int64_t> axes(rank);
  std::iota(axes.begin(), axes.end(), std::int64_t{0});
  if (has_input(node, 3)) {
    auto given = const_ints(graph, node, 3);
    if (!given) return std::unexpected(given.error());
    axes = std::move(*given);
  }
  if (pads->size() != 2 * axes.size()) return node_refusal(node, "pads porte un début et une fin par axe");
  std::vector<std::int64_t> begins(rank, 0);
  for (std::size_t k = 0; k < axes.size(); ++k) {
    const std::int64_t axis = axes[k] < 0 ? axes[k] + static_cast<std::int64_t>(rank) : axes[k];
    if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) return node_refusal(node, "axe hors du rang");
    begins[static_cast<std::size_t>(axis)] = (*pads)[k];
  }
  return begins;
}

float pad_value(const ir::Graph& graph, const ir::Node& node) {
  if (!has_input(node, 2)) return 0.0F;
  const ir::Tensor& t = input_tensor(graph, node, 2);
  return t.value.f.empty() ? 0.0F : t.value.f[0];
}

std::expected<std::vector<bool>, Refusal> reduced_axes(const ir::Graph& graph, const ir::Node& node) {
  auto attrs = attrs_of<ir::ReduceAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const std::size_t rank = input_tensor(graph, node, 0).dims.size();
  std::vector<std::int64_t> axes;
  if (has_input(node, 1)) {
    auto given = const_ints(graph, node, 1);
    if (!given) return std::unexpected(given.error());
    axes = std::move(*given);
  }
  if (axes.empty() && (*attrs)->noop_with_empty_axes) return std::vector<bool>{};
  std::vector<bool> reduced(rank, axes.empty());
  for (const std::int64_t axis : axes) {
    const std::int64_t a = axis < 0 ? axis + static_cast<std::int64_t>(rank) : axis;
    if (a < 0 || a >= static_cast<std::int64_t>(rank)) return node_refusal(node, "axe hors du rang");
    reduced[static_cast<std::size_t>(a)] = true;
  }
  return reduced;
}

}  // namespace ajp::gpu::vk::net
