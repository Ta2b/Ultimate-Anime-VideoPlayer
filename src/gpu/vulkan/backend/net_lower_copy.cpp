// Abaissement des copies à pas (Concat, Split, Slice, Transpose : net_copy.comp) et des
// réarrangements de pixels (DepthToSpace, SpaceToDepth, Pad). Le reste est dans net_lower.cpp.
#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <variant>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"
#include "gpu/vulkan/backend/net_alias.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/backend/net_lower_detail.h"
#include "gpu/vulkan/backend/net_params.h"

namespace ajp::gpu::vk::net {

using detail::copy_push;
using detail::CopyPush;
using detail::count_of;
using detail::dims_of;
using detail::put;
using detail::put_f;
using detail::refuse_node;

std::expected<void, Refusal> Lowerer::concat(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const auto* attrs = std::get_if<ir::AxisAttrs>(&n.attrs);
  if (attrs == nullptr) return refuse_node(RefusalCode::GraphInvalid, n, "attributs absents");
  auto axis = axis4(n, attrs->axis, y->tensor->dims.size());
  if (!axis) return std::unexpected(axis.error());
  const auto rank = static_cast<std::int64_t>(y->tensor->dims.size());
  const auto logical = static_cast<std::size_t>(attrs->axis < 0 ? attrs->axis + rank : attrs->axis);
  std::uint32_t offset = 0;
  std::int64_t start = 0;
  for (const ir::TensorId id : n.inputs) {
    auto x = bind(id);
    if (!x) return std::unexpected(x.error());
    CopyPush c{.iter = dims_of(x->desc)};
    c.out_offset[*axis] = offset;
    offset += c.iter[*axis];
    // Entrée déjà écrite en place par son producteur.
    const auto bytes = slab_offset(*y->tensor, logical, start);
    start += x->tensor->dims[logical];
    if (bytes && aliased(aliases_, id, n.outputs[0], *bytes)) continue;
    const std::array tensors{*x, *y};
    auto name = shader("net_copy", tensors);
    if (!name) return std::unexpected(name.error());
    emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, copy_push(x->desc, y->desc, c),
                count_of(x->desc));
  }
  return {};
}

std::expected<void, Refusal> Lowerer::split(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  const auto* attrs = std::get_if<ir::SplitAttrs>(&n.attrs);
  if (attrs == nullptr) return refuse_node(RefusalCode::GraphInvalid, n, "attributs absents");
  auto axis = axis4(n, attrs->axis, x->tensor->dims.size());
  if (!axis) return std::unexpected(axis.error());
  std::int32_t offset = 0;
  for (const ir::TensorId id : n.outputs) {
    auto y = bind(id);
    if (!y) return std::unexpected(y.error());
    CopyPush c{.iter = dims_of(y->desc)};
    c.start[*axis] = offset;
    offset += static_cast<std::int32_t>(c.iter[*axis]);
    if (id < aliases_.size() && aliases_[id].parent == n.inputs[0]) continue;  // lue dans sa source
    const std::array tensors{*x, *y};
    auto name = shader("net_copy", tensors);
    if (!name) return std::unexpected(name.error());
    emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, copy_push(x->desc, y->desc, c),
                count_of(y->desc));
  }
  return {};
}

std::expected<void, Refusal> Lowerer::slice(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  if (viewed(n)) return {};
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  auto plan = slice_plan(graph_, n);
  if (!plan) return std::unexpected(plan.error());
  const std::size_t skip = 4 - plan->size();
  CopyPush c{.iter = dims_of(y->desc)};
  for (std::size_t k = 0; k < plan->size(); ++k) {
    c.start[skip + k] = static_cast<std::int32_t>((*plan)[k].start);
    c.step[skip + k] = static_cast<std::int32_t>((*plan)[k].step);
  }
  const std::array tensors{*x, *y};
  auto name = shader("net_copy", tensors);
  if (!name) return std::unexpected(name.error());
  emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, copy_push(x->desc, y->desc, c),
              count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::transpose(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const std::size_t rank = x->tensor->dims.size();
  ir::Dims perm;
  if (const auto* attrs = std::get_if<ir::TransposeAttrs>(&n.attrs)) perm = attrs->perm;
  if (perm.empty()) {
    for (std::size_t k = rank; k-- > 0;) perm.push_back(static_cast<std::int64_t>(k));
  }
  if (perm.size() != rank) return refuse_node(RefusalCode::ShapeInvalid, n, "permutation d'un autre rang");
  const std::size_t skip = 4 - rank;
  CopyPush c{.iter = dims_of(y->desc), .perm = 0};
  for (std::size_t k = 0; k < 4; ++k) {
    const auto axis = k < skip
                          ? static_cast<std::uint32_t>(k)
                          : static_cast<std::uint32_t>(perm[k - skip]) + static_cast<std::uint32_t>(skip);
    c.perm |= axis << (2 * k);
  }
  const std::array tensors{*x, *y};
  auto name = shader("net_copy", tensors);
  if (!name) return std::unexpected(name.error());
  emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, copy_push(x->desc, y->desc, c),
              count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::depth_space(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const auto* attrs = std::get_if<ir::BlockAttrs>(&n.attrs);
  if (attrs == nullptr || attrs->blocksize < 1)
    return refuse_node(RefusalCode::GraphInvalid, n, "blocksize absent");
  const std::uint32_t crd = attrs->mode == ir::BlockMode::Crd ? 1 : 0;
  const std::uint32_t mode = (n.op == ir::OpKind::SpaceToDepth ? 2U : 0U) + crd;
  const std::array tensors{*x, *y};
  auto name = shader("net_depth_space", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, y->desc);
  push.push_back(static_cast<std::uint32_t>(attrs->blocksize));
  emit_linear(index, *name, {x->type, y->type, mode}, {x->binding, y->binding}, push, count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::pad(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  auto begins = pad_begins(graph_, n);
  if (!begins) return std::unexpected(begins.error());
  std::uint32_t mode = 0;
  if (const auto* attrs = std::get_if<ir::PadAttrs>(&n.attrs)) {
    mode = attrs->mode == ir::PadMode::Reflect ? 1U : (attrs->mode == ir::PadMode::Edge ? 2U : 0U);
  }
  const std::array tensors{*x, *y};
  auto name = shader("net_pad", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, y->desc);
  std::array<std::int32_t, 4> begin4{};
  const std::size_t skip = 4 - begins->size();
  for (std::size_t k = 0; k < begins->size(); ++k) begin4[skip + k] = static_cast<std::int32_t>((*begins)[k]);
  for (const std::int32_t v : begin4) push.push_back(std::bit_cast<std::uint32_t>(v));
  put_f(push, pad_value(graph_, n));
  emit_linear(index, *name, {x->type, y->type, mode}, {x->binding, y->binding}, push, count_of(y->desc));
  return {};
}

}  // namespace ajp::gpu::vk::net
