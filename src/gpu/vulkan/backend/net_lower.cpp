// Abaissement des opérations sans réglage : élément par élément, copies à pas, pixels,
// échantillonnage, réductions. Les convolutions sont dans net_lower_conv.cpp.
#include "gpu/vulkan/backend/net_lower.h"

#include <algorithm>
#include <bit>
#include <format>
#include <limits>
#include <variant>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"
#include "gpu/vulkan/backend/net_lower_detail.h"
#include "gpu/vulkan/backend/net_params.h"
#include "gpu/vulkan/backend/net_weights.h"

namespace ajp::gpu::vk::net {

using detail::put;
using detail::put_f;

namespace {

constexpr std::uint32_t kIdentityPerm = 0U | (1U << 2U) | (2U << 4U) | (3U << 6U);

std::unexpected<Refusal> refuse_node(RefusalCode code, const ir::Node& node, std::string_view detail) {
  return std::unexpected(
      Refusal{.code = code, .detail = std::format("{} {} : {}", ir::name(node.op), node.name, detail)});
}

bool broadcastable(const Desc4& in, const Desc4& out) {
  const auto fits = [](std::uint32_t a, std::uint32_t y) { return a == y || a == 1; };
  return fits(in.n, out.n) && fits(in.c, out.c) && fits(in.h, out.h) && fits(in.w, out.w);
}

std::uint32_t count_of(const Desc4& d) {
  return d.n * d.c * d.h * d.w;
}

// Bloc push de net_copy.comp : vues, boîte parcourue, décalage de sortie, départ, pas, perm.
struct CopyPush {
  std::array<std::uint32_t, 4> iter{};
  std::array<std::uint32_t, 4> out_offset{};
  std::array<std::int32_t, 4> start{};
  std::array<std::int32_t, 4> step{1, 1, 1, 1};
  std::uint32_t perm = kIdentityPerm;
};

std::vector<std::uint32_t> copy_push(const Desc4& x, const Desc4& y, const CopyPush& c) {
  std::vector<std::uint32_t> p;
  put(p, x);
  put(p, y);
  p.insert(p.end(), c.iter.begin(), c.iter.end());
  p.insert(p.end(), c.out_offset.begin(), c.out_offset.end());
  for (const std::int32_t v : c.start) p.push_back(std::bit_cast<std::uint32_t>(v));
  for (const std::int32_t v : c.step) p.push_back(std::bit_cast<std::uint32_t>(v));
  p.push_back(c.perm);
  return p;
}

std::array<std::uint32_t, 4> dims_of(const Desc4& d) {
  return {d.n, d.c, d.h, d.w};
}

std::uint32_t activation_code(ir::OpKind op) {
  switch (op) {
    case ir::OpKind::Relu:
      return 1;
    case ir::OpKind::LeakyRelu:
      return 2;
    case ir::OpKind::Sigmoid:
      return 4;
    case ir::OpKind::Tanh:
      return 5;
    case ir::OpKind::Clip:
      return 6;
    default:
      return 0;
  }
}

}  // namespace

namespace detail {

void put(std::vector<std::uint32_t>& push, const Desc4& d) {
  push.insert(push.end(), {d.n, d.c, d.h, d.w, d.order, d.cp});
}

void put_f(std::vector<std::uint32_t>& push, float value) {
  push.push_back(std::bit_cast<std::uint32_t>(value));
}

}  // namespace detail

std::expected<Lowerer::Bound, Refusal> Lowerer::bind(ir::TensorId id) {
  if (id == ir::kNoTensor || id >= graph_.tensors.size()) {
    return std::unexpected(Refusal{RefusalCode::GraphInvalid, "tenseur absent"});
  }
  auto desc = desc_of(graph_, id);
  if (!desc) return std::unexpected(desc.error());
  const ir::Tensor& t = graph_.tensors[id];
  Binding binding{.kind = BindingKind::Tensor, .index = id};
  if (t.kind == ir::TensorKind::Constant) {
    auto found = constants_.find(id);
    if (found == constants_.end()) found = constants_.emplace(id, add_blob(constant_blob(t))).first;
    binding = {.kind = BindingKind::Blob, .index = found->second};
  }
  return Bound{.binding = binding, .desc = *desc, .type = type_code(graph_, id), .tensor = &t};
}

std::uint32_t Lowerer::add_blob(std::vector<std::byte> bytes) {
  program_.blobs.push_back(std::move(bytes));
  return static_cast<std::uint32_t>(program_.blobs.size() - 1);
}

std::expected<std::string, Refusal> Lowerer::shader(std::string_view base,
                                                    std::span<const Bound> tensors) const {
  const bool half = std::ranges::any_of(tensors, [](const Bound& b) { return b.type == 1; });
  if (!half) return std::string(base);
  if (!target_.f16_storage) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable,
                                   std::format("{} : tenseur float16 sans stockage 16 bits", base)});
  }
  return std::format("{}_f16", base);
}

void Lowerer::emit(std::size_t node, std::string shader, std::vector<std::uint32_t> constants,
                   std::vector<Binding> bindings, std::vector<std::uint32_t> push,
                   std::array<std::uint32_t, 3> groups) {
  program_.dispatches.push_back(Dispatch{.node = static_cast<std::uint32_t>(node),
                                         .shader = std::move(shader),
                                         .constants = std::move(constants),
                                         .bindings = std::move(bindings),
                                         .push = std::move(push),
                                         .groups = groups});
}

void Lowerer::emit_linear(std::size_t node, std::string shader, std::vector<std::uint32_t> constants,
                          std::vector<Binding> bindings, std::vector<std::uint32_t> push,
                          std::uint64_t count) {
  // Groupes de 256 invocations sur une grille 2D (linear_invocation de net_tensor.glsl).
  const std::uint64_t groups = std::max<std::uint64_t>(1, (count + 255) / 256);
  const std::uint64_t x = std::min<std::uint64_t>(groups, 32768);
  const std::uint64_t y = (groups + x - 1) / x;
  emit(node, std::move(shader), std::move(constants), std::move(bindings), std::move(push),
       {static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), 1});
}

std::expected<void, Refusal> Lowerer::node(std::size_t index, const KernelChoice& choice) {
  const ir::Node& n = graph_.nodes[index];
  const bool plain = choice.variant == "default";
  switch (n.op) {
    case ir::OpKind::Conv:
      return choice.variant == "coopmat" ? conv_matrix(index, choice) : conv(index, choice);
    case ir::OpKind::ConvTranspose:
      return conv_transpose(index, choice);
    default:
      break;
  }
  if (!plain)
    return refuse_node(RefusalCode::InputMismatch, n, std::format("variante {} inconnue", choice.variant));
  switch (n.op) {
    case ir::OpKind::Relu:
    case ir::OpKind::LeakyRelu:
    case ir::OpKind::Sigmoid:
    case ir::OpKind::Tanh:
    case ir::OpKind::Clip:
    case ir::OpKind::Cast:
    case ir::OpKind::Identity:
    case ir::OpKind::Reshape:
    case ir::OpKind::Unsqueeze:
    case ir::OpKind::Squeeze:
      return unary(index);
    case ir::OpKind::Add:
    case ir::OpKind::Sub:
    case ir::OpKind::Mul:
    case ir::OpKind::Div:
    case ir::OpKind::Mod:
    case ir::OpKind::PRelu:
      return binary(index);
    case ir::OpKind::Concat:
      return concat(index);
    case ir::OpKind::Split:
      return split(index);
    case ir::OpKind::Slice:
      return slice(index);
    case ir::OpKind::Transpose:
      return transpose(index);
    case ir::OpKind::DepthToSpace:
    case ir::OpKind::SpaceToDepth:
      return depth_space(index);
    case ir::OpKind::Pad:
      return pad(index);
    case ir::OpKind::Resize:
      return resize(index);
    case ir::OpKind::GridSample:
      return grid_sample(index);
    case ir::OpKind::ReduceMean:
      return reduce_mean(index);
    case ir::OpKind::AveragePool:
      return average_pool(index);
    default:
      return refuse_node(RefusalCode::OperatorUnknown, n, "aucun noyau Vulkan pour cet opérateur");
  }
}

std::expected<void, Refusal> Lowerer::unary(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const bool flat = x->desc.order == 0 && y->desc.order == 0;
  if (count_of(x->desc) != count_of(y->desc) ||
      (!flat && (x->desc.n != y->desc.n || x->desc.c != y->desc.c || x->desc.h != y->desc.h ||
                 x->desc.w != y->desc.w))) {
    return refuse_node(RefusalCode::ShapeInvalid, n, "formes d'entrée et de sortie incompatibles");
  }
  float alpha = 0.0F;
  float lo = -std::numeric_limits<float>::infinity();
  float hi = std::numeric_limits<float>::infinity();
  if (const auto* leaky = std::get_if<ir::LeakyReluAttrs>(&n.attrs)) alpha = leaky->alpha;
  if (n.op == ir::OpKind::Clip) {
    for (const std::size_t k : {std::size_t{1}, std::size_t{2}}) {
      if (!has_input(n, k)) continue;
      auto bound = const_floats(graph_, n, k);
      if (!bound || bound->size() != 1)
        return refuse_node(RefusalCode::ShapeInvalid, n, "bornes non constantes");
      (k == 1 ? lo : hi) = (*bound)[0];
    }
  }
  const std::array tensors{*x, *y};
  auto name = shader("net_unary", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, y->desc);
  put_f(push, alpha);
  put_f(push, lo);
  put_f(push, hi);
  push.push_back(flat ? 1 : 0);
  emit_linear(index, *name, {x->type, y->type, activation_code(n.op)}, {x->binding, y->binding}, push,
              count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::binary(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto a = bind(n.inputs[0]);
  if (!a) return std::unexpected(a.error());
  auto b = bind(n.inputs[1]);
  if (!b) return std::unexpected(b.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  if (!broadcastable(a->desc, y->desc) || !broadcastable(b->desc, y->desc)) {
    return refuse_node(RefusalCode::ShapeInvalid, n, "diffusion hors des quatre axes");
  }
  std::uint32_t op = 0;
  switch (n.op) {
    case ir::OpKind::Sub:
      op = 1;
      break;
    case ir::OpKind::Mul:
      op = 2;
      break;
    case ir::OpKind::Div:
      op = 3;
      break;
    case ir::OpKind::Mod: {
      const auto* attrs = std::get_if<ir::ModAttrs>(&n.attrs);
      if (attrs == nullptr || !attrs->fmod)
        return refuse_node(RefusalCode::ShapeInvalid, n, "Mod flottant sans fmod");
      op = 4;
      break;
    }
    case ir::OpKind::PRelu:
      op = 5;
      break;
    default:
      break;
  }
  const std::array tensors{*a, *b, *y};
  auto name = shader("net_binary", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, a->desc);
  put(push, b->desc);
  put(push, y->desc);
  emit_linear(index, *name, {a->type, b->type, y->type, op}, {a->binding, b->binding, y->binding}, push,
              count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::concat(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const auto* attrs = std::get_if<ir::AxisAttrs>(&n.attrs);
  if (attrs == nullptr) return refuse_node(RefusalCode::GraphInvalid, n, "attributs absents");
  auto axis = axis4(n, attrs->axis, y->tensor->dims.size());
  if (!axis) return std::unexpected(axis.error());
  std::uint32_t offset = 0;
  for (const ir::TensorId id : n.inputs) {
    auto x = bind(id);
    if (!x) return std::unexpected(x.error());
    const std::array tensors{*x, *y};
    auto name = shader("net_copy", tensors);
    if (!name) return std::unexpected(name.error());
    CopyPush c{.iter = dims_of(x->desc)};
    c.out_offset[*axis] = offset;
    offset += c.iter[*axis];
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
    const std::array tensors{*x, *y};
    auto name = shader("net_copy", tensors);
    if (!name) return std::unexpected(name.error());
    CopyPush c{.iter = dims_of(y->desc)};
    c.start[*axis] = offset;
    offset += static_cast<std::int32_t>(c.iter[*axis]);
    emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, copy_push(x->desc, y->desc, c),
                count_of(y->desc));
  }
  return {};
}

std::expected<void, Refusal> Lowerer::slice(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
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

std::expected<void, Refusal> Lowerer::resize(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const auto* attrs = std::get_if<ir::ResizeAttrs>(&n.attrs);
  if (attrs == nullptr) return refuse_node(RefusalCode::GraphInvalid, n, "attributs absents");
  auto scales = resize_scales(graph_, n);
  if (!scales) return std::unexpected(scales.error());
  const std::array tensors{*x, *y};
  auto name = shader("net_resize", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, y->desc);
  put_f(push, (*scales)[0]);
  put_f(push, (*scales)[1]);
  const std::uint32_t linear = attrs->mode == ir::ResizeMode::Linear ? 1 : 0;
  emit_linear(index, *name,
              {x->type, y->type, linear, static_cast<std::uint32_t>(attrs->coord),
               static_cast<std::uint32_t>(attrs->nearest)},
              {x->binding, y->binding}, push, count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::grid_sample(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto g = bind(n.inputs[1]);
  if (!g) return std::unexpected(g.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const auto* attrs = std::get_if<ir::GridSampleAttrs>(&n.attrs);
  if (attrs == nullptr) return refuse_node(RefusalCode::GraphInvalid, n, "attributs absents");
  const std::array tensors{*x, *g, *y};
  auto name = shader("net_grid_sample", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, g->desc);
  put(push, y->desc);
  emit_linear(index, *name,
              {x->type, g->type, y->type, attrs->mode == ir::GridMode::Nearest ? 1U : 0U,
               attrs->padding == ir::GridPadding::Border ? 1U : 0U, attrs->align_corners ? 1U : 0U},
              {x->binding, g->binding, y->binding}, push, count_of(y->desc));
  return {};
}

std::expected<void, Refusal> Lowerer::reduce_mean(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto reduced = reduced_axes(graph_, n);
  if (!reduced) return std::unexpected(reduced.error());
  if (reduced->empty()) return unary(index);  // noop_with_empty_axes : une copie
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  if (y->desc.order != 0) return refuse_node(RefusalCode::ShapeInvalid, n, "sortie Nhwc16");
  // La sortie vue avec 1 sur chaque axe réduit : keepdims ne change pas la mémoire.
  std::array<std::uint32_t, 4> kept = dims_of(x->desc);
  std::uint32_t mask = 0;
  const std::size_t skip = 4 - reduced->size();
  for (std::size_t k = 0; k < reduced->size(); ++k) {
    if (!(*reduced)[k]) continue;
    mask |= 1U << (skip + k);
    kept[skip + k] = 1;
  }
  const Desc4 out{.n = kept[0], .c = kept[1], .h = kept[2], .w = kept[3], .order = 0, .cp = kept[1]};
  const std::array tensors{*x, *y};
  auto name = shader("net_reduce_mean", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, out);
  push.push_back(mask);
  emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, push, count_of(out));
  return {};
}

std::expected<void, Refusal> Lowerer::average_pool(std::size_t index) {
  const ir::Node& n = graph_.nodes[index];
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  auto win = pool_window(graph_, n);
  if (!win) return std::unexpected(win.error());
  const auto* attrs = std::get_if<ir::PoolAttrs>(&n.attrs);
  const std::array tensors{*x, *y};
  auto name = shader("net_avg_pool", tensors);
  if (!name) return std::unexpected(name.error());
  std::vector<std::uint32_t> push;
  put(push, x->desc);
  put(push, y->desc);
  push.insert(push.end(), {win->kh, win->kw, win->sh, win->sw, win->pt, win->pl,
                           attrs != nullptr && attrs->count_include_pad ? 1U : 0U});
  emit_linear(index, *name, {x->type, y->type}, {x->binding, y->binding}, push, count_of(y->desc));
  return {};
}

std::expected<Program, Refusal> lower(const ir::Graph& graph, std::span<const KernelChoice> choices,
                                      const Target& target) {
  if (choices.size() != graph.nodes.size()) {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch,
                std::format("{} choix pour {} nœuds", choices.size(), graph.nodes.size())});
  }
  Lowerer lowerer(graph, target);
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    if (auto ok = lowerer.node(k, choices[k]); !ok) return std::unexpected(ok.error());
  }
  return lowerer.take();
}

std::expected<std::vector<KernelChoice>, Refusal> default_choices(const ir::Graph& graph,
                                                                  const Target& target) {
  std::vector<KernelChoice> choices;
  choices.reserve(graph.nodes.size());
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    auto options = candidates(graph, k, target);
    if (!options) return std::unexpected(options.error());
    choices.push_back(options->front());
  }
  return choices;
}

}  // namespace ajp::gpu::vk::net
