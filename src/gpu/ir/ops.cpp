#include "gpu/ir/ops.h"

#include <array>
#include <format>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace ajp::gpu::ir {
namespace {

using namespace detail;

constexpr std::size_t kMany = 4096;  // Concat, Split : borne de l'arité variable

// Une ligne par OpKind, dans l'ordre de l'énumération (vérifié par op_info).
constexpr std::array<OpInfo, 33> kOps{{
    {OpKind::Conv, "Conv", 2, 3, 5, 1, 1, infer_conv, eval_conv},
    {OpKind::ConvTranspose, "ConvTranspose", 2, 3, 3, 1, 1, infer_conv_transpose, eval_conv_transpose},
    {OpKind::Add, "Add", 2, 2, 2, 1, 1, infer_binary, eval_binary},
    {OpKind::Sub, "Sub", 2, 2, 2, 1, 1, infer_binary, eval_binary},
    {OpKind::Mul, "Mul", 2, 2, 2, 1, 1, infer_binary, eval_binary},
    {OpKind::Div, "Div", 2, 2, 2, 1, 1, infer_binary, eval_binary},
    {OpKind::Mod, "Mod", 2, 2, 2, 1, 1, infer_binary, eval_binary},
    {OpKind::Relu, "Relu", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::LeakyRelu, "LeakyRelu", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::PRelu, "PRelu", 2, 2, 2, 1, 1, infer_binary, eval_binary},
    {OpKind::Sigmoid, "Sigmoid", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::Tanh, "Tanh", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::Clip, "Clip", 1, 3, 3, 1, 1, infer_unary, eval_unary},
    {OpKind::Cast, "Cast", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::Identity, "Identity", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::Concat, "Concat", 1, kMany, kMany, 1, 1, infer_concat, eval_concat},
    {OpKind::Split, "Split", 1, 2, 2, 1, kMany, infer_split, eval_split},
    {OpKind::Slice, "Slice", 3, 5, 5, 1, 1, infer_slice, eval_slice},
    {OpKind::Reshape, "Reshape", 2, 2, 2, 1, 1, infer_reshape, eval_reshape},
    {OpKind::Transpose, "Transpose", 1, 1, 1, 1, 1, infer_transpose, eval_transpose},
    {OpKind::Unsqueeze, "Unsqueeze", 2, 2, 2, 1, 1, infer_reshape, eval_reshape},
    {OpKind::Squeeze, "Squeeze", 1, 2, 2, 1, 1, infer_reshape, eval_reshape},
    {OpKind::Shape, "Shape", 1, 1, 1, 1, 1, infer_shape, eval_shape},
    {OpKind::Gather, "Gather", 2, 2, 2, 1, 1, infer_gather, eval_gather},
    {OpKind::DepthToSpace, "DepthToSpace", 1, 1, 1, 1, 1, infer_depth_to_space, eval_depth_to_space},
    {OpKind::SpaceToDepth, "SpaceToDepth", 1, 1, 1, 1, 1, infer_space_to_depth, eval_space_to_depth},
    {OpKind::Pad, "Pad", 2, 4, 4, 1, 1, infer_pad, eval_pad},
    {OpKind::Resize, "Resize", 1, 4, 4, 1, 1, infer_resize, eval_resize},
    {OpKind::GridSample, "GridSample", 2, 2, 2, 1, 1, infer_grid_sample, eval_grid_sample},
    {OpKind::ReduceMean, "ReduceMean", 1, 2, 2, 1, 1, infer_reduce_mean, eval_reduce_mean},
    {OpKind::AveragePool, "AveragePool", 1, 1, 1, 1, 1, infer_average_pool, eval_average_pool},
    {OpKind::Reciprocal, "Reciprocal", 1, 1, 1, 1, 1, infer_unary, eval_unary},
    {OpKind::Range, "Range", 3, 3, 3, 1, 1, infer_range, eval_range},
}};

constexpr bool table_in_order() {
  for (std::size_t k = 0; k < kOps.size(); ++k) {
    if (static_cast<std::size_t>(kOps[k].kind) != k) return false;
  }
  return true;
}
static_assert(table_in_order(), "kOps suit l'ordre de OpKind");

std::expected<void, Refusal> check_arity(const Node& node, std::span<const Operand> in) {
  const OpInfo& info = op_info(node.op);
  if (in.size() < info.min_inputs || in.size() > info.max_inputs) {
    return graph_error(
        node, std::format("{} entrées, {} à {} attendues", in.size(), info.min_inputs, info.max_inputs));
  }
  for (std::size_t k = 0; k < info.min_inputs; ++k) {
    if (!present(in, k)) return graph_error(node, std::format("entrée {} obligatoire absente", k));
  }
  if (node.outputs.size() < info.min_outputs || node.outputs.size() > info.max_outputs) {
    return graph_error(node, std::format("{} sorties", node.outputs.size()));
  }
  return {};
}

std::expected<void, Refusal> check_output(const Node& node, const OutputInfo& out) {
  if (out.dims.size() > kMaxRank)
    return shape_error(node, std::format("rang {} au-delà de {}", out.dims.size(), kMaxRank));
  std::uint64_t count = 1;
  for (const std::int64_t dim : out.dims) {
    if (dim < 0 || dim > kMaxDim)
      return shape_error(node, std::format("dimension {} hors de [0, 2^24]", dim));
    count *= static_cast<std::uint64_t>(dim);
    if (count > kMaxElements)
      return shape_error(node, std::format("sortie {} trop grande", format_dims(out.dims)));
  }
  return {};
}

}  // namespace

const OpInfo& op_info(OpKind op) {
  const auto index = static_cast<std::size_t>(op);
  return kOps[index < kOps.size() ? index : static_cast<std::size_t>(OpKind::Identity)];
}

std::string_view name(OpKind op) {
  return op_info(op).name;
}

std::optional<OpKind> op_from_onnx(std::string_view op_type) {
  for (const OpInfo& info : kOps) {
    if (info.name == op_type) return info.kind;
  }
  return std::nullopt;
}

Inferred infer_node(const Node& node, std::span<const Operand> inputs) {
  if (auto ok = check_arity(node, inputs); !ok) return std::unexpected(ok.error());
  auto outs = op_info(node.op).infer(node, inputs);
  if (!outs) return outs;
  if (outs->size() != node.outputs.size()) return graph_error(node, "nombre de sorties incohérent");
  for (const OutputInfo& out : *outs) {
    if (auto ok = check_output(node, out); !ok) return std::unexpected(ok.error());
  }
  return outs;
}

Evaluated evaluate(const Node& node, std::span<const Value* const> inputs) {
  const std::vector<Operand> operands = operands_of(inputs);
  auto outs = infer_node(node, operands);
  if (!outs) return std::unexpected(outs.error());
  return op_info(node.op).eval(node, inputs, *outs);
}

}  // namespace ajp::gpu::ir
