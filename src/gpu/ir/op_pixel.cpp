// Réarrangements de pixels (DepthToSpace, SpaceToDepth) et ReduceMean.
//
// DCR range les canaux par bloc puis par canal, comme ONNX par défaut ; CRD par
// canal puis par bloc, comme pixel_shuffle et pixel_unshuffle de PyTorch.
#include <algorithm>
#include <format>
#include <optional>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace ajp::gpu::ir::detail {
namespace {

struct Block {
  std::int64_t size = 1;
  BlockMode mode = BlockMode::Dcr;
};

std::expected<Block, Refusal> block_of(const Node& node, const Dims& x) {
  auto attrs = attrs_of<BlockAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  if (auto ok = expect_rank(node, x, 4, "l'entrée"); !ok) return std::unexpected(ok.error());
  const std::int64_t b = (*attrs)->blocksize;
  if (b < 1 || b > 64) return graph_error(node, std::format("blocksize {} hors de [1, 64]", b));
  return Block{.size = b, .mode = (*attrs)->mode};
}

std::size_t at4(const Dims& d, std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t e) {
  return static_cast<std::size_t>((((((a * d[1]) + b) * d[2]) + c) * d[3]) + e);
}

// Axes réduits, ou std::nullopt quand ReduceMean ne fait rien (noop_with_empty_axes).
std::expected<std::optional<std::vector<bool>>, Refusal> reduced_axes(const Node& node,
                                                                      std::span<const Operand> in) {
  auto attrs = attrs_of<ReduceAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const std::size_t rank = in[0].dims->size();
  std::vector<std::int64_t> axes;
  if (present(in, 1)) {
    auto given = const_ints(node, in, 1, "les axes");
    if (!given) return std::unexpected(given.error());
    axes = std::move(*given);
  }
  if (axes.empty() && (*attrs)->noop_with_empty_axes) return std::optional<std::vector<bool>>{};
  std::vector<bool> reduced(rank, axes.empty());
  for (const std::int64_t axis : axes) {
    auto a = normalize_axis(node, axis, rank);
    if (!a) return std::unexpected(a.error());
    reduced[*a] = true;
  }
  return std::optional{std::move(reduced)};
}

}  // namespace

std::vector<std::size_t> depth_to_space_source(const Dims& x, const Dims& out, std::int64_t b,
                                               BlockMode mode) {
  const std::int64_t channels = out[1];
  std::vector<std::size_t> source(element_count(out));
  for_each_index(out, [&](std::span<const std::int64_t> o, std::size_t pos) {
    const std::int64_t i = o[2] % b;
    const std::int64_t j = o[3] % b;
    const std::int64_t c =
        mode == BlockMode::Dcr ? (((i * b) + j) * channels) + o[1] : (o[1] * b * b) + (i * b) + j;
    source[pos] = at4(x, o[0], c, o[2] / b, o[3] / b);
  });
  return source;
}

Inferred infer_depth_to_space(const Node& node, std::span<const Operand> in) {
  const Dims& x = *in[0].dims;
  auto block = block_of(node, x);
  if (!block) return std::unexpected(block.error());
  const std::int64_t b = block->size;
  if (x[1] % (b * b) != 0) return shape_error(node, "canaux non multiples de blocksize²");
  Dims out{x[0], x[1] / (b * b), x[2] * b, x[3] * b};
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_depth_to_space(const Node& node, std::span<const Value* const> in,
                              std::span<const OutputInfo> outs) {
  const Value& x = *in[0];
  auto block = block_of(node, x.dims);
  if (!block) return std::unexpected(block.error());
  return std::vector<Value>{
      gather_elements(x, outs[0], depth_to_space_source(x.dims, outs[0].dims, block->size, block->mode))};
}

Inferred infer_space_to_depth(const Node& node, std::span<const Operand> in) {
  const Dims& x = *in[0].dims;
  auto block = block_of(node, x);
  if (!block) return std::unexpected(block.error());
  const std::int64_t b = block->size;
  if (x[2] % b != 0 || x[3] % b != 0)
    return shape_error(node, "hauteur ou largeur non multiple de blocksize");
  Dims out{x[0], x[1] * b * b, x[2] / b, x[3] / b};
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_space_to_depth(const Node& node, std::span<const Value* const> in,
                              std::span<const OutputInfo> outs) {
  const Value& x = *in[0];
  auto block = block_of(node, x.dims);
  if (!block) return std::unexpected(block.error());
  const std::int64_t b = block->size;
  const bool dcr = block->mode == BlockMode::Dcr;
  const std::int64_t channels = x.dims[1];
  std::vector<std::size_t> source(element_count(outs[0].dims));
  for_each_index(outs[0].dims, [&](std::span<const std::int64_t> o, std::size_t pos) {
    const std::int64_t cell = dcr ? o[1] / channels : o[1] % (b * b);
    const std::int64_t c = dcr ? o[1] % channels : o[1] / (b * b);
    source[pos] = at4(x.dims, o[0], c, (o[2] * b) + (cell / b), (o[3] * b) + (cell % b));
  });
  return std::vector<Value>{gather_elements(x, outs[0], source)};
}

Inferred infer_reduce_mean(const Node& node, std::span<const Operand> in) {
  if (auto ok = expect_float(node, in[0], "l'entrée"); !ok) return std::unexpected(ok.error());
  auto reduced = reduced_axes(node, in);
  if (!reduced) return std::unexpected(reduced.error());
  const Dims& x = *in[0].dims;
  if (!*reduced) return std::vector<OutputInfo>{{.type = in[0].type, .dims = x}};
  auto attrs = attrs_of<ReduceAttrs>(node);
  if (!attrs) return std::unexpected(attrs.error());
  const bool keepdims = (*attrs)->keepdims;
  Dims out;
  for (std::size_t k = 0; k < x.size(); ++k) {
    if (!(**reduced)[k]) {
      out.push_back(x[k]);
    } else if (keepdims) {
      out.push_back(1);
    }
  }
  return std::vector<OutputInfo>{{.type = in[0].type, .dims = std::move(out)}};
}

Evaluated eval_reduce_mean(const Node& node, std::span<const Value* const> in,
                           std::span<const OutputInfo> outs) {
  const Value& x = *in[0];
  auto reduced = reduced_axes(node, operands_of(in));
  if (!reduced) return std::unexpected(reduced.error());
  if (!*reduced) return std::vector<Value>{x};

  // Chaque élément s'ajoute à la case de sortie dont les axes réduits valent 0.
  Dims kept = x.dims;
  for (std::size_t k = 0; k < kept.size(); ++k) {
    if ((**reduced)[k]) kept[k] = 1;
  }
  const std::vector<std::size_t> strides = row_strides(kept);
  Value out = make_output(outs[0]);
  for_each_index(x.dims, [&](std::span<const std::int64_t> index, std::size_t pos) {
    std::size_t at = 0;
    for (std::size_t k = 0; k < index.size(); ++k) {
      if (!(**reduced)[k]) at += static_cast<std::size_t>(index[k]) * strides[k];
    }
    out.f[at] += x.f[pos];
  });
  const std::size_t total = element_count(x.dims);
  const std::size_t per_output = out.f.empty() ? 1 : total / out.f.size();
  for (float& value : out.f) value /= static_cast<float>(per_output);
  return std::vector<Value>{std::move(out)};
}

}  // namespace ajp::gpu::ir::detail
