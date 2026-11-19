#include "gpu/vulkan/backend/net_conv_low_bit.h"

#include <string>
#include <utility>
#include <variant>

#include "gpu/ir/fp8_weights.h"
#include "gpu/vulkan/backend/net_lower_detail.h"
#include "gpu/vulkan/backend/net_params.h"
#include "gpu/vulkan/backend/net_weights.h"
#include "kirari_vk_backend_spirv.h"

namespace kirari::gpu::vk::net::detail {
namespace {

constexpr std::string_view kFp8Shader = "net_conv_coopmat_fp8_f16";
constexpr std::string_view kInt8Shader = "net_conv_coopmat_int8_f16";

const ir::Tensor* constant_weights(const ir::Graph& graph, const ir::Node& node) {
  if (!has_input(node, ir::kConvW)) return nullptr;
  const ir::Tensor& t = graph.tensors[node.inputs[ir::kConvW]];
  return t.kind == ir::TensorKind::Constant ? &t : nullptr;
}

bool weights_on_grid(const ir::Value& weights, ir::QuantGrid grid) {
  return grid == ir::QuantGrid::Int8 ? ir::recover_int8_weights(weights).has_value()
                                     : ir::recover_fp8_weights(weights).has_value();
}

bool tiles_offered(const Target& target, ir::QuantGrid grid) {
  return grid == ir::QuantGrid::Int8 ? target.matrix_i8 : target.matrix_f8;
}

}  // namespace

std::optional<ir::QuantGrid> low_bit_grid(std::string_view variant) {
  if (variant == kMatrixFp8) return ir::QuantGrid::E4m3;
  if (variant == kMatrixInt8) return ir::QuantGrid::Int8;
  return std::nullopt;
}

std::string_view low_bit_shader(ir::QuantGrid grid) {
  return grid == ir::QuantGrid::Int8 ? kInt8Shader : kFp8Shader;
}

float grid_scale(const ir::ConvAttrs& attrs, float scale) {
  return attrs.quant_grid == ir::QuantGrid::Int8 && scale > 0.0F ? -scale : scale;
}

std::uint32_t matrix_subgroup(const Target& target) {
  if (target.subgroup_min != 0 && target.subgroup_min <= 32 && target.subgroup_max >= 32) return 32;
  return target.subgroup_size;
}

bool low_bit_eligible(const ir::Graph& graph, const ir::Node& node, const Target& target,
                      ir::QuantGrid grid) {
  const auto* attrs = std::get_if<ir::ConvAttrs>(&node.attrs);
  if (node.op != ir::OpKind::Conv || attrs == nullptr || !tiles_offered(target, grid) ||
      attrs->quant_grid != grid || !(attrs->in_quant > 0.0F) || attrs->group != 1 ||
      target.subgroup_size == 0)
    return false;
  if (spirv::words(low_bit_shader(grid)).empty()) return false;
  const ir::Tensor& x = graph.tensors[node.inputs[ir::kConvX]];
  const ir::Tensor& y = graph.tensors[node.outputs[0]];
  const ir::Tensor* w = constant_weights(graph, node);
  if (!x.known || !y.known || y.dims.size() != 4 || w == nullptr) return false;
  const ir::Dims result = ir::conv_result_dims(y.dims, attrs->epilogue);
  return x.type == ir::DataType::F16 && x.layout == ir::Layout::Nhwc16 && (result[1] + 15) / 16 <= 16 &&
         weights_on_grid(w->value, grid);
}

void add_low_bit_candidates(const ir::Graph& graph, const ir::Node& node, const Target& target,
                            std::vector<KernelChoice>& out) {
  for (const ir::QuantGrid grid : {ir::QuantGrid::E4m3, ir::QuantGrid::Int8}) {
    if (!low_bit_eligible(graph, node, target, grid)) continue;
    const std::string variant(grid == ir::QuantGrid::Int8 ? kMatrixInt8 : kMatrixFp8);
    const auto& attrs = std::get<ir::ConvAttrs>(node.attrs);
    const ir::Dims result = ir::conv_result_dims(graph.tensors[node.outputs[0]].dims, attrs.epilogue);
    const auto tiles = static_cast<std::uint32_t>((result[1] + 15) / 16);
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(result[2]) * static_cast<std::uint64_t>(result[3]);
    for (const std::uint32_t subgroups : {4U, 2U, 8U}) {
      for (const std::uint32_t tm : {2U, 1U, 4U}) {
        const std::uint64_t block = static_cast<std::uint64_t>(subgroups) * tm * 16;
        if ((pixels + block - 1) / block > target.max_workgroup_count_x) continue;
        if (tm * tiles > 16 || subgroups * matrix_subgroup(target) > target.max_workgroup_invocations ||
            matrix_shared_bytes(subgroups, tm, 16, 0) > target.max_shared_memory) {
          continue;
        }
        out.push_back({.variant = variant, .tuning = {tm, subgroups}});
        if (tm == 2 && matrix_shared_bytes(subgroups, tm, 16, tiles * 16) <= target.max_shared_memory)
          out.push_back({.variant = variant, .tuning = {tm, subgroups, 1}});
      }
    }
  }
}

std::optional<LowBitBlobs> low_bit_blobs(const ir::Tensor& weights, const ir::Tensor* bias,
                                         ir::QuantGrid grid, float in_scale, std::uint32_t cin_padded,
                                         std::uint32_t cout_padded) {
  std::vector<std::uint8_t> bytes;
  std::vector<float> scale;
  if (grid == ir::QuantGrid::Int8) {
    auto recovered = ir::recover_int8_weights(weights.value);
    if (!recovered) return std::nullopt;
    bytes = std::move(recovered->bytes);
    scale = std::move(recovered->scale);
  } else {
    auto recovered = ir::recover_fp8_weights(weights.value);
    if (!recovered) return std::nullopt;
    bytes = std::move(recovered->bytes);
    scale = std::move(recovered->scale);
  }
  return LowBitBlobs{.weights = matrix_weights_fp8(weights.dims, bytes, cin_padded, cout_padded),
                     .bias = fp8_bias_and_scales(bias, scale, in_scale, cout_padded)};
}

}  // namespace kirari::gpu::vk::net::detail
