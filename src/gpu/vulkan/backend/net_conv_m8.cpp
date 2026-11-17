#include "gpu/vulkan/backend/net_conv_m8.h"

#include <array>
#include <string>
#include <variant>

#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/backend/net_lower_detail.h"
#include "gpu/vulkan/backend/net_params.h"
#include "gpu/vulkan/backend/net_weights.h"

namespace ajp::gpu::vk::net {
namespace detail {
namespace {

constexpr std::uint32_t kSubgroup = 16;
constexpr std::uint32_t kMaxGroups = 65535;

const ir::Tensor* constant_input(const ir::Graph& graph, const ir::Node& node, std::size_t k) {
  if (!has_input(node, k)) return nullptr;
  const ir::Tensor& t = graph.tensors[node.inputs[k]];
  return t.kind == ir::TensorKind::Constant ? &t : nullptr;
}

// Le réglage tient-il dans l'appareil ? Au plus 16 tuiles de sommes par sous-groupe (tm × tiles), les groupes
// du dispatch dans la limite en x, les invocations du groupe, la mémoire partagée au plus petit paquet de
// canaux (16) avec B partagée sur np canaux, ou sans B partagée (np = 0).
bool tuning_fits(const Target& target, std::uint64_t pixels, std::uint32_t tiles, std::uint32_t subgroups,
                 std::uint32_t tm, std::uint32_t np) {
  if (tm < 1 || tm > 16 || subgroups < 1 || subgroups > 64 || tm * tiles > 16) return false;
  const std::uint64_t block = static_cast<std::uint64_t>(subgroups) * tm * kMatrixM8Pixels;
  return (pixels + block - 1) / block <= target.max_workgroup_count_x &&
         subgroups * kSubgroup <= target.max_workgroup_invocations &&
         matrix_m8_shared_bytes(subgroups, tm, 16, np) <= target.max_shared_memory;
}

// Canaux d'entrée par étape : le plus grand de 64, 32, 16 qui divise cinp et tient en mémoire partagée (comme
// matrix_step_channels, net_lower_conv.cpp).
std::uint32_t step_channels(std::uint32_t cinp, std::uint32_t subgroups, std::uint32_t tm, std::uint32_t np,
                            std::uint32_t max_shared) {
  for (const std::uint32_t kc : {64U, 32U}) {
    if (cinp % kc == 0 && matrix_m8_shared_bytes(subgroups, tm, kc, np) <= max_shared) return kc;
  }
  return 16;
}

}  // namespace

std::uint32_t matrix_m8_subgroup(const Target& target) {
  return target.subgroup_min != 0 && target.subgroup_min <= kSubgroup && target.subgroup_max >= kSubgroup
             ? kSubgroup
             : 0;
}

std::uint32_t matrix_m8_shared_bytes(std::uint32_t subgroups, std::uint32_t tm, std::uint32_t kc,
                                     std::uint32_t np) {
  const std::uint32_t row = (kc / 8) + ((kc / 32) + 1) / 2;  // AS : uvec4 d'une ligne de A
  const std::uint32_t b_quads = np == 0 ? 1 : kc * np / 8;   // BBUF : un uvec4 sans B partagée
  return (2 * subgroups * tm * kMatrixM8Pixels * row * 16) + (subgroups * kMatrixM8Pixels * 16 * 4) +
         (2 * b_quads * 16);
}

bool matrix_m8_eligible(const ir::Graph& graph, const ir::Node& node, const Target& target) {
  const auto* attrs = std::get_if<ir::ConvAttrs>(&node.attrs);
  if (node.op != ir::OpKind::Conv || attrs == nullptr || !target.matrix_m8 ||
      matrix_m8_subgroup(target) == 0 || attrs->group != 1 || attrs->in_quant > 0.0F ||
      attrs->out_quant > 0.0F || constant_input(graph, node, ir::kConvW) == nullptr)
    return false;
  const ir::Tensor& x = graph.tensors[node.inputs[ir::kConvX]];
  const ir::Tensor& y = graph.tensors[node.outputs[0]];
  if (!x.known || !y.known || y.dims.size() != 4) return false;
  const ir::Dims result = ir::conv_result_dims(y.dims, attrs->epilogue);
  return x.type == ir::DataType::F16 && x.layout == ir::Layout::Nhwc16 && (result[1] + 15) / 16 <= 16;
}

void add_matrix_m8_candidates(const ir::Graph& graph, const ir::Node& node, const Target& target,
                              std::vector<KernelChoice>& out) {
  if (!matrix_m8_eligible(graph, node, target)) return;
  const auto& attrs = std::get<ir::ConvAttrs>(node.attrs);
  const ir::Dims result = ir::conv_result_dims(graph.tensors[node.outputs[0]].dims, attrs.epilogue);
  const auto tiles = static_cast<std::uint32_t>((result[1] + 15) / 16);
  const std::uint64_t pixels = static_cast<std::uint64_t>(result[2]) * static_cast<std::uint64_t>(result[3]);
  for (const std::uint32_t subgroups : {4U, 2U, 8U}) {
    for (const std::uint32_t tm : {2U, 1U, 4U}) {
      if (!tuning_fits(target, pixels, tiles, subgroups, tm, 0)) continue;
      out.push_back({.variant = std::string(kMatrixM8), .tuning = {tm, subgroups}});
      if (tm == 2 && tuning_fits(target, pixels, tiles, subgroups, tm, tiles * 16))
        out.push_back({.variant = std::string(kMatrixM8), .tuning = {tm, subgroups, 1}});
    }
  }
}

}  // namespace detail

std::expected<void, Refusal> Lowerer::conv_matrix_m8(std::size_t index, const KernelChoice& choice) {
  const ir::Node& n = graph_.nodes[index];
  if (choice.variant != detail::kMatrixM8 || choice.tuning.size() < 2 || choice.tuning.size() > 3)
    return detail::refuse_node(RefusalCode::InputMismatch, n, "réglage coopmat_m8 incomplet");
  if (!detail::matrix_m8_eligible(graph_, n, target_)) {
    return detail::refuse_node(RefusalCode::FeatureUnavailable, n,
                               "tuiles 8 × 16 × 16 indisponibles pour ce nœud");
  }
  const std::uint32_t tm = choice.tuning[0];
  const std::uint32_t subgroups = choice.tuning[1];
  const std::uint32_t shared_b = choice.tuning.size() == 3 ? choice.tuning[2] : 0;  // B en mémoire partagée
  const auto& attrs = std::get<ir::ConvAttrs>(n.attrs);
  auto x = bind(n.inputs[ir::kConvX]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const Desc4 yc =
      detail::result_desc(y->desc, attrs.epilogue);  // forme calculée, avant le DepthToSpace fondu
  const std::uint32_t tiles = (yc.c + 15) / 16;
  const std::uint32_t np_shared = shared_b == 1 ? tiles * 16 : 0;
  const std::uint64_t pixels = static_cast<std::uint64_t>(yc.h) * yc.w;
  if (shared_b > 1 || !detail::tuning_fits(target_, pixels, tiles, subgroups, tm, np_shared) ||
      x->desc.n > detail::kMaxGroups)
    return detail::refuse_node(RefusalCode::InputMismatch, n, "réglage coopmat_m8 hors de ses bornes");
  auto win = conv_window(graph_, n);
  if (!win) return std::unexpected(win.error());

  detail::ConvConstants e{.act = static_cast<std::uint32_t>(attrs.epilogue.act),
                          .alpha = attrs.epilogue.alpha,
                          .lo = attrs.epilogue.lo,
                          .hi = attrs.epilogue.hi};
  Binding residual{};
  if (attrs.epilogue.residual || attrs.epilogue.gate) {
    auto r = bind(n.inputs[ir::kConvResidual]);
    if (!r) return std::unexpected(r.error());
    e.tr = r->type;
    e.lr = r->desc.order;
    e.crp = r->desc.cp;
    e.residual = attrs.epilogue.gate ? 2U : 1U;
    residual = r->binding;
  }
  Binding slope{};
  if (attrs.epilogue.act == ir::Activation::PRelu) {
    const ir::Tensor* s = detail::constant_input(graph_, n, ir::kConvSlope);
    if (s == nullptr) return detail::refuse_node(RefusalCode::ShapeInvalid, n, "pente PRelu non constante");
    slope = {.kind = BindingKind::Blob, .index = add_blob(channel_floats(s, yc.c))};
  }
  // Constantes 30 à 39 de net_conv_coopmat.glsl : TM, WG, SGS, KC, SPACE, SPACE_CRD, SHARED_B, puis
  // DIRECT_EPI, HALO et NTG à 0 (absents du noyau _m8).
  const std::uint32_t sgs = detail::matrix_m8_subgroup(target_);
  std::vector<std::uint32_t> constants = detail::conv_constants(x->desc, x->type, yc, y->type, *win, 1, e);
  constants.push_back(tm);
  constants.push_back(subgroups * sgs);
  constants.push_back(sgs);
  constants.push_back(detail::step_channels(x->desc.cp, subgroups, tm, np_shared, target_.max_shared_memory));
  for (const std::uint32_t c : detail::space_constants(attrs.epilogue)) constants.push_back(c);
  constants.push_back(shared_b);
  constants.push_back(0);
  constants.push_back(0);
  constants.push_back(0);
  const Binding w{
      .kind = BindingKind::Blob,
      .index = add_blob(matrix_weights(graph_.tensors[n.inputs[ir::kConvW]], x->desc.cp, tiles * 16))};
  const Binding b{.kind = BindingKind::Blob,
                  .index = add_blob(channel_floats(detail::constant_input(graph_, n, ir::kConvB), yc.c))};
  const std::uint64_t bm = static_cast<std::uint64_t>(subgroups) * tm * detail::kMatrixM8Pixels;
  emit(index, std::string(detail::kMatrixM8Shader), std::move(constants),
       {x->binding, w, b, slope, residual, y->binding}, {},
       {static_cast<std::uint32_t>((pixels + bm - 1) / bm), 1, x->desc.n});
  // Toujours exigé, même s'il égale subgroupSize : sans lui, un pilote Intel choisit SIMD8, 16 ou 32 shader
  // par shader, et la tuile se répartirait sur un autre nombre d'invocations que celui que SGS suppose.
  program_.dispatches.back().subgroup = sgs;
  return {};
}

}  // namespace ajp::gpu::vk::net
