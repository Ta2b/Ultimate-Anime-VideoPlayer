// Abaissement des convolutions et choix des candidats de chaque nœud : Conv directe (float ou
// fp16 empaqueté), Conv 3 × 3 par Winograd, Conv par matrices coopératives, ConvTranspose directe.
#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <optional>
#include <variant>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/backend/net_lower_detail.h"
#include "gpu/vulkan/backend/net_params.h"
#include "gpu/vulkan/backend/net_weights.h"

namespace ajp::gpu::vk::net {

std::uint32_t detail::matrix_shared_bytes(std::uint32_t subgroups, std::uint32_t tm, std::uint32_t kc,
                                          std::uint32_t np) {
  const std::uint32_t row = (kc / 8) + ((kc / 32) + 1) / 2;
  const std::uint32_t b_quads = np == 0 ? 1 : kc * np / 8;  // BBUF : un uvec4 sans B partagé
  return (2 * subgroups * tm * 16 * row * 16) + (subgroups * 256 * 4) + (2 * b_quads * 16);
}

namespace {

constexpr std::string_view kDirect = "direct";
constexpr std::string_view kPacked = "direct_f16p";
constexpr std::string_view kMatrix = "coopmat";
constexpr std::string_view kMatrixAcc16 = "coopmat_acc16";     // sommes en float16
constexpr std::string_view kWinograd = "winograd";             // F(2×2, 3×3)
constexpr std::string_view kWinogradPacked = "winograd_f16p";  // F(2×2, 3×3) en fp16 empaqueté
constexpr std::uint32_t kMaxGroups = 65535;

std::unexpected<Refusal> refuse_node(RefusalCode code, const ir::Node& node, std::string_view detail) {
  return std::unexpected(
      Refusal{.code = code, .detail = std::format("{} {} : {}", ir::name(node.op), node.name, detail)});
}

std::uint32_t div_up(std::uint32_t a, std::uint32_t b) {
  return (a + b - 1) / b;
}

// Quads par défaut : 16 canaux de sortie par invocation quand il y en a assez.
std::uint32_t default_quads(std::uint32_t cout_g) {
  return cout_g >= 16 ? 4 : (cout_g >= 8 ? 2 : 1);
}

const ir::Tensor* constant_input(const ir::Graph& graph, const ir::Node& node, std::size_t k) {
  if (!has_input(node, k)) return nullptr;
  const ir::Tensor& t = graph.tensors[node.inputs[k]];
  return t.kind == ir::TensorKind::Constant ? &t : nullptr;
}

std::uint32_t groups_of(const ir::Node& node) {
  if (const auto* a = std::get_if<ir::ConvAttrs>(&node.attrs)) return static_cast<std::uint32_t>(a->group);
  if (const auto* a = std::get_if<ir::ConvTransposeAttrs>(&node.attrs))
    return static_cast<std::uint32_t>(a->group);
  return 1;
}

// Tuiles de 16 canaux de sortie de la variante coopmat.
std::uint32_t matrix_tiles(std::uint32_t cout) {
  return div_up(cout, 16);
}

// Canaux d'entrée par étape : le plus grand de 64, 32, 16 qui divise cinp et tient en mémoire
// partagée. Moins d'étapes, moins de barrières.
std::uint32_t matrix_step_channels(std::uint32_t cinp, std::uint32_t subgroups, std::uint32_t tm,
                                   std::uint32_t np, std::uint32_t max_shared) {
  for (const std::uint32_t kc : {64U, 32U}) {
    if (cinp % kc == 0 && detail::matrix_shared_bytes(subgroups, tm, kc, np) <= max_shared) return kc;
  }
  return 16;
}

// Taille de sous-groupe des matrices coopératives : 32 exigé quand l'appareil le permet (les WMMA
// des RDNA en wave32, comme llama.cpp ; le pilote AMD annonce 64 par défaut), sinon celle du
// pilote. Les noyaux en déduisent leurs sous-groupes par groupe de travail (SGS).
std::uint32_t matrix_subgroup(const Target& target) {
  if (target.subgroup_min != 0 && target.subgroup_min <= 32 && target.subgroup_max >= 32) return 32;
  return target.subgroup_size;
}

// Forme calculée par une Conv avant son DepthToSpace fondu (dims sinon).
ir::Dims result_dims(const ir::Node& node, const ir::Dims& dims) {
  const auto* attrs = std::get_if<ir::ConvAttrs>(&node.attrs);
  return attrs != nullptr ? ir::conv_result_dims(dims, attrs->epilogue) : dims;
}

// Même chose en Desc4 : le rangement (disposition, canaux rangés) reste celui de la sortie.
Desc4 result_desc(const Desc4& y, const ir::Epilogue& e) {
  if (e.space < 2) return y;
  const auto b = static_cast<std::uint32_t>(e.space);
  Desc4 r = y;
  r.c = y.c * b * b;
  r.h = y.h / b;
  r.w = y.w / b;
  return r;
}

// Constantes du DepthToSpace fondu : taille des blocs (0 sans), ordre CRD.
std::array<std::uint32_t, 2> space_constants(const ir::Epilogue& e) {
  return {static_cast<std::uint32_t>(e.space), e.space_mode == ir::BlockMode::Crd ? 1U : 0U};
}

// La variante coopmat convient-elle à ce nœud ? Entrée float16 en Nhwc16, groupe unique.
bool matrix_eligible(const ir::Graph& graph, const ir::Node& node, const Target& target) {
  if (!target.matrix_units || target.subgroup_size == 0 || groups_of(node) != 1) return false;
  const ir::Tensor& x = graph.tensors[node.inputs[ir::kConvX]];
  const ir::Tensor& y = graph.tensors[node.outputs[0]];
  if (!x.known || !y.known || y.dims.size() != 4) return false;
  const ir::Dims result = result_dims(node, y.dims);
  return x.type == ir::DataType::F16 && x.layout == ir::Layout::Nhwc16 &&
         matrix_tiles(static_cast<std::uint32_t>(result[1])) <= 16 &&
         constant_input(graph, node, ir::kConvW) != nullptr;
}

// Winograd F(2×2, 3×3) (net_conv_wino.glsl) : Conv 3 × 3 de pas 1 sans dilatation ni groupes,
// entrée float16 en Nhwc16 (lue deux canaux par uint), poids constants.
bool winograd_eligible(const ir::Graph& graph, const ir::Node& node, const Target& target) {
  if (node.op != ir::OpKind::Conv || !target.f16_storage || groups_of(node) != 1 ||
      constant_input(graph, node, ir::kConvW) == nullptr) {
    return false;
  }
  const ir::Tensor& x = graph.tensors[node.inputs[ir::kConvX]];
  if (!x.known || x.type != ir::DataType::F16 || x.layout != ir::Layout::Nhwc16) return false;
  const auto win = conv_window(graph, node);
  return win && win->kh == 3 && win->kw == 3 && win->sh == 1 && win->sw == 1 && win->dh == 1 && win->dw == 1;
}

}  // namespace

std::expected<std::vector<KernelChoice>, Refusal> candidates(const ir::Graph& graph, std::size_t index,
                                                             const Target& target) {
  const ir::Node& node = graph.nodes[index];
  switch (node.op) {
    case ir::OpKind::Shape:
    case ir::OpKind::Gather:
      return refuse_node(RefusalCode::OperatorUnknown, node,
                         "aucun noyau Vulkan (entiers calculés pendant l'exécution)");
    case ir::OpKind::Conv:
    case ir::OpKind::ConvTranspose:
      break;
    default:
      return std::vector<KernelChoice>{{.variant = "default"}};
  }
  const ir::Tensor& y = graph.tensors[node.outputs[0]];
  if (!y.known || y.dims.size() != 4)
    return refuse_node(RefusalCode::ShapeInvalid, node, "sortie de rang 4 attendue");
  const ir::Dims result = result_dims(node, y.dims);
  const std::uint32_t groups = std::max<std::uint32_t>(1, groups_of(node));
  const std::uint32_t cout_g = static_cast<std::uint32_t>(result[1]) / groups;
  const std::uint32_t quads = default_quads(cout_g);
  std::vector<KernelChoice> out;
  // Les variantes coopmat, en tête pour une famille de la liste d'autorisation, après les autres
  // sinon (Target::matrix_first) : sans autotuner, un appareil hors liste garde le noyau direct.
  std::vector<KernelChoice> matrix;
  const bool conv = node.op == ir::OpKind::Conv;
  if (conv && matrix_eligible(graph, node, target)) {
    const std::uint32_t tiles = matrix_tiles(static_cast<std::uint32_t>(result[1]));
    // 8 sous-groupes (256 invocations en wave32) : les grandes images des RDNA4 et des Ada.
    // Troisième réglage à 1 : B de chaque étape copiée en mémoire partagée une fois pour tout le
    // groupe, plutôt que lue par chaque sous-groupe dans le tampon des poids.
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(result[2]) * static_cast<std::uint64_t>(result[3]);
    // 16 sous-groupes et tm 8 en fin de liste : la mémoire partagée plus grande des RDNA 4 les
    // permet ; l'autotuner les écarte (taille, sortie discordante) quand l'appareil ne les tient pas.
    for (const std::uint32_t subgroups : {4U, 2U, 8U, 16U}) {
      for (const std::uint32_t tm : {2U, 1U, 4U, 8U}) {
        // Les groupes du dispatch (net_lower_conv.cpp, conv coopmat) tiennent dans la limite en x.
        const std::uint64_t block = static_cast<std::uint64_t>(subgroups) * tm * 16;
        if ((pixels + block - 1) / block > target.max_workgroup_count_x) continue;
        if (tm * tiles > 16 || subgroups * matrix_subgroup(target) > target.max_workgroup_invocations ||
            detail::matrix_shared_bytes(subgroups, tm, 16, 0) > target.max_shared_memory) {
          continue;
        }
        matrix.push_back({.variant = std::string(kMatrix), .tuning = {tm, subgroups}});
        if (tm == 2 && detail::matrix_shared_bytes(subgroups, tm, 16, tiles * 16) <= target.max_shared_memory)
          matrix.push_back({.variant = std::string(kMatrix), .tuning = {tm, subgroups, 1}});
        if (target.matrix_f16_acc)
          matrix.push_back({.variant = std::string(kMatrixAcc16), .tuning = {tm, subgroups}});
      }
    }
  }
  if (target.matrix_first) std::swap(out, matrix);
  const bool packed = conv && target.f16_math && target.f16_storage &&
                      graph.tensors[node.inputs[0]].type == ir::DataType::F16 && y.type == ir::DataType::F16;
  for (const std::uint32_t q : {quads, 1U, 2U, 4U}) {
    if (q > quads && q != quads) continue;
    for (const std::uint32_t wg : {64U, 128U}) {
      if (wg > target.max_workgroup_invocations) continue;
      KernelChoice choice{.variant = std::string(kDirect), .tuning = {q, wg}};
      if (std::ranges::find(out, choice) == out.end()) out.push_back(choice);
      if (packed) {
        KernelChoice half{.variant = std::string(kPacked), .tuning = {q, wg}};
        if (std::ranges::find(out, half) == out.end()) out.push_back(half);
      }
    }
  }
  // Conv : plusieurs pixels par invocation (net_conv.glsl, PIX), aux quads par défaut, dans la
  // variante la plus rapide d'ordinaire (fp16 empaqueté si l'appareil l'a).
  const auto width = static_cast<std::uint32_t>(result[3]);
  for (const std::uint32_t pix : {2U, 4U}) {
    if (!conv || width < pix * 2) continue;
    for (const std::uint32_t wg : {64U, 128U}) {
      if (wg > target.max_workgroup_invocations) continue;
      out.push_back({.variant = std::string(packed ? kPacked : kDirect), .tuning = {quads, wg, pix}});
    }
  }
  // Conv : une autre taille de sous-groupe que celle du pilote, quand l'appareil laisse choisir
  // (AMD RDNA : 32 ou 64 ; Intel : 8 à 32).
  for (std::uint32_t sg = target.subgroup_min; conv && sg != 0 && sg <= target.subgroup_max; sg *= 2) {
    if (sg == target.subgroup_size || sg < 8) continue;
    for (const std::uint32_t pix : {1U, 2U}) {
      if (width < pix * 2) continue;
      for (const std::uint32_t wg : {64U, 128U}) {
        if (wg > target.max_workgroup_invocations || wg % sg != 0) continue;
        out.push_back({.variant = std::string(packed ? kPacked : kDirect), .tuning = {quads, wg, pix, sg}});
      }
    }
  }
  // Conv 3 × 3 : Winograd, 2,25 fois moins de produits, pour les GPU sans unités matricielles
  // (Arc intégré, RDNA2, UHD) ; aux tailles de sous-groupe que l'appareil laisse choisir aussi
  // (Intel : SIMD8, SIMD16 ou SIMD32). Là où la variante coopmat convient, elle l'emporte : pas
  // de Winograd à mesurer.
  if (!matrix_eligible(graph, node, target) && winograd_eligible(graph, node, target)) {
    for (const std::uint32_t q : {1U, 2U}) {
      if (q == 2 && cout_g < 8) continue;
      for (const std::uint32_t wg : {64U, 128U}) {
        if (wg > target.max_workgroup_invocations) continue;
        out.push_back({.variant = std::string(kWinograd), .tuning = {q, wg}});
        if (packed) out.push_back({.variant = std::string(kWinogradPacked), .tuning = {q, wg}});
      }
    }
    for (std::uint32_t sg = target.subgroup_min; sg != 0 && sg <= target.subgroup_max; sg *= 2) {
      if (sg == target.subgroup_size || sg < 8) continue;
      for (const std::uint32_t wg : {64U, 128U}) {
        if (wg <= target.max_workgroup_invocations && wg % sg == 0)
          out.push_back(
              {.variant = std::string(packed ? kWinogradPacked : kWinograd), .tuning = {1, wg, 1, sg}});
      }
    }
  }
  if (!target.matrix_first) out.insert(out.end(), matrix.begin(), matrix.end());
  return out;
}

namespace {

// Constantes 0 à 29 communes aux shaders de convolution (net_conv.comp).
struct ConvConstants {
  std::uint32_t tr = 0;
  std::uint32_t lr = 0;
  std::uint32_t crp = 1;
  std::uint32_t act = 0;
  float alpha = 0.0F;
  float lo = 0.0F;
  float hi = 0.0F;
  std::uint32_t residual = 0;  // 1 : ajouté avant l'activation ; 2 : porte (Epilogue::gate)
};

std::vector<std::uint32_t> conv_constants(const Desc4& x, std::uint32_t tx, const Desc4& y, std::uint32_t ty,
                                          const Window& w, std::uint32_t groups, const ConvConstants& e) {
  return {tx,
          ty,
          e.tr,
          x.order,
          y.order,
          e.lr,
          x.cp,
          y.cp,
          e.crp,
          x.n,
          x.c,
          x.h,
          x.w,
          y.c,
          y.h,
          y.w,
          w.kh,
          w.kw,
          w.sh,
          w.sw,
          w.dh,
          w.dw,
          w.pt,
          w.pl,
          groups,
          e.act,
          std::bit_cast<std::uint32_t>(e.alpha),
          std::bit_cast<std::uint32_t>(e.lo),
          std::bit_cast<std::uint32_t>(e.hi),
          e.residual};
}

}  // namespace

std::expected<void, Refusal> Lowerer::conv(std::size_t index, const KernelChoice& choice) {
  const ir::Node& n = graph_.nodes[index];
  const bool winograd = choice.variant == kWinograd || choice.variant == kWinogradPacked;
  const bool packed = choice.variant == kPacked || choice.variant == kWinogradPacked;
  if ((!packed && !winograd && choice.variant != kDirect) || choice.tuning.size() < 2 ||
      choice.tuning.size() > 4) {
    return refuse_node(RefusalCode::InputMismatch, n, std::format("variante {} inconnue", choice.variant));
  }
  const std::uint32_t quads = choice.tuning[0];
  const std::uint32_t wg = choice.tuning[1];
  const std::uint32_t pix = choice.tuning.size() >= 3 ? choice.tuning[2] : 1;  // pixels par invocation
  const std::uint32_t sg = choice.tuning.size() == 4 ? choice.tuning[3] : 0;   // sous-groupe exigé
  if (quads < 1 || quads > 8 || wg < 1 || wg > target_.max_workgroup_invocations || pix < 1 || pix > 4) {
    return refuse_node(RefusalCode::InputMismatch, n, "réglage hors de ses bornes");
  }
  if (sg != 0 && (sg < target_.subgroup_min || sg > target_.subgroup_max || wg % sg != 0)) {
    return refuse_node(RefusalCode::FeatureUnavailable, n, "taille de sous-groupe hors de l'appareil");
  }
  if (packed && !(target_.f16_math && target_.f16_storage)) {
    return refuse_node(RefusalCode::FeatureUnavailable, n, "fp16 de calcul absent");
  }
  if (winograd && (pix != 1 || !winograd_eligible(graph_, n, target_)))
    return refuse_node(RefusalCode::FeatureUnavailable, n,
                       "Winograd : Conv 3 × 3 de pas 1 en float16 seulement");
  const auto* attrs = std::get_if<ir::ConvAttrs>(&n.attrs);
  const ir::Tensor* weights = constant_input(graph_, n, ir::kConvW);
  if (attrs == nullptr || weights == nullptr)
    return refuse_node(RefusalCode::ShapeInvalid, n, "poids non constants");
  auto x = bind(n.inputs[ir::kConvX]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const Desc4 yc = result_desc(y->desc, attrs->epilogue);  // forme calculée, avant le DepthToSpace fondu
  auto win = conv_window(graph_, n);
  if (!win) return std::unexpected(win.error());
  const auto groups = static_cast<std::uint32_t>(attrs->group);
  const std::uint32_t cout_g = yc.c / groups;
  const std::uint32_t blocks = div_up(cout_g, 4 * quads);

  ConvConstants e{.act = static_cast<std::uint32_t>(attrs->epilogue.act),
                  .alpha = attrs->epilogue.alpha,
                  .lo = attrs->epilogue.lo,
                  .hi = attrs->epilogue.hi};
  std::vector<Bound> tensors{*x, *y};
  Binding residual{};
  if (attrs->epilogue.residual || attrs->epilogue.gate) {
    auto r = bind(n.inputs[ir::kConvResidual]);
    if (!r) return std::unexpected(r.error());
    e = {.tr = r->type,
         .lr = r->desc.order,
         .crp = r->desc.cp,
         .act = e.act,
         .alpha = e.alpha,
         .lo = e.lo,
         .hi = e.hi,
         .residual = attrs->epilogue.gate ? 2U : 1U};
    residual = r->binding;
    tensors.push_back(*r);
  }
  Binding slope{};
  if (attrs->epilogue.act == ir::Activation::PRelu) {
    const ir::Tensor* s = constant_input(graph_, n, ir::kConvSlope);
    if (s == nullptr) return refuse_node(RefusalCode::ShapeInvalid, n, "pente PRelu non constante");
    slope = {.kind = BindingKind::Blob, .index = add_blob(channel_floats(s, yc.c))};
  }
  auto name = shader("net_conv", tensors);
  if (!name) return std::unexpected(name.error());
  if (packed) *name = "net_conv_f16p";
  if (winograd) *name = packed ? "net_conv_wino_f16p" : "net_conv_wino_f16";
  const std::uint32_t z = x->desc.n * groups * blocks;
  if (yc.h > kMaxGroups || z > kMaxGroups)
    return refuse_node(RefusalCode::ShapeInvalid, n, "trop de groupes");

  std::vector<std::uint32_t> constants = conv_constants(x->desc, x->type, yc, y->type, *win, groups, e);
  constants.push_back(quads);
  constants.push_back(wg);
  constants.push_back(pix);
  for (const std::uint32_t c : space_constants(attrs->epilogue)) constants.push_back(c);
  const Binding w{
      .kind = BindingKind::Blob,
      .index = add_blob(
          winograd ? winograd_weights(*weights, 4 * quads, packed ? BlobType::F16 : BlobType::F32)
                   : conv_weights(*weights, groups, 4 * quads, packed ? BlobType::F16 : BlobType::F32))};
  const Binding b{.kind = BindingKind::Blob,
                  .index = add_blob(channel_floats(constant_input(graph_, n, ir::kConvB), yc.c))};
  // Winograd : une tuile de 2 × 2 pixels par invocation.
  const std::uint32_t across = winograd ? div_up(yc.w, 2) : div_up(yc.w, pix);
  emit(index, *name, std::move(constants), {x->binding, w, b, slope, residual, y->binding}, {},
       {div_up(across, wg), winograd ? div_up(yc.h, 2) : yc.h, z});
  program_.dispatches.back().subgroup = sg;
  return {};
}

std::expected<void, Refusal> Lowerer::conv_matrix(std::size_t index, const KernelChoice& choice) {
  const ir::Node& n = graph_.nodes[index];
  const bool acc16 = choice.variant == kMatrixAcc16;
  if ((!acc16 && choice.variant != kMatrix) || (choice.tuning.size() != 2 && choice.tuning.size() != 3))
    return refuse_node(RefusalCode::InputMismatch, n, "réglage coopmat incomplet");
  if (acc16 && !target_.matrix_f16_acc) {
    return refuse_node(RefusalCode::FeatureUnavailable, n, "tuiles à sommes float16 absentes du pilote");
  }
  if (!matrix_eligible(graph_, n, target_)) {
    return refuse_node(RefusalCode::FeatureUnavailable, n,
                       "matrices coopératives indisponibles pour ce nœud");
  }
  const std::uint32_t tm = choice.tuning[0];
  const std::uint32_t subgroups = choice.tuning[1];
  const std::uint32_t shared_b = choice.tuning.size() == 3 ? choice.tuning[2] : 0;  // B en mémoire partagée
  const std::uint32_t sgs = matrix_subgroup(target_);
  const std::uint32_t wg = subgroups * sgs;
  const auto* attrs = std::get_if<ir::ConvAttrs>(&n.attrs);
  auto x = bind(n.inputs[ir::kConvX]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  const Desc4 yc = result_desc(y->desc, attrs->epilogue);  // forme calculée, avant le DepthToSpace fondu
  const std::uint32_t tiles = matrix_tiles(yc.c);
  const std::uint32_t np_shared = shared_b == 1 ? tiles * 16 : 0;
  if (tm < 1 || subgroups < 1 || tm * tiles > 16 || wg > target_.max_workgroup_invocations || shared_b > 1 ||
      detail::matrix_shared_bytes(subgroups, tm, 16, np_shared) > target_.max_shared_memory) {
    return refuse_node(RefusalCode::InputMismatch, n, "réglage coopmat hors de ses bornes");
  }
  auto win = conv_window(graph_, n);
  if (!win) return std::unexpected(win.error());

  ConvConstants e{.act = static_cast<std::uint32_t>(attrs->epilogue.act),
                  .alpha = attrs->epilogue.alpha,
                  .lo = attrs->epilogue.lo,
                  .hi = attrs->epilogue.hi};
  std::vector<Bound> tensors{*x, *y};
  Binding residual{};
  if (attrs->epilogue.residual || attrs->epilogue.gate) {
    auto r = bind(n.inputs[ir::kConvResidual]);
    if (!r) return std::unexpected(r.error());
    e = {.tr = r->type,
         .lr = r->desc.order,
         .crp = r->desc.cp,
         .act = e.act,
         .alpha = e.alpha,
         .lo = e.lo,
         .hi = e.hi,
         .residual = attrs->epilogue.gate ? 2U : 1U};
    residual = r->binding;
    tensors.push_back(*r);
  }
  Binding slope{};
  if (attrs->epilogue.act == ir::Activation::PRelu) {
    const ir::Tensor* s = constant_input(graph_, n, ir::kConvSlope);
    if (s == nullptr) return refuse_node(RefusalCode::ShapeInvalid, n, "pente PRelu non constante");
    slope = {.kind = BindingKind::Blob, .index = add_blob(channel_floats(s, yc.c))};
  }
  const std::uint32_t pixels = yc.h * yc.w;
  const std::uint32_t bm = subgroups * tm * 16;
  std::vector<std::uint32_t> constants = conv_constants(x->desc, x->type, yc, y->type, *win, 1, e);
  constants.push_back(tm);
  constants.push_back(wg);
  constants.push_back(sgs);
  constants.push_back(matrix_step_channels(x->desc.cp, subgroups, tm, np_shared, target_.max_shared_memory));
  for (const std::uint32_t c : space_constants(attrs->epilogue)) constants.push_back(c);
  constants.push_back(shared_b);
  const ir::Tensor* weights = constant_input(graph_, n, ir::kConvW);
  const Binding w{.kind = BindingKind::Blob,
                  .index = add_blob(matrix_weights(*weights, x->desc.cp, tiles * 16))};
  const Binding b{.kind = BindingKind::Blob,
                  .index = add_blob(channel_floats(constant_input(graph_, n, ir::kConvB), yc.c))};
  if (x->desc.n > kMaxGroups) return refuse_node(RefusalCode::ShapeInvalid, n, "trop d'images");
  emit(index, acc16 ? "net_conv_coopmat_acc16" : "net_conv_coopmat_f16", std::move(constants),
       {x->binding, w, b, slope, residual, y->binding}, {}, {div_up(pixels, bm), 1, x->desc.n});
  if (sgs != target_.subgroup_size) program_.dispatches.back().subgroup = sgs;
  return {};
}

std::expected<void, Refusal> Lowerer::conv_transpose(std::size_t index, const KernelChoice& choice) {
  const ir::Node& n = graph_.nodes[index];
  if (choice.variant != kDirect || choice.tuning.size() != 2) {
    return refuse_node(RefusalCode::InputMismatch, n, std::format("variante {} inconnue", choice.variant));
  }
  const std::uint32_t quads = choice.tuning[0];
  const std::uint32_t wg = choice.tuning[1];
  if (quads < 1 || quads > 8 || wg < 1 || wg > target_.max_workgroup_invocations) {
    return refuse_node(RefusalCode::InputMismatch, n, "réglage hors de ses bornes");
  }
  const ir::Tensor* weights = constant_input(graph_, n, 1);
  if (weights == nullptr) return refuse_node(RefusalCode::ShapeInvalid, n, "poids non constants");
  auto x = bind(n.inputs[0]);
  if (!x) return std::unexpected(x.error());
  auto y = bind(n.outputs[0]);
  if (!y) return std::unexpected(y.error());
  auto win = conv_transpose_window(graph_, n);
  if (!win) return std::unexpected(win.error());
  const std::uint32_t groups = groups_of(n);
  const std::uint32_t blocks = div_up(y->desc.c / groups, 4 * quads);
  const std::array tensors{*x, *y};
  auto name = shader("net_conv_transpose", tensors);
  if (!name) return std::unexpected(name.error());
  const std::uint32_t z = x->desc.n * groups * blocks;
  if (y->desc.h > kMaxGroups || z > kMaxGroups)
    return refuse_node(RefusalCode::ShapeInvalid, n, "trop de groupes");
  std::vector<std::uint32_t> constants = conv_constants(x->desc, x->type, y->desc, y->type, *win, groups, {});
  constants.push_back(quads);
  constants.push_back(wg);
  const Binding w{.kind = BindingKind::Blob,
                  .index = add_blob(conv_transpose_weights(*weights, groups, 4 * quads))};
  const Binding b{.kind = BindingKind::Blob,
                  .index = add_blob(channel_floats(constant_input(graph_, n, 2), y->desc.c))};
  emit(index, *name, std::move(constants), {x->binding, w, b, Binding{}, Binding{}, y->binding}, {},
       {div_up(y->desc.w, wg), y->desc.h, z});
  return {};
}

}  // namespace ajp::gpu::vk::net
