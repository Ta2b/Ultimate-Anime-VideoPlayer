// La variante coopmat_m8 (tuiles 8 × 16 × 16 des Intel Xe2, net_conv_m8.h) : tuiles lues dans les propriétés
// du pilote, proposée seulement là où elles sont annoncées et où des sous-groupes de 16 s'exigent, toujours
// derrière le noyau direct ; abaissée avec son noyau, ses constantes, ses poids rangés comme la variante
// coopmat et des sous-groupes de 16 exigés ; mémoire partagée comptée par l'hôte égale à celle que déclare le
// module SPIR-V (évalué avec les constantes du dispatch). Sur un appareil qui annonce ces tuiles, ses sorties
// contre l'interpréteur ; ailleurs (lavapipe), elle n'est pas exécutée.
#include "gpu/vulkan/backend/net_conv_m8.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "ajp_vk_backend_spirv.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/backend/net_lower_detail.h"
#include "net_rig.h"
#include "spirv_shared.h"
#include "test_models.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
using ajp::RefusalCode;
using ir::test::attr_float;
using ir::test::attr_ints;
using ir::test::ModelBuilder;

namespace {

constexpr std::string_view kM8 = "coopmat_m8";

ir::Graph build(const ModelBuilder& model) {
  ir::Graph graph = require_value(ir::load_onnx(model.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  return graph;
}

// Conv 3x3 + Relu → Conv 3x3 + résiduel → Conv 1x1 + Sigmoid → LeakyRelu, en float16 (net_conv_test.cpp) : la
// deuxième et la troisième Conv lisent du Nhwc16.
ModelBuilder chain_model(std::int64_t cin, std::int64_t mid) {
  ModelBuilder m;
  m.input("x", {1, cin, 13, 17}, ir::onnx::kFloat16);
  add_weight(m, "w1", {mid, cin, 3, 3}, true, 21);
  add_weight(m, "b1", {mid}, true, 22);
  add_weight(m, "w2", {mid, mid, 3, 3}, true, 23, 0.1F);
  add_weight(m, "b2", {mid}, true, 24);
  add_weight(m, "w3", {8, mid, 1, 1}, true, 25);
  add_weight(m, "b3", {8}, true, 26);
  const auto same = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"x", "w1", "b1"}, {"c1"}, {same}).node("Relu", {"c1"}, {"r1"});
  m.node("Conv", {"r1", "w2", "b2"}, {"c2"}, {same}).node("Add", {"c2", "r1"}, {"s2"});
  m.node("Conv", {"s2", "w3", "b3"}, {"c3"}).node("Sigmoid", {"c3"}, {"g3"});
  m.node("LeakyRelu", {"g3"}, {"y"}, {attr_float("alpha", 0.2F)}).output("y", ir::onnx::kFloat16);
  return m;
}

// Un Intel Xe2 sous ANV, sans GPU : tuiles 8 × 16 × 16, sous-groupes de 8 à 32 exigeables.
vk::net::Target xe2_target() {
  return {.f16_storage = true,
          .f16_math = true,
          .matrix_m8 = true,
          .subgroup_size = 32,
          .subgroup_min = 8,
          .subgroup_max = 32,
          .max_shared_memory = 65536,
          .max_workgroup_invocations = 1024};
}

// Le nth-ième nœud Conv du graphe, à partir de 1.
std::size_t nth_conv(const ir::Graph& graph, std::size_t nth) {
  for (std::size_t k = 0, seen = 0; k < graph.nodes.size(); ++k) {
    if (graph.nodes[k].op == ir::OpKind::Conv && ++seen == nth) return k;
  }
  FAIL("pas de Conv numéro " << nth);
  return 0;
}

bool is_m8(const vk::net::KernelChoice& c) {
  return c.variant == kM8;
}

std::vector<vk::net::KernelChoice> options_of(const ir::Graph& graph, std::size_t node,
                                              const vk::net::Target& target) {
  return require_value(vk::net::candidates(graph, node, target));
}

const vk::net::Dispatch& dispatch_of(const vk::net::Program& program, std::size_t node) {
  const auto d =
      std::ranges::find(program.dispatches, static_cast<std::uint32_t>(node), &vk::net::Dispatch::node);
  REQUIRE(d != program.dispatches.end());
  return *d;
}

// Le dispatch de la deuxième Conv de chain_model(16, 16) abaissée avec option : noyau, sous-groupes de 16,
// constantes 29 à 39, grille, et mémoire partagée du module évaluée avec ses constantes, égale au compte de
// l'hôte et dans l'appareil.
void check_m8_dispatch(const vk::net::Dispatch& d, const vk::net::KernelChoice& option,
                       const vk::net::Target& target) {
  const std::uint32_t tm = option.tuning[0];
  const std::uint32_t subgroups = option.tuning[1];
  const std::uint32_t shared_b = option.tuning.size() == 3 ? option.tuning[2] : 0;
  INFO("tm " << tm << ", sous-groupes " << subgroups << ", B partagée " << shared_b);
  CHECK(d.shader == "net_conv_coopmat_m8_f16");
  CHECK(d.subgroup == 16U);
  REQUIRE(d.constants.size() == 40U);
  // 29 RESIDUAL (le Add fondu), 30 TM, 31 WG, 32 SGS, 33 KC (16 canaux d'entrée), 34 et 35 DepthToSpace
  // (aucun), 36 SHARED_B, 37 DIRECT_EPI, 38 HALO, 39 NTG.
  const std::vector<std::uint32_t> tail(d.constants.begin() + 29, d.constants.end());
  CHECK(tail == std::vector<std::uint32_t>{1, tm, subgroups * 16, 16, 16, 0, 0, shared_b, 0, 0, 0});
  const std::uint32_t block = subgroups * tm * 8;
  CHECK(d.groups == std::array<std::uint32_t, 3>{(13U * 17U + block - 1) / block, 1, 1});
  const std::span<const std::uint32_t> words = vk::net::spirv::words(vk::net::detail::kMatrixM8Shader);
  REQUIRE_FALSE(words.empty());
  const std::uint64_t declared = spirv_shared_bytes(words, spec_constants(d.constants)).value_or(0);
  CHECK(declared == vk::net::detail::matrix_m8_shared_bytes(subgroups, tm, 16, shared_b == 1 ? 16 : 0));
  CHECK(declared > 0U);
  CHECK(declared <= target.max_shared_memory);
}

// Compte de l'hôte contre module, B partagée ou non : la variante _m8 en sous-groupes de 16, la variante
// coopmat en sous-groupes de 32. SpecId : 13 COUT, 30 TM, 31 WG, 32 SGS, 33 KC, 36 SHARED_B.
void check_matrix_shared(std::uint32_t subgroups, std::uint32_t tm, std::uint32_t kc, std::uint32_t tiles) {
  const std::span<const std::uint32_t> m8 = vk::net::spirv::words(vk::net::detail::kMatrixM8Shader);
  const std::span<const std::uint32_t> wide = vk::net::spirv::words("net_conv_coopmat_f16");
  REQUIRE_FALSE(m8.empty());
  REQUIRE_FALSE(wide.empty());
  for (const std::uint32_t shared_b : {0U, 1U}) {
    INFO(subgroups << " sous-groupes, tm " << tm << ", kc " << kc << ", " << tiles << " tuiles, B "
                   << shared_b);
    const std::uint32_t np = shared_b == 1 ? tiles * 16 : 0;
    std::map<std::uint32_t, std::uint32_t> c{{13, tiles * 16}, {30, tm}, {31, subgroups * 16},
                                             {32, 16},         {33, kc}, {36, shared_b}};
    CHECK(spirv_shared_bytes(m8, c).value_or(0) ==
          vk::net::detail::matrix_m8_shared_bytes(subgroups, tm, kc, np));
    c[31] = subgroups * 32;
    c[32] = 32;
    CHECK(spirv_shared_bytes(wide, c).value_or(0) ==
          vk::net::detail::matrix_shared_bytes(subgroups, tm, kc, np));
  }
}

// Tuiles à halo d'une Conv 3 × 3 de la variante coopmat, B jamais partagée (SpecId 16 et 17 : KH, KW ; 20 et
// 21 : DH, DW ; 38 : HALO).
void check_halo_shared(std::uint32_t subgroups, std::uint32_t tm) {
  const std::span<const std::uint32_t> wide = vk::net::spirv::words("net_conv_coopmat_f16");
  REQUIRE_FALSE(wide.empty());
  for (const std::uint32_t kc : {16U, 32U}) {
    INFO(subgroups << " sous-groupes, tm " << tm << ", kc " << kc);
    const std::uint32_t hpix = vk::net::detail::matrix_halo_pixels(subgroups * tm, 3, 3, 1, 1);
    const std::map<std::uint32_t, std::uint32_t> c{
        {13, 48}, {16, 3},  {17, 3}, {20, 1}, {21, 1}, {30, tm}, {31, subgroups * 32},
        {32, 32}, {33, kc}, {38, 1}};
    CHECK(spirv_shared_bytes(wide, c).value_or(0) ==
          vk::net::detail::matrix_halo_shared_bytes(subgroups, kc, hpix));
  }
}

VkCooperativeMatrixPropertiesKHR tile(std::uint32_t m, std::uint32_t n, std::uint32_t k,
                                      VkComponentTypeKHR in, VkComponentTypeKHR out,
                                      VkScopeKHR scope = VK_SCOPE_SUBGROUP_KHR) {
  return {.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
          .MSize = m,
          .NSize = n,
          .KSize = k,
          .AType = in,
          .BType = in,
          .CType = out,
          .ResultType = out,
          .saturatingAccumulation = VK_FALSE,
          .scope = scope};
}

}  // namespace

// Les listes de Mesa (src/intel/dev/intel_device_info.c) : Xe2 en 8 × 16 × 16, DG2 (Arc A) en 8 × 8 × 16.
TEST_CASE("Profil : tuiles 8 × 16 × 16 lues dans les propriétés du pilote", "[vk_backend][conv][m8]") {
  constexpr auto kF16 = VK_COMPONENT_TYPE_FLOAT16_KHR;
  constexpr auto kF32 = VK_COMPONENT_TYPE_FLOAT32_KHR;
  constexpr auto kS8 = VK_COMPONENT_TYPE_SINT8_KHR;
  constexpr auto kS32 = VK_COMPONENT_TYPE_SINT32_KHR;
  const std::vector xe2{tile(8, 16, 16, kF16, kF16), tile(8, 16, 16, kF16, kF32), tile(8, 16, 32, kS8, kS32)};
  const vk::MatrixTiles x = vk::matrix_tiles(xe2);
  CHECK(x.m8_f32_sums);
  CHECK_FALSE(x.f32_sums);
  CHECK_FALSE(x.f16_sums);
  CHECK_FALSE(x.s8);
  CHECK(x.shapes.find("8×16×16 f16→f32") != std::string::npos);
  const std::vector dg2{tile(8, 8, 16, kF16, kF32), tile(8, 8, 32, kS8, kS32)};
  const vk::MatrixTiles a = vk::matrix_tiles(dg2);
  CHECK_FALSE(a.m8_f32_sums);  // N = 8 : aucun noyau ne la tuile
  CHECK_FALSE(a.f32_sums);
  CHECK(a.shapes == "8×8×16 f16→f32, 8×8×32 s8→s32");
  // Une tuile de groupe de travail, ou à sommes float16 seulement, ne compte pas.
  CHECK_FALSE(vk::matrix_tiles(std::vector{tile(8, 16, 16, kF16, kF32, VK_SCOPE_WORKGROUP_KHR)}).m8_f32_sums);
  CHECK_FALSE(vk::matrix_tiles(std::vector{tile(8, 16, 16, kF16, kF16)}).m8_f32_sums);
  // Un appareil en 16 × 16 × 16 (RDNA 3) : rien de la variante _m8.
  const std::vector rdna3{tile(16, 16, 16, kF16, kF32), tile(16, 16, 16, kF16, kF16),
                          tile(16, 16, 16, kS8, kS32)};
  const vk::MatrixTiles r = vk::matrix_tiles(rdna3);
  CHECK(r.f32_sums);
  CHECK(r.f16_sums);
  CHECK(r.s8);
  CHECK_FALSE(r.m8_f32_sums);
}

TEST_CASE("Réseau : variante coopmat_m8 jamais proposée sans ses tuiles ni des sous-groupes de 16",
          "[vk_backend][conv][m8]") {
  const ir::Graph graph = build(chain_model(16, 16));
  vk::net::Target target = xe2_target();
  const std::size_t conv = nth_conv(graph, 2);
  REQUIRE(std::ranges::any_of(options_of(graph, conv, target), is_m8));
  target.subgroup_min = 16;  // Xe2 : SIMD16 au moins
  CHECK(std::ranges::any_of(options_of(graph, conv, target), is_m8));
  const auto none = [&](const vk::net::Target& t) {
    for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
      INFO("nœud " << k);
      CHECK_FALSE(std::ranges::any_of(options_of(graph, k, t), is_m8));
    }
  };
  vk::net::Target t = xe2_target();
  t.matrix_m8 = false;
  none(t);
  t.matrix_units = true;  // une carte en 16 × 16 × 16 (AMD, NVIDIA)
  t.matrix_first = true;
  none(t);
  t = xe2_target();
  t.subgroup_min = 0;  // pas de taille exigeable
  t.subgroup_max = 0;
  none(t);
  t.subgroup_min = 32;
  t.subgroup_max = 32;
  none(t);
  t.subgroup_min = 8;
  t.subgroup_max = 8;
  none(t);
  // La première Conv lit l'entrée du graphe, hors Nhwc16 : pas de matrices coopératives.
  CHECK_FALSE(std::ranges::any_of(options_of(graph, nth_conv(graph, 1), xe2_target()), is_m8));
}

TEST_CASE("Réseau : variante coopmat_m8 derrière le noyau direct, abaissée en sous-groupes de 16",
          "[vk_backend][conv][m8]") {
  const ir::Graph graph = build(chain_model(16, 16));
  vk::net::Target target = xe2_target();
  const std::size_t conv = nth_conv(graph, 2);
  for (const bool first : {false, true}) {
    target.matrix_first = first;  // sans effet sur _m8 : jamais en tête
    const auto options = options_of(graph, conv, target);
    CHECK(options.front().variant == "direct");
    CHECK(std::ranges::is_partitioned(options, [](const vk::net::KernelChoice& c) { return !is_m8(c); }));
    CHECK(std::ranges::count_if(options, is_m8) >= 9);
  }
  target.matrix_first = false;
  const std::vector<vk::net::KernelChoice> defaults = require_value(vk::net::default_choices(graph, target));
  for (const vk::net::KernelChoice& option : options_of(graph, conv, target)) {
    if (!is_m8(option)) continue;
    std::vector<vk::net::KernelChoice> choices = defaults;
    choices[conv] = option;
    check_m8_dispatch(dispatch_of(require_value(vk::net::lower(graph, choices, target)), conv), option,
                      target);
  }
  // Le sous-groupe s'exige même quand il égale celui du pilote : un Intel choisit sinon sa largeur SIMD.
  target.subgroup_size = 16;
  std::vector<vk::net::KernelChoice> choices = defaults;
  choices[conv] = {.variant = std::string(kM8), .tuning = {2, 4}};
  CHECK(dispatch_of(require_value(vk::net::lower(graph, choices, target)), conv).subgroup == 16U);
}

TEST_CASE("Réseau : coopmat_m8 range ses poids comme coopmat et refuse ce que son noyau n'a pas",
          "[vk_backend][conv][m8]") {
  const ir::Graph graph = build(chain_model(16, 32));
  const vk::net::Target target = xe2_target();
  const std::size_t conv = nth_conv(graph, 2);
  std::vector<vk::net::KernelChoice> choices = require_value(vk::net::default_choices(graph, target));
  choices[conv] = {.variant = std::string(kM8), .tuning = {2, 4, 1}};
  const vk::net::Program m8 = require_value(vk::net::lower(graph, choices, target));
  // La même Conv en tuiles 16 × 16 × 16, sur une carte qui les a : mêmes poids, même biais.
  vk::net::Target wide = target;
  wide.matrix_units = true;
  wide.matrix_m8 = false;
  std::vector<vk::net::KernelChoice> wide_choices = require_value(vk::net::default_choices(graph, wide));
  wide_choices[conv] = {.variant = "coopmat", .tuning = {2, 4, 1}};
  const vk::net::Program full = require_value(vk::net::lower(graph, wide_choices, wide));
  const vk::net::Dispatch& a = dispatch_of(m8, conv);
  const vk::net::Dispatch& b = dispatch_of(full, conv);
  REQUIRE(a.bindings[1].kind == vk::net::BindingKind::Blob);
  REQUIRE(b.bindings[1].kind == vk::net::BindingKind::Blob);
  CHECK(m8.blobs[a.bindings[1].index] == full.blobs[b.bindings[1].index]);
  CHECK(m8.blobs[a.bindings[2].index] == full.blobs[b.bindings[2].index]);
  CHECK(m8.blobs[a.bindings[1].index].size() == std::size_t{9} * 32 * 32 * 2);
  // Les constantes 0 à 29 (formes, épilogue) sont celles de la variante coopmat.
  CHECK(std::equal(a.constants.begin(), a.constants.begin() + 30, b.constants.begin()));

  const auto refused = [&](const vk::net::KernelChoice& choice, const vk::net::Target& t, RefusalCode code) {
    std::vector<vk::net::KernelChoice> c = require_value(vk::net::default_choices(graph, t));
    c[conv] = choice;
    require_refusal(vk::net::lower(graph, c, t), code);
  };
  refused({.variant = std::string(kM8), .tuning = {2, 4, 0, 1}}, target,
          RefusalCode::InputMismatch);  // DIRECT_EPI
  refused({.variant = std::string(kM8), .tuning = {2, 4, 2}}, target, RefusalCode::InputMismatch);
  refused({.variant = std::string(kM8), .tuning = {9, 4}}, target, RefusalCode::InputMismatch);  // 9 × 2 > 16
  refused({.variant = std::string(kM8), .tuning = {0, 4}}, target, RefusalCode::InputMismatch);
  refused({.variant = std::string(kM8), .tuning = {2}}, target, RefusalCode::InputMismatch);
  vk::net::Target small = target;
  small.max_shared_memory = 4096;
  refused({.variant = std::string(kM8), .tuning = {4, 8}}, small, RefusalCode::InputMismatch);
  vk::net::Target without = target;
  without.matrix_m8 = false;
  refused({.variant = std::string(kM8), .tuning = {2, 4}}, without, RefusalCode::FeatureUnavailable);
  vk::net::Target simd8 = target;
  simd8.subgroup_max = 8;
  refused({.variant = std::string(kM8), .tuning = {2, 4}}, simd8, RefusalCode::FeatureUnavailable);
}

// SPAN en petit, en float16 : PRelu (pente par canal) et DepthToSpace fondus passent par l'épilogue du noyau
// _m8.
TEST_CASE("Réseau : coopmat_m8 sur un SPAN en petit, PRelu et DepthToSpace fondus",
          "[vk_backend][conv][m8]") {
  const ir::Graph graph = build(span_like(true));
  const vk::net::Target target = xe2_target();
  std::vector<vk::net::KernelChoice> choices = require_value(vk::net::default_choices(graph, target));
  std::vector<std::size_t> chosen;
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto options = options_of(graph, k, target);
    const auto m8 = std::ranges::find_if(options, is_m8);
    if (m8 == options.end()) continue;
    choices[k] = *m8;
    chosen.push_back(k);
  }
  REQUIRE(chosen.size() >= 2);
  const vk::net::Program program = require_value(vk::net::lower(graph, choices, target));
  bool prelu = false;
  bool space = false;
  for (const std::size_t k : chosen) {
    const vk::net::Dispatch& d = dispatch_of(program, k);
    CHECK(d.shader == "net_conv_coopmat_m8_f16");
    CHECK(d.subgroup == 16U);
    prelu = prelu || d.bindings[3].kind == vk::net::BindingKind::Blob;  // pente PRelu
    space = space || d.constants[34] == 2U;                             // DepthToSpace 2 × 2
  }
  CHECK(prelu);
  CHECK(space);
}

// Le compte de l'hôte contre la déclaration du module, évaluée comme le pilote : coopmat_m8, et aussi la
// variante coopmat en 16 × 16 × 16 (B partagée ou non, tuiles à halo), dont un écart de 32 octets a déjà
// perdu l'appareil.
TEST_CASE("Réseau : mémoire partagée des variantes coopmat égale à celle de leur module SPIR-V",
          "[vk_backend][conv][m8]") {
  for (const std::uint32_t subgroups : {1U, 2U, 4U, 8U}) {
    for (const std::uint32_t tm : {1U, 2U, 4U}) {
      for (const std::uint32_t kc : {16U, 32U, 64U}) {
        for (const std::uint32_t tiles : {1U, 3U, 16U}) check_matrix_shared(subgroups, tm, kc, tiles);
      }
      check_halo_shared(subgroups, tm);
    }
  }
  // En tuiles de 8 pixels, A et l'épilogue prennent deux fois moins de place.
  CHECK(vk::net::detail::matrix_m8_shared_bytes(4, 2, 32, 0) <
        vk::net::detail::matrix_shared_bytes(4, 2, 32, 0));
}

TEST_CASE("Autotuner et artefacts : un appareil à tuiles 8 × 16 × 16 remesure, les autres non",
          "[vk_backend][conv][m8]") {
  vk::net::Target plain;
  plain.matrix_units = true;
  vk::net::Target m8 = plain;
  m8.matrix_units = false;
  m8.matrix_m8 = true;
  CHECK(vk::net::candidate_set(m8) == vk::net::candidate_set(plain) + "m8");
  CHECK(vk::net::candidate_set(plain) == vk::net::candidate_set(vk::net::Target{}));
  vk::DeviceProfile profile;
  profile.key = "8086:e20b:Intel open-source Mesa driver:Mesa 25.2:0";
  const vk::net::BuildRequest request{
      .model_id = "span", .model_sha256 = "00", .input_shape = {1, 3, 64, 64}};
  const std::uint64_t before = vk::net::artifact_key(request, profile);
  profile.target.matrix_m8 = true;
  CHECK(vk::net::artifact_key(request, profile) != before);
}

// Sur un appareil qui annonce les tuiles 8 × 16 × 16 (Intel Xe2) : chaque réglage de la variante contre
// l'interpréteur. Ailleurs, rien ne s'exécute : le noyau n'a jamais tourné sous lavapipe, qui n'a pas de
// matrices coopératives.
TEST_CASE("Réseau : coopmat_m8 contre l'interpréteur sur l'appareil", "[vk_backend][net][conv][m8]") {
  NetRig rig = make_net_rig();
  if (!rig.target.matrix_m8) {
    WARN("pas de tuiles 8 × 16 × 16 sur " << rig.context->caps().device_name
                                          << " : la variante coopmat_m8 n'est pas exécutée ici");
    return;
  }
  for (const auto& [cin, mid] : {std::pair<std::int64_t, std::int64_t>{16, 16}, {12, 48}, {16, 20}}) {
    INFO("cin " << cin << ", mid " << mid);
    const ir::Graph graph = build(chain_model(cin, mid));
    const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F16, {1, cin, 13, 17}, 32)}};
    const std::vector<vk::net::KernelChoice> defaults =
        require_value(vk::net::default_choices(graph, rig.target));
    for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
      for (const vk::net::KernelChoice& option : options_of(graph, k, rig.target)) {
        if (!is_m8(option)) continue;
        std::vector<vk::net::KernelChoice> choices = defaults;
        choices[k] = option;
        INFO("nœud " << k << " : tm " << option.tuning[0] << ", sous-groupes " << option.tuning[1]);
        check_graph(rig, graph, inputs, kFp16, choices);
      }
    }
  }
}
