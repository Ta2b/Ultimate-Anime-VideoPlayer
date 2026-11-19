#include "gpu/vulkan/backend/device_profile.h"

#include <algorithm>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kirari::gpu::vk {
namespace {

// VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT / _E5M2_EXT (VK_EXT_shader_float8), absents des en-têtes
// plus anciens : valeurs de l'extension.
constexpr auto kComponentFloat8E4M3 = static_cast<VkComponentTypeKHR>(1000491002);
constexpr auto kComponentFloat8E5M2 = static_cast<VkComponentTypeKHR>(1000491003);

std::string_view component(VkComponentTypeKHR type) {
  if (type == kComponentFloat8E4M3) return "e4m3";
  if (type == kComponentFloat8E5M2) return "e5m2";
  switch (type) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
      return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
      return "f32";
    case VK_COMPONENT_TYPE_SINT8_KHR:
      return "s8";
    case VK_COMPONENT_TYPE_UINT8_KHR:
      return "u8";
    case VK_COMPONENT_TYPE_SINT32_KHR:
      return "s32";
    case VK_COMPONENT_TYPE_UINT32_KHR:
      return "u32";
    default:
      return "?";
  }
}

// Une tuile m × 16 × 16 par sous-groupe, A et B du type inputs, sommes et résultat du type sums.
bool offers_tile(std::span<const VkCooperativeMatrixPropertiesKHR> properties, VkComponentTypeKHR inputs,
                 VkComponentTypeKHR sums, std::uint32_t m = 16) {
  return std::ranges::any_of(properties, [&](const VkCooperativeMatrixPropertiesKHR& p) {
    return p.MSize == m && p.NSize == 16 && p.KSize == 16 && p.AType == inputs && p.BType == inputs &&
           p.CType == sums && p.ResultType == sums && p.scope == VK_SCOPE_SUBGROUP_KHR;
  });
}

MatrixTiles query_matrix_tiles(const VkContext& context) {
  const auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      context.get_instance_proc_addr()(context.instance(),
                                       "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (query == nullptr) return {};
  std::uint32_t count = 0;
  if (query(context.physical_device(), &count, nullptr) != VK_SUCCESS || count == 0) return {};
  std::vector<VkCooperativeMatrixPropertiesKHR> properties(
      count, VkCooperativeMatrixPropertiesKHR{.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  if (query(context.physical_device(), &count, properties.data()) != VK_SUCCESS) return {};
  properties.resize(count);
  return matrix_tiles(properties);
}

}  // namespace

MatrixTiles matrix_tiles(std::span<const VkCooperativeMatrixPropertiesKHR> properties) {
  const auto tile = [&](VkComponentTypeKHR sums, std::uint32_t m) {
    return offers_tile(properties, VK_COMPONENT_TYPE_FLOAT16_KHR, sums, m);
  };
  // Toutes les tuiles du pilote au journal (« 8×8×16 f16→f32 » des Arc A, qu'aucun noyau ne tuile), plutôt
  // que de supposer 16 × 16 × 16.
  std::string shapes;
  for (const VkCooperativeMatrixPropertiesKHR& p : properties) {
    if (p.scope != VK_SCOPE_SUBGROUP_KHR) continue;
    const std::string shape =
        std::format("{}×{}×{} {}→{}", p.MSize, p.NSize, p.KSize, component(p.AType), component(p.ResultType));
    if (shapes.find(shape) == std::string::npos) shapes += (shapes.empty() ? "" : ", ") + shape;
  }
  return {.f32_sums = tile(VK_COMPONENT_TYPE_FLOAT32_KHR, 16),
          .f16_sums = tile(VK_COMPONENT_TYPE_FLOAT16_KHR, 16),
          .f8_e4m3 = offers_tile(properties, kComponentFloat8E4M3, VK_COMPONENT_TYPE_FLOAT32_KHR),
          .s8 = offers_tile(properties, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR),
          .m8_f32_sums = tile(VK_COMPONENT_TYPE_FLOAT32_KHR, 8),
          .shapes = std::move(shapes)};
}

DeviceProfile profile_device(const VkContext& context) {
  const DeviceCaps& caps = context.caps();
  DeviceProfile profile;
  profile.family = detect_family(caps.vendor_id, caps.device_id, caps.type == VK_PHYSICAL_DEVICE_TYPE_CPU,
                                 caps.device_name);
  profile.driver = driver_kind(caps.driver_name);
  const Features& f = caps.enabled;
  const bool matrix_features = f.cooperative_matrix && f.vulkan_memory_model && f.storage_buffer_16bit &&
                               f.shader_float16 && caps.subgroup_size > 0;
  const MatrixTiles tiles = matrix_features ? query_matrix_tiles(context) : MatrixTiles{};
  profile.family = family_from_tiles(profile.family, caps.device_name, tiles.f32_sums, tiles.f8_e4m3);
  profile.matrix_hardware = tiles.f32_sums;
  profile.matrix_shapes = tiles.shapes;
  profile.matrix_allowed = matrix_units_allowed(profile.family, profile.driver);
  profile.matrix_unlisted = matrix_units_unlisted(profile.family, profile.driver);
  const bool offered = profile.matrix_hardware && profile.matrix_allowed;
  const bool sized = f.subgroup_size_control && f.compute_full_subgroups;
  // Tuiles 8 × 16 × 16 (coopmat_m8) : là seulement où la tuile 16 × 16 × 16 manque (rien ne change ailleurs),
  // et où des sous-groupes de 16 peuvent s'exiger (net_conv_m8.h).
  const bool m8 = tiles.m8_f32_sums && !tiles.f32_sums && profile.matrix_allowed && sized &&
                  caps.min_subgroup_size <= 16 && caps.max_subgroup_size >= 16;
  profile.target = net::Target{
      .f16_storage = f.storage_buffer_16bit,
      .f16_math = f.storage_buffer_16bit && f.shader_float16,
      .matrix_units = offered,
      .matrix_first = offered && !profile.matrix_unlisted,
      .matrix_f16_acc = offered && tiles.f16_sums,
      .matrix_f8 = offered && tiles.f8_e4m3 && f.float8_matrix && matrix_f8_allowed(profile.family),
      .matrix_i8 = offered && tiles.s8 && f.shader_int8,
      .matrix_m8 = m8,
      .subgroup_size = caps.subgroup_size,
      // Taille exigée seulement avec des sous-groupes pleins garantis.
      .subgroup_min = sized ? caps.min_subgroup_size : 0,
      .subgroup_max = sized ? caps.max_subgroup_size : 0,
      .max_push_constants = caps.max_push_constants,
      .max_shared_memory = caps.max_shared_memory,
      .max_workgroup_invocations = caps.max_workgroup_invocations,
      .max_workgroup_count_x = caps.max_workgroup_count[0],
      .storage_alignment =
          static_cast<std::uint32_t>(std::max<VkDeviceSize>(caps.min_storage_buffer_alignment, 4))};
  // driverVersion se code autrement selon le constructeur : il entre brut, avec driverInfo lisible.
  profile.stable_key = std::format("{:04x}:{:04x}:{}", caps.vendor_id, caps.device_id, caps.driver_name);
  profile.driver_build = std::format("{}:{:08x}", caps.driver_info, caps.driver_version);
  profile.key = std::format("{}:{}", profile.stable_key, profile.driver_build);
  return profile;
}

}  // namespace kirari::gpu::vk
