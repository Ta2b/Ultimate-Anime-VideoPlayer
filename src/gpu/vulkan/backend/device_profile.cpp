#include "gpu/vulkan/backend/device_profile.h"

#include <algorithm>
#include <format>
#include <vector>

namespace ajp::gpu::vk {
namespace {

// Tuiles 16 × 16 × 16 de produits float16 par sous-groupe qu'annonce le pilote
// (VK_KHR_cooperative_matrix), sommes en float (net_conv_coopmat.glsl) ou en float16 (_acc16).
struct MatrixTiles {
  bool f32_sums = false;
  bool f16_sums = false;
};

MatrixTiles matrix_tiles(const VkContext& context) {
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
  const auto tile = [&](VkComponentTypeKHR sums) {
    return std::ranges::any_of(properties, [&](const VkCooperativeMatrixPropertiesKHR& p) {
      return p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
             p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.CType == sums && p.ResultType == sums &&
             p.scope == VK_SCOPE_SUBGROUP_KHR;
    });
  };
  return {.f32_sums = tile(VK_COMPONENT_TYPE_FLOAT32_KHR), .f16_sums = tile(VK_COMPONENT_TYPE_FLOAT16_KHR)};
}

}  // namespace

DeviceProfile profile_device(const VkContext& context) {
  const DeviceCaps& caps = context.caps();
  DeviceProfile profile;
  profile.family = detect_family(caps.vendor_id, caps.device_id, caps.type == VK_PHYSICAL_DEVICE_TYPE_CPU);
  profile.driver = driver_kind(caps.driver_name);
  const Features& f = caps.enabled;
  const bool matrix_features = f.cooperative_matrix && f.vulkan_memory_model && f.storage_buffer_16bit &&
                               f.shader_float16 && caps.subgroup_size > 0;
  const MatrixTiles tiles = matrix_features ? matrix_tiles(context) : MatrixTiles{};
  profile.matrix_hardware = tiles.f32_sums;
  profile.matrix_allowed = matrix_units_allowed(profile.family, profile.driver);
  profile.target = net::Target{.f16_storage = f.storage_buffer_16bit,
                               .f16_math = f.storage_buffer_16bit && f.shader_float16,
                               .matrix_units = profile.matrix_hardware && profile.matrix_allowed,
                               .matrix_f16_acc = profile.matrix_hardware && profile.matrix_allowed && tiles.f16_sums,
                               .subgroup_size = caps.subgroup_size,
                               .subgroup_min = f.subgroup_size_control ? caps.min_subgroup_size : 0,
                               .subgroup_max = f.subgroup_size_control ? caps.max_subgroup_size : 0,
                               .max_push_constants = caps.max_push_constants,
                               .max_shared_memory = caps.max_shared_memory,
                               .max_workgroup_invocations = caps.max_workgroup_invocations,
                               .storage_alignment = static_cast<std::uint32_t>(
                                   std::max<VkDeviceSize>(caps.min_storage_buffer_alignment, 4))};
  // driverVersion se code autrement selon le constructeur : il entre brut, avec driverInfo lisible.
  profile.key = std::format("{:04x}:{:04x}:{}:{}:{:08x}", caps.vendor_id, caps.device_id, caps.driver_name,
                            caps.driver_info, caps.driver_version);
  return profile;
}

}  // namespace ajp::gpu::vk
