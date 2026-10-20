#include "gpu/vulkan/backend/device_profile.h"

#include <algorithm>
#include <format>
#include <vector>

namespace ajp::gpu::vk {
namespace {

// VK_KHR_cooperative_matrix : le format 16 × 16 × 16 des noyaux (net_conv_coopmat.glsl).
bool has_f16_tile(const VkContext& context) {
  const auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      context.get_instance_proc_addr()(context.instance(),
                                       "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (query == nullptr) return false;
  std::uint32_t count = 0;
  if (query(context.physical_device(), &count, nullptr) != VK_SUCCESS || count == 0) return false;
  std::vector<VkCooperativeMatrixPropertiesKHR> properties(
      count, VkCooperativeMatrixPropertiesKHR{.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  if (query(context.physical_device(), &count, properties.data()) != VK_SUCCESS) return false;
  properties.resize(count);
  return std::ranges::any_of(properties, [](const VkCooperativeMatrixPropertiesKHR& p) {
    return p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
           p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.scope == VK_SCOPE_SUBGROUP_KHR;
  });
}

}  // namespace

DeviceProfile profile_device(const VkContext& context) {
  const DeviceCaps& caps = context.caps();
  DeviceProfile profile;
  profile.family = detect_family(caps.vendor_id, caps.device_id, caps.type == VK_PHYSICAL_DEVICE_TYPE_CPU);
  profile.driver = driver_kind(caps.driver_name);
  const Features& f = caps.enabled;
  profile.matrix_hardware = f.cooperative_matrix && f.vulkan_memory_model && f.storage_buffer_16bit &&
                            f.shader_float16 && caps.subgroup_size > 0 && has_f16_tile(context);
  profile.matrix_allowed = matrix_units_allowed(profile.family, profile.driver);
  profile.target = net::Target{.f16_storage = f.storage_buffer_16bit,
                               .f16_math = f.storage_buffer_16bit && f.shader_float16,
                               .matrix_units = profile.matrix_hardware && profile.matrix_allowed,
                               .subgroup_size = caps.subgroup_size,
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
