#include "gpu/vulkan/compute/caps.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <iterator>
#include <string_view>

namespace ajp::gpu::vk {
namespace {

bool yes(VkBool32 value) {
  return value == VK_TRUE;
}

// Texte d'un tableau de taille fixe, jusqu'au premier zéro.
std::string text(const char* chars, std::size_t capacity) {
  const std::string_view all(chars, capacity);
  return std::string(all.substr(0, all.find('\0')));
}

std::string version_text(std::uint32_t version) {
  return std::format("{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
}

// Fonctions lues dans une structure de la chaîne ; les autres structures sont ignorées.
void read_link(const VkBaseInStructure& link, Features& f) {
  switch (link.sType) {
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
      const auto& v11 = reinterpret_cast<const VkPhysicalDeviceVulkan11Features&>(link);
      f.storage_buffer_16bit = f.storage_buffer_16bit || yes(v11.storageBuffer16BitAccess);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: {
      const auto& v12 = reinterpret_cast<const VkPhysicalDeviceVulkan12Features&>(link);
      f.timeline_semaphore = f.timeline_semaphore || yes(v12.timelineSemaphore);
      f.shader_float16 = f.shader_float16 || yes(v12.shaderFloat16);
      f.vulkan_memory_model = f.vulkan_memory_model || yes(v12.vulkanMemoryModel);
      f.vulkan_memory_model_device_scope =
          f.vulkan_memory_model_device_scope || yes(v12.vulkanMemoryModelDeviceScope);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES: {
      const auto& model = reinterpret_cast<const VkPhysicalDeviceVulkanMemoryModelFeatures&>(link);
      f.vulkan_memory_model = f.vulkan_memory_model || yes(model.vulkanMemoryModel);
      f.vulkan_memory_model_device_scope =
          f.vulkan_memory_model_device_scope || yes(model.vulkanMemoryModelDeviceScope);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: {
      const auto& matrix = reinterpret_cast<const VkPhysicalDeviceCooperativeMatrixFeaturesKHR&>(link);
      f.cooperative_matrix = f.cooperative_matrix || yes(matrix.cooperativeMatrix);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT: {
      const auto& atomic = reinterpret_cast<const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT&>(link);
      f.float_atomic_add = f.float_atomic_add || yes(atomic.shaderBufferFloat32AtomicAdd);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
      const auto& v13 = reinterpret_cast<const VkPhysicalDeviceVulkan13Features&>(link);
      f.subgroup_size_control = f.subgroup_size_control || yes(v13.subgroupSizeControl);
      f.compute_full_subgroups = f.compute_full_subgroups || yes(v13.computeFullSubgroups);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: {
      const auto& timeline = reinterpret_cast<const VkPhysicalDeviceTimelineSemaphoreFeatures&>(link);
      f.timeline_semaphore = f.timeline_semaphore || yes(timeline.timelineSemaphore);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: {
      const auto& storage = reinterpret_cast<const VkPhysicalDevice16BitStorageFeatures&>(link);
      f.storage_buffer_16bit = f.storage_buffer_16bit || yes(storage.storageBuffer16BitAccess);
      break;
    }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: {
      const auto& float16 = reinterpret_cast<const VkPhysicalDeviceShaderFloat16Int8Features&>(link);
      f.shader_float16 = f.shader_float16 || yes(float16.shaderFloat16);
      break;
    }
    default:
      break;
  }
}

}  // namespace

std::vector<VkExtensionProperties> device_extensions(const InstanceFunctions& fn, VkPhysicalDevice device) {
  std::uint32_t count = 0;
  if (fn.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) return {};
  std::vector<VkExtensionProperties> extensions(count);
  if (fn.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()) != VK_SUCCESS) {
    return {};
  }
  extensions.resize(count);
  return extensions;
}

bool has_extension(std::span<const VkExtensionProperties> extensions, const char* name) {
  return std::ranges::any_of(extensions, [name](const VkExtensionProperties& e) {
    return std::strncmp(e.extensionName, name, VK_MAX_EXTENSION_NAME_SIZE) == 0;
  });
}

DeviceCaps probe_device(const InstanceFunctions& fn, VkPhysicalDevice device, std::uint32_t instance_api) {
  VkPhysicalDeviceProperties base{};
  fn.vkGetPhysicalDeviceProperties(device, &base);
  const std::uint32_t api = std::min(instance_api, base.apiVersion);

  VkPhysicalDeviceSubgroupSizeControlProperties sizes{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
                                          .pNext = api >= VK_API_VERSION_1_3 ? &sizes : nullptr};
  VkPhysicalDeviceSubgroupProperties subgroup{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                                              .pNext = &driver};
  VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
                                   .pNext = &subgroup};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &ids};
  fn.vkGetPhysicalDeviceProperties2(device, &properties);

  const VkPhysicalDeviceLimits& limits = base.limits;
  DeviceCaps caps{
      .device_name = text(base.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
      .driver_name = text(driver.driverName, VK_MAX_DRIVER_NAME_SIZE),
      .driver_info = text(driver.driverInfo, VK_MAX_DRIVER_INFO_SIZE),
      .api_version = api,
      .driver_version = base.driverVersion,
      .vendor_id = base.vendorID,
      .device_id = base.deviceID,
      .type = base.deviceType,
      .max_push_constants = limits.maxPushConstantsSize,
      .max_workgroup_invocations = limits.maxComputeWorkGroupInvocations,
      .max_workgroup_count = {limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1],
                              limits.maxComputeWorkGroupCount[2]},
      .max_shared_memory = limits.maxComputeSharedMemorySize,
      .max_storage_buffer_range = limits.maxStorageBufferRange,
      .min_storage_buffer_alignment = limits.minStorageBufferOffsetAlignment,
      .max_image_dimension_2d = limits.maxImageDimension2D,
  };
  std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), caps.uuid.begin());
  if ((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0) {
    caps.subgroup_size = subgroup.subgroupSize;
    caps.subgroup_operations = subgroup.supportedOperations;
  }
  if ((sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0) {
    caps.min_subgroup_size = sizes.minSubgroupSize;
    caps.max_subgroup_size = sizes.maxSubgroupSize;
  }
  caps.cooperative_matrix = has_extension(device_extensions(fn, device), "VK_KHR_cooperative_matrix");
  return caps;
}

Features available_features(const InstanceFunctions& fn, VkPhysicalDevice device, std::uint32_t api_version) {
  VkPhysicalDeviceVulkan13Features v13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan11Features v11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                       .pNext = &v12};
  if (api_version >= VK_API_VERSION_1_3) v12.pNext = &v13;
  // La structure d'une extension n'entre dans la chaîne que si l'appareil a l'extension.
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, .pNext = v12.pNext};
  const std::vector<VkExtensionProperties> extensions = device_extensions(fn, device);
  if (has_extension(extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) v12.pNext = &matrix;
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT, .pNext = v12.pNext};
  if (has_extension(extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) v12.pNext = &atomic;
  VkPhysicalDeviceFeatures2 all{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v11};
  fn.vkGetPhysicalDeviceFeatures2(device, &all);
  return features_from_chain(&all);
}

Features features_from_chain(const VkPhysicalDeviceFeatures2* chain) {
  Features f;
  if (chain == nullptr) return f;
  f.shader_int16 = yes(chain->features.shaderInt16);
  for (const auto* link = static_cast<const VkBaseInStructure*>(chain->pNext); link != nullptr;
       link = link->pNext) {
    read_link(*link, f);
  }
  return f;
}

std::string uuid_text(const DeviceUuid& uuid) {
  std::string out;
  for (const std::uint8_t byte : uuid) out += std::format("{:02x}", byte);
  return out;
}

std::string describe(const DeviceCaps& caps) {
  const Features& f = caps.enabled;
  const auto flag = [](bool on, const char* name) { return on ? std::format(", {}", name) : std::string{}; };
  return std::format(
      "{} ({} {}), Vulkan {}, UUID {}{}{}{}{}{}{}{}, sous-groupes de {}, push {} o, groupe {} invocations, "
      "partagée {} o, tampon {} o{}",
      caps.device_name, caps.driver_name, caps.driver_info, version_text(caps.api_version),
      uuid_text(caps.uuid), flag(f.timeline_semaphore, "timeline"),
      flag(f.storage_buffer_16bit, "stockage 16 bits"), flag(f.shader_float16, "fp16"),
      flag(f.shader_int16, "int16"), flag(f.subgroup_size_control, "contrôle des sous-groupes"),
      flag(f.compute_full_subgroups, "sous-groupes pleins"), flag(f.float_atomic_add, "atomiques float"),
      caps.subgroup_size, caps.max_push_constants, caps.max_workgroup_invocations, caps.max_shared_memory,
      caps.max_storage_buffer_range, flag(caps.cooperative_matrix, "VK_KHR_cooperative_matrix présent"));
}

}  // namespace ajp::gpu::vk
