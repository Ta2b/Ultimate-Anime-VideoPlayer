// Création d'un appareil propre : instance, choix de l'appareil (par UUID ou le premier), file
// compute, et fonctions utiles activées quand l'appareil les a.
#include <algorithm>
#include <cstring>
#include <format>
#include <string_view>
#include <vector>

#include <spdlog/logger.h>

#include "gpu/vulkan/compute/context.h"

namespace ajp::gpu::vk {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr const char* kPortabilityEnumeration = "VK_KHR_portability_enumeration";
// Le socle se contente de Vulkan 1.3 : rien au-delà ne lui sert encore.
constexpr std::uint32_t kMaxApiVersion = VK_API_VERSION_1_3;

std::string version_text(std::uint32_t version) {
  return std::format("{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
}

bool has_layer(const GlobalFunctions& g, const char* name) {
  std::uint32_t count = 0;
  if (g.vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) return false;
  std::vector<VkLayerProperties> layers(count);
  if (g.vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) return false;
  return std::ranges::any_of(layers, [name](const VkLayerProperties& l) {
    return std::strncmp(l.layerName, name, VK_MAX_EXTENSION_NAME_SIZE) == 0;
  });
}

bool has_instance_extension(const GlobalFunctions& g, const char* name) {
  std::uint32_t count = 0;
  if (g.vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) return false;
  std::vector<VkExtensionProperties> extensions(count);
  if (g.vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS)
    return false;
  return has_extension(extensions, name);
}

// L'appareil d'UUID wanted, ou le premier énuméré.
std::expected<VkPhysicalDevice, Refusal> choose_device(const InstanceFunctions& fn, VkInstance instance,
                                                       const std::optional<DeviceUuid>& wanted,
                                                       std::uint32_t api) {
  std::uint32_t count = 0;
  if (auto ok = check(fn.vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
      !ok) {
    return std::unexpected(ok.error());
  }
  std::vector<VkPhysicalDevice> devices(count);
  if (auto ok = check(fn.vkEnumeratePhysicalDevices(instance, &count, devices.data()),
                      "vkEnumeratePhysicalDevices");
      !ok) {
    return std::unexpected(ok.error());
  }
  if (devices.empty()) return refuse(RefusalCode::DeviceUnavailable, "aucun appareil Vulkan");
  if (!wanted) return devices.front();
  std::string seen;
  for (VkPhysicalDevice device : devices) {
    const DeviceCaps caps = probe_device(fn, device, api);
    if (caps.uuid == *wanted) return device;
    seen += std::format(" {} ({})", uuid_text(caps.uuid), caps.device_name);
  }
  return refuse(RefusalCode::DeviceUnavailable,
                std::format("aucun appareil d'UUID {} ; présents :{}", uuid_text(*wanted), seen));
}

// Une famille de calcul : sans graphisme de préférence (file asynchrone), sinon la première.
std::expected<std::uint32_t, Refusal> compute_family(const InstanceFunctions& fn, VkPhysicalDevice device) {
  std::uint32_t count = 0;
  fn.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  fn.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
  std::optional<std::uint32_t> any;
  for (std::uint32_t i = 0; i < count; ++i) {
    const VkQueueFlags flags = families[i].queueFlags;
    if ((flags & VK_QUEUE_COMPUTE_BIT) == 0 || families[i].queueCount == 0) continue;
    if ((flags & VK_QUEUE_GRAPHICS_BIT) == 0) return i;
    if (!any) any = i;
  }
  if (any) return *any;
  return refuse(RefusalCode::DeviceUnavailable, "aucune file compute sur l'appareil");
}

VkBool32 flag(bool on) {
  return on ? VK_TRUE : VK_FALSE;
}

}  // namespace

std::expected<std::unique_ptr<VkContext>, Refusal> VkContext::create(const CreateOptions& options) {
  std::unique_ptr<VkContext> ctx(new VkContext());
  ctx->get_instance_proc_addr_ = options.get_instance_proc_addr;
  if (ctx->get_instance_proc_addr_ == nullptr) {
    auto library = options.library ? VulkanLibrary::open(*options.library) : VulkanLibrary::open_system();
    if (!library) return std::unexpected(library.error());
    ctx->library_ = std::move(*library);
    ctx->get_instance_proc_addr_ = ctx->library_->get_instance_proc_addr();
  }
  if (auto ok = ctx->create_instance(options); !ok) return std::unexpected(ok.error());
  if (auto ok = ctx->create_device(options); !ok) return std::unexpected(ok.error());
  if (auto ok = ctx->finish_setup(); !ok) return std::unexpected(ok.error());
  ctx->log().info("appareil créé : {}", describe(ctx->caps_));
  return ctx;
}

Done VkContext::create_instance(const CreateOptions& options) {
  auto global = load_global(get_instance_proc_addr_);
  if (!global) return std::unexpected(global.error());
  std::uint32_t loader = 0;
  if (auto ok = check(global->vkEnumerateInstanceVersion(&loader), "vkEnumerateInstanceVersion"); !ok)
    return ok;
  if (loader < kMinApiVersion) {
    return refuse(
        RefusalCode::FeatureUnavailable,
        std::format("chargeur Vulkan {}, {} au moins", version_text(loader), version_text(kMinApiVersion)));
  }
  std::vector<const char*> layers;
  if (options.validation) {
    if (!has_layer(*global, kValidationLayer)) {
      return refuse(RefusalCode::FeatureUnavailable, std::format("couche {} absente", kValidationLayer));
    }
    layers.push_back(kValidationLayer);
  }
  // MoltenVK n'est énuméré qu'avec l'extension de portabilité ; ailleurs elle est sans effet.
  std::vector<const char*> extensions;
  VkInstanceCreateFlags flags = 0;
  if (has_instance_extension(*global, kPortabilityEnumeration)) {
    extensions.push_back(kPortabilityEnumeration);
    flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  const std::uint32_t api = std::min(loader, kMaxApiVersion);
  const VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                              .pApplicationName = "ajp",
                              .pEngineName = "ajp vk_compute",
                              .apiVersion = api};
  const VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                  .flags = flags,
                                  .pApplicationInfo = &app,
                                  .enabledLayerCount = static_cast<std::uint32_t>(layers.size()),
                                  .ppEnabledLayerNames = layers.data(),
                                  .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
                                  .ppEnabledExtensionNames = extensions.data()};
  if (auto ok = check(global->vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance"); !ok)
    return ok;
  owns_instance_ = true;
  // Détruite sans attendre les autres fonctions : une table incomplète ne la fait pas fuir.
  instance_fn_.vkDestroyInstance =
      reinterpret_cast<PFN_vkDestroyInstance>(get_instance_proc_addr_(instance_, "vkDestroyInstance"));
  auto fns = load_instance(get_instance_proc_addr_, instance_);
  if (!fns) return std::unexpected(fns.error());
  instance_fn_ = *fns;
  auto device = choose_device(instance_fn_, instance_, options.device, api);
  if (!device) return std::unexpected(device.error());
  physical_device_ = *device;
  caps_ = probe_device(instance_fn_, physical_device_, api);
  if (caps_.api_version < kMinApiVersion) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} : Vulkan {}, {} au moins", caps_.device_name,
                              version_text(caps_.api_version), version_text(kMinApiVersion)));
  }
  return {};
}

Done VkContext::create_device(const CreateOptions& options) {
  auto family = compute_family(instance_fn_, physical_device_);
  if (!family) return std::unexpected(family.error());
  queue_family_ = *family;
  queue_index_ = 0;

  const Features available = available_features(instance_fn_, physical_device_, caps_.api_version);
  if (!available.timeline_semaphore) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} : pas de timeline semaphores", caps_.device_name));
  }
  const std::vector<VkExtensionProperties> present = device_extensions(instance_fn_, physical_device_);
  std::vector<const char*> extensions;
  for (const std::string& name : options.extensions) {
    if (!has_extension(present, name.c_str())) {
      return refuse(RefusalCode::FeatureUnavailable,
                    std::format("{} : extension {} absente", caps_.device_name, name));
    }
    extensions.push_back(name.c_str());
  }

  const bool matrix = options.cooperative_matrix && available.cooperative_matrix &&
                      available.vulkan_memory_model &&
                      has_extension(present, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  std::vector<std::string> enabled_extensions = options.extensions;
  if (matrix && std::ranges::find(options.extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ==
                    options.extensions.end()) {
    extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    enabled_extensions.emplace_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  }
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      .cooperativeMatrix = VK_TRUE};
  VkPhysicalDeviceVulkan13Features v13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                       .pNext = matrix ? &matrix_features : nullptr,
                                       .subgroupSizeControl = flag(available.subgroup_size_control),
                                       .computeFullSubgroups = flag(available.compute_full_subgroups)};
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       .pNext = matrix ? &matrix_features : nullptr,
                                       .shaderFloat16 = flag(available.shader_float16),
                                       .timelineSemaphore = VK_TRUE,
                                       .vulkanMemoryModel = flag(matrix)};
  if (caps_.api_version >= VK_API_VERSION_1_3) v12.pNext = &v13;
  VkPhysicalDeviceVulkan11Features v11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                       .pNext = &v12,
                                       .storageBuffer16BitAccess = flag(available.storage_buffer_16bit)};
  VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v11};
  features.features.shaderInt16 = flag(available.shader_int16);

  const float priority = 1.0F;
  const VkDeviceQueueCreateInfo queue{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                      .queueFamilyIndex = queue_family_,
                                      .queueCount = 1,
                                      .pQueuePriorities = &priority};
  const VkDeviceCreateInfo info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                .pNext = &features,
                                .queueCreateInfoCount = 1,
                                .pQueueCreateInfos = &queue,
                                .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
                                .ppEnabledExtensionNames = extensions.data()};
  if (auto ok =
          check(instance_fn_.vkCreateDevice(physical_device_, &info, nullptr, &device_), "vkCreateDevice");
      !ok) {
    return ok;
  }
  owns_device_ = true;
  device_fn_.vkDestroyDevice =
      reinterpret_cast<PFN_vkDestroyDevice>(instance_fn_.vkGetDeviceProcAddr(device_, "vkDestroyDevice"));
  auto fns = load_device(instance_fn_, device_);
  if (!fns) return std::unexpected(fns.error());
  device_fn_ = *fns;
  caps_.enabled = features_from_chain(&features);
  caps_.extensions = std::move(enabled_extensions);
  return {};
}

}  // namespace ajp::gpu::vk
