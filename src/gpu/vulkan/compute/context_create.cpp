// Création d'un appareil propre : instance, choix de l'appareil (par UUID ou le premier), file
// compute, et fonctions utiles activées quand l'appareil les a.
#include <algorithm>
#include <cstring>
#include <format>
#include <string_view>
#include <vector>

#include <spdlog/logger.h>

#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/log.h"

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

// Rappel de la couche de validation (VK_EXT_debug_utils) : chaque message va au journal et au
// relais ; la couche appelle depuis le thread de l'appel Vulkan, quel qu'il soit.
VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
  auto* relay = static_cast<ValidationRelay*>(user);
  if (relay == nullptr || data == nullptr || data->pMessage == nullptr) return VK_FALSE;
  const bool error = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
  const std::string message = std::format(
      "validation {} : {}", data->pMessageIdName != nullptr ? data->pMessageIdName : "", data->pMessage);
  if (relay->log) {
    if (error) {
      relay->log->error("{}", message);
    } else {
      relay->log->warn("{}", message);
    }
  }
  if (relay->sink) relay->sink(error, message);
  return VK_FALSE;
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
  std::vector<const char*> extensions;
  bool debug_utils = false;
  if (options.validation) {
    if (!has_layer(*global, kValidationLayer)) {
      return refuse(RefusalCode::FeatureUnavailable, std::format("couche {} absente", kValidationLayer));
    }
    layers.push_back(kValidationLayer);
    validation_ = std::make_unique<ValidationRelay>(
        ValidationRelay{.log = make_logger(), .sink = options.validation_sink});
    // Le chargeur porte VK_EXT_debug_utils depuis Vulkan 1.1 ; sans lui, les messages de la
    // couche vont sur sa sortie standard, hors journal et hors relais.
    debug_utils = has_instance_extension(*global, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debug_utils) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else {
      validation_->log->warn("VK_EXT_debug_utils absent : la couche de validation écrit sur sa sortie");
    }
  }
  // Pour l'instance, puis pour tout ce qui suit : les messages de vkCreateInstance passent par le
  // pNext, ceux des autres appels par le messager créé après.
  const VkDebugUtilsMessengerCreateInfoEXT messenger{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity =
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = on_validation,
      .pUserData = validation_.get()};
  // MoltenVK n'est énuméré qu'avec l'extension de portabilité ; ailleurs elle est sans effet.
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
                                  .pNext = debug_utils ? &messenger : nullptr,
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
  if (debug_utils) {
    auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        get_instance_proc_addr_(instance_, "vkCreateDebugUtilsMessengerEXT"));
    destroy_messenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        get_instance_proc_addr_(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (create_messenger == nullptr || destroy_messenger_ == nullptr) {
      return refuse(RefusalCode::LibraryUnavailable, "vkCreateDebugUtilsMessengerEXT absent du chargeur");
    }
    if (auto ok = check(create_messenger(instance_, &messenger, nullptr, &messenger_),
                        "vkCreateDebugUtilsMessengerEXT");
        !ok) {
      return ok;
    }
  }
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
  // Les additions atomiques de float, quand l'appareil les a : softmax splatting de GMFSS.
  const bool atomic =
      available.float_atomic_add && has_extension(present, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
  if (atomic && std::ranges::find(options.extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) ==
                    options.extensions.end()) {
    extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    enabled_extensions.emplace_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
  }
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
      .shaderBufferFloat32AtomicAdd = VK_TRUE};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      .pNext = atomic ? &atomic_features : nullptr,
      .cooperativeMatrix = VK_TRUE};
  void* const extension_chain = matrix   ? static_cast<void*>(&matrix_features)
                                : atomic ? static_cast<void*>(&atomic_features)
                                         : nullptr;
  VkPhysicalDeviceVulkan13Features v13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                       .pNext = extension_chain,
                                       .subgroupSizeControl = flag(available.subgroup_size_control),
                                       .computeFullSubgroups = flag(available.compute_full_subgroups)};
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       .pNext = extension_chain,
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
