// Messager VK_EXT_debug_utils du contexte créé : chaque message d'erreur ou d'avertissement de
// la couche de validation, du chargeur ou du pilote est journalisé et compté. Les tests vérifient
// qu'aucune erreur n'a été reçue ; le lecteur l'active avec --vulkan-validation.
#include <spdlog/logger.h>

#include "gpu/vulkan/compute/context.h"

namespace ajp::gpu::vk {
namespace {

constexpr VkDebugUtilsMessageSeverityFlagsEXT kSeverities =
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
constexpr VkDebugUtilsMessageTypeFlagsEXT kTypes = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

}  // namespace

// Appelé par le chargeur, depuis n'importe quel thread qui fait un appel Vulkan. Ne lève rien :
// spdlog n'est appelé que pour journaliser, ses erreurs sont absorbées par son gestionnaire.
VKAPI_ATTR VkBool32 VKAPI_CALL VkContext::on_debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                           VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                                           const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                           void* user) {
  auto* sink = static_cast<DebugSink*>(user);
  if (sink == nullptr) return VK_FALSE;
  const char* id = data != nullptr && data->pMessageIdName != nullptr ? data->pMessageIdName : "?";
  const char* text = data != nullptr && data->pMessage != nullptr ? data->pMessage : "";
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    sink->errors.fetch_add(1);
    if (sink->log) sink->log->error("Vulkan [{}] {}", id, text);
  } else {
    sink->warnings.fetch_add(1);
    if (sink->log) sink->log->warn("Vulkan [{}] {}", id, text);
  }
  // VK_FALSE : l'appel qui a produit le message continue normalement.
  return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT VkContext::messenger_info() const {
  return {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
          .messageSeverity = kSeverities,
          .messageType = kTypes,
          .pfnUserCallback = &VkContext::on_debug_message,
          .pUserData = debug_.get()};
}

Done VkContext::create_messenger() {
  const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      get_instance_proc_addr_(instance_, "vkCreateDebugUtilsMessengerEXT"));
  const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      get_instance_proc_addr_(instance_, "vkDestroyDebugUtilsMessengerEXT"));
  if (create == nullptr || destroy == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  "vkCreateDebugUtilsMessengerEXT ou vkDestroyDebugUtilsMessengerEXT absente du chargeur");
  }
  const VkDebugUtilsMessengerCreateInfoEXT info = messenger_info();
  if (auto ok = check(create(instance_, &info, nullptr, &messenger_), "vkCreateDebugUtilsMessengerEXT"); !ok) {
    return ok;
  }
  destroy_messenger_ = destroy;
  return {};
}

void VkContext::destroy_messenger() {
  if (messenger_ != VK_NULL_HANDLE && destroy_messenger_ != nullptr) {
    destroy_messenger_(instance_, messenger_, nullptr);
  }
  messenger_ = VK_NULL_HANDLE;
}

}  // namespace ajp::gpu::vk
