#include "gpu/vulkan/compute/functions.h"

#include <format>
#include <string_view>

#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {
namespace {

// Pointeur générique rendu par vk*ProcAddr vers le type exact de la fonction : les deux sont
// des pointeurs de fonction, et la spécification garantit la conversion.
template <class Pfn>
bool assign(Pfn& slot, PFN_vkVoidFunction address) {
  slot = reinterpret_cast<Pfn>(address);
  return address != nullptr;
}

std::unexpected<Refusal> missing(std::string_view name) {
  return refuse(RefusalCode::LibraryUnavailable, std::format("fonction Vulkan {} absente", name));
}

}  // namespace

std::expected<GlobalFunctions, Refusal> load_global(PFN_vkGetInstanceProcAddr get_instance_proc_addr) {
  if (get_instance_proc_addr == nullptr) return missing("vkGetInstanceProcAddr");
  GlobalFunctions f;
#define AJP_VK_LOAD(name) \
  if (!assign(f.name, get_instance_proc_addr(VK_NULL_HANDLE, #name))) return missing(#name);
  AJP_VK_GLOBAL_FUNCTIONS(AJP_VK_LOAD)
#undef AJP_VK_LOAD
  return f;
}

std::expected<InstanceFunctions, Refusal> load_instance(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                                        VkInstance instance) {
  if (get_instance_proc_addr == nullptr) return missing("vkGetInstanceProcAddr");
  InstanceFunctions f;
#define AJP_VK_LOAD(name) \
  if (!assign(f.name, get_instance_proc_addr(instance, #name))) return missing(#name);
  AJP_VK_INSTANCE_FUNCTIONS(AJP_VK_LOAD)
#undef AJP_VK_LOAD
  return f;
}

std::expected<DeviceFunctions, Refusal> load_device(const InstanceFunctions& instance, VkDevice device) {
  DeviceFunctions f;
#define AJP_VK_LOAD(name) \
  if (!assign(f.name, instance.vkGetDeviceProcAddr(device, #name))) return missing(#name);
  AJP_VK_DEVICE_FUNCTIONS(AJP_VK_LOAD)
#undef AJP_VK_LOAD
  return f;
}

}  // namespace ajp::gpu::vk
