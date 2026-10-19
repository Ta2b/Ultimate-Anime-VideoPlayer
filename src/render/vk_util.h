// Outils Vulkan internes à render : handles non dispatchables en 64 bits, chargement des
// fonctions par vkGetInstanceProcAddr.
#pragma once

#include <cstdint>
#include <type_traits>

#include <vulkan/vulkan.h>

namespace ajp::render::detail {

// VkImage, VkSemaphore, VkSurfaceKHR… : un pointeur opaque sur 64 bits, un entier sur 32 bits.
template <class Handle>
std::uint64_t handle_to_u64(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
  } else {
    return static_cast<std::uint64_t>(handle);
  }
}

template <class Handle>
Handle handle_from_u64(std::uint64_t value) {
  if constexpr (std::is_pointer_v<Handle>) {
    // Un handle non dispatchable voyage en entier de 64 bits par contrat (core/ring.h).
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(value));  // NOLINT(performance-no-int-to-ptr)
  } else {
    return static_cast<Handle>(value);
  }
}

// Fonction d'instance ou d'appareil physique, par son nom ; nullptr si le pilote ne l'a pas.
template <class Pfn>
Pfn instance_function(PFN_vkGetInstanceProcAddr get_proc_addr, VkInstance instance, const char* name) {
  return reinterpret_cast<Pfn>(get_proc_addr(instance, name));
}

}  // namespace ajp::render::detail
