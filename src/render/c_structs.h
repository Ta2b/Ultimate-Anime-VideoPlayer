// Outils internes pour les structures et drapeaux C de libplacebo et de Vulkan.
#pragma once

#include <cstring>
#include <type_traits>

namespace ajp::render::detail {

// Une structure C mise à zéro octet par octet. Dans les API C de libplacebo et de Vulkan, 0 vaut
// « aucun » ou « non renseigné », même pour une énumération sans énumérateur nul
// (pl_handle_type, VkDriverId) : c'est leur valeur par défaut documentée.
template <class T>
T zeroed() {
  static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
  T value;
  std::memset(&value, 0, sizeof(T));
  return value;
}

// Union de drapeaux C (pl_fmt_caps, ImGuiBackendFlags…), en non signé.
template <class... Flags>
constexpr unsigned flag_bits(Flags... bits) {
  return (static_cast<unsigned>(bits) | ...);
}

// Tous les drapeaux de needed sont dans have.
template <class Have>
constexpr bool has_all(Have have, unsigned needed) {
  return (static_cast<unsigned>(have) & needed) == needed;
}

}  // namespace ajp::render::detail
