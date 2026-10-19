// Handles Vulkan non dispatchables rangés en std::uint64_t, comme core/ring.h le fait :
// des pointeurs opaques sur 64 bits, déjà des uint64_t sur 32 bits ; même taille partout.
// En-tête interne, inclus par les .cpp du module seulement, après vulkan.h.
#pragma once

#include <bit>
#include <cstdint>

namespace ajp::present {

template <class Handle>
std::uint64_t to_u64(Handle handle) {
  static_assert(sizeof(Handle) == sizeof(std::uint64_t));
  return std::bit_cast<std::uint64_t>(handle);
}

template <class Handle>
Handle from_u64(std::uint64_t value) {
  static_assert(sizeof(Handle) == sizeof(std::uint64_t));
  return std::bit_cast<Handle>(value);
}

}  // namespace ajp::present
