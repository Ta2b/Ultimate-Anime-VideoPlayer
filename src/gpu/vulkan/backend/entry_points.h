// Points d'entrée extern "C" d'une bibliothèque de backend bâtie sur VulkanBackend
// (gpu/backend.h) : ajp_backend_vulkan, et ajp_backend_coreml dont les réseaux passent par Core ML. Aucune
// exception ne traverse la frontière : le backend rendu est enveloppé par GuardedBackend
// (gpu/guarded_backend.h), une exception interne devient un refus nommé. Les canaux de la bibliothèque
// (core/log.h) vont au rappel BackendInit::log ; avec plusieurs backends créés, le dernier rappel reçoit
// tout.
//
// Inclus par un seul fichier de chaque bibliothèque, qui écrit :
//   AJP_VULKAN_BACKEND_ENTRY_POINTS("Vulkan", options_function)
#pragma once

#include <exception>
#include <format>
#include <memory>
#include <string>
#include <string_view>

#include "gpu/backend.h"
#include "gpu/guarded_backend.h"
#include "gpu/module_logs.h"
#include "gpu/vulkan/backend/vulkan_backend.h"

namespace ajp::gpu::vk::entry {

inline void tell(const BackendInit* init, Refusal refusal) {
  if (init != nullptr && init->refusal != nullptr) *init->refusal = std::move(refusal);
}

template <class MakeOptions>
Backend* create(int abi, const BackendInit* init, std::string_view label, MakeOptions make_options) noexcept {
  if (init == nullptr) return nullptr;
  if (abi != kBackendAbi) {
    tell(init, {RefusalCode::AbiMismatch,
                std::format("backend {} d'interface {}, appel en version {}", label, kBackendAbi, abi)});
    return nullptr;
  }
  try {
    route_module_logs(init->log);
    auto backend = VulkanBackend::create(*init, make_options());
    if (!backend) {
      tell(init, backend.error());
      return nullptr;
    }
    return new GuardedBackend(std::move(*backend));  // NOLINT(cppcoreguidelines-owning-memory)
  } catch (const std::exception& e) {
    tell(init, {RefusalCode::LibraryUnavailable, std::format("backend {} : exception {}", label, e.what())});
  } catch (...) {
    tell(init, {RefusalCode::LibraryUnavailable, std::format("backend {} : exception inconnue", label)});
  }
  return nullptr;
}

}  // namespace ajp::gpu::vk::entry

#define AJP_VULKAN_BACKEND_ENTRY_POINTS(label, make_options)                                    \
  extern "C" AJP_BACKEND_EXPORT int ajp_backend_abi() noexcept {                                \
    return ajp::gpu::kBackendAbi;                                                               \
  }                                                                                             \
  extern "C" AJP_BACKEND_EXPORT std::uint64_t ajp_backend_layout() noexcept {                   \
    return ajp::gpu::kBackendLayout;                                                            \
  }                                                                                             \
  extern "C" AJP_BACKEND_EXPORT ajp::gpu::Backend* ajp_backend_create(                          \
      int abi, const ajp::gpu::BackendInit* init) noexcept {                                    \
    return ajp::gpu::vk::entry::create(abi, init, label, make_options);                         \
  }                                                                                             \
  extern "C" AJP_BACKEND_EXPORT void ajp_backend_destroy(ajp::gpu::Backend* backend) noexcept { \
    delete backend;                                                                             \
  }
