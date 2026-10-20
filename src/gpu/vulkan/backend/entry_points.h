// Points d'entrée extern "C" d'une bibliothèque de backend bâtie sur VulkanBackend
// (gpu/backend.h) : ajp_backend_vulkan, et ajp_backend_migraphx, ajp_backend_openvino dont les
// réseaux passent par un autre runtime. Aucune exception ne traverse la frontière : une exception
// interne devient un refus nommé. Le journal de la bibliothèque (sa propre copie de spdlog) va au
// rappel BackendInit::log ; avec plusieurs backends créés, le dernier rappel reçoit tout.
//
// Inclus par un seul fichier de chaque bibliothèque, qui écrit :
//   AJP_VULKAN_BACKEND_ENTRY_POINTS("Vulkan", options_function)
#pragma once

#include <exception>
#include <format>
#include <memory>
#include <string>
#include <string_view>

#include <spdlog/sinks/callback_sink.h>

#include "core/log.h"
#include "gpu/backend.h"
#include "gpu/vulkan/backend/vulkan_backend.h"

namespace ajp::gpu::vk::entry {

inline BackendLogLevel level_of(spdlog::level::level_enum level) {
  switch (level) {
    case spdlog::level::trace:
    case spdlog::level::debug:
      return BackendLogLevel::Debug;
    case spdlog::level::info:
      return BackendLogLevel::Info;
    case spdlog::level::warn:
      return BackendLogLevel::Warn;
    default:
      return BackendLogLevel::Error;
  }
}

// Les canaux de spdlog de la bibliothèque vers le rappel du lecteur.
inline void route_logs(const BackendLog& log) {
  if (!log) return;
  auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>([log](const spdlog::details::log_msg& msg) {
    log(level_of(msg.level), std::string_view(msg.payload.data(), msg.payload.size()));
  });
  install_log_sinks({sink}, spdlog::level::debug);
}

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
    route_logs(init->log);
    auto backend = VulkanBackend::create(*init, make_options());
    if (!backend) {
      tell(init, backend.error());
      return nullptr;
    }
    return backend->release();
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
