// Points d'entrée de ajp_backend_vulkan (gpu/backend.h, ABI 3), seuls symboles exportés de la
// bibliothèque. Aucune exception ne traverse la frontière : une exception interne devient un
// refus nommé. Le journal du backend (sa propre copie de spdlog : vk_compute, gpu_ir) va au
// rappel BackendInit::log ; avec plusieurs backends créés, le dernier rappel reçoit tout.
#include <exception>
#include <format>
#include <memory>
#include <string>

#include <spdlog/sinks/callback_sink.h>

#include "core/log.h"
#include "gpu/backend.h"
#include "gpu/vulkan/backend/vulkan_backend.h"

#if defined(_WIN32)
#define AJP_BACKEND_EXPORT __declspec(dllexport)
#else
#define AJP_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

namespace {

ajp::gpu::BackendLogLevel level_of(spdlog::level::level_enum level) {
  switch (level) {
    case spdlog::level::trace:
    case spdlog::level::debug:
      return ajp::gpu::BackendLogLevel::Debug;
    case spdlog::level::info:
      return ajp::gpu::BackendLogLevel::Info;
    case spdlog::level::warn:
      return ajp::gpu::BackendLogLevel::Warn;
    default:
      return ajp::gpu::BackendLogLevel::Error;
  }
}

// Les canaux de spdlog de la bibliothèque vers le rappel du lecteur.
void route_logs(const ajp::gpu::BackendLog& log) {
  if (!log) return;
  auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>([log](const spdlog::details::log_msg& msg) {
    log(level_of(msg.level), std::string_view(msg.payload.data(), msg.payload.size()));
  });
  ajp::install_log_sinks({sink}, spdlog::level::debug);
}

void tell(const ajp::gpu::BackendInit* init, ajp::Refusal refusal) {
  if (init != nullptr && init->refusal != nullptr) *init->refusal = std::move(refusal);
}

}  // namespace

extern "C" AJP_BACKEND_EXPORT int ajp_backend_abi() noexcept {
  return ajp::gpu::kBackendAbi;
}

extern "C" AJP_BACKEND_EXPORT std::uint64_t ajp_backend_layout() noexcept {
  return ajp::gpu::kBackendLayout;
}

extern "C" AJP_BACKEND_EXPORT ajp::gpu::Backend* ajp_backend_create(
    int abi, const ajp::gpu::BackendInit* init) noexcept {
  if (init == nullptr) return nullptr;
  if (abi != ajp::gpu::kBackendAbi) {
    tell(init,
         {ajp::RefusalCode::AbiMismatch,
          std::format("backend Vulkan d'interface {}, appel en version {}", ajp::gpu::kBackendAbi, abi)});
    return nullptr;
  }
  try {
    route_logs(init->log);
    auto backend = ajp::gpu::vk::VulkanBackend::create(*init);
    if (!backend) {
      tell(init, backend.error());
      return nullptr;
    }
    return backend->release();
  } catch (const std::exception& e) {
    tell(init,
         {ajp::RefusalCode::LibraryUnavailable, std::format("backend Vulkan : exception {}", e.what())});
  } catch (...) {
    tell(init, {ajp::RefusalCode::LibraryUnavailable, "backend Vulkan : exception inconnue"});
  }
  return nullptr;
}

extern "C" AJP_BACKEND_EXPORT void ajp_backend_destroy(ajp::gpu::Backend* backend) noexcept {
  delete backend;
}
