// Les canaux d'une bibliothèque de backend (core/log.h) vers le rappel BackendInit::log du lecteur, pour
// chaque backend chargé à l'exécution (Vulkan, TensorRT). Avec plusieurs backends créés, le dernier
// rappel reçoit tout.
#pragma once

#include <memory>
#include <string_view>

#include <spdlog/sinks/callback_sink.h>

#include "core/log.h"
#include "gpu/backend.h"

namespace ajp::gpu {

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

// Par le répartiteur de la bibliothèque et non par le journal par défaut : sous Windows, spdlog.dll est
// partagé avec le lecteur, dont le rappel écrit dans ce journal ; il s'y appellerait lui-même (« resource
// deadlock would occur »).
inline void route_module_logs(const BackendLog& log) {
  if (!log) return;
  auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>([log](const spdlog::details::log_msg& msg) {
    log(level_of(msg.level), std::string_view(msg.payload.data(), msg.payload.size()));
  });
  install_module_log_sinks({sink}, spdlog::level::debug);
}

}  // namespace ajp::gpu
