// Le journal de libplacebo (pl_log) relié au canal du module render. Interne à render.
#pragma once

#include <libplacebo/log.h>
#include <spdlog/logger.h>

namespace ajp::render {

// Les messages d'information de libplacebo (description de l'appareil, passes) sont bavards :
// ils descendent d'un niveau, comme mpv les range en « verbose ».
constexpr spdlog::level::level_enum to_spdlog(pl_log_level level) {
  switch (level) {
    case PL_LOG_FATAL:
      return spdlog::level::critical;
    case PL_LOG_ERR:
      return spdlog::level::err;
    case PL_LOG_WARN:
      return spdlog::level::warn;
    case PL_LOG_INFO:
      return spdlog::level::debug;
    case PL_LOG_DEBUG:
    case PL_LOG_TRACE:  // PL_LOG_ALL a la même valeur
      return spdlog::level::trace;
    case PL_LOG_NONE:
      break;
  }
  return spdlog::level::off;
}

// Niveau le plus bavard que le canal publiera : libplacebo ne formate rien en dessous.
constexpr pl_log_level to_placebo(spdlog::level::level_enum level) {
  switch (level) {
    case spdlog::level::trace:
      return PL_LOG_TRACE;
    case spdlog::level::debug:
      return PL_LOG_INFO;
    case spdlog::level::info:
    case spdlog::level::warn:
      return PL_LOG_WARN;
    case spdlog::level::err:
      return PL_LOG_ERR;
    case spdlog::level::critical:
      return PL_LOG_FATAL;
    case spdlog::level::off:
    case spdlog::level::n_levels:
      break;
  }
  return PL_LOG_NONE;
}

// Un pl_log qui écrit dans logger ; logger doit lui survivre.
inline pl_log make_placebo_log(spdlog::logger& logger) {
  const pl_log_params params{
      .log_cb =
          [](void* priv, pl_log_level level, const char* msg) {
            static_cast<spdlog::logger*>(priv)->log(to_spdlog(level), "libplacebo : {}", msg);
          },
      .log_priv = &logger,
      .log_level = to_placebo(logger.level()),
  };
  return pl_log_create(PL_API_VER, &params);
}

}  // namespace ajp::render
