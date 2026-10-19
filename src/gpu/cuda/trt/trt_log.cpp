#include "gpu/cuda/trt/trt_log.h"

#include <utility>

#include <spdlog/logger.h>

namespace ajp::gpu::trt {
namespace {

spdlog::level::level_enum level_of(nvinfer1::ILogger::Severity severity) {
  using Severity = nvinfer1::ILogger::Severity;
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return spdlog::level::critical;
    case Severity::kERROR:
      return spdlog::level::err;
    case Severity::kWARNING:
      return spdlog::level::warn;
    case Severity::kINFO:
      return spdlog::level::debug;
    case Severity::kVERBOSE:
      break;
  }
  return spdlog::level::trace;
}

}  // namespace

TrtLog::TrtLog(std::shared_ptr<spdlog::logger> log) : log_(std::move(log)) {}

void TrtLog::log(Severity severity, nvinfer1::AsciiChar const* message) noexcept {
  if (message == nullptr) return;
  const bool error = severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR;
  // Rien ne doit sortir de ce rappel noexcept, pas même un bad_alloc.
  try {
    if (error) {
      const std::scoped_lock lock(mutex_);
      last_error_ = message;
    }
    const auto level = level_of(severity);
    if (log_ && log_->should_log(level)) log_->log(level, "{}", message);
  } catch (...) {  // NOLINT(bugprone-empty-catch) : le message est perdu, TensorRT continue
  }
}

std::string TrtLog::take_last_error() {
  const std::scoped_lock lock(mutex_);
  return std::exchange(last_error_, {});
}

std::string error_suffix(TrtLog& log) {
  std::string error = log.take_last_error();
  return error.empty() ? std::string{} : " ; TensorRT : " + error;
}

}  // namespace ajp::gpu::trt
