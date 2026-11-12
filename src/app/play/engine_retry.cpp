#include "app/play/engine_retry.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "core/model_index.h"

namespace ajp::app::play {
namespace {

bool contains(const std::vector<core::EngineId>& list, core::EngineId id) {
  return std::ranges::contains(list, id);
}

// Le plus rapide des moteurs de réseaux choisis ET installés ; Vulkan, livré partout, sinon.
void prefer_installed(core::Settings& engines) {
  core::EngineId networks = core::EngineId::Vulkan;
  for (const core::EngineId id : {core::EngineId::Metal, core::EngineId::TensorRt}) {
    if (contains(engines.chosen_engines, id) && contains(engines.installed_engines, id)) networks = id;
  }
  for (const core::Usage usage : {core::Usage::Upscale, core::Usage::Interpolate})
    engines.preferred[static_cast<std::size_t>(usage)] = networks;
}

}  // namespace

bool install_failure_is_transient(RefusalCode code) {
  switch (code) {
    case RefusalCode::HashMismatch:
    case RefusalCode::InsufficientSpace:
    case RefusalCode::FileUnwritable:
    case RefusalCode::SdkRejected:
    case RefusalCode::FormatVersion:
    case RefusalCode::Cancelled:
      return false;
    default:
      return true;
  }
}

core::Settings postponed_engine(core::Settings engines, core::EngineId id) {
  std::erase(engines.installed_engines, id);
  prefer_installed(engines);
  return engines;
}

core::Settings installed_engine(core::Settings engines, core::EngineId id) {
  if (!contains(engines.installed_engines, id)) engines.installed_engines.push_back(id);
  prefer_installed(engines);
  return engines;
}

bool is_postponed(const core::Settings& engines, core::EngineId id) {
  return contains(engines.chosen_engines, id) && !contains(engines.installed_engines, id);
}

std::filesystem::path retry_marker(const std::filesystem::path& engines_root, core::EngineId id) {
  return engines_root / (std::string(core::engine_name(id)) + ".retry");
}

bool retry_install_due(const std::filesystem::path& marker) {
  std::error_code error;
  const auto written = std::filesystem::last_write_time(marker, error);
  if (error) return true;
  const auto age = std::filesystem::file_time_type::clock::now() - written;
  return age < std::filesystem::file_time_type::duration::zero() || age >= kRetryInstallAfter;
}

void note_install_attempt(const std::filesystem::path& marker) {
  std::error_code error;
  std::filesystem::create_directories(marker.parent_path(), error);
  std::ofstream(marker, std::ios::binary | std::ios::trunc).put('\n');
}

}  // namespace ajp::app::play
