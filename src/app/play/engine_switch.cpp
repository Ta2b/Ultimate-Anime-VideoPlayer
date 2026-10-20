#include "app/play/engine_switch.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace ajp::app::play {
namespace {

// Les moteurs des réseaux, dans l'ordre de l'onglet, et le mot de leur bibliothèque.
constexpr std::array<std::pair<std::string_view, core::EngineId>, 5> kLibraries{{
    {"vulkan", core::EngineId::Vulkan},
    {"tensorrt", core::EngineId::TensorRt},
    {"migraphx", core::EngineId::MiGraphX},
    {"openvino", core::EngineId::OpenVino},
    {"coreml", core::EngineId::Metal},  // le moteur natif du Mac : Core ML derrière MoltenVK
}};

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

}  // namespace

std::optional<core::EngineId> engine_of_library(const std::filesystem::path& library) {
  const std::string file = library.filename().string();
  if (!file.contains("ajp_backend_")) return std::nullopt;
  for (const auto& [word, id] : kLibraries) {
    if (file.contains(word)) return id;
  }
  return std::nullopt;
}

std::vector<core::EngineId> switchable_engines(std::span<const std::filesystem::path> candidates) {
  std::vector<core::EngineId> out;
  for (const auto& entry : kLibraries) {
    const core::EngineId id = entry.second;
    if (std::ranges::any_of(candidates, [&](const auto& path) { return engine_of_library(path) == id; }))
      out.push_back(id);
  }
  return out;
}

core::Settings using_engine(core::Settings settings, core::EngineId id) {
  if (!std::ranges::contains(settings.chosen_engines, id)) settings.chosen_engines.push_back(id);
  if (!std::ranges::contains(settings.installed_engines, id)) settings.installed_engines.push_back(id);
  settings.preferred[usage(core::Usage::Upscale)] = id;
  settings.preferred[usage(core::Usage::Interpolate)] = id;
  return settings;
}

std::vector<std::filesystem::path> preferred_first(std::vector<std::filesystem::path> candidates,
                                                   const core::Settings& engines) {
  const std::optional<core::EngineId> preferred = engines.preferred[usage(core::Usage::Upscale)];
  if (!preferred) return candidates;
  std::ranges::stable_partition(candidates,
                                [&](const auto& path) { return engine_of_library(path) == preferred; });
  return candidates;
}

}  // namespace ajp::app::play
