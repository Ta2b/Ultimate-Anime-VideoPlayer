// L'onglet Moteurs : SPAN et RIFE passent sur un autre moteur installé sans relancer le lecteur.
// Le moteur choisi devient le préféré des réseaux (settings.toml) ; sa bibliothèque s'essaie alors
// en premier, et Vulkan reste le recours si elle refuse le GPU.
#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "core/engines.h"
#include "core/settings.h"

namespace ajp::app::play {

// Le moteur d'une bibliothèque de backend, par son nom (« ajp_backend_tensorrt.so ») ; rien
// pour un fichier qui n'en est pas une.
std::optional<core::EngineId> engine_of_library(const std::filesystem::path& library);

// Les moteurs des réseaux dont la bibliothèque est parmi candidates, sans doublon, dans l'ordre de
// l'onglet : Vulkan, TensorRT, MIGraphX, OpenVINO, Core ML.
std::vector<core::EngineId> switchable_engines(std::span<const std::filesystem::path> candidates);

// settings avec id choisi, installé et préféré pour SPAN et RIFE.
core::Settings using_engine(core::Settings settings, core::EngineId id);

// candidates, la bibliothèque du moteur préféré pour les réseaux en tête ; les autres gardent leur
// ordre.
std::vector<std::filesystem::path> preferred_first(std::vector<std::filesystem::path> candidates,
                                                   const core::Settings& engines);

}  // namespace ajp::app::play
