// Canal de journal du module vk_compute. En-tête interne, inclus par les .cpp seulement.
#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace ajp::gpu::vk {

// Clone du journal par défaut, nommé « vk_compute » : mêmes sorties, sans toucher au registre
// global de spdlog. Chaque objet du module qui journalise garde le sien.
std::shared_ptr<spdlog::logger> make_logger();

}  // namespace ajp::gpu::vk
