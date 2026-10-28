// Canal de journal du module vk_compute. En-tête interne, inclus par les .cpp seulement.
#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace ajp::gpu::vk {

// Le canal « vk_compute » (core/log.h) : dans un backend, celui-ci journalise par le rappel du
// lecteur. Chaque objet du module qui journalise garde le sien.
std::shared_ptr<spdlog::logger> make_logger();

}  // namespace ajp::gpu::vk
