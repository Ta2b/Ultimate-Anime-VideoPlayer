// Canal de journal du module audio. En-tête interne, inclus par les .cpp seulement.
#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace ajp::audio {

// Clone du journal par défaut, nommé « audio » : mêmes sorties, sans toucher au
// registre global de spdlog. Chaque objet du module qui journalise garde le sien.
std::shared_ptr<spdlog::logger> make_logger();

}  // namespace ajp::audio
