// Canal de journal du module subs. Interne au module.
#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace ajp::subs {

// Clone du journal par défaut, nommé « subs » : mêmes sorties, sans toucher au
// registre global de spdlog. Chaque objet du module garde le sien.
std::shared_ptr<spdlog::logger> make_logger();

}  // namespace ajp::subs
