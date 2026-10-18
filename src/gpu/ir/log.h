// Canal de journal du module gpu_ir.
#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace ajp::gpu::ir {

// Copie du journal par défaut nommée « gpu_ir », sans l'inscrire au registre
// global de spdlog : chaque appelant garde la sienne.
std::shared_ptr<spdlog::logger> make_logger();

}  // namespace ajp::gpu::ir
