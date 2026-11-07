#include "app/play/pumped_task.h"

#include <string>

#include <SDL3/SDL_events.h>

#include "app/play/engine_install.h"

namespace ajp::app::play {

void pump_window_events() {
  SDL_PumpEvents();
}

std::string pumped_cuda_arch() {
  return run_pumped<std::string>("nvidia-smi", [] { return detect_cuda_arch(); }).value_or(std::string());
}

}  // namespace ajp::app::play
