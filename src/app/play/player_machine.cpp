#include "app/play/player_machine.h"

#include <chrono>
#include <string>
#include <utility>

#include "app/play/engine_install.h"
#include "app/play/engine_switch.h"
#include "app/play/host_facts.h"
#include "app/play/player.h"
#include "app/play/pumped_task.h"
#include "app/play/relaunch_target.h"
#include "app/play/resume.h"
#include "core/log.h"

namespace ajp::app::play {

Machine machine_of(SDL_Window* window, Tier tier, const BenchResult* explored,
                   std::optional<CalibratedChain> calibrated, bool span_fp8) {
  return Machine{.tier = tier,
                 .display_height = display_height(window),
                 .display_hz = display_refresh_hz(window),
                 .explored_work_height = explored != nullptr ? explored->work_height : 0,
                 .explored_rife_model = explored != nullptr ? explored->rife_model : std::string(),
                 .calibrated = std::move(calibrated),
                 .display_width = display_width(window),
                 .span_fp8 = span_fp8};
}

bool relaunch_allowed(bool lost_recently, std::int64_t launched_s) {
  return !(lost_recently && relaunch_now_s() - launched_s < kRelaunchAgainS);
}

std::optional<std::int64_t> resume_from(const std::filesystem::path& store,
                                        const std::filesystem::path& file) {
  return run_pumped_for<std::optional<std::int64_t>>("resume", std::chrono::milliseconds(5000), true,
                                                     [store, file] { return resume_position(store, file); })
      .value_or(std::nullopt);
}

PlaySettings rederived(const PlaySettings& settings, const Machine& machine, const core::SourceInfo& source,
                       const PlayOptions& file_options, const KeptChoices& kept) {
  PlaySettings out = settings_for(apply_preset(settings, machine, source), file_options);
  if (kept.rife) {
    out.rife_factor = settings.rife_factor;
    out.rife_factor_den = settings.rife_factor_den;
  }
  log_channel("app")->info("écran changé : chaîne refaite pour lui ({} p, ×{})", out.work_height,
                           out.rife_factor);
  return out;
}

// La machine à laquelle une calibration se compare (A25-02, A07-05). Le moteur qui jouera SPAN et RIFE : la
// bibliothèque déjà ouverte si le choix n'a pas bougé depuis (opened_for), sinon le préféré s'il est
// installé.
CalibrationHost calibration_host(const GpuFacts& here, SDL_Window* window, const BackendLibrary* library,
                                 const std::optional<core::EngineId>& opened_for,
                                 const core::Settings& engines, const EngineCatalog& catalog,
                                 const PlayerPaths& paths) {
  const std::optional<core::EngineId> preferred =
      engines.preferred[static_cast<std::size_t>(core::Usage::Upscale)];
  const bool tensorrt =
      library != nullptr && preferred == opened_for
          ? engine_of_library(library->path()) == core::EngineId::TensorRt
          : preferred == core::EngineId::TensorRt && tensorrt_backend_present(catalog, paths);
  return {.gpu = here.name,
          .display_height = display_height(window),
          .tensorrt = tensorrt,
          .driver = here.driver,
          .display_hz = display_refresh_hz(window)};  // C37-08
}

}  // namespace ajp::app::play
