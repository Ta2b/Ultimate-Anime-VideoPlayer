#include "app/play/host_choices.h"

#include <optional>
#include <string>
#include <utility>

#include "app/play/auto_bench.h"
#include "app/play/default_player.h"
#include "app/play/engine_screen.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/machine_calibration.h"
#include "app/play/paths.h"
#include "app/play/placebo_screen.h"
#include "audio/devices.h"
#include "core/gpu_device.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

void fill_host_choices(SettingsChoices& choices, const PlaceboScreen& screen, const PlayerPaths& paths,
                       const EngineCatalog& catalog, const std::filesystem::path& settings_dir) {
  // La tranche de la machine pour Vidéo > Post-traitement : la mesure de « Calibrer cette machine » si elle
  // est de cette carte et de cette branche de pilote (A37-09), sinon l'estimation.
  std::optional<BenchResult> measured;
  if (auto bench = load_bench(paths::bench(settings_dir))) measured = std::move(*bench);
  const GpuFacts machine = gpu_facts(screen);
  choices.machine_tier = static_cast<int>(effective_tier(machine, measured));
  choices.estimated_tier = static_cast<int>(estimate_tier(machine).tier);
  const bool explored = measured && measured_here(*measured, machine);
  // La machine que le lecteur passe à apply_preset à l'ouverture d'une vidéo (player.cpp), pour que « Auto »
  // choisi au panneau donne la même chaîne (A23-03).
  choices.explored_work_height = explored ? measured->work_height : 0;
  choices.explored_rife_model = explored ? measured->rife_model : std::string();
  // Le moteur en service : celui de la bibliothèque chargée, ou à l'accueil le préféré s'il est installé.
  choices.calibration_host = {.gpu = machine.name,
                              .display_height = display_height(screen.window()),
                              .tensorrt = choices.engine_in_use == core::EngineId::TensorRt &&
                                          tensorrt_backend_present(catalog, paths),
                              .driver = machine.driver};
  choices.display_width = display_width(screen.window());  // un écran 21:9 se mesure à sa vraie taille
  choices.calibrated = calibrated_chain(settings_dir, choices.calibration_host);
  choices.gpu_name = machine.name;
  choices.gpus = screen.device_names();
  choices.nvidia = vendor_from_pci(screen.device_facts().pci_vendor) == GpuVendor::Nvidia;
  choices.luts_dir = paths::luts(paths);
  choices.switchable = installed_switchable(catalog, paths);
  choices.engines_dir_default = prepare::utf8_path(paths::backends_install(paths));
  choices.place = place_of(paths);
  choices.settings_dir = settings_dir;
  choices.bench_clip = paths::bench_clip(paths);
  refresh_host_choices(choices, catalog);
}

void refresh_host_choices(SettingsChoices& choices, const EngineCatalog& catalog) {
  choices.audio_outputs.clear();
  for (const audio::AudioDevice& device : audio::playback_devices())
    choices.audio_outputs.push_back(
        {.name = device.name, .channels = device.channels, .is_default = device.is_default});
  choices.engines_dir_free = shown_size(free_bytes(catalog.layout.engines));
  choices.default_player_is_ajp = is_default_player(choices.place);
}

}  // namespace ajp::app::play
