#include "app/play/host_choices.h"

#include <optional>
#include <string>
#include <utility>

#include "app/play/auto_bench.h"
#include "app/play/default_player.h"
#include "app/play/engine_screen.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/placebo_screen.h"
#include "audio/devices.h"
#include "core/gpu_device.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

void fill_host_choices(SettingsChoices& choices, const PlaceboScreen& screen, const PlayerPaths& paths,
                       const EngineCatalog& catalog, const std::filesystem::path& settings_dir) {
  // La tranche de la machine pour Vidéo > Post-traitement : la mesure de « Évaluer ma machine » si elle est
  // de cette carte et de ce pilote, sinon l'estimation.
  std::optional<BenchResult> measured;
  if (auto bench = load_bench(settings_dir / "bench.toml")) measured = std::move(*bench);
  const GpuFacts machine = gpu_facts(screen);
  choices.machine_tier = static_cast<int>(effective_tier(machine, measured));
  choices.bench_text = measured && measured->gpu == machine.name && measured->driver == machine.driver
                           ? i18n::tr_format("Last optimization: {}", measured->date)
                           : std::string();
  choices.gpu_name = machine.name;
  choices.gpus = screen.device_names();
  choices.nvidia = vendor_from_pci(screen.device_facts().pci_vendor) == GpuVendor::Nvidia;
  choices.audio_outputs.clear();
  for (const audio::AudioDevice& device : audio::playback_devices())
    choices.audio_outputs.push_back(
        {.name = device.name, .channels = device.channels, .is_default = device.is_default});
  choices.luts_dir = paths.engines.parent_path() / "luts";
  choices.switchable = installed_switchable(catalog, paths);
  choices.engines_dir_default = prepare::utf8_path(paths.engines.parent_path() / "backends");
  choices.engines_dir_free = format_size(free_bytes(catalog.layout.engines));
  choices.place = place_of(paths);
  choices.default_player_is_ajp = is_default_player(choices.place);
  choices.settings_dir = settings_dir;
}

}  // namespace ajp::app::play
