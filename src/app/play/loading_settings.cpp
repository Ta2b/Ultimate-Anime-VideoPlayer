#include "app/play/loading_settings.h"

#include <type_traits>
#include <utility>
#include <variant>

#include "app/play/player_engines.h"
#include "app/play/player_run.h"
#include "app/play/ui.h"
#include "core/log.h"

namespace ajp::app::play {

PlaySettings LoadingChange::settings_or(const PlaySettings& current) const {
  if (restart) return *restart;
  if (reopen) return *reopen;
  return live.value_or(current);
}

void take_loading_inputs(PlayerUi& ui, std::vector<Input>& actions, LoadingChange& change) {
  std::vector<Input> kept;
  for (Input& input : actions) {
    const bool taken = std::visit(
        [&]<typename T>(T& value) {
          if constexpr (std::is_same_v<T, ToggleSettings>) {
            ui.toggle_settings();
          } else if constexpr (std::is_same_v<T, ApplySettings>) {
            if (value.reopen) {
              change.reopen = value.settings;
            } else {
              change.live = value.settings;
              ui.set_running(value.settings);  // le panneau ne redemande pas ce qu'il vient de livrer
            }
          } else if constexpr (std::is_same_v<T, RestartPlayer>) {
            change.restart = std::move(value.settings);
          } else if constexpr (std::is_same_v<T, UseEngine> || std::is_same_v<T, OpenEngineScreen>) {
            change.engines = EngineChange(value);
          } else {
            return false;
          }
          return true;
        },
        input);
    if (!taken) kept.push_back(std::move(input));
  }
  actions = std::move(kept);
  if (change.stops())
    log_channel("app")->info("préparation arrêtée : un réglage du menu demande de rouvrir la vidéo");
}

std::optional<Reopen> stopped_preparation(Player& p, const Refusal& refusal, const PlaySettings& settings,
                                          std::int64_t start_us, bool start_paused) {
  if (refusal.key == kDroppedVideoKey && p.dropped_video) {
    // A08-08 : une vidéo glissée pendant la construction d'un moteur : la lecture passe à elle.
    return Reopen{
        .settings = settings, .position_us = start_us, .next_file = std::exchange(p.dropped_video, {})};
  }
  if (refusal.key != kSettingsChangedKey) return std::nullopt;
  LoadingChange change = std::exchange(p.loading, {});
  Reopen reopen{.settings = change.settings_or(settings),
                .position_us = start_us,
                .paused = start_paused,
                .engines = std::move(change.engines),
                .restart = change.restart.has_value()};
  return reopen;
}

}  // namespace ajp::app::play
