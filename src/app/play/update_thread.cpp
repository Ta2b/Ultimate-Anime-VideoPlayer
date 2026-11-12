#include "app/play/update_thread.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stop_token>
#include <utility>

#include "app/play/paths.h"
#include "app/play/player.h"
#include "app/play/settings.h"
#include "app/play/updater.h"
#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {

std::jthread start_update_check(const std::filesystem::path& settings_path, const PlayerPaths& paths,
                                std::atomic<bool>& playing, UpdateNews& news) {
  const std::filesystem::path update_file = paths::update_state(paths.data);
  auto started = start_thread(
      "updates", ThreadReport([](const Refusal& refused) {
        log_channel("app")->warn("mise à jour : {}", refused.detail);  // i18n-ok : journal
      }),
      [=, &playing, &news](const std::stop_token& stop, const ThreadReport&) {
        std::mutex mutex;
        std::condition_variable_any wake;
        for (bool first = true; !stop.stop_requested(); first = false) {
          // B36-02 : relu sans toucher à la base de fusion de play.toml, que seul le lecteur avance.
          const auto saved = peek_settings(settings_path);
          const UpdateMode mode = saved ? saved->update_mode : UpdateMode::Notify;
          const UpdateFrequency frequency = saved ? saved->update_frequency : UpdateFrequency::Daily;
          const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                                       std::chrono::system_clock::now().time_since_epoch())
                                       .count();
          check_updates_if_due(update_file, mode, frequency, first || !playing.load(), now, stop);
          if (mode == UpdateMode::Notify && !news.shown.load() &&
              !load_update_state(update_file).latest_version.empty())
            news.fresh.store(true);
          if (frequency == UpdateFrequency::Launch) return;
          std::unique_lock lock(mutex);
          wake.wait_for(lock, stop, std::chrono::minutes(30), [] { return false; });
        }
      });
  if (started) return std::move(*started);
  log_channel("app")->warn("mise à jour : {}", started.error().detail);  // i18n-ok : journal
  return {};
}

}  // namespace ajp::app::play
