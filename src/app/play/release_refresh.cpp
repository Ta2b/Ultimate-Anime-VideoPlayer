#include "app/play/release_refresh.h"

#include <chrono>
#include <format>
#include <utility>

#include "app/play/paths.h"
#include "app/play/player.h"
#include "app/play/pumped_task.h"
#include "app/play/updater.h"
#include "core/log.h"
#include "core/thread.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

ReleaseRefresh start_release_refresh(const PlayOptions& options, const PlayerPaths& paths,
                                     const PlaySettings& saved) {
  ReleaseRefresh refresh{.cache = paths::data_root(paths) / "engines-nvidia.lock"};
  refresh.before = read_release_index(refresh.cache);
  // « Jamais » coupe les vérifications de version, pas le catalogue des moteurs : l'index apporte le paquet
  // TensorRT de la version installée. Il se relit alors une fois par version (copie ou 404 sans limite
  // d'âge), pas chaque jour (C26-02).
  const bool online = !options.data_dir;
  const bool never = saved.update_mode == UpdateMode::Never;
  if (!online || release_repo().empty() || !release_index_due(refresh.cache, installed_version(), never))
    return refresh;
  // Une copie gardée d'une autre version (le lecteur vient d'être remplacé) ne sert à rien : traitée comme
  // absente, le lancement attend la neuve plutôt que de rester sans le paquet TensorRT de cette version
  // (C26-01).
  if (release_index_tag(refresh.before.text) != std::format("v{}", installed_version())) refresh.before = {};
  auto done = std::make_shared<std::atomic<bool>>(false);
  auto started = start_thread(
      "release-index", ThreadReport([done](const Refusal& refused) {
        log_channel("app")->warn("moteurs de la release : {}", refused.detail);
        done->store(true);
      }),
      [cache = refresh.cache, done](const std::stop_token& stop, const ThreadReport&) {
        refresh_release_index(cache, release_repo(), installed_version(), UpdateFetch(fetch_https), stop);
        done->store(true);
      });
  if (!started) {
    log_channel("app")->warn("moteurs de la release : {}", started.error().detail);
    return refresh;
  }
  refresh.done = std::move(done);
  refresh.thread = std::move(*started);
  return refresh;
}

EngineCatalog engine_catalog(const PlayerPaths& paths, const PlaySettings& saved, ReleaseRefresh& refresh) {
  EngineCatalog catalog = load_catalog(
      paths.engine_index, paths::data_root(paths),
      saved.engines_dir.empty() ? std::filesystem::path() : prepare::path_from_utf8(saved.engines_dir));
  ReleaseIndexCopy copy = std::move(refresh.before);
  if (refresh.done) {
    // Sans copie gardée, l'écran des moteurs et l'assistant ne verraient pas TensorRT : la réponse s'attend
    // (16 s au plus, curl borné), la fenêtre pompée. Avec une copie, elle sert ce lancement-ci, et la neuve
    // le suivant.
    // La fermeture de la fenêtre coupe l'attente (C26-08).
    if (copy.text.empty()) {
      const auto started = std::chrono::steady_clock::now();
      while (!refresh.done->load() && !window_close_requested() &&
             std::chrono::steady_clock::now() - started < std::chrono::seconds(20)) {
        pump_window_events();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
      }
      log_channel("app")->info(
          "moteurs de la release : index attendu {} ms",
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started)
              .count());
    }
    if (refresh.done->load()) copy = read_release_index(refresh.cache);
    if (!refresh.done->load()) refresh.thread.request_stop();  // curl coupé, la copie neuve attendra
  }
  merge_release_copy(catalog, copy, release_repo(),
                     update_public_keys(load_update_state(paths::update_state(paths.data)).revoked_keys),
                     installed_version());
  if (!catalog.problem.empty()) log_channel("app")->info("{}", catalog.problem);
  return catalog;
}

}  // namespace ajp::app::play
