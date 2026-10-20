#include "app/play/engine_downloads.h"

#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include "app/play/controls_draw.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kFrameMs = 100;

struct Shared {
  std::mutex mutex;
  float fraction = 0;
  std::string stage;
  std::optional<std::expected<std::filesystem::path, Refusal>> result;
};

// Un paquet, la fenêtre vivante ; rien si elle se ferme.
std::optional<std::expected<std::filesystem::path, Refusal>> install_one(PlaceboScreen& screen, PlayerUi& ui,
                                                                         const EngineCatalog& catalog,
                                                                         const core::EnginePackage& package) {
  auto shared = std::make_shared<Shared>();
  auto worker = start_thread("install", ThreadReport([shared](const Refusal& refusal) {
                               const std::scoped_lock lock(shared->mutex);
                               if (!shared->result) shared->result = std::unexpected(refusal);
                             }),
                             [shared, &catalog, package](const std::stop_token& stop, const ThreadReport&) {
                               auto done =
                                   install_package(catalog, package,
                                                   {.report =
                                                        [shared](float fraction, const std::string& stage) {
                                                          const std::scoped_lock lock(shared->mutex);
                                                          shared->fraction = fraction;
                                                          shared->stage = stage;
                                                        },
                                                    .stop = stop});
                               const std::scoped_lock lock(shared->mutex);
                               shared->result = std::move(done);
                             });
  if (!worker) return std::unexpected(std::move(worker.error()));
  const std::string name(core::engine_name(package.engine));
  while (true) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        worker->request_stop();
        return std::nullopt;
      }
    }
    PrepareView view;
    {
      const std::scoped_lock lock(shared->mutex);
      if (shared->result) return std::move(*shared->result);
      view = PrepareView{.label = std::format("Installation de {}", engine_label(package.engine)),
                         .fraction = shared->fraction,
                         .details = {shared->stage}};
    }
    if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
      log_channel("app")->warn("installation : {}", shown.error().detail);
    SDL_Delay(kFrameMs);
  }
}

}  // namespace

std::expected<bool, Refusal> install_chosen(PlaceboScreen& screen, PlayerUi& ui, const EngineCatalog& catalog,
                                            const FirstRunView& view, core::Settings& engines) {
  const auto log = log_channel("app");
  for (const EngineRow& row : view.engines) {
    if (!row.download || !chosen(engines, row.offer.id)) continue;
    const core::EnginePackage* package = package_for(catalog, row.offer.id);
    if (package == nullptr) continue;
    auto outcome = install_one(screen, ui, catalog, *package);
    if (!outcome) return false;
    if (*outcome) {
      log->info("{} {} installé : {}", core::engine_name(package->engine), package->version,
                (*outcome)->string());
      continue;
    }
    log->warn("{} non installé : {}", core::engine_name(package->engine), outcome->error().detail);
    ui.controls().notify(std::format("{} non installé : Vulkan à la place", engine_label(row.offer.id)),
                         host_now_us());
    engines = without_engine(std::move(engines), row.offer.id);
  }
  return true;
}

}  // namespace ajp::app::play
