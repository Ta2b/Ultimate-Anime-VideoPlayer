#include "app/play/engine_downloads.h"

#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include "app/play/backend_library.h"
#include "app/play/controls_draw.h"
#include "app/play/maxine_install.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

constexpr std::int32_t kFrameMs = 100;

struct Shared {
  std::mutex mutex;
  float fraction = 0;
  std::string stage;
  std::optional<std::expected<void, Refusal>> result;
};

using Job = std::function<std::expected<void, Refusal>(const InstallProgress&)>;

// Une installation sur son thread, la fenêtre vivante ; rien si elle se ferme.
std::optional<std::expected<void, Refusal>> run_install(PlaceboScreen& screen, PlayerUi& ui,
                                                        const std::string& label, Job job) {
  auto shared = std::make_shared<Shared>();
  auto worker =
      start_thread("install", ThreadReport([shared](const Refusal& refusal) {
                     const std::scoped_lock lock(shared->mutex);
                     if (!shared->result) shared->result = std::unexpected(refusal);
                   }),
                   [shared, job = std::move(job)](const std::stop_token& stop, const ThreadReport&) {
                     auto done = job({.report =
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
      if (shared->result) return std::move(shared->result);
      view = PrepareView{.label = label, .fraction = shared->fraction, .details = {shared->stage}};
    }
    if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
      log_channel("app")->warn("installation : {}", shown.error().detail);
    // Une image toutes les 100 ms, plus tôt si un événement arrive.
    (void)SDL_WaitEventTimeout(nullptr, kFrameMs);
  }
}

}  // namespace

std::expected<bool, Refusal> install_chosen(PlaceboScreen& screen, PlayerUi& ui, const EngineCatalog& catalog,
                                            const FirstRunView& view, core::Settings& engines,
                                            const std::filesystem::path& sdk_dir) {
  const auto log = log_channel("app");
  for (const EngineRow& row : view.engines) {
    if (!row.download || !chosen(engines, row.offer.id)) continue;
    Job job;
    if (row.needs_key && catalog.maxine) {
      // La clé quitte la vue pour le seul thread d'installation.
      job = [&catalog, source = *catalog.maxine, key = std::string(row.key.data()),
             sdk_dir](const InstallProgress& progress) {
        return install_maxine(catalog, source, key, sdk_dir, progress);
      };
    } else if (const core::EnginePackage* package = package_for(catalog, row.offer.id);
               package != nullptr && row.offer.id == core::EngineId::MaxineVfx) {
      job = [&catalog, package = *package, sdk_dir](const InstallProgress& progress) {
        return install_maxine_package(catalog, package, sdk_dir, progress);
      };
    } else if (package != nullptr) {
      job = [&catalog, package = *package](const InstallProgress& progress) -> std::expected<void, Refusal> {
        auto library = install_package(catalog, package, progress);
        if (!library) return std::unexpected(library.error());
        return {};
      };
    } else {
      continue;
    }
    auto outcome = run_install(screen, ui, std::format("Installation de {}", engine_label(row.offer.id)),
                               std::move(job));
    if (!outcome) return false;
    if (*outcome) {
      log->info("{} installé", engine_label(row.offer.id));
      continue;
    }
    log->warn("{} non installé : {}", engine_label(row.offer.id), outcome->error().detail);
    ui.controls().notify(std::format("{} non installé : {}", engine_label(row.offer.id),
                                     row.offer.id == core::EngineId::MaxineVfx ? "HDR adaptatif à la place"
                                                                               : "Vulkan à la place"),
                         host_now_us());
    engines = without_engine(std::move(engines), row.offer.id);
  }
  return true;
}

std::expected<bool, Refusal> update_stale_engines(PlaceboScreen& screen, PlayerUi& ui,
                                                  const EngineCatalog& catalog,
                                                  const core::Settings& engines) {
  const auto log = log_channel("app");
  for (const core::EnginePackage& package : catalog.packages) {
    if (package.engine == core::EngineId::MaxineVfx || !chosen(engines, package.engine) ||
        !installed(catalog, package))
      continue;
    const std::filesystem::path library = core::backend_library(catalog.layout, package);
    if (auto opened = BackendLibrary::open(std::span(&library, 1));
        opened || opened.error().code != RefusalCode::AbiMismatch)
      continue;  // la bibliothèque se referme ici, avant que l'installation ne remplace son dossier
    const std::string label(engine_label(package.engine));
    // Rien n'est noté pour un moteur installé avant la note du paquet : il ne l'a été que par
    // l'écran des moteurs, licence acceptée, et c'est la même que celle du paquet de cette version.
    if (const auto previous = installed_package(catalog, package);
        previous && previous->license != package.license) {
      log->warn("{} : module d'une version précédente, licence changée ({}) : onglet Moteurs", label,
                package.license);
      ui.controls().notify(std::format("{} : nouvelle licence à accepter (Moteurs)", label), host_now_us());
      continue;
    }
    log->info("{} : module d'une version précédente du lecteur, réinstallation", label);
    auto outcome =
        run_install(screen, ui, std::format("Mise à jour de {}", label),
                    [&catalog, package](const InstallProgress& progress) -> std::expected<void, Refusal> {
                      auto installed_library = install_package(catalog, package, progress);
                      if (!installed_library) return std::unexpected(installed_library.error());
                      return {};
                    });
    if (!outcome) return false;
    if (*outcome) {
      log->info("{} mis à jour", label);
    } else {
      log->warn("{} non mis à jour : {}", label, outcome->error().detail);
      ui.controls().notify(std::format("{} non mis à jour : Vulkan à la place", label), host_now_us());
    }
  }
  return true;
}

}  // namespace ajp::app::play
