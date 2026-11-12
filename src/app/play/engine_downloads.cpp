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
#include "app/play/engine_retry.h"
#include "app/play/engine_versions.h"
#include "app/play/home_screen.h"
#include "app/play/i18n.h"
#include "app/play/player_engines.h"
#include "app/play/pumped_task.h"
#include "app/play/sleep_watch.h"
#include "app/play/visible_error.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"

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
                                                        const i18n::Translated& label, Job job) {
  auto shared = std::make_shared<Shared>();
  // L'architecture du GPU (nvidia-smi, sinon le nom de la carte) : sans elle, les fichiers de toutes les
  // architectures étaient tirés (2 Go au lieu de 740 Mo pour TensorRT sm89).
  const std::string arch =
      effective_cuda_arch(pumped_cuda_arch(screen.device_facts().name), screen.device_facts().name);
  auto worker =
      start_thread("install", ThreadReport([shared](const Refusal& refusal) {
                     const std::scoped_lock lock(shared->mutex);
                     if (!shared->result) shared->result = std::unexpected(refusal);
                   }),
                   [shared, arch, job = std::move(job)](const std::stop_token& stop, const ThreadReport&) {
                     auto done = job({.report =
                                          [shared](float fraction, const std::string& stage) {
                                            const std::scoped_lock lock(shared->mutex);
                                            shared->fraction = fraction;
                                            shared->stage = stage;
                                          },
                                      .stop = stop,
                                      .cuda_arch = arch});
                     const std::scoped_lock lock(shared->mutex);
                     shared->result = std::move(done);
                   });
  if (!worker) return std::unexpected(std::move(worker.error()));
  DeferredDrops drops;  // une vidéo envoyée par un second lancement pendant l'installation (B29-04)
  // « Annuler et regarder » : le téléchargement s'arrête (le thread rend la main vite, il lit le jeton), la
  // vidéo se lit sans ce moteur, l'ancien paquet gardé ou Vulkan. Fermer la fenêtre reste une sortie.
  const AwakeWhile awake;  // 650 Mo à tirer : l'écran ne s'éteint pas en route
  bool watch_now = false;
  bool cancelling = false;
  while (true) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (drops.keep(event)) continue;
      if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        worker->request_stop();
        return std::nullopt;
      }
    }
    if (watch_now && !cancelling) {
      cancelling = true;
      worker->request_stop();
    }
    PrepareView view;
    {
      const std::scoped_lock lock(shared->mutex);
      if (shared->result) {
        if (!cancelling) return std::move(shared->result);
        // i18n-ok : journal
        const std::string detail = "installation annulée";
        return std::expected<void, Refusal>(std::unexpected(
            Refusal{.code = RefusalCode::Cancelled, .detail = detail, .key = std::string(kWatchNowKey)}));
      }
      view = PrepareView{.label = label,
                         .fraction = shared->fraction,
                         .details = {shared->stage},
                         .watch_now = cancelling ? nullptr : &watch_now};
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
                                            std::vector<core::EngineId>* failed) {
  const auto log = log_channel("app");
  for (const EngineRow& row : view.engines) {
    if (!row.download || !chosen(engines, row.offer.id)) continue;
    Job job;
    if (const core::EnginePackage* package = package_for(catalog, row.offer.id); package != nullptr) {
      job = [&catalog, package = *package](const InstallProgress& progress) -> std::expected<void, Refusal> {
        auto library = install_package(catalog, package, progress);
        if (!library) return std::unexpected(library.error());
        return {};
      };
    } else {
      continue;
    }
    auto outcome = run_install(screen, ui, i18n::ui_tr_format("Installing {}", engine_label(row.offer.id)),
                               std::move(job));
    if (!outcome) return false;
    if (*outcome) {
      log->info("{} installé", engine_label(row.offer.id));
      continue;
    }
    log->warn("{} not installed: {}", engine_label(row.offer.id), outcome->error().detail);
    if (install_failure_is_transient(outcome->error().code)) {
      // Coupure du réseau, 5xx, portail captif : le choix reste, Vulkan prend le relais, le lancement suivant
      // réessaie (C06-02).
      ui.controls().defer(i18n::ui_tr_format(
          "{} not installed: Vulkan instead, the next launch will try again", engine_label(row.offer.id)));
      engines = postponed_engine(std::move(engines), row.offer.id);
      continue;
    }
    ui.controls().defer(
        i18n::ui_tr_format("{} not installed: {}", engine_label(row.offer.id),
                           i18n::tr("Vulkan for now; install it in Settings > General > Engines")));
    engines = without_engine(std::move(engines), row.offer.id);
    if (failed != nullptr) failed->push_back(row.offer.id);
  }
  return true;
}

std::expected<bool, Refusal> retry_postponed_installs(PlaceboScreen& screen, PlayerUi& ui,
                                                      const EngineCatalog& catalog, core::Settings& engines,
                                                      bool& changed) {
  const auto log = log_channel("app");
  for (const core::EnginePackage& package : catalog.packages) {
    if (!is_postponed(engines, package.engine)) continue;
    recover_interrupted_install(catalog, package);
    const std::string label(engine_label(package.engine));
    // Un module déjà là (installé sans que le choix l'ait noté) : rien à télécharger ; son état de version
    // est celui que update_stale_engines traite ensuite.
    if (installed_state(catalog, package).state != PackageState::Absent) {
      engines = installed_engine(std::move(engines), package.engine);
      changed = true;
      continue;
    }
    const std::filesystem::path marker = retry_marker(catalog.layout.engines, package.engine);
    if (!retry_install_due(marker)) continue;
    const std::string arch =
        effective_cuda_arch(pumped_cuda_arch(screen.device_facts().name), screen.device_facts().name);
    if (const auto fits = check_space(install_space_in(catalog, package, arch, {}),
                                      free_bytes(catalog.layout.engines), catalog.layout.engines);
        !fits) {
      log->warn("{} non installé : {}", label, fits.error().detail);
      ui.controls().defer(i18n::ui_tr_format("{} not installed: {}", label, visible_error(fits.error())));
      continue;
    }
    note_install_attempt(marker);
    log->info("{} : installation reprise (échec passager au lancement précédent)", label);
    auto outcome =
        run_install(screen, ui, i18n::ui_tr_format("Installing {}", label),
                    [&catalog, package](const InstallProgress& progress) -> std::expected<void, Refusal> {
                      auto library = install_package(catalog, package, progress);
                      if (!library) return std::unexpected(library.error());
                      return {};
                    });
    if (!outcome) return false;
    if (*outcome) {
      log->info("{} installé", label);
      engines = installed_engine(std::move(engines), package.engine);
      changed = true;
      continue;
    }
    log->warn("{} not installed: {}", label, outcome->error().detail);
    if (install_failure_is_transient(outcome->error().code)) {
      ui.controls().defer(
          i18n::ui_tr_format("{} not installed: Vulkan instead, the next launch will try again", label));
      continue;
    }
    ui.controls().defer(i18n::ui_tr_format("{} not installed: {}", label, i18n::tr("Vulkan instead")));
    engines = without_engine(std::move(engines), package.engine);
    changed = true;
  }
  return true;
}

std::expected<bool, Refusal> update_stale_engines(PlaceboScreen& screen, PlayerUi& ui,
                                                  const EngineCatalog& catalog, const core::Settings& engines,
                                                  bool& updated) {
  const auto log = log_channel("app");
  for (const core::EnginePackage& package : catalog.packages) {
    if (!chosen(engines, package.engine)) continue;
    recover_interrupted_install(catalog, package);
    const std::string arch =
        effective_cuda_arch(pumped_cuda_arch(screen.device_facts().name), screen.device_facts().name);
    const InstalledState found = installed_state(catalog, package, arch);
    if (found.state == PackageState::Absent) continue;
    const std::string label(engine_label(package.engine));
    if (found.state == PackageState::Current) {
      const std::filesystem::path library = core::backend_library(catalog.layout, package);
      // Ouverte sur un thread, la fenêtre pompée ; à jour, elle reste ouverte pour le choix du moteur qui
      // suit (keep_open).
      auto opened = run_pumped<std::shared_ptr<BackendLibrary>>(
          "backend", [library] { return BackendLibrary::open(std::span(&library, 1)); });
      if (!opened && opened.error().key == kWindowClosedKey)
        return false;  // la croix : plus rien à mettre à jour
      if (opened) BackendLibrary::keep_open(std::move(*opened));
      if (opened || opened.error().code != RefusalCode::AbiMismatch) continue;
      // Le paquet de l'index est celui qui est installé : le retélécharger ne changerait rien (A06-03). La
      // release a gardé un module construit pour un autre lecteur ; Vulkan prend la suite, et c'est dit.
      log->warn("{} : module du paquet de cette version périmé, Vulkan", label);
      ui.controls().defer(
          i18n::ui_tr_format("{}: the engine package of this version is out of date, Vulkan instead", label));
      continue;
    }
    // Un autre paquet à installer : SHA-256 changé à version égale (A26-01), ou version de TensorRT changée
    // (A06-01). Rien n'est noté pour un moteur installé avant la note du paquet : il ne l'a été que par
    // l'écran des moteurs, licence acceptée, et c'est la même que celle du paquet de cette version.
    if (found.note && found.note->license != package.license) {
      log->warn("{}: module from a previous version, licence changed ({}): General > Engines", label,
                package.license);
      ui.controls().defer(i18n::ui_tr_format("{}: new licence to accept (Engines)", label));
      continue;
    }
    // Pas de place : rien n'est tenté (ni boucle à chaque lancement), l'utilisateur est prévenu.
    if (const auto fits = check_space(install_space_in(catalog, package, arch, found.dir),
                                      free_bytes(catalog.layout.engines), catalog.layout.engines);
        !fits) {
      log->warn("{} non mis à jour : {}", label, fits.error().detail);
      ui.controls().defer(i18n::ui_tr_format("{} not updated: {}", label, visible_error(fits.error())));
      continue;
    }
    if (found.state == PackageState::NewVersion)
      log->info("{} : version {} demandée, {} installée : réinstallation", label, package.version,
                core::to_utf8(found.dir.filename()));
    else
      log->info("{} : paquet remplacé dans la release (SHA-256 différent) : réinstallation", label);
    auto outcome =
        run_install(screen, ui, i18n::ui_tr_format("Updating {}", label),
                    [&catalog, package](const InstallProgress& progress) -> std::expected<void, Refusal> {
                      auto installed_library = install_package(catalog, package, progress);
                      if (!installed_library) return std::unexpected(installed_library.error());
                      return {};
                    });
    if (!outcome) return false;
    if (*outcome) {
      log->info("{} mis à jour", label);
      remove_other_versions(catalog, package);
      updated = true;
    } else if (found.state == PackageState::Replaced) {
      // Le paquet d'avant reste en place, entier : il sert jusqu'au prochain essai.
      log->warn("{} non mis à jour : {} ; le paquet installé reste", label, outcome->error().detail);
      ui.controls().defer(i18n::ui_tr_format("{} not updated: {}", label, visible_error(outcome->error())));
    } else {
      log->warn("{} non mis à jour : {}", label, outcome->error().detail);
      ui.controls().defer(i18n::ui_tr_format("{} not updated: Vulkan instead", label));
    }
  }
  return true;
}

}  // namespace ajp::app::play
