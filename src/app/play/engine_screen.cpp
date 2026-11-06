#include "app/play/engine_screen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include "app/play/default_player.h"
#include "app/play/engine_downloads.h"
#include "app/play/engine_switch.h"
#include "app/play/first_run.h"
#include "app/play/gpu_tier.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/machine_calibration.h"
#include "app/play/profiles.h"
#include "app/play/scripted_input.h"
#include "app/play/settings.h"
#include "app/play/wizard.h"
#include "core/clock.h"
#include "core/keys.h"
#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

// L'écran des moteurs suit la souris : ~60 images par seconde.
constexpr std::int32_t kWelcomeFrameMs = 16;

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

// L'écran jusqu'à « Commencer » ou « Annuler » ; rien si l'utilisateur quitte. Les touches d'essai
// (--keys) qui tombent pendant l'écran y sont pressées, comptées depuis son ouverture.
std::expected<std::optional<FirstRunView>, Refusal> welcome(PlaceboScreen& screen, PlayerUi& ui,
                                                            FirstRunView view, const PlayOptions& options,
                                                            std::size_t& next_key, SnapshotPlan& snapshot,
                                                            int& furthest) {
  const core::KeyMap& keys = configured_keys();
  const std::int64_t started = host_now_us();
  std::vector<Input> actions;
  while (!view.done && !view.cancelled) {
    for (; next_key < options.keys.size() && host_now_us() - started >= options.keys[next_key].first;
         ++next_key)
      press(options.keys[next_key].second, screen.window());
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (ui.handle(event, host_now_us(), actions)) continue;
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    for (const Input& input : actions) {
      if (const auto* command = std::get_if<Command>(&input)) {
        if (std::holds_alternative<cmd::Quit>(*command)) {
          furthest = view.wizard.seen;
          return std::nullopt;
        }
      } else if (std::holds_alternative<ToggleFullscreen>(input)) {
        screen.set_fullscreen(!screen.fullscreen());
      } else if (std::holds_alternative<LeaveFullscreen>(input)) {
        screen.set_fullscreen(false);
      }
    }
    actions.clear();
    snapshot.maybe(screen, started);
    if (auto shown = screen.show(ShowRequest{.ui = ui.welcome(host_now_us(), view)}); !shown)
      return std::unexpected(std::move(shown.error()));
    // Une image toutes les 16 ms, plus tôt si un événement arrive.
    (void)SDL_WaitEventTimeout(nullptr, kWelcomeFrameMs);
  }
  return view;
}

// Un moteur des réseaux ajouté entre before et after.
bool network_added(const core::Settings& before, const core::Settings& after) {
  const std::vector<core::EngineId> networks = {core::EngineId::TensorRt, core::EngineId::MiGraphX,
                                                core::EngineId::OpenVino, core::EngineId::Metal};
  return std::ranges::any_of(networks,
                             [&](core::EngineId id) { return chosen(after, id) && !chosen(before, id); });
}

// La calibration de la machine de l'assistant (machine_calibration.h) sur son thread : l'utilisateur ne voit
// qu'une barre de chargement. Rien si la fenêtre se ferme ; une calibration qui échoue garde base (Auto).
std::optional<PlaySettings> calibrate_on_screen(PlaceboScreen& screen, PlayerUi& ui, const PlayerPaths& paths,
                                                const std::filesystem::path& data_dir, PlaySettings base) {
  const auto log = log_channel("app");
  struct Shared {
    std::mutex mutex;
    float fraction = 0.0F;
    std::optional<std::expected<MachineCalibration, Refusal>> result;
  };
  auto shared = std::make_shared<Shared>();
  std::error_code error;
  BenchPlan plan{.executable = paths.executable,
                 .video = data_dir / "bench-clip-v3.mkv",
                 .work_dir = std::filesystem::temp_directory_path(error) / "ajp-calibration",
                 .data = data_dir,
                 .display_width = display_width(screen.window()),
                 .display_height = display_height(screen.window()),
                 .display_hdr = screen.hdr_output()};
  const int height = plan.display_height;
  auto worker = start_thread("calibration", ThreadReport([shared](const Refusal& refusal) {
                               const std::scoped_lock lock(shared->mutex);
                               if (!shared->result) shared->result = std::unexpected(refusal);
                             }),
                             [shared, plan, base, height](const std::stop_token& stop, const ThreadReport&) {
                               auto done =
                                   calibrate_machine(plan, base, height, stop, [shared](float fraction) {
                                     const std::scoped_lock lock(shared->mutex);
                                     shared->fraction = fraction;
                                   });
                               const std::scoped_lock lock(shared->mutex);
                               shared->result = std::move(done);
                             });
  if (!worker) {
    log->warn("calibration de la machine : {}", worker.error().detail);
    return base;
  }
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
      if (shared->result) break;
      view = PrepareView{.label = i18n::tr("Calibrating this machine"), .fraction = shared->fraction};
    }
    if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
      log->warn("calibration de la machine : {}", shown.error().detail);
    (void)SDL_WaitEventTimeout(nullptr, 100);
  }
  const std::scoped_lock lock(shared->mutex);
  if (!*shared->result) {
    log->warn("calibration de la machine : {} ; préréglage Auto", shared->result->error().detail);
    ui.controls().defer(i18n::tr("Machine calibration failed: Auto preset (see log)"));
    return base;
  }
  return candidate_settings((*shared->result)->chosen, std::move(base));
}

}  // namespace

std::expected<bool, Refusal> run_engine_screen(const EngineScreen& s, core::Settings& engines,
                                               ProfileState& profiles, bool revisit, std::size_t& next_key,
                                               SnapshotPlan& snapshot) {
  const auto log = log_channel("app");
  const GpuFacts gpu = gpu_facts(s.screen);
  ProfilesView choice = view_of(profiles);
  if (!engines.first_run_done) choice.start = start_profile_for(gpu);
  log->info("{} : {}", revisit ? "écran des moteurs" : "bienvenue", gpu_line(gpu));
  FirstRunView view =
      first_run_view(gpu, engines_here(s.options, s.paths, s.catalog, gpu.name), std::move(choice), engines);
  // L'assistant du premier lancement (ou de --first-run, ou relancé depuis les paramètres) ; rouvert depuis
  // l'onglet Moteurs, l'ancien écran des seuls moteurs.
  const bool wizard_mode = !revisit || s.restart_wizard;
  view.revisit = revisit && !wizard_mode;
  if (wizard_mode) {
    const TierEstimate estimate = estimate_tier(gpu);
    view.facts = WizardFacts{
        .engine_choice = std::ranges::any_of(
            view.engines,
            [](const EngineRow& row) { return row.offer.id != core::EngineId::Vulkan && row.usable; }),
        // Avant la première image, la sortie n'est pas encore choisie : l'écran HDR se lit dans le système,
        // comme à l'accueil.
        .hdr_display = !s.options.sdr && content_peak_nits(s.screen.window()).has_value()};
    view.wizard =
        begin_wizard(view.facts, s.options.first_run || s.restart_wizard ? 0 : engines.first_run_step);
    view.tier_line = i18n::tr_format("Estimated level: {}", tier_label(estimate.tier));
    const int height = display_height(s.screen.window());
    const double hz = display_refresh_hz(s.screen.window());
    if (height > 0) view.display_line = i18n::tr_format("Display {}p · {:.0f} Hz", height, hz);
  }
  int furthest = engines.first_run_step;
  auto chosen_view = welcome(s.screen, s.ui, std::move(view), s.options, next_key, snapshot, furthest);
  if (!chosen_view) return std::unexpected(std::move(chosen_view.error()));
  if (!*chosen_view) {
    // Fermé en route : l'assistant reprendra après le dernier écran vu.
    if (wizard_mode && !revisit && furthest > engines.first_run_step) {
      engines.first_run_step = furthest;
      if (auto written = core::save_settings(s.engines_path, engines); !written)
        log->warn("{}", written.error().detail);
    }
    return false;
  }
  if ((*chosen_view)->cancelled) return true;
  const core::Settings before = engines;
  engines = settings_after(**chosen_view, engines);
  auto downloaded = install_chosen(s.screen, s.ui, s.catalog, **chosen_view, engines);
  if (!downloaded) return std::unexpected(std::move(downloaded.error()));
  if (!*downloaded) return false;
  // Le moteur des réseaux choisi à l'onglet Moteurs reste, sauf si un moteur vient d'arriver : le
  // plus rapide des choisis prend alors sa place (settings_after).
  const std::optional<core::EngineId> kept = before.preferred[usage(core::Usage::Upscale)];
  if (revisit && kept && chosen(engines, *kept) && !network_added(before, engines))
    engines = using_engine(std::move(engines), *kept);
  std::optional<PlaySettings> calibrated_chain;  // la chaîne de la calibration de la machine
  if (wizard_mode) {
    const Wizard& wizard = (*chosen_view)->wizard;
    engines = wizard_settings(wizard, std::move(engines));
    const std::filesystem::path play_path = s.engines_path.parent_path() / "play.toml";
    auto play = load_settings(play_path);
    PlaySettings chosen_play;
    if (play) chosen_play = std::move(*play);
    chosen_play = wizard_play_settings(wizard, std::move(chosen_play));
    if (wants_machine_calibration(wizard)) {
      // Les essais relisent settings.toml (moteurs choisis, assistant vu) : il s'écrit avant eux.
      if (auto written = core::save_settings(s.engines_path, engines); !written)
        log->warn("{}", written.error().detail);
      auto calibrated =
          calibrate_on_screen(s.screen, s.ui, s.paths, s.engines_path.parent_path(), std::move(chosen_play));
      if (!calibrated) return false;
      chosen_play = std::move(*calibrated);
      calibrated_chain = chosen_play;
    }
    if (auto written = save_settings(play_path, chosen_play); !written)
      log->warn("{}", written.error().detail);
    // « Ouvrir avec » : une fois, sans confirmation (elle ajoute une entrée, ne prend la place de personne).
    if (!engines.open_with_registered) {
      const PlayerPlace place{.executable = s.paths.executable, .home = s.paths.home, .data = s.paths.data};
      if (auto registered = register_player(place)) {
        engines.open_with_registered = true;
        log->info("Ouvrir avec : {}", *registered);
      } else {
        log->warn("Ouvrir avec : {}", registered.error().detail);
      }
    }
    for (const std::string& line : wizard_summary(wizard)) log->info("assistant : {}", line);
  }
  if (auto written = core::save_settings(s.engines_path, engines); !written)
    log->warn("{}", written.error().detail);
  const std::array<core::UserEdit, 1> start{core::edit::StartProfile{.slot = (*chosen_view)->profiles.start}};
  if (auto edited = edit_profiles(profiles.user_path, profiles.data, start)) {
    profiles.data = std::move(*edited);
  } else {
    log->warn("user.toml : {}", edited.error().detail);
  }
  if (!revisit) profiles.active = profiles.data.file.start_profile;
  // La chaîne calibrée devient celle du profil de démarrage : sans quoi elle prendrait le pas sur play.toml.
  if (calibrated_chain) remember_chain(profiles, *calibrated_chain);
  log->info("moteurs choisis : {} ; profil de démarrage {}", chosen_line(engines),
            profiles.data.file.start_profile);
  return true;
}

std::vector<std::filesystem::path> backend_candidates(const EngineCatalog& catalog,
                                                      const PlayerPaths& paths) {
  std::vector<std::filesystem::path> out = installed_backends(catalog);
  for (const std::filesystem::path& backend : paths.backends) {
    const std::optional<core::EngineId> id = engine_of_library(backend);
    if (!id || core::engine_plugged(*id)) out.push_back(backend);  // MIGraphX, OpenVINO : débranchés
  }
  return out;
}

std::vector<core::EngineId> installed_switchable(const EngineCatalog& catalog, const PlayerPaths& paths) {
  std::vector<std::filesystem::path> present = backend_candidates(catalog, paths);
  std::error_code error;
  std::erase_if(present,
                [&](const std::filesystem::path& path) { return !std::filesystem::exists(path, error); });
  return switchable_engines(present);
}

std::expected<std::shared_ptr<BackendLibrary>, Refusal> open_backend(const EngineCatalog& catalog,
                                                                     const PlayerPaths& paths,
                                                                     const core::Settings& engines,
                                                                     const gpu::BackendInit& probe) {
  const std::vector<std::filesystem::path> candidates =
      preferred_first(backend_candidates(catalog, paths), engines);
  auto library = BackendLibrary::choose(backends_within(candidates, engines), probe);
  // Un moteur préféré qui refuse l'appareil laisse sa place sans bruit : sa raison va au journal.
  if (library && !(*library)->passed_over().empty())
    log_channel("app")->warn("moteurs : refus avant {} :{}", (*library)->path().string(),
                             (*library)->passed_over());
  return library;
}

std::expected<bool, Refusal> handle_engine_change(Player& p, const EngineChange& request) {
  const auto log = log_channel("app");
  const auto* open = std::get_if<OpenEngineScreen>(&request);
  const EngineScreen screen{.screen = p.screen,
                            .ui = p.ui,
                            .options = p.options,
                            .paths = p.paths,
                            .catalog = p.catalog,
                            .engines_path = p.settings_path.parent_path() / "settings.toml",
                            .restart_wizard = open != nullptr && open->wizard};
  const auto* use = std::get_if<UseEngine>(&request);
  if (use != nullptr) {
    p.engines = using_engine(std::move(p.engines), use->id);
    if (auto written = core::save_settings(screen.engines_path, p.engines); !written)
      log->warn("{}", written.error().detail);
  } else {
    auto shown = run_engine_screen(screen, p.engines, p.profiles, true, p.next_key, p.snapshot);
    if (!shown) return std::unexpected(std::move(shown.error()));
    if (!*shown) return false;
  }
  // Le moteur choisi en cours de lecture, installé par une version précédente : son module se met à jour
  // ici comme au lancement, sinon il serait refusé (interface changée) et Vulkan resterait en service.
  bool updated = false;
  auto stale = update_stale_engines(p.screen, p.ui, p.catalog, p.engines, updated);
  if (!stale) return std::unexpected(std::move(stale.error()));
  if (!*stale) return false;
  if (updated && p.paths.relaunch != nullptr) {
    log->info("moteurs mis à jour : le lecteur se relance");
    *p.paths.relaunch = true;
    return false;
  }
  show_loading(p.screen, p.ui, i18n::tr("Loading the engine"));
  const gpu::BackendInit probe{
      .render = &p.screen.device(), .cache_dir = p.paths.engines, .log = backend_log()};
  // Le module en service remplacé par l'écran des moteurs (mise à jour) : le nouveau ne se chargerait pas
  // dans ce processus, le lecteur se relance.
  if (p.library->replaced_on_disk() && p.paths.relaunch != nullptr) {
    log->info("moteur mis à jour : le lecteur se relance");
    *p.paths.relaunch = true;
    return false;
  }
  auto library = open_backend(p.catalog, p.paths, p.engines, probe);
  if (!library) {
    log->warn("moteurs : {} ; {} reste en service", library.error().detail, p.library->path().string());
    p.notice = i18n::tr("Engine unchanged: no other one opens (see log)");
    return true;
  }
  p.library = std::move(*library);
  log->info("moteurs : {}", p.library->path().string());
  const std::optional<core::EngineId> now = engine_of_library(p.library->path());
  const std::string name = now ? std::string(engine_label(*now)) : p.library->path().filename().string();
  if (use != nullptr && now != use->id) {
    p.notice = i18n::tr_format("{} does not open on this GPU: {} stays active", engine_label(use->id), name);
  } else {
    p.notice = i18n::tr_format("Engine: {}", name);
  }
  return true;
}

std::string fallback_note(const core::Settings& engines, const std::filesystem::path& library) {
  const std::optional<core::EngineId> preferred = engines.preferred[usage(core::Usage::Upscale)];
  const std::optional<core::EngineId> in_use = engine_of_library(library);
  if (!preferred || !in_use || preferred == in_use) return {};
  return i18n::tr_format("{} is chosen but does not open on this GPU (see log): {} replaces it.",
                         engine_label(*preferred), engine_label(*in_use));
}

}  // namespace ajp::app::play
