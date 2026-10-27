#include "app/play/home_screen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_video.h>

#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/display_edid.h"
#include "app/play/drba_model.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/first_run.h"
#include "app/play/gmfss_model.h"
#include "app/play/host_facts.h"
#include "app/play/input.h"
#include "app/play/model_choice.h"
#include "app/play/profiles.h"
#include "app/play/scripted_input.h"
#include "core/clock.h"
#include "core/engines.h"
#include "core/keys.h"
#include "core/log.h"
#include "prepare/utf8.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {
namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

// L'accueil suit la souris : ~60 images par seconde.
constexpr std::int32_t kHomeFrameMs = 16;

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

// Les filtres de la boîte « Ouvrir une vidéo » : SDL les lit jusqu'à la réponse, d'où le static.
constexpr std::array<SDL_DialogFileFilter, 2> kVideoFilters{{
    {.name = "Vidéos", .pattern = "mkv;mp4;m4v;mov;avi;webm;ts;m2ts;mts;wmv;flv;ogv;mpg;mpeg"},
    {.name = "Tous les fichiers", .pattern = "*"},
}};

// La réponse de la boîte, rendue par SDL sur un autre thread selon le système. Partagée avec le
// rappel : l'accueil peut se fermer avant elle.
class FilePick {
 public:
  bool open() const {
    const std::scoped_lock lock(mutex_);
    return open_;
  }
  std::optional<std::filesystem::path> take() {
    const std::scoped_lock lock(mutex_);
    return std::exchange(chosen_, std::nullopt);
  }

  // Une boîte à la fois, ouverte depuis la fenêtre du lecteur, dans le dossier des vidéos.
  static void show(const std::shared_ptr<FilePick>& pick, SDL_Window* window) {
    {
      const std::scoped_lock lock(pick->mutex_);
      if (pick->open_) return;
      pick->open_ = true;
    }
    const char* videos = SDL_GetUserFolder(SDL_FOLDER_VIDEOS);
    pick->start_in_ = videos != nullptr ? videos : "";
    SDL_ShowOpenFileDialog(&FilePick::answered, new std::shared_ptr<FilePick>(pick), window,
                           kVideoFilters.data(), static_cast<int>(kVideoFilters.size()),
                           pick->start_in_.empty() ? nullptr : pick->start_in_.c_str(), false);
  }

 private:
  static void SDLCALL answered(void* userdata, const char* const* files, int /*filter*/) {
    const std::unique_ptr<std::shared_ptr<FilePick>> owner(static_cast<std::shared_ptr<FilePick>*>(userdata));
    FilePick& pick = **owner;
    const std::scoped_lock lock(pick.mutex_);
    pick.open_ = false;
    if (files == nullptr) {
      log_channel("app")->warn("Ouvrir une vidéo : {}", SDL_GetError());
    } else if (files[0] != nullptr) {
      pick.chosen_ = prepare::path_from_utf8(files[0]);
    }
  }

  mutable std::mutex mutex_;
  bool open_ = false;
  std::optional<std::filesystem::path> chosen_;
  std::string start_in_;
};

// Ce que le menu propose sans vidéo ouverte : les modèles et les moteurs du poste, la source type.
SettingsChoices home_choices(const HomeScreen& h, const HomeState& st, const PlaySettings& running) {
  const auto model = choose_model(h.options, h.paths.home, running, h.paths.models);
  std::optional<ModelChoice> rife;
  if (auto found = choose_rife_model(h.options, h.paths.home, running, rife_preference(gpu_facts(h.screen)),
                                     h.paths.models))
    rife = std::move(*found);
  const std::vector<std::filesystem::path> places = gmfss_dirs(h.paths, rife);
  const bool gmfss = find_gmfss(places).has_value();
  const bool drba = find_distil_drba(places).has_value();
  const std::int64_t period = display_period_us(h.screen.window());
  const std::optional<DisplayFacts> display = read_display(display_name(h.screen.window()));
  return SettingsChoices{
      .models = model ? model->available : std::vector<std::string>{},
      .default_model = model ? model->default_name : std::string(),
      .source_width = kHomeSource.width,
      .source_height = kHomeSource.height,
      .source_fps = kHomeSource.fps,
      .source_rate = media::Rational{.num = 24000, .den = 1001},
      .display_hz = period > 0 ? 1e6 / static_cast<double>(period) : 0.0,
      .rife_available = rife.has_value() || gmfss || drba || offered(MotionEngine::Apple),
      .gmfss_available = gmfss,
      .gmfss_union_available = find_gmfss(places, GmfssVariant::Union).has_value(),
      .distil_drba_available = drba,
      .rife_loaded = false,
      .rife_models = rife ? rife->available : std::vector<std::string>{},
      .default_rife_model = rife ? rife->default_name : std::string(),
      .rife_folder = places.empty() ? std::string() : prepare::utf8_path(places.front()),
      .nvidia = vendor_from_pci(h.screen.device_facts().pci_vendor) == GpuVendor::Nvidia,
      .true_hdr_available = maxine_available(h.options, h.paths) || running.true_hdr,
      .hdr_output = !h.options.sdr && content_peak_nits(h.screen.window()).has_value(),
      .auto_peak = vendor_hdr::peak_from_display(content_peak_nits(h.screen.window())),
      .engines = engine_badges(h.paths.engines, {}, WorkSize{}, std::nullopt, kernels::RifeGeometry{}),
      .chosen_engines = chosen_line(st.engines),
      .switchable = installed_switchable(h.catalog, h.paths),
      .engine_in_use = st.engines.preferred[usage(core::Usage::Upscale)],
      .display_text = display ? describe(*display) : std::string("EDID illisible : OLED inconnu"),
      .source_file = {},
      .place = {.executable = h.paths.executable, .home = h.paths.home, .data = h.paths.data}};
}

// Les réglages que la lecture prendra : play.toml, la chaîne du profil actif pour la source type,
// puis les options de la ligne de commande.
PlaySettings home_settings(const HomeScreen& h, const HomeState& st, const PlaySettings& base) {
  PlaySettings settings = base;
  if (const auto& profile =
          st.profiles.data.file
              .profiles[static_cast<std::size_t>(std::clamp(st.profiles.active, 1, 9) - 1)]) {
    std::vector<std::string> notes;
    settings = apply_profile(*profile, kHomeSource, base, notes);
  }
  return settings_for(settings, h.options);
}

}  // namespace

std::optional<std::filesystem::path> dropped_file(const SDL_Event& event) {
  if (event.type != SDL_EVENT_DROP_FILE || event.drop.data == nullptr) return std::nullopt;
  return prepare::path_from_utf8(event.drop.data);
}

std::expected<std::optional<std::filesystem::path>, Refusal> run_home(const HomeScreen& h, HomeState& st,
                                                                      const std::string& problem) {
  const auto log = log_channel("app");
  log->info("accueil : Ouvrir une vidéo, ou un fichier glissé sur la fenêtre");
  SDL_SetWindowTitle(h.screen.window(), "ajp");
  st.profiles.source = kHomeSource;
  const PlaySettings from_file = st.saved;
  PlaySettings running = home_settings(h, st, st.saved);
  const PlaySettings started = running;
  const auto refresh = [&] {
    h.ui.set_settings(running, home_choices(h, st, running));
    h.ui.set_profiles(view_of(st.profiles));
  };
  refresh();
  h.ui.open_settings();
  const auto pick = std::make_shared<FilePick>();
  HomeView view{.problem = problem};
  bool dirty = false;
  std::int64_t saved_at = host_now_us();
  const auto save = [&] {
    st.saved = settings_to_save(running, from_file, started, h.options);
    if (auto written = save_settings(h.settings_path, st.saved); !written)
      log->warn("{}", written.error().detail);
    dirty = false;
    saved_at = host_now_us();
  };
  const auto changed = [&](const PlaySettings& wanted) {
    running = wanted;
    running.volume = h.ui.controls().volume();
    h.ui.set_running(running);
    dirty = true;
  };
  const auto use_profile = [&] {
    changed(home_settings(h, st, running));
    h.ui.set_profiles(view_of(st.profiles));
  };

  const core::KeyMap keys = core::KeyMap::defaults();
  const std::int64_t opened = host_now_us();
  std::vector<Input> actions;
  std::optional<std::filesystem::path> chosen;
  bool quit = false;
  while (!chosen && !quit) {
    for (; st.next_key < h.options.keys.size() && host_now_us() - opened >= h.options.keys[st.next_key].first;
         ++st.next_key)
      press(h.options.keys[st.next_key].second, h.screen.window());
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (auto dropped = dropped_file(event)) {
        if (!chosen) chosen = std::move(*dropped);
        continue;
      }
      if (h.ui.handle(event, host_now_us(), actions)) continue;
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    if (!chosen) chosen = pick->take();
    for (const Input& input : actions) {
      std::visit(
          Overloaded{[&](const Command& command) {
                       if (std::holds_alternative<cmd::Quit>(command)) {
                         quit = true;
                       } else if (const auto* select = std::get_if<cmd::SelectProfile>(&command)) {
                         if (!st.profiles.data.file
                                  .profiles[static_cast<std::size_t>(std::clamp(select->slot, 1, 9) - 1)])
                           return;
                         st.profiles.active = std::clamp(select->slot, 1, 9);
                         log->info("profil {}", st.profiles.active);
                         use_profile();
                       }
                     },
                     [&](const ToggleFullscreen&) { h.screen.set_fullscreen(!h.screen.fullscreen()); },
                     [&](const LeaveFullscreen&) {
                       if (h.ui.settings_open()) {
                         h.ui.close_settings();
                       } else if (h.screen.fullscreen()) {
                         h.screen.set_fullscreen(false);
                       }
                     },
                     [&](const ToggleSettings&) { h.ui.toggle_settings(); },
                     [&](const ToggleMute&) {
                       static_cast<void>(h.ui.controls().toggle_mute());
                       changed(running);
                     },
                     [&](const VolumeStep& step) {
                       static_cast<void>(h.ui.controls().step_volume(step.delta));
                       changed(running);
                     },
                     [&](const ApplySettings& apply) {
                       // Comme pendant la lecture : changer la chaîne au menu modifie le profil actif.
                       if (chain_differs(running, apply.settings)) {
                         remember_chain(st.profiles, apply.settings);
                         h.ui.set_profiles(view_of(st.profiles));
                       }
                       changed(apply.settings);
                     },
                     [&](const EditProfile& edit) {
                       const std::array<core::UserEdit, 1> edits{edit.edit};
                       auto edited = edit_profiles(st.profiles.user_path, st.profiles.data, edits);
                       if (!edited) {
                         log->warn("user.toml : {}", edited.error().detail);
                         return;
                       }
                       st.profiles.data = std::move(*edited);
                       h.ui.set_profiles(view_of(st.profiles));
                       if (std::holds_alternative<core::edit::ResetProfile>(edit.edit)) use_profile();
                     },
                     [&](const UseEngine& use) {
                       st.engines = using_engine(std::move(st.engines), use.id);
                       if (auto written = core::save_settings(h.settings_path.parent_path() / "settings.toml",
                                                              st.engines);
                           !written)
                         log->warn("{}", written.error().detail);
                       log->info("moteur des réseaux : {}", engine_label(use.id));
                       refresh();
                     },
                     [&](const OpenEngineScreen&) {
                       const EngineScreen screen{
                           .screen = h.screen,
                           .ui = h.ui,
                           .options = h.options,
                           .paths = h.paths,
                           .catalog = h.catalog,
                           .engines_path = h.settings_path.parent_path() / "settings.toml"};
                       auto shown =
                           run_engine_screen(screen, st.engines, st.profiles, true, st.next_key, st.snapshot);
                       if (!shown) {
                         log->warn("écran des moteurs : {}", shown.error().detail);
                       } else if (!*shown) {
                         quit = true;
                       }
                       refresh();
                     },
                     [](const auto&) {}},  // pistes, chapitres, export : rien sans vidéo
          input);
    }
    actions.clear();
    if (dirty && host_now_us() - saved_at >= 1'000'000) save();
    st.snapshot.maybe(h.screen, opened);
    view.dialog_open = pick->open();
    const ImDrawData* draw = h.ui.home(host_now_us(), view, actions);
    if (std::exchange(view.open_file, false)) FilePick::show(pick, h.screen.window());
    if (std::exchange(view.show_settings, false)) h.ui.open_settings();
    if (auto shown = h.screen.show(ShowRequest{.ui = draw}); !shown)
      return std::unexpected(std::move(shown.error()));
    (void)SDL_WaitEventTimeout(nullptr, kHomeFrameMs);
  }
  if (dirty) save();
  if (!chosen) return std::nullopt;
  log->info("accueil : {}", prepare::utf8_path(*chosen));
  h.ui.close_settings();
  return chosen;
}

}  // namespace ajp::app::play
