#include "app/play/home_screen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include "app/play/auto_bench.h"
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/default_player.h"
#include "app/play/display_edid.h"
#include "app/play/drba_model.h"
#include "app/play/engine_install.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/first_run.h"
#include "app/play/home_idle.h"
#include "app/play/host_choices.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/latest_writer.h"
#include "app/play/model_choice.h"
#include "app/play/paths.h"
#include "app/play/scripted_input.h"
#include "app/play/single_instance.h"
#include "app/play/span_auto.h"
#include "app/play/span_int8.h"
#include "app/play/user_keys.h"
#include "audio/devices.h"
#include "core/clock.h"
#include "core/engines.h"
#include "core/keys.h"
#include "core/log.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

// L'accueil suit la souris : ~60 images par seconde.

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

// Les filtres de la boîte « Ouvrir une vidéo » : SDL les lit jusqu'à la réponse, d'où le static. Les
// extensions sont celles de video_extensions() (P3.85).
const std::array<SDL_DialogFileFilter, 2> kVideoFiltersEn{{
    {.name = "Videos", .pattern = video_dialog_pattern()},
    {.name = "All files", .pattern = "*"},
}};
const std::array<SDL_DialogFileFilter, 2> kVideoFiltersFr{{
    {.name = "Vidéos", .pattern = video_dialog_pattern()},  // i18n-ok : le filtre de la boîte suit la langue
    {.name = "Tous les fichiers", .pattern = "*"},          // i18n-ok
}};
const std::array<SDL_DialogFileFilter, 2>& video_filters() {
  return i18n::language() == i18n::Language::French ? kVideoFiltersFr : kVideoFiltersEn;
}

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
                           video_filters().data(), static_cast<int>(video_filters().size()),
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
  const std::vector<std::filesystem::path> places = motion_model_dirs(h.paths, rife);
  const std::optional<DrbaModel> drba_file = find_distil_drba(places);
  const bool drba = drba_file.has_value();
  // Sans moteur chargé, le FP8 se présume sur une RTX (le moteur TensorRT le calcule dès Ada) ; la chaîne
  // retombe sur l'Équilibre au chargement si la carte ne le calcule pas.
  const bool fp8_engine = gpu_facts(h.screen).vendor == GpuVendor::Nvidia && gpu_facts(h.screen).tensor_cores;
  // Le mode Performance de SPAN (noyaux CUTLASS) : une RTX 40 (Ada, 8.9, portable compris : le nom de la
  // carte le dit, « RTX 4050 Laptop GPU » comme « RTX 4070 Ti ») et le module TensorRT ; le moteur le
  // confirme à l'ouverture d'une vidéo.
  const SpanFp8Facts fp8_facts{
      .backend_name = tensorrt_backend_present(h.catalog, h.paths) ? "tensorrt" : "vulkan",
      .gpu_name = gpu_facts(h.screen).name,
      .tensorrt_present = tensorrt_backend_present(h.catalog, h.paths),
      .capability = gpu_facts(h.screen).vendor == GpuVendor::Nvidia &&
                    cuda_arch_from_name(gpu_facts(h.screen).name) == "sm89" &&
                    tensorrt_backend_present(h.catalog, h.paths)};
  const std::int64_t period = display_period_us(h.screen.window());
  const std::optional<DisplayFacts> display = window_display(h.screen.window());
  SettingsChoices choices{
      .models = model ? model->available : std::vector<std::string>{},
      .default_model = model ? model->default_name : std::string(),
      .source_width = kHomeSource.width,
      .source_height = kHomeSource.height,
      .source_fps = kHomeSource.fps,
      .source_rate = media::Rational{.num = 24000, .den = 1001},
      .display_hz = period > 0 ? 1e6 / static_cast<double>(period) : 0.0,
      .rife_available = rife.has_value() || drba || offered(MotionEngine::Apple),
      .span_tiers = model
                        ? span_tiers(model->path, paths::derived_models(h.paths), fp8_engine,
                                     fp8_facts.capability)  // les noyaux CUTLASS se présument sur une RTX 40
                        : TierSupport{},
      .rife_tiers = rife ? sibling_tiers(rife->path, fp8_engine) : TierSupport{},
      .drba_tiers = drba_file ? sibling_tiers(drba_file->path, fp8_engine) : TierSupport{},
      .distil_drba_available = drba,
      .rife_loaded = false,
      .rife_models = rife ? rife->available : std::vector<std::string>{},
      .default_rife_model = rife ? rife->default_name : std::string(),
      .rife_folder = places.empty() ? std::string() : prepare::utf8_path(places.front()),
      .hdr_output = !h.options.sdr && system_hdr(content_peak_nits(h.screen.window())),
      .auto_peak = auto_peak_nits(content_peak_nits(h.screen.window())),
      .engines = engine_badges(h.paths.engines, {}, WorkSize{}, std::nullopt, kernels::RifeGeometry{}),
      .chosen_engines = chosen_line(st.engines),
      .engine_in_use = st.engines.preferred[usage(core::Usage::Upscale)],
      .display_text = display ? describe(*display) : std::string("EDID illisible : OLED inconnu"),
      .source_file = {},
      .display_height = display_height(h.screen.window()),
      .span_prepare = {.note = i18n::tr("Open a video: the player prepares Performance mode on its frames.")},
      .fp8_reason = span_fp8_unavailable_reason(fp8_facts)};
  // Vidéo > Post-traitement sans vidéo ouverte : la carte, sa tranche, les sorties son, comme en lecture.
  fill_host_choices(choices, h.screen, h.paths, h.catalog, h.settings_path.parent_path());
  return choices;
}

// Les réglages que la lecture prendra : play.toml, puis les options de la ligne de commande.
PlaySettings home_settings(const HomeScreen& h, const PlaySettings& base) {
  return settings_for(base, h.options);
}

// Quelque chose avance sans événement et se regarde : l'accueil garde son rythme plein (home_idle.h).
bool home_busy(const HomeScreen& h, const HomeView& view) {
  const bool scripted = !h.options.keys.empty() || h.options.snapshot.has_value();
  return scripted || view.dialog_open || h.ui.controls().toast(host_now_us()).has_value();
}

// L'écran des moteurs depuis l'accueil, ou l'assistant en entier pour « Relancer l'assistant » (A24-24 :
// avant, l'accueil n'ouvrait que les moteurs). L'assistant écrit play.toml : st.saved le relit, sans quoi
// l'accueil réécrirait par-dessus ses réglages d'avant. Rend faux quand la fenêtre se ferme.
bool home_engines(const HomeScreen& h, HomeState& st, bool wizard, LatestWriter& writer) {
  writer.flush();  // l'écran qui s'ouvre lit et écrit play.toml : la dernière écriture de l'accueil d'abord
  const EngineScreen screen{.screen = h.screen,
                            .ui = h.ui,
                            .options = h.options,
                            .paths = h.paths,
                            .catalog = h.catalog,
                            .engines_path = paths::engine_settings(h.settings_path.parent_path()),
                            .restart_wizard = wizard};
  auto shown = run_engine_screen(screen, st.engines, true, st.next_key, st.snapshot);
  if (!shown) log_channel("app")->warn("écran des moteurs : {}", shown.error().detail);
  if (wizard) {
    if (auto reread = load_settings(h.settings_path)) st.saved = std::move(*reread);
  }
  return !shown || *shown;
}

}  // namespace

std::optional<std::filesystem::path> dropped_file(const SDL_Event& event) {
  auto drop = dropped(event);
  if (!drop || drop->kind != DropKind::Video) return std::nullopt;
  return std::move(drop->path);
}

void raise_for_forwarded(const SDL_Event& event) {
  if (event.type != SDL_EVENT_DROP_FILE || event.drop.source == nullptr) return;
  if (std::string_view(event.drop.source) != kForwardedDrop) return;
  int count = 0;
  SDL_Window** windows = SDL_GetWindows(&count);
  if (windows == nullptr) return;
  if (count > 0) {
    if ((SDL_GetWindowFlags(windows[0]) & SDL_WINDOW_MINIMIZED) != 0) SDL_RestoreWindow(windows[0]);
    SDL_RaiseWindow(windows[0]);
  }
  SDL_free(static_cast<void*>(windows));
}

DropKind drop_kind(const std::filesystem::path& path) {
  std::string extension = prepare::utf8_path(path.extension());
  std::ranges::transform(extension, extension.begin(),
                         [](char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c); });
  if (std::ranges::find(video_extensions(), extension) != video_extensions().end()) return DropKind::Video;
  if (extension == ".ass" || extension == ".ssa" || extension == ".srt") return DropKind::Subtitles;
  return DropKind::Other;
}

// A24-15 : les trois gestes sont séparés et nommés (la fenêtre qui passe devant, la lecture du disque, le
// tri par extension). Le disque reste lu une fois par fichier glissé : c'est ce qui écarte un dossier.
std::optional<Dropped> dropped(const SDL_Event& event) {
  if (event.type != SDL_EVENT_DROP_FILE || event.drop.data == nullptr) return std::nullopt;
  raise_for_forwarded(event);
  std::filesystem::path path = prepare::path_from_utf8(event.drop.data);
  // Un dossier ou un fichier qui n'est pas une vidéo (image, texte, PDF) ne ferme pas la lecture en cours :
  // il ne se lirait pas, et la vidéo serait perdue pour rien.
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error) return std::nullopt;
  const DropKind kind = drop_kind(path);
  return Dropped{.path = std::move(path), .kind = kind};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<std::optional<std::filesystem::path>, Refusal> run_home(const HomeScreen& h, HomeState& st,
                                                                      const std::string& problem) {
  const auto log = log_channel("app");
  log->info("accueil : Ouvrir une vidéo, ou un fichier glissé sur la fenêtre");
  SDL_SetWindowTitle(h.screen.window(), "ajp");
  PlaySettings from_file = st.saved;
  PlaySettings running = home_settings(h, st.saved);
  PlaySettings started = running;
  const auto refresh = [&] { h.ui.set_settings(running, home_choices(h, st, running)); };
  refresh();
  h.ui.open_settings();
  const auto pick = std::make_shared<FilePick>();
  HomeView view{.problem = problem};
  bool dirty = false;
  std::int64_t saved_at = host_now_us();
  // A24-16 : play.toml s'écrit hors du thread de l'interface, comme en lecture (une écriture lente, disque
  // qui dort ou partage réseau, ne fige plus l'accueil). La sortie de la fonction attend la dernière.
  LatestWriter writer;
  const auto save = [&] {
    st.saved = settings_to_save(running, from_file, started, h.options);
    writer.post("play.toml",
                [path = h.settings_path, saved = st.saved] { return save_settings(path, saved); });
    dirty = false;
    saved_at = host_now_us();
  };
  const auto changed = [&](const PlaySettings& wanted) {
    running = wanted;
    running.volume = h.ui.controls().volume();
    h.ui.set_running(running);
    dirty = true;
  };
  const core::KeyMap& keys = configured_keys();
  const std::int64_t opened = host_now_us();
  std::vector<Input> actions;
  std::optional<std::filesystem::path> chosen;
  bool quit = false;
  std::int64_t last_event = opened;  // A24-17 : au repos, l'accueil ne redessine plus 60 fois par seconde
  while (!chosen && !quit) {
    for (; st.next_key < h.options.keys.size() && host_now_us() - opened >= h.options.keys[st.next_key].first;
         ++st.next_key)
      press(h.options.keys[st.next_key].second, h.screen.window());
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      last_event = host_now_us();
      if (auto dropped = dropped_file(event)) {
        if (!chosen) chosen = std::move(*dropped);
        continue;
      }
      if (h.ui.handle(event, host_now_us(), actions)) continue;
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    if (!chosen) chosen = pick->take();
    for (const Input& input : actions) {
      std::visit(Overloaded{[&](const Command& command) {
                              if (std::holds_alternative<cmd::Quit>(command)) quit = true;
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
                            [&](const ApplySettings& apply) { changed(apply.settings); },
                            [&](const UseEngine& use) {
                              st.engines = using_engine(std::move(st.engines), use.id);
                              if (auto written = core::save_settings(
                                      paths::engine_settings(h.settings_path.parent_path()), st.engines);
                                  !written)
                                log->warn("{}", written.error().detail);
                              log->info("moteur des réseaux : {}", engine_label(use.id));
                              refresh();
                            },
                            [&](const OpenEngineScreen& open) {
                              if (open.wizard) save();  // l'assistant part des réglages à jour
                              quit = !home_engines(h, st, open.wizard, writer);
                              if (open.wizard) {  // et l'accueil repart de ce qu'il a écrit
                                from_file = st.saved;
                                started = running = home_settings(h, st.saved);
                              }
                              refresh();
                            },
                            [&](const PlayFile& play) {
                              if (play.file.empty() || chosen) return;
                              chosen = play.file;
                              st.start = play;
                            },
                            [](const auto&) {}},  // pistes, chapitres, export : rien sans vidéo
                 input);
    }
    actions.clear();
    if (dirty && host_now_us() - saved_at >= kSettingsSaveEveryUs) save();
    st.snapshot.maybe(h.screen, opened);
    view.dialog_open = pick->open();
    const ImDrawData* draw = h.ui.home(host_now_us(), view, actions);
    if (std::exchange(view.open_file, false)) FilePick::show(pick, h.screen.window());
    if (std::exchange(view.show_settings, false)) h.ui.open_settings();
    if (auto shown = h.screen.show(ShowRequest{.ui = draw}); !shown)
      return std::unexpected(std::move(shown.error()));
    (void)SDL_WaitEventTimeout(nullptr, home_wait_ms(host_now_us(), last_event, home_busy(h, view)));
  }
  if (dirty) save();
  if (!chosen) return std::nullopt;
  log->info("accueil : {}", prepare::utf8_path(chosen->filename()));
  if (!st.start) h.ui.close_settings();  // PlayFile : le panneau reste ouvert
  return chosen;
}

}  // namespace ajp::app::play
