#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include "app/play/backend_library.h"
#include "app/play/build_progress.h"
#include "app/play/display_edid.h"
#include "app/play/engine_build.h"
#include "app/play/first_run.h"
#include "app/play/input.h"
#include "app/play/motion_model.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/profiles.h"
#include "app/play/session.h"
#include "app/play/ui.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "kernels/spec/rife.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kRingSlots = 4;
// Hauteur où RIFE estime le mouvement quand la chaîne travaille plus haut (sortie 4K).
constexpr int kRifeMotionHeight = 720;
static_assert(kRingSlots <= kMaxKeptSubtitles, "les sous-titres gardés doivent couvrir tout le ring");

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

gpu::BackendLog backend_log() {
  return [channel = log_channel("backend")](gpu::BackendLogLevel level, std::string_view message) {
    switch (level) {
      case gpu::BackendLogLevel::Debug:
        channel->debug("{}", message);
        break;
      case gpu::BackendLogLevel::Info:
        channel->info("{}", message);
        break;
      case gpu::BackendLogLevel::Warn:
        channel->warn("{}", message);
        break;
      case gpu::BackendLogLevel::Error:
        channel->error("{}", message);
        break;
    }
  };
}

// La piste vidéo que la source choisira, lue d'avance : sa taille fixe le moteur et le ring.
std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video)
    return refuse(RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", file.string()));
  return *demuxer->track(*choice.video);
}

// Un moteur à construire, tel que la préparation le montre.
struct EngineLabel {
  std::string name;   // « SPAN », « RIFE »
  std::string model;  // nom du modèle
  int width = 0;      // entrée du moteur
  int height = 0;
  double expected_s = kDefaultBuildSeconds;
};

// Une construction plus courte ne montre rien : un moteur du cache revient tout de suite.
constexpr std::int64_t kQuietBuildUs = 250'000;
// Pendant une construction, la superposition est redessinée 10 fois par seconde au plus : le GPU
// sert à mesurer les tactiques.
constexpr std::uint32_t kBuildFrameMs = 100;
// Seuils du vrai noir recalés au plus 5 fois par seconde pendant qu'on règle.
constexpr std::int64_t kBlackRefreshUs = 200'000;
// L'écran de bienvenue suit la souris : ~60 images par seconde.
constexpr std::uint32_t kWelcomeFrameMs = 16;

// Instantané demandé (--snapshot) : pris pendant la première construction de moteur qui dure jusqu'à
// at_us, sinon pendant la lecture.
struct SnapshotPlan {
  std::optional<std::filesystem::path> path;
  std::int64_t at_us = 0;
  int frames = 1;
  // Demande l'instantané à l'écran si since_us remonte à at_us ; true s'il vient d'être demandé
  // (une présentation doit suivre, même en pause).
  bool maybe(PlaceboScreen& screen, std::int64_t since_us) {
    if (!path || host_now_us() - since_us < at_us) return false;
    screen.snapshot_next(*path, frames);
    path.reset();
    return true;
  }
};

// Tours d'attente de main pendant la construction d'un moteur : la fenêtre reste vivante et la
// superposition montre la préparation. Rien si l'utilisateur ferme la fenêtre ou quitte (q) ; Tab
// montre les détails (details, repris ensuite par les statistiques de la lecture).
std::expected<std::optional<gpu::Artifact>, Refusal> wait_engine(EngineBuild& build, PlaceboScreen& screen,
                                                                 PlayerUi& ui, const EngineLabel& engine,
                                                                 bool& details, SnapshotPlan& snapshot) {
  const core::KeyMap keys = core::KeyMap::defaults();
  const std::int64_t started = host_now_us();
  BuildProgress progress(engine.expected_s, started);
  std::vector<Input> actions;
  while (true) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (ui.handle(event, host_now_us(), actions)) continue;
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    for (const Input& input : actions) {
      if (const auto* command = std::get_if<Command>(&input)) {
        if (std::holds_alternative<cmd::Quit>(*command)) return std::nullopt;
        if (std::holds_alternative<cmd::ToggleOsd>(*command)) details = !details;
      } else if (std::holds_alternative<ToggleFullscreen>(input)) {
        screen.set_fullscreen(!screen.fullscreen());
      } else if (std::holds_alternative<LeaveFullscreen>(input)) {
        screen.set_fullscreen(false);
      }
    }
    actions.clear();
    if (auto result = build.result()) {
      if (!*result) return std::unexpected(std::move(result->error()));
      log_channel("app")->info("moteur {} prêt en {:.1f} s (construction prévue {:.0f} s)", engine.name,
                               progress.elapsed_seconds(host_now_us()), engine.expected_s);
      return std::optional(std::move(**result));
    }
    const EngineBuild::Status status = build.status();
    progress.marker(status.fraction);
    const std::int64_t now = host_now_us();
    if (now - started >= kQuietBuildUs) {
      snapshot.maybe(screen, started);
      const auto seconds = [](double s) { return format_clock(static_cast<std::int64_t>(s * 1e6), 0); };
      const PrepareView view{
          .label = std::format("Préparation du moteur {}", engine.name),
          .fraction = progress.fraction(now),
          .details = {status.stage,
                      std::format("écoulé {} · prévu ~{}", seconds(progress.elapsed_seconds(now)),
                                  seconds(progress.expected_seconds())),
                      std::format("{} · entrée {}x{} · TensorRT", engine.model, engine.width, engine.height)},
          .show_details = details};
      auto shown = screen.show(ShowRequest{.ui = ui.preparation(now, view)});
      if (!shown) return std::unexpected(std::move(shown.error()));
    }
    SDL_Delay(kBuildFrameMs);
  }
}

// Une image de la superposition avant une étape bloquante (chargement d'un moteur, RTX HDR) : la
// fenêtre dit ce qu'elle attend au lieu de rester noire.
void show_loading(PlaceboScreen& screen, PlayerUi& ui, std::string label) {
  const PrepareView view{.label = std::move(label)};
  if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
    log_channel("app")->warn("préparation : {}", shown.error().detail);
}

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

// « mouse=XxY » ou « click=XxY » de --keys : la souris passe (et clique) à ce point de la
// fenêtre (fractions de sa taille).
std::optional<std::pair<float, float>> mouse_point(std::string_view text) {
  if (!text.starts_with("mouse=") && !text.starts_with("click=")) return std::nullopt;
  text.remove_prefix(6);
  const std::size_t cross = text.find('x');
  if (cross == std::string_view::npos) return std::nullopt;
  float x = 0.0F;
  float y = 0.0F;
  const std::string_view xs = text.substr(0, cross);
  const std::string_view ys = text.substr(cross + 1);
  if (std::from_chars(xs.data(), xs.data() + xs.size(), x).ec != std::errc{} ||
      std::from_chars(ys.data(), ys.data() + ys.size(), y).ec != std::errc{})
    return std::nullopt;
  return std::pair{x, y};
}

// Touche pressée d'office (--keys) : un événement SDL de plus dans la file, comme le clavier.
// « mouse=XxY » déplace la souris (la barre de lecture paraît), « click=XxY » y clique.
void press(const std::string& name, SDL_Window* window) {
  if (const auto point = mouse_point(name)) {
    int w = 0;
    int h = 0;
    SDL_GetWindowSize(window, &w, &h);
    SDL_Event motion{};
    motion.type = SDL_EVENT_MOUSE_MOTION;
    motion.motion.windowID = SDL_GetWindowID(window);
    motion.motion.x = point->first * static_cast<float>(w);
    motion.motion.y = point->second * static_cast<float>(h);
    log_channel("app")->info("--keys : souris en {:.0f}, {:.0f}", motion.motion.x, motion.motion.y);
    SDL_PushEvent(&motion);
    if (name.starts_with("click=")) {
      for (const bool down : {true, false}) {
        SDL_Event button{};
        button.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
        button.button.windowID = motion.motion.windowID;
        button.button.button = SDL_BUTTON_LEFT;
        button.button.down = down;
        button.button.clicks = 1;
        button.button.x = motion.motion.x;
        button.button.y = motion.motion.y;
        SDL_PushEvent(&button);
      }
    }
    return;
  }
  SDL_Event event{};
  event.type = SDL_EVENT_KEY_DOWN;
  event.key.windowID = SDL_GetWindowID(window);
  std::string_view key = name;
  if (key.starts_with("ctrl+")) {  // « ctrl+2 » : Ctrl tenu
    event.key.mod = SDL_KMOD_LCTRL;
    key.remove_prefix(5);
  }
  event.key.key = SDL_GetKeyFromName(std::string(key).c_str());
  event.key.down = true;
  if (event.key.key == SDLK_UNKNOWN) {
    log_channel("app")->warn("--keys : touche inconnue « {} »", name);
    return;
  }
  log_channel("app")->info("--keys : {}", name);
  SDL_PushEvent(&event);
}

// Période de rafraîchissement de l'écran de la fenêtre, d'après SDL ; 0 si inconnue (la session
// la mesure alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window) {
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr || mode->refresh_rate <= 0.0F) return 0;
  return static_cast<std::int64_t>(1e6 / static_cast<double>(mode->refresh_rate));
}

// Crête vue par le contenu sur l'écran de la fenêtre : la marge HDR de SDL3 (relative au blanc de
// référence du compositeur) × 203 nits ; rien si l'écran n'est pas en HDR (display.md).
std::optional<double> content_peak_nits(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  if (!SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false)) return std::nullopt;
  const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F);
  if (!(headroom > 1.0F)) return std::nullopt;
  return static_cast<double>(headroom) * 203.0;
}

// Couleur de la sortie de RTX HDR : PQ BT.2020 au pic donné à TrueHDR.
ColorInfo pq_color(int peak_nits) {
  return ColorInfo{.matrix = ColorMatrix::Rgb,
                   .range = ColorRange::Full,
                   .transfer = Transfer::Pq,
                   .primaries = Primaries::Bt2020,
                   .hdr = HdrMetadata{.max_luminance = static_cast<float>(peak_nits)}};
}

vendor_hdr::TrueHdrParams true_hdr_params(const PlaySettings& s, int auto_peak) {
  return vendor_hdr::TrueHdrParams{.contrast = s.true_hdr_contrast,
                                   .saturation = s.true_hdr_saturation,
                                   .middle_gray = s.true_hdr_gray,
                                   .peak_nits = s.true_hdr_peak > 0 ? s.true_hdr_peak : auto_peak,
                                   .debanding = s.true_hdr_debanding};
}

// RTX HDR : le SDK Maxine importé une fois dans le magasin du lecteur (environ 135 Mo), revérifié
// à chaque lancement, puis l'effet ouvert pour la sortie de SPAN ; l'écran reçoit alors du PQ
// BT.2020 au pic donné à TrueHDR, que rend la fonction.
std::expected<int, Refusal> open_rtx_hdr(const std::filesystem::path& sdk_source, const PlayerPaths& paths,
                                         const PlaySettings& settings, int auto_peak, PlaceboScreen& screen,
                                         gpu::KernelOps& kernels, int width, int height) {
  const auto log = log_channel("app");
  vendor_hdr::SdkStore store(paths.engines.parent_path() / "sdk", vendor_hdr::SdkKind::MaxineVfx);
  vendor_hdr::SdkStatus sdk = store.status();
  if (!sdk.installed) {
    log->info("RTX HDR : import du SDK Maxine depuis {} (environ 135 Mo, une seule fois)",
              sdk_source.string());
    auto imported = store.import(sdk_source);
    if (!imported) return std::unexpected(std::move(imported.error()));
    sdk = *imported;
  }
  if (!sdk.installed) return refuse(RefusalCode::HdrUnavailable, std::format("RTX HDR : {}", sdk.reason));
  auto params = vendor_hdr::validate(true_hdr_params(settings, auto_peak));
  if (!params) return std::unexpected(std::move(params.error()));
  if (auto ok = kernels.open_true_hdr(
          *sdk.installed,
          vendor_hdr::TrueHdrConfig{
              .width = width, .height = height, .source = ColorInfo{}, .params = *params});
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  screen.set_source_color(pq_color(params->peak_nits));
  log->info("RTX HDR : Maxine {} · pic {} nits · contraste {} · saturation {} · gris moyen {} · débandage {}",
            sdk.installed->version().text(), params->peak_nits, params->contrast, params->saturation,
            params->middle_gray, params->debanding ? "oui" : "non");
  return params->peak_nits;
}

std::string volume_text(const Controls& controls) {
  if (controls.muted()) return "Son coupé";
  return std::format("Volume {} %", std::lround(controls.volume() * 100.0F));
}

std::string label_of(const std::vector<TrackEntry>& entries, int track) {
  const auto found = std::ranges::find(entries, track, &TrackEntry::index);
  return found != entries.end() ? found->label : std::format("piste {}", track);
}

// Le grade HDR de la chaîne : route TrueHDR si RTX HDR est ouvert, native si le HDR natif est choisi
// (le SDR converti en PQ par libplacebo, aspect TrueHDR émulé avec les réglages de RTX HDR), aucun
// en SDR. Pic : celui de RTX HDR, ou celui de l'écran (1000 nits s'il ne le dit pas).
// Le vrai noir OLED est voulu : forcé, ou « auto » sur un écran détecté OLED.
bool black_wanted(const PlaySettings& s, const std::optional<DisplayFacts>& display) {
  switch (s.oled_black) {
    case OledBlack::On:
      return true;
    case OledBlack::Off:
      return false;
    case OledBlack::Auto:
      return display && display->oled;
  }
  return false;
}

std::optional<render::GradeParams> grade_of(const PlaySettings& s, bool true_hdr_open, int peak_nits,
                                            int auto_peak, const std::optional<DisplayFacts>& display = {},
                                            const std::optional<render::BlackThresholds>& black = {}) {
  const int peak = true_hdr_open && peak_nits > 0 ? peak_nits
                   : s.true_hdr_peak > 0          ? s.true_hdr_peak
                   : auto_peak > 0                ? auto_peak
                                                  : 1000;
  render::GradeSettings g = s.grade_values;
  g.contrast = s.true_hdr_contrast;
  g.saturation = s.true_hdr_saturation;
  g.middle_gray = s.true_hdr_gray;
  g.oled = black_wanted(s, display);
  if (g.oled && black) {
    g.black_lo = black->lo;
    g.black_hi = black->hi;
  }
  const core::HdrMode route = true_hdr_open  ? core::HdrMode::TrueHdr
                              : s.hdr_native ? core::HdrMode::Native
                                             : core::HdrMode::Off;
  auto decision = render::decide_grade(g, render::GradeContext{.route = route,
                                                               .source_hdr = false,
                                                               .output_pq = true,
                                                               .enabled = s.grade,
                                                               .peak_nits = static_cast<double>(peak)});
  if (!decision) {
    log_channel("app")->warn("grade : {}", decision.error().detail);
    return std::nullopt;
  }
  if (decision->mode == render::GradeMode::Neutral) return std::nullopt;
  return decision->params;
}

std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, int peak_nits) {
  std::string text = screen.output_description();
  if (peak_nits > 0) {
    text += std::format(" · RTX HDR, pic {} nits", peak_nits);
  } else if (s.hdr_native && screen.hdr_output()) {
    text += " · HDR natif";
  }
  if (s.grade && (peak_nits > 0 || s.hdr_native) && screen.hdr_output())
    text += std::format(" · grade {}", render::preset_label(s.grade_preset));
  return text;
}

double fps_of(const media::Rational& rate) {
  return rate.den > 0 ? static_cast<double>(rate.num) / static_cast<double>(rate.den) : 0.0;
}

// Les profils de la lecture : user.toml, le profil actif, la source pour choisir leur règle.
struct ProfileState {
  std::filesystem::path user_path;
  Profiles data;
  int active = 1;
  core::SourceInfo source;
  std::string default_model;  // le modèle SPAN par défaut, pour un réglage « modèle par défaut »
};

ProfilesView view_of(const ProfileState& state) {
  ProfilesView view{.active = state.active, .start = state.data.file.start_profile};
  for (std::size_t i = 0; i < view.rows.size(); ++i) {
    if (const auto& profile = state.data.file.profiles[i])
      view.rows[i] = ProfileRow{.name = profile->name, .summary = profile_summary(*profile, state.source)};
  }
  return view;
}

// La chaîne en cours devient celle du profil actif, dans user.toml (rien n'y est écrit si elle
// redevient celle de la base).
void remember_chain(ProfileState& state, const PlaySettings& s) {
  const std::array<core::UserEdit, 2> edits{
      core::edit::ProfileRules{.slot = state.active, .rules = rules_of(s, state.default_model)},
      core::edit::ProfileHdr{.slot = state.active, .mode = hdr_of(s)}};
  auto edited = edit_profiles(state.user_path, state.data, edits);
  if (!edited) {
    log_channel("app")->warn("user.toml : {}", edited.error().detail);
    return;
  }
  state.data = std::move(*edited);
}

bool chain_differs(const PlaySettings& a, const PlaySettings& b) {
  return a.upscale != b.upscale || a.model != b.model || a.work_height != b.work_height ||
         a.downscale != b.downscale || a.rife_factor != b.rife_factor || a.true_hdr != b.true_hdr;
}

// Ce qui vit le temps d'une ouverture de la chaîne, et ce que les actions y changent.
struct Chain {
  Session& session;
  PlaceboScreen& screen;
  PlayerUi& ui;
  Pipeline& pipeline;
  LiveSettings& live;
  PlaySettings& running;  // réglages en cours, gardés dans play.toml
  ProfileState& profiles;
  bool rife_loaded = false;
  bool true_hdr_open = false;
  int auto_peak = 0;                   // pic automatique de RTX HDR
  int peak_nits = 0;                   // pic donné à RTX HDR, 0 sans lui
  bool dirty = false;                  // play.toml à réécrire
  std::optional<PlaySettings> reopen;  // demandé par le menu des paramètres
  const std::optional<DisplayFacts>& display;
  // Vrai noir OLED : la rampe passée par TrueHDR (codes PQ, de gpu), les seuils en cours, et le
  // recalage demandé (au plus 5 fois par seconde).
  std::vector<float> truehdr_ramp;
  std::optional<render::BlackThresholds> black;
  bool black_dirty = true;
  std::int64_t black_at_us = 0;
};

// Le grade de la chaîne avec ses seuils du vrai noir.
void set_grade(Chain& to, const PlaySettings& s) {
  to.screen.set_grade(grade_of(s, to.true_hdr_open, to.peak_nits, to.auto_peak, to.display, to.black));
}

// Vrai noir OLED : recale les seuils sur la rampe de 32 marches mesurée à la sortie (route native :
// les marches SDR ; RTX HDR : celles que gpu a passées par TrueHDR), grade compris, puis pose le
// grade avec eux.
void refresh_black(Chain& to) {
  to.black_dirty = false;
  to.black_at_us = host_now_us();
  const PlaySettings& s = to.running;
  const bool routed = to.true_hdr_open || s.hdr_native;
  const bool wanted = black_wanted(s, to.display) && s.grade && routed && to.screen.hdr_output();
  to.live.set_black_wanted(wanted && to.true_hdr_open);
  to.black.reset();
  if (!wanted) {
    set_grade(to, s);
    to.ui.set_black_status(black_wanted(s, to.display)
                               ? "Vrai noir : sans effet ici (grade coupé ou sortie SDR)."
                               : "Vrai noir : coupé.");
    return;
  }
  std::vector<float> levels;
  if (to.true_hdr_open) {
    if (to.truehdr_ramp.empty()) {
      set_grade(to, s);
      to.ui.set_black_status("Vrai noir : rampe TrueHDR en cours de mesure…");
      return;
    }
    levels = to.truehdr_ramp;
  } else {
    const auto sdr = black_ramp_levels();
    levels.assign(sdr.begin(), sdr.end());
  }
  set_grade(to, s);  // sans seuils : la rampe se mesure sans le vrai noir
  auto nits = to.screen.grade_ramp(levels);
  if (!nits) {
    log_channel("app")->warn("vrai noir : {}", nits.error().detail);
    to.ui.set_black_status("Vrai noir : mesure impossible (voir le journal).");
    return;
  }
  to.black = render::black_thresholds(*nits, s.oled_black_steps);
  set_grade(to, s);
  if (!to.black) {
    to.ui.set_black_status("Vrai noir : la rampe ne monte pas, rien n'est changé.");
    return;
  }
  log_channel("app")->info(
      "vrai noir : noir de la source {:.4f} nit, marches 1 et 2 {:.4f} et {:.4f} nit ; 0 sous {:.4f}, pied "
      "jusqu'à {:.4f} nit",
      (*nits)[0], (*nits)[1], (*nits)[2], to.black->lo, to.black->hi);
  to.ui.set_black_status(
      std::format("Vrai noir : 0 sous {:.4f} nit, pied jusqu'à {:.4f} nit (noir de la source "
                  "à {:.4f} nit avant).",
                  to.black->lo, to.black->hi, (*nits)[0]));
}

// Piste audio : le pont change de piste à l'époque du seek qui suit, à la position courante.
void switch_audio(Chain& to, int track, std::int64_t now_us) {
  if (to.pipeline.tracks().audio_selected == track) return;
  const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
  auto switched = to.pipeline.select_audio(track, to.session.epoch() + 1, [&] {
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  });
  if (!switched) {
    log_channel("app")->warn("piste audio {} : {}", track, switched.error().detail);
    to.ui.controls().notify("Piste audio refusée", now_us);
    return;
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(std::format("Audio : {}", label_of(to.pipeline.tracks().audio, track)), now_us);
}

// Sous-titres : subs prend la nouvelle piste, un seek à la position courante relit ses paquets.
void switch_subtitles(Chain& to, std::optional<int> track, std::int64_t now_us) {
  if (to.pipeline.tracks().subtitle_selected == track) return;
  auto switched = to.pipeline.select_subtitles(track);
  if (!switched) {
    log_channel("app")->warn("sous-titres : {}", switched.error().detail);
    to.ui.controls().notify("Piste de sous-titres refusée", now_us);
    return;
  }
  to.session.set_subtitles(track.has_value());
  if (track) {
    const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(
      track ? std::format("Sous-titres : {}", label_of(to.pipeline.tracks().subtitles, *track))
            : std::string("Sous-titres désactivés"),
      now_us);
}

// Réglages en direct du menu des paramètres : RIFE, réduction, RTX HDR, son, statistiques.
void apply_live(Chain& to, const PlaySettings& wanted, std::int64_t now_us) {
  PlaySettings& r = to.running;
  Controls& controls = to.ui.controls();
  if (wanted.rife_factor != r.rife_factor) {
    log_channel("app")->info("RIFE en direct : ×{}", wanted.rife_factor);
    to.live.set_rife_factor(static_cast<std::uint32_t>(std::max(wanted.rife_factor, 0)));
    controls.notify(wanted.rife_factor >= 2 ? std::format("Interpolation ×{}", wanted.rife_factor)
                                            : std::string("Interpolation désactivée"),
                    now_us);
  }
  if (wanted.downscale != r.downscale) {
    log_channel("app")->info("réduction en direct : {}", label(wanted.downscale));
    to.live.set_reduction(reduction_of(wanted.downscale));
    controls.notify(std::format("Réduction : {}", label(wanted.downscale)), now_us);
  }
  const vendor_hdr::TrueHdrParams before = true_hdr_params(r, to.auto_peak);
  const vendor_hdr::TrueHdrParams after = true_hdr_params(wanted, to.auto_peak);
  if (to.true_hdr_open && after != before) {
    log_channel("app")->info(
        "RTX HDR en direct : contraste {} · saturation {} · gris moyen {} · pic {} nits · débandage {}",
        after.contrast, after.saturation, after.middle_gray, after.peak_nits,
        after.debanding ? "oui" : "non");
    to.live.set_true_hdr(after);
    if (after.peak_nits != before.peak_nits) {
      to.screen.set_source_color(pq_color(after.peak_nits));
      to.peak_nits = after.peak_nits;
    }
  }
  if (wanted.grade != r.grade || wanted.grade_preset != r.grade_preset ||
      wanted.grade_values != r.grade_values || wanted.hdr_native != r.hdr_native || after != before ||
      wanted.oled_black != r.oled_black || wanted.oled_black_steps != r.oled_black_steps) {
    set_grade(to, wanted);
    to.black_dirty = true;
    to.session.set_output(output_text(to.screen, wanted, to.peak_nits));
    if (wanted.grade_preset != r.grade_preset || wanted.grade != r.grade) {
      controls.notify(wanted.grade ? std::format("Grade : {}", render::preset_label(wanted.grade_preset))
                                   : std::string("Grade désactivé"),
                      now_us);
    }
  }
  if (wanted.audio_delay_ms != r.audio_delay_ms)
    to.session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{wanted.audio_delay_ms} * 1000});
  if (wanted.stats != r.stats) to.session.set_stats(wanted.stats);
  if (wanted.subtitle_scale != r.subtitle_scale) {
    to.session.set_subtitle_scale(wanted.subtitle_scale);
    controls.notify(std::format("Sous-titres : {} %", std::lround(wanted.subtitle_scale * 100.0F)), now_us);
  }
  r = wanted;
  r.volume = controls.volume();
  to.ui.set_running(r);
  to.dirty = true;
}

// Profil slot : sa chaîne pour cette source, en direct si elle le permet, sinon par réouverture.
void switch_profile(Chain& to, int slot, std::int64_t now_us) {
  const auto& profile = to.profiles.data.file.profiles[static_cast<std::size_t>(std::clamp(slot, 1, 9) - 1)];
  if (!profile) {
    to.ui.controls().notify(std::format("Profil {} : vide", slot), now_us);
    return;
  }
  std::vector<std::string> notes;
  const PlaySettings wanted = apply_profile(*profile, to.profiles.source, to.running, notes);
  for (const std::string& line : notes) log_channel("app")->warn("profil {} : {}", slot, line);
  log_channel("app")->info("profil {} « {} »", slot, profile->name);
  to.profiles.active = slot;
  to.ui.set_profiles(view_of(to.profiles));
  to.ui.controls().notify(std::format("Profil {} · {}", slot, profile->name), now_us);
  const Change change = change_between(to.running, wanted, to.rife_loaded);
  if (change.cost == ChangeCost::Reopen) {
    to.reopen = wanted;
  } else if (change.cost == ChangeCost::Live) {
    apply_live(to, wanted, now_us);
  }
}

// Une action du clavier ou de l'interface. La barre paraît aussi après un seek au clavier.
void apply(const Input& input, Chain& to, std::int64_t now_us) {
  Session& session = to.session;
  PlaceboScreen& screen = to.screen;
  Controls& controls = to.ui.controls();
  const auto volume_changed = [&] {
    to.running.volume = controls.volume();
    to.ui.set_running(to.running);
    to.dirty = true;
    controls.notify(volume_text(controls), now_us);
  };
  std::visit(
      Overloaded{
          [&](const Command& command) {
            if (const auto* select = std::get_if<cmd::SelectProfile>(&command)) {  // Ctrl+1 à 9, menu
              switch_profile(to, select->slot, now_us);
              return;
            }
            if (std::holds_alternative<cmd::TogglePause>(command)) {
              log_channel("app")->info("action : {}", session.paused() ? "lecture" : "pause");
            } else if (const auto* seek = std::get_if<cmd::Seek>(&command)) {
              log_channel("app")->info("action : seek {}{}", format_position(seek->target_us),
                                       seek->exact ? "" : " (point clé)");
            }
            session.command(command);
            if (std::holds_alternative<cmd::SeekRelative>(command)) controls.activity(now_us);
            if (std::holds_alternative<cmd::SetVolume>(command)) {  // curseur de la barre
              to.running.volume = controls.volume();
              to.ui.set_running(to.running);
              to.dirty = true;
            }
            if (std::holds_alternative<cmd::ToggleOsd>(command)) {
              to.running.stats.all = session.stats_settings().all;
              to.ui.set_running(to.running);
              to.dirty = true;
              controls.notify(to.running.stats.all ? "Statistiques affichées" : "Statistiques masquées",
                              now_us);
            }
          },
          [&](const ToggleFullscreen&) { session.toggle_fullscreen(); },
          [&](const LeaveFullscreen&) {
            if (to.ui.settings_open()) {
              to.ui.close_settings();
            } else if (screen.fullscreen()) {
              session.toggle_fullscreen();
            }
          },
          [&](const ToggleMute&) {
            session.command(Command{controls.toggle_mute()});
            volume_changed();
          },
          [&](const VolumeStep& step) {
            session.command(Command{controls.step_volume(step.delta)});
            volume_changed();
          },
          [&](const SelectAudio& select) { switch_audio(to, select.track, now_us); },
          [&](const SelectSubtitles& select) { switch_subtitles(to, select.track, now_us); },
          [&](const CycleAudio&) {
            if (auto next = next_audio(to.pipeline.tracks())) switch_audio(to, *next, now_us);
          },
          [&](const CycleSubtitles&) { switch_subtitles(to, next_subtitle(to.pipeline.tracks()), now_us); },
          [&](const ToggleSettings&) { to.ui.toggle_settings(); },
          [&](const ApplySettings& apply) {
            // Changer la chaîne au menu modifie le profil actif.
            if (chain_differs(to.running, apply.settings)) {
              remember_chain(to.profiles, apply.settings);
              to.ui.set_profiles(view_of(to.profiles));
            }
            if (apply.reopen) {
              log_channel("app")->info("paramètres : réouverture de la chaîne");
              to.reopen = apply.settings;
            } else {
              apply_live(to, apply.settings, now_us);
            }
          },
          [&](const EditProfile& edit) {
            const std::array<core::UserEdit, 1> edits{edit.edit};
            auto edited = edit_profiles(to.profiles.user_path, to.profiles.data, edits);
            if (!edited) {
              log_channel("app")->warn("user.toml : {}", edited.error().detail);
              to.ui.controls().notify("Profil : modification refusée", now_us);
              return;
            }
            to.profiles.data = std::move(*edited);
            to.ui.set_profiles(view_of(to.profiles));
            // Revenir à la base change la chaîne du profil actif : elle s'applique.
            if (std::holds_alternative<core::edit::ResetProfile>(edit.edit))
              switch_profile(to, to.profiles.active, now_us);
          }},
      input);
}

void write_record(std::ofstream& stats, const PresentRecord& record) {
  stats << record.host_us << '\t' << record.pts_us << '\t' << (record.fresh ? 1 : 0) << '\t'
        << record.render_us << '\t' << record.interval_us << '\t' << record.dropped << '\t' << record.why
        << '\n';
}

void log_summary(const Session& session) {
  const FrameStats& stats = session.stats();
  log_channel("app")->info(
      "fin : {} images neuves, {} présentations, {} sautées, {} jetées au seek, {} sans sous-titres en "
      "attente ; "
      "temps par image médian {} (σ {}), p95 {} ; composition médiane {}",
      stats.new_frames, stats.presents, stats.dropped, stats.stale, stats.subtitle_gaps,
      format_ms(stats.frame_interval.median()),
      format_ms(static_cast<std::int64_t>(stats.frame_interval.stddev())),
      format_ms(stats.frame_interval.percentile(95)), format_ms(stats.render.median()));
}

// Ce qui survit à une réouverture de la chaîne : la fenêtre, l'interface, la bibliothèque du
// backend, les touches et l'instantané d'essai, les pistes choisies.
struct Player {
  const PlayOptions& options;
  const PlayerPaths& paths;
  PlaceboScreen& screen;
  PlayerUi& ui;
  std::shared_ptr<BackendLibrary> library;
  media::TrackInfo video;
  std::filesystem::path settings_path;
  PlaySettings from_file;  // play.toml au lancement
  PlaySettings started;    // réglages du début de la lecture (options comprises)
  SnapshotPlan snapshot;
  bool details = false;  // Tab pendant une préparation
  std::size_t next_key = 0;
  std::int64_t began_us = 0;
  std::optional<int> audio_track;
  std::optional<std::optional<int>> subtitle_track;
  ProfileState profiles;
  core::Settings engines;               // settings.toml : les moteurs choisis à l'écran de bienvenue
  std::optional<DisplayFacts> display;  // l'écran selon son EDID (OLED ou non)
};

// Fin d'une ouverture de la chaîne : la lecture est finie (rien), ou le menu demande de la rouvrir
// avec ces réglages à cette position.
struct Reopen {
  PlaySettings settings;
  std::int64_t position_us = 0;
};

// Le modèle SPAN : --model, sinon celui choisi dans le menu (à côté du modèle par défaut), sinon
// celui par défaut. Rend aussi le dossier des modèles et le nom par défaut, pour le menu.
struct ModelChoice {
  std::filesystem::path path;
  std::vector<std::string> available;  // modèles ×2 du dossier
  std::string default_name;
};
std::expected<ModelChoice, Refusal> choose_model(const Player& p, const PlaySettings& settings) {
  PlayOptions defaults = p.options;
  if (!p.options.was_given("--model")) defaults.model.reset();
  auto found = find_model(defaults, default_model_candidates(p.paths.home));
  if (!found) return std::unexpected(std::move(found.error()));
  ModelChoice choice{.path = *found, .default_name = found->filename().string()};
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(found->parent_path(), error)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with("2x_") && name.ends_with(".onnx")) choice.available.push_back(name);
  }
  std::ranges::sort(choice.available);
  if (!p.options.was_given("--model") && !settings.model.empty()) {
    const std::filesystem::path chosen = found->parent_path() / settings.model;
    if (std::filesystem::exists(chosen, error)) {
      choice.path = chosen;
    } else {
      log_channel("app")->warn("modèle {} introuvable : {}", settings.model, found->filename().string());
    }
  }
  return choice;
}

// Les moteurs du cache, pour l'onglet Moteurs du menu.
std::vector<EngineBadge> engine_badges(const std::filesystem::path& cache_dir,
                                       const std::filesystem::path& span, WorkSize span_size,
                                       const std::optional<std::filesystem::path>& rife,
                                       const kernels::RifeGeometry& rife_size) {
  std::vector<EngineBadge> badges;
  for (const PastBuild& past : past_builds(cache_dir)) {
    std::string file = past.onnx.filename().string();
    std::string lower_file = file;
    std::ranges::transform(lower_file, lower_file.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool is_rife = lower_file.find("rife") != std::string::npos;
    const bool loaded = is_rife ? rife && past.onnx.filename() == rife->filename() &&
                                      past.width == rife_size.padded_width &&
                                      past.height == rife_size.padded_height
                                : past.onnx.filename() == span.filename() && past.width == span_size.width &&
                                      past.height == span_size.height;
    badges.push_back(EngineBadge{
        .name = is_rife ? std::format("RIFE · {}", past.onnx.stem().string())
                        : std::format("SPAN · {}", model_label(file)),
        .detail = std::format("entrée {}x{} · construit en {:.0f} s", past.width, past.height, past.seconds),
        .loaded = loaded});
  }
  std::ranges::stable_partition(badges, &EngineBadge::loaded);
  return badges;
}

// Mesures de la chaîne pour l'OSD : compteurs de RIFE, mémoire vidéo.
class ChainStats final : public StatsSource {
 public:
  ChainStats(const LiveSettings& live, gpu::KernelOps& kernels, bool rife)
      : live_(live), kernels_(kernels), rife_(rife) {}
  std::optional<std::string> rife() const override {
    if (!rife_) return std::nullopt;
    const RifeCounts c = live_.counts();
    return std::format("RIFE ×{} : {} paires, {} interpolées, {} dessins tenus, {} coupes, {} en retard",
                       live_.rife_factor(), c.pairs, c.interpolated, c.held, c.cuts, c.late);
  }
  std::optional<std::string> memory() const override {
    const auto memory = kernels_.device_memory();
    if (!memory) return std::nullopt;
    constexpr double kMiB = 1024.0 * 1024.0;
    return std::format("mémoire vidéo {:.0f} / {:.0f} Mio", static_cast<double>(memory->used) / kMiB,
                       static_cast<double>(memory->total) / kMiB);
  }

 private:
  const LiveSettings& live_;
  gpu::KernelOps& kernels_;
  bool rife_;
};

// Construit (ou lit du cache) puis charge un moteur, la préparation à l'écran. Rien si
// l'utilisateur ferme la fenêtre.
std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_engine(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, const std::filesystem::path& onnx,
    const gpu::Shape& shape, EngineLabel label) {
  auto sha256 = prepare::sha256_file(onnx);
  if (!sha256) return std::unexpected(std::move(sha256.error()));
  auto build = EngineBuild::start(
      EngineRequest{.library = p.library,
                    .init = init,
                    .model = {.id = onnx.stem().string(), .onnx_path = onnx, .sha256 = *sha256},
                    .shapes = {.min = shape, .opt = shape, .max = shape}});
  if (!build) return std::unexpected(std::move(build.error()));
  label.expected_s = expected_build_seconds(p.paths.engines, onnx, std::int64_t{label.width} * label.height);
  auto artifact = wait_engine(**build, p.screen, p.ui, label, p.details, p.snapshot);
  if (!artifact) return std::unexpected(std::move(artifact.error()));
  if (!*artifact) return std::nullopt;  // fenêtre fermée pendant la construction
  log_channel("app")->info("moteur {} {}", label.name, (*artifact)->path.string());
  show_loading(p.screen, p.ui, "Chargement des moteurs");
  auto loaded = backend.load(**artifact);
  if (!loaded) return std::unexpected(std::move(loaded.error()));
  return std::optional(std::move(*loaded));
}

// Une ouverture de la chaîne : moteurs, RTX HDR, threads, session et boucle de main, jusqu'à la
// fin du fichier, q, ou un réglage qui demande de la rouvrir.
std::expected<std::optional<Reopen>, Refusal> run_chain(Player& p, PlaySettings settings,
                                                        std::int64_t start_us) {
  const auto log = log_channel("app");
  std::vector<std::string> unavailable;
  settings = within_engines(settings, p.engines, unavailable);
  for (const std::string& line : unavailable) log->warn("{}", line);
  const PlayOptions options = options_for(p.options, settings);
  const media::TrackInfo& video = p.video;
  std::optional<ModelChoice> model;
  if (settings.upscale) {
    auto chosen = choose_model(p, settings);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    model = std::move(*chosen);
  }
  const int factor = model ? kSpanFactor : 1;
  const WorkSize work = work_size(video.width, video.height, settings.work_height);
  const bool reduced = work != WorkSize{video.width, video.height};
  const std::string reduce = reduced ? std::format("{}x{} → {} {}x{}", video.width, video.height,
                                                   label(settings.downscale), work.width, work.height)
                                     : std::format("{}x{}", video.width, video.height);
  // Le modèle RIFE est cherché même sans interpolation : le menu peut l'allumer.
  std::optional<std::filesystem::path> rife_model;
  if (auto found = find_rife_model(options, default_rife_candidates(p.paths.home))) rife_model = *found;
  if (settings.rife_factor >= 2 && !rife_model)
    return std::unexpected(find_rife_model(options, default_rife_candidates(p.paths.home)).error());
  const bool use_rife = settings.rife_factor >= 2 && rife_model;
  const std::string label_text = std::format(
      "{}{}{}",
      use_rife ? std::format("RIFE ×{} · {} · ", settings.rife_factor, rife_model->stem().string()) : "",
      settings.true_hdr ? "RTX HDR · " : "",
      model ? std::format("SPAN ×{} · {} · TensorRT · {} → {}x{}", factor, model->path.stem().string(),
                          reduce, work.width * factor, work.height * factor)
            : std::format("sans agrandissement · {}", reduce));
  log->info("{} : {}", options.file.string(), label_text);

  show_loading(p.screen, p.ui, "Chargement");
  const gpu::BackendInit init{
      .render = &p.screen.device(), .cache_dir = p.paths.engines, .log = backend_log()};
  auto backend = BackendLibrary::create(p.library, init);
  if (!backend) return std::unexpected(std::move(backend.error()));
  log->info("backend {} ({}) depuis {}", (*backend)->capabilities().name,
            (*backend)->capabilities().device_name, p.library->path().string());

  std::unique_ptr<gpu::Engine> engine;
  if (model) {
    auto loaded =
        prepare_engine(p, init, **backend, model->path, gpu::Shape{.dims = {1, 3, work.height, work.width}},
                       EngineLabel{.name = "SPAN",
                                   .model = model->path.stem().string(),
                                   .width = work.width,
                                   .height = work.height});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    engine = std::move(**loaded);
  }
  // Au-delà de 720 lignes de travail (la 4K sans réduction), RIFE estime le mouvement sur une copie
  // réduite à 720p et le warp déforme les images de travail : la variante mouvement seulement.
  const WorkSize rife_size = !p.options.rife_image && work.height > kRifeMotionHeight
                                 ? work_size(video.width, video.height, kRifeMotionHeight)
                                 : work;
  const kernels::RifeGeometry g = kernels::rife_geometry(rife_size.width, rife_size.height);
  const kernels::RifeGeometry image_g = kernels::rife_geometry(work.width, work.height);
  std::unique_ptr<gpu::Engine> rife;
  std::unique_ptr<gpu::Engine> rife_motion;
  // RIFE par son mouvement : une passe au milieu de la paire la sert toute quand le mouvement est
  // faible ; sans variante dérivable, RIFE image comme avant.
  if (use_rife && !p.options.rife_image) {
    auto derived = motion_model(*rife_model, p.paths.engines.parent_path() / "modeles");
    if (!derived) {
      log->warn("RIFE mouvement indisponible ({}) : RIFE image", derived.error().detail);
    } else {
      auto loaded = prepare_engine(
          p, init, **backend, *derived,
          gpu::Shape{.dims = {1, kernels::kRifeInputChannels, g.padded_height, g.padded_width}},
          EngineLabel{.name = "RIFE",
                      .model = derived->stem().string(),
                      .width = g.padded_width,
                      .height = g.padded_height});
      if (!loaded) return std::unexpected(std::move(loaded.error()));
      if (!*loaded) return std::nullopt;
      rife_motion = std::move(**loaded);
    }
  }
  if (use_rife && !rife_motion) {
    auto loaded = prepare_engine(
        p, init, **backend, *rife_model,
        gpu::Shape{.dims = {1, kernels::kRifeInputChannels, image_g.padded_height, image_g.padded_width}},
        EngineLabel{.name = "RIFE",
                    .model = rife_model->stem().string(),
                    .width = image_g.padded_width,
                    .height = image_g.padded_height});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    rife = std::move(**loaded);
  }
  if (rife_motion) {
    log->info("RIFE mouvement : estimé en {}x{}, warp en {}x{}", rife_size.width, rife_size.height,
              work.width, work.height);
  }
  settings.stats.all = p.details;  // Tab pendant la préparation

  const int auto_peak = vendor_hdr::peak_from_display(content_peak_nits(p.screen.window()));
  int peak_nits = 0;
  if (settings.true_hdr) {
    show_loading(p.screen, p.ui, "Préparation de RTX HDR");
    auto opened = open_rtx_hdr(p.options.true_hdr_sdk, p.paths, settings, auto_peak, p.screen,
                               *(*backend)->kernels(), work.width * factor, work.height * factor);
    if (opened) {
      peak_nits = *opened;
    } else {
      // Le plan : TrueHDR qui échoue laisse la place à la route native, et l'écran le dit.
      log->warn("{} ; HDR natif à la place", opened.error().detail);
      p.ui.controls().notify("RTX HDR indisponible : HDR natif", host_now_us());
      settings.true_hdr = false;
      settings.hdr_native = true;
      p.screen.set_source_color(ColorInfo{});
    }
  } else {
    p.screen.set_source_color(ColorInfo{});  // celle de la source, reprise à la première image
  }

  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(static_cast<std::uint32_t>(use_rife ? settings.rife_factor : 0));
  live->set_reduction(reduction_of(settings.downscale));
  PipelineConfig config{
      .file = options.file,
      .decoder = {.attempts = {media::HwApi::Cuda}, .render_device = p.screen.device().uuid},
      .audio_track = p.audio_track,
      .subtitle_track = p.subtitle_track};
  PlaceboScreen& out = p.screen;
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = work_size(width, height, settings.work_height);
        auto slots = out.create_ring(sized.width * factor, sized.height * factor, kRingSlots);
        if (!slots) return std::unexpected(std::move(slots.error()));
        const MasterClock* clock = stage.clock;
        return UpscaleStage::start(
            std::move(stage),
            UpscaleSetup{.backend = backend->get(),
                         .engine = engine.get(),
                         .rife = rife.get(),
                         .rife_motion = rife_motion.get(),
                         .rife_factor = static_cast<std::uint32_t>(std::max(settings.rife_factor, 1)),
                         .true_hdr = settings.true_hdr,
                         .live = live,
                         .clock = clock,
                         .source_width = width,
                         .source_height = height,
                         .work_width = sized.width,
                         .work_height = sized.height,
                         .rife_width = rife_motion ? rife_size.width : 0,
                         .rife_height = rife_motion ? rife_size.height : 0,
                         .slots = std::move(*slots)});
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));

  const bool rife_loaded = rife != nullptr || rife_motion != nullptr;
  const ChainStats chain_stats(*live, *(*backend)->kernels(), rife_loaded);
  SessionConfig session_config = (*pipeline)->session_config();
  session_config.start_us = start_us;
  session_config.stop_after_us = options.stop_after_us;
  session_config.fullscreen_after_us = options.fullscreen_after_us;
  session_config.header = {label_text};
  session_config.tracks = (*pipeline)->describe();
  p.screen.set_grade(grade_of(settings, settings.true_hdr, peak_nits, auto_peak));
  session_config.output = output_text(out, settings, peak_nits);
  session_config.display_period_us = display_period_us(out.window());
  session_config.stats = settings.stats;
  session_config.subtitle_scale = settings.subtitle_scale;
  session_config.extra = &chain_stats;
  log->info("{}", label_text);
  for (const std::string& line : session_config.tracks) log->info("{}", line);
  log->info("{}", session_config.output);
  Session session(std::move(session_config), (*pipeline)->stages(), out, (*pipeline)->clock());
  p.ui.set_tracks((*pipeline)->tracks());
  p.ui.set_settings(
      settings,
      SettingsChoices{
          .models = model ? model->available : std::vector<std::string>{},
          .default_model = model ? model->default_name : std::string(),
          .source_height = video.height,
          .rife_available = rife_model.has_value(),
          .rife_loaded = rife_loaded,
          .true_hdr_available = std::filesystem::exists(p.options.true_hdr_sdk) || settings.true_hdr,
          .hdr_output = p.screen.hdr_output(),
          .auto_peak = auto_peak,
          .engines = engine_badges(p.paths.engines, model ? model->path : std::filesystem::path(), work,
                                   rife_loaded ? rife_model : std::nullopt, rife_motion ? g : image_g),
          .chosen_engines = chosen_line(p.engines),
          .display_text = p.display ? describe(*p.display) : std::string("EDID illisible : OLED inconnu")});
  p.profiles.source =
      core::SourceInfo{.width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};
  if (model) p.profiles.default_model = model->default_name;
  p.ui.set_profiles(view_of(p.profiles));
  Chain chain{.session = session,
              .screen = out,
              .ui = p.ui,
              .pipeline = **pipeline,
              .live = *live,
              .running = settings,
              .profiles = p.profiles,
              .rife_loaded = rife_loaded,
              .true_hdr_open = settings.true_hdr,
              .auto_peak = auto_peak,
              .peak_nits = peak_nits,
              .display = p.display};

  std::ofstream stats;
  if (options.stats) {
    stats.open(*options.stats);
    stats << "host_us\tpts_us\tfresh\trender_us\tinterval_us\tdropped\twhy\n";
  }
  const core::KeyMap keys = core::KeyMap::defaults();
  std::optional<Refusal> failure;
  if (p.began_us == 0) p.began_us = host_now_us();
  std::vector<Input> actions;
  session.begin(host_now_us());
  // Son : le volume de la barre et le délai des réglages valent dès la première image.
  session.command(Command{p.ui.controls().volume_command()});
  if (settings.audio_delay_ms != 0)
    session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{settings.audio_delay_ms} * 1000});
  std::int64_t saved_at = host_now_us();
  while (!session.finished() && !chain.reopen) {
    for (;
         p.next_key < p.options.keys.size() && host_now_us() - p.began_us >= p.options.keys[p.next_key].first;
         ++p.next_key) {
      press(p.options.keys[p.next_key].second, out.window());
    }
    const bool snapshot_now = p.snapshot.maybe(out, p.began_us);
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (p.ui.handle(event, host_now_us(), actions)) continue;  // saisie dans l'interface
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) session.handle(std::move(*message));
    // Vrai noir : la rampe que gpu a passée par TrueHDR, puis les seuils recalés.
    if (auto ramp = chain.live.take_black_ramp()) {
      chain.truehdr_ramp = std::move(*ramp);
      chain.black_dirty = true;
    }
    if (chain.black_dirty && host_now_us() - chain.black_at_us >= kBlackRefreshUs) refresh_black(chain);
    UiFrame ui_frame = p.ui.frame(host_now_us(), session.view(), actions);
    ui_frame.changed |= snapshot_now;
    for (const Input& input : actions) apply(input, chain, host_now_us());
    actions.clear();
    // play.toml réécrit au plus une fois par seconde pendant qu'on règle.
    if (chain.dirty && host_now_us() - saved_at >= 1'000'000) {
      if (auto saved = save_settings(p.settings_path,
                                     settings_to_save(chain.running, p.from_file, p.started, p.options));
          !saved)
        log->warn("{}", saved.error().detail);
      chain.dirty = false;
      saved_at = host_now_us();
    }
    auto record = session.tick(host_now_us(), ui_frame);
    if (!record) {
      failure = std::move(record.error());
      break;
    }
    if (*record && stats.is_open()) write_record(stats, **record);
    // Rien de présenté (rien n'a changé, ou fenêtre cachée) : la FIFO n'impose pas le rythme,
    // la boucle reprend dans 1 ms pour ne manquer ni l'image suivante ni une touche.
    if (!*record) SDL_Delay(1);
  }
  if (chain.dirty) {
    if (auto saved = save_settings(p.settings_path,
                                   settings_to_save(chain.running, p.from_file, p.started, p.options));
        !saved)
      log->warn("{}", saved.error().detail);
  }
  log_summary(session);
  for (const std::string& refusal : session.refusals()) log->warn("{}", refusal);
  std::optional<Reopen> reopen;
  if (chain.reopen && !failure) {
    reopen = Reopen{.settings = *chain.reopen,
                    .position_us = std::max<std::int64_t>(session.view().position_us, 0)};
    reopen->settings.volume = p.ui.controls().volume();
    p.audio_track = (*pipeline)->tracks().audio_selected;
    p.subtitle_track = (*pipeline)->tracks().subtitle_selected;
  }
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, le moteur avant son backend, le
  // backend (et ses imports du ring) avant l'écran qui possède le ring.
  session.end();
  out.flush();
  pipeline->reset();
  rife_motion.reset();
  rife.reset();
  engine.reset();
  backend->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return reopen;
}

// Ce qui est sur ce poste pour l'écran de bienvenue : le backend TensorRT, le SDK Maxine importé
// ou à importer.
EnginesHere engines_here(const PlayOptions& options, const PlayerPaths& paths) {
  EnginesHere here;
  std::error_code error;
  for (const std::filesystem::path& backend : paths.backends)
    here.tensorrt |= std::filesystem::exists(backend, error);
  vendor_hdr::SdkStore store(paths.engines.parent_path() / "sdk", vendor_hdr::SdkKind::MaxineVfx);
  if (store.status().installed) {
    here.maxine = true;
    here.maxine_status = "importé";
  } else if (std::filesystem::exists(options.true_hdr_sdk, error)) {
    here.maxine = true;
    here.maxine_status = std::format("à importer depuis {} (une fois)", options.true_hdr_sdk.string());
  } else {
    here.maxine_status = std::format("SDK absent ({})", options.true_hdr_sdk.string());
  }
  return here;
}

// Le nom de l'écran de la fenêtre selon SDL, pour choisir son EDID.
std::string display_name(SDL_Window* window) {
  const char* name = SDL_GetDisplayName(SDL_GetDisplayForWindow(window));
  return name != nullptr ? std::string(name) : std::string();
}

GpuFacts gpu_facts(const PlaceboScreen& screen) {
  const PlaceboScreen::DeviceFacts facts = screen.device_facts();
  const GpuVendor vendor = vendor_from_pci(facts.pci_vendor);
  return GpuFacts{.name = facts.name,
                  .vendor = vendor,
                  .vram_bytes = facts.vram_bytes,
                  .tensor_cores = vendor == GpuVendor::Nvidia && facts.cooperative_matrix};
}

// L'écran de bienvenue jusqu'à « Commencer » ; rien si l'utilisateur quitte. Les touches d'essai
// (--keys) qui tombent pendant l'écran y sont pressées, comptées depuis son ouverture.
std::expected<std::optional<FirstRunView>, Refusal> welcome(PlaceboScreen& screen, PlayerUi& ui,
                                                            FirstRunView view, const PlayOptions& options,
                                                            std::size_t& next_key, SnapshotPlan& snapshot) {
  const core::KeyMap keys = core::KeyMap::defaults();
  const std::int64_t started = host_now_us();
  std::vector<Input> actions;
  while (!view.done) {
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
        if (std::holds_alternative<cmd::Quit>(*command)) return std::nullopt;
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
    SDL_Delay(kWelcomeFrameMs);
  }
  return view;
}

}  // namespace

std::expected<void, Refusal> play(const PlayOptions& options, const PlayerPaths& paths) {
  const auto log = log_channel("app");
  auto video = probe_video(options.file);
  if (!video) return std::unexpected(std::move(video.error()));
  const std::filesystem::path settings_path = paths.engines.parent_path() / "play.toml";
  auto saved = load_settings(settings_path);
  if (!saved) {
    log->warn("{} ; réglages par défaut", saved.error().detail);
    saved = PlaySettings{};
  }
  // Le profil de démarrage fixe la chaîne ; les options données la remplacent pour cette lecture.
  ProfileState profiles{
      .user_path = settings_path.parent_path() / "user.toml",
      .source = {.width = video->width, .height = video->height, .fps = fps_of(video->frame_rate)}};
  if (auto loaded = load_profiles(profiles.user_path)) {
    profiles.data = std::move(*loaded);
  } else {
    log->warn("{} ; profils de base", loaded.error().detail);
    profiles.data = Profiles{.file = *core::parse_profile(base_profiles_toml())};
  }
  profiles.active = profiles.data.file.start_profile;
  // Premier lancement avec profils : la chaîne de play.toml (celle d'avant les profils) devient
  // celle du profil de démarrage, au lieu d'être remplacée par la base.
  std::error_code missing;
  if (!std::filesystem::exists(profiles.user_path, missing) &&
      std::filesystem::exists(settings_path, missing)) {
    profiles.default_model = std::string(kSpanModelId);
    remember_chain(profiles, *saved);
    log->info("chaîne de play.toml reprise dans le profil {}", profiles.active);
  }
  auto screen =
      PlaceboScreen::open(ScreenOptions{.title = std::format("ajp — {}", options.file.filename().string()),
                                        .width = options.window_width,
                                        .height = options.window_height,
                                        .fullscreen = options.fullscreen});
  if (!screen) return std::unexpected(std::move(screen.error()));
  auto ui = PlayerUi::create((*screen)->window(), options.file.filename().string());
  if (!ui) return std::unexpected(std::move(ui.error()));
  // L'interface rend ses textures à l'écran avant qu'ils ne partent, quelle que soit la sortie.
  class Release {
   public:
    Release(PlayerUi& ui, PlaceboScreen& screen) : ui_(ui), screen_(screen) {}
    Release(const Release&) = delete;
    Release& operator=(const Release&) = delete;
    Release(Release&&) = delete;
    Release& operator=(Release&&) = delete;
    ~Release() { ui_.release(screen_); }

   private:
    PlayerUi& ui_;
    PlaceboScreen& screen_;
  } const release(**ui, **screen);
  SnapshotPlan snapshot{
      .path = options.snapshot, .at_us = options.snapshot_at_us, .frames = options.snapshot_frames};
  std::size_t next_key = 0;

  // Les moteurs : settings.toml, choisis à l'écran de bienvenue au premier lancement.
  const std::filesystem::path engines_path = settings_path.parent_path() / "settings.toml";
  auto engines = core::load_settings(engines_path);
  if (!engines) {
    log->warn("{} ; écran de bienvenue", engines.error().detail);
    engines = core::Settings{};
  }
  if (!engines->first_run_done || options.first_run) {
    const GpuFacts gpu = gpu_facts(**screen);
    ProfilesView choice = view_of(profiles);
    if (!engines->first_run_done) choice.start = start_profile_for(gpu);
    log->info("bienvenue : {}", gpu_line(gpu));
    auto chosen = welcome(**screen, **ui,
                          first_run_view(gpu, engines_here(options, paths), std::move(choice), *engines),
                          options, next_key, snapshot);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    if (!*chosen) return {};
    engines = settings_after(**chosen, *engines);
    if (auto written = core::save_settings(engines_path, *engines); !written)
      log->warn("{}", written.error().detail);
    const std::array<core::UserEdit, 1> start{core::edit::StartProfile{.slot = (*chosen)->profiles.start}};
    if (auto edited = edit_profiles(profiles.user_path, profiles.data, start)) {
      profiles.data = std::move(*edited);
    } else {
      log->warn("user.toml : {}", edited.error().detail);
    }
    profiles.active = profiles.data.file.start_profile;
    log->info("moteurs choisis : {} ; profil de démarrage {}", chosen_line(*engines), profiles.active);
  }

  PlaySettings from_profile = *saved;
  if (const auto& start = profiles.data.file.profiles[static_cast<std::size_t>(profiles.active - 1)]) {
    std::vector<std::string> notes;
    from_profile = apply_profile(*start, profiles.source, *saved, notes);
    for (const std::string& line : notes) log->warn("profil {} : {}", profiles.active, line);
    log->info("profil {} « {} »", profiles.active, start->name);
  }
  PlaySettings settings = settings_for(from_profile, options);
  if (auto valid = validate(settings); !valid) return std::unexpected(std::move(valid.error()));
  static_cast<void>((*ui)->controls().set_volume(settings.volume));
  show_loading(**screen, **ui, "Chargement");
  auto library = BackendLibrary::open(paths.backends);
  if (!library) return std::unexpected(std::move(library.error()));

  Player player{.options = options,
                .paths = paths,
                .screen = **screen,
                .ui = **ui,
                .library = *library,
                .video = *video,
                .settings_path = settings_path,
                .from_file = *saved,
                .started = settings,
                .snapshot = std::move(snapshot),
                .details = settings.stats.all,
                .next_key = next_key,
                .profiles = std::move(profiles),
                .engines = std::move(*engines),
                .display = read_display(display_name((*screen)->window()))};
  if (player.display) {
    log->info("écran {} : {}", player.display->connector, describe(*player.display));
  } else {
    log->info("écran : EDID illisible, OLED inconnu");
  }
  std::int64_t start_us = options.start_us;
  std::expected<void, Refusal> result;
  while (true) {
    auto outcome = run_chain(player, settings, start_us);
    if (!outcome) {
      result = std::unexpected(std::move(outcome.error()));
      break;
    }
    if (!*outcome) break;
    settings = (*outcome)->settings;
    start_us = (*outcome)->position_us;
    player.details = settings.stats.all;
    if (auto written = save_settings(settings_path,
                                     settings_to_save(settings, player.from_file, player.started, options));
        !written)
      log->warn("{}", written.error().detail);
    log->info("réouverture à {}", format_position(start_us));
  }
  return result;
}

}  // namespace ajp::app::play
