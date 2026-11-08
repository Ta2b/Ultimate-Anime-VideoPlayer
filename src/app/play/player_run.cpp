#include "app/play/player_run.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_watch.h"
#include "app/play/backend_library.h"
#include "app/play/bench_events.h"
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/default_player.h"
#include "app/play/display_contract.h"
#include "app/play/display_edid.h"
#include "app/play/drba_model.h"
#include "app/play/engine_build.h"
#include "app/play/engine_install.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/episodes.h"
#include "app/play/export_run.h"
#include "app/play/first_run.h"
#include "app/play/gpu_profile_out.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/home_screen.h"
#include "app/play/host_choices.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/latest_writer.h"
#include "app/play/look_catalog.h"
#include "app/play/loop_wait.h"
#include "app/play/machine_calibration.h"
#include "app/play/model_choice.h"
#include "app/play/paths.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player_engines.h"
#include "app/play/present_stats.h"
#include "app/play/resume.h"
#include "app/play/scripted_input.h"
#include "app/play/session.h"
#include "app/play/span_auto.h"
#include "app/play/span_derive.h"
#include "app/play/span_int8.h"
#include "app/play/ui.h"
#include "app/play/upscale_stage.h"
#include "app/play/user_keys.h"
#include "app/play/visible_error.h"
#include "app/play/vrr_watch.h"
#include "audio/devices.h"
#include "core/auto_memory.h"
#include "core/auto_rate.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "core/vrr_floor.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// Seuils du vrai noir recalés au plus 5 fois par seconde pendant qu'on règle.
constexpr std::int64_t kBlackRefreshUs = 200'000;
// En pause, l'attente d'un événement SDL au plus (P4.7 : 33 ms, l'interface à 30 i/s ; une image réveille).
constexpr std::int32_t kPausedWaitMs = 33;
// L'écran de la fenêtre est relu deux fois par seconde (sa fréquence, son VRR : la cadence s'y règle) ; le
// HDR et le pic du système toutes les 2 s, l'appel coûte plus et l'utilisateur ne bascule pas plus vite.
constexpr std::int64_t kDisplayPollUs = 500'000;
constexpr std::int64_t kHdrPollUs = 2'000'000;
// Une source qui change de taille 4 fois en une minute est refusée : la chaîne se rouvrirait sans fin.
constexpr std::int64_t kResizeWindowUs = 60'000'000;
constexpr std::size_t kResizesRefused = 4;

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<std::optional<Reopen>, Refusal> run_chain(Player& p, PlaySettings settings,
                                                        std::int64_t start_us, bool start_paused) {
  const auto log = log_channel("app");
  std::vector<std::string> unavailable;
  settings = within_engines(settings, p.engines, unavailable);
  for (const std::string& line : unavailable) log->warn("{}", line);
  if (is_gmfss(settings.motion_engine)) {  // débranché de la lecture (settings.h, offered)
    log->info("interpolation {} : débranchée de la lecture, RIFE à sa place", label(settings.motion_engine));
    settings.motion_engine = MotionEngine::Rife;
  }
  const PlayOptions options = options_for(p.options, settings);
  const media::TrackInfo& video = p.video;
  // La pré-analyse des plans part avant la préparation des moteurs : elle a de l'avance quand l'image vient.
  const std::optional<ColorInfo> early_color = media::track_color(video);
  const bool early_hdr =
      early_color && (early_color->transfer == Transfer::Pq || early_color->transfer == Transfer::Hlg);
  if (p.shots != nullptr) open_shots(*p.shots, p.options.file, settings, start_us, early_hdr);
  auto prepared = prepare_chain(p, settings);
  if (!prepared) return std::unexpected(std::move(prepared.error()));
  if (!*prepared) {  // q ou la fenêtre fermée pendant la préparation des moteurs
    p.quit = true;
    return std::nullopt;
  }
  ChainEngines& c = **prepared;
  // La variante de SPAN chargée, pour la mesure d'un essai du bench (--bench-out).
  const std::string span_variant = c.span_int8 ? std::string(variant_text(c.span_int8->kind))
                                   : c.model   ? std::string("FP16")
                                               : std::string();
  const int factor = c.factor;
  const WorkSize work = c.work;
  const bool rife_loaded = c.interpolates();

  settings.stats.all = p.details;  // Tab pendant la préparation

  log->info("{}", describe_hdr(p.screen.window()));
  const std::optional<double> peak_seen = content_peak_nits(p.screen.window());
  const int auto_peak = auto_peak_nits(peak_seen);
  // Le HDR suit le système (system_hdr) : sortie HDR10 si l'écran en reçoit assez, sinon SDR gradée, la
  // swapchain en SDR comme avec --sdr.
  settings.hdr_native = !p.options.sdr && system_hdr(peak_seen);
  if (!settings.hdr_native)
    log->info("sortie SDR (HDR du système absent, coupé ou sous {} nits) : image SDR gradée",
              kMinHdrPeakNits);
  p.screen.set_sdr_output(!settings.hdr_native);
  // La couleur de la source telle que le flux l'étiquette : l'écran en convertit gamut, courbe, HDR10, HLG.
  const std::optional<ColorInfo> source_color = media::track_color(video);
  const ColorInfo source = source_color.value_or(ColorInfo{});
  const bool source_hdr = source.transfer == Transfer::Pq || source.transfer == Transfer::Hlg;
  p.screen.set_source_color(source);

  const std::string label_text = std::format("{}{}{}", c.motion_label(settings),
                                             // Le HDR, s'il tourne vraiment sur une sortie HDR10.
                                             settings.hdr_native ? "HDR · " : "", c.span_label());
  log->info("{} : {}", prepare::utf8_path(options.file.filename()), label_text);
  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(rife_loaded ? factor_of(settings) : Factor{});
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  PipelineConfig config{
      .file = options.file,
      .decoder = decoder_options(c.backend_name, p.screen.device(), p.screen.shared_devices()),
      .languages = {.presets = settings.language_presets},
      .audio_device = settings.audio_device,
      .audio_pll = p.options.audio_pll,
      .audio_track = p.audio_track,
      .subtitle_track = p.subtitle_track};
  PlaceboScreen& out = p.screen;
  LoopWake wake{.event = loop_wake_event()};  // vit plus longtemps que la chaîne, qui l'appelle
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = work_size(width, height, settings.work_height);
        // Étape 5.5a : anneau RGB10A2 derrière --ring-rgb10a2, tant que la porte de banding n'est pas faite.
        const RingFormat ring_format =
            p.options.ring_rgb10a2 && c.backend && c.backend->capabilities().ring_rgb10a2
                ? RingFormat::Rgb10a2Unorm
                : RingFormat::Rgba16Unorm;
        auto slots = out.create_ring(sized.width * factor, sized.height * factor,
                                     ring_depth(settings, rife_loaded ? factor_of(settings) : Factor{}),
                                     c.backend_name == "tensorrt", ring_format);
        if (!slots) return std::unexpected(std::move(slots.error()));
        const MasterClock* clock = stage.clock;
        return UpscaleStage::start(std::move(stage),
                                   c.upscale(settings, live, clock, width, height, std::move(*slots)));
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));
  (*pipeline)->inbox().set_waker(&LoopWake::push, &wake);
  const WriteWatch watching(out, p.options.ring_watcher ? &LoopWake::push : nullptr, &wake);  // 6.5

  const std::unique_ptr<GpuTelemetry> telemetry = GpuTelemetry::open(p.screen.device());
  const std::uint64_t energy_start_mj =
      telemetry ? telemetry->sample(host_now_us()).value_or(GpuSample{}).energy_mj : 0;
  ChainStats chain_stats(*live, *c.backend->kernels(), rife_loaded, std::string(c.motion_name()),
                         fps_of(video.frame_rate), telemetry.get());
  SessionConfig session_config = (*pipeline)->session_config();
  session_config.start_us = start_us;
  p.screen.set_sample_aspect(session_config.video_sar);
  session_config.stop_after_us = options.stop_after_us;
  session_config.fullscreen_after_us = options.fullscreen_after_us;
  session_config.vsync_index = options.vsync_index;
  session_config.header = {label_text};
  session_config.tracks = (*pipeline)->describe();
  {
    // Le dossier des LUT de l'utilisateur existe dès le premier lancement : il n'a qu'à y déposer ses .cube.
    std::error_code made;
    std::filesystem::create_directories(paths::luts(p.paths), made);
  }
  if (std::ranges::any_of(look_slots(settings), [](const LookLayer& l) { return !l.key.empty(); }))
    apply_looks(p.screen, settings, paths::luts(p.paths));
  p.screen.set_grade(grade_of(settings, auto_peak, {}, {}, source_hdr));
  p.screen.set_finish(settings.final_scaler, settings.deband, settings.sharpness);
  session_config.output = output_text(out, settings, source_hdr);
  session_config.display_period_us = display_period_us(out.window());
  const std::int64_t bench_period_us = session_config.display_period_us;
  session_config.display_hz = display_refresh_hz(out.window());
  session_config.display_sync = settings.display_sync;
  session_config.swapchain_depth = kSwapchainDepth;
  session_config.video_latency_us = std::int64_t{settings.video_latency_ms} * 1000;
  session_config.stats = settings.stats;
  session_config.subtitle_scale = settings.subtitle_scale;
  session_config.subtitle_delay_us = std::int64_t{settings.subtitle_delay_ms} * 1000;
  session_config.extra = &chain_stats;
  session_config.latency = &live->latency();
  session_config.on_written = [live](const RingFrame& frame, std::int64_t done_us) {
    live->auto_written(frame.left.value, frame.right.value, frame.submitted_us, done_us, frame.repeated);
  };
  log->info("{}", label_text);
  for (const std::string& line : session_config.tracks) log->info("{}", line);
  log->info("{}", session_config.output);
  Session session(std::move(session_config), (*pipeline)->stages(), out, (*pipeline)->clock());
  if (start_paused) session.command(cmd::Pause{});
  const bool fp8_engine = c.backend && c.backend->capabilities().fp8;
  // Le mode Performance de SPAN : ce que le moteur annonce, et sinon pourquoi (moteur absent ou autre carte).
  const SpanFp8Facts fp8_facts{.backend_name = c.backend_name,
                               .gpu_name = gpu_facts(p.screen).name,
                               .tensorrt_present = tensorrt_backend_present(p.catalog, p.paths),
                               .capability = c.backend && c.backend->capabilities().span_fp8_cutlass};
  const std::string fp8_reason = span_fp8_unavailable_reason(fp8_facts);
  log_channel("app")->info("SPAN Performance (FP8) : {} (moteur {}, capacite CUTLASS {})",
                           fp8_reason.empty() ? "disponible" : fp8_reason, c.backend_name,
                           fp8_facts.capability);
  const SpanPrepare span_prepare =
      c.model
          ? make_span_prepare(
                p.library,
                gpu::BackendInit{
                    .render = &p.screen.device(), .cache_dir = p.paths.engines, .log = backend_log()},
                fp8_facts.capability, c.model->path, paths::derived_models(p.paths), p.options.file,
                fp8_reason, span_still_reduce(work.width, work.height, settings.downscale),
                SpanFp8Context{.gpu_arch = c.backend ? c.backend->capabilities().gpu_arch : std::string(),
                               .kernels_version = c.backend ? c.backend->capabilities().span_fp8_kernels : 0,
                               .width = work.width,
                               .height = work.height},
                settings.preset != Preset::Custom)
          : SpanPrepare{};
  const std::optional<DrbaModel> drba_file = find_distil_drba(c.model_places);
  p.ui.set_tracks((*pipeline)->tracks());
  SettingsChoices choices{
      .models = c.model ? c.model->available : std::vector<std::string>{},
      .default_model = c.model ? c.model->default_name : std::string(),
      .source_width = video.width,
      .source_height = video.height,
      .source_fps = fps_of(video.frame_rate),
      .source_rate = video.frame_rate,
      .display_hz = display_period_us(out.window()) > 0
                        ? 1e6 / static_cast<double>(display_period_us(out.window()))
                        : 0.0,
      .display_vrr = p.display && p.display->vrr.value_or(false),
      .display_vrr_armed = p.display && p.display->vrr_armed.value_or(false),
      .rife_available = c.rife_model.has_value() || c.drba_found || offered(MotionEngine::Apple),
      .span_tiers = c.model ? span_tiers(c.model->path, paths::derived_models(p.paths), fp8_engine,
                                         c.backend && c.backend->capabilities().span_fp8_cutlass)
                            : TierSupport{},
      .rife_tiers = c.rife_choice ? sibling_tiers(c.rife_choice->path, fp8_engine) : TierSupport{},
      .drba_tiers = drba_file ? sibling_tiers(drba_file->path, fp8_engine) : TierSupport{},
      .distil_drba_available = c.drba_found,
      .rife_loaded = rife_loaded,
      .rife_models = c.rife_choice ? c.rife_choice->available : std::vector<std::string>{},
      .default_rife_model = c.rife_choice ? c.rife_choice->default_name : std::string(),
      .rife_folder = c.model_places.front().string(),
      .hdr_output = p.screen.hdr_output(),
      .auto_peak = auto_peak,
      .engines = engine_badges(p.paths.engines,
                               c.span_int8 ? c.span_int8->onnx
                               : c.model   ? c.model->path
                                           : std::filesystem::path(),
                               work, c.rife || c.rife_motion ? c.rife_model : std::nullopt,
                               c.rife_motion ? c.geometry : c.image_geometry),
      .chosen_engines = chosen_line(p.engines),
      .engine_in_use = engine_of_library(p.library->path()),
      .engine_note = fallback_note(p.engines, p.library->path()),
      .display_text = p.display ? describe(*p.display) : std::string("EDID illisible : OLED inconnu"),
      .source_file = p.options.file,
      .display_height = display_height(out.window()),
      .calibrated = calibrated_chain(p.settings_path.parent_path(), gpu_facts(p.screen).name,
                                     display_height(out.window()))
                        .has_value(),
      .span_prepare = span_prepare,
      .fp8_reason = fp8_reason,
      .span_job = p.span_job.note};
  fill_host_choices(choices, p.screen, p.paths, p.catalog, p.settings_path.parent_path());
  p.ui.set_settings(settings, choices);
  Chain chain{.session = session,
              .screen = out,
              .ui = p.ui,
              .pipeline = **pipeline,
              .live = *live,
              .running = settings,
              .rife_loaded = rife_loaded,
              .auto_peak = auto_peak,
              .file = p.options.file,
              .display = p.display,
              .header_text =
                  [&c](const PlaySettings& s) {
                    return std::format("{}{}{}", c.motion_label(s), s.hdr_native ? "HDR · " : "",
                                       c.span_label());
                  },
              .luts_dir = paths::luts(p.paths),
              .source_hdr = source_hdr,
              .kernels = c.backend ? c.backend->kernels() : nullptr,
              .shots = p.shots,
              .shot_cuts = p.options.shot_cuts,
              .anime_shots = p.options.anime_shots};
  apply_oled_care(chain, settings);

  std::ofstream stats;
  if (options.stats) {
    stats.open(*options.stats);
    stats << stats_header();
  }
  const core::KeyMap& keys = configured_keys();
  std::optional<Refusal> failure;
  if (p.began_us == 0) p.began_us = host_now_us();
  std::vector<Input> actions;  // vidée à chaque tour en s'appliquant (la 0.10.0 la rejouait sans fin)
  session.begin(host_now_us());
  if (p.unsaved) {
    if (auto saved =
            save_settings(p.settings_path, settings_to_save(*p.unsaved, p.from_file, p.started, p.options));
        !saved)
      log->warn("{}", saved.error().detail);
    p.unsaved.reset();
  }
  if (p.resumed) {
    p.ui.controls().notify(
        i18n::ui_tr_format("Resumed at {}", format_clock(start_us, session.view().duration_us)),
        host_now_us());
    p.resumed = false;
  }
  if (!p.notice.empty()) {
    p.ui.controls().notify(p.notice, host_now_us());
    p.notice = {};
  }
  // play.toml, la position et la mémoire de l'Auto s'écrivent sur leur thread, sans attendre le disque.
  LatestWriter writer;
  if (p.shots != nullptr) p.shots->writer = &writer;  // le cache des plans aussi
  const std::unique_ptr<ShotWork, void (*)(ShotWork*)> shots_writer(
      p.shots, [](ShotWork* shot_work) { shot_work->writer = nullptr; });
  // Ce que play.toml garde.
  const auto saved_settings = [&] {
    return settings_to_save(with_kept_choices(chain, chain.running), p.from_file, p.started, p.options);
  };
  // La position retenue toutes les kResumeEveryUs, et à la fin : un arrêt brutal en perd peu.
  std::int64_t kept_us = kNoPts;
  const auto remember = [&] {
    if (p.resume_path.empty()) return;
    const ControlsView view = session.view();
    if (view.position_us == kNoPts || view.position_us == kept_us) return;
    kept_us = view.position_us;
    writer.post("resume", [store = p.resume_path, file = p.options.file, at = view.position_us,
                           duration = view.duration_us > 0 || !session.finished() ? view.duration_us
                                                                                  : view.position_us] {
      return remember_position(store, file, at, duration);  // P3.83 : sans durée, la fin vue oublie
    });
  };
  std::int64_t remembered_at = host_now_us();
  // Son : le volume de la barre et le délai des réglages valent dès la première image.
  session.command(Command{p.ui.controls().volume_command()});
  if (settings.audio_delay_ms != 0)
    session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{settings.audio_delay_ms} * 1000});
  std::int64_t saved_at = host_now_us();
  std::int64_t display_polled_at = saved_at;
  std::int64_t hdr_polled_at = host_now_us();
  p.ui.controls().show_deferred(host_now_us());
  // Le mode auto : son plafond suit la fréquence de l'écran.
  bool vrr_armed = !p.options.no_vrr && p.display && p.display->vrr.value_or(false) &&
                   p.display->vrr_armed.value_or(false);
  std::string free_note;  // pourquoi le mode Rapide ne tourne pas alors qu'il est choisi (vide : il tourne ou
                          // n'est pas voulu)
  // La fréquence de l'écran selon SDL, gardée quand SDL ne la donne plus : un écran rallumé passe un moment
  // sans mode courant (0 Hz), et le plafond de l'auto retombait au rythme de la source.
  double known_hz = display_refresh_hz(out.window());
  std::int64_t known_period_us = display_period_us(out.window());
  SDL_DisplayID known_display = SDL_GetDisplayForWindow(out.window());
  const int opened_height = display_height(out.window());
  const GrowingFile opened_file =
      GrowingFile::of(p.options.bench_out ? std::filesystem::path() : p.options.file);
  double session_hz_seen = known_hz;
  ContractWatch contract_watch(out.window(), p.display, log);  // 8.2 : relu sur HDR_STATE_CHANGED et KWin
  const auto read_display_facts = [&] {
    if (const double hz = display_refresh_hz(out.window()); hz > 0.0) {
      known_hz = hz;
      known_period_us = display_period_us(out.window());
    }
  };
  // Repli du Rapide pour la séance seulement : play.toml garde le choix, qu'un choix explicite annule.
  core::FreeFallback free_fallback;  // revient au Rapide après un calme (core/free_rate.h)
  bool free_chosen = chain.running.auto_free;
  const auto apply_auto = [&] {
    if (chain.running.auto_free != free_chosen) {
      free_chosen = chain.running.auto_free;
      free_fallback.clear();
    }
    const bool free_wanted = free_requested(chain.running, free_fallback.active(host_now_us()));
    const bool on = chain.running.rife_auto && chain.rife_loaded && chain.running.rife_factor >= 2;
    const double limit_hz =
        core::auto_limit_hz(known_hz, power_auto_cap(chain.running.auto_cap, on_battery()));
    // mode Rapide (core/free_grid.h) : écran VRR armé, RIFE, fréquence de l'écran connue ; sinon Auto.
    std::string why;
    if (on && free_wanted) {
      if (free_instants_limit(chain.running.motion_engine) == 0)
        why = i18n::tr_format("{} cannot compute at a free instant", label(chain.running.motion_engine));
      else if (limit_hz <= 0.0 || (!vrr_armed && known_period_us <= 0))
        why = i18n::tr("display refresh rate unknown");
    }
    const bool free = on && free_wanted && why.empty();
    if (why != free_note) {
      free_note = why;
      // Rapide est le défaut : là où il ne tient pas, Stable le remplace sans message à l'écran.
      if (!why.empty()) log->info("Mode Rapide indisponible ({}) : mode Stable", why);
    }
    chain.live.set_auto(on && !free, core::auto_limits(limit_hz, fps_of(video.frame_rate), vrr_armed,
                                                       chain.running.rife_factor));
    // Écart minimal : la période de l'écran ou du plafond, à 1 % près ; à fréquence fixe, sur les vsyncs.
    const std::int64_t refresh_us = known_period_us;
    const bool vsync = free && !vrr_armed;
    chain.live.set_free(free, limit_hz > 0.0 ? std::llround(1e6 / (vsync ? limit_hz : limit_hz * 0.99)) : 0,
                        host_now_us());
    chain.live.set_steady(free && p.options.fast_steady, {.vrr = vrr_armed, .hz = limit_hz},
                          fps_of(video.frame_rate));
    chain.live.set_vsync(vsync ? refresh_us : 0, session.visible_delay_us());
  };
  apply_auto();
  // Le dernier palier tenu pour cette vidéo, ce moteur, cette hauteur de travail et ce GPU : le départ.
  std::filesystem::path auto_memory_path;
  std::string auto_memory_key;
  if (!p.settings_path.empty() && !p.options.file.empty()) {
    std::error_code size_error;
    const std::uintmax_t file_size = std::filesystem::file_size(p.options.file, size_error);
    auto_memory_path = paths::auto_memory(p.settings_path.parent_path());
    auto_memory_key =
        core::auto_memory_key(prepare::utf8_path(p.options.file), size_error ? 0 : file_size,
                              std::string(c.motion_name()), settings.work_height, gpu_facts(p.screen).name);
    if (chain.live.auto_active()) {
      if (const auto remembered = core::load_auto_step(auto_memory_path, auto_memory_key)) {
        chain.live.auto_seed(*remembered, host_now_us());
        log->info("auto : départ mémorisé ×{}", auto_factor_text(*remembered));
      }
    }
  }
  bool vrr_fullscreen = true;  // relu au premier tour : VRR, compositeur (P3.45)
  VrrWatch vrr_watch(vrr_armed);
  // VRR armé : l'image tenue se représente à intervalles égaux au-dessus du plancher de l'écran (EDID 0xFD).
  const auto apply_floor = [&] {
    const int min_hz = p.display ? p.display->vrr_floor_hz() : 0;
    session.set_vrr_floor(vrr_armed ? core::vrr_floor_us(min_hz) : 0);
  };
  apply_floor();
  if (p.display)
    log->info("VRR : {} · {}", p.display->vrr.value_or(false) ? "capable" : "non capable",
              p.display->vrr_armed.value_or(false) ? "armé" : "non armé");
  AutoWatch auto_watch;  // le palier de l'Auto : journal, annonce, mémoire (auto_watch.h)
  std::int64_t auto_missing_seen = 0;
  std::int64_t free_dropped_seen = 0;
  std::int64_t free_missing_seen = 0;
  const double auto_source_fps = fps_of(video.frame_rate);
  // Un changement de palier : une ligne de journal avec sa raison. Le message à l'écran n'est dit qu'une fois
  // le palier tenu 3 s (pas de clignotement), et le palier tenu 6 s est retenu pour la prochaine lecture.
  const auto show_auto = [&] {
    const Factor now = chain.live.factor_now();
    const bool free = chain.live.free_active();
    const std::int64_t at = host_now_us();
    const core::AutoInfo info = chain.live.auto_info(at);
    const AutoActions todo = auto_watch.turn(
        {.at_us = at,
         .centi = chain.running.rife_auto && !free && now.den >= 1 ? now.num * 100 / now.den : 0,
         .free = free,
         .at_floor = info.step <= info.granularity && info.ceiling > info.granularity,
         .can_store = !auto_memory_path.empty()});
    if (todo.log_free) log->info("{}", free_status_text(chain.live.free_info(at)));
    if (todo.log_change) log->info("auto : {}", auto_status_text(info, auto_source_fps));
    // Figé au plancher alors que l'écran permet plus : on le dit au journal (avec ce qui peut le tenir là),
    // la régulation tente d'elle-même un cran toutes les 8 s (core::AutoRate::kFloorEscapeUs).
    if (todo.floor_us) {
      const int g = std::max(info.granularity, 1);
      log->warn(
          "Auto : plafond figé à ×1 depuis {} s (dernier changement : {} ; plafond de l'écran ×{}, écran "
          "{:.1f} "
          "Hz, plafond choisi {}, palier prédit ×{}, limite apprise ×{})",
          *todo.floor_us / 1'000'000, core::auto_reason_text(info.reason),
          auto_factor_text(info.ceiling * 100 / g), known_hz, chain.running.auto_cap,
          auto_factor_text(info.predicted * 100 / g), auto_factor_text(info.cap * 100 / g));
    }
    if (todo.notify)
      p.ui.controls().notify(i18n::ui_tr_format("Auto ×{} ({:.0f} fps)", auto_factor_text(*todo.notify),
                                                auto_source_fps * *todo.notify / 100.0),
                             at);
    if (todo.store)
      writer.post("auto_memory", [store = auto_memory_path, key = auto_memory_key, centi = *todo.store] {
        return core::store_auto_step(store, key, centi);
      });
  };
  while (!session.finished() && !chain.reopen) {
    for (;
         p.next_key < p.options.keys.size() && host_now_us() - p.began_us >= p.options.keys[p.next_key].first;
         ++p.next_key) {
      const std::string& name = p.options.keys[p.next_key].second;
      if (auto job = scripted_export(name, chain.running, p.video, p.options.file))
        actions.emplace_back(StartExport{.job = std::move(*job)});
      else
        press(name, out.window());
    }
    const bool snapshot_now = p.snapshot.maybe(out, p.began_us);
    // Remis à faux avant de vider la file (P2.24) : un message arrivé pendant ce tour repousse son réveil, au
    // lieu de trouver pending vrai et de laisser l'attente suivante aller à son terme (jusqu'à 4 ms).
    wake.pending.store(false, std::memory_order_release);
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (auto drop = dropped(event)) {  // vidéo, sous-titres ou refus (P3.85)
        take_drop(chain, *drop, host_now_us());
        continue;
      }
      if (p.ui.handle(event, host_now_us(), actions)) continue;  // saisie dans l'interface
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    // L'écran de la fenêtre peut changer (déplacement, mode d'affichage) : on relit sa cadence.
    if (host_now_us() - display_polled_at >= kDisplayPollUs || out.vrr_changed()) {  // 6.10 : sans attendre
      display_polled_at = host_now_us();
      // Le mode Performance de SPAN se prépare tout seul (span_auto.h), pris à la vidéo suivante (P3.24).
      if (!p.options.file.empty() && c.model &&
          !p.options.bench_out) {  // le bench ne valide pas sur son clip
        const SpanAutoTick span =
            tick_span_auto(chain.running.span_precision, span_prepare, p.span_job,
                           prepare::utf8_path(c.model->path) + "|" + prepare::utf8_path(p.options.file),
                           chain.running.preset != Preset::Custom);
        if (span.notify) p.ui.controls().notify(*span.notify, host_now_us());
      }
      // L'écran change (éteint, autre sortie, nouveau mode) : sa fréquence, son VRR et son état sont relus.
      read_display_facts();
      if (const auto contract = contract_watch.tick()) out.set_contract(*contract);  // 8.2
      // Le HDR et le pic suivent le système et l'écran (Windows, KDE), relus toutes les 2 s.
      if (!p.options.sdr && host_now_us() - hdr_polled_at >= kHdrPollUs) {
        hdr_polled_at = host_now_us();
        const std::optional<double> peak_now = content_peak_nits(out.window());
        if (system_hdr(peak_now) != chain.running.hdr_native) {
          PlaySettings switched = chain.running;
          switched.hdr_native = !switched.hdr_native;
          chain.auto_peak = auto_peak_nits(peak_now);
          log->info("HDR du système {} : sortie {}", switched.hdr_native ? "allumé" : "coupé",
                    switched.hdr_native ? "HDR10" : "SDR gradée");
          apply(Input{ApplySettings{.settings = switched}}, chain, host_now_us());
        } else {
          follow_display(chain, peak_now);  // P3.58
        }
      }
      const SDL_DisplayID display_id = SDL_GetDisplayForWindow(out.window());
      const bool display_changed =
          display_id != known_display || (display_id != 0 && known_hz > 0.0 && known_hz != session_hz_seen);
      known_display = display_id;
      session_hz_seen = known_hz;
      session.set_display(known_period_us, known_hz);
      apply_auto();
      // Le mode auto écoute les images manquées à l'écran (sans paire en retard, le GPU peut ne pas suivre)
      // et le chien de garde veille sur le moteur d'interpolation.
      const FrameStats& shown = session.stats();
      chain.live.auto_missed(host_now_us(), shown.missing - auto_missing_seen);
      auto_missing_seen = shown.missing;
      if (chain.live.free_active()) {
        // Images perdues : sautées à la présentation (arrivées trop tard) ou manquées (trous dans les pts).
        const std::int64_t lost =
            std::max(shown.missing - free_missing_seen, shown.dropped - free_dropped_seen);
        chain.live.free_missed(host_now_us(), lost);
        if (chain.live.free_should_fall_back(host_now_us())) {
          log->warn("Mode Rapide : trop d'images perdues ({}) : retour au mode Stable",
                    free_status_text(chain.live.free_info(host_now_us())));
          if (free_fallback.fell_back(host_now_us()))
            p.ui.controls().notify(i18n::ui_tr("Fast mode: too many frames lost, back to Stable"),
                                   host_now_us());
          apply_auto();
        }
      }
      if (const auto notice = session.take_notice())
        p.ui.controls().notify(
            i18n::ui_tr_format("Playback goes on after an error: {}", visible_error(*notice)), host_now_us());
      free_missing_seen = shown.missing;
      free_dropped_seen = shown.dropped;
      chain.live.auto_tick(host_now_us(), !session.view().paused);
      show_auto();
      // KDE arme et désarme le VRR selon la fenêtre active : relu à chaque tour (un seul ioctl).
      if (p.display && !p.display->connector.empty()) {
        if (const auto now =
                vrr_watch.observe(out.vrr_armed(), window_vrr_armed(out.window(), p.display->connector))) {
          const bool armed = !p.options.no_vrr && p.display->vrr.value_or(false) && *now;
          log->info("VRR : {}", *now ? "armé" : "non armé");
          p.ui.set_display_vrr(p.display->vrr.value_or(false), *now,
                               (SDL_GetWindowFlags(out.window()) & SDL_WINDOW_FULLSCREEN) != 0);
          if (armed != vrr_armed) {
            vrr_armed = armed;
            apply_floor();
            apply_auto();
          }
        }
      }
      // KDE arme le VRR en plein écran seulement : on le relit quand la fenêtre change d'état.
      const bool fullscreen = (SDL_GetWindowFlags(out.window()) & SDL_WINDOW_FULLSCREEN) != 0;
      if (fullscreen != vrr_fullscreen || display_changed) {
        if (display_changed) log->info("écran : changement détecté ({} Hz), relecture du VRR", known_hz);
        if (display_changed && !p.options.bench_out && !p.options.file.empty() &&
            display_moves_chain(chain.running.preset, opened_height, display_height(out.window())))
          actions.emplace_back(PlayFile{p.options.file, session.view().position_us, session.paused()});
        vrr_fullscreen = fullscreen;
        session.set_composited(!fullscreen);
        if (const auto display = window_display(out.window())) {
          if (display_changed) p.display = display;  // P3.40 : l'écran courant, son connecteur et son EDID
          p.ui.set_display_vrr(display->vrr.value_or(false), display->vrr_armed.value_or(false), fullscreen);
          const bool armed =
              !p.options.no_vrr && display->vrr.value_or(false) && display->vrr_armed.value_or(false);
          if (armed != vrr_armed) {
            vrr_armed = armed;
            apply_floor();
            apply_auto();  // les demi-paliers suivent le VRR
          }
          log->info("VRR : {} · {}", display->vrr.value_or(false) ? "capable" : "non capable",
                    display->vrr_armed.value_or(false) ? "armé" : "non armé");
        }
      }
    }
    grade_tick(chain, host_now_us());
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) session.handle(std::move(*message));
    if (chain.black_dirty && host_now_us() - chain.black_at_us >= kBlackRefreshUs) refresh_black(chain);
    UiFrame ui_frame = p.ui.frame(host_now_us(), session.view(), actions);
    ui_frame.changed |= snapshot_now;
    for (const Input& input : std::exchange(actions, {})) apply(input, chain, host_now_us());
    if (chain.dirty && host_now_us() - saved_at >= kSettingsSaveEveryUs) {
      writer.post("play.toml",
                  [path = p.settings_path, saved = saved_settings()] { return save_settings(path, saved); });
      chain.dirty = false;
      saved_at = host_now_us();
    }
    if (host_now_us() - remembered_at >= kResumeEveryUs) {
      remember();
      remembered_at = host_now_us();
    }
    auto record = session.tick(host_now_us(), ui_frame);
    if (!record) {
      failure = std::move(record.error());
      break;
    }
    if (*record && (*record)->fresh)
      chain.live.vsync_note((*record)->host_us, out.vblank_grid(), (*record)->display_gpu_us);
    if (*record && stats.is_open()) write_record(stats, **record);
    // Rien de présenté (rien n'a changé, ou fenêtre cachée) : la FIFO n'impose pas le rythme. En
    // lecture, la boucle dort jusqu'à l'échéance de l'image suivante (Session::idle_wait_us) ; en pause,
    // au plus kPausedWaitMs. Un message des autres threads (une image, un seek) la réveille.
    if (!*record) {
      const std::int64_t wait_us =
          session.view().paused ? std::int64_t{kPausedWaitMs} * 1000 : session.idle_wait_us(host_now_us());
      wait_loop(wait_us, p.options.precise_wait);  // --precise-wait : à la microseconde (6.7)
    }
  }
  if (chain.dirty) {
    writer.post("play.toml",
                [path = p.settings_path, saved = saved_settings()] { return save_settings(path, saved); });
  }
  remember();
  writer.flush();  // la vidéo suivante, ou le prochain lancement, relit ces fichiers
  p.quit = session.quit_asked();
  log_summary(session);
  if (telemetry) {
    // Énergie de la lecture : le compteur NVML relevé au début et à la fin (au plus 500 ms de retard).
    const std::uint64_t energy_end_mj = telemetry->sample(host_now_us()).value_or(GpuSample{}).energy_mj;
    if (const auto per_frame =
            millijoules_per_frame(energy_start_mj, energy_end_mj, session.stats().new_frames))
      log->info("énergie : {:.1f} J, {:.1f} mJ par image neuve",
                static_cast<double>(energy_end_mj - energy_start_mj) / 1000.0, *per_frame);
  }
  if (p.options.bench_out) {
    write_bench_reading(*p.options.bench_out, gpu_facts(p.screen), session.stats(),
                        BenchChain{.level = static_cast<Tier>(p.options.bench_level.value_or(0)),
                                   .period_us = bench_period_us,
                                   .work_height = chain.running.work_height,
                                   .rife_model = chain.running.rife_model,
                                   .span_variant = span_variant});
  }
  for (const std::string& refusal : session.refusals()) log->warn("{}", refusal);
  // Un étage arrêté sur un refus fatal : un échec, jamais une fin d'épisode (pas d'épisode suivant).
  if (!failure && session.failure()) failure = *session.failure();
  if (failure && p.options.bench_out)
    tell_bench_parent(
        {.event = "refusal", .code = std::string(name(failure->code)), .detail = failure->detail});
  // La source a changé de taille en cours de lecture : la chaîne se rouvre à la nouvelle taille, au même
  // endroit ; au-delà de 4 réouvertures par minute, c'est un refus.
  if (session.resize_asked() && !chain.reopen && !failure) {
    std::deque<std::int64_t>& resizes = p.resizes;
    const std::int64_t now = host_now_us();
    while (!resizes.empty() && now - resizes.front() > kResizeWindowUs) resizes.pop_front();
    if (resizes.size() >= kResizesRefused) {
      failure = Refusal{RefusalCode::SourceResized, "la source change de taille trop souvent"};
    } else {
      resizes.push_back(now);
      log->info("la source a changé de taille : réouverture de la chaîne");
      chain.reopen = chain.running;
    }
  }
  if (!chain.reopen && !failure && !session.quit_asked() && opened_file.grew(p.options.file)) {
    log->info("le fichier grandit encore : relu au même endroit");
    apply(Input{PlayFile{p.options.file, session.view().position_us, false}}, chain, host_now_us());
  }
  // Fin de la vidéo, sans demande de quitter : l'épisode suivant du dossier, si le réglage le veut.
  if (!chain.reopen && !failure && !session.quit_asked() && !p.options.bench_out &&
      chain.running.autoplay_next && !p.options.file.empty()) {
    if (const auto next = sibling_episode(p.options.file, 1)) {
      log->info("épisode suivant : {}", prepare::utf8_path(next->filename()));
      chain.next_file = *next;
      chain.reopen = chain.running;
    }
  }
  std::optional<Reopen> reopen;
  if (chain.reopen && !failure) {
    reopen = Reopen{.settings = *chain.reopen,
                    .position_us = std::max<std::int64_t>(session.view().position_us, 0),
                    .paused = session.paused() && !chain.next_file,
                    .engines = chain.engines,
                    .export_job = chain.export_job,
                    .calibrate = chain.calibrate,
                    .next_file = chain.next_file,
                    .next_start_us = chain.next_start_us,
                    .next_paused = chain.next_paused};
    reopen->settings.volume = p.ui.controls().volume();
    p.audio_track = (*pipeline)->tracks().audio_selected;
    p.subtitle_track = (*pipeline)->tracks().subtitle_selected;
  }
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, le moteur avant son backend, le
  // backend (et ses imports du ring) avant l'écran qui possède le ring.
  session.end();
  out.flush();
  pipeline->reset();
  if (c.backend)
    keep_gpu_profile(c.backend->kernels()->gpu_profile(), options.stats, p.options.bench_out);  // 0.2
  prepared->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return reopen;
}

}  // namespace ajp::app::play
