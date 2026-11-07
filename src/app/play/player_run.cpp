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
#include "app/play/gmfss_model.h"
#include "app/play/gpu_profile_out.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/home_screen.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/latest_writer.h"
#include "app/play/look_catalog.h"
#include "app/play/loop_wait.h"
#include "app/play/machine_calibration.h"
#include "app/play/model_choice.h"
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
// En pause, l'attente d'un événement SDL au plus : le seuil de réactivité aux messages des threads.
constexpr std::int32_t kPausedWaitMs = 10;

}  // namespace

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
  const ChainStats chain_stats(*live, *c.backend->kernels(), rife_loaded, std::string(c.motion_name()),
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
    std::filesystem::create_directories(p.paths.engines.parent_path() / "luts", made);
  }
  if (std::ranges::any_of(look_slots(settings), [](const LookLayer& l) { return !l.key.empty(); }))
    apply_looks(p.screen, settings, p.paths.engines.parent_path() / "luts");
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
  // La tranche de la machine pour Vidéo > Post-traitement : la mesure de « Évaluer ma machine » si elle est
  // de cette carte et de ce pilote, sinon l'estimation.
  std::optional<BenchResult> measured;
  if (auto bench = load_bench(p.settings_path.parent_path() / "bench.toml")) measured = std::move(*bench);
  const GpuFacts machine = gpu_facts(p.screen);
  const Tier machine_tier = effective_tier(machine, measured);
  const std::string bench_text =
      measured && measured->gpu == machine.name && measured->driver == machine.driver
          ? i18n::tr_format("Last optimization: {}", measured->date)
          : std::string();
  std::vector<SettingsChoices::AudioOut> audio_outputs;
  for (const audio::AudioDevice& device : audio::playback_devices())
    audio_outputs.push_back(
        {.name = device.name, .channels = device.channels, .is_default = device.is_default});
  const bool default_player_is_ajp = is_default_player(place_of(p.paths));
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
                fp8_facts.capability, c.model->path, p.paths.engines.parent_path() / "modeles",
                p.options.file, fp8_reason, span_still_reduce(work.width, work.height, settings.downscale),
                SpanFp8Context{.gpu_arch = c.backend ? c.backend->capabilities().gpu_arch : std::string(),
                               .kernels_version = c.backend ? c.backend->capabilities().span_fp8_kernels : 0,
                               .width = work.width,
                               .height = work.height},
                settings.preset != Preset::Custom)
          : SpanPrepare{};
  const std::optional<DrbaModel> drba_file = find_distil_drba(c.gmfss_places);
  p.ui.set_tracks((*pipeline)->tracks());
  p.ui.set_settings(
      settings,
      SettingsChoices{
          .luts_dir = p.paths.engines.parent_path() / "luts",
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
          .span_tiers = c.model
                            ? span_tiers(c.model->path, p.paths.engines.parent_path() / "modeles", fp8_engine,
                                         c.backend && c.backend->capabilities().span_fp8_cutlass)
                            : TierSupport{},
          .rife_tiers = c.rife_choice ? sibling_tiers(c.rife_choice->path, fp8_engine) : TierSupport{},
          .drba_tiers = drba_file ? sibling_tiers(drba_file->path, fp8_engine) : TierSupport{},
          .distil_drba_available = c.drba_found,
          .rife_loaded = rife_loaded,
          .rife_models = c.rife_choice ? c.rife_choice->available : std::vector<std::string>{},
          .default_rife_model = c.rife_choice ? c.rife_choice->default_name : std::string(),
          .rife_folder = c.gmfss_places.front().string(),
          .nvidia = vendor_from_pci(p.screen.device_facts().pci_vendor) == GpuVendor::Nvidia,
          .hdr_output = p.screen.hdr_output(),
          .auto_peak = auto_peak,
          .engines = engine_badges(p.paths.engines,
                                   c.span_int8 ? c.span_int8->onnx
                                   : c.model   ? c.model->path
                                               : std::filesystem::path(),
                                   work, c.rife || c.rife_motion ? c.rife_model : std::nullopt,
                                   c.rife_motion ? c.geometry : c.image_geometry),
          .chosen_engines = chosen_line(p.engines),
          .switchable = installed_switchable(p.catalog, p.paths),
          .engine_in_use = engine_of_library(p.library->path()),
          .engine_note = fallback_note(p.engines, p.library->path()),
          .display_text = p.display ? describe(*p.display) : std::string("EDID illisible : OLED inconnu"),
          .source_file = p.options.file,
          .machine_tier = static_cast<int>(machine_tier),
          .display_height = display_height(out.window()),
          .calibrated =
              calibrated_chain(p.settings_path.parent_path(), machine.name, display_height(out.window()))
                  .has_value(),
          .gpu_name = gpu_facts(p.screen).name,
          .audio_outputs = audio_outputs,
          .engines_dir_default = prepare::utf8_path(p.paths.engines.parent_path() / "backends"),
          .engines_dir_free = format_size(free_bytes(p.catalog.layout.engines)),
          .default_player_is_ajp = default_player_is_ajp,
          .bench_text = bench_text,
          .place = place_of(p.paths),
          .span_prepare = span_prepare,
          .fp8_reason = fp8_reason,
          .span_job = p.span_job.note,
          .settings_dir = p.settings_path.parent_path()});
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
              .luts_dir = p.paths.engines.parent_path() / "luts",
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
  std::vector<Input> actions;
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
    writer.post("resume",
                [store = p.resume_path, file = p.options.file, at = view.position_us,
                 duration = view.duration_us] { return remember_position(store, file, at, duration); });
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
  // La fréquence de l'écran telle que SDL la donne, gardée quand SDL ne la donne plus : un écran éteint puis
  // rallumé passe un moment sans mode courant (0 Hz), et le plafond de l'auto retombait au rythme de la
  // source.
  double known_hz = display_refresh_hz(out.window());
  std::int64_t known_period_us = display_period_us(out.window());
  SDL_DisplayID known_display = SDL_GetDisplayForWindow(out.window());
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
    // Au fps près (1/24 de multiplicateur) sur un écran VRR armé : la cadence n'y est plus liée au
    // rafraîchissement. À fréquence fixe, des paliers entiers (les autres feraient vibrer).
    const int granularity = vrr_armed ? core::kFineGranularity : 1;
    const double limit_hz = core::auto_limit_hz(known_hz, chain.running.auto_cap);
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
    chain.live.set_auto(on && !free, core::auto_ceiling(limit_hz, fps_of(video.frame_rate), granularity),
                        granularity);
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
    auto_memory_path = p.settings_path.parent_path() / "auto_memory.toml";
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
  int auto_stored = 0;
  // Le palier du mode auto : un message à l'écran et une ligne de journal à chaque changement.
  bool vrr_fullscreen = false;
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
  int auto_shown = 0;  // centièmes de multiplicateur ; 0 : le mode auto est coupé
  int auto_notified = 0;
  std::int64_t auto_changed_us = 0;
  std::int64_t auto_floor_since_us = 0;
  std::int64_t auto_floor_logged_us = 0;
  std::int64_t auto_missing_seen = 0;
  std::int64_t free_dropped_seen = 0;
  std::int64_t free_missing_seen = 0;
  std::int64_t free_logged_us = 0;
  const double auto_source_fps = fps_of(video.frame_rate);
  // Un changement de palier : une ligne de journal avec sa raison. Le message à l'écran n'est dit qu'une fois
  // le palier tenu 3 s (pas de clignotement), et le palier tenu 6 s est retenu pour la prochaine lecture.
  const auto show_auto = [&] {
    const Factor now = chain.live.factor_now();
    const bool free = chain.live.free_active();
    const int centi = chain.running.rife_auto && !free && now.den >= 1 ? now.num * 100 / now.den : 0;
    const std::int64_t at = host_now_us();
    if (free && at - free_logged_us >= 5'000'000) {  // mode Rapide : une ligne de journal toutes les 5 s
      free_logged_us = at;
      log->info("{}", free_status_text(chain.live.free_info(at)));
    }
    if (centi != auto_shown) {
      if (centi > 0) {
        log->info("auto : {}", auto_status_text(chain.live.auto_info(at), auto_source_fps));
      }
      auto_shown = centi;
      auto_changed_us = at;
      return;
    }
    if (centi <= 0) return;
    // Figé au plancher alors que l'écran permet plus : on le dit au journal (avec ce qui peut le tenir là),
    // la régulation tente d'elle-même un cran toutes les 8 s (core::AutoRate::kFloorEscapeUs).
    {
      const core::AutoInfo info = chain.live.auto_info(at);
      if (info.step <= info.granularity && info.ceiling > info.granularity) {
        if (auto_floor_since_us == 0) auto_floor_since_us = at;
        if (at - auto_floor_since_us >= 8'000'000 && at - auto_floor_logged_us >= 30'000'000) {
          auto_floor_logged_us = at;
          log->warn(
              "Auto : plafond figé à ×1 depuis {} s (dernier changement : {} ; plafond de l'écran ×{}, écran "
              "{:.1f} Hz, "
              "plafond choisi {}, palier prédit ×{}, limite apprise ×{})",
              (at - auto_floor_since_us) / 1'000'000, core::auto_reason_text(info.reason),
              auto_factor_text(info.ceiling * 100 / std::max(info.granularity, 1)), known_hz,
              chain.running.auto_cap, auto_factor_text(info.predicted * 100 / std::max(info.granularity, 1)),
              auto_factor_text(info.cap * 100 / std::max(info.granularity, 1)));
        }
      } else {
        auto_floor_since_us = 0;
      }
    }
    if (centi != auto_notified && at - auto_changed_us >= 3'000'000) {
      if (auto_notified > 0)
        p.ui.controls().notify(i18n::ui_tr_format("Auto ×{} ({:.0f} fps)", auto_factor_text(centi),
                                                  auto_source_fps * centi / 100.0),
                               at);
      auto_notified = centi;
    }
    if (centi != auto_stored && at - auto_changed_us >= 6'000'000 && !auto_memory_path.empty()) {
      writer.post("auto_memory", [store = auto_memory_path, key = auto_memory_key, centi] {
        return core::store_auto_step(store, key, centi);
      });
      auto_stored = centi;  // même en cas d'échec : pas de réécriture à chaque tour
    }
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
      if (auto dropped = dropped_file(event)) {  // une autre vidéo : la chaîne se ferme pour elle
        log->info("fichier glissé : {}", prepare::utf8_path(dropped->filename()));
        chain.next_file = std::move(*dropped);
        chain.reopen = chain.running;
        continue;
      }
      if (p.ui.handle(event, host_now_us(), actions)) continue;  // saisie dans l'interface
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    // L'écran de la fenêtre peut changer (déplacement, mode d'affichage) : on relit sa cadence.
    if (host_now_us() - display_polled_at >= 500'000 || out.vrr_changed()) {  // 6.10 : sans attendre
      display_polled_at = host_now_us();
      // Le mode Performance de SPAN se prépare tout seul au cran Performance (span_auto.h) : la lecture reste
      // en FP16 pendant ce temps, puis la chaîne se rouvre pour prendre le paquet.
      if (!p.options.file.empty() && c.model &&
          !p.options.bench_out) {  // le bench ne valide pas sur son clip
        const SpanAutoTick span =
            tick_span_auto(chain.running.span_precision, span_prepare, p.span_job,
                           prepare::utf8_path(c.model->path) + "|" + prepare::utf8_path(p.options.file),
                           chain.running.preset != Preset::Custom);
        if (span.notify) p.ui.controls().notify(*span.notify, host_now_us());
        if (span.reopen) chain.reopen = chain.running;
      }
      // L'écran change (éteint puis rallumé, autre sortie, nouveau mode) : on relit sa fréquence, son VRR et
      // son état, sans quoi la cadence restait celle de l'écran d'avant.
      read_display_facts();
      if (const auto contract = contract_watch.tick()) out.set_contract(*contract);  // 8.2
      // Le HDR suit le système : allumé ou coupé pendant la lecture (Windows, KDE), la sortie suit en direct.
      // Relu toutes les 2 s : sous Windows, la crête se lit dans DXGI et l'EDID.
      if (!p.options.sdr && host_now_us() - hdr_polled_at >= 2'000'000) {
        hdr_polled_at = host_now_us();
        const std::optional<double> peak_now = content_peak_nits(out.window());
        if (system_hdr(peak_now) != chain.running.hdr_native) {
          PlaySettings switched = chain.running;
          switched.hdr_native = !switched.hdr_native;
          chain.auto_peak = auto_peak_nits(peak_now);
          log->info("HDR du système {} : sortie {}", switched.hdr_native ? "allumé" : "coupé",
                    switched.hdr_native ? "HDR10" : "SDR gradée");
          apply(Input{ApplySettings{.settings = switched}}, chain, host_now_us());
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
          free_fallback.fell_back(host_now_us());
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
        vrr_fullscreen = fullscreen;
        if (const auto display = window_display(out.window())) {
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
    for (const Input& input : actions) apply(input, chain, host_now_us());
    actions.clear();
    // play.toml réécrit au plus une fois par seconde pendant qu'on règle.
    if (chain.dirty && host_now_us() - saved_at >= 1'000'000) {
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
    static std::deque<std::int64_t> resizes;
    const std::int64_t now = host_now_us();
    while (!resizes.empty() && now - resizes.front() > 60'000'000) resizes.pop_front();
    if (resizes.size() >= 4) {
      failure = Refusal{RefusalCode::SourceResized, "la source change de taille trop souvent"};
    } else {
      resizes.push_back(now);
      log->info("la source a changé de taille : réouverture de la chaîne");
      chain.reopen = chain.running;
    }
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
