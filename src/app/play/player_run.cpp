#include "app/play/player_run.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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

#include "app/play/after_gap.h"
#include "app/play/auto_bench.h"
#include "app/play/auto_config.h"
#include "app/play/auto_watch.h"
#include "app/play/backend_library.h"
#include "app/play/bench_events.h"
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/chain_choices.h"
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/close_watch.h"
#include "app/play/default_player.h"
#include "app/play/display_edid.h"
#include "app/play/display_follow.h"
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
#include "app/play/model_choice.h"
#include "app/play/paths.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player_engines.h"
#include "app/play/present_stats.h"
#include "app/play/pumped_task.h"
#include "app/play/relaunch_target.h"
#include "app/play/resume.h"
#include "app/play/save_failure.h"
#include "app/play/scripted_input.h"
#include "app/play/session.h"
#include "app/play/sleep_notice.h"
#include "app/play/sleep_watch.h"
#include "app/play/span_auto.h"
#include "app/play/span_derive.h"
#include "app/play/span_int8.h"
#include "app/play/ui.h"
#include "app/play/upscale_stage.h"
#include "app/play/user_keys.h"
#include "app/play/visible_error.h"
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
// L'écran de la fenêtre est relu deux fois par seconde (sa fréquence, son VRR : la cadence s'y règle), sur
// le fil de DisplayProbe ; le HDR et le pic sur événement, sinon toutes les 10 s (display_follow.cpp).
constexpr std::int64_t kDisplayPollUs = 500'000;
// Les E/S du dossier de la vidéo (taille, fichier qui grandit, épisode suivant) : au-delà, laissées à leur
// sort, la lecture continue comme si elles n'avaient rien trouvé.
constexpr std::chrono::milliseconds kFolderIoLimit{5000};
// Une source qui change de taille 4 fois en une minute est refusée : la chaîne se rouvrirait sans fin.
constexpr std::int64_t kResizeWindowUs = 60'000'000;
constexpr std::size_t kResizesRefused = 4;
// Le démontage d'une chaîne saine (moteurs TensorRT compris) se compte en secondes (placebo_open.cpp).
constexpr std::chrono::seconds kTeardownLimit{60};

// La période d'une fréquence déjà relue (display_period_us referait le relevé du mode, deux
// QueryDisplayConfig par tour sous Windows) ; une fréquence inutilisable laisse SDL répondre.
// Les réglages d'une réouverture demandée, écrits une fois la chaîne rouverte (Player::unsaved), sans les
// raccourcis de séance (A08-04).
void save_reopened(Player& p) {
  if (!p.unsaved) return;
  const PlaySettings kept = with_kept_choices(p.kept, *p.unsaved);
  if (auto saved = save_settings(p.settings_path, settings_to_save(kept, p.from_file, p.started, p.options));
      !saved)
    log_channel("app")->warn("{}", saved.error().detail);
  p.unsaved.reset();
}

// Pourquoi le mode Rapide (core/free_grid.h) ne peut pas tourner : il lui faut un moteur qui calcule à un
// instant libre et la fréquence de l'écran (ou un VRR armé). Vide : il le peut ; sinon l'Auto Stable le
// remplace.
std::string free_refusal(MotionEngine engine, bool limit_known, bool display_known) {
  if (free_instants_limit(engine) == 0)
    return i18n::tr_format("{} cannot compute at a free instant", label(engine));
  return limit_known && display_known ? std::string() : std::string(i18n::tr("display refresh rate unknown"));
}

// B31-02 : l'étage que ce refus retire pour la réouverture (stage_fallback.h, retreat_stage) ; rien pour le
// banc, qui mesure la chaîne demandée ou rien.
std::optional<ChainStage> retreat(Player& p, const ChainEngines& c, const Refusal& refusal) {
  return retreat_stage(p.options.bench_out ? nullptr : p.stages, refusal, c.played, c.interpolates(),
                       c.engine != nullptr, p.video.width, p.video.height, library_name(p), p.options.file);
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<std::optional<Reopen>, Refusal> run_chain(Player& p, PlaySettings settings,
                                                        std::int64_t start_us, bool start_paused) {
  const auto log = log_channel("app");
  // B31-03 : une perte du GPU pendant la préparation relance au départ demandé, pas à 0:00.
  p.stopped_at_us = std::max<std::int64_t>(start_us, 0);
  std::vector<std::string> unavailable;
  settings = within_engines(settings, p.engines, unavailable);
  for (const std::string& line : unavailable) log->warn("{}", line);
  const PlayOptions options = options_for(p.options, settings);
  const media::TrackInfo& video = p.video;
  // La pré-analyse des plans part avant la préparation des moteurs : elle a de l'avance quand l'image vient.
  if (p.shots != nullptr)
    open_shots_early(*p.shots, p.options.file, settings, start_us, video,
                     content_peak_nits(p.screen.window()), p.options.sdr);
  // Un étage refusé ou annulé laisse la vidéo jouer sans lui (décisions 6 et 10) ; le banc mesure la chaîne
  // demandée ou rien.
  const RingRelease ring_release(p.screen);  // détruit après prepared, donc après le backend (A33-01)
  auto prepared = prepare_chain(p, settings, p.options.bench_out ? nullptr : p.stages);
  if (!prepared && prepared.error().key == kDroppedVideoKey && p.dropped_video) {
    // A08-08 : une vidéo glissée pendant la construction d'un moteur : la lecture passe à elle.
    Reopen to_dropped{
        .settings = settings, .position_us = start_us, .next_file = std::exchange(p.dropped_video, {})};
    return to_dropped;
  }
  if (!prepared) return std::unexpected(std::move(prepared.error()));
  if (!*prepared) {  // q ou la fenêtre fermée pendant la préparation des moteurs
    p.quit = true;
    return std::nullopt;
  }
  ChainEngines& c = **prepared;
  // Ce que les moteurs jouent : settings sans les étages refusés ; settings reste le choix de l'utilisateur,
  // que le panneau montre et que play.toml garde.
  const PlaySettings& played = c.played;
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
  announce_playback(p, source, settings.hdr_native, label_text);
  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(rife_loaded ? factor_of(played) : Factor{});
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  const PipelineConfig config = pipeline_config(p, settings, options.file, c.backend_name);
  PlaceboScreen& out = p.screen;
  LoopWake wake{.event = loop_wake_event()};  // vit plus longtemps que la chaîne, qui l'appelle
  bool ring_refused = false;                  // B31-02 : l'anneau ne tient pas en mémoire vidéo
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = chain_work_size(played, width, height);
        // Anneau RGBA16 ; RGB10A2 derrière --ring-rgb10a2 tant que la porte de banding n'est pas mesurée (ADR
        // 0008).
        const RingFormat ring_format = p.options.ring_rgb10a2 && c.backend->capabilities().ring_rgb10a2
                                           ? RingFormat::Rgb10a2Unorm
                                           : RingFormat::Rgba16Unorm;
        auto slots = out.create_ring(sized.width * factor, sized.height * factor, c.opened.shape.ring,
                                     c.backend_name == "tensorrt", ring_format);
        if (!slots) {
          ring_refused = true;
          return std::unexpected(std::move(slots.error()));
        }
        const MasterClock* clock = stage.clock;
        return UpscaleStage::start(std::move(stage),
                                   c.upscale(played, live, clock, width, height, std::move(*slots)));
      });
  if (!pipeline) {
    // B31-02 : l'anneau refusé retire un étage, et la chaîne se rouvre au même endroit sans lui.
    if (ring_refused && retreat(p, c, pipeline.error()))
      return Reopen{.settings = settings, .position_us = start_us, .paused = start_paused};
    return std::unexpected(std::move(pipeline.error()));
  }
  if ((*pipeline)->chapters_flattened() && p.notice.empty())
    p.notice = i18n::ui_tr("Chapters stored in other files are not played (linked opening or ending)");
  (*pipeline)->inbox().set_waker(&LoopWake::push, &wake);
  const WriteWatch watching(out, p.options.ring_watcher ? &LoopWake::push : nullptr, &wake);  // 6.5

  const std::unique_ptr<GpuTelemetry> telemetry = GpuTelemetry::open(p.screen.device());
  const std::uint64_t energy_start_mj = telemetry ? telemetry->energy_mj(host_now_us()) : 0;
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
  // Le mode Performance de SPAN : ce que le moteur annonce, et sinon pourquoi (moteur absent ou autre carte).
  const SpanFp8Facts fp8_facts{.backend_name = c.backend_name,
                               .gpu_name = gpu_facts(p.screen).name,
                               .tensorrt_present = tensorrt_backend_present(p.catalog, p.paths),
                               .capability = c.backend->capabilities().span_fp8_cutlass};
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
                settings.preset != Preset::Custom, true)
          : SpanPrepare{};
  p.ui.set_tracks((*pipeline)->tracks());
  SettingsChoices choices = chain_choices(p, c, span_prepare, fp8_reason, auto_peak);
  fill_host_choices(choices, p.screen, p.paths, p.catalog, p.settings_path.parent_path());
  p.ui.set_settings(settings, choices, &p.catalog);
  Chain chain{.session = session,
              .screen = out,
              .ui = p.ui,
              .pipeline = **pipeline,
              .live = *live,
              .running = settings,
              .opened = c.opened,
              .auto_peak = auto_peak,
              .kept = p.kept,
              .file = p.options.file,
              .display = p.display,
              .header_text =
                  [&c](const PlaySettings& s) {
                    return std::format("{}{}{}", c.motion_label(s), s.hdr_native ? "HDR · " : "",
                                       c.span_label());
                  },
              .luts_dir = paths::luts(p.paths),
              .source_hdr = source_hdr,
              .kernels = c.backend->kernels(),
              .shots = p.shots,
              .shot_cuts = p.options.shot_cuts,
              .anime_shots = p.options.anime_shots};
  carry_session(chain, p.stages, p.dropped_subtitles, p.subtitle_track);
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
  save_reopened(p);
  if (p.resumed) {
    p.ui.controls().notify(
        i18n::ui_tr_format("Resumed at {}", format_clock(start_us, session.view().duration_us)),
        host_now_us());
    p.resumed = false;
  }
  if (const i18n::Translated dropped = stage_notice(c.stage_notes); !dropped.empty())
    p.notice = p.notice.empty() ? dropped : dropped + i18n::Translated::data(" · ") + p.notice;
  if (!p.notice.empty()) {
    p.ui.controls().warn(p.notice, host_now_us());
    p.notice = {};
  }
  // play.toml, la position et la mémoire de l'Auto s'écrivent sur leur thread, sans attendre le disque.
  LatestWriter writer;
  if (p.shots != nullptr) p.shots->writer = &writer;  // le cache des plans aussi
  std::optional<StatsSink> stats_sink;                // B32-04 : --stats s'écrit sur ce même fil
  if (stats.is_open()) stats_sink.emplace(stats, writer);
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
  p.ui.controls().show_deferred(host_now_us());
  // Le mode auto : son plafond suit la fréquence de l'écran.
  DisplayState screen_state{.known_hz = display_refresh_hz(out.window()),
                            .known_period_us = display_period_us(out.window()),
                            .vrr_armed = !p.options.no_vrr && p.display && p.display->vrr.value_or(false) &&
                                         p.display->vrr_armed.value_or(false)};
  const double& known_hz = screen_state.known_hz;
  const std::int64_t& known_period_us = screen_state.known_period_us;
  const bool& vrr_armed = screen_state.vrr_armed;
  std::string free_note;  // pourquoi le mode Rapide ne tourne pas alors qu'il est choisi (vide : il tourne ou
                          // n'est pas voulu)
  // B31-12 : la taille du fichier hors du thread de la fenêtre, bornée (un partage réseau muet).
  const std::uintmax_t file_bytes =
      p.options.file.empty()
          ? 0
          : run_pumped_for<GrowingFile>("file-size", kFolderIoLimit, true,
                                        [file = p.options.file] { return GrowingFile::of(file); })
                .value_or(GrowingFile{})
                .size;
  const GrowingFile opened_file{p.options.bench_out ? 0 : file_bytes};
  // Repli du Rapide pour la séance seulement : play.toml garde le choix, qu'un choix explicite annule.
  core::FreeFallback free_fallback;  // revient au Rapide après un calme (core/free_rate.h)
  bool free_chosen = chain.running.auto_free;
  const auto apply_auto = [&] {
    if (chain.running.auto_free != free_chosen) {
      free_chosen = chain.running.auto_free;
      free_fallback.clear();
    }
    const bool free_wanted = free_requested(chain.running, free_fallback.active(host_now_us()));
    const bool on = chain.running.rife_auto && chain.opened.rife_loaded && chain.running.rife_factor >= 2;
    // Le plafond choisi borne la cadence ; le calage se fait toujours sur l'écran (A12-04), et jamais sur
    // un plafond de batterie (décision 11, A31-10).
    const double cap_hz = static_cast<double>(chain.running.auto_cap);
    const double limit_hz = core::auto_limit_hz(known_hz, chain.running.auto_cap);
    const std::string why = on && free_wanted ? free_refusal(chain.running.motion_engine, limit_hz > 0.0,
                                                             vrr_armed || known_period_us > 0)
                                              : std::string();
    const bool free = on && free_wanted && why.empty();
    if (why != free_note) {
      free_note = why;
      // Rapide est le défaut : là où il ne tient pas, Stable le remplace sans message à l'écran.
      if (!why.empty()) log->info("Mode Rapide indisponible ({}) : mode Stable", why);
    }
    // B12-02, B12-07 : les paliers calés et la grille des vsyncs sur la fréquence mesurée, celle que la
    // session suit, et non l'annoncée (164,78 Hz pour 165 : le facteur exact ne se calait plus).
    const double lock_hz = session.display_hz() > 0.0 ? session.display_hz() : known_hz;
    chain.live.set_auto(on && !free, core::auto_limits(lock_hz, fps_of(video.frame_rate), vrr_armed,
                                                       chain.running.rife_factor, cap_hz));
    const std::int64_t refresh_us =
        session.display_period_us() > 0 ? session.display_period_us() : known_period_us;
    // Écart minimal : la période de l'écran ou du plafond, à 1 % près ; à fréquence fixe, sur les vsyncs.
    const bool vsync = free && !vrr_armed;
    chain.live.set_free(free, limit_hz > 0.0 ? std::llround(1e6 / (vsync ? limit_hz : limit_hz * 0.99)) : 0,
                        host_now_us());
    chain.live.set_steady(free && p.options.fast_steady,
                          {.vrr = vrr_armed, .hz = vrr_armed ? limit_hz : known_hz, .cap_hz = cap_hz},
                          fps_of(video.frame_rate));
    chain.live.set_vsync(vsync ? refresh_us : 0, session.visible_delay_us());
  };
  apply_auto();
  // Le dernier palier tenu pour cette vidéo, ce moteur, cette hauteur de travail et ce GPU : le départ.
  std::filesystem::path auto_memory_path;
  std::string auto_memory_key;
  if (!p.settings_path.empty() && !p.options.file.empty()) {
    auto_memory_path = paths::auto_memory(p.settings_path.parent_path());
    auto_memory_key =
        // B32-06 : la variante de SPAN chargée (FP16, FP8, INT8) entre dans la clé : un palier appris sur
        // l'une ne vaut pas pour l'autre.
        core::auto_memory_key(prepare::utf8_path(p.options.file), file_bytes,
                              std::format("{} {}", c.motion_name(), span_variant), settings.work_height,
                              gpu_facts(p.screen).name);
    if (chain.live.auto_active()) {
      if (const auto remembered = core::load_auto_step(auto_memory_path, auto_memory_key)) {
        chain.live.auto_seed(*remembered, host_now_us());
        log->info("auto : départ mémorisé ×{}", auto_factor_text(*remembered));
      }
    }
  }
  // VRR armé : l'image tenue se représente à intervalles égaux au-dessus du plancher de l'écran (EDID 0xFD).
  const auto apply_floor = [&] {
    const int min_hz = p.display ? p.display->vrr_floor_hz() : 0;
    session.set_vrr_floor(vrr_armed ? core::vrr_floor_us(min_hz) : 0);
  };
  apply_floor();
  DisplayFollow follow(p, chain, session, screen_state, fps_of(video.frame_rate),
                       {.apply_auto = apply_auto, .apply_floor = apply_floor}, log, &LoopWake::push, &wake);
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
    // B32-06 : la préparation du FP8 de SPAN charge le GPU pendant la lecture ; un palier tenu sous elle ne
    // dit rien de la charge de la vidéo, il n'est pas retenu.
    if (todo.store && p.span_job.note && p.span_job.note->pending)
      log->info("auto : palier ×{} non retenu, préparation du mode Performance en cours",
                auto_factor_text(*todo.store));
    else if (todo.store)
      writer.post("auto_memory", [store = auto_memory_path, key = auto_memory_key, centi = *todo.store] {
        return core::store_auto_step(store, key, centi);
      });
  };
  SleepWatch sleep_watch;
  SleepNotice sleep_notice(&LoopWake::push, &wake);  // B31-01 : la veille dite avant
  bool sleep_announced = false;
  const WindowEvents window_events;  // B31-13 : la barre de titre tenue sous Windows n'est pas une veille
  ScreenSaverHold screen_saver;      // coupé seulement pendant qu'une vidéo avance (A33-02)
  while (!session.finished() && !chain.reopen) {
    if (sleep_notice.take()) {
      // Pause avant la veille : le son s'arrête, le GPU finit ce qu'il a en vol, puis la veille suit.
      session.command(cmd::Pause{});
      out.flush();
      sleep_notice.ready();
      sleep_announced = true;
    }
    if (AfterGap after = after_gap(session, c, p.ui, sleep_watch, window_events, sleep_announced);
        after.lost || after.reopen) {
      if (after.lost) {
        failure = std::move(*after.lost);
        break;
      }
      chain.reopen = chain.running;
    }
    screen_saver.want(must_stay_awake(true, session.view().paused, session.finished()));
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
      // L'écran change (éteint, autre sortie, nouveau mode) : sa fréquence, son VRR et son état sont relus,
      // hors du thread qui présente (display_follow.h).
      follow.poll();
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
            p.ui.controls().warn(i18n::ui_tr("Instant mode: too many frames lost, back to Stable"),
                                 host_now_us());
          apply_auto();
        }
      }
      if (const auto notice = session.take_notice())
        p.ui.controls().warn(
            i18n::ui_tr_format("Playback goes on after an error: {}", visible_error(*notice)), host_now_us());
      free_missing_seen = shown.missing;
      free_dropped_seen = shown.dropped;
      chain.live.auto_tick(host_now_us(), !session.view().paused);
      show_auto();
    }
    follow.take();  // le relevé de l'écran fini sur son fil, s'il y en a un
    grade_tick(chain, host_now_us());
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) session.handle(std::move(*message));
    if (chain.black_dirty && host_now_us() - chain.black_at_us >= kBlackRefreshUs) refresh_black(chain);
    UiFrame ui_frame = p.ui.frame(host_now_us(), session.view(), actions);
    ui_frame.changed |= snapshot_now;
    for (const Input& input : std::exchange(actions, {})) apply(input, chain, host_now_us());
    if (settings_save_due(writer, p.ui, chain.dirty, saved_at, host_now_us())) {  // B31-08, B36-01
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
    // La phase des vsyncs ne se nourrit que d'un vsync vu (acquisition qui a attendu l'écran), jamais de
    // l'heure de la présentation, que la boucle a elle-même choisie (A12-03).
    if (*record && (*record)->fresh)
      chain.live.vsync_note((*record)->vsync_us, out.vblank_grid(), (*record)->display_gpu_us);
    if (*record && stats_sink) stats_sink->add(**record, host_now_us());
    // Rien de présenté (rien n'a changé, ou fenêtre cachée) : la FIFO n'impose pas le rythme. En
    // lecture, la boucle dort jusqu'à l'échéance de l'image suivante (Session::idle_wait_us) ; en pause,
    // au plus kPausedWaitMs. Un message des autres threads (une image, un seek) la réveille.
    if (!*record) {
      const std::int64_t wait_us =
          session.view().paused ? std::int64_t{kPausedWaitMs} * 1000 : session.idle_wait_us(host_now_us());
      wait_loop(wait_us, p.options.precise_wait);  // --precise-wait : à la microseconde (6.7)
    }
  }
  screen_saver.want(false);
  if (chain.dirty) {
    writer.post("play.toml",
                [path = p.settings_path, saved = saved_settings()] { return save_settings(path, saved); });
  }
  remember();
  stats_sink.reset();
  finish_settings_writes(writer, p.settings_path, saved_settings());  // attente bornée (B30-05)
  p.quit = session.quit_asked();
  log_summary(session);
  if (telemetry) {
    // Énergie de la lecture : le compteur NVML relu au début et à la fin, à l'instant même (B11-14).
    const std::uint64_t energy_end_mj = telemetry->energy_mj(host_now_us());
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
                                   .span_variant = span_variant,
                                   .engine = c.backend_name});
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
  // B13-10, B29-08, B31-12 : l'attente du fichier qui grandit et le parcours du dossier se font hors du
  // thread de la fenêtre, pompée, et bornés (un NAS lent ou muet ne la gèle plus).
  if (!chain.reopen && !failure && !session.quit_asked() &&
      run_pumped_for<bool>("growing-file", kFolderIoLimit, true, [file = p.options.file, opened_file] {
        return opened_file.grew(file);
      }).value_or(false)) {
    log->info("le fichier grandit encore : relu au même endroit");
    chain.reopen = chain.running;  // A29-03 : comme une réouverture de la chaîne, pistes gardées
  }
  // Fin de la vidéo, sans demande de quitter : l'épisode suivant du dossier, si le réglage le veut.
  if (!chain.reopen && !failure && !session.quit_asked() && !p.options.bench_out &&
      chain.running.autoplay_next && !p.options.file.empty()) {
    if (const auto next = run_pumped_for<std::optional<std::filesystem::path>>(
                              "next-episode", kFolderIoLimit, true,
                              [file = p.options.file] { return sibling_episode(file, 1); })
                              .value_or(std::nullopt)) {
      log->info("épisode suivant : {}", prepare::utf8_path(next->filename()));
      chain.next_file = *next;
      chain.reopen = chain.running;
    }
  }
  // B31-02 : un refus de l'étage GPU pendant la lecture (enqueue, mémoire) retire un étage ; la chaîne se
  // rouvre au même endroit sans lui, avec l'avis et « Réessayer ». Un appareil perdu reste une perte.
  if (failure && failure->key == kFatalStageKey && failure->code != RefusalCode::DeviceLost &&
      c.backend->kernels()->health() && retreat(p, c, *failure)) {
    failure.reset();
    chain.reopen = chain.running;
    chain.next_file.reset();
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
                    .next_paused = chain.next_paused,
                    .rederive = chain.rederive};
    reopen->settings.volume = p.ui.controls().volume();
    p.kept = chain.kept;  // la chaîne rouverte garde les raccourcis de séance (A08-04)
    p.audio_track = (*pipeline)->tracks().audio_selected;
    p.subtitle_track = (*pipeline)->tracks().subtitle_selected;
  }
  // Ordre d'arrêt : slots, threads, moteur, backend (imports du ring), puis le ring (ring_release).
  if (const std::int64_t at = session.view().position_us; at != kNoPts)
    p.stopped_at_us = std::max<std::int64_t>(at, 0);
  // B31-04, B30-07 : ce démontage attend le GPU sans délai (jointure des fils, cudaStreamSynchronize,
  // cudaFree) ; gardé comme celui de l'écran. Après une perte du GPU, la relance est demandée d'abord : un
  // démontage bloqué la fait au lieu de laisser une fenêtre figée.
  if (failure && failure->code == RefusalCode::DeviceLost && p.relaunch_on_loss)
    (void)ask_relaunch(p.paths, p.options.file, p.stopped_at_us, false, RelaunchReason::GpuLost);
  const CloseWatch teardown(kTeardownLimit, [] { exit_stuck("fermeture de la chaîne bloquée depuis 60 s"); });
  session.end();
  out.flush();
  chain_stats.stop();  // son relevé lit les noyaux du backend, détruits par prepared->reset() (A08-02)
  pipeline->reset();
  // A31-16 : un refus fatal vu d'abord ailleurs (NVDEC au réveil, l'écran) sur un appareil mort : le backend
  // le dit, et la lecture se traite comme une perte du GPU (message, relance).
  if (failure && failure->code != RefusalCode::DeviceLost) {
    if (auto healthy = c.backend->kernels()->health();
        !healthy && healthy.error().code == RefusalCode::DeviceLost) {
      log->warn("{} : l'appareil est perdu ({})", failure->detail, healthy.error().detail);
      failure = std::move(healthy.error());
    }
  }
  keep_gpu_profile(c.backend->kernels()->gpu_profile(), options.stats, p.options.bench_out);  // 0.2
  prepared->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return reopen;
}

}  // namespace ajp::app::play
