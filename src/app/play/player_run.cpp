#include "app/play/player_run.h"

#include <algorithm>
#include <array>
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
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/default_player.h"
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
#include "app/play/gpu_telemetry.h"
#include "app/play/home_screen.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/model_choice.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player_engines.h"
#include "app/play/profiles.h"
#include "app/play/resume.h"
#include "app/play/scripted_input.h"
#include "app/play/session.h"
#include "app/play/span_auto.h"
#include "app/play/span_derive.h"
#include "app/play/span_int8.h"
#include "app/play/ui.h"
#include "app/play/upscale_stage.h"
#include "audio/devices.h"
#include "core/auto_memory.h"
#include "core/auto_rate.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kRingSlots = 4;
static_assert(kRingSlots <= kMaxKeptSubtitles, "les sous-titres gardés doivent couvrir tout le ring");

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
  // Le mode HDR demandé, avant tout repli de l'environnement : c'est lui que les profils gardent.
  const HdrIntent wanted_hdr{.true_hdr = settings.true_hdr, .hdr_native = settings.hdr_native};
  const PlayOptions options = options_for(p.options, settings);
  const media::TrackInfo& video = p.video;
  auto prepared = prepare_chain(p, settings);
  if (!prepared) return std::unexpected(std::move(prepared.error()));
  if (!*prepared) {  // q ou la fenêtre fermée pendant la préparation des moteurs
    p.quit = true;
    return std::nullopt;
  }
  ChainEngines& c = **prepared;
  const int factor = c.factor;
  const WorkSize work = c.work;
  const bool rife_loaded = c.interpolates();

  settings.stats.all = p.details;  // Tab pendant la préparation

  const int auto_peak = vendor_hdr::peak_from_display(content_peak_nits(p.screen.window()));
  int peak_nits = 0;
  // La sortie est HDR10 : l'écran en reçoit, et --sdr ne l'a pas coupée. Sinon (sortie SDR),
  // TrueHDR ferait du HDR que libplacebo ramènerait aussitôt en SDR ; le grade SDR prend sa place.
  const bool hdr_out = !p.options.sdr && content_peak_nits(p.screen.window()).has_value();
  // Un écran SDR n'a pas de HDR à offrir : ni RTX HDR ni HDR adaptatif (le panneau les grise) ; l'image reste
  // en SDR, et le grade (contraste, saturation, préréglage) s'y applique.
  if (!hdr_out && (settings.true_hdr || settings.hdr_native)) {
    log->info("sortie SDR : ni RTX HDR ni HDR adaptatif, image SDR gradée");
    settings.true_hdr = false;
    settings.hdr_native = false;
  }
  // La couleur de la source telle que le flux l'étiquette : l'écran convertit depuis elle (gamut, courbe,
  // HDR10 ou HLG), RTX HDR s'ouvre pour elle.
  const std::optional<ColorInfo> source_color = media::track_color(video);
  const ColorInfo source = source_color.value_or(ColorInfo{});
  const bool source_hdr = source.transfer == Transfer::Pq || source.transfer == Transfer::Hlg;
  if (settings.true_hdr && source_hdr) {
    log->info("RTX HDR : la source est déjà HDR, TrueHDR n'est pas ouvert");
    p.ui.controls().notify(i18n::tr("Source is already HDR: RTX HDR not needed"), host_now_us());
    settings.true_hdr = false;
  }
  // TrueHDR s'ouvre par le moteur TensorRT (open_true_hdr n'existe que là) : sous Vulkan, la route
  // native prend sa place sans essayer, et le panneau ne propose plus RTX HDR (true_hdr_available).
  if (settings.true_hdr && c.backend_name != "tensorrt") {
    log->info("RTX HDR : le moteur {} ne l'ouvre pas (TensorRT requis) ; HDR adaptatif à la place",
              c.backend_name);
    p.ui.controls().notify(i18n::tr("RTX HDR needs TensorRT: using adaptive HDR"), host_now_us());
    settings.true_hdr = false;
    settings.hdr_native = true;
  }
  bool rtx_hdr_failed = false;  // TrueHDR a refusé de s'ouvrir : le panneau ne le propose plus
  if (settings.true_hdr) {
    show_loading(p.screen, p.ui, i18n::tr("Preparing RTX HDR"));
    auto opened = open_rtx_hdr(p.options.true_hdr_sdk, p.paths, settings, auto_peak, p.screen,
                               *c.backend->kernels(), work.width * factor, work.height * factor, source);
    if (opened) {
      peak_nits = *opened;
    } else {
      // Le plan : TrueHDR qui échoue laisse la place à la route native, et l'écran le dit.
      log->warn("{} ; HDR adaptatif à la place", opened.error().detail);
      p.ui.controls().notify(i18n::tr("RTX HDR unavailable: using adaptive HDR"), host_now_us());
      settings.true_hdr = false;
      settings.hdr_native = true;
      rtx_hdr_failed = true;
      p.screen.set_source_color(source);
    }
  } else {
    p.screen.set_source_color(source);
  }

  const std::string label_text = std::format(
      "{}{}{}", c.motion_label(settings),
      // Le moteur HDR vraiment pris : TrueHDR ouvert, ou le HDR adaptatif sur une sortie HDR10.
      settings.true_hdr ? "RTX HDR · " : (settings.hdr_native && hdr_out ? "HDR adaptatif · " : ""),
      c.span_label());
  log->info("{} : {}", options.file.string(), label_text);
  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(rife_loaded ? factor_of(settings) : Factor{});
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  PipelineConfig config{
      .file = options.file,
      .decoder = decoder_options(c.backend_name, p.screen.device(), p.screen.shared_devices()),
      .languages = {.presets = settings.language_presets},
      .audio_device = settings.audio_device,
      .mix = {},
      .audio_track = p.audio_track,
      .subtitle_track = p.subtitle_track};
  {
    // L'écoute de cet appareil : son profil propre, sinon le global (settings.h : effective_listening).
    const auto device = audio::resolve_device(audio::playback_devices(), settings.audio_device);
    const EffectiveListening listening = effective_listening(settings, device ? device->name : std::string());
    config.mix.listen = static_cast<audio::Listen>(listening.listen);
    config.mix.faithful = listening.faithful;
    config.mix.tuning = listening.tuning;
    config.mix.anime = listening.anime;
    if (auto loaded = load_listening_hrir(listening.hrir_path)) {
      config.mix.hrir = std::move(*loaded);
    } else {
      log->warn("{} ; modèle paramétrique à la place", loaded.error().detail);
    }
  }
  PlaceboScreen& out = p.screen;
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = work_size(width, height, settings.work_height);
        auto slots = out.create_ring(sized.width * factor, sized.height * factor, kRingSlots,
                                     c.backend_name == "tensorrt");
        if (!slots) return std::unexpected(std::move(slots.error()));
        const MasterClock* clock = stage.clock;
        return UpscaleStage::start(std::move(stage),
                                   c.upscale(settings, live, clock, width, height, std::move(*slots)));
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));

  const std::unique_ptr<GpuTelemetry> telemetry = GpuTelemetry::open(p.screen.device());
  const ChainStats chain_stats(*live, *c.backend->kernels(), rife_loaded, std::string(c.motion_name()),
                               fps_of(video.frame_rate), telemetry.get());
  SessionConfig session_config = (*pipeline)->session_config();
  session_config.start_us = start_us;
  p.screen.set_sample_aspect(session_config.video_sar);
  session_config.stop_after_us = options.stop_after_us;
  session_config.fullscreen_after_us = options.fullscreen_after_us;
  session_config.header = {label_text};
  session_config.tracks = (*pipeline)->describe();
  p.screen.set_grade(grade_of(settings, settings.true_hdr, peak_nits, auto_peak, {}, {}, source_hdr));
  p.screen.set_finish(settings.final_scaler, settings.deband);
  session_config.output = output_text(out, settings, peak_nits);
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
    live->auto_written(frame.left.value, frame.right.value, frame.submitted_us, done_us);
  };
  log->info("{}", label_text);
  for (const std::string& line : session_config.tracks) log->info("{}", line);
  log->info("{}", session_config.output);
  Session session(std::move(session_config), (*pipeline)->stages(), out, (*pipeline)->clock());
  if (start_paused) session.command(cmd::Pause{});
  // La tranche de la machine pour Général > Configuration : la mesure de « Évaluer ma machine » si elle est
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
  const bool default_player_is_ajp =
      is_default_player({.executable = p.paths.executable, .home = p.paths.home, .data = p.paths.data});
  const bool fp8_engine = c.backend && c.backend->capabilities().fp8;
  // Le mode Performance de SPAN : ce que le moteur annonce, et sinon pourquoi (moteur absent ou autre carte).
  const SpanFp8Facts fp8_facts{.backend_name = c.backend_name,
                               .gpu_name = gpu_facts(p.screen).name,
                               .tensorrt_present = tensorrt_backend_present(p.paths),
                               .capability = c.backend && c.backend->capabilities().span_fp8_cutlass};
  const std::string fp8_reason = span_fp8_unavailable_reason(fp8_facts);
  log_channel("app")->info("SPAN Performance (FP8) : {} (moteur {}, capacite CUTLASS {})",
                           fp8_reason.empty() ? "disponible" : fp8_reason, c.backend_name,
                           fp8_facts.capability);
  const SpanPrepare span_prepare =
      c.model ? make_span_prepare(
                    p.library,
                    gpu::BackendInit{
                        .render = &p.screen.device(), .cache_dir = p.paths.engines, .log = backend_log()},
                    fp8_facts.capability, c.model->path, p.paths.engines.parent_path() / "modeles",
                    p.options.file, fp8_reason)
              : SpanPrepare{};
  const std::optional<DrbaModel> drba_file = find_distil_drba(c.gmfss_places);
  p.ui.set_tracks((*pipeline)->tracks());
  p.ui.set_settings(
      settings,
      SettingsChoices{
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
          .rife_available =
              c.rife_model.has_value() || c.gmfss_found || c.drba_found || offered(MotionEngine::Apple),
          // Seul TensorRT sait importer les graphes de GMFSS (MatMul, Softmax, LayerNorm) : le
          // moteur Vulkan les refuserait, le menu ne les propose donc pas.
          .gmfss_available = c.backend_name == "tensorrt" && find_gmfss(c.gmfss_places).has_value(),
          .gmfss_union_available =
              c.backend_name == "tensorrt" && find_gmfss(c.gmfss_places, GmfssVariant::Union).has_value(),
          .gmfss_tiers = c.backend_name == "tensorrt"
                             ? gmfss_tiers(c.gmfss_places, GmfssVariant::Base, fp8_engine)
                             : TierSupport{},
          .gmfss_union_tiers = c.backend_name == "tensorrt"
                                   ? gmfss_tiers(c.gmfss_places, GmfssVariant::Union, fp8_engine)
                                   : TierSupport{},
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
          .true_hdr_available = c.backend_name == "tensorrt" && !rtx_hdr_failed &&
                                (maxine_available(p.options, p.paths) || settings.true_hdr),
          .true_hdr_backend = c.backend_name == "tensorrt",
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
          .gpu_name = gpu_facts(p.screen).name,
          .audio_outputs = audio_outputs,
          .engines_dir_default = prepare::utf8_path(p.paths.engines.parent_path() / "backends"),
          .engines_dir_free = format_size(free_bytes(p.catalog.layout.engines)),
          .default_player_is_ajp = default_player_is_ajp,
          .offer_default_player = !default_player_is_ajp && !settings.default_player_dismissed,
          .bench_text = bench_text,
          .place = {.executable = p.paths.executable, .home = p.paths.home, .data = p.paths.data},
          .span_prepare = span_prepare,
          .fp8_reason = fp8_reason,
          .span_job = p.span_job.note,
          .settings_dir = p.settings_path.parent_path()});
  p.profiles.source =
      core::SourceInfo{.width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};
  if (c.model) p.profiles.default_model = c.model->default_name;
  p.ui.set_profiles(view_of(p.profiles));
  Chain chain{
      .session = session,
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
      .file = p.options.file,
      .display = p.display,
      .header_text =
          [&c, hdr_out](const PlaySettings& s) {
            return std::format(
                "{}{}{}", c.motion_label(s),
                s.true_hdr ? "RTX HDR · " : (s.hdr_native && hdr_out ? "HDR adaptatif · " : ""),
                c.span_label());
          },
      .hdr_wanted = settings.true_hdr != wanted_hdr.true_hdr || settings.hdr_native != wanted_hdr.hdr_native
                        ? std::optional(wanted_hdr)
                        : std::nullopt,
      .source_hdr = source_hdr};
  apply_oled_care(chain, settings);

  std::ofstream stats;
  if (options.stats) {
    stats.open(*options.stats);
    stats << "host_us\tpts_us\tfresh\trender_us\tinterval_us\tdropped\twhy\tmissing\tlag_us\n";
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
        i18n::tr_format("Resumed at {}", format_clock(start_us, session.view().duration_us)), host_now_us());
    p.resumed = false;
  }
  if (!p.notice.empty()) {
    p.ui.controls().notify(p.notice, host_now_us());
    p.notice.clear();
  }
  // La position retenue toutes les kResumeEveryUs, et à la fin : un arrêt brutal en perd peu.
  std::int64_t kept_us = kNoPts;
  const auto remember = [&] {
    if (p.resume_path.empty()) return;
    const ControlsView view = session.view();
    if (view.position_us == kNoPts || view.position_us == kept_us) return;
    kept_us = view.position_us;
    if (auto kept = remember_position(p.resume_path, p.options.file, view.position_us, view.duration_us);
        !kept)
      log->warn("{}", kept.error().detail);
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
  bool vrr_armed = !p.options.no_vrr && p.display && p.display->vrr.value_or(false) &&
                   p.display->vrr_armed.value_or(false);
  std::string free_note;  // pourquoi le mode Rapide ne tourne pas alors qu'il est choisi (vide : il tourne ou
                          // n'est pas voulu)
  const auto apply_auto = [&] {
    const bool on = chain.running.rife_auto && chain.rife_loaded && chain.running.rife_factor >= 2;
    // Au fps près (1/24 de multiplicateur) sur un écran VRR armé : la cadence n'y est plus liée au
    // rafraîchissement. À fréquence fixe, des paliers entiers (les autres feraient vibrer).
    const int granularity = vrr_armed ? core::kFineGranularity : 1;
    const double limit_hz = core::auto_limit_hz(display_refresh_hz(out.window()), chain.running.auto_cap);
    // mode Rapide (core/free_grid.h) : écran VRR armé, RIFE, fréquence de l'écran connue ; sinon Auto.
    std::string why;
    if (on && chain.running.auto_free) {
      if (free_instants_limit(chain.running.motion_engine) == 0)
        why = i18n::tr_format("{} cannot compute at a free instant", label(chain.running.motion_engine));
      else if (limit_hz <= 0.0 || (!vrr_armed && display_period_us(out.window()) <= 0))
        why = i18n::tr("display refresh rate unknown");
    }
    const bool free = on && chain.running.auto_free && why.empty();
    if (why != free_note) {
      free_note = why;
      if (!why.empty()) {
        log->warn("Mode Rapide indisponible ({}) : retour au mode Stable", why);
        p.ui.controls().notify(i18n::tr_format("Fast mode unavailable: {}", why), host_now_us());
      }
    }
    chain.live.set_auto(on && !free, core::auto_ceiling(limit_hz, fps_of(video.frame_rate), granularity),
                        granularity);
    // Écart minimal entre deux images : la période de l'écran ou du plafond, à 1 % près (jamais plus vite).
    // Écran à fréquence fixe (sans VRR, ou VRR coupé) : les instants tombent sur les vsyncs, l'écart minimal
    // est la période de l'écran (ou du plafond, s'il est plus lent).
    const std::int64_t refresh_us = display_period_us(out.window());
    const bool vsync = free && !vrr_armed;
    chain.live.set_free(free, limit_hz > 0.0 ? std::llround(1e6 / (vsync ? limit_hz : limit_hz * 0.99)) : 0,
                        host_now_us());
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
  if (p.display)
    log->info("VRR : {} · {}", p.display->vrr.value_or(false) ? "capable" : "non capable",
              p.display->vrr_armed.value_or(false) ? "armé" : "non armé");
  int auto_shown = 0;  // centièmes de multiplicateur ; 0 : le mode auto est coupé
  int auto_notified = 0;
  std::int64_t auto_changed_us = 0;
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
    if (centi != auto_notified && at - auto_changed_us >= 3'000'000) {
      if (auto_notified > 0)
        p.ui.controls().notify(i18n::tr_format("Auto ×{} ({:.0f} fps)", auto_factor_text(centi),
                                               auto_source_fps * centi / 100.0),
                               at);
      auto_notified = centi;
    }
    if (centi != auto_stored && at - auto_changed_us >= 6'000'000 && !auto_memory_path.empty()) {
      if (auto kept = core::store_auto_step(auto_memory_path, auto_memory_key, centi); !kept)
        log->warn("{}", kept.error().detail);
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
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (auto dropped = dropped_file(event)) {  // une autre vidéo : la chaîne se ferme pour elle
        log->info("fichier glissé : {}", prepare::utf8_path(*dropped));
        chain.next_file = std::move(*dropped);
        chain.reopen = chain.running;
        continue;
      }
      if (p.ui.handle(event, host_now_us(), actions)) continue;  // saisie dans l'interface
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    // L'écran de la fenêtre peut changer (déplacement, mode d'affichage) : on relit sa cadence.
    if (host_now_us() - display_polled_at >= 500'000) {
      display_polled_at = host_now_us();
      // Le mode Performance de SPAN se prépare tout seul au cran Performance (span_auto.h) : la lecture reste
      // en FP16 pendant ce temps, puis la chaîne se rouvre pour prendre le paquet.
      if (!p.options.file.empty() && c.model) {
        const SpanAutoTick span =
            tick_span_auto(chain.running.span_precision, span_prepare, p.span_job,
                           prepare::utf8_path(c.model->path) + "|" + prepare::utf8_path(p.options.file));
        if (span.notify) p.ui.controls().notify(*span.notify, host_now_us());
        if (span.reopen) chain.reopen = chain.running;
      }
      session.set_display(display_period_us(out.window()), display_refresh_hz(out.window()));
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
          chain.running.auto_free = false;
          p.ui.set_running(chain.running);
          p.ui.controls().notify(i18n::tr("Fast mode: too many frames lost, back to Stable"), host_now_us());
          apply_auto();
        }
      }
      free_missing_seen = shown.missing;
      free_dropped_seen = shown.dropped;
      if (chain.hdr_refresh_us != 0 && host_now_us() >= chain.hdr_refresh_us) {
        chain.hdr_refresh_us = 0;
        if (session.view().paused)
          session.command(
              cmd::Seek{.target_us = std::max<std::int64_t>(session.view().position_us, 0), .exact = true});
      }
      chain.live.auto_tick(host_now_us(), !session.view().paused);
      show_auto();
      // KDE arme le VRR en plein écran seulement : on le relit quand la fenêtre change d'état.
      const bool fullscreen = (SDL_GetWindowFlags(out.window()) & SDL_WINDOW_FULLSCREEN) != 0;
      if (fullscreen != vrr_fullscreen) {
        vrr_fullscreen = fullscreen;
        if (const auto display = read_display(display_name(out.window()))) {
          p.ui.set_display_vrr(display->vrr.value_or(false), display->vrr_armed.value_or(false), fullscreen);
          const bool armed =
              !p.options.no_vrr && display->vrr.value_or(false) && display->vrr_armed.value_or(false);
          if (armed != vrr_armed) {
            vrr_armed = armed;
            apply_auto();  // les demi-paliers suivent le VRR
          }
          log->info("VRR : {} · {}", display->vrr.value_or(false) ? "capable" : "non capable",
                    display->vrr_armed.value_or(false) ? "armé" : "non armé");
        }
      }
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
      if (auto saved = save_settings(
              p.settings_path,
              settings_to_save(with_intended_hdr(chain, chain.running), p.from_file, p.started, p.options));
          !saved)
        log->warn("{}", saved.error().detail);
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
    if (*record && (*record)->fresh) chain.live.vsync_note((*record)->host_us);
    if (*record && stats.is_open()) write_record(stats, **record);
    // Rien de présenté (rien n'a changé, ou fenêtre cachée) : la FIFO n'impose pas le rythme. En
    // lecture, la boucle reprend dans 1 ms pour ne pas manquer l'image suivante ; en pause, elle
    // attend une touche ou la souris, au plus kPausedWaitMs : les messages des autres threads
    // (image d'un seek, avancée d'un export) ne réveillent pas SDL.
    if (!*record) {
      if (session.view().paused) {
        (void)SDL_WaitEventTimeout(nullptr, kPausedWaitMs);
      } else {
        SDL_Delay(1);
      }
    }
  }
  if (chain.dirty) {
    if (auto saved = save_settings(p.settings_path, settings_to_save(with_intended_hdr(chain, chain.running),
                                                                     p.from_file, p.started, p.options));
        !saved)
      log->warn("{}", saved.error().detail);
  }
  remember();
  p.quit = session.quit_asked();
  log_summary(session);
  if (p.options.bench_out) {
    const GpuFacts gpu = gpu_facts(p.screen);
    const FrameStats& seen = session.stats();
    const BenchReading reading{.gpu = gpu.name,
                               .driver = gpu.driver,
                               .run = {.level = static_cast<Tier>(p.options.bench_level.value_or(0)),
                                       .presents = seen.presents,
                                       .dropped = seen.dropped,
                                       .missing = seen.missing,
                                       .lag_p95_us = seen.lag.percentile(95),
                                       .period_us = bench_period_us},
                               .interval_median_us = seen.frame_interval.median(),
                               .interval_p95_us = seen.frame_interval.percentile(95),
                               .render_median_us = seen.render.median(),
                               .work_height = chain.running.work_height,
                               .rife_model = chain.running.rife_model};
    if (auto written = save_reading(*p.options.bench_out, reading); !written)
      log->warn("{}", written.error().detail);
  }
  for (const std::string& refusal : session.refusals()) log->warn("{}", refusal);
  // Fin de la vidéo, sans demande de quitter : l'épisode suivant du dossier, si le réglage le veut.
  if (!chain.reopen && !failure && !session.quit_asked() && !p.options.bench_out &&
      chain.running.autoplay_next && !p.options.file.empty()) {
    if (const auto next = sibling_episode(p.options.file, 1)) {
      log->info("épisode suivant : {}", prepare::utf8_path(*next));
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
                    .next_file = chain.next_file};
    reopen->settings.volume = p.ui.controls().volume();
    p.audio_track = (*pipeline)->tracks().audio_selected;
    p.subtitle_track = (*pipeline)->tracks().subtitle_selected;
  }
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, le moteur avant son backend, le
  // backend (et ses imports du ring) avant l'écran qui possède le ring.
  session.end();
  out.flush();
  pipeline->reset();
  prepared->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return reopen;
}

}  // namespace ajp::app::play
