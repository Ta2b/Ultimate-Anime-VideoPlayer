#include "app/play/chain_setup.h"

#include <algorithm>
#include <format>
#include <span>
#include <utility>

#include "app/play/drba_model.h"
#include "app/play/first_run.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/host_facts.h"
#include "app/play/motion_model.h"
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

// Hauteur où RIFE estime le mouvement quand la chaîne travaille plus haut (sortie 4K).
constexpr int kRifeMotionHeight = 720;

}  // namespace

std::vector<std::filesystem::path> gmfss_dirs(const PlayerPaths& paths,
                                              const std::optional<ModelChoice>& rife) {
  std::vector<std::filesystem::path> dirs;
  if (rife) dirs.push_back(rife->path.parent_path());
  for (const std::filesystem::path& candidate : default_rife_candidates(paths.home, paths.models)) {
    if (std::ranges::find(dirs, candidate.parent_path()) == dirs.end())
      dirs.push_back(candidate.parent_path());
  }
  return dirs;
}

std::vector<media::HwApi> decoder_attempts(std::string_view backend, bool shared_vulkan, bool nvidia_gpu) {
  if (backend == "tensorrt") return {media::HwApi::Cuda, media::HwApi::Software};
  std::vector<media::HwApi> attempts;
#if defined(__APPLE__)
  // Le bloc média de la puce : MoltenVK n'offre pas Vulkan Video.
  attempts.push_back(media::HwApi::VideoToolbox);
#endif
  if (shared_vulkan) attempts.push_back(media::HwApi::Vulkan);
#if defined(__linux__)
  if (!nvidia_gpu) attempts.push_back(media::HwApi::Vaapi);
#endif
  attempts.push_back(media::HwApi::Software);
  return attempts;
}

media::DecoderOptions decoder_options(std::string_view backend, const GpuDevice& render,
                                      const media::HwDeviceSource* shared) {
  const bool tensorrt = backend == "tensorrt";
  return {.attempts = decoder_attempts(backend, shared != nullptr, render.vendor() == GpuVendor::Nvidia),
          .render_device = render.uuid,
          .devices = shared,
          .download = !tensorrt,
          .direct_surfaces = tensorrt ? kHeldDecodedFrames : 0};
}

std::string_view ChainEngines::motion_name() const {
  return use_gmfss() ? "GMFSS" : use_drba() ? "DistilDRBA" : use_apple() ? "Apple" : "RIFE";
}

std::string ChainEngines::motion_label(const PlaySettings& settings) const {
  if (use_gmfss())
    return std::format("GMFSS Fortuna{} {} · ", gmfss_variant == GmfssVariant::Union ? " Union" : "",
                       factor_text(factor_of(settings)));
  if (use_drba()) return std::format("DistilDRBA {} · ", factor_text(factor_of(settings)));
  if (use_apple()) return std::format("Apple {} · VideoToolbox · ", factor_text(factor_of(settings)));
  if ((rife || rife_motion) && rife_model)
    return std::format("RIFE {} · {} · ", factor_text(factor_of(settings)), rife_model->stem().string());
  return {};
}

std::string ChainEngines::span_label() const {
  if (!model) return std::format("sans agrandissement · {}", reduce);
  return std::format("SPAN ×{} · {}{} · {} · {} → {}x{}", factor, model->path.stem().string(),
                     span_int8 ? " INT8" : "", backend_name, reduce, work.width * factor,
                     work.height * factor);
}

UpscaleSetup ChainEngines::upscale(const PlaySettings& settings, std::shared_ptr<LiveSettings> live,
                                   const MasterClock* clock, int width, int height,
                                   std::vector<RingSlotDesc> slots) const {
  const WorkSize sized = work_size(width, height, settings.work_height);
  return UpscaleSetup{.backend = backend.get(),
                      .engine = engine.get(),
                      .rife = rife.get(),
                      .rife_motion = rife_motion.get(),
                      .rife_planes = rife_planes,
                      .rife_align = rife_align,
                      .gmfss_pair = gmfss.pair.get(),
                      .gmfss_fusion = gmfss.fusion.get(),
                      .gmfss_align = gmfss.align,
                      .distil_drba = drba.get(),
                      .drba_align = drba_align,
                      .rife_factor = factor_of(settings),
                      .true_hdr = settings.true_hdr,
                      .one_lane = one_lane,
                      .live = std::move(live),
                      .clock = clock,
                      .source_width = width,
                      .source_height = height,
                      .work_width = sized.width,
                      .work_height = sized.height,
                      .rife_width = rife_motion ? rife_size.width : 0,
                      .rife_height = rife_motion ? rife_size.height : 0,
                      .warp_after_span = warp_after_span(settings.span_order, sized.height * kSpanFactor),
                      .slots = std::move(slots)};
}

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings) {
  const auto log = log_channel("app");
  const media::TrackInfo& video = p.video;
  ChainEngines c;
  c.one_lane = p.options.one_lane;
  if (settings.upscale) {
    auto chosen = choose_model(p.options, p.paths.home, settings, p.paths.models);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    c.model = std::move(*chosen);
  }
  c.factor = c.model ? kSpanFactor : 1;
  c.work = work_size(video.width, video.height, settings.work_height);
  const bool reduced = c.work != WorkSize{video.width, video.height};
  c.reduce = reduced ? std::format("{}x{} → {} {}x{}", video.width, video.height, label(settings.downscale),
                                   c.work.width, c.work.height)
                     : std::format("{}x{}", video.width, video.height);
  const GpuFacts gpu = gpu_facts(p.screen);
  log->info("GPU : {}", gpu_line(gpu));
  if (const std::string rebar = rebar_text(gpu); rebar.rfind("inactif", 0) == 0)
    log->warn("Resizable BAR inactif : activer Above 4G Decoding et Resizable BAR dans le BIOS");
  if (p.screen.device().vendor() == GpuVendor::Nvidia) {
    const std::string& driver = p.screen.device().driver_version;
    if (settings.true_hdr && nvidia_driver_older_than(driver, kRtxHdrDriverMajor))
      log->warn("pilote NVIDIA {} : RTX HDR demande {} ou plus récent", driver, kRtxHdrDriverMajor);
    if (settings.smooth_motion && nvidia_driver_older_than(driver, 580, 82, 7))
      log->warn("pilote NVIDIA {} : Smooth Motion demande 580.82.07 ou plus récent", driver);
  }
  const std::span<const std::string_view> rife_order = rife_preference(gpu);
  if (auto found = choose_rife_model(p.options, p.paths.home, settings, rife_order, p.paths.models))
    c.rife_choice = std::move(*found);
  if (c.rife_choice) c.rife_model = c.rife_choice->path;
  c.gmfss_places = gmfss_dirs(p.paths, c.rife_choice);
  c.gmfss_variant =
      settings.motion_engine == MotionEngine::GmfssUnion ? GmfssVariant::Union : GmfssVariant::Base;
  c.gmfss_found = find_gmfss(c.gmfss_places, c.gmfss_variant).has_value();
  const bool want_gmfss = settings.rife_factor >= 2 && is_gmfss(settings.motion_engine);
  c.drba_found = find_distil_drba(c.gmfss_places).has_value();
  const bool want_drba = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::DistilDrba;
  const bool want_apple = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::Apple;
  if (settings.rife_factor >= 2 && !c.rife_model && !(want_gmfss && c.gmfss_found) &&
      !(want_drba && c.drba_found) && !want_apple)
    return std::unexpected(
        choose_rife_model(p.options, p.paths.home, settings, rife_order, p.paths.models).error());
  show_loading(p.screen, p.ui, "Chargement");
  const gpu::BackendInit init{
      .render = &p.screen.device(), .cache_dir = p.paths.engines, .log = backend_log()};
  auto backend = BackendLibrary::create(p.library, init);
  if (!backend) return std::unexpected(std::move(backend.error()));
  c.backend = std::move(*backend);
  c.backend_name = c.backend->capabilities().name;
  log->info("backend {} ({}) depuis {}", c.backend_name, c.backend->capabilities().device_name,
            p.library->path().string());

  // SPAN INT8 sous TensorRT quand tools/span/quantize_span.py en a laissé une variante qui passe ses
  // garde-fous (span_int8.h) ; le SPAN choisi sinon, et aussi si TensorRT refuse la variante.
  if (c.model && c.backend->capabilities().int8) {
    SpanInt8Lookup int8 = find_span_int8(c.model->path, p.paths.engines.parent_path() / "modeles");
    if (!int8.note.empty()) log->info("SPAN INT8 : {}", int8.note);
    c.span_int8 = std::move(int8.variant);
  }
  if (c.model) {
    const auto prepare_span = [&] {
      return prepare_engine(p, init, *c.backend, c.span_int8 ? c.span_int8->onnx : c.model->path,
                            gpu::Shape{.dims = {1, 3, c.work.height, c.work.width}},
                            EngineLabel{.name = c.span_int8 ? "SPAN INT8" : "SPAN",
                                        .model = c.model->path.stem().string(),
                                        .width = c.work.width,
                                        .height = c.work.height},
                            gpu::Lane::Upscale);
    };
    auto loaded = prepare_span();
    if (!loaded && c.span_int8) {
      log->warn("SPAN INT8 refusé ({}) : SPAN FP16", loaded.error().detail);
      c.span_int8.reset();
      loaded = prepare_span();
    }
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    c.engine = std::move(**loaded);
  }
  // Au-delà de 720 lignes de travail (la 4K sans réduction), RIFE estime le mouvement sur une copie
  // réduite à 720p et le warp déforme les images de travail : la variante mouvement seulement.
  c.rife_size = !p.options.rife_image && c.work.height > kRifeMotionHeight
                    ? work_size(video.width, video.height, kRifeMotionHeight)
                    : c.work;
  // GMFSS Fortuna à la place de RIFE, à la taille de travail ; RIFE reprend si ses graphes manquent.
  if (want_gmfss) {
    auto loaded =
        prepare_gmfss(p, init, *c.backend, c.gmfss_places, c.gmfss_variant, c.work.width, c.work.height);
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(c.rife_model ? "GMFSS indisponible : RIFE" : "GMFSS indisponible",
                             host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      c.gmfss = std::move(**loaded);
    }
  }
  // DistilDRBA à la place de RIFE, à la taille de travail ; RIFE reprend si son graphe manque.
  if (want_drba) {
    auto loaded =
        prepare_distil_drba(p, init, *c.backend, c.gmfss_places, c.work.width, c.work.height, c.drba_align);
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(c.rife_model ? "DistilDRBA indisponible : RIFE" : "DistilDRBA indisponible",
                             host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      c.drba = std::move(**loaded);
    }
  }
  // L'interpolation d'Apple à la place de RIFE image, à la taille de travail ; RIFE reprend sous un
  // autre backend que Core ML, ou sur un Mac qui ne l'offre pas.
  if (want_apple) {
    auto loaded = prepare_apple(p, init, *c.backend, c.work.width, c.work.height);
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(
          c.rife_model ? "Interpolation d'Apple indisponible : RIFE" : "Interpolation d'Apple indisponible",
          host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      c.rife = std::move(**loaded);
      c.apple = true;
      c.rife_planes = kernels::kRifeV2InputChannels;
    }
  }
  const bool use_rife =
      settings.rife_factor >= 2 && c.rife_model && !c.use_gmfss() && !c.use_drba() && !c.use_apple();
  c.rife_align = use_rife ? rife_padding_multiple(*c.rife_model) : kernels::kRifeAlign;
  c.geometry = kernels::rife_geometry(c.rife_size.width, c.rife_size.height, c.rife_align);
  c.image_geometry = kernels::rife_geometry(c.work.width, c.work.height, c.rife_align);
  if (use_rife) {
    auto planes = rife_input_planes(*c.rife_model);
    if (!planes) return std::unexpected(std::move(planes.error()));
    c.rife_planes = *planes;
  }
  // RIFE par son mouvement : une passe au milieu de la paire la sert toute quand le mouvement est
  // faible ; sans variante dérivable, RIFE image comme avant.
  if (use_rife && !p.options.rife_image) {
    auto derived = motion_model(*c.rife_model, p.paths.engines.parent_path() / "modeles");
    if (!derived) {
      log->warn("RIFE mouvement indisponible ({}) : RIFE image", derived.error().detail);
    } else {
      const kernels::RifeGeometry& g = c.geometry;
      auto loaded = prepare_engine(p, init, *c.backend, *derived,
                                   gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                                   EngineLabel{.name = "RIFE",
                                               .model = derived->stem().string(),
                                               .width = g.padded_width,
                                               .height = g.padded_height},
                                   gpu::Lane::Interpolate);
      if (!loaded) return std::unexpected(std::move(loaded.error()));
      if (!*loaded) return std::nullopt;
      c.rife_motion = std::move(**loaded);
    }
  }
  if (use_rife && !c.rife_motion) {
    const kernels::RifeGeometry& g = c.image_geometry;
    auto loaded = prepare_engine(p, init, *c.backend, *c.rife_model,
                                 gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                                 EngineLabel{.name = "RIFE",
                                             .model = c.rife_model->stem().string(),
                                             .width = g.padded_width,
                                             .height = g.padded_height},
                                 gpu::Lane::Interpolate);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    c.rife = std::move(**loaded);
  }
  if (c.rife_motion) {
    log->info("RIFE mouvement : estimé en {}x{}, warp en {}x{}", c.rife_size.width, c.rife_size.height,
              c.work.width, c.work.height);
  }
  return std::optional<ChainEngines>(std::move(c));
}

}  // namespace ajp::app::play
