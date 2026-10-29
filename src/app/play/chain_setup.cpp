#include "app/play/chain_setup.h"

#include <algorithm>
#include <format>
#include <span>
#include <utility>

#include "app/play/drba_fused_choice.h"
#include "app/play/drba_model.h"
#include "app/play/first_run.h"
#include "app/play/gmfss_fused_choice.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/motion_model.h"
#include "app/play/rife_fused_choice.h"
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

std::vector<media::HwApi> decoder_attempts(std::string_view backend, bool shared_vulkan,
                                           [[maybe_unused]] bool nvidia_gpu) {
  if (backend == "tensorrt") return {media::HwApi::Cuda, media::HwApi::Software};
  std::vector<media::HwApi> attempts;
#if defined(__APPLE__)
  // Le bloc média de la puce : MoltenVK n'offre pas Vulkan Video.
  attempts.push_back(media::HwApi::VideoToolbox);
#endif
  if (shared_vulkan) attempts.push_back(media::HwApi::Vulkan);
#if defined(_WIN32)
  // Le bloc vidéo du GPU par D3D11VA quand Vulkan Video ne prend pas le codec : copié en mémoire
  // hôte, bien plus léger que le décodage logiciel.
  attempts.push_back(media::HwApi::D3d11va);
#endif
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
          .render_luid = render.luid,
          .devices = shared,
          .download = !tensorrt,
          .direct_surfaces = tensorrt ? kHeldDecodedFrames : 0};
}

std::string_view ChainEngines::motion_name() const {
  return use_gmfss() ? "GMFSS" : use_drba() ? "DistilDRBA" : use_apple() ? "Apple" : "RIFE";
}

std::string ChainEngines::motion_label(const PlaySettings& settings) const {
  if (use_gmfss())
    return std::format("GMFSS Fortuna{} {}{} · ", gmfss_variant == GmfssVariant::Union ? " Union" : "",
                       factor_text(factor_of(settings)), gmfss_fused ? i18n::tr(" fused") : "");
  if (use_drba())
    return std::format("DistilDRBA {}{} · ", factor_text(factor_of(settings)),
                       drba_fused ? i18n::tr(" fused") : "");
  if (use_apple()) return std::format("Apple {} · VideoToolbox · ", factor_text(factor_of(settings)));
  if ((rife || rife_motion) && rife_model)
    return std::format("RIFE {} · {}{} · ", factor_text(factor_of(settings)), rife_model->stem().string(),
                       rife_fused ? i18n::tr(" fused") : "");
  return {};
}

std::string ChainEngines::span_label() const {
  if (!model) return i18n::tr_format("no upscaling · {}", reduce);
  return std::format("SPAN ×{} · {}{} · {} · {} → {}x{}", factor, model->path.stem().string(),
                     span_int8 ? std::format(" {}", variant_text(span_int8->kind)) : std::string(),
                     backend_name, reduce, work.width * factor, work.height * factor);
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
                      .gmfss_features = gmfss.features.get(),
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

namespace {

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain_once(Player& p,
                                                                       const PlaySettings& settings);

}  // namespace

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings) {
  auto chain = prepare_chain_once(p, settings);
  if (chain || (settings.span_precision == PrecisionTier::Balanced &&
                settings.motion_precision == PrecisionTier::Balanced &&
                settings.gmfss_precision == PrecisionTier::Balanced))
    return chain;
  // Un graphe FP8 ou FP32 que le moteur refuse : la chaîne repart au cran Équilibre (FP16), celui de
  // toujours.
  log_channel("app")->warn("chaîne refusée aux crans de précision choisis ({}) : Équilibre (FP16)",
                           chain.error().detail);
  PlaySettings balanced = settings;
  balanced.span_precision = balanced.motion_precision = balanced.gmfss_precision = PrecisionTier::Balanced;
  return prepare_chain_once(p, balanced);
}

namespace {

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain_once(Player& p,
                                                                       const PlaySettings& settings) {
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
  // Spline36 ne tient que 32 prises : au-delà d'un rapport d'environ 5,3 le filtre est tronqué.
  constexpr double kSplineMaxRatio = 5.3;
  if (reduced && settings.downscale == Downscale::Spline36 &&
      static_cast<double>(video.width) / c.work.width > kSplineMaxRatio)
    log->warn(
        "réduction Spline36 de {}x{} vers {}x{} : rapport au-delà de 5,3, filtre tronqué ; Area ou une "
        "hauteur de travail plus haute l'évite",
        video.width, video.height, c.work.width, c.work.height);
  const GpuFacts gpu = gpu_facts(p.screen);
  log->info("GPU : {}", gpu_line(gpu));
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
  // Sous TensorRT les copies passent par CUDA : le BAR ne joue que pour le moteur Vulkan.
  if (const std::string rebar = rebar_text(gpu);
      c.backend_name != "tensorrt" && rebar.rfind("inactif", 0) == 0)
    log->warn("Resizable BAR inactif : activer Above 4G Decoding et Resizable BAR dans le BIOS");

  // La variante de SPAN du cran (precision.h) quand tools/span/quantize_span.py ou derive_span_fp8.py en ont
  // laissé une qui passe ses garde-fous (span_int8.h) : Performance prend le SPAN entier en noyaux CUTLASS
  // FP8 (moteur qui l'annonce : RTX 40), puis le FP8 par Q/DQ (moteur qui l'accélère), puis l'INT8 ;
  // Équilibre l'INT8, sinon le FP16 tel que livré ; Qualité le FP32. Chaque variante refusée au chargement
  // (autre GPU, taille que les noyaux ne prennent pas, erreur CUDA, TensorRT qui la refuse) laisse la place à
  // la suivante, et la dernière au SPAN choisi, en FP16 sous TensorRT.
  if (c.model) {
    const std::filesystem::path variants = p.paths.engines.parent_path() / "modeles";
    const gpu::BackendCaps caps = c.backend->capabilities();
    const PrecisionTier tier = settings.span_precision;
    std::vector<SpanVariantKind> wanted;
    if (tier == PrecisionTier::Performance && caps.span_fp8_cutlass)
      wanted.push_back(SpanVariantKind::Fp8Cutlass);
    if (tier == PrecisionTier::Performance && caps.fp8) wanted.push_back(SpanVariantKind::Fp8);
    if (tier != PrecisionTier::Quality && caps.int8) wanted.push_back(SpanVariantKind::Int8);
    if (tier == PrecisionTier::Quality) wanted.push_back(SpanVariantKind::Fp32);
    if (tier == PrecisionTier::Performance && !caps.fp8)
      log->info("SPAN Performance (FP8) : ce moteur ne le calcule pas, variante INT8 ou FP16");
    const auto prepare_span = [&] {
      return prepare_engine(
          p, init, *c.backend, c.span_int8 ? c.span_int8->onnx : c.model->path,
          gpu::Shape{.dims = {1, 3, c.work.height, c.work.width}},
          EngineLabel{.name = c.span_int8 ? std::format("SPAN {}", variant_text(c.span_int8->kind)) : "SPAN",
                      .model = c.model->path.stem().string(),
                      .width = c.work.width,
                      .height = c.work.height},
          gpu::Lane::Upscale);
    };
    using Prepared = decltype(prepare_span());
    std::optional<Prepared> loaded;
    for (const SpanVariantKind kind : wanted) {
      SpanInt8Lookup found = find_span_variant(kind, c.model->path, variants);
      if (!found.note.empty()) log->info("SPAN {} : {}", variant_text(kind), found.note);
      if (!found.variant) continue;
      c.span_int8 = std::move(found.variant);
      loaded = prepare_span();
      if (*loaded) break;
      log->warn("SPAN {} refusé ({}) : variante suivante ou SPAN FP16", variant_text(kind),
                loaded->error().detail);
      c.span_int8.reset();
      loaded.reset();
    }
    if (!c.span_int8 && tier != PrecisionTier::Balanced)
      log->info("SPAN {} : aucune variante utilisable, FP16 à la place", label(tier));
    if (!loaded) loaded = prepare_span();
    if (!*loaded) return std::unexpected(std::move(loaded->error()));
    if (!**loaded) return std::nullopt;
    c.engine = std::move(***loaded);
  }
  // Au-delà de 720 lignes de travail (la 4K sans réduction), RIFE estime le mouvement sur une copie
  // réduite à 720p et le warp déforme les images de travail : la variante mouvement seulement.
  c.rife_size = !p.options.rife_image && c.work.height > kRifeMotionHeight
                    ? work_size(video.width, video.height, kRifeMotionHeight)
                    : c.work;
  // GMFSS Fortuna à la place de RIFE, à la taille de travail ; RIFE reprend si ses graphes manquent.
  if (want_gmfss) {
    // GMFSS fusionné (prepare/gmfss_split.h) : la paire en deux moteurs, les caractéristiques d'une image
    // gardées pour la paire suivante, quand le backend le sait, que le réglage le veut et que le cran est
    // Équilibre ; tout échec garde la paire entière.
    GmfssFusedUse fused;
    fused.declined = gmfss_fused_declined(settings, c.backend->capabilities().gmfss_fused);
    if (settings.gmfss_fused && !fused.declined.empty())
      log->info("GMFSS fusionné non utilisé : {}", fused.declined);
    const bool try_fused = fused.declined.empty();
    auto loaded = prepare_gmfss(p, init, *c.backend, c.gmfss_places, c.gmfss_variant, c.work.width,
                                c.work.height, settings.gmfss_precision, try_fused ? &fused : nullptr);
    if (loaded && *loaded) {
      c.gmfss_fused = fused.used;
      if (fused.used)
        log->info("GMFSS fusionné : caractéristiques et flot en deux moteurs");
      else if (try_fused)
        log->warn("GMFSS fusionné indisponible ({}) : paire entière", fused.declined);
    }
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(
          c.rife_model ? i18n::tr("GMFSS unavailable: using RIFE") : i18n::tr("GMFSS unavailable"),
          host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      c.gmfss = std::move(**loaded);
    }
  }
  // DistilDRBA à la place de RIFE, à la taille de travail ; RIFE reprend si son graphe manque.
  if (want_drba) {
    // DistilDRBA fusionné (kernels/spec/drba_fused.h) : quatre petits moteurs et des noyaux de colle à la
    // place du moteur entier, quand le backend le sait, que le réglage le veut et que le cran est Équilibre ;
    // tout échec garde le moteur entier.
    DrbaFusedUse fused;
    fused.declined = drba_fused_declined(settings, c.backend->capabilities().drba_fused);
    if (settings.drba_fused && !fused.declined.empty())
      log->info("DistilDRBA fusionné non utilisé : {}", fused.declined);
    const bool try_fused = fused.declined.empty();
    auto loaded = prepare_distil_drba(p, init, *c.backend, c.gmfss_places, c.work.width, c.work.height,
                                      c.drba_align, settings.motion_precision, try_fused ? &fused : nullptr);
    if (loaded && *loaded) {
      c.drba_fused = fused.used;
      if (fused.used)
        log->info("DistilDRBA fusionné : quatre moteurs et noyaux de colle");
      else if (try_fused)
        log->warn("DistilDRBA fusionné indisponible ({}) : DistilDRBA entier", fused.declined);
    }
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(
          c.rife_model ? i18n::tr("DistilDRBA unavailable: using RIFE") : i18n::tr("DistilDRBA unavailable"),
          host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      c.drba = std::move(**loaded);
    }
  }
  // RIFE au cran de précision demandé : le graphe « _fp8 » ou « _fp32 » à côté du modèle quand il y est (FP8
  // sur un moteur qui l'accélère seulement) ; le modèle tel que livré sinon.
  if (c.rife_model && settings.motion_engine == MotionEngine::Rife &&
      settings.motion_precision != PrecisionTier::Balanced) {
    const PrecisionTier tier = settings.motion_precision;
    if (tier == PrecisionTier::Performance && !c.backend->capabilities().fp8) {
      log->info("RIFE Performance (FP8) : ce moteur ne le calcule pas, Équilibre à la place");
    } else if (const std::filesystem::path variant = tier_variant(*c.rife_model, tier);
               variant != *c.rife_model) {
      log->info("RIFE {} : {}", label(tier), variant.filename().string());
      c.rife_model = variant;
    } else {
      log->info("RIFE {} : aucun graphe {}, Équilibre à la place", label(tier), precision_text(tier));
    }
  }
  // L'interpolation d'Apple à la place de RIFE image, à la taille de travail ; RIFE reprend sous un
  // autre backend que Core ML, ou sur un Mac qui ne l'offre pas.
  if (want_apple) {
    auto loaded = prepare_apple(p, init, *c.backend, c.work.width, c.work.height);
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(c.rife_model ? i18n::tr("Apple interpolation unavailable: using RIFE")
                                          : i18n::tr("Apple interpolation unavailable"),
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
  // RIFE fusionné (kernels/spec/rife_fused.h) : six petits moteurs et des noyaux de colle à la place du
  // moteur entier, quand le backend le sait, que le réglage le veut, que le cran est Équilibre (le modèle tel
  // que livré) et que le modèle se coupe ; tout échec garde le RIFE entier, tel qu'il est chargé plus bas.
  const std::filesystem::path models_dir = p.paths.engines.parent_path() / "modeles";
  const std::string fused_declined =
      use_rife ? rife_fused_declined(settings, c.backend->capabilities().rife_fused, c.rife_planes)
               : i18n::tr("no RIFE");
  const bool fused_wanted = fused_declined.empty();
  if (use_rife && settings.rife_fused && !fused_wanted)
    log->info("RIFE fusionné non utilisé : {}", fused_declined);
  // Rien (fenêtre fermée pendant la construction) : « arrêt » ; un refus : journal et RIFE entier.
  enum class Fused : std::uint8_t { Loaded, Declined, Closed };
  const auto try_fused = [&](const kernels::RifeGeometry& g, bool motion,
                             std::unique_ptr<gpu::Engine>& into) {
    if (!fused_wanted) return Fused::Declined;
    auto fused = prepare_rife_fused(p, init, *c.backend, *c.rife_model, models_dir, g, motion);
    if (!fused) {
      log->warn("RIFE fusionné indisponible ({}) : RIFE entier", fused.error().detail);
      return Fused::Declined;
    }
    if (!*fused) return Fused::Closed;
    into = std::move(**fused);
    c.rife_fused = true;
    log->info("RIFE fusionné : six moteurs et noyaux de colle, {}", motion ? "flot et masque" : "image");
    return Fused::Loaded;
  };
  // RIFE par son mouvement : une passe au milieu de la paire la sert toute quand le mouvement est
  // faible ; sans variante dérivable, RIFE image comme avant.
  if (use_rife && !p.options.rife_image) {
    auto derived = motion_model(*c.rife_model, models_dir);
    if (!derived) {
      log->warn("RIFE mouvement indisponible ({}) : RIFE image", derived.error().detail);
    } else if (const Fused tried = try_fused(c.geometry, true, c.rife_motion); tried == Fused::Closed) {
      return std::nullopt;
    } else if (tried == Fused::Declined) {
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
    if (try_fused(g, false, c.rife) == Fused::Closed) return std::nullopt;
  }
  if (use_rife && !c.rife_motion && !c.rife) {
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

}  // namespace

}  // namespace ajp::app::play
