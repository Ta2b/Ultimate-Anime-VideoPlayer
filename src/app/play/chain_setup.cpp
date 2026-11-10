#include "app/play/chain_setup.h"

#include <algorithm>
#include <format>
#include <span>
#include <utility>

#include "app/play/auto_config.h"
#include "app/play/drba_fused_choice.h"
#include "app/play/drba_model.h"
#include "app/play/first_run.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/motion_model.h"
#include "app/play/paths.h"
#include "app/play/problem_report.h"
#include "app/play/rife_fused_choice.h"
#include "core/auto_rate.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// Hauteur où RIFE estime le mouvement quand la chaîne travaille plus haut (sortie 4K).
constexpr int kRifeMotionHeight = 720;

// Ce que le rapport « Signaler un problème » dit de la carte et du préréglage joué (A39-02).
void note_gpu_facts(const GpuFacts& gpu, const PlaySettings& settings, const GpuDevice& device) {
  note_report_fact("GPU", gpu_line(gpu));
  note_report_fact("Préréglage", label(settings.preset));
  if (device.vendor() == GpuVendor::Nvidia) note_report_fact("Pilote NVIDIA", device.driver_version);
}

// Le cran de précision voulu n'est pas tenu : le journal et le rapport le disent (A39-07).
void warn_span_fallback(spdlog::logger& log, PrecisionTier tier) {
  log.warn("SPAN {} : aucune variante utilisable, FP16 à la place", label(tier));
  note_report_fact("Précision SPAN", std::string("FP16 à la place de ") + std::string(label(tier)));
}

}  // namespace

std::vector<std::filesystem::path> motion_model_dirs(const PlayerPaths& paths,
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
  return use_drba() ? "DistilDRBA" : use_apple() ? "Apple" : "RIFE";
}

std::string ChainEngines::motion_label(const PlaySettings& settings) const {
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
  return std::format("SPAN ×{} · {} {} · {} · {} → {}x{}", factor, model->path.stem().string(),
                     span_int8 ? std::string(variant_text(span_int8->kind)) : std::string("FP16"),
                     backend_name, reduce, work.width * factor, work.height * factor);
}

UpscaleSetup ChainEngines::upscale(const PlaySettings& settings, std::shared_ptr<LiveSettings> live,
                                   const MasterClock* clock, int width, int height,
                                   std::vector<RingSlotDesc> slots) const {
  const WorkSize sized = chain_work_size(settings, width, height);
  return UpscaleSetup{
      .backend = backend.get(),
      .engine = engine.get(),
      .skip_held = skip_held && clock != nullptr,
      .rife = rife.get(),
      .rife_motion = rife_motion.get(),
      .rife_motion_calm = rife_motion_calm.get(),
      .rife_planes = rife_planes,
      .rife_align = rife_align,
      .distil_drba = drba.get(),
      .drba_align = drba_align,
      .drba_motion = drba_motion,
      .rife_factor = factor_of(settings),
      .one_lane = one_lane,
      .rife_warm = rife_warm,
      .live = std::move(live),
      .clock = clock,
      .source_width = width,
      .source_height = height,
      .work_width = sized.width,
      .work_height = sized.height,
      .rife_width = rife_motion   ? rife_size.width
                    : drba_motion ? drba_size.width
                                  : 0,
      .rife_height = rife_motion   ? rife_size.height
                     : drba_motion ? drba_size.height
                                   : 0,
      .warp_after_span = warp_after_span(settings.span_order, sized.height * kSpanFactor, top_factor),
      // En lecture (horloge présente), un instant déjà passé n'est ni calculé ni écrit, en Rapide comme à pas
      // constant : l'image serait jetée par la présentation (audit F11).
      .late_fate = clock != nullptr,
      .smooth_drawings = smooth_drawings,
      .slots = std::move(slots)};
}

namespace {

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain_once(Player& p,
                                                                       const PlaySettings& settings);

// La forme de l'ouverture (chain_shape.h) : l'anneau de ces réglages et l'ordre du warp que les moteurs
// chargés ont pris (c.top_factor), avec l'écran et la source d'où ils viennent.
void note_opened(ChainEngines& c, const Player& p, const PlaySettings& settings) {
  c.opened = {.display_hz = display_refresh_hz(p.screen.window()),
              .source_fps = fps_of(p.video.frame_rate),
              .output_height = c.work.height * kSpanFactor,
              .span = c.engine != nullptr,
              .rife_loaded = c.interpolates()};
  c.opened.shape = {
      .ring = ring_depth(settings, c.opened.rife_loaded ? factor_of(settings) : Factor{}),
      .warp_after_span =
          c.opened.span && warp_after_span(settings.span_order, c.opened.output_height, c.top_factor)};
}

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain_tiers(Player& p,
                                                                        const PlaySettings& settings) {
  auto chain = prepare_chain_once(p, settings);
  if (chain || (settings.span_precision == PrecisionTier::Balanced &&
                settings.motion_precision == PrecisionTier::Balanced))
    return chain;
  // Un graphe FP8 ou FP32 que le moteur refuse : la chaîne repart au cran Équilibre (FP16), celui de
  // toujours.
  log_channel("app")->warn("chaîne refusée aux crans de précision choisis ({}) : Équilibre (FP16)",
                           chain.error().detail);
  PlaySettings balanced = settings;
  balanced.span_precision = balanced.motion_precision = PrecisionTier::Balanced;
  return prepare_chain_once(p, balanced);
}

}  // namespace

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings) {
  auto chain = prepare_chain_tiers(p, settings);
  if (chain && *chain) note_opened(**chain, p, settings);
  return chain;
}

namespace {

// Les options d'essai que la chaîne porte telles quelles.
void take_trials(ChainEngines& c, const PlayOptions& o) {
  c.one_lane = o.one_lane;
  c.rife_warm = o.rife_warm;
  c.skip_held = o.skip_held;
  c.smooth_drawings = o.smooth_drawings;
}

// P6.17, essai (--rife-calm-model) : la variante mouvement d'un RIFE plus léger, à la forme de rife_motion,
// pour la passe du milieu des paires calmes. Facultative : un modèle aux autres plans ou à un autre
// remplissage, une dérivation ou une construction refusée la laissent de côté, la raison au journal (essai :
// un refus n'arrête pas la lecture).
void load_calm_rife(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                    const std::filesystem::path& models_dir) {
  if (!c.rife_motion || !p.options.rife_calm_model) return;
  const auto log = log_channel("app");
  const std::filesystem::path& calm = *p.options.rife_calm_model;
  auto planes = rife_input_planes(calm);
  if (!planes || *planes != c.rife_planes || rife_padding_multiple(calm) != c.rife_align) {
    log->warn("RIFE des paires calmes ignoré : {} n'a pas les plans ni le remplissage du RIFE principal",
              core::to_utf8(calm.filename()));
    return;
  }
  auto derived = motion_model(calm, models_dir);
  if (!derived) {
    log->warn("RIFE des paires calmes ignoré : {}", derived.error().detail);
    return;
  }
  const kernels::RifeGeometry& g = c.geometry;
  auto loaded = prepare_engine(p, init, *c.backend, *derived,
                               gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                               EngineLabel{.name = "RIFE calme",
                                           .model = core::to_utf8(derived->stem()),
                                           .width = g.padded_width,
                                           .height = g.padded_height},
                               gpu::Lane::Interpolate);
  if (!loaded) {
    log->warn("RIFE des paires calmes ignoré : {}", loaded.error().detail);
  } else if (*loaded) {
    c.rife_motion_calm = std::move(**loaded);
    log->info("RIFE des paires calmes : {}", core::to_utf8(derived->stem()));
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<std::optional<ChainEngines>, Refusal> prepare_chain_once(Player& p,
                                                                       const PlaySettings& settings) {
  const auto log = log_channel("app");
  const media::TrackInfo& video = p.video;
  ChainEngines c;
  take_trials(c, p.options);
  const bool span = span_runs(settings, video.height);
  if (settings.upscale && !span)
    log_channel("app")->info("restaurer les traits coupé : source {}p gardée telle quelle, sans SPAN",
                             video.height);
  if (span) {
    auto chosen = choose_model(p.options, p.paths.home, settings, p.paths.models);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    c.model = std::move(*chosen);
  }
  c.factor = c.model ? kSpanFactor : 1;
  c.top_factor = top_factor(settings, display_refresh_hz(p.screen.window()), fps_of(p.video.frame_rate));
  c.work = chain_work_size(settings, video.width, video.height);
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
  note_gpu_facts(gpu, settings, p.screen.device());
  if (p.screen.device().vendor() == GpuVendor::Nvidia) {
    const std::string& driver = p.screen.device().driver_version;
    if (settings.smooth_motion && nvidia_driver_older_than(driver, 580, 82, 7))
      log->warn("pilote NVIDIA {} : Smooth Motion demande 580.82.07 ou plus récent", driver);
  }
  const std::span<const std::string_view> rife_order = rife_preference(gpu);
  if (auto found = choose_rife_model(p.options, p.paths.home, settings, rife_order, p.paths.models))
    c.rife_choice = std::move(*found);
  if (c.rife_choice) c.rife_model = c.rife_choice->path;
  c.model_places = motion_model_dirs(p.paths, c.rife_choice);
  c.drba_found = find_distil_drba(c.model_places).has_value();
  const bool want_drba = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::DistilDrba;
  const bool want_apple = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::Apple;
  if (settings.rife_factor >= 2 && !c.rife_model && !(want_drba && c.drba_found) && !want_apple)
    return std::unexpected(
        choose_rife_model(p.options, p.paths.home, settings, rife_order, p.paths.models).error());
  show_loading(p.screen, p.ui, i18n::ui_tr("Loading"));
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
    const std::filesystem::path variants = paths::derived_models(p.paths);
    const gpu::BackendCaps caps = c.backend->capabilities();
    const PrecisionTier tier = settings.span_precision;
    // Un cran posé par un préréglage (Auto et les autres, auto_config.h) est strict : ni FP8 par Q/DQ, ni
    // reçu v1 du FP8 CUTLASS (étape 3.6).
    const bool strict = settings.preset != Preset::Custom;
    const std::vector<SpanVariantKind> wanted = span_variant_order(
        tier, SpanVariantCaps{.cutlass = caps.span_fp8_cutlass, .fp8 = caps.fp8, .int8 = caps.int8}, strict);
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
    const SpanFp8Context fp8_context{.gpu_arch = caps.gpu_arch,
                                     .kernels_version = caps.span_fp8_kernels,
                                     .width = c.work.width,
                                     .height = c.work.height};
    // Le SPAN FP8 choisi à la main dans la liste des modèles se prépare ici, sur l'écran de chargement, comme
    // un moteur qui se construit ; la tâche de fond (span_auto.h) ne le refait pas pour cette vidéo.
    // Le bench (--bench-out) ne valide jamais sur son clip : une mire synthétique aux bords nets jugerait le
    // FP8 bien plus durement qu'un épisode et écrirait ses refus dans le reçu de l'utilisateur.
    const bool bench = p.options.bench_out.has_value();
    if (tier == PrecisionTier::Performance && !strict && caps.span_fp8_cutlass && !p.options.file.empty() &&
        !bench) {
      const std::optional<bool> ready =
          prepare_span_fp8(p, init, c.model->path, variants,
                           span_still_reduce(c.work.width, c.work.height, settings.downscale), fp8_context);
      if (!ready) return std::nullopt;
      p.span_job.attempted = prepare::utf8_path(c.model->path) + "|" + prepare::utf8_path(p.options.file);
    }
    for (const SpanVariantKind kind : wanted) {
      SpanInt8Lookup found = find_span_variant(kind, c.model->path, variants,
                                               kind == SpanVariantKind::Fp8Cutlass ? &fp8_context : nullptr);
      if (!found.note.empty()) log->info("SPAN {} : {}", variant_text(kind), found.note);
      if (!found.variant && bench && found.unjudged) {
        // Le bench mesure le temps du FP8 à cette taille ; la lecture d'un épisode la jugera sur ses images.
        log->info("SPAN {} : pas jugé à {}x{}, joué pour le bench seulement", variant_text(kind),
                  c.work.width, c.work.height);
        found.variant = std::move(found.unjudged);
      }
      if (!found.variant) continue;
      if (strict && kind == SpanVariantKind::Fp8Cutlass && !found.variant->receipt_v2) {
        log->info("SPAN FP8 CUTLASS : reçu v1, pas pris par un préréglage (préparer de nouveau)");
        continue;
      }
      c.span_int8 = std::move(found.variant);
      loaded = prepare_span();
      if (*loaded) break;
      log->warn("SPAN {} refusé ({}) : variante suivante ou SPAN FP16", variant_text(kind),
                loaded->error().detail);
      c.span_int8.reset();
      loaded.reset();
    }
    if (!c.span_int8 && tier != PrecisionTier::Balanced) warn_span_fallback(*log, tier);
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
    // Warp après SPAN (sortie haute) : la variante mouvement, un SPAN par source au lieu d'un par image
    // créée.
    bool motion =
        settings.upscale && warp_after_span(settings.span_order, c.work.height * kSpanFactor, c.top_factor);
    // D4 : la variante mouvement estime à la taille de RIFE mouvement (720p pour un travail plus grand),
    // comme RIFE ; sans variante mouvement, l'image se refait à la taille de travail.
    const bool smaller = motion && c.rife_size.height < c.work.height;
    const WorkSize at = smaller ? c.rife_size : c.work;
    auto loaded = prepare_distil_drba(p, init, *c.backend, c.model_places, at.width, at.height, c.drba_align,
                                      settings.motion_precision, try_fused ? &fused : nullptr, &motion);
    if (smaller && loaded && *loaded && !motion) {
      fused = DrbaFusedUse{.declined = fused.declined};
      loaded =
          prepare_distil_drba(p, init, *c.backend, c.model_places, c.work.width, c.work.height, c.drba_align,
                              settings.motion_precision, try_fused ? &fused : nullptr, &motion);
    }
    c.drba_motion = loaded && *loaded && motion;
    c.drba_size = c.drba_motion ? at : c.work;
    if (c.drba_motion && smaller)
      log->info("DistilDRBA mouvement : estimé en {}x{}, warp en {}x{}", at.width, at.height, c.work.width,
                c.work.height);
    if (loaded && *loaded) {
      c.drba_fused = fused.used;
      if (fused.used)
        log->info("DistilDRBA fusionné : quatre moteurs et noyaux de colle");
      else if (try_fused)
        log->warn("DistilDRBA fusionné indisponible ({}) : DistilDRBA entier", fused.declined);
    }
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().notify(c.rife_model ? i18n::ui_tr("DistilDRBA unavailable: using RIFE")
                                          : i18n::ui_tr("DistilDRBA unavailable"),
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
      p.ui.controls().notify(c.rife_model ? i18n::ui_tr("Apple interpolation unavailable: using RIFE")
                                          : i18n::ui_tr("Apple interpolation unavailable"),
                             host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      c.rife = std::move(**loaded);
      c.apple = true;
      c.rife_planes = kernels::kRifeV2InputChannels;
    }
  }
  const bool use_rife = settings.rife_factor >= 2 && c.rife_model && !c.use_drba() && !c.use_apple();
  c.rife_align = use_rife ? rife_padding_multiple(*c.rife_model) : kernels::kRifeAlign;
  c.geometry = kernels::rife_geometry(c.rife_size.width, c.rife_size.height, c.rife_align);
  c.image_geometry = kernels::rife_geometry(c.work.width, c.work.height, c.rife_align);
  if (use_rife) {
    auto planes = rife_input_planes(*c.rife_model);
    if (!planes) return std::unexpected(std::move(planes.error()));
    c.rife_planes = *planes;
  }
  // RIFE fusionné (kernels/spec/rife_fused.h) à la place du moteur entier quand rife_fused_choice.h le veut
  // et que le modèle se coupe ; tout échec garde le RIFE entier, tel qu'il est chargé plus bas.
  const std::filesystem::path models_dir = paths::derived_models(p.paths);
  const std::string fused_declined = use_rife
                                         ? rife_fused_declined(settings, c.backend->capabilities().rife_fused,
                                                               c.rife_planes, model_tier(c.rife_model))
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
  load_calm_rife(p, init, c, models_dir);
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
