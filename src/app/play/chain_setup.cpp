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
#include "app/play/pumped_task.h"
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

std::vector<std::pair<std::string, std::string>> playback_facts(const PlaybackFacts& facts) {
  std::string rate =
      facts.frame_rate > 0.0 ? std::format("{:.3f} i/s", facts.frame_rate) : "cadence inconnue";
  std::vector<std::pair<std::string, std::string>> out;
  out.emplace_back("Vidéo", std::format("{} · {} · {}x{} · {} · source {}", facts.file, facts.codec,
                                        facts.width, facts.height, rate, facts.transfer));
  out.emplace_back("Écran", facts.screen);
  out.emplace_back("Sortie", facts.hdr_output ? "HDR10" : "SDR (image gradée)");
  out.emplace_back("Chaîne jouée", facts.chain);
  return out;
}

void note_playback_facts(const media::TrackInfo& video, const ColorInfo& source, const std::string& screen,
                         bool hdr_output, const std::string& chain, const std::filesystem::path& file) {
  const PlaybackFacts facts{
      .file = core::to_utf8(file.filename()),
      .codec = video.codec,
      .width = video.width,
      .height = video.height,
      .frame_rate = video.frame_rate.den > 0 ? static_cast<double>(video.frame_rate.num) /
                                                   static_cast<double>(video.frame_rate.den)
                                             : 0.0,
      .transfer = source.transfer == Transfer::Pq ? "PQ" : (source.transfer == Transfer::Hlg ? "HLG" : "SDR"),
      .screen = screen,
      .hdr_output = hdr_output,
      .chain = chain};
  for (const auto& [name, value] : playback_facts(facts)) note_report_fact(name, value);
}

void announce_playback(const Player& p, const ColorInfo& source, bool hdr_output, const std::string& chain) {
  log_channel("app")->info("{} : {}", prepare::utf8_path(p.options.file.filename()), chain);
  note_playback_facts(p.video, source, describe_hdr(p.screen.window()), hdr_output, chain, p.options.file);
}

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
                                      const media::HwDeviceSource* shared, bool smooth_drawings) {
  const bool tensorrt = backend == "tensorrt";
  return {.attempts = decoder_attempts(backend, shared != nullptr, render.vendor() == GpuVendor::Nvidia),
          .render_device = render.uuid,
          .render_luid = render.luid,
          .devices = shared,
          .download = !tensorrt,
          // Les deux ingestions lisent Yuv444p16 et P016 en mémoire hôte ; celle de TensorRT lit aussi le
          // 4:2:2 (Yuv422p16) : la chroma d'une source logicielle n'est jamais réduite (B01-03).
          .full_host_formats = true,
          .host_422 = tensorrt,
          .direct_surfaces = tensorrt ? held_decoded_frames(smooth_drawings) : 0};
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
  if (anime4k)
    return std::format("Anime4K ×2 · {} · mode {} · libplacebo · {} → {}x{}", render::name(anime4k->size),
                       render::name(anime4k->mode), reduce, work.width * 2, work.height * 2);
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
      .warp_after_span = warp_after_span(settings.span_order, sized.height * kSpanFactor, order_factor),
      // En lecture (horloge présente), un instant déjà passé n'est ni calculé ni écrit, en Rapide comme à pas
      // constant : l'image serait jetée par la présentation (audit F11).
      .late_fate = clock != nullptr,
      .smooth_drawings = smooth_drawings,
      .slots = std::move(slots)};
}

namespace {

using Prepared = std::expected<std::optional<ChainEngines>, StageFailure>;

Prepared prepare_chain_once(Player& p, const PlaySettings& settings);

// La forme de l'ouverture (chain_shape.h) : l'anneau de ces réglages et l'ordre du warp que les moteurs
// chargés ont pris (c.order_factor), avec l'écran et la source d'où ils viennent.
void note_opened(ChainEngines& c, const Player& p, const PlaySettings& settings) {
  c.opened = {.display_hz = display_refresh_hz(p.screen.window()),
              .source_fps = fps_of(p.video.frame_rate),
              .output_height = c.work.height * kSpanFactor,
              .span = c.engine != nullptr,
              .rife_loaded = c.interpolates()};
  c.opened.drba = c.drba != nullptr;
  c.opened.drba_fused = c.opened.drba && c.drba_fused;
  c.opened.drba_motion = c.opened.drba && c.drba_motion;
  c.opened.drba_fused_capable = c.backend != nullptr && c.backend->capabilities().drba_fused;
  c.opened.drba_fused_refused = c.opened.drba ? c.drba_fused_refused : std::nullopt;
  c.opened.rife_fused_refused = c.rife_fused ? std::nullopt : c.rife_fused_refused;
  if (c.opened.rife_loaded) c.opened.loaded = loaded_by(settings);
  c.opened.shape = {
      .ring = ring_depth(settings, c.opened.rife_loaded ? factor_of(settings) : Factor{}, c.opened.display_hz,
                         c.opened.source_fps),
      .warp_after_span =
          c.opened.span && warp_after_span(settings.span_order, c.opened.output_height, c.order_factor)};
}

}  // namespace

std::string library_name(const Player& p) {
  return p.library ? core::to_utf8(p.library->path().filename()) : std::string();
}

std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings,
                                                                  StageMemory* memory) {
  const auto log = log_channel("app");
  // Sans mémoire de séance (export, banc), une mémoire de cet appel : les crans retombent, rien d'autre.
  StageMemory local;
  StageMemory& refused = memory != nullptr ? *memory : local;
  const std::string library = library_name(p);
  // B08-03 : une instance de construction pour toute l'ouverture, rendue en sortant (pas pendant la lecture).
  p.builder = std::make_shared<BackendLibrary::Instance>();
  const struct BuilderRelease {
    Player& p;
    ~BuilderRelease() { p.builder.reset(); }
  } builder_release{p};
  p.loaded_engines.clear();
  const int width = p.video.width;
  const int height = p.video.height;
  // Chaque tour retient un refus de plus : au plus deux crans et deux étages, puis un dernier essai.
  constexpr int kAttempts = 5;
  std::optional<StageFailure> last;
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    std::optional<StagePlan> plan =
        plan_stages(settings, refused, width, height, library, p.options.file, memory != nullptr);
    if (!plan) break;  // un étage refusé que seule la lecture peut laisser de côté
    for (const StageNote& note : plan->notes)
      log->warn("{} : {} ({})", note.stage == ChainStage::Upscale ? "SPAN" : "interpolation",
                note.change == StageChange::Cancelled ? "construction annulée, lecture sans lui"
                : note.change == StageChange::Dropped ? "refusé, lecture sans lui"
                                                      : "cran refusé, Équilibre (FP16) à la place",
                note.refusal.detail);
    auto chain = prepare_chain_once(p, plan->settings);
    if (chain) {
      if (*chain) {
        ChainEngines& c = **chain;
        note_opened(c, p, plan->settings);
        // A03-21 : le repli du fusionné vers le réseau entier se dit à l'écran, pas au journal seulement ;
        // une fois, à son refus, pas à chaque réouverture qui reprend le repli retenu (B08-02).
        if (p.notice.empty() && c.fallback_fresh) p.notice = fused_refusal_notice(c.opened);
        c.played = std::move(plan->settings);
        c.stage_notes.insert(c.stage_notes.begin(), plan->notes.begin(), plan->notes.end());
        c.loaded_engines = std::exchange(p.loaded_engines, {});
      }
      return std::move(*chain);
    }
    StageFailure& failure = chain.error();
    // Une vidéo glissée arrête toute la préparation : rien n'est retenu, la lecture passe à elle (A08-08).
    // Un appareil perdu n'est pas un refus de l'étage : la chaîne entière s'arrête, la relance suit (A31-02).
    if (!failure.stage || failure.refusal.key == kDroppedVideoKey ||
        failure.refusal.code == RefusalCode::DeviceLost)
      return std::unexpected(std::move(failure.refusal));
    // Le refus retenu sous la signature de l'étage tel qu'il a été essayé : la suite (ce tour, la vidéo
    // suivante, une réouverture) ne le reconstruit pas.
    const ChainStage stage = *failure.stage;
    log->warn("{} refusé : {}", stage == ChainStage::Upscale ? "SPAN" : "interpolation",
              failure.refusal.detail);
    // La fin du journal de construction part avec « Signaler un problème » (B39-03). Une annulation n'est pas
    // un échec à signaler.
    if (!watch_now(failure.refusal)) note_build_failure(name(failure.refusal.code), failure.refusal.detail);
    refused.remember({.stage = stage,
                      .signature = stage_signature(stage, plan->settings, width, height, library),
                      .refusal = failure.refusal,
                      .cancelled = watch_now(failure.refusal),
                      .file = p.options.file});
    last = std::move(failure);
  }
  return std::unexpected(last ? std::move(last->refusal)
                              : Refusal{RefusalCode::BuilderFailed, "chaîne refusée"});
}

namespace {

// Ce que DistilDRBA chargé a pris : le fusionné, ou l'entier après un refus du fusionné (A03-21).
void note_drba_fused(ChainEngines& c, const DrbaFusedUse& fused, bool tried) {
  c.drba_fused = fused.used;
  c.drba_fused_refused = fused.refused;
  if (fused.used)
    log_channel("app")->info("DistilDRBA fusionné : quatre moteurs et noyaux de colle");
  else if (tried)
    log_channel("app")->warn("DistilDRBA fusionné indisponible ({}) : DistilDRBA entier", fused.declined);
}

// Les options d'essai que la chaîne porte telles quelles.
void take_trials(ChainEngines& c, const PlayOptions& o) {
  c.one_lane = o.one_lane;
  c.rife_warm = o.rife_warm;
  c.skip_held = o.skip_held;
}

// Ce que le backend de la chaîne reçoit du lecteur, options d'essai comprises (--profile-dispatches).
gpu::BackendInit chain_backend_init(Player& p) {
  return {.render = &p.screen.device(),
          .cache_dir = p.paths.engines,
          .log = backend_log(),
          .profile_dispatches = p.options.profile_dispatches};
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

// Un refus de l'étage qui le rend : la chaîne peut se préparer sans lui (stage_fallback.h).
std::unexpected<StageFailure> stage_refused(Refusal refusal, std::optional<ChainStage> stage) {
  return std::unexpected(StageFailure{.refusal = std::move(refusal), .stage = stage});
}

// La variante de ce genre que SPAN peut prendre ici, rien sinon : celle qui passe ses garde-fous, ou pour le
// banc celle pas encore jugée à cette taille ; un préréglage (strict) ne prend pas un reçu v1 du FP8 CUTLASS.
std::optional<SpanInt8> usable_variant(SpanVariantKind kind, const ModelChoice& model,
                                       const std::filesystem::path& variants,
                                       const SpanFp8Context& fp8_context, WorkSize work, bool bench,
                                       bool strict) {
  const auto log = log_channel("app");
  SpanInt8Lookup found = find_span_variant(kind, model.path, variants,
                                           kind == SpanVariantKind::Fp8Cutlass ? &fp8_context : nullptr);
  if (!found.note.empty()) log->info("SPAN {} : {}", variant_text(kind), found.note);
  if (!found.variant && bench && found.unjudged) {
    // Le bench mesure le temps du FP8 à cette taille ; la lecture d'un épisode la jugera sur ses images.
    log->info("SPAN {} : pas jugé à {}x{}, joué pour le bench seulement", variant_text(kind), work.width,
              work.height);
    found.variant = std::move(found.unjudged);
  }
  if (found.variant && strict && kind == SpanVariantKind::Fp8Cutlass && !found.variant->receipt_v2) {
    log->info("SPAN FP8 CUTLASS : reçu v1, pas pris par un préréglage (préparer de nouveau)");
    return std::nullopt;
  }
  return std::move(found.variant);
}

// Le SPAN FP8 choisi à la main dans la liste des modèles se prépare ici, sur l'écran de chargement, comme
// un moteur qui se construit ; la tâche de fond (span_auto.h) ne le refait pas pour cette vidéo.
// Le bench (--bench-out) ne valide jamais sur son clip : une mire synthétique aux bords nets jugerait le
// FP8 bien plus durement qu'un épisode et écrirait ses refus dans le reçu de l'utilisateur. Faux : la fenêtre
// fermée ; refus : une vidéo glissée pendant la préparation.
std::expected<bool, StageFailure> prepare_manual_fp8(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                                     const PlaySettings& settings,
                                                     const SpanFp8Context& fp8_context) {
  if (!c.model || p.options.file.empty() || p.options.bench_out) return true;
  const std::optional<bool> ready =
      prepare_span_fp8(p, init, c.model->path, paths::derived_models(p.paths),
                       span_still_reduce(c.work.width, c.work.height, settings.downscale), fp8_context);
  if (!ready && p.dropped_video) return stage_refused(dropped_video_refusal(), std::nullopt);
  if (!ready) return false;
  p.span_job.attempted = prepare::utf8_path(c.model->path) + "|" + prepare::utf8_path(p.options.file);
  return true;
}

// La variante de SPAN du cran (precision.h) quand tools/span/quantize_span.py ou derive_span_fp8.py en ont
// laissé une qui passe ses garde-fous (span_int8.h) : Performance prend le SPAN entier en noyaux CUTLASS
// FP8 (moteur qui l'annonce : RTX 40), puis le FP8 par Q/DQ (moteur qui l'accélère), puis l'INT8 ;
// Équilibre l'INT8, sinon le FP16 tel que livré ; Qualité le FP32. Chaque variante refusée au chargement
// (autre GPU, taille que les noyaux ne prennent pas, erreur CUDA, TensorRT qui la refuse) laisse la place à
// la suivante, et la dernière au SPAN choisi, en FP16 sous TensorRT. Faux : la fenêtre fermée.
std::expected<bool, StageFailure> load_span(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                            const PlaySettings& settings) {
  if (!c.model) return true;  // rien à charger : l'appelant ne vient qu'avec un modèle
  const auto log = log_channel("app");
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
  using Loaded = decltype(prepare_span());
  std::optional<Loaded> loaded;
  const SpanFp8Context fp8_context{.gpu_arch = caps.gpu_arch,
                                   .kernels_version = caps.span_fp8_kernels,
                                   .width = c.work.width,
                                   .height = c.work.height};
  const bool bench = p.options.bench_out.has_value();
  if (tier == PrecisionTier::Performance && !strict && caps.span_fp8_cutlass) {
    auto ready = prepare_manual_fp8(p, init, c, settings, fp8_context);
    if (!ready || !*ready) return ready;
  }
  InnerFallbacks inner(p.stages, c.stage_notes, c.fallback_fresh);  // B08-02
  for (const SpanVariantKind kind : wanted) {
    std::optional<SpanInt8> found =
        usable_variant(kind, *c.model, variants, fp8_context, c.work, bench, strict);
    if (!found) continue;
    const std::string key = inner_key(std::format("span-variant {}", variant_text(kind)), library_name(p),
                                      c.model->path, c.work.width, c.work.height);
    if (inner.refused(ChainStage::Upscale, key) != nullptr) continue;
    c.span_int8 = std::move(found);
    loaded = prepare_span();
    if (*loaded) break;
    // « Annuler et regarder » n'est pas un refus de la variante : aucune autre ne se construit (A08-01).
    if (watch_now(loaded->error())) return stage_refused(std::move(loaded->error()), ChainStage::Upscale);
    log->warn("SPAN {} refusé ({}) : variante suivante ou SPAN FP16", variant_text(kind),
              loaded->error().detail);
    inner.failed(ChainStage::Upscale, key, loaded->error());
    c.span_int8.reset();
    loaded.reset();
  }
  if (!c.span_int8 && tier != PrecisionTier::Balanced) warn_span_fallback(*log, tier);
  if (!loaded) loaded = prepare_span();
  if (!*loaded) return stage_refused(std::move(loaded->error()), ChainStage::Upscale);
  if (!**loaded) return false;
  c.engine = std::move(***loaded);
  return true;
}

// DistilDRBA fusionné (kernels/spec/drba_fused.h) : quatre petits moteurs et des noyaux de colle à la place
// du moteur entier, quand le backend le sait, que le réglage le veut et que le cran est Équilibre ; tout
// échec garde le moteur entier. Un refus retenu pour la séance (B08-02) le décline sans reconstruire.
DrbaFusedUse drba_fused_use(const ChainEngines& c, const PlaySettings& settings, InnerFallbacks& inner,
                            const std::string& fused_key) {
  DrbaFusedUse fused;
  fused.declined = drba_fused_declined(settings, c.backend->capabilities().drba_fused);
  if (settings.drba_fused && !fused.declined.empty())
    log_channel("app")->info("DistilDRBA fusionné non utilisé : {}", fused.declined);
  if (!fused.declined.empty()) return fused;
  if (const Refusal* before = inner.refused(ChainStage::Motion, fused_key)) {
    fused.declined = before->detail;
    fused.refused = before->code;
  }
  return fused;
}

// Le moteur DistilDRBA, la variante mouvement d'abord quand le warp suit SPAN ; pose drba_motion et
// drba_size.
std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> build_drba(Player& p,
                                                                               const gpu::BackendInit& init,
                                                                               ChainEngines& c,
                                                                               const PlaySettings& settings,
                                                                               DrbaFusedUse* fused) {
  // Warp après SPAN (sortie haute) : la variante mouvement, un SPAN par source au lieu d'un par image
  // créée.
  bool motion =
      settings.upscale && warp_after_span(settings.span_order, c.work.height * kSpanFactor, c.order_factor);
  // D4 : la variante mouvement estime à la taille de RIFE mouvement (720p pour un travail plus grand),
  // comme RIFE ; sans variante mouvement, l'image se refait à la taille de travail.
  const bool smaller = motion && c.rife_size.height < c.work.height;
  const WorkSize at = smaller ? c.rife_size : c.work;
  auto loaded = prepare_distil_drba(p, init, *c.backend, c.model_places, at.width, at.height, c.drba_align,
                                    settings.motion_precision, fused, &motion, smaller);
  // Sans variante mouvement, l'image se fait à la taille de travail : rien n'a été construit plus bas.
  if (smaller &&
      ((!loaded && loaded.error().key == kDrbaImageElsewhereKey) || (loaded && *loaded && !motion))) {
    if (fused != nullptr) *fused = DrbaFusedUse{.declined = fused->declined};
    loaded = prepare_distil_drba(p, init, *c.backend, c.model_places, c.work.width, c.work.height,
                                 c.drba_align, settings.motion_precision, fused, &motion);
  }
  c.drba_motion = loaded && *loaded && motion;
  c.drba_size = c.drba_motion ? at : c.work;
  if (c.drba_motion && smaller)
    log_channel("app")->info("DistilDRBA mouvement : estimé en {}x{}, warp en {}x{}", at.width, at.height,
                             c.work.width, c.work.height);
  return loaded;
}

// DistilDRBA à la place de RIFE, à la taille de travail ; RIFE reprend si son graphe manque. Un refus retenu
// pour la séance (B08-02) passe à RIFE sans reconstruire. Faux : construction annulée ; vrai : chargé, ou
// refusé avec RIFE (ou rien) à la place.
std::expected<bool, StageFailure> load_drba(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                            const PlaySettings& settings, InnerFallbacks& inner,
                                            const std::string& library) {
  const auto log = log_channel("app");
  const std::string drba_key = inner_key(std::format("drba {}", name(settings.motion_precision)), library, {},
                                         c.work.width, c.work.height);
  if (inner.refused(ChainStage::Motion, drba_key) != nullptr) {
    log->info("DistilDRBA refusé plus tôt dans la séance : {}", c.rife_model ? "RIFE" : "sans interpolation");
    return true;
  }
  const std::string fused_key = inner_key("drba-fused", library, {}, c.work.width, c.work.height);
  DrbaFusedUse fused = drba_fused_use(c, settings, inner, fused_key);
  const bool try_fused = fused.declined.empty();
  auto loaded = build_drba(p, init, c, settings, try_fused ? &fused : nullptr);
  if (loaded && *loaded) {
    note_drba_fused(c, fused, try_fused);
    if (try_fused && fused.refused)
      inner.failed(ChainStage::Motion, fused_key, Refusal{*fused.refused, fused.declined});
  }
  // Annulée, la construction ne passe pas la main à RIFE : l'interpolation part, SPAN reste (A31-15).
  if (!loaded && watch_now(loaded.error()))
    return stage_refused(std::move(loaded.error()), ChainStage::Motion);
  if (!loaded) {
    log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
    inner.failed(ChainStage::Motion, drba_key, loaded.error());
    p.ui.controls().warn(c.rife_model ? i18n::ui_tr("DistilDRBA unavailable: using RIFE")
                                      : i18n::ui_tr("DistilDRBA unavailable"),
                         host_now_us());
    return true;
  }
  if (!*loaded) return false;
  c.drba = std::move(**loaded);
  return true;
}

// Le moteur retenu : au journal et au rapport de problème (B39-02).
void announce_backend(spdlog::logger& log, const gpu::Backend& backend, const std::string& name,
                      const std::filesystem::path& library) {
  const auto& caps = backend.capabilities();
  log.info("backend {} ({}) depuis {}/{}", name, caps.device_name,
           core::to_utf8(library.parent_path().filename()), core::to_utf8(library.filename()));
  note_report_fact("Backend", std::format("{} sur {} · {}", name, caps.device_name, caps.driver_version));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
Prepared prepare_chain_once(Player& p, const PlaySettings& settings) {
  const auto log = log_channel("app");
  const media::TrackInfo& video = p.video;
  ChainEngines c;
  take_trials(c, p.options);
  c.smooth_drawings = settings.smooth_drawings;  // --smooth-drawings compris (settings_for)
  c.work = chain_work_size(settings, video.width, video.height);
  const bool span = take_upscaler(p, c, settings);
  if (span) {
    auto chosen = choose_model(p.options, p.paths.home, settings, p.paths.models);
    if (!chosen) return stage_refused(std::move(chosen.error()), ChainStage::Upscale);
    c.model = std::move(*chosen);
  }
  c.factor = c.model ? kSpanFactor : 1;
  c.order_factor = order_factor(settings, display_refresh_hz(p.screen.window()), fps_of(p.video.frame_rate));
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
  // B08-14 : un seul choix du modèle, son refus gardé pour le cas où l'interpolation n'a rien d'autre.
  auto rife_found = choose_rife_model(p.options, p.paths.home, settings, rife_order, p.paths.models);
  if (rife_found) c.rife_choice = std::move(*rife_found);
  if (c.rife_choice) c.rife_model = c.rife_choice->path;
  c.model_places = motion_model_dirs(p.paths, c.rife_choice);
  c.drba_found = find_distil_drba(c.model_places).has_value();
  const bool want_drba = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::DistilDrba;
  const bool want_apple = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::Apple;
  if (settings.rife_factor >= 2 && !c.rife_model && !(want_drba && c.drba_found) && !want_apple)
    return stage_refused(std::move(rife_found.error()), ChainStage::Motion);
  show_loading(p.screen, p.ui, i18n::ui_tr("Loading"));
  const gpu::BackendInit init = chain_backend_init(p);
  auto backend = run_pumped<BackendLibrary::Instance>(  // B29-09 : la fenêtre pompée
      "backend", [&p, &init] { return BackendLibrary::create(p.library, init); });
  if (!backend) return stage_refused(std::move(backend.error()), std::nullopt);
  c.backend = std::move(*backend);
  c.backend_name = c.backend->capabilities().name;
  announce_backend(*log, *c.backend, c.backend_name, p.library->path());
  // Sous TensorRT les copies passent par CUDA : le BAR ne joue que pour le moteur Vulkan.
  if (const std::string rebar = rebar_text(gpu);
      c.backend_name != "tensorrt" && rebar.rfind("inactif", 0) == 0)
    log->warn("Resizable BAR inactif : activer Above 4G Decoding et Resizable BAR dans le BIOS");

  if (c.model) {
    auto upscaled = load_span(p, init, c, settings);
    if (!upscaled) return std::unexpected(std::move(upscaled.error()));
    if (!*upscaled) return std::nullopt;
  }
  // Au-delà de 720 lignes de travail (la 4K sans réduction), RIFE estime le mouvement sur une copie
  // réduite à 720p et le warp déforme les images de travail : la variante mouvement seulement.
  c.rife_size = !p.options.rife_image && c.work.height > kRifeMotionHeight
                    ? work_size(video.width, video.height, kRifeMotionHeight)
                    : c.work;
  InnerFallbacks inner(p.stages, c.stage_notes, c.fallback_fresh);
  const std::string library = library_name(p);
  if (want_drba) {
    auto drba = load_drba(p, init, c, settings, inner, library);
    if (!drba) return std::unexpected(std::move(drba.error()));
    if (!*drba) return std::nullopt;
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
    if (!loaded && watch_now(loaded.error()))
      return stage_refused(std::move(loaded.error()), ChainStage::Motion);
    if (!loaded) {
      log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
      p.ui.controls().warn(c.rife_model ? i18n::ui_tr("Apple interpolation unavailable: using RIFE")
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
    if (!planes) return stage_refused(std::move(planes.error()), ChainStage::Motion);
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
  // Rien (fenêtre fermée pendant la construction) : « arrêt » ; un refus : journal et RIFE entier ; une
  // annulation (« Annuler et regarder ») : l'interpolation part sans autre construction (A08-01, A31-15).
  enum class Fused : std::uint8_t { Loaded, Declined, Closed, Cancelled };
  std::optional<Refusal> cancelled;
  const auto try_fused = [&](const kernels::RifeGeometry& g, bool motion,
                             std::unique_ptr<gpu::Engine>& into) {
    if (!fused_wanted) return Fused::Declined;
    const std::string key = inner_key(motion ? "rife-fused motion" : "rife-fused", library, *c.rife_model,
                                      g.padded_width, g.padded_height);
    if (const Refusal* before = inner.refused(ChainStage::Motion, key)) {
      c.rife_fused_refused = before->code;  // le panneau dit encore pourquoi l'entier tourne
      return Fused::Declined;
    }
    auto fused = prepare_rife_fused(p, init, *c.backend, *c.rife_model, models_dir, g, motion);
    if (!fused && watch_now(fused.error())) {
      cancelled = std::move(fused.error());
      return Fused::Cancelled;
    }
    if (!fused) {
      log->warn("RIFE fusionné indisponible ({}) : RIFE entier", fused.error().detail);
      c.rife_fused_refused = fused.error().code;
      inner.failed(ChainStage::Motion, key, fused.error());
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
    } else if (tried == Fused::Cancelled) {
      return stage_refused(std::move(*cancelled), ChainStage::Motion);
    } else if (const std::string key = inner_key("rife-motion", library, *derived, c.geometry.padded_width,
                                                 c.geometry.padded_height);
               tried == Fused::Declined && inner.refused(ChainStage::Motion, key) != nullptr) {
      log->info("RIFE mouvement refusé plus tôt dans la séance : RIFE image");
    } else if (tried == Fused::Declined) {
      const kernels::RifeGeometry& g = c.geometry;
      auto loaded = prepare_engine(p, init, *c.backend, *derived,
                                   gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                                   EngineLabel{.name = "RIFE",
                                               .model = derived->stem().string(),
                                               .width = g.padded_width,
                                               .height = g.padded_height},
                                   gpu::Lane::Interpolate);
      if (!loaded && watch_now(loaded.error()))
        return stage_refused(std::move(loaded.error()), ChainStage::Motion);
      if (!loaded) {
        // La variante mouvement refusée (construction, mémoire) : RIFE image reste à essayer (A29-01).
        log->warn("RIFE mouvement refusé ({}) : RIFE image", loaded.error().detail);
        inner.failed(ChainStage::Motion, key, loaded.error());
      } else if (!*loaded) {
        return std::nullopt;
      } else {
        c.rife_motion = std::move(**loaded);
      }
    }
  }
  load_calm_rife(p, init, c, models_dir);
  if (use_rife && !c.rife_motion) {
    const kernels::RifeGeometry& g = c.image_geometry;
    const Fused tried = try_fused(g, false, c.rife);
    if (tried == Fused::Closed) return std::nullopt;
    if (tried == Fused::Cancelled) return stage_refused(std::move(*cancelled), ChainStage::Motion);
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
    if (!loaded) return stage_refused(std::move(loaded.error()), ChainStage::Motion);
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
