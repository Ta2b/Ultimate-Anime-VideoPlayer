// L'étage gpu (upscale_stage.h) : sa construction, ses tampons et son bilan au journal. Sorti
// d'upscale_stage.cpp (C09-09).
#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <string_view>
#include <utility>

#include "app/play/options.h"
#include "app/play/rife_plan.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "kernels/spec/rife.h"

namespace kirari::app::play {
// « profil hôte · attente … : médiane, p95 (nombre) » au journal de fin, si l'attente a été chronométrée.
void UpscaleStage::HostWaits::log(std::string_view what) {
  if (us.empty()) return;
  std::ranges::sort(us);
  log_channel("gpu")->info("profil hôte · attente {} : médiane {:.2f} ms, p95 {:.2f} ms ({})", what,
                           static_cast<double>(us[us.size() / 2]) / 1000.0,
                           static_cast<double>(us[us.size() * 95 / 100]) / 1000.0, us.size());
}
namespace {

gpu::Shape planes(int channels, int width, int height) {
  return gpu::Shape{.dims = {1, channels, height, width}};
}

// La sortie de l'étage : celle de SPAN, ou sans lui la taille de travail (les copies gradées, warps_spanned).
gpu::Shape output_shape(const UpscaleSetup& s) {
  const int factor = s.engine != nullptr ? kSpanFactor : 1;
  return planes(3, s.work_width * factor, s.work_height * factor);
}

// Sans SPAN, le grade avant le warp grade une copie de chaque source (warps_spanned), que RIFE ne lit pas.
// Ni GMFSS, ni les calques, ni Smooth, ni TemporalFix : ils lisent les sources de travail (l'écran grade).
bool graded_copies(const UpscaleSetup& s) {
  return s.engine == nullptr && s.grader != nullptr && !s.yuv_slots && s.gmfss.use == GmfssUse::Off &&
         !s.layers.layers && !s.layers.plate_on() && !s.smooth_drawings && s.temporal_fix == nullptr;
}

// L'écran grade chaque image de cette chaîne : le journal le dit, sans quoi seule l'absence de la ligne de
// fin du grade avant le warp le montre.
void note_screen_grade(const UpscaleSetup& s, bool interpolates, bool spanned) {
  if (s.grader == nullptr || !interpolates || spanned) return;
  log_channel("gpu")->info(
      "grade avant le warp : pas sur cette chaîne (GMFSS, calques, Smooth, TemporalFix ou warp avant SPAN), "
      "l'écran grade chaque image");
}

}  // namespace

UpscaleStage::UpscaleStage(VideoStageInit init, UpscaleSetup setup)
    : RingStage(
          std::move(init), static_cast<std::uint32_t>(setup.slots.size()), setup.slots.front().generation,
          smooth_extra_frames(setup.smooth_drawings) + temporal_extra_frames(setup.temporal_fix != nullptr)),
      setup_(std::move(setup)),
      scene_(setup_.scene_map, setup_.nvof_roundtrip) {
  graded_copies_ = graded_copies(setup_);
  note_screen_grade(setup_, interpolates(), warps_spanned());
  reuse_.enabled = setup_.span_reuse.whole;
  triage_.for_gmfss(gmfss_only());
  reuse_.tiles.enabled = setup_.span_reuse.tiles;
  reuse_.check = setup_.span_reuse.check;
  if (setup_.engine != nullptr) reuse_.tiles.set_grid(setup_.engine->span_tile_grid());
  // SPAN seul : la paire de la reprise ou des tuiles se mesure en avance (look_ahead), comme sous RIFE.
  reuse_.alone = setup_.engine != nullptr && (reuse_.enabled || reuse_.tiles.active()) && !interpolates() &&
                 setup_.distil_drba == nullptr;
  // Ce que les tuiles feront, dit au démarrage quand elles sont demandées : leur grille, ou pourquoi ce SPAN
  // n'en calcule pas.
  if (setup_.engine != nullptr && reuse_.tiles.enabled) {
    const kernels::SpanTileGrid& grid = reuse_.tiles.grid();
    if (reuse_.tiles.active())
      log_channel("gpu")->info("SPAN par tuiles : grille de {} × {} tuiles de {} × {} pixels de travail",
                               grid.cols, grid.rows, grid.tile_w, grid.tile_h);
    else
      log_channel("gpu")->info(
          "SPAN par tuiles : ce moteur ne calcule que l'image entière (seul le SPAN natif les calcule)");
  }
}

UpscaleStage::~UpscaleStage() {
  stop();
  log_temporal();
  log_shots();
  scene_.log_summary();
  layers_.log_summary(setup_.layers);
  if (fences_mapped_) setup_.backend->kernels()->unmap_fences(setup_.yuv_slots->fence_block());
  if (setup_.rife_motion != nullptr) {
    log_channel("gpu")->info(
        "RIFE mouvement : {} passes ; {} paires à une passe, {} à plusieurs (dont {} sans passe au milieu)",
        rife_runs_, single_pairs_, multi_pairs_, direct_pairs_);
  }
  triage_.log();  // --pair-triage : la ligne à comparer d'une séance à l'autre (rife_triage.h)
  if (strong_derived_ != 0)
    log_channel("gpu")->info("paires fortes : {} instants déduits du milieu (moins de passes RIFE)",
                             strong_derived_);
  if (setup_.rife_motion_calm != nullptr)
    log_channel("gpu")->info("RIFE des paires calmes : {} passes sur {}", calm_runs_, rife_runs_);
  if (halves())
    log_channel("gpu")->info("échelle par paire : {} passes à demi-taille sur {}", half_.runs, rife_runs_);
  if (setup_.gmfss.use != GmfssUse::Off) {
    log_channel("gpu")->info(
        "GMFSS union : {} paires, {} fusions, {} passes de RIFE entre clés, {} paires fortes laissées à "
        "RIFE en retard, {} clés sous le seuil, {} clés de plus sur les grands mouvements, {} paires en "
        "avance dont {} reprises et {} sans clé, {} clés composées, {} paires par la caméra ({} images)",
        gmfss_.pairs + gmfss_.ahead_pairs, gmfss_.fusions, gmfss_.halves, gmfss_.behind_pairs,
        gmfss_.spare_keys, gmfss_.more_keys, gmfss_.ahead_pairs, gmfss_.ahead_used, gmfss_.ahead_wasted,
        gmfss_.composed, gmfss_.camera_pairs, gmfss_.camera_images);
    if (gmfss_only())
      log_channel("gpu")->info(
          "GMFSS seul : {} avances reprises sans instant à temps, {} avances évitées (échéance passée)",
          gmfss_.ahead_adopted_all_late, gmfss_.ahead_declined_expired);
  }
  // RIFE à chaud, un compromis qualité contre vitesse (B45-07) : le milieu d'une paire calme qui accélère
  // garde l'échelle grossière de la paire d'avant. Ni le gain ni la perte ne sont mesurés : l'essai reste
  // hors défaut.
  if (setup_.rife_warm)
    log_channel("gpu")->info(
        "RIFE à chaud (qualité et gain non mesurés) : {} passes sur {} sans les blocs "
        "grossiers{}",
        warm_runs_, rife_runs_,
        warm_runs_ == 0 && rife_runs_ > 0
            ? " (ce moteur ne reprend pas à chaud : l'essai est sans effet hors du RIFE "
              "fusionné de Vulkan)"
            : "");
  if (moving_.pairs > 0) {
    const auto pairs = static_cast<double>(moving_.pairs);
    log_channel("gpu")->info(
        "Zone qui bouge sur {} paires : blocs changés {:.0f} %, boîte {:.0f} %, boîte sous la moitié {} fois",
        moving_.pairs, 100.0 * moving_.block_sum / pairs, 100.0 * moving_.box_sum / pairs,
        moving_.small_boxes);
  }
  if (const std::string reuse = reuse_.summary(); !reuse.empty()) log_channel("gpu")->info("{}", reuse);
  if (const std::string tiles = reuse_.tiles.summary(); !tiles.empty()) log_channel("gpu")->info("{}", tiles);
  // Le travail d'une image ne s'enfile qu'avec un slot libre : ces attentes sont du temps où l'étage ne
  // nourrissait pas le GPU parce que l'écran n'avait pas encore rendu de slot.
  if (const SlotWaits& waits = slot_waits(); waits.count > 0)
    log_channel("gpu")->info(
        "anneau : {} attentes d'un slot libre, {:.1f} ms en tout, la plus longue {:.1f} ms", waits.count,
        static_cast<double>(waits.total_us) / 1000.0, static_cast<double>(waits.longest_us) / 1000.0);
  if (early_.passes > 0)
    log_channel("gpu")->info("RIFE : {} passes enfilées à l'arrivée de {} paires, sans attendre leur slot",
                             early_.passes, early_.pairs);
  if (setup_.span_b_ahead)
    log_channel("gpu")->info(
        "SPAN B : {} lancés avant le premier slot, {} parvenus à une soumission d'écriture, "
        "{:.1f} ms d'avance hôte cumulée jusqu'à cette soumission",
        span_body_ahead_, span_body_seen_, static_cast<double>(span_body_headroom_us_) / 1000.0);
  if (encodes_kept_ + encodes_followed_ + encodes_after_gap_ + encodes_cold_ > 0)
    log_channel("gpu")->info(
        "RIFE : encodage repris sur {} passes, B seule sur {}, A et B sur {} après une paire sans passe et "
        "{} "
        "à froid",
        encodes_kept_, encodes_followed_, encodes_after_gap_, encodes_cold_);
  if (setup_.distil_drba != nullptr) {
    log_channel("gpu")->info(
        "DistilDRBA : {} passes, dont {} sans l'image d'un côté (début, coupe) ; {} instants sans passe "
        "(côté tenu)",
        drba_.runs, drba_.edges, drba_.held);
  }
  if (setup_.smooth_drawings) {
    const auto& h = smoothed_holds_;
    log_channel("gpu")->info(
        "Smooth : {} tenues fondues en un mouvement (sur 2 sources : {}, 3 : {}, 4 : {}, 5 : {}, 6 : {})",
        h[2] + h[3] + h[4] + h[5] + h[6], h[2], h[3], h[4], h[5], h[6]);
  }
  const char* motion = gmfss_only()                                            ? "GMFSS union"
                       : setup_.distil_drba != nullptr && !setup_.drba_by_shot ? "DistilDRBA"
                                                                               : "RIFE";
  if (setup_.late_fate) {
    log_channel("gpu")->info(
        "Interpolation : {} instants laissés en retard sans calcul, {} calculés après leur instant",
        late_dropped_, late_computed_);
  }
  if (late_pairs_ > 0) {
    log_channel("gpu")->info("{} : {} paires en retard sur l'horloge, {} instants sautés pour la rattraper",
                             motion, late_pairs_, late_instants_);
  }
  if (skipped_ > 0) {
    log_channel("gpu")->info(
        "{} images source sautées avant calcul, déjà passées : la chaîne est plus lente "
        "que la vidéo",
        skipped_);
  }
  if (interpolates()) {
    const std::string cadence =
        setup_.live && setup_.live->free_active()
            ? "Auto"
            : factor_text(setup_.live ? setup_.live->factor_now() : setup_.rife_factor);
    log_channel("gpu")->info("{} {} : {} paires, {} interpolées, {} coupes de scène, {} dessins tenus",
                             motion, cadence, pairs_, interpolated_, cuts_, held_);
  }
  if (setup_.live && setup_.live->cuts().active())
    log_channel("gpu")->info(
        "coupes : {} désaccords entre RIFE et les plans de la pré-analyse, {} coupes de RIFE en rafale "
        "écartées",
        cut_disagreements_, burst_cuts_);
  scores_waits_.log("des mesures de la paire");
  motion_waits_.log("du mouvement");
}

std::expected<std::unique_ptr<VideoStage>, Refusal> UpscaleStage::start(VideoStageInit init,
                                                                        UpscaleSetup setup) {
  if (setup.backend == nullptr || setup.backend->kernels() == nullptr) {
    return refuse(RefusalCode::FieldMissing, "étage gpu : backend sans noyaux");
  }
  if (setup.work_width == 0 && setup.work_height == 0) {
    setup.work_width = setup.source_width;
    setup.work_height = setup.source_height;
  }
  if (setup.work_width < 1 || setup.work_height < 1 || setup.work_width > setup.source_width ||
      setup.work_height > setup.source_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("étage gpu : taille de travail {}x{} hors de la source {}x{}", setup.work_width,
                              setup.work_height, setup.source_width, setup.source_height));
  }
  if (auto ok = check_by_shot(setup); !ok) return std::unexpected(std::move(ok.error()));
  const int estimators = estimators_of(setup);  // plan par plan : DistilDRBA et RIFE comptent pour un
  if (estimators > 1) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "étage gpu : RIFE image, RIFE mouvement ou DistilDRBA, un seul des trois");
  }
  if ((setup.rife != nullptr || setup.rife_motion != nullptr) &&
      !kernels::is_rife_input_planes(setup.rife_planes))
    return refuse(RefusalCode::InputMismatch,
                  std::format("étage gpu : RIFE à {} plans d'entrée", setup.rife_planes));
  if (estimators == 1 && setup.live == nullptr &&
      (setup.rife_factor.num <= setup.rife_factor.den || !valid_factor(setup.rife_factor))) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("étage gpu : RIFE {} hors de ]1, {}]",
                                                            factor_text(setup.rife_factor), kMaxRifeFactor));
  }
  if (setup.rife_width == 0 && setup.rife_height == 0) {
    setup.rife_width = setup.work_width;
    setup.rife_height = setup.work_height;
  }
  if (setup.rife_width < 1 || setup.rife_height < 1 || setup.rife_width > setup.work_width ||
      setup.rife_height > setup.work_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("étage gpu : RIFE en {}x{} hors du travail {}x{}", setup.rife_width,
                              setup.rife_height, setup.work_width, setup.work_height));
  }
  if ((setup.rife != nullptr || (setup.distil_drba != nullptr && !setup.drba_motion)) &&
      (setup.rife_width != setup.work_width || setup.rife_height != setup.work_height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "étage gpu : RIFE image et DistilDRBA image à la taille de travail seulement");
  }
  if (setup.slots.empty() || setup.slots.size() > kMaxRingSlots) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("étage gpu : 1 à {} slots", kMaxRingSlots));
  }
  const int factor = setup.engine != nullptr ? kSpanFactor : 1;
  for (const RingSlotDesc& slot : setup.slots) {
    if (slot.width != setup.work_width * factor || slot.height != setup.work_height * factor) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("slot {} : {}x{} au lieu de la sortie {}x{}", slot.index, slot.width,
                                slot.height, setup.work_width * factor, setup.work_height * factor));
    }
  }
  std::unique_ptr<UpscaleStage> stage(new UpscaleStage(std::move(init), std::move(setup)));
  if (auto ok = stage->allocate(); !ok) return std::unexpected(std::move(ok.error()));
  if (auto started = stage->RingStage::start("gpu"); !started)
    return std::unexpected(std::move(started.error()));
  return stage;
}

void UpscaleStage::map_yuv_fences() {
  log_channel("gpu")->info("images écrites en YUV dans celles de l'encodeur");
  const std::span<std::byte> block = setup_.yuv_slots->fence_block();
  fences_mapped_ = !block.empty() && setup_.backend->kernels()->map_fences(block);
  if (fences_mapped_) log_channel("gpu")->info("signaux de l'export écrits par le GPU, sans rappel hôte");
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<void, Refusal> UpscaleStage::allocate() {
  gpu::Backend& backend = *setup_.backend;
  const gpu::BackendCaps caps = backend.capabilities();
  copies_ = caps.lanes > 1 && !setup_.one_lane ? 2U : 1U;
  if (setup_.yuv_slots && !caps.yuv_targets) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("étage gpu : le backend {} n'écrit pas en YUV pour l'encodeur", caps.name));
  }
  // Calques : l'image intermédiaire passe par un tenseur, composée sur place avant l'écriture du slot.
  ring_direct_ =
      layers_ring_direct(setup_.layers, caps.ring_targets && !setup_.yuv_slots &&
                                            (warps_spanned() || (by_motion() && setup_.engine == nullptr)));
  if (ring_direct_)
    log_channel("gpu")->info("images intermédiaires écrites directement dans le slot du ring");
  if (setup_.yuv_slots) map_yuv_fences();
  const int w = setup_.work_width;
  const int h = setup_.work_height;
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, const gpu::Shape& shape) {
    auto tensor = backend.alloc_tensor(gpu::TensorType::F16, shape);
    if (tensor) into = std::move(*tensor);
    return tensor ? std::expected<void, Refusal>{} : std::unexpected(std::move(tensor.error()));
  };
  for (auto& rgb : rgb_) {
    if (auto ok = alloc(rgb, planes(3, w, h)); !ok) return ok;
  }
  const gpu::Shape out_shape = output_shape(setup_);
  if (warps_spanned()) {
    for (auto& spanned : spanned_) {
      if (auto ok = alloc(spanned, out_shape); !ok) return ok;
    }
    for (std::uint32_t i = 0; i < copies_ && !ring_direct_; ++i) {
      if (auto ok = alloc(middle_outs_[i], out_shape); !ok) return ok;
    }
  } else if (setup_.engine != nullptr) {
    if (auto ok = alloc(output_, out_shape); !ok) return ok;
  }
  if (interpolates() && (setup_.rife_width != w || setup_.rife_height != h)) {
    for (auto& small : small_) {
      if (auto ok = alloc(small, planes(3, setup_.rife_width, setup_.rife_height)); !ok) return ok;
    }
  }
  if (interpolates() && !warps_spanned() && !ring_direct_) {
    for (std::uint32_t i = 0; i < copies_; ++i) {
      if (auto ok = alloc(middles_[i], planes(3, w, h)); !ok) return ok;
    }
  }
  if (auto ok = layers_.allocate(backend, setup_.layers, planes(3, w, h), out_shape, warps_spanned()); !ok)
    return ok;
  if (auto ok = allocate_temporal(); !ok) return ok;
  if (looks_ahead()) {  // plan par plan aussi : RIFE d'abord, DistilDRBA avec (allocate_ahead)
    if (auto ok = allocate_ahead(); !ok) return ok;
  }
  if (drba()) return allocate_drba();
  if (interpolates()) {
    const kernels::RifeGeometry g =
        kernels::rife_geometry(setup_.rife_width, setup_.rife_height, setup_.rife_align);
    for (std::uint32_t i = 0; i < copies_; ++i) {
      if (auto ok = alloc(rife_ins_[i], planes(setup_.rife_planes, g.padded_width, g.padded_height)); !ok)
        return ok;
    }
    if (setup_.rife != nullptr) {
      if (auto ok = alloc(rife_out_, planes(3, g.padded_width, g.padded_height)); !ok) return ok;
    } else {
      for (MotionBuffers& mid : mids_) {
        if (auto ok = alloc(mid.flow, planes(4, g.padded_width, g.padded_height)); !ok) return ok;
        if (auto ok = alloc(mid.mask, planes(1, g.padded_width, g.padded_height)); !ok) return ok;
      }
      if (auto ok = alloc(flow_, planes(4, g.padded_width, g.padded_height)); !ok) return ok;
      if (auto ok = alloc(mask_, planes(1, g.padded_width, g.padded_height)); !ok) return ok;
    }
  }
  return allocate_gmfss();  // après allocate_half
}

}  // namespace kirari::app::play
