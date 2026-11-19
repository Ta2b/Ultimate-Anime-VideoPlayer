// L'Auto plan par plan dans l'étage gpu (shot_plan.h) : DistilDRBA et RIFE mouvement chargés ensemble, l'un
// ou l'autre selon le plan de l'image qui arrive, et TemporalFix seulement sur les plans qui le demandent.
//
// Le moteur ne change qu'à une coupe connue des plans (ou au début d'une époque) : la paire qui enjambe la
// coupe n'est pas interpolée, aucun des deux moteurs ne lit les images de l'autre plan. Ce que le moteur
// quitté avait lancé en avance pour l'image qui arrive (DistilDRBA : la suivante ingérée et mesurée dans
// drba_.after ; RIFE : ahead_ et la mesure du mouvement) est relu et oublié ; l'image s'ingère de nouveau.
#include <utility>

#include "app/play/upscale_stage.h"
#include "core/log.h"

namespace kirari::app::play {

std::expected<void, Refusal> UpscaleStage::follow_shot(const DecodedFrame& frame) {
  if (!setup_.drba_by_shot || setup_.live == nullptr || frame.pts_us == kNoPts) return {};
  const ShotPlanBook& book = setup_.live->shot_plan();
  const std::optional<ShotChoice> choice = book.at(frame.pts_us);
  if (!choice) {  // pas encore analysé : le moteur reste
    ++shot_.unknown_frames;
  } else if (const bool want = choice->motion == ShotMotion::Drba; want != shot_.drba) {
    const bool fresh = !previous_ || previous_->epoch != frame.epoch || previous_->pts_us == kNoPts;
    const bool cut = !fresh && book.cut_between(previous_->pts_us, frame.pts_us).value_or(false);
    if (fresh || cut) {
      gpu::KernelOps& ops = *setup_.backend->kernels();
      if (shot_.drba) {  // DistilDRBA → RIFE : la suivante lue en avance par DistilDRBA ne sert plus
        for (auto* ticket : {&drba_.after_ticket, &drba_.bridge_ticket}) {
          if (const auto pending = std::exchange(*ticket, std::nullopt)) {
            if (auto taken = ops.scores_take(*pending); !taken) return std::unexpected(std::move(taken.error()));
          }
        }
        drba_.after_scores.reset();
        drba_.after_id.reset();
        drba_.bridge = false;
        drba_.continuous = false;
        drba_.small_taken = false;
      } else {  // RIFE → DistilDRBA : la suivante de RIFE et sa mesure du mouvement non plus
        if (const std::uint64_t pending = std::exchange(ahead_.ticket, 0); pending != 0) {
          if (auto taken = ops.scores_take(pending); !taken) return std::unexpected(std::move(taken.error()));
        }
        ahead_.id.reset();
        if (auto ok = settle_motion(); !ok) return ok;
      }
      forget_motion();
      log_channel("gpu")->info("plan par plan : {} à {:.1f} s, {}", want ? "DistilDRBA" : "RIFE",
                               static_cast<double>(frame.pts_us) / 1e6, book.describe(frame.pts_us));
      shot_.drba = want;
      ++(want ? shot_.to_drba : shot_.to_rife);
    } else {
      ++shot_.held_back;  // jamais au milieu d'un plan
    }
  }
  ++(shot_.drba ? shot_.drba_frames : shot_.rife_frames);
  return {};
}

bool UpscaleStage::temporal_wanted(const DecodedFrame& frame) {
  if (!setup_.temporal_by_shot || setup_.live == nullptr) return true;
  // Un plan pas encore analysé garde TemporalFix : le scintillement coûte plus qu'une passe de trop.
  const std::optional<ShotChoice> choice = setup_.live->shot_plan().at(frame.pts_us);
  const bool wanted = !choice || choice->temporal;
  ++(wanted ? shot_.temporal_frames : shot_.temporal_skipped);
  return wanted;
}

void UpscaleStage::log_shots() const {
  if (setup_.drba_by_shot) {
    log_channel("gpu")->info(
        "plan par plan : {} images sous DistilDRBA, {} sous RIFE ({} avant l'analyse de leur plan) ; {} "
        "bascules vers DistilDRBA, {} vers RIFE, {} gardées faute de coupe connue",
        shot_.drba_frames, shot_.rife_frames, shot_.unknown_frames, shot_.to_drba, shot_.to_rife,
        shot_.held_back);
  }
  if (setup_.temporal_by_shot) {
    log_channel("gpu")->info("plan par plan : TemporalFix sur {} ingestions, sauté sur {}", shot_.temporal_frames,
                             shot_.temporal_skipped);
  }
}

}  // namespace kirari::app::play
