// Apprentissage du grade adaptatif pendant la lecture (grade_learning.h) : les corrections de l'utilisateur
// et ses approbations silencieuses s'écrivent dans adaptive-learning.tsv ; « Apprendre » en tire la courbe.
#include <ctime>
#include <format>
#include <string>
#include <system_error>

#include "app/play/calibration_images.h"
#include "app/play/chain.h"
#include "app/play/i18n.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSettleUs = 3'000'000;    // une correction est retenue une fois les curseurs posés
constexpr std::int64_t kApproveUs = 30'000'000;  // une scène regardée sans rien changer vaut approbation

void count_samples(LearnTrack& l) {
  l.corrections = 0;
  l.approvals = 0;
  for (const LearnSample& s : read_learn_samples(l.log)) {
    if (s.kind == LearnSample::Kind::Approval) {
      ++l.approvals;
    } else if (s.touched != 0) {
      ++l.corrections;
    }
  }
}

void write(LearnTrack& l, LearnSample sample, const GradeCurve& curve, const SceneMix& mix) {
  sample.unix_s = static_cast<std::int64_t>(std::time(nullptr));
  sample.look = look_at(curve, mix.t, mix.glow, mix.tv);
  append_learn_sample(l.log, sample);
  ++(sample.kind == LearnSample::Kind::Approval ? l.approvals : l.corrections);
}

}  // namespace

void learn_tick(Chain& to, std::int64_t now_us) {
  LearnTrack& l = to.learn;
  if (l.log.empty() || to.adaptive_luma < 0.0) return;
  if (l.corrections < 0) count_samples(l);
  const double force = to.running.grade_values.adaptive_force;
  const SceneMix mix{.t = to.adaptive_t, .glow = to.adaptive_glow, .tv = to.adaptive_tv};
  const LearnScene scene{.x = to.adaptive_luma / series_ratio(to.series_luma),
                         .glow_seen = to.adaptive_glow_seen,
                         .tv = to.adaptive_tv,
                         .force = force};
  to.ui.set_adaptive_scene(AdaptiveScene{.live = force < 0.0,
                                         .t = mix.t,
                                         .glow = mix.glow,
                                         .tv = mix.tv,
                                         .x = scene.x,
                                         .glow_seen = scene.glow_seen,
                                         .corrections = l.corrections,
                                         .approvals = l.approvals});
  // La vidéo de calibration : ses réglages vont à la calibration (settings_calibration.cpp), pas au journal.
  if (to.file == calibration_clip()) {
    l.seen.reset();
    l.pending.reset();
    return;
  }
  const GradeCurve curve = curve_of(to.running);
  if (!l.seen) {
    l.seen = curve;
    l.still_us = now_us;
    return;
  }
  if (const std::uint8_t touched = touched_families(*l.seen, curve); touched != 0) {
    l.seen = curve;
    l.still_us = now_us;
    // Plusieurs familles d'un coup : un profil, un préréglage ou un ajustement, pas une main sur un curseur.
    if (touched != kShadowsFamily && touched != kColorsFamily && touched != kGrayFamily) {
      l.pending.reset();
      return;
    }
    if (!l.pending) l.pending.emplace();
    l.pending->kind = force >= 0.0 ? LearnSample::Kind::Calibration : LearnSample::Kind::Correction;
    l.pending->touched |= touched;
    l.pending->scene = scene;  // la scène que l'utilisateur regarde en réglant
    l.pending_mix = mix;
    l.pending_us = now_us;
    return;
  }
  if (l.pending) {
    if (now_us - l.pending_us < kSettleUs) return;
    write(l, *l.pending, curve, l.pending_mix);
    l.pending.reset();
    l.still_us = now_us;
    return;
  }
  if (force >= 0.0 || to.session.view().paused) {
    l.still_us = now_us;  // une pause ou une calibration n'approuve rien
    return;
  }
  if (now_us - l.still_us < kApproveUs) return;
  write(l, LearnSample{.kind = LearnSample::Kind::Approval, .scene = scene}, curve, mix);
  l.still_us = now_us;
}

std::optional<PlaySettings> learned_settings(Chain& to, std::int64_t now_us) {
  Controls& controls = to.ui.controls();
  if (to.learn.log.empty()) return std::nullopt;
  const std::vector<LearnSample> samples = read_learn_samples(to.learn.log);
  const GradeCurve now = curve_of(to.running);
  auto report = learn_grade(samples, now, static_cast<std::int64_t>(std::time(nullptr)));
  if (!report) {
    controls.notify(report.error(), now_us);
    return std::nullopt;
  }
  const GradeCurve& c = report->curve;
  // Le journal : en clair pour le développeur (sans accents, une ligne par sujet).
  const auto log = log_channel("app");
  log->info("grade appris : {} corrections, {} approbations, poids sombre {:.1f} halo {:.1f} clair {:.1f}",
            report->corrections, report->approvals, report->support_dark, report->support_glow,
            report->support_bright);
  log->info("grade appris : erreur {:.1f} % puis {:.1f} %, seuils {:.0f}/{:.0f} puis {:.0f}/{:.0f}",
            100.0 * report->error_before, 100.0 * report->error_after, now.lo, now.hi, c.lo, c.hi);
  log->info("grade appris : noirs {:.2f}/{:.2f}/{:.2f}, gris {:.0f}/{:.0f}/{:.0f}", c.shadows,
            c.shadows_bright, c.shadows_glow, c.gray, c.gray_bright, c.gray_glow);
  log->info("grade appris : couleurs {:.2f}, coupe {:.2f}, halo {:.2f}, ciel {:.2f}", c.colors, c.colors_cut,
            c.colors_glow, c.colors_sky);
  PlaySettings next = to.running;
  set_curve(next, c);
  // Le grade calibré de l'écran se rouvre à chaque lecture (with_calibrated_grade) : il reçoit la courbe
  // apprise, sinon elle serait perdue à la prochaine vidéo.
  if (to.display) {
    const std::string key = display_key(*to.display);
    if (const PlaySettings::DisplayGrade* kept = calibrated_grade(next, key); kept != nullptr) {
      const std::string name = kept->name;
      keep_calibrated_grade(next, key, name);
    }
  }
  controls.notify(
      i18n::tr_format("Grade learned from {} corrections: gap {:.0f} % → {:.0f} %", report->corrections,
                      100.0 * report->error_before, 100.0 * report->error_after),
      now_us);
  to.learn.seen = curve_of(next);  // l'ajustement n'est pas une correction
  to.learn.pending.reset();
  return next;
}

void forget_learning(Chain& to, std::int64_t now_us) {
  if (to.learn.log.empty()) return;
  std::error_code error;
  std::filesystem::remove(to.learn.log, error);
  to.learn.corrections = 0;
  to.learn.approvals = 0;
  to.learn.pending.reset();
  to.ui.controls().notify(i18n::tr("Grade corrections forgotten"), now_us);
}

}  // namespace ajp::app::play
