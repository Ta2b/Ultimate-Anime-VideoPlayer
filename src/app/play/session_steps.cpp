// La session hors du choix d'image : reprise, image par image, garde de l'horloge et fin de lecture (sortis
// de session.cpp, C13-09).
#include <algorithm>
#include <utility>

#include "app/play/i18n.h"
#include "app/play/session.h"

namespace ajp::app::play {
namespace {

// Fin de flux : plus rien n'arrive depuis ce délai, la lecture est finie (si l'horloge a passé la fin).
constexpr std::int64_t kEndGraceUs = 1'000'000;
// Au plus ce temps après la dernière image neuve, même si l'horloge n'a pas atteint la durée annoncée.
constexpr std::int64_t kEndWaitMaxUs = 60'000'000;
// Des images attendent l'horloge audio depuis 1,5 s (Session::audio_stall_grace_us_) : l'horloge
// hôte prend la main. Après une première perte, le délai est plus court : la piste est
// probablement illisible.
constexpr std::int64_t kAudioStallRepeatUs = 400'000;
constexpr std::int64_t kAssumedFramePeriodUs = 41'667;  // une image de la source à 24 i/s, faute de mieux

}  // namespace

void Session::resume() {
  if (!paused_) return;
  paused_ = false;
  // A13-07 : des pas pris dans la file n'ont pas déplacé l'audio, resté à la pause. La lecture repart par un
  // seek exact sur l'image montrée, comme après un pas en arrière.
  if (std::exchange(stepped_in_queue_, false) && !prerolling_ && current_ && current_->pts_us != kNoPts) {
    seek_to(current_->pts_us, true);
    return;
  }
  if (prerolling_ || start_pts_ || !current_) return;
  // C13-02 : l'horloge repart d'où elle s'est figée, et non du pts de l'image montrée, choisie jusqu'à une
  // demi-période et le délai d'affichage en avance : l'horloge hôte qui mène sautait de −24 à +18 ms à chaque
  // reprise, et l'audio rattrapait l'écart en une quinzaine de secondes. Après un seek fait en pause,
  // l'horloge n'est pas de cette époque : le pts de l'image.
  const ClockSnapshot frozen = clock_.snapshot();
  const bool held = frozen.valid && frozen.epoch == epoch_ && frozen.rate == 0.0;
  send_play(epoch_, held ? frozen.media_us : current_->pts_us);
}

// Image par image : en pause, un seek exact d'une image de la source. Le seek part de la position visée, donc
// des pressions rapprochées avancent d'autant. P3.42 : vers l'avant, l'image suivante déjà dans la file, sans
// seek.
void Session::step_frame(int direction) {
  if (!paused_) command(cmd::Pause{});
  if (direction > 0 && step_from_queue()) return;
  const std::int64_t period = source_period();
  // P3.78 : la cible est le milieu de l'image voisine, pas son début : des pts arrondis à la microseconde
  // (23,976 i/s) ne font plus tomber le seek exact deux images plus loin. Une cible en attente est déjà un
  // milieu.
  std::int64_t base = 0;
  int hop = direction;
  if (pending_seek_us_) {
    base = *pending_seek_us_ - period / 2;
  } else if (current_ && current_->pts_us != kNoPts) {
    // B13-01 : d'une image interpolée, le pas part de sa source de gauche : sans cela, la cible tombait une
    // source plus loin (t = 1/2 + une période et demie : la source d'après la suivante).
    base = current_->pts_us;
    if (current_->t_num > 0 && current_->t_den > 0) {
      base -= period * current_->t_num / current_->t_den;
      // C13-07 : en arrière, l'image précédente est cette source de gauche elle-même, montrée juste avant.
      if (direction < 0) hop = 0;
    }
    // A13-05 : le pas d'avant est retombé dans l'image montrée (elle dure plus que la période moyenne, en
    // VFR) ; le suivant part de sa cible, sans quoi rien ne bougerait jamais.
    if (direction > 0 && last_step_ && last_step_->from_pts_us == base)
      base = std::max(base, last_step_->target_us - period / 2);
  }
  const std::int64_t target = std::max<std::int64_t>(base + hop * period + period / 2, 0);
  const std::optional<std::int64_t> from =
      current_ ? std::optional(current_->pts_us) : std::optional<std::int64_t>();
  seek_to(target, true);
  if (direction > 0 && from) last_step_ = StepMark{*from, target};
}

bool Session::step_from_queue() {
  if (pending_seek_us_ || prerolling_ || !current_ || current_->pts_us == kNoPts) return false;
  const std::int64_t period = source_period();
  const RingFrame shown = *current_;
  // B13-01 : la source suivante (t = 0), au plus une période et un quart plus loin ; les images interpolées
  // d'entre deux sont passées. Elle et celles d'avant doivent être écrites, sinon le seek exact la donne.
  const auto ready = [this](const RingFrame& f) { return screen_.ready(f); };
  const RingFrame* source =
      queue_.written_source_after(epoch_, shown.pts_us, shown.pts_us + period + period / 4, ready);
  if (source == nullptr) return false;
  late_.clear();
  const std::optional<RingFrame> next = queue_.pick(source->pts_us, 0, late_, ready);
  for (const RingFrame& frame : late_) retire(frame);
  late_.clear();
  if (!next) return false;
  retire(shown);
  current_ = next;
  target_host_us_ = 0;       // B13-05 : la date visée de l'image d'avant ne vaut pas pour celle-ci
  stepped_ = true;           // montrée au prochain tour
  stepped_in_queue_ = true;  // l'audio est resté à la pause : la reprise le recale (A13-07)
  return true;
}

// Des images attendent une horloge qui ne court pas : l'audio maître n'en fournit plus (piste
// illisible, trou, début tardif, fin avant la vidéo). Passé le délai, l'horloge hôte mène et la
// vidéo repart ; sans cela elle resterait figée, la source bloquée sur l'image tenue.
void Session::watch_clock(const ClockSnapshot& now, std::int64_t host_now) {
  const bool running = now.valid && now.epoch == epoch_ && now.rate != 0.0;
  if (running || paused_ || current_ == std::nullopt || queue_.first_of(epoch_) == nullptr) {
    clock_stalled_since_us_.reset();
    return;
  }
  if (!clock_stalled_since_us_) {
    clock_stalled_since_us_ = host_now;
    return;
  }
  // P3.75 : passé la fin connue de la piste audio, l'horloge hôte mène tout de suite et sans message : ce
  // n'est pas une panne, la vidéo continue seule.
  const bool audio_over = config_.audio_end_us > 0 && current_->pts_us >= config_.audio_end_us - 200'000;
  if (host_now - *clock_stalled_since_us_ < (audio_over ? 0 : audio_stall_grace_us_)) return;
  clock_stalled_since_us_.reset();
  if (!stages_.audio_lost(epoch_, current_->pts_us) || audio_over) return;
  audio_stall_grace_us_ = kAudioStallRepeatUs;  // l'audio a déjà manqué : la prochaine fois, plus vite
  on_refusal({.stage = "main",
              .refusal = {RefusalCode::DeviceUnavailable, i18n::tr("audio missing: the display clock leads")},
              .fatal = false});
}

void Session::check_end(std::int64_t host_now) {
  if (playing_since_us_) {
    const std::int64_t played = host_now - *playing_since_us_;
    // --seconds atteint : un arrêt demandé, sans épisode suivant (A13-14).
    if (config_.stop_after_us > 0 && played >= config_.stop_after_us) finished_ = quit_ = true;
    if (config_.fullscreen_after_us > 0 && !fullscreen_done_ && played >= config_.fullscreen_after_us) {
      screen_.set_fullscreen(true);
      fullscreen_done_ = true;
    }
  }
  if (plan_.active() && current_ && current_->epoch == epoch_ && current_->pts_us != kNoPts) {
    const EditionPlan::Step step = plan_.shown(current_->pts_us, pacing_period_us());
    if (const auto* jump = std::get_if<EditionPlan::Jump>(&step)) {
      plan_jump_ = true;
      seek_to(jump->to_us, true);
      plan_jump_ = false;
    } else if (std::holds_alternative<EditionPlan::End>(step)) {
      finished_ = true;
    }
  }
  if (stream_ended_ && !paused_ && queue_.empty() && stages_.idle() &&
      host_now - last_fresh_host_us_ > kEndGraceUs && media_over(host_now)) {
    finished_ = true;
  }
}

// C13-03 : la fin tombe quand l'horloge a passé la dernière image (son pts et sa durée) et la durée du
// fichier, qui couvre un son plus long que la vidéo ou un carton de fin tenu plusieurs secondes. Horloge
// arrêtée ou d'une autre époque : la grâce d'une seconde suffit. Une durée annoncée fausse ne retient pas la
// fin plus de kEndWaitMaxUs.
bool Session::media_over(std::int64_t host_now) const {
  if (host_now - last_fresh_host_us_ > kEndWaitMaxUs) return true;
  const ClockSnapshot now = clock_.snapshot();
  if (!now.valid || now.epoch != epoch_ || now.rate <= 0.0 || !current_ || current_->pts_us == kNoPts)
    return true;
  std::int64_t end_us = current_->pts_us + source_period();
  // Une édition ordonnée (EditionPlan) a sa propre fin ; la durée du fichier n'y vaut pas.
  if (!plan_.active()) end_us = std::max(end_us, config_.duration_us);
  return now.media_at(host_now) >= end_us;
}

std::int64_t Session::source_period() const {
  const std::int64_t period = config_.extra != nullptr ? config_.extra->source_period_us() : 0;
  return period > 0 ? period : kAssumedFramePeriodUs;
}

}  // namespace ajp::app::play
