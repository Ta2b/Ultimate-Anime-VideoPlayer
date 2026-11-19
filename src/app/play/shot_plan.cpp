#include "app/play/shot_plan.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

#include "app/play/cut_book.h"

namespace kirari::app::play {

ShotChoice judge_shot(const media::ShotRhythm& rhythm) {
  ShotChoice choice;
  const float cadence = media::drawing_cadence(rhythm);
  const float strong =
      rhythm.pairs > 0 ? static_cast<float>(rhythm.strong) / static_cast<float>(rhythm.pairs) : 0.0F;
  // Le sakuga d'abord : un passage en deux mais rapide reste à RIFE, que DistilDRBA ne bat pas sur les
  // grands mouvements.
  const bool sakuga = strong >= shot_plan::kSakugaShare;
  choice.motion = !sakuga && cadence >= shot_plan::kLimitedCadence ? ShotMotion::Drba : ShotMotion::Rife;
  choice.temporal = rhythm.still >= shot_plan::kFlickerStill && rhythm.grain >= shot_plan::kFlickerGrain;
  return choice;
}

std::vector<ShotChoice> plan_shots(std::span<const media::Shot> shots, ShotChoice first) {
  std::vector<ShotChoice> out;
  out.reserve(shots.size());
  for (std::size_t i = 0; i < shots.size(); ++i) {
    const media::Shot& shot = shots[i];
    const bool follows = i > 0 && shots[i - 1].end_us == shot.start_us;
    ShotChoice previous = follows ? out.back() : first;
    // Sans coupe devant (reprise d'une analyse au milieu d'un plan) ou trop court pour se juger : le choix
    // d'avant, pour ne jamais basculer au milieu d'un plan réel ni sur un éclair.
    if ((follows && !shot.cut_before) || shot.rhythm.pairs < shot_plan::kMinPairs) {
      previous.judged = false;
      out.push_back(previous);
      continue;
    }
    out.push_back(judge_shot(shot.rhythm));
  }
  return out;
}

std::string rhythm_text(const media::ShotRhythm& rhythm) {
  const float cadence = media::drawing_cadence(rhythm);
  const auto pairs = static_cast<float>(std::max<std::uint32_t>(rhythm.pairs, 1));
  const int drawn = std::clamp(static_cast<int>(std::lround(cadence)), 1, 4);
  static constexpr const char* kCadence[] = {"en un", "en deux", "en trois", "plus lent"};
  std::string out = std::format(
      "{} ({:.1f} images par dessin), {:.0f} % de paires fortes, grain {:.1f} sur {:.0f} % immobile",
      kCadence[drawn - 1], cadence, 100.0F * static_cast<float>(rhythm.strong) / pairs, rhythm.grain,
      100.0F * rhythm.still);
  if (rhythm.flow_pairs > 0)
    out += std::format(", flux : caméra {:.2f} % et dessins {:.2f} % de la hauteur par image",
                       100.0F * rhythm.camera, 100.0F * rhythm.drawing);
  return out;
}

void ShotPlanBook::set_shots(std::shared_ptr<const std::vector<media::Shot>> shots) {
  std::vector<ShotChoice> choices = shots ? plan_shots(*shots) : std::vector<ShotChoice>{};
  const std::scoped_lock lock(mutex_);
  shots_ = std::move(shots);
  choices_ = std::move(choices);
}

namespace {

std::optional<std::size_t> index_of(const std::vector<media::Shot>& shots, std::int64_t pts_us) {
  const auto after = std::ranges::upper_bound(shots, pts_us, {}, &media::Shot::start_us);
  if (after == shots.begin()) return std::nullopt;
  const auto index = static_cast<std::size_t>(after - shots.begin()) - 1;
  if (pts_us >= shots[index].end_us) return std::nullopt;
  return index;
}

}  // namespace

std::optional<ShotChoice> ShotPlanBook::at(std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  if (!shots_) return std::nullopt;
  const std::optional<std::size_t> index = index_of(*shots_, pts_us);
  if (!index || *index >= choices_.size()) return std::nullopt;
  return choices_[*index];
}

std::string ShotPlanBook::describe(std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  if (!shots_) return {};
  const std::optional<std::size_t> index = index_of(*shots_, pts_us);
  return index ? rhythm_text((*shots_)[*index].rhythm) : std::string();
}

std::optional<bool> ShotPlanBook::cut_between(std::int64_t a_us, std::int64_t b_us) const {
  const std::scoped_lock lock(mutex_);
  if (!shots_) return std::nullopt;
  return known_cut(*shots_, a_us, b_us);
}

bool ShotPlanBook::active() const {
  const std::scoped_lock lock(mutex_);
  return shots_ && !shots_->empty();
}

}  // namespace kirari::app::play
