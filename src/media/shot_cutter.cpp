#include "media/shot_cutter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace ajp::media {
namespace {

// Seuils mesurés sur la vidéo de calibration et des épisodes d'anime : une coupe franche dépasse 0,35,
// un panoramique rapide reste sous 0,2 d'une image à l'autre. Le seuil monte avec le mouvement récent
// (2,5 fois la médiane des dernières distances), sans dépasser 0,6.
//
// La médiane ne compte que les distances entre dessins distincts (A42-01) : en animation tenue sur deux ou
// trois images, deux distances sur trois sont presque nulles, la médiane valait 0 et chaque dessin d'un
// panoramique ouvrait un plan (24 plans sur 96 images). Elle oublie aussi un mouvement vieux de plus d'une
// seconde et demie : un plan fixe ne garde pas le seuil du panoramique d'avant.
constexpr float kCutBase = 0.28F;
constexpr float kCutCap = 0.6F;
constexpr float kCutMotion = 2.5F;
constexpr std::size_t kRecent = 24;
constexpr std::int64_t kRecentUs = 1'500'000;
constexpr float kHeldDrawing = 0.02F;  // sous cette distance, le même dessin tenu
// Un éclair dure au plus trois images (A42-02) : tenu deux images en animation sur deux, il ouvrait un plan
// à l'entrée et un autre à la sortie.
constexpr std::size_t kFlashFrames = 3;
// Les bords d'un plan gardés pour juger un fondu : le sixième du plan, au plus deux secondes à 24 i/s.
constexpr std::size_t kEdgeKept = 48;
// Les palettes de visages gardées d'un plan : un long plan n'en garde pas des milliers.
constexpr std::size_t kFacesKept = 256;
// Un plan dont la luminance moyenne change de plus de 0,08 entre son début et sa fin (sur au moins une
// seconde à 24 images par seconde) est suivi image par image.
constexpr float kFadeDelta = 0.08F;
constexpr std::size_t kFadeFrames = 12;

// 70 % du blanc en lumière, en luma gamma 2,4 : le seuil des grands aplats clairs du grade.
const float kLitLuma = std::pow(0.7F, 1.0F / 2.4F);

float percentile(const std::array<float, kShotBins>& histogram, float share) {
  float seen = 0.0F;
  for (int i = 0; i < kShotBins; ++i) {
    const float part = histogram.at(static_cast<std::size_t>(i));
    if (part > 0.0F && seen + part >= share) {
      const float within = (share - seen) / part;
      return (static_cast<float>(i) + std::clamp(within, 0.0F, 1.0F)) / static_cast<float>(kShotBins);
    }
    seen += part;
  }
  return 1.0F;
}

// La part de l'histogramme sous (below) ou au-dessus de (!below) la luminance at, cases coupées comprises.
float share_beyond(const std::array<float, kShotBins>& histogram, float at, bool below) {
  float sum = 0.0F;
  for (int i = 0; i < kShotBins; ++i) {
    const float lo = static_cast<float>(i) / static_cast<float>(kShotBins);
    const float hi = static_cast<float>(i + 1) / static_cast<float>(kShotBins);
    const float under = std::clamp((at - lo) / (hi - lo), 0.0F, 1.0F);
    sum += histogram.at(static_cast<std::size_t>(i)) * (below ? under : 1.0F - under);
  }
  return sum;
}

float median(std::vector<float> values) {
  if (values.empty()) return 0.0F;
  const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  return *mid;
}

}  // namespace

ShotStats stats_of(const std::array<float, kShotBins>& histogram, float mean, float blue, float chroma) {
  return ShotStats{.mean = mean,
                   .p05 = percentile(histogram, 0.05F),
                   .p50 = percentile(histogram, 0.50F),
                   .p95 = percentile(histogram, 0.95F),
                   .p995 = percentile(histogram, 0.995F),
                   .dark = share_beyond(histogram, 0.1F, true),
                   .bright = share_beyond(histogram, 0.8F, false),
                   .blue = blue,
                   .chroma = chroma,
                   .lit = share_beyond(histogram, kLitLuma, false)};
}

ShotFrame shot_frame_of(std::int64_t pts_us, const kernels::FrameStats& stats) {
  ShotFrame out;
  out.pts_us = pts_us;
  out.histogram = stats.luma_hist;
  out.grid = stats.zones;
  out.mean = stats.mean;
  out.blue = stats.blue;
  out.chroma = stats.chroma;
  out.light = ShotLight{.maxrgb_nits = stats.maxrgb_nits,
                        .apl_nits = stats.apl_nits,
                        .maxrgb_mean_nits = stats.maxrgb_mean_nits,
                        .bright = stats.bright,
                        .highlights = stats.highlights,
                        .skin = stats.skin};
  // I d'ICtCp en kLinearBins cases ramenées à kShotLumBins.
  static_assert(kernels::frame_stats::kLinearBins % kShotLumBins == 0);
  constexpr std::size_t kGroup = kernels::frame_stats::kLinearBins / kShotLumBins;
  for (std::size_t i = 0; i < stats.i_hist.size(); ++i)
    out.light.lum_hist.at(i / kGroup) += stats.i_hist.at(i);
  return out;
}

auto zone_nits_of(const ShotFrame& frame) -> decltype(frame.grid) {
  decltype(frame.grid) light{};
  float sum = 0.0F;
  for (std::size_t i = 0; i < light.size(); ++i) {
    light.at(i) = std::pow(std::max(frame.grid.at(i), 0.0F), 2.4F);
    sum += light.at(i);
  }
  const float mean = sum / static_cast<float>(light.size());
  for (float& zone : light) zone = mean > 0.0F ? frame.light.apl_nits * zone / mean : frame.light.apl_nits;
  return light;
}

namespace {

// Ce qui change entre deux images : la lumière (histogramme), sa place (grille), la couleur. De 0 à 1.
struct Change {
  float light = 0.0F;
  float place = 0.0F;
  float colour = 0.0F;
  float distance() const { return std::max({light, place, colour}); }
};

Change change_between(const ShotFrame& a, const ShotFrame& b) {
  float histogram = 0.0F;
  for (std::size_t i = 0; i < a.histogram.size(); ++i)
    histogram += std::abs(a.histogram.at(i) - b.histogram.at(i));
  float grid = 0.0F;
  for (std::size_t i = 0; i < a.grid.size(); ++i) grid += std::abs(a.grid.at(i) - b.grid.at(i));
  grid /= static_cast<float>(a.grid.size());
  // L'histogramme dit que la lumière a changé ; la grille, qu'elle a changé de place (deux plans de même
  // luminosité, cadrés autrement) ; la couleur (P3.31), un champ-contrechamp de même luminosité.
  const float colour = std::abs(a.blue - b.blue) + std::abs(a.chroma - b.chroma);
  return Change{.light = std::min(0.5F * histogram, 1.0F),
                .place = std::min(2.5F * grid, 1.0F),
                .colour = std::min(3.0F * colour, 1.0F)};
}

}  // namespace

float frame_distance(const ShotFrame& a, const ShotFrame& b) {
  return change_between(a, b).distance();
}

void ShotCutter::Totals::add(const ShotFrame& f) {
  const ShotLight& l = f.light;
  apl_min = count == 0 ? l.apl_nits : std::min(apl_min, l.apl_nits);
  apl_max = std::max(apl_max, l.apl_nits);
  max_scl = std::max(max_scl, l.maxrgb_nits.back());
  ++count;
  for (std::size_t i = 0; i < histogram.size(); ++i) histogram.at(i) += f.histogram.at(i);
  mean += f.mean;
  blue += f.blue;
  chroma += f.chroma;
  for (std::size_t i = 0; i < maxrgb.size(); ++i) maxrgb.at(i) += l.maxrgb_nits.at(i);
  apl += l.apl_nits;
  maxrgb_mean += l.maxrgb_mean_nits;
  bright += l.bright;
  highlights += l.highlights;
  skin += l.skin;
  for (std::size_t i = 0; i < lum_hist.size(); ++i) lum_hist.at(i) += l.lum_hist.at(i);
  for (const SkinTone& tone : f.faces)
    if (faces.size() < kFacesKept) faces.push_back(tone);
}

ShotStats ShotCutter::Totals::stats() const {
  const auto n = static_cast<double>(std::max<std::size_t>(count, 1));
  std::array<float, kShotBins> parts{};
  for (std::size_t i = 0; i < parts.size(); ++i) parts.at(i) = static_cast<float>(histogram.at(i) / n);
  ShotStats out = stats_of(parts, static_cast<float>(mean / n), static_cast<float>(blue / n),
                           static_cast<float>(chroma / n));
  out.apl_nits = static_cast<float>(apl / n);
  return out;
}

ShotMeta ShotCutter::Totals::meta() const {
  ShotMeta out;
  if (count == 0) return out;
  const auto n = static_cast<double>(count);
  out.apl_min_nits = apl_min;
  out.apl_max_nits = apl_max;
  out.max_scl_nits = max_scl;
  ShotLight& m = out.mean;
  for (std::size_t i = 0; i < maxrgb.size(); ++i) m.maxrgb_nits.at(i) = static_cast<float>(maxrgb.at(i) / n);
  m.apl_nits = static_cast<float>(apl / n);
  m.maxrgb_mean_nits = static_cast<float>(maxrgb_mean / n);
  m.bright = static_cast<float>(bright / n);
  m.highlights = static_cast<float>(highlights / n);
  m.skin = static_cast<float>(skin / n);
  for (std::size_t i = 0; i < lum_hist.size(); ++i) m.lum_hist.at(i) = static_cast<float>(lum_hist.at(i) / n);
  out.apl_mean_nits = m.apl_nits;
  out.skins = plan_skins(faces);
  return out;
}

void ShotCutter::keep(const ShotFrame& frame) {
  if (totals_.count == 0) start_us_ = frame.pts_us;
  totals_.add(frame);
  if (head_.size() < kEdgeKept) head_.push_back(frame);
  tail_.push_back(frame);
  if (tail_.size() > kEdgeKept) tail_.pop_front();
}

float ShotCutter::limit(std::int64_t pts_us) {
  while (!recent_.empty() && (recent_.size() > kRecent || recent_.front().pts_us < pts_us - kRecentUs))
    recent_.pop_front();
  std::vector<float> distances;
  distances.reserve(recent_.size());
  for (const Motion& m : recent_) distances.push_back(m.distance);
  return std::min(std::max(kCutBase, kCutMotion * median(std::move(distances))), kCutCap);
}

std::optional<Shot> ShotCutter::push(const ShotFrame& frame) {
  if (!anchor_) {
    keep(frame);
    anchor_ = frame;
    return std::nullopt;
  }
  const ShotFrame anchor = *anchor_;  // copie : l'ancre change en route
  if (jumped_.empty()) return judge(anchor, frame);
  // Un saut attend : l'image qui revient près de l'ancre en fait un éclair, gardé dans le plan sans servir
  // de référence ; après kFlashFrames images sans retour, c'était une coupe, à la première image du saut.
  if (frame_distance(anchor, frame) < std::min(0.5F * jump_, jump_limit_)) {
    for (const ShotFrame& f : jumped_) keep(f);
    jumped_.clear();
    return judge(anchor, frame);
  }
  // Le départ d'un mouvement rapide (A42-01) : la lumière et la couleur n'ont pas changé, seule leur place,
  // et le dessin suivant s'éloigne encore d'autant. Le saut suit alors la règle du mouvement récent, jugée
  // sur le mouvement qui vient : sans cela, le premier dessin d'un panoramique en trois ouvrait un plan.
  const float step = frame_distance(jumped_.back(), frame);
  jumped_.push_back(frame);
  if (jump_moves_ && step >= kHeldDrawing && jump_ <= kCutMotion * step) {
    recent_.push_back({.pts_us = jumped_.front().pts_us, .distance = jump_});
    recent_.push_back({.pts_us = frame.pts_us, .distance = step});
    for (const ShotFrame& f : jumped_) keep(f);
    jumped_.clear();
    anchor_ = frame;
    return std::nullopt;
  }
  if (step >= kHeldDrawing) jump_moves_ = false;  // un autre dessin, trop proche : pas ce mouvement-là
  if (jumped_.size() <= kFlashFrames) return std::nullopt;
  const std::vector<ShotFrame> after = std::move(jumped_);
  jumped_.clear();
  std::optional<Shot> closed = close_shot(after.front().pts_us);
  recent_.push_back({.pts_us = after.front().pts_us, .distance = jump_});
  keep(after.front());
  anchor_ = after.front();
  // Les images d'après la coupe repassent une à une : aucune n'a pu ouvrir un autre plan en si peu d'images
  // sans que la première coupe soit elle-même un éclair.
  for (std::size_t i = 1; i < after.size(); ++i) {
    if (auto again = push(after.at(i)); again && !closed) closed = again;
  }
  return closed;
}

// L'image face à l'ancre : dans le plan, ou mise en attente si elle saute plus loin que le seuil.
std::optional<Shot> ShotCutter::judge(const ShotFrame& anchor, const ShotFrame& frame) {
  const Change change = change_between(anchor, frame);
  const float jump = change.distance();
  const float threshold = limit(frame.pts_us);
  if (jump > threshold) {
    jumped_.push_back(frame);
    jump_ = jump;
    jump_limit_ = threshold;
    jump_moves_ = jump <= kCutCap && std::max(change.light, change.colour) < 0.5F * kCutBase;
    return std::nullopt;
  }
  keep(frame);
  anchor_ = frame;
  if (jump >= kHeldDrawing) recent_.push_back({.pts_us = frame.pts_us, .distance = jump});
  return std::nullopt;
}

std::optional<Shot> ShotCutter::finish(std::int64_t end_us) {
  // Un saut encore en attente à la fin (trois images au plus) reste dans le dernier plan.
  for (const ShotFrame& f : jumped_) keep(f);
  jumped_.clear();
  std::optional<Shot> closed = close_shot(end_us);
  recent_.clear();
  anchor_.reset();
  return closed;
}

// B42-03 : la part du plan où la lumière change vraiment. Un plan long fini par un fondu au noir restait
// interpolé sur toute sa durée : il dérivait vers le noir dès son début. Le fondu commence après la dernière
// image de la fin du plan (tail_) encore au niveau du début, et finit à la première image du début du plan
// (head_) déjà au niveau de la fin. Non vu dans ces images gardées (un fondu plus long qu'elles) : tout le
// plan, comme avant.
void ShotCutter::ramp_of(Shot& shot, float begin, float end) const {
  const float near = 0.5F * kFadeDelta;
  std::int64_t from = shot.start_us;
  std::int64_t to = shot.end_us;
  for (std::size_t i = tail_.size(); i-- > 0;) {
    if (std::abs(tail_.at(i).mean - begin) <= near) {
      if (i + 1 < tail_.size()) from = std::max(from, tail_.at(i).pts_us);
      break;
    }
  }
  for (std::size_t i = 0; i < head_.size(); ++i) {
    if (std::abs(head_.at(i).mean - end) <= near) {
      if (i > 0) to = std::min(to, head_.at(i).pts_us);
      break;
    }
  }
  if (from >= to) return;  // un aller-retour dans le plan : tout le plan
  if (from == shot.start_us && to == shot.end_us) return;
  shot.ramp_from_us = from;
  shot.ramp_to_us = to;
}

std::optional<Shot> ShotCutter::close_shot(std::int64_t end_us) {
  if (totals_.count == 0) return std::nullopt;
  Shot shot;
  shot.start_us = start_us_;
  shot.end_us = std::max(end_us, shot.start_us + 1);
  shot.cut_before = first_shot_ ? starts_at_cut_ : true;
  first_shot_ = false;
  shot.meta = totals_.meta();
  shot.first = totals_.stats();
  shot.last = shot.first;
  if (totals_.count >= 2 * kFadeFrames) {
    const std::size_t edge = std::clamp(totals_.count / 6, std::size_t{4}, kEdgeKept);
    Totals first, last;
    for (std::size_t i = 0; i < edge; ++i) {
      first.add(head_.at(i));
      last.add(tail_.at(tail_.size() - edge + i));
    }
    const ShotStats begin = first.stats();
    const ShotStats end = last.stats();
    if (std::abs(begin.mean - end.mean) > kFadeDelta) {
      shot.first = begin;
      shot.last = end;
      shot.fade = true;
      ramp_of(shot, begin.mean, end.mean);
    }
  }
  totals_ = {};
  head_.clear();
  tail_.clear();
  return shot;
}

}  // namespace ajp::media
