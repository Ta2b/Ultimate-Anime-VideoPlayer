#include "media/shot_cutter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace ajp::media {
namespace {

// Seuils mesurés sur la vidéo de calibration et des épisodes d'anime : une coupe franche dépasse 0,35,
// un panoramique rapide reste sous 0,2. Le seuil monte avec le mouvement récent (2,5 fois la médiane
// des 24 dernières distances), sans dépasser 0,6.
constexpr float kCutBase = 0.28F;
constexpr float kCutCap = 0.6F;
constexpr float kCutMotion = 2.5F;
constexpr std::size_t kRecent = 24;
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

ShotStats average(std::span<const ShotFrame> frames) {
  std::array<float, kShotBins> histogram{};
  float mean = 0.0F, blue = 0.0F, chroma = 0.0F, apl = 0.0F;
  for (const ShotFrame& f : frames) {
    for (std::size_t i = 0; i < histogram.size(); ++i) histogram.at(i) += f.histogram.at(i);
    apl += f.light.apl_nits;
    mean += f.mean;
    blue += f.blue;
    chroma += f.chroma;
  }
  const auto n = static_cast<float>(std::max<std::size_t>(frames.size(), 1));
  for (float& part : histogram) part /= n;
  ShotStats stats = stats_of(histogram, mean / n, blue / n, chroma / n);
  stats.apl_nits = apl / n;
  return stats;
}

ShotMeta meta_of(std::span<const ShotFrame> frames) {
  ShotMeta meta;
  if (frames.empty()) return meta;
  meta.apl_min_nits = frames.front().light.apl_nits;
  ShotLight& m = meta.mean;
  for (const ShotFrame& f : frames) {
    const ShotLight& l = f.light;
    meta.apl_min_nits = std::min(meta.apl_min_nits, l.apl_nits);
    meta.apl_max_nits = std::max(meta.apl_max_nits, l.apl_nits);
    meta.max_scl_nits = std::max(meta.max_scl_nits, l.maxrgb_nits.back());
    for (std::size_t i = 0; i < m.maxrgb_nits.size(); ++i) m.maxrgb_nits.at(i) += l.maxrgb_nits.at(i);
    m.apl_nits += l.apl_nits;
    m.maxrgb_mean_nits += l.maxrgb_mean_nits;
    m.bright += l.bright;
    m.highlights += l.highlights;
    m.skin += l.skin;
    for (std::size_t i = 0; i < m.lum_hist.size(); ++i) m.lum_hist.at(i) += l.lum_hist.at(i);
  }
  const auto n = static_cast<float>(frames.size());
  for (float& v : m.maxrgb_nits) v /= n;
  m.apl_nits /= n;
  m.maxrgb_mean_nits /= n;
  m.bright /= n;
  m.highlights /= n;
  m.skin /= n;
  for (float& part : m.lum_hist) part /= n;
  meta.apl_mean_nits = m.apl_nits;
  return meta;
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

float frame_distance(const ShotFrame& a, const ShotFrame& b) {
  float histogram = 0.0F;
  for (std::size_t i = 0; i < a.histogram.size(); ++i)
    histogram += std::abs(a.histogram.at(i) - b.histogram.at(i));
  float grid = 0.0F;
  for (std::size_t i = 0; i < a.grid.size(); ++i) grid += std::abs(a.grid.at(i) - b.grid.at(i));
  grid /= static_cast<float>(a.grid.size());
  // L'histogramme dit que la lumière a changé ; la grille, qu'elle a changé de place (deux plans de même
  // luminosité, cadrés autrement).
  return std::max(std::min(0.5F * histogram, 1.0F), std::min(2.5F * grid, 1.0F));
}

std::optional<Shot> ShotCutter::push(const ShotFrame& frame) {
  std::optional<Shot> closed;
  if (pending_ && anchor_) {
    // L'image d'avant l'image en attente, éclairs écartés.
    const ShotFrame before = *anchor_;
    const ShotFrame held = *pending_;  // copie : close_shot() ne laisse pas suivre l'optional
    const float jump = frame_distance(before, held);
    const float limit = std::min(std::max(kCutBase, kCutMotion * median(recent_)), kCutCap);
    bool cut = jump > limit;
    // Un éclair : l'image d'après ressemble à celle d'avant, l'image en attente reste dans le plan sans
    // servir de référence à la suivante.
    const bool flash = cut && frame_distance(before, frame) < std::min(0.5F * jump, limit);
    if (flash) cut = false;
    if (cut) {
      closed = close_shot(held.pts_us);
      frames_.clear();
    }
    frames_.push_back(held);
    if (!flash) {
      anchor_ = held;
      recent_.push_back(jump);
      if (recent_.size() > kRecent) recent_.erase(recent_.begin());
    }
  } else if (pending_) {
    frames_.push_back(*pending_);
    anchor_ = *pending_;
  }
  pending_ = frame;
  return closed;
}

std::optional<Shot> ShotCutter::finish(std::int64_t end_us) {
  if (pending_) {
    frames_.push_back(*pending_);
    pending_.reset();
  }
  std::optional<Shot> closed = close_shot(end_us);
  frames_.clear();
  recent_.clear();
  anchor_.reset();
  return closed;
}

std::optional<Shot> ShotCutter::close_shot(std::int64_t end_us) {
  if (frames_.empty()) return std::nullopt;
  Shot shot;
  shot.start_us = frames_.front().pts_us;
  shot.end_us = std::max(end_us, shot.start_us + 1);
  shot.cut_before = first_shot_ ? starts_at_cut_ : true;
  first_shot_ = false;
  const std::span<const ShotFrame> all(frames_);
  shot.meta = meta_of(all);
  if (frames_.size() >= 2 * kFadeFrames) {
    const std::size_t edge = std::max(frames_.size() / 6, std::size_t{4});
    const ShotStats first = average(all.first(edge));
    const ShotStats last = average(all.last(edge));
    if (std::abs(first.mean - last.mean) > kFadeDelta) {
      shot.first = first;
      shot.last = last;
      shot.fade = true;
      return shot;
    }
  }
  shot.first = average(all);
  shot.last = shot.first;
  return shot;
}

}  // namespace ajp::media
