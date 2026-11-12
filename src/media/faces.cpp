#include "media/faces.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ajp::media {
namespace {

float area(const FaceBox& b) {
  return std::max(b.x1 - b.x0, 0.0F) * std::max(b.y1 - b.y0, 0.0F);
}

float median_of(std::vector<float>& v) {
  const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}

float smoothstep(float e0, float e1, float x) {
  const float t = std::clamp((x - e0) / (e1 - e0), 0.0F, 1.0F);
  return t * t * (3.0F - 2.0F * t);
}

// Deux palettes du même personnage : la teinte à 6° près, la saturation à 30 % près.
bool same_skin(const SkinTone& a, const SkinTone& b) {
  constexpr float kHue = 6.0F;
  constexpr float kSat = 1.3F;
  const float lo = std::min(a.sat, b.sat);
  const float hi = std::max(a.sat, b.sat);
  return std::abs(a.hue - b.hue) <= kHue && hi <= kSat * lo;
}

}  // namespace

float face_iou(const FaceBox& a, const FaceBox& b) {
  const FaceBox inter{.x0 = std::max(a.x0, b.x0),
                      .y0 = std::max(a.y0, b.y0),
                      .x1 = std::min(a.x1, b.x1),
                      .y1 = std::min(a.y1, b.y1)};
  const float i = area(inter);
  const float u = area(a) + area(b) - i;
  return u > 0.0F ? i / u : 0.0F;
}

std::vector<FaceBox> suppress_faces(std::vector<FaceBox> boxes, float iou) {
  std::ranges::stable_sort(boxes, [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });
  std::vector<FaceBox> kept;
  for (const FaceBox& b : boxes)
    if (std::ranges::none_of(kept, [&](const FaceBox& k) { return face_iou(k, b) > iou; })) kept.push_back(b);
  return kept;
}

std::optional<SkinTone> skin_tone_of(float r, float g, float b) {
  if (!(r > 1e-4F) || r < g || g < b) return std::nullopt;
  return SkinTone{.hue = 60.0F * (g - b) / std::max(r - b, 1e-6F), .sat = (r - b) / r};
}

float skin_wide(const SkinTone& t) {
  return smoothstep(10.0F, 16.0F, t.hue) * (1.0F - smoothstep(44.0F, 52.0F, t.hue)) *
         smoothstep(0.08F, 0.12F, t.sat) * (1.0F - smoothstep(0.60F, 0.70F, t.sat));
}

std::optional<SkinTone> face_palette(std::span<const SkinTone> samples) {
  if (samples.empty()) return std::nullopt;
  std::vector<float> hue;
  std::vector<float> sat;
  hue.reserve(samples.size());
  sat.reserve(samples.size());
  for (const SkinTone& s : samples) {
    hue.push_back(s.hue);
    sat.push_back(s.sat);
  }
  return SkinTone{.hue = median_of(hue), .sat = median_of(sat)};
}

PlanSkins plan_skins(std::span<const SkinTone> faces) {
  // Regroupement glouton autour de la palette la plus entourée : chaque palette est jugée comme centre, celle
  // qui a le plus de voisines forme un groupe, qui sort de la liste ; on recommence.
  std::vector<SkinTone> left;
  for (const SkinTone& t : faces)
    if (skin_wide(t) > 0.0F) left.push_back(t);
  PlanSkins out;
  while (!left.empty() && out.count < kMaxPlanSkins) {
    std::size_t best = 0;
    std::size_t best_n = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
      const auto n = static_cast<std::size_t>(
          std::ranges::count_if(left, [&](const SkinTone& t) { return same_skin(left[i], t); }));
      if (n > best_n) {
        best = i;
        best_n = n;
      }
    }
    if (best_n < 2) break;
    const SkinTone centre = left[best];
    std::vector<SkinTone> group;
    std::erase_if(left, [&](const SkinTone& t) {
      if (!same_skin(centre, t)) return false;
      group.push_back(t);
      return true;
    });
    if (const auto tone = face_palette(group)) out.tones.at(out.count++) = *tone;
  }
  return out;
}

}  // namespace ajp::media
