#include "render/faces.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>

namespace ajp::render {
namespace {

float area(const FaceBox& b) {
  return std::max(b.x1 - b.x0, 0.0F) * std::max(b.y1 - b.y0, 0.0F);
}

float median_of(std::vector<float>& v) {
  const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}

float ramp(float d, float margin) {
  if (d <= 0.0F) return 1.0F;
  if (d >= margin || margin <= 0.0F) return 0.0F;
  return 1.0F - d / margin;
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

std::vector<FaceBox> decode_faces(std::span<const float> output, int rows, int stride, int input_w,
                                  int input_h, float min_score) {
  std::vector<FaceBox> boxes;
  if (rows <= 0 || stride < 5 || input_w <= 0 || input_h <= 0) return boxes;
  const auto need = static_cast<std::size_t>(rows) * static_cast<std::size_t>(stride);
  if (output.size() < need) return boxes;
  const auto w = static_cast<float>(input_w);
  const auto h = static_cast<float>(input_h);
  for (int r = 0; r < rows; ++r) {
    const float* row = &output[static_cast<std::size_t>(r) * static_cast<std::size_t>(stride)];
    const float score = row[4];
    if (!(score >= min_score) || !(row[2] > 0.0F) || !(row[3] > 0.0F)) continue;
    FaceBox b{.x0 = std::clamp((row[0] - 0.5F * row[2]) / w, 0.0F, 1.0F),
              .y0 = std::clamp((row[1] - 0.5F * row[3]) / h, 0.0F, 1.0F),
              .x1 = std::clamp((row[0] + 0.5F * row[2]) / w, 0.0F, 1.0F),
              .y1 = std::clamp((row[1] + 0.5F * row[3]) / h, 0.0F, 1.0F),
              .score = score};
    if (area(b) > 0.0F) boxes.push_back(b);
  }
  return boxes;
}

std::vector<FaceBox> suppress_faces(std::vector<FaceBox> boxes, float iou) {
  std::ranges::stable_sort(boxes, [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });
  std::vector<FaceBox> kept;
  for (const FaceBox& b : boxes)
    if (std::ranges::none_of(kept, [&](const FaceBox& k) { return face_iou(k, b) > iou; })) kept.push_back(b);
  return kept;
}

std::optional<SkinPalette> face_palette(std::span<const SkinPalette> samples) {
  if (samples.empty()) return std::nullopt;
  std::vector<float> i;
  std::vector<float> ct;
  std::vector<float> cp;
  i.reserve(samples.size());
  ct.reserve(samples.size());
  cp.reserve(samples.size());
  for (const SkinPalette& s : samples) {
    i.push_back(s.i);
    ct.push_back(s.ct);
    cp.push_back(s.cp);
  }
  return SkinPalette{.i = median_of(i), .ct = median_of(ct), .cp = median_of(cp)};
}

void FaceTracker::update(std::span<const FaceBox> detections) {
  std::vector<bool> taken(detections.size(), false);
  for (FaceTrack& t : tracks_) {
    float best = kMatchIou;
    std::ptrdiff_t at = -1;
    for (std::size_t d = 0; d < detections.size(); ++d) {
      if (taken[d]) continue;
      const float v = face_iou(t.box, detections[d]);
      if (v >= best) {
        best = v;
        at = static_cast<std::ptrdiff_t>(d);
      }
    }
    if (at < 0) {
      ++t.misses;  // la piste reste confirmée le temps de la retrouver
      continue;
    }
    const FaceBox& d = detections[static_cast<std::size_t>(at)];
    taken[static_cast<std::size_t>(at)] = true;
    t.box = FaceBox{.x0 = std::lerp(t.box.x0, d.x0, kSmooth),
                    .y0 = std::lerp(t.box.y0, d.y0, kSmooth),
                    .x1 = std::lerp(t.box.x1, d.x1, kSmooth),
                    .y1 = std::lerp(t.box.y1, d.y1, kSmooth),
                    .score = d.score};
    ++t.hits;
    t.misses = 0;
  }
  std::erase_if(tracks_, [](const FaceTrack& t) { return t.misses >= kMaxMisses; });
  for (std::size_t d = 0; d < detections.size(); ++d)
    if (!taken[d]) tracks_.push_back(FaceTrack{.id = next_id_++, .box = detections[d], .hits = 1});
}

void FaceTracker::set_palette(std::uint32_t id, const SkinPalette& palette) {
  for (FaceTrack& t : tracks_)
    if (t.id == id) t.palette = palette;
}

void FaceTracker::cut() {
  tracks_.clear();
}

float face_exclusion(std::span<const FaceTrack> tracks, float x, float y) {
  float part = 0.0F;
  for (const FaceTrack& t : tracks) {
    if (!t.confirmed()) continue;
    const FaceBox& b = t.box;
    const float margin = 0.1F * std::max(b.x1 - b.x0, b.y1 - b.y0);
    const float dx = std::max({b.x0 - x, x - b.x1, 0.0F});
    const float dy = std::max({b.y0 - y, y - b.y1, 0.0F});
    part = std::max(part, ramp(std::hypot(dx, dy), margin));
  }
  return part;
}

}  // namespace ajp::render
