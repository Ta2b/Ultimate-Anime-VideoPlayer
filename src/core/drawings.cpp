// Réécrit d'après Ta2b/mpv-animejanai-hdr@55a8ae53 :
//   - native/linux-cuda/vf_animejanai-drawings.patch (process_drawings) : le mouvement
//     du dessin à l'écran au suivant, réparti sur sa tenue, cherché 4 sources plus loin ;
//   - native/linux-ort/aji_ort.cpp:1038-1072 : tenue de 2 à 4 sources, instant
//     (run + t) / hold, jamais d'interpolation à travers une coupe ;
// et d'après animejanai-inference-rtx@118c272 src/aji_trt.cpp:3510-3531 : seuil du
// dessin tenu inclus, en niveaux 8 bits sur le pire bloc 32x32.
// Comportement fixé par tests/unit/core/drawings_test.cpp.
#include "core/drawings.h"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

#include "core/cadence.h"

namespace ajp::core {
namespace {

// La paire (k, k+1) cherche le dessin suivant jusqu'à k + kMaxSpreadHold.
static_assert(drawings::kMaxSpreadHold - 1 <= drawings::kSmoothPreload);
static_assert(drawings::kSmoothPreload + 2 <= drawings::kWindow);

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::expected<void, Refusal> check(const PairMeasure& m) {
  if (!(m.worst_block_diff >= 0.0 && m.worst_block_diff <= drawings::kMaxDiff)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("différence {} hors de [0, 255]", m.worst_block_diff));
  }
  if (m.identical && (m.worst_block_diff != 0.0 || m.scene_cut)) {
    return refuse(RefusalCode::FieldOutOfRange, "doublon exact avec une différence ou une coupe");
  }
  return {};
}

bool is_one(Ratio t) {
  return t.num == t.den;
}

// (whole + t) / parts, réduit : la position d'un instant dans une tenue.
Ratio within_hold(std::uint64_t whole, Ratio t, std::uint64_t parts) {
  const auto w = static_cast<std::int64_t>(whole);
  const auto p = static_cast<std::int64_t>(parts);
  return reduced({w * t.den + t.num, p * t.den});
}

InstantDecision source(SourceId a) {
  return {.show = Show::Source, .a = a};
}
InstantDecision repeated(SourceId a) {
  return {.show = Show::Repeated, .a = a};
}
InstantDecision interpolate(SourceId a, SourceId b, Ratio u) {
  return {.show = Show::Interpolate, .a = a, .b = b, .u = reduced(u)};
}

}  // namespace

void DrawingsTimeline::start(SourceId first) {
  *this = DrawingsTimeline{};
  started_ = true;
  store({.id = first, .drawing = {.kind = DrawingClass::New, .first = first}});
}

std::expected<SourceDrawing, Refusal> DrawingsTimeline::push(SourceId id, const PairMeasure& measure) {
  if (!started_ || finished_) {
    return refuse(RefusalCode::SequenceOrder,
                  finished_ ? "source après la fin du flux" : "source avant start()");
  }
  if (auto ok = check(measure); !ok) return std::unexpected(ok.error());

  const Entry& previous = at(count_ - 1);
  Entry entry{.id = id};
  if (!measure.scene_cut && measure.worst_block_diff <= drawings::kHeldThreshold) {
    entry.drawing = {
        .kind = DrawingClass::Held, .identical = measure.identical, .first = previous.drawing.first};
    entry.first_at = previous.first_at;
  } else {
    entry.drawing = {.kind = DrawingClass::New, .scene_cut = measure.scene_cut, .first = id};
    entry.first_at = count_;
    if (measure.scene_cut) {
      hold_count_ = 0;  // nouveau plan : le rythme repart
    } else {
      record_hold(count_ - previous.first_at);
    }
  }
  store(entry);
  return entry.drawing;
}

Rhythm DrawingsTimeline::rhythm() const {
  if (hold_count_ == 0) return Rhythm::Unknown;
  std::array<std::size_t, 5> votes{};  // index : tenue de 1 à 4, 4 valant « 4 ou plus »
  const std::size_t kept = std::min(hold_count_, drawings::kRhythmWindow);
  for (std::size_t i = 0; i < kept; ++i) ++votes[holds_[i]];
  // Le mode ; à égalité, la tenue la plus courte, qui demande le plus de calcul.
  std::size_t best = 1;
  for (std::size_t hold = 2; hold <= 4; ++hold) {
    if (votes[hold] > votes[best]) best = hold;
  }
  constexpr std::array kRhythms{Rhythm::Unknown, Rhythm::Ones, Rhythm::Twos, Rhythm::Threes, Rhythm::Longer};
  return kRhythms[best];
}

bool DrawingsTimeline::ready(SourceId left, Pacing pacing) const {
  const std::optional<std::uint64_t> position = find(left);
  if (!position || *position + 1 >= count_) return false;
  return finished_ || *position + 1 + static_cast<std::uint64_t>(preload(pacing)) < count_;
}

std::expected<InstantDecision, Refusal> DrawingsTimeline::decide(Policy policy, Pacing pacing, SourceId left,
                                                                 Ratio t) const {
  if (t.den < 1 || t.num < 1 || t.num > t.den) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("instant {}/{} hors de ]0, 1]", t.num, t.den));
  }
  if (policy == Policy::Quality && pacing != Pacing::Source) {
    return refuse(RefusalCode::PacingForbidden, "Quality impose le rythme source");
  }
  const std::optional<std::uint64_t> position = find(left);
  if (!position || (finished_ && *position + 1 >= count_)) {
    return refuse(RefusalCode::SourceUnknown, std::format("paire de la source {}", left.value));
  }
  if (!ready(left, pacing)) {
    return refuse(RefusalCode::LookaheadMissing, std::format("paire de la source {}", left.value));
  }
  if (pacing == Pacing::Smooth) return on_drawings(*position, t);
  return on_grid(policy, at(*position), at(*position + 1), t);
}

InstantDecision DrawingsTimeline::on_grid(Policy policy, const Entry& left, const Entry& right,
                                          Ratio t) const {
  if (is_one(t)) return source(right.id);
  if (right.drawing.scene_cut || right.drawing.identical) return repeated(left.id);
  // Fast et Refine déforment : un dessin tenu se reprend tel quel.
  if (policy != Policy::Quality && right.drawing.kind == DrawingClass::Held) return repeated(left.id);
  return interpolate(left.id, right.id, t);
}

InstantDecision DrawingsTimeline::on_drawings(std::uint64_t left, Ratio t) const {
  const Entry& k = at(left);
  const Entry& right = at(left + 1);
  const SourceId drawing = k.drawing.first;
  const std::uint64_t into = left - k.first_at;  // sources du dessin avant celle de gauche

  if (right.drawing.kind == DrawingClass::New) {
    if (is_one(t)) return source(right.id);
    if (right.drawing.scene_cut) return repeated(drawing);
    const std::uint64_t hold = left + 1 - k.first_at;
    // Au-delà de kMaxSpreadHold, un plan fixe s'achève : le mouvement tient dans la dernière paire.
    if (hold > drawings::kMaxSpreadHold) return interpolate(drawing, right.id, t);
    return interpolate(drawing, right.id, within_hold(into, t, hold));
  }

  // La source droite tient le dessin : où commence le suivant ?
  for (std::uint64_t n = left + 2; n <= k.first_at + drawings::kMaxSpreadHold && n < count_; ++n) {
    const Entry& next = at(n);
    if (next.drawing.kind == DrawingClass::Held) continue;
    if (next.drawing.scene_cut) break;
    return interpolate(drawing, next.id, within_hold(into, t, n - k.first_at));
  }
  return repeated(drawing);  // plan fixe, coupe à venir, ou fin du flux
}

const DrawingsTimeline::Entry& DrawingsTimeline::at(std::uint64_t position) const {
  return ring_[position % drawings::kWindow];
}

std::optional<std::uint64_t> DrawingsTimeline::find(SourceId id) const {
  for (std::uint64_t back = 1; back <= count_ && back <= drawings::kWindow; ++back) {
    if (ring_[(count_ - back) % drawings::kWindow].id == id) return count_ - back;
  }
  return std::nullopt;
}

void DrawingsTimeline::store(const Entry& entry) {
  ring_[count_ % drawings::kWindow] = entry;
  ++count_;
}

void DrawingsTimeline::record_hold(std::uint64_t hold) {
  holds_[hold_count_ % drawings::kRhythmWindow] = std::min<std::uint64_t>(hold, 4);
  ++hold_count_;
}

}  // namespace ajp::core
