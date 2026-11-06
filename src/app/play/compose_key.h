// Ce dont dépend l'image composée (vidéo gradée et sous-titres) : tant que la clé ne change pas, l'écran
// recopie l'image déjà composée au lieu de la recomposer, puis pose l'OSD et l'interface par-dessus. L'OSD
// n'y entre pas : il change à chaque présentation et se dessine à part.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include "app/play/final_scaler.h"
#include "core/ring.h"
#include "subs/subtitle_frame.h"

namespace ajp::app::play {

struct ComposeKey {
  bool has_frame = false;  // faux : image noire
  std::uint32_t slot = 0;
  std::uint32_t generation = 0;
  std::uint32_t epoch = 0;
  std::uint64_t ready_value = 0;
  std::uint64_t subs_generation = 0;  // celle de l'atlas ; 0 : aucun sous-titre
  std::uint64_t subs_rects_hash = 0;
  int width = 0;
  int height = 0;
  std::array<float, 4> crop{};  // la place de l'image dans la cible
  int ui_bottom = 0;            // ne compte qu'avec des sous-titres, qu'elle fait monter
  // Augmenté par l'écran à chaque changement de ce que le grade applique (paramètres, looks, KWin, couleur de
  // la source, nouvelle scène).
  std::uint64_t grade_serial = 0;
  FinalScaler final_scaler = FinalScaler::Default;
  bool deband = false;
  int sharpness = 0;
  double sar = 1.0;
  int transfer = 0;  // ceux de la cible (pl_color_transfer, pl_color_primaries)
  int primaries = 0;
  float max_luma = 0.0F;
  friend bool operator==(const ComposeKey&, const ComposeKey&) = default;
};

// FNV-1a sur la place et la couleur de chaque rectangle.
inline std::uint64_t rects_hash(std::span<const subs::SubtitleRect> rects) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  const auto mix = [&h](std::int64_t v) {
    for (int i = 0; i < 8; ++i) {
      h ^= static_cast<std::uint64_t>(v >> (i * 8)) & 0xFFU;
      h *= 0x100000001b3ULL;
    }
  };
  for (const subs::SubtitleRect& r : rects) {
    mix(r.src.x);
    mix(r.src.y);
    mix(r.src.width);
    mix(r.src.height);
    mix(r.dst_x);
    mix(r.dst_y);
    mix(static_cast<std::int64_t>(r.color.r) | static_cast<std::int64_t>(r.color.g) << 8 |
        static_cast<std::int64_t>(r.color.b) << 16 | static_cast<std::int64_t>(r.color.a) << 24);
  }
  return h;
}

// Ce que l'écran tient d'une présentation à l'autre, hors de la demande.
struct ComposeState {
  std::array<float, 4> crop{};
  std::uint64_t grade_serial = 0;
  FinalScaler final_scaler = FinalScaler::Default;
  bool deband = false;
  int sharpness = 0;
  double sar = 1.0;
  int transfer = 0;
  int primaries = 0;
  float max_luma = 0.0F;
};

inline ComposeKey compose_key(const RingFrame* frame, const subs::SubtitleFrame* subtitles, int width,
                              int height, int ui_bottom, const ComposeState& state) {
  ComposeKey key;
  if (frame != nullptr) {
    key.has_frame = true;
    key.slot = frame->slot;
    key.generation = frame->generation;
    key.epoch = frame->epoch;
    key.ready_value = frame->ready_value;
  }
  // Un atlas vide ou sans rectangle ne dessine rien : comme aucun sous-titre.
  if (subtitles != nullptr && subtitles->atlas != nullptr && !subtitles->rects.empty()) {
    key.subs_generation = subtitles->atlas->generation;
    key.subs_rects_hash = rects_hash(subtitles->rects);
    key.ui_bottom = ui_bottom;
  }
  key.width = width;
  key.height = height;
  key.crop = state.crop;
  key.grade_serial = state.grade_serial;
  key.final_scaler = state.final_scaler;
  key.deband = state.deband;
  key.sharpness = state.sharpness;
  key.sar = state.sar;
  key.transfer = state.transfer;
  key.primaries = state.primaries;
  key.max_luma = state.max_luma;
  return key;
}

// La composition gardée sous `kept` sert encore pour `now`.
inline bool reusable(const std::optional<ComposeKey>& kept, const ComposeKey& now) {
  return kept.has_value() && *kept == now;
}

}  // namespace ajp::app::play
