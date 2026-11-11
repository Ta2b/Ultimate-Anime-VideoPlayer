// Détection des flashs selon la recommandation UIT-R BT.1702-2 (P3.51) : un flash est une paire de variations
// opposées de luminance d'au moins 20 cd/m² sur un quart de l'écran au moins, l'image la plus sombre sous
// 160 cd/m². Plus de trois flashs dans une seconde rendent la séquence dangereuse pour les personnes
// photosensibles.
//
// Le lecteur juge les seize zones de l'analyse (la grille 4 × 4 de FrameStats), chacune en cd/m² : une
// variation compte quand quatre zones au moins (le quart de l'écran) tournent dans le même sens à 150 ms
// près (A42-08 ; la moyenne de toute l'image manquait un quart d'écran passant de 20 à 60 cd/m²). Les
// luminances sont celles de la source multipliées par `gain` : le grade HDR éclaircit, et la recommandation
// parle de ce que l'écran montre.
//
// Le plafond que le lecteur pose alors (250 nits) retire l'amplification du grade ; il ne rend pas sûre une
// source déjà dangereuse.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>

namespace ajp::media {

inline constexpr double kFlashStepNits = 20.0;
inline constexpr double kFlashDarkNits = 160.0;
inline constexpr int kFlashesPerSecond = 3;  // au-delà, la séquence est dangereuse
// Le plafond tient une seconde après le dernier instant dangereux.
inline constexpr std::int64_t kFlashHoldUs = 1'000'000;
inline constexpr std::size_t kFlashZones = 16;
inline constexpr int kFlashQuarter = 4;                    // zones : le quart de l'écran
inline constexpr std::int64_t kFlashTogetherUs = 150'000;  // des zones qui tournent ensemble

class FlashGuard {
 public:
  // Ce que le grade fait à la lumière de la source (blanc de sortie sur blanc de la source) ; 1 sans grade.
  void set_gain(double gain) { gain_ = gain > 0.0 ? gain : 1.0; }

  // L'image source à pts_us, ses seize zones en cd/m². Vrai tant que le plafond doit tenir.
  bool push(std::int64_t pts_us, std::span<const float, kFlashZones> zone_nits) {
    if (pts_us <= last_pts_) reset();
    last_pts_ = pts_us;
    missing_ = false;
    int ups = 0, downs = 0;
    for (std::size_t i = 0; i < kFlashZones; ++i) {
      Zone& zone = zones_.at(i);
      const double nits = static_cast<double>(zone_nits[i]) * gain_;
      if (!started_) {
        zone = Zone{.low = nits, .high = nits};
        continue;
      }
      // Suit l'extrême du sens courant ; une variation opposée d'au moins kFlashStepNits fait tourner la
      // zone. Au départ le sens est inconnu : la première variation compte, montée ou descente (A42-17).
      if (zone.turned >= 0) zone.high = std::max(zone.high, nits);
      if (zone.turned <= 0) zone.low = std::min(zone.low, nits);
      if (zone.turned >= 0 && zone.high - nits >= kFlashStepNits && nits < kFlashDarkNits) {
        zone = Zone{.low = nits, .high = nits, .at_us = pts_us, .turned = -1};
      } else if (zone.turned <= 0 && nits - zone.low >= kFlashStepNits && zone.low < kFlashDarkNits) {
        zone = Zone{.low = nits, .high = nits, .at_us = pts_us, .turned = 1};
      }
      if (zone.turned != 0 && pts_us - zone.at_us <= kFlashTogetherUs) ++(zone.turned > 0 ? ups : downs);
    }
    started_ = true;
    // Une transition de l'image : le quart de l'écran tourne dans un sens, l'autre que la fois d'avant.
    if (ups >= kFlashQuarter && sense_ <= 0) {
      sense_ = 1;
      turns_.push_back(pts_us);
    } else if (downs >= kFlashQuarter && sense_ >= 0) {
      sense_ = -1;
      turns_.push_back(pts_us);
    }
    return judge(pts_us);
  }

  // La même luminance partout (les essais, et une source sans zones).
  bool push(std::int64_t pts_us, double nits) {
    std::array<float, kFlashZones> zones{};
    zones.fill(static_cast<float>(nits));
    return push(pts_us, zones);
  }

  // Une image dont l'analyse manque (A42-18) : au milieu d'une suite de variations déjà rapide (cinq
  // transitions dans la seconde : un flash caché de plus la rendrait dangereuse), elle compte pour une
  // transition, une fois par suite d'images perdues. Hors de là, rien : une seule transition suivie d'images
  // perdues (le début d'une vidéo, un GPU chargé) plafonnait le grade sur une image fixe.
  bool missed(std::int64_t pts_us) {
    if (pts_us <= last_pts_) return pts_us < limited_until_;
    while (!turns_.empty() && turns_.front() <= pts_us - 1'000'000) turns_.pop_front();
    if (!missing_ && std::ssize(turns_) >= 2 * kFlashesPerSecond - 1) turns_.push_back(pts_us);
    missing_ = true;
    return judge(pts_us);
  }

  // Le début de la séquence dangereuse en cours : la première transition de la seconde qui a déclenché le
  // plafond. Rien hors plafond.
  std::optional<std::int64_t> danger_from() const {
    return last_pts_ < limited_until_ ? std::optional(danger_from_) : std::nullopt;
  }
  std::int64_t limited_until() const { return limited_until_; }

  void reset() {
    const double gain = gain_;
    *this = FlashGuard{};
    gain_ = gain;
  }

 private:
  struct Zone {
    double low = 0.0, high = 0.0;
    std::int64_t at_us = 0;  // l'instant de son dernier tour
    int turned = 0;          // 1 : elle vient de monter ; -1 : de descendre ; 0 : pas encore tourné
  };

  bool judge(std::int64_t pts_us) {
    while (!turns_.empty() && turns_.front() <= pts_us - 1'000'000) turns_.pop_front();
    // Deux transitions font un flash : plus de trois flashs, au moins sept transitions dans la seconde.
    if (std::ssize(turns_) > 2 * kFlashesPerSecond) {
      if (pts_us >= limited_until_) danger_from_ = turns_.front();
      limited_until_ = pts_us + kFlashHoldUs;
    }
    return pts_us < limited_until_;
  }

  std::array<Zone, kFlashZones> zones_{};
  std::deque<std::int64_t> turns_;
  std::int64_t last_pts_ = INT64_MIN;
  std::int64_t limited_until_ = INT64_MIN;
  std::int64_t danger_from_ = 0;
  double gain_ = 1.0;
  int sense_ = 0;  // le sens de la dernière transition comptée
  bool started_ = false;
  bool missing_ = false;  // la dernière image vue est une analyse perdue
};

}  // namespace ajp::media
