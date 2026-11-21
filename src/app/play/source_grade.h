// Le grade avant le warp (toujours pris, sans réglage) : l'image source sortie de SPAN, ou sans SPAN sa copie
// à la taille de travail, est gradée une fois, sur place, avant que le warp de RIFE n'en tire les images
// intermédiaires. Avec RIFE ×8, le grade tourne une fois par source au lieu de huit.
//
// L'étage d'agrandissement (upscale_stage.h) appelle grade() juste après SPAN, sur son propre fil : le
// tenseur est écrit dans une image du même protocole que le ring (core/ring.h), gradé par libplacebo hors
// écran avec les paramètres que l'écran applique à cet instant, puis relu dans le même tenseur
// (KernelOps::read_ring). Le warp mélange alors des images gradées, et l'écran les montre telles quelles
// (RingFrame::graded).
//
// Ce qui change à l'image : le grade d'une image intermédiaire est le mélange des grades de ses deux sources
// au lieu du grade de leur mélange ; un réglage changé en pause ne se voit qu'à l'image source suivante ; les
// paramètres du plan sont ceux de l'écran au moment de l'agrandissement, une image source plus tôt environ.
// Les éclats sur les seuls effets (effect_book.h) suivent la carte de la scène de chaque source gradée, à son
// propre masque lissé, et non celui de l'écran.
// Sortie HDR10 seulement : en SDR, la marge des éclats change à la coupe de plan et doit suivre l'image
// affichée.
#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

#include <libplacebo/colorspace.h>

#include "app/play/effect_book.h"
#include "core/frame.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "render/cube_lut.h"
#include "render/grade.h"
#include "render/grade_params.h"

namespace kirari::app::play {

// Ce que l'écran applique à sa dernière présentation, publié pour le fil de l'agrandissement.
struct SourceGradeState {
  // Le grade de la sortie ; rien : pas de grade avant le warp maintenant (grade coupé ou absent, sortie SDR,
  // source HDR, Anime4K en jeu).
  std::optional<render::GradeParams> params;
  std::array<std::shared_ptr<const render::CubeLut>, render::kLookSlots> looks;
  std::optional<render::Grade::KwinMatch> kwin;
  pl_color_space source{};  // la couleur des images du ring
  pl_color_space target{};  // celle de la sortie : les images gradées y sont
  bool deband = false;      // débandage de libplacebo, sans grain (celui de l'écran sous le grade)
  std::string_view off;     // sans params : pourquoi, pour le journal de fin du grade avant le warp
};

class SourceGradeShare {
 public:
  void put(SourceGradeState state) {
    const std::scoped_lock lock(mutex_);
    state_ = std::move(state);
  }
  SourceGradeState get() const {
    const std::scoped_lock lock(mutex_);
    return state_;
  }
  bool available() const {
    const std::scoped_lock lock(mutex_);
    return state_.params.has_value();
  }
  // L'autre sens : le masque des effets des dernières sources gradées, pour le journal d'un instantané. Le
  // grade avant le warp travaille en avance sur l'écran : l'image montrée retrouve le sien par son pts.
  void note_effects(const EffectSummary& seen) {
    const std::scoped_lock lock(mutex_);
    effects_[next_effect_] = seen;
    next_effect_ = (next_effect_ + 1) % effects_.size();
  }
  // Celui de la source de pts pts_us s'il est encore gardé, sinon le plus récent (son pts le dit).
  EffectSummary effects(std::int64_t pts_us) const {
    const std::scoped_lock lock(mutex_);
    for (const EffectSummary& seen : effects_) {
      if (pts_us != kNoPts && seen.pts_us == pts_us) return seen;
    }
    return effects_[(next_effect_ + effects_.size() - 1) % effects_.size()];
  }

 private:
  mutable std::mutex mutex_;
  SourceGradeState state_;
  std::array<EffectSummary, 64> effects_{};
  std::size_t next_effect_ = 0;
};

// La source gradée : son époque, son pts (le journal de fin dit où voir le masque), et la note des éclats sur
// les seuls effets de sa paire (EffectBook::for_source) ; rien sans carte, sans --scene-map ou avec
// --no-effect-highlights.
struct GradedSource {
  std::uint32_t epoch = 0;
  std::int64_t pts_us = kNoPts;
  std::optional<EffectNote> effects;
  std::optional<kernels::SceneGlowTiles> glow;
};

class SourceGrader {
 public:
  SourceGrader() = default;
  SourceGrader(const SourceGrader&) = delete;
  SourceGrader& operator=(const SourceGrader&) = delete;
  SourceGrader(SourceGrader&&) = delete;
  SourceGrader& operator=(SourceGrader&&) = delete;
  virtual ~SourceGrader() = default;
  virtual bool available() const { return false; }
  // rgb (RGB planaire de la taille des slots du ring, la sortie de SPAN) gradé sur place, sur la voie
  // courante du backend. Vrai : gradé ; faux : rien à grader maintenant, l'image reste telle quelle et
  // l'écran la gradera. Refus : le backend sans read_ring, une image ou un rendu refusés ; l'étage cesse
  // alors de grader, et l'écran reprend le grade. debanded : la source a été débandée avant SPAN
  // (KernelOps::deband), libplacebo ne la débande plus. source : l'époque et le masque des effets de l'image.
  virtual std::expected<bool, Refusal> grade(gpu::KernelOps& ops, const gpu::Tensor& rgb, bool debanded,
                                             const GradedSource& source) = 0;
};

}  // namespace kirari::app::play
