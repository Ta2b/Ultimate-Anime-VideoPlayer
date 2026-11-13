// Le grade avant le warp (PlaySettings::grade_before_warp, essai éteint par défaut) : l'image source sortie
// de SPAN est gradée une fois, sur place, avant que le warp de RIFE n'en tire les images intermédiaires. Avec
// RIFE ×8, le grade tourne ainsi une fois par source au lieu de huit fois, une par image affichée.
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
// Sortie HDR10 seulement : en SDR, la marge des éclats change à la coupe de plan et doit suivre l'image
// affichée.
#pragma once

#include <array>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>

#include <libplacebo/colorspace.h>

#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "render/cube_lut.h"
#include "render/grade.h"
#include "render/grade_params.h"

namespace ajp::app::play {

// Ce que l'écran applique à sa dernière présentation, publié pour le fil de l'agrandissement.
struct SourceGradeState {
  // Le grade de la sortie ; rien : pas de grade avant le warp maintenant (grade coupé ou absent, sortie SDR,
  // source HDR, netteté ou Anime4K en jeu).
  std::optional<render::GradeParams> params;
  std::array<std::shared_ptr<const render::CubeLut>, render::kLookSlots> looks;
  std::optional<render::Grade::KwinMatch> kwin;
  pl_color_space source{};  // la couleur des images du ring
  pl_color_space target{};  // celle de la sortie : les images gradées y sont
  bool deband = false;      // débandage de libplacebo, sans grain (celui de l'écran sous le grade)
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

 private:
  mutable std::mutex mutex_;
  SourceGradeState state_;
};

class SourceGrader {
 public:
  SourceGrader() = default;
  SourceGrader(const SourceGrader&) = delete;
  SourceGrader& operator=(const SourceGrader&) = delete;
  SourceGrader(SourceGrader&&) = delete;
  SourceGrader& operator=(SourceGrader&&) = delete;
  virtual ~SourceGrader() = default;
  // rgb (RGB planaire de la taille des slots du ring, la sortie de SPAN) gradé sur place, sur la voie
  // courante du backend. Vrai : gradé ; faux : rien à grader maintenant, l'image reste telle quelle et
  // l'écran la gradera. Refus : le backend sans read_ring, une image ou un rendu refusés ; l'étage cesse
  // alors de grader, et l'écran reprend le grade. debanded : la source a été débandée avant SPAN
  // (KernelOps::deband), libplacebo ne la débande plus.
  virtual std::expected<bool, Refusal> grade(gpu::KernelOps& ops, const gpu::Tensor& rgb, bool debanded) = 0;
};

}  // namespace ajp::app::play
