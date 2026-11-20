// Les images, les mesures de paires et les tenseurs de DistilDRBA dans l'étage GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "core/frame.h"
#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"

namespace kirari::app::play {

// DistilDRBA : l'image d'avant la paire et celle d'après, à la taille de travail, chacune
// valable si rien ne la sépare de la paire (même époque, pas de coupe) ; les deux entrées du
// graphe ; les mesures de (b, suivante), gardées pour la paire suivante.
struct DrbaBuffers {
  std::unique_ptr<gpu::TensorBuffer> before;
  std::unique_ptr<gpu::TensorBuffer> after;
  std::unique_ptr<gpu::TensorBuffer> pair01;  // 1 × 7 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> pair12;
  std::unique_ptr<gpu::TensorBuffer> flow;  // variante mouvement : 1 × 4 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> mask;  // 1 × 1 × ph × pw
  // Les deux images dont pair01 et pair12 tiennent les plans RGB : sur le même trio, seul l'instant
  // change (rife_time), y compris quand la paire suivante a repris les trois tampons tels quels.
  struct Filled {
    const gpu::Tensor* first = nullptr;
    const gpu::Tensor* second = nullptr;
  };
  Filled filled01;
  Filled filled12;
  bool time_only = true;  // faux si le backend refuse rife_time
  bool before_ok = false;
  bool after_ok = false;
  bool after_held = false;  // (b, after) est un dessin tenu : les instants après ½ sont b
  bool continuous = false;  // la paire en cours suit la précédente sans coupe, mesures prises
  std::optional<SourceId> after_id;
  bool after_debanded = false;  // l'image d'après, ingérée en avance, a été débandée (A09-07)
  std::uint32_t after_epoch = 0;
  gpu::Reduction after_reduce;  // la réduction de son ingestion : changée en direct, refaite (C09-07)
  std::optional<gpu::PairScores> after_scores;  // mesures de (b, after), si prises
  // Mesure de (b, after) lancée sans attente (scores_start) : relue à la première sortie qui
  // en a besoin, ou à l'arrivée de after, le GPU ayant alors fini.
  std::optional<std::uint64_t> after_ticket;
  SourceId ticket_left;
  SourceId ticket_right;
  // Les dates de la paire mesurée (b, after) et de celle du pont : la coupe se juge comme pour (a, b), avec
  // les plans de la pré-analyse (A09-01).
  std::int64_t left_pts_us = kNoPts;
  std::int64_t right_pts_us = kNoPts;
  std::int64_t bridge_left_pts_us = kNoPts;
  std::int64_t bridge_right_pts_us = kNoPts;
  // Image sautée en retard (passed) : déjà ingérée comme after, elle reste l'image d'avant de la
  // paire qui suit la reprise (D3). bridge : à mesurer contre l'image de reprise ; bridge_ticket : cette
  // mesure, relue à l'arrivée de l'image d'après.
  bool bridge = false;
  // Variante mouvement estimée plus petite (D4) : before et after réduits, comme small_ pour a et b ;
  // small_taken : la réduite de b vient de small_after, sans nouvelle ingestion.
  std::unique_ptr<gpu::TensorBuffer> small_before;
  std::unique_ptr<gpu::TensorBuffer> small_after;
  bool small_taken = false;
  std::optional<std::uint64_t> bridge_ticket;
  std::uint64_t runs = 0;
  std::uint64_t edges = 0;  // passes sans l'image d'un côté (début, coupe, fin)
  std::uint64_t held = 0;   // instants rendus sans passe : le côté de la paire est un dessin tenu
};

}  // namespace kirari::app::play
