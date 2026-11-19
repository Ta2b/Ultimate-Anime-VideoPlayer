// L'interpolation par calques et la plaque de décor dans l'étage gpu (upscale_stage.h, docs/calques.md) :
// l'état d'une lecture, les décisions de layer_plan.h appliquées au GPU, et le journal de fin.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>

#include "app/play/layer_plan.h"
#include "app/play/rife_plan.h"
#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"

namespace kirari::app::play {

struct LayerSetup {
  // play.toml [interpolation] layers : les calques sur les images intermédiaires.
  bool layers = false;
  // Le SPAN de la plaque (play.toml [interpolation] background_plate) : le modèle de SPAN en FP16, aux formes
  // de travail ; nullptr : sans plaque.
  gpu::Engine* plate = nullptr;
};

// Les images intermédiaires écrites directement dans le slot (BackendCaps::ring_targets) : pas avec les
// calques, qui composent sur place dans un tenseur avant l'écriture du slot.
constexpr bool layers_ring_direct(const LayerSetup& setup, bool direct) {
  return direct && !setup.layers;
}

class LayerStage {
 public:
  // La taille de travail (work_shape, celle de la carte) ; le tenseur de la plaque, à la taille de sortie de
  // SPAN (out_shape), quand elle tourne.
  std::expected<void, Refusal> allocate(gpu::Backend& backend, const LayerSetup& setup,
                                        const gpu::Shape& work_shape, const gpu::Shape& out_shape,
                                        bool spanned);
  // La paire (a, b) arrive : sa carte (facts.scene) décide les calques de ses images intermédiaires et la
  // plaque de b ; une autre époque ou une coupe (pair_cut) repart de zéro. smooth_first ou smooth_second
  // (« Lisser les dessins tenus » mène la paire, ses instants ne sont plus ceux de la caméra) : ni calques ni
  // plaque.
  void arrive(const PairFacts& facts, const LayerSetup& setup, bool smooth_first, bool smooth_second);
  // Un saut, une nouvelle époque : la plaque et la paire s'oublient.
  void forget();
  bool pair_active() const { return pair_.has_value(); }
  // L'image intermédiaire à l'instant t, déjà dans out (warp de RIFE) : son décor repris de a et b.
  std::expected<void, Refusal> compose_pair(gpu::KernelOps& ops, const gpu::Tensor& a, const gpu::Tensor& b,
                                            const gpu::Tensor& out, float t, bool cubic);
  // SPAN de b vient d'être enfilé dans spanned (sur la voie de SPAN) : la plaque, si la paire en a une. Une
  // clé passe d'abord rgb (l'image de travail de b) dans le SPAN de la plaque. Avant le grade : la plaque
  // n'est jamais gradée, l'image composée l'est comme toute autre.
  std::expected<void, Refusal> after_span(gpu::Backend& backend, const LayerSetup& setup,
                                          const gpu::Tensor& rgb, const gpu::Tensor& spanned, bool cubic);
  // Le journal de fin : paires avec calques, sans (par raison), clés et sources de la plaque.
  void log_summary(const LayerSetup& setup) const;

 private:
  std::optional<PairLayers> pair_;
  bool pair_layers_ = false;  // pair_ sert aussi aux images intermédiaires (setup.layers)
  PlateTrack plate_track_;
  PlateStep plate_step_;
  std::unique_ptr<gpu::TensorBuffer> plate_;
  bool plate_ready_ = false;  // plate_ tient la clé du suivi (son SPAN a tourné)
  bool key_pending_ = false;  // la clé de la paire attend le SPAN de la plaque (after_span)
  struct Counts {
    std::uint64_t pairs = 0;     // paires qui ont des calques
    std::uint64_t composed = 0;  // images intermédiaires composées
    std::array<std::uint64_t, kLayerSkipCount> skipped{};
    std::uint64_t smooth = 0;  // paires menées par le Smooth
    double share = 0.0;        // somme des parts de décor des paires avec calques
    std::uint64_t keys = 0;    // images passées par le SPAN de la plaque
    std::uint64_t plated = 0;  // sources dont le décor vient de la plaque
    std::uint64_t refused = 0;
  } counts_;
  int work_width_ = 0;
  int work_height_ = 0;
  bool refused_ = false;  // un refus du backend coupe calques et plaque pour la lecture
};

}  // namespace kirari::app::play
