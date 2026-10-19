// L'étage gpu de la tranche verticale sur un backend chargé (gpu/backend.h). Chaque image
// décodée est ingérée en RGB FP16 à la taille de travail (réduction Area fusionnée si elle est
// plus petite que la source). Avec RIFE, la paire (image précédente, image) donne d'abord les
// images intermédiaires à t = k/N (rife_plan.h décide si RIFE tourne), puis l'image elle-même.
// Chaque sortie est agrandie par SPAN (×2) et écrite dans un slot du ring que render présente.
// Sans moteur SPAN, la sortie passe à la taille de travail.
//
// Ordre de la chaîne : RIFE à la taille de travail (720p), avant SPAN. RIFE y traite quatre fois
// moins de pixels qu'à 1440p, et SPAN nettoie ensuite ses petits défauts.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <vector>

#include "app/play/pipeline.h"
#include "app/play/ring_stage.h"
#include "core/ring.h"
#include "gpu/backend.h"

namespace ajp::app::play {

struct UpscaleSetup {
  gpu::Backend* backend = nullptr;  // vit plus longtemps que l'étage
  gpu::Engine* engine = nullptr;    // SPAN aux formes de travail ; nullptr : sans agrandissement
  gpu::Engine* rife = nullptr;      // RIFE à rife_geometry(travail) ; nullptr : sans interpolation
  std::uint32_t rife_factor = 1;    // ×N quand RIFE tourne (2 à kMaxRifeFactor)
  bool true_hdr = false;            // RTX HDR à l'écriture du ring (backend déjà ouvert pour la sortie)
  int source_width = 0;             // images décodées
  int source_height = 0;
  int work_width = 0;  // entrée des moteurs, au plus la source ; 0 : la source
  int work_height = 0;
  std::vector<RingSlotDesc> slots;  // ring de render, à la taille de sortie de SPAN
};

class UpscaleStage final : public RingStage {
 public:
  // Refus : backend sans noyaux, slots absents ou d'une taille qui n'est pas la sortie du
  // moteur (ShapeInvalid), facteur RIFE hors plage, tenseurs refusés par le backend, thread refusé.
  static std::expected<std::unique_ptr<VideoStage>, Refusal> start(VideoStageInit init, UpscaleSetup setup);
  ~UpscaleStage() override;

 private:
  UpscaleStage(VideoStageInit init, UpscaleSetup setup);
  std::expected<void, Refusal> allocate();
  std::expected<std::uint32_t, Refusal> outputs(const DecodedFrame& frame) override;
  std::expected<Output, Refusal> produce(const DecodedFrame& frame, std::uint32_t output, std::uint32_t slot,
                                         std::uint64_t wait_value, std::uint64_t signal_value) override;
  // SPAN (s'il est là) puis l'écriture dans le slot.
  std::expected<void, Refusal> finish(const gpu::Tensor& rgb, std::uint32_t slot, const gpu::RingWrite& sync);

  UpscaleSetup setup_;
  // Deux images de travail RGB, à tour de rôle : current_ vient d'être ingérée, l'autre est la
  // précédente (valable si previous_ est rempli).
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> rgb_;
  std::uint32_t current_ = 0;
  struct Previous {
    std::int64_t pts_us = 0;
    SourceId id;
    std::uint32_t epoch = 0;
  };
  std::optional<Previous> previous_;
  std::uint32_t outputs_ = 1;                    // sorties de l'image en cours
  vendor_hdr::InputPreparation prepared_;        // couleur de l'image en cours, pour RTX HDR
  std::unique_ptr<gpu::TensorBuffer> rife_in_;   // 1 × 11 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> rife_out_;  // 1 × 3 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> middle_;    // 1 × 3 × h × w : image intermédiaire recadrée
  std::unique_ptr<gpu::TensorBuffer> output_;    // 1 × 3 × 2h × 2w ; absent sans SPAN
  // Pour le journal de fin : paires vues, interpolées, coupes, dessins tenus.
  std::uint64_t pairs_ = 0;
  std::uint64_t interpolated_ = 0;
  std::uint64_t cuts_ = 0;
  std::uint64_t held_ = 0;
};

}  // namespace ajp::app::play
