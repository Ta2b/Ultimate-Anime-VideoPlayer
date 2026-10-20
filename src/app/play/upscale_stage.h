// L'étage gpu de la tranche verticale sur un backend chargé (gpu/backend.h). Chaque image
// décodée est ingérée en RGB FP16 à la taille de travail (réduction Area fusionnée si elle est
// plus petite que la source). Avec RIFE, la paire (image précédente, image) donne les instants de
// sa grille de sortie (core/cadence.h : t = k/N pour ×N ; ×1,5 alterne t = 2/3, puis t = 1/3 et
// l'image), l'image elle-même quand elle est sur la grille (rife_plan.h décide si RIFE tourne).
// Chaque sortie est agrandie par SPAN (×2) et écrite dans un slot du ring que render présente.
// Sans moteur SPAN, la sortie passe à la taille de travail.
//
// GMFSS Fortuna (kernels/spec/gmfss.h) remplace RIFE, à la taille de travail : son graphe « paire »
// une fois par paire, puis par image intermédiaire six softmax splats et son graphe « fusion ».
//
// Ordre de la chaîne : RIFE à la taille de travail (720p), avant SPAN. RIFE y traite quatre fois
// moins de pixels qu'à 1440p, et SPAN nettoie ensuite ses petits défauts.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/ring_stage.h"
#include "core/cadence.h"
#include "core/ring.h"
#include "gpu/backend.h"

namespace ajp::app::play {

struct UpscaleSetup {
  gpu::Backend* backend = nullptr;  // vit plus longtemps que l'étage
  gpu::Engine* engine = nullptr;    // SPAN aux formes de travail ; nullptr : sans agrandissement
  gpu::Engine* rife = nullptr;      // RIFE à rife_geometry(travail) ; nullptr : sans interpolation
  // Variante mouvement de RIFE (motion_model.h) : flot et masque, warp à la sortie. Remplace rife ;
  // une passe au milieu de la paire la sert toute quand le mouvement est faible.
  gpu::Engine* rife_motion = nullptr;
  // Flux optique matériel à la place de RIFE (KernelOps::open_optical_flow, déjà ouvert pour
  // rife_width × rife_height) : la grille de ses vecteurs, en pixels ; 0 : RIFE ou rien. Un seul
  // mouvement par paire, au milieu ; les autres instants s'en déduisent au warp.
  int optical_flow_grid = 0;
  // GMFSS à la place de RIFE : ses deux moteurs, aux plans remplis de rife_geometry(travail) ; les
  // deux ou aucun.
  gpu::Engine* gmfss_pair = nullptr;
  gpu::Engine* gmfss_fusion = nullptr;
  Factor rife_factor{.num = 1, .den = 1};  // ×num/den quand l'interpolation tourne (au plus ×8)
  bool true_hdr = false;  // RTX HDR à l'écriture du ring (backend déjà ouvert pour la sortie)
  // Réglages changés en cours de lecture (facteur RIFE, réduction, RTX HDR) et compteurs de RIFE ;
  // nullptr : rife_factor et Area pour toute la lecture.
  std::shared_ptr<LiveSettings> live;
  // Horloge maître : une paire dont le premier instant intermédiaire est déjà passé n'est pas
  // interpolée (la chaîne rattrape au lieu de prendre du retard sur le son). nullptr : jamais.
  const MasterClock* clock = nullptr;
  int source_width = 0;  // images décodées
  int source_height = 0;
  int work_width = 0;  // entrée des moteurs, au plus la source ; 0 : la source
  int work_height = 0;
  // Taille où RIFE (variante mouvement seulement) estime le mouvement, au plus la taille de travail ;
  // 0 : la taille de travail. Plus petite (720p pour une sortie 4K), une seconde ingestion la
  // donne, et le warp déforme les images de travail avec ce mouvement mis à l'échelle.
  int rife_width = 0;
  int rife_height = 0;
  std::vector<RingSlotDesc> slots;  // ring de render, à la taille de sortie de SPAN
};

class UpscaleStage final : public RingStage {
 public:
  // Refus : backend sans noyaux, slots absents ou d'une taille qui n'est pas la sortie du
  // moteur (ShapeInvalid), facteur RIFE hors plage, tenseurs refusés par le backend, thread refusé.
  static std::expected<std::unique_ptr<VideoStage>, Refusal> start(VideoStageInit init, UpscaleSetup setup);
  ~UpscaleStage() override;

 private:
  // Vrai noir : rampe TrueHDR au plus toutes les 250 ms.
  static constexpr std::int64_t kBlackRampGapUs = 250'000;
  bool black_ramp_pending_ = true;  // la première image la fait, si le vrai noir est voulu
  std::int64_t black_ramp_at_us_ = 0;
  UpscaleStage(VideoStageInit init, UpscaleSetup setup);
  std::expected<void, Refusal> allocate();
  std::expected<std::uint32_t, Refusal> outputs(const DecodedFrame& frame) override;
  std::expected<Output, Refusal> produce(const DecodedFrame& frame, std::uint32_t output, std::uint32_t slot,
                                         std::uint64_t wait_value, std::uint64_t signal_value) override;
  // Image intermédiaire output + 1 sur outputs_ par le mouvement (rife_motion ou flux optique)
  // dans middle_.
  std::expected<void, Refusal> interpolate_by_motion(std::uint32_t output);
  // RIFE image à l'instant t, recadrée dans middle_.
  std::expected<void, Refusal> interpolate_image(float t);
  // La variante mouvement de RIFE à l'instant t, dans flow et mask.
  std::expected<void, Refusal> run_motion(float t, const gpu::Tensor& flow, const gpu::Tensor& mask);
  bool interpolates() const {
    return setup_.rife != nullptr || setup_.rife_motion != nullptr || setup_.optical_flow_grid > 0 || gmfss();
  }
  bool gmfss() const { return setup_.gmfss_pair != nullptr; }
  // GMFSS (upscale_gmfss.cpp) : ses tenseurs, rangés dans l'ordre de ses moteurs ; puis l'image
  // intermédiaire output + 1 sur outputs_ dans middle_.
  std::expected<void, Refusal> allocate_gmfss();
  std::expected<void, Refusal> interpolate_gmfss(std::uint32_t output);
  // Le mouvement est estimé à la taille de RIFE puis déformé au warp (variante mouvement ou flux
  // optique), plutôt que l'image de RIFE elle-même.
  bool by_motion() const { return setup_.rife_motion != nullptr || setup_.optical_flow_grid > 0; }
  // L'image où RIFE et les mesures de la paire travaillent : la réduite s'il y en a une.
  const gpu::Tensor& rife_source(std::uint32_t index) const {
    return small_[index] ? small_[index]->tensor() : rgb_[index]->tensor();
  }
  // SPAN (s'il est là) puis l'écriture dans le slot.
  std::expected<void, Refusal> finish(const gpu::Tensor& rgb, std::uint32_t slot, const gpu::RingWrite& sync);

  UpscaleSetup setup_;
  // Deux images de travail RGB, à tour de rôle : current_ vient d'être ingérée, l'autre est la
  // précédente (valable si previous_ est rempli).
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> rgb_;
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> small_;  // à la taille de RIFE, si plus petite
  std::uint32_t current_ = 0;
  struct Previous {
    std::int64_t pts_us = 0;
    SourceId id;
    std::uint32_t epoch = 0;
  };
  std::optional<Previous> previous_;
  // Les sorties de l'image en cours, dans l'ordre : instants de la grille, t = 1 pour l'image.
  struct Instant {
    float t = 1.0F;
    std::int64_t pts_us = 0;
  };
  std::array<Instant, core::kMaxInstantsPerPair> instants_{};
  std::uint32_t outputs_ = 1;
  std::uint32_t intermediates_ = 0;  // celles qui sont interpolées
  std::optional<core::CadenceAccumulator> cadence_;
  Factor cadence_factor_;
  vendor_hdr::InputPreparation prepared_;        // couleur de l'image en cours, pour RTX HDR
  std::unique_ptr<gpu::TensorBuffer> rife_in_;   // 1 × 11 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> rife_out_;  // 1 × 3 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> middle_;    // 1 × 3 × h × w : image intermédiaire recadrée
  // Mouvement au milieu de la paire (variante mouvement de RIFE, ou flux optique sur sa grille) :
  // deux, à tour de rôle, celui de la paire précédente servant à garder les poses.
  struct MotionBuffers {
    std::unique_ptr<gpu::TensorBuffer> flow;  // 1 × 4 × ph × pw (RIFE) ou cells_y × cells_x
    std::unique_ptr<gpu::TensorBuffer> mask;  // 1 × 1, même taille
  };
  std::array<MotionBuffers, 2> mids_;
  std::uint32_t mid_ = 0;  // celui de la paire en cours
  gpu::Motion mid_motion(std::uint32_t index) const;
  // Variante mouvement, mouvement fort : celui d'un autre instant que le milieu.
  std::unique_ptr<gpu::TensorBuffer> flow_;
  std::unique_ptr<gpu::TensorBuffer> mask_;
  std::uint64_t optical_flow_pairs_ = 0;
  // GMFSS : l'entrée de la paire, chaque tenseur nommé de ses graphes (sorties de la paire,
  // sources splattées), et les six splats d'une image intermédiaire.
  struct GmfssSplat {
    int side = 0;
    gpu::Tensor source;
    gpu::Tensor motion;
    gpu::Tensor warped;
  };
  struct GmfssBuffers {
    std::unique_ptr<gpu::TensorBuffer> frames;  // 1 × 6 × ph × pw
    std::vector<std::pair<std::string, std::unique_ptr<gpu::TensorBuffer>>> named;
    std::vector<gpu::Tensor> pair_outputs;   // dans l'ordre de output_names() du moteur paire
    std::vector<gpu::Tensor> fusion_inputs;  // dans l'ordre de input_names() du moteur fusion
    std::vector<GmfssSplat> splats;
  };
  GmfssBuffers gmfss_;
  std::uint64_t gmfss_pairs_ = 0;
  // Garder les poses (kernels/spec/poses.h) : ce qu'a bougé la paire précédente, et la paire en
  // cours. Inconnu après une coupe, un saut, une paire non interpolée : le warp est alors le simple.
  enum class PairMotion : std::uint8_t { Unknown, Still, Moving };
  PairMotion before_ = PairMotion::Unknown;
  PairMotion now_ = PairMotion::Unknown;
  bool poses_ = false;       // garder les poses pour la paire en cours (réglage lu à son début)
  bool single_run_ = false;  // la paire en cours n'a qu'une passe RIFE, au milieu
  std::uint64_t single_pairs_ = 0;
  std::uint64_t multi_pairs_ = 0;
  std::uint64_t rife_runs_ = 0;
  std::uint64_t late_pairs_ = 0;               // paires non interpolées pour rattraper l'horloge
  std::unique_ptr<gpu::TensorBuffer> output_;  // 1 × 3 × 2h × 2w ; absent sans SPAN
  // Pour le journal de fin : paires vues, interpolées, coupes, dessins tenus.
  std::uint64_t pairs_ = 0;
  std::uint64_t interpolated_ = 0;
  std::uint64_t cuts_ = 0;
  std::uint64_t held_ = 0;
  std::vector<std::int64_t> scores_wait_us_;  // attente hôte de scores_now, pour le profil
};

}  // namespace ajp::app::play
