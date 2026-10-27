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
// Déformé après SPAN, chaque image intermédiaire se compose des images agrandies splattées par son
// flot, corrigées du résidu de la fusion : une passe de SPAN par image source au lieu d'une par sortie.
// DistilDRBA (kernels/spec/distil_drba.h) aussi, à la taille de travail : une passe de son graphe
// par image intermédiaire, sur trois images ; il lit l'image d'après la paire, une image de retard.
//
// Ordre de la chaîne : RIFE à la taille de travail (720p), avant SPAN. RIFE y traite quatre fois
// moins de pixels qu'à 1440p, et SPAN nettoie ensuite ses petits défauts.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/ring_stage.h"
#include "core/cadence.h"
#include "core/ring.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/warp.h"

namespace ajp::app::play {

// Les slots de l'export par puits CUDA (export_sink.h) : chaque slot est une image de l'encodeur,
// que write_slot remplit en YUV (KernelOps::write_yuv) au lieu du ring ; done_value est la valeur
// qu'aurait signalée le slot du ring (RingFrame::ready_value). Appelé depuis le thread gpu.
class YuvSlots {
 public:
  virtual ~YuvSlots() = default;
  virtual gpu::YuvTarget target(std::uint32_t slot, std::uint64_t done_value) const = 0;
};

struct UpscaleSetup {
  gpu::Backend* backend = nullptr;  // vit plus longtemps que l'étage
  gpu::Engine* engine = nullptr;    // SPAN aux formes de travail ; nullptr : sans agrandissement
  gpu::Engine* rife = nullptr;      // RIFE à rife_geometry(travail) ; nullptr : sans interpolation
  // Variante mouvement de RIFE (motion_model.h) : flot et masque, warp à la sortie. Remplace rife ;
  // une passe au milieu de la paire la sert toute quand le mouvement est faible.
  gpu::Engine* rife_motion = nullptr;
  int rife_planes = 11;  // plans d'entrée de rife et rife_motion : 11 (v1) ou 7 (v2)
  int rife_align = 64;   // multiple de remplissage du modèle RIFE : 64, ou 128 pour 4.25 lite
  // GMFSS à la place de RIFE : ses deux moteurs, aux plans remplis de rife_geometry(travail) ; les
  // deux ou aucun.
  gpu::Engine* gmfss_pair = nullptr;
  gpu::Engine* gmfss_fusion = nullptr;
  int gmfss_align = 64;  // multiple des plans remplis de GMFSS : 128 pour le flot à l'échelle 0,5
  // DistilDRBA à la place de RIFE, aux plans remplis de rife_geometry(travail).
  gpu::Engine* distil_drba = nullptr;
  int drba_align = 64;                     // multiple des plans remplis de DistilDRBA : 128 au demi-flot
  Factor rife_factor{.num = 1, .den = 1};  // ×num/den quand l'interpolation tourne (au plus ×8)
  bool true_hdr = false;  // RTX HDR à l'écriture du ring (backend déjà ouvert pour la sortie)
  // --one-lane : tout sur la voie Upscale, sans recouvrement. Plus lent, mais le temps GPU de
  // chaque moteur redevient son coût propre au lieu d'inclure celui des autres voies.
  bool one_lane = false;
  // Réglages changés en cours de lecture (facteur RIFE, réduction, RTX HDR) et compteurs de RIFE ;
  // nullptr : rife_factor et Area pour toute la lecture.
  std::shared_ptr<LiveSettings> live;
  // Horloge maître : les instants intermédiaires qui seront passés avant d'être prêts sont sautés,
  // la paire entière s'il n'en reste aucun (la chaîne rattrape au lieu de prendre du retard sur le
  // son). nullptr : jamais.
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
  // Variante mouvement de RIFE ou GMFSS avec SPAN : SPAN une fois par image source, le warp à la
  // taille de sortie sur les images agrandies (kernels/spec/warp.h : a, b et la sortie à une taille,
  // le mouvement mis à l'échelle ; kernels/spec/gmfss.h : les images agrandies splattées par le flot
  // du niveau 1, corrigées du résidu de la fusion). Sans effet pour les autres estimateurs ou sans
  // SPAN.
  bool warp_after_span = false;
  std::vector<RingSlotDesc> slots;  // ring de render, à la taille de sortie de SPAN
  // Export par puits CUDA : les images de l'encodeur derrière les slots (sans image ni sémaphore
  // dans slots), écrites par write_yuv ; ni écriture directe du ring, ni RTX HDR. nullptr : le ring.
  std::shared_ptr<const YuvSlots> yuv_slots;
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
  bool passed(const DecodedFrame& frame, const DecodedFrame& next) override;
  std::expected<Output, Refusal> produce(const DecodedFrame& frame, std::uint32_t output, std::uint32_t slot,
                                         std::uint64_t wait_value, std::uint64_t signal_value) override;
  // Où va une image intermédiaire : un tenseur, ou directement le slot du ring (ring_direct_).
  struct Target {
    const gpu::Tensor* tensor = nullptr;
    const gpu::RingTarget* ring = nullptr;
  };
  // Image intermédiaire output + 1 sur outputs_ par le mouvement de rife_motion, de a et b dans out
  // (les trois à une taille : celle de travail, ou de sortie après SPAN).
  std::expected<void, Refusal> interpolate_by_motion(std::uint32_t output, const gpu::Tensor& a,
                                                     const gpu::Tensor& b, const Target& out);
  // warp_blend et warp_blend_poses vers out (KernelOps, ou leurs variantes _ring).
  std::expected<void, Refusal> blend(const gpu::Tensor& a, const gpu::Tensor& b, const gpu::Motion& motion,
                                     float t, const Target& out) const;
  std::expected<void, Refusal> blend_poses(const gpu::Tensor& a, const gpu::Tensor& b,
                                           const gpu::Motion& motion, const gpu::Motion* previous, float t,
                                           const Target& out) const;
  // Le warp et la composition écrivent l'image intermédiaire directement dans le slot
  // (BackendCaps::ring_targets, étude 1) : après SPAN, ou sans SPAN par la variante mouvement ; pas
  // avec RTX HDR, qui lit un tenseur. middle_outs_ et middles_ ne sont alors pas alloués.
  bool ring_direct_ = false;
  // RIFE image à l'instant t, recadrée dans middle().
  std::expected<void, Refusal> interpolate_image(float t);
  // La variante mouvement de RIFE à l'instant t, dans flow et mask.
  std::expected<void, Refusal> run_motion(float t, const gpu::Tensor& flow, const gpu::Tensor& mask);
  bool interpolates() const {
    return setup_.rife != nullptr || setup_.rife_motion != nullptr || gmfss() || drba();
  }
  bool gmfss() const { return setup_.gmfss_pair != nullptr; }
  bool drba() const { return setup_.distil_drba != nullptr; }
  bool wants_next() const override { return drba(); }
  // DistilDRBA (upscale_drba.cpp) : ses tenseurs ; l'arrivée de l'image b à la place de son
  // ingestion (rgb_[current_] reçoit b, l'image d'avant la paire passe dans drba_.before) ; les
  // mesures de la paire (reprises de l'arrivée précédente si elle les a prises pour sa suivante) ;
  // la suivante ingérée dans drba_.after et mesurée contre b ; puis l'image intermédiaire
  // output + 1 sur outputs_ dans middle().
  std::expected<void, Refusal> allocate_drba();
  std::expected<void, Refusal> drba_arrive(const DecodedFrame& frame, const gpu::IngestSpec& spec);
  std::expected<gpu::PairScores, Refusal> drba_scores(const DecodedFrame& frame);
  std::expected<void, Refusal> drba_next(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                         bool measure);
  std::expected<void, Refusal> settle_after_scores();
  std::expected<void, Refusal> interpolate_drba(std::uint32_t output);
  // Mesures en avance (upscale_ahead.cpp) : quand l'image suivante attend déjà derrière celle-ci,
  // elle est ingérée et mesurée contre elle avant le travail de la paire en cours. À son arrivée,
  // l'image n'attend alors que cette mesure, finie depuis longtemps, et non toute la file du GPU :
  // l'hôte enfile la paire suivante pendant que le GPU finit celle-ci. Pas avec DistilDRBA, qui lit
  // déjà la suivante (drba_next).
  struct Ahead {
    std::unique_ptr<gpu::TensorBuffer> rgb;    // à la taille de travail
    std::unique_ptr<gpu::TensorBuffer> small;  // à la taille de RIFE, si small_
    std::optional<SourceId> id;                // l'image ingérée, si elle l'est
    std::uint32_t epoch = 0;
    gpu::Reduction reduce;
    std::uint64_t ticket = 0;  // mesures (left, id) lancées ; 0 : aucune
    SourceId left;
  };
  Ahead ahead_;
  bool ahead_scores_ = true;  // faux si le backend n'a pas scores_start
  bool looks_ahead() const { return interpolates() && !drba(); }
  // Retard mesuré des images de sortie (FrameLatency) ; 0 sans mesure.
  std::int64_t latency_us() const { return setup_.live ? setup_.live->latency().us() : 0; }
  std::expected<void, Refusal> allocate_ahead();
  // L'image déjà ingérée en avance passe dans rgb_[current_] (et small_) ; faux sinon.
  bool take_ahead(const DecodedFrame& frame, const gpu::IngestSpec& spec);
  // Les mesures de la paire (image précédente, image en cours) : lancées en avance si elles l'ont été.
  std::expected<gpu::PairScores, Refusal> pair_scores();
  // La suivante de frame, si elle attend, ingérée et mesurée contre frame.
  std::expected<void, Refusal> look_ahead(const DecodedFrame& frame, const gpu::IngestSpec& spec);
  // rife_in() à l'instant t pour la paire en cours : ses images y sont écrites une fois par paire
  // (prepare_rife_input), l'instant seul ensuite (rife_time), quand le backend le sait ; rien si
  // l'instant y est déjà.
  std::expected<void, Refusal> fill_rife_input(float t);
  // Les images de la paire dans l'autre entrée de RIFE, sur la voie des mesures, avant le travail
  // de la paire : elles s'écrivent pendant que RIFE lit encore l'entrée de la paire précédente.
  std::expected<void, Refusal> prepare_rife_input();
  // Les images de la paire en cours et l'instant t dans rife_in() : sans réécrire le maillage s'il
  // y est déjà (rife_pair), sinon tout (rife_input).
  std::expected<void, Refusal> write_rife_pair(float t);
  bool rife_in_pair_ = false;        // rife_in() tient les images de la paire en cours
  float rife_in_t_ = 0.0F;           // et cet instant
  bool rife_time_ = true;            // faux si le backend n'a pas rife_time
  bool rife_pair_op_ = true;         // faux si le backend n'a pas rife_pair
  std::array<bool, 2> rife_mesh_{};  // rife_ins_[i] tient le maillage, écrit par un rife_input
  // GMFSS (upscale_gmfss.cpp) : ses tenseurs, rangés dans l'ordre de ses moteurs ; puis l'image
  // intermédiaire output + 1 sur outputs_ dans middle(), ou, déformée après SPAN, de a et b (les
  // images agrandies) dans out.
  std::expected<void, Refusal> allocate_gmfss();
  std::expected<void, Refusal> interpolate_gmfss(std::uint32_t output);
  std::expected<void, Refusal> interpolate_gmfss_spanned(std::uint32_t output, const gpu::Tensor& a,
                                                         const gpu::Tensor& b, const Target& out);
  // Le graphe « paire » à la première image intermédiaire de la paire, puis les splats de l'instant
  // (les six de la fusion, et extra) et le graphe « fusion » dans rife_out_.
  std::expected<void, Refusal> run_gmfss(std::uint32_t output,
                                         std::span<const gpu::KernelOps::SplatJob> extra);
  // Le mouvement est estimé à la taille de RIFE puis déformé au warp (variante mouvement), plutôt
  // que l'image de RIFE elle-même.
  bool by_motion() const { return setup_.rife_motion != nullptr; }
  // L'image où RIFE et les mesures de la paire travaillent : la réduite s'il y en a une.
  const gpu::Tensor& rife_source(std::uint32_t index) const {
    return small_[index] ? small_[index]->tensor() : rgb_[index]->tensor();
  }
  // Warp après SPAN : les images sources sont agrandies à l'ingestion, dans spanned_.
  bool warps_spanned() const {
    return setup_.warp_after_span && (by_motion() || gmfss()) && setup_.engine != nullptr;
  }
  // SPAN de l'image en cours dans son tampon de spanned_, une fois (aucune passe sur un dessin tenu :
  // celui de la précédente).
  std::expected<void, Refusal> span_current();
  const gpu::Tensor& spanned(std::uint32_t index) const { return spanned_[span_of_[index]]->tensor(); }
  bool spanned_now_ = false;
  std::array<std::uint32_t, 2> span_of_{0, 1};  // tampon de spanned_ de chaque image de rgb_
  // Dessin tenu : la sortie de SPAN de la précédente sert (output_ si output_has_source_, ou son
  // tampon de spanned_).
  static constexpr int kMaxSpanReuse = 2;
  bool reuse_span_ = false;
  bool output_has_source_ = false;  // output_ tient SPAN de l'image précédente
  int spans_reused_in_row_ = 0;
  std::uint64_t spans_reused_ = 0;
  // SPAN (s'il est là) puis l'écriture dans le slot.
  std::expected<void, Refusal> finish(const gpu::Tensor& rgb, std::uint32_t slot, const gpu::RingWrite& sync);
  // L'écriture d'une image à la taille du ring (RTX HDR s'il est demandé, sa rampe de vrai noir).
  std::expected<void, Refusal> write_slot(const gpu::Tensor& image, std::uint32_t slot,
                                          const gpu::RingWrite& sync);

  UpscaleSetup setup_;
  // Voie du backend (gpu/lanes.h) des travaux qui suivent : ingestion et mesures sur Measure, SPAN
  // et écriture du ring sur Upscale (les écritures du ring signalent dans l'ordre sur une seule
  // file), RIFE, GMFSS, DistilDRBA et warp sur Interpolate. Le backend ordonne les voies d'après
  // les tenseurs lus et écrits.
  void use_lane(gpu::Lane lane) {
    lane_ = setup_.one_lane ? gpu::Lane::Upscale : lane;
    setup_.backend->use_lane(lane_);
  }
  gpu::Lane lane_ = gpu::Lane::Upscale;
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
  bool uninterpolated_ = false;      // la paire courante, laissée sans interpolation à dessein
  std::optional<core::CadenceAccumulator> cadence_;
  Factor cadence_factor_;
  vendor_hdr::InputPreparation prepared_;  // couleur de l'image en cours, pour RTX HDR
  // Tampons passés d'une voie à l'autre (entrée de RIFE, image intermédiaire) : deux quand le
  // backend a plusieurs voies, l'un s'écrit pendant que l'autre est encore lu ; un sinon.
  std::uint32_t copies_ = 1;
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> rife_ins_;  // 1 × rife_planes × ph × pw
  std::uint32_t rife_pair_ = 0;                                 // celle de la paire en cours
  const gpu::Tensor& rife_in() const { return rife_ins_[rife_pair_]->tensor(); }
  std::unique_ptr<gpu::TensorBuffer> rife_out_;  // 1 × 3 × ph × pw
  // 1 × 3 × h × w : image intermédiaire recadrée, une par instant à tour de rôle (next_middle).
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> middles_;
  std::uint32_t middle_at_ = 0;
  const gpu::Tensor& middle() const { return middles_[middle_at_]->tensor(); }
  void next_middle() { middle_at_ = (middle_at_ + 1) % copies_; }
  // Mouvement au milieu de la paire (variante mouvement de RIFE) : deux, à tour de rôle, celui de
  // la paire précédente servant à garder les poses.
  struct MotionBuffers {
    std::unique_ptr<gpu::TensorBuffer> flow;  // 1 × 4 × ph × pw
    std::unique_ptr<gpu::TensorBuffer> mask;  // 1 × 1, même taille
  };
  std::array<MotionBuffers, 2> mids_;
  std::uint32_t mid_ = 0;  // celui de la paire en cours
  gpu::Motion mid_motion(std::uint32_t index) const;
  // Variante mouvement, mouvement fort : celui d'un autre instant que le milieu.
  std::unique_ptr<gpu::TensorBuffer> flow_;
  std::unique_ptr<gpu::TensorBuffer> mask_;
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
    std::vector<gpu::KernelOps::SplatJob> jobs;  // les splats de l'instant, un seul passage
    std::optional<gpu::Tensor> time;             // union : l'instant de l'image intermédiaire, pour sa fusion
    // Déformation après SPAN : le flot du niveau 1 de chaque côté, sa grille (spec MotionGrid) pour
    // l'image de travail et pour l'image agrandie, la fenêtre de la fusion ; les images de travail
    // splattées (1 × 4 × h × w, couverture comprise) et les agrandies (1 × 4 × 2h × 2w).
    std::array<gpu::Tensor, 2> level1{};
    kernels::MotionGrid work_grid;
    kernels::MotionGrid spanned_grid;
    kernels::Window window;
    std::array<std::unique_ptr<gpu::TensorBuffer>, 2> low;
    std::array<std::unique_ptr<gpu::TensorBuffer>, 2> high;
  };
  GmfssBuffers gmfss_;
  std::uint64_t gmfss_pairs_ = 0;
  // DistilDRBA : l'image d'avant la paire et celle d'après, à la taille de travail, chacune
  // valable si rien ne la sépare de la paire (même époque, pas de coupe) ; les deux entrées du
  // graphe ; les mesures de (b, suivante), gardées pour la paire suivante.
  struct DrbaBuffers {
    std::unique_ptr<gpu::TensorBuffer> before;
    std::unique_ptr<gpu::TensorBuffer> after;
    std::unique_ptr<gpu::TensorBuffer> pair01;  // 1 × 7 × ph × pw
    std::unique_ptr<gpu::TensorBuffer> pair12;
    // Les deux images dont pair01 et pair12 tiennent les plans d'images : une autre sortie de la
    // même paire sur les mêmes images ne réécrit que l'instant (rife_time).
    struct Filled {
      const gpu::Tensor* first = nullptr;
      const gpu::Tensor* second = nullptr;
    };
    Filled filled01;
    Filled filled12;
    bool time_only = true;  // faux si le backend refuse rife_time
    bool before_ok = false;
    bool after_ok = false;
    bool continuous = false;  // la paire en cours suit la précédente sans coupe, mesures prises
    std::optional<SourceId> after_id;
    std::uint32_t after_epoch = 0;
    std::optional<gpu::PairScores> after_scores;  // mesures de (b, after), si prises
    // Mesure de (b, after) lancée sans attente (scores_start) : relue à la première sortie qui
    // en a besoin, ou à l'arrivée de after, le GPU ayant alors fini.
    std::optional<std::uint64_t> after_ticket;
    SourceId ticket_left;
    SourceId ticket_right;
    std::uint64_t runs = 0;
    std::uint64_t edges = 0;  // passes sans l'image d'un côté (début, coupe, fin)
  };
  DrbaBuffers drba_;
  // Garder les poses (kernels/spec/poses.h) : ce qu'a bougé la paire précédente, et la paire en
  // cours. Inconnu après une coupe, un saut, une paire non interpolée : le warp est alors le simple.
  enum class PairMotion : std::uint8_t { Unknown, Still, Moving };
  PairMotion before_ = PairMotion::Unknown;
  PairMotion now_ = PairMotion::Unknown;
  bool poses_ = false;       // garder les poses pour la paire en cours (réglage lu à son début)
  bool single_run_ = false;  // la paire en cours n'a qu'une passe RIFE, au milieu
  // Mouvement fort (upscale_motion.cpp) : la paire précédente en avait, la paire en cours passe
  // RIFE à chaque instant sans passe au milieu (direct_run_), et la mesure de son premier instant,
  // lancée sans attente, décide de la suivante. Oublié à une coupe ou à un saut (forget_motion).
  bool strong_motion_ = false;
  bool direct_run_ = false;
  bool flow_max_async_ = true;  // faux si le backend n'a pas flow_max_start
  struct PendingMotion {
    std::uint64_t ticket = 0;  // 0 : aucune
    float scale = 1.0F;        // vers le mouvement maximal au milieu
    bool known = false;        // faux après une coupe ou un saut : relue, mais ignorée
  };
  PendingMotion motion_;
  void forget_motion();
  // La mesure en vol relue (sans attente : une paire est passée), strong_motion_ mis à jour.
  std::expected<void, Refusal> settle_motion();
  std::uint64_t single_pairs_ = 0;
  std::uint64_t multi_pairs_ = 0;
  std::uint64_t direct_pairs_ = 0;  // dont sans passe au milieu
  std::uint64_t rife_runs_ = 0;
  std::uint64_t skipped_ = 0;     // images source sautées avant calcul (passed)
  std::uint64_t late_pairs_ = 0;  // paires dont des instants sont sautés pour rattraper l'horloge
  std::uint64_t late_instants_ = 0;
  std::unique_ptr<gpu::TensorBuffer> output_;  // 1 × 3 × 2h × 2w ; absent sans SPAN
  // Warp après SPAN : deux images sources agrandies (span_of_), et l'image intermédiaire (une par
  // instant à tour de rôle, comme middles_).
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> spanned_;
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> middle_outs_;
  const gpu::Tensor& middle_out() const { return middle_outs_[middle_at_]->tensor(); }
  // Pour le journal de fin : paires vues, interpolées, coupes, dessins tenus.
  std::uint64_t pairs_ = 0;
  std::uint64_t interpolated_ = 0;
  std::uint64_t cuts_ = 0;
  std::uint64_t held_ = 0;
  static constexpr std::size_t kScoresWaitKept = 4096;
  std::vector<std::int64_t> scores_wait_us_;  // les dernières attentes hôte de scores_now, pour le profil
  std::size_t scores_wait_next_ = 0;          // case réécrite quand le tampon est plein
};

}  // namespace ajp::app::play
