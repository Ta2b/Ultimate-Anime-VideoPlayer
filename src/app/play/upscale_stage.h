// L'étage gpu de la tranche verticale sur un backend chargé (gpu/backend.h). Chaque image
// décodée est ingérée en RGB FP16 à la taille de travail (réduction Area fusionnée si elle est
// plus petite que la source). Avec RIFE, la paire (image précédente, image) donne les instants de
// sa grille de sortie (core/cadence.h : t = k/N pour ×N ; ×1,5 alterne t = 2/3, puis t = 1/3 et
// l'image), l'image elle-même quand elle est sur la grille (rife_plan.h décide si RIFE tourne).
// Chaque sortie est agrandie par SPAN (×2) et écrite dans un slot du ring que render présente.
// Sans moteur SPAN, la sortie passe à la taille de travail.
//
// DistilDRBA (kernels/spec/distil_drba.h) aussi, à la taille de travail : une passe de son graphe
// par image intermédiaire, sur trois images ; il lit l'image d'après la paire, une image de retard.
//
// Ordre de la chaîne : RIFE estime le mouvement à la taille de travail. Le warp et le mélange se font avant
// SPAN (SPAN nettoie ensuite leurs petits défauts), ou après lui (warp_after_span : un SPAN par source au
// lieu d'un par image créée) ; l'ordre Auto met le warp après SPAN au-dessus d'une sortie de 1080 lignes ou
// dès ×3 (settings.h, warp_after_span).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "app/play/free_engines.h"
#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/rife_plan.h"
#include "app/play/ring_stage.h"
#include "core/cadence.h"
#include "core/free_grid.h"
#include "core/moving_area.h"
#include "core/ring.h"
#include "core/steady_grid.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/distil_drba.h"
#include "kernels/spec/frame_stats.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

namespace ajp::app::play {

// Les slots de l'export par puits CUDA (export_sink.h) : chaque slot est une image de l'encodeur,
// que write_slot remplit en YUV (KernelOps::write_yuv) au lieu du ring ; done_value est la valeur
// qu'aurait signalée le slot du ring (RingFrame::ready_value). Appelé depuis le thread gpu.
class YuvSlots {
 public:
  virtual ~YuvSlots() = default;
  virtual gpu::YuvTarget target(std::uint32_t slot, std::uint64_t done_value) const = 0;
  // Le bloc qui loge les signaux done des cibles, aligné sur une page et en multiple de pages, pour que le
  // backend y écrive depuis le GPU (KernelOps::map_fences) ; vide : rappels hôtes.
  virtual std::span<std::byte> fence_block() const { return {}; }
};

// Échelle par paire (PlaySettings::pair_scale) : le côté où le second RIFE estime les paires à mouvement
// fort, la moitié de celui de RIFE, pair (1080 → 540, 1280 → 640).
constexpr int pair_scale_side(int rife_side) {
  return std::max(2, rife_side / 4 * 2);
}

struct UpscaleSetup {
  gpu::Backend* backend = nullptr;  // vit plus longtemps que l'étage
  gpu::Engine* engine = nullptr;    // SPAN aux formes de travail ; nullptr : sans agrandissement
  // Lecture (horloge présente) : un instant identique à l'image affichée n'est ni calculé ni écrit.
  bool skip_held = false;
  gpu::Engine* rife = nullptr;  // RIFE à rife_geometry(travail) ; nullptr : sans interpolation
  // Variante mouvement de RIFE (motion_model.h) : flot et masque, warp à la sortie. Remplace rife ;
  // une passe au milieu de la paire la sert toute quand le mouvement est faible.
  gpu::Engine* rife_motion = nullptr;
  // P6.17, essai : un RIFE mouvement plus léger, aux mêmes formes, pour la passe du milieu quand la paire
  // précédente était calme ; nullptr : rife_motion partout.
  gpu::Engine* rife_motion_calm = nullptr;
  // Échelle par paire (upscale_half.cpp) : rife_motion à pair_scale_side de rife_width × rife_height, pour
  // les passes des paires à mouvement fort ; nullptr : rife_motion partout.
  gpu::Engine* rife_motion_half = nullptr;
  int rife_planes = 11;  // plans d'entrée de rife et rife_motion : 11 (v1) ou 7 (v2)
  int rife_align = 64;   // multiple de remplissage du modèle RIFE : 64, ou 128 pour 4.25 lite
  // DistilDRBA à la place de RIFE, aux plans remplis de rife_geometry(travail).
  gpu::Engine* distil_drba = nullptr;
  int drba_align = 64;  // multiple des plans remplis de DistilDRBA : 128 au demi-flot
  // distil_drba rend le flot (1 × 4) et le masque (1 × 1) du warp final au lieu de l'image (variante
  // « mouvement », drba_model.h, ou le fusionné à deux sorties) : le warp et le mélange se font ici, après
  // SPAN quand warp_after_span (un SPAN par source au lieu d'un par image créée).
  bool drba_motion = false;
  Factor rife_factor{.num = 1,
                     .den = 1};  // ×num/den quand l'interpolation tourne (au plus ×10, limits::kRifeFactor)
  // --one-lane : tout sur la voie Upscale, sans recouvrement. Plus lent, mais le temps GPU de
  // chaque moteur redevient son coût propre au lieu d'inclure celui des autres voies.
  bool one_lane = false;
  // P6.15, essai : la passe du milieu d'une paire calme reprend les blocs grossiers de la passe du milieu de
  // la paire précédente, sur le même moteur (gpu::Engine::next_warm).
  bool rife_warm = false;
  // --rife-encode-all, mesure : aucune annonce au moteur de ce que sa passe précédente a encodé
  // (enqueue_rife) ; le RIFE fusionné encode alors les deux images à chaque passe.
  bool rife_encode_all = false;
  // Réglages changés en cours de lecture (facteur RIFE, réduction, débandage) et compteurs de RIFE ;
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
  // Variante mouvement de RIFE (ou de DistilDRBA) avec SPAN : SPAN une fois par image source, le warp à
  // la taille de sortie sur les images agrandies (kernels/spec/warp.h : a, b et la sortie à une taille, le
  // mouvement mis à l'échelle). Sans effet pour les autres estimateurs ou sans SPAN.
  bool warp_after_span = false;
  // Rapide à pas constant (chantier 7.6) : chaque instant est comparé à l'horloge prédite juste avant son
  // calcul ; passé, il n'est ni calculé ni écrit (Fate::Late). Une source en retard garde son SPAN quand la
  // paire suivante le lit. Sans copie de b (poses gardées) ; DistilDRBA compris (A09-07).
  bool late_fate = false;
  // « Lisser les dessins tenus » (play.toml smooth_drawings, --smooth-drawings, chantier 7.9) : un dessin
  // tenu de deux à kSmoothMaxHold sources, suivi d'un dessin nouveau déjà lu, se fond en un seul mouvement
  // (rife_plan.h, smooth_hold_u) au lieu de rester figé puis de sauter. RIFE mouvement seulement, sans poses
  // gardées ; l'étage lit alors jusqu'à kSmoothMaxHold - 1 images en avance (frames_ahead). Une tenue plus
  // longue, ou coupée, garde le rythme des sources.
  bool smooth_drawings = false;
  // Le filtre qui lit les images au warp de la variante mouvement de RIFE (PlaySettings::cubic_warp).
  kernels::RifeWarp warp = kernels::RifeWarp::Bilinear;
  std::vector<RingSlotDesc> slots;  // ring de render, à la taille de sortie de SPAN
  // Export par puits CUDA : les images de l'encodeur derrière les slots (sans image ni sémaphore
  // dans slots), écrites par write_yuv ; ni écriture directe du ring. nullptr : le ring.
  std::shared_ptr<const YuvSlots> yuv_slots;
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
  std::expected<void, Refusal> health() override { return setup_.backend->kernels()->health(); }
  bool passed(const DecodedFrame& frame, const DecodedFrame& next) override;
  Fate fate(const DecodedFrame& frame, std::uint32_t output) override;
  std::int64_t planned_gap_us(std::uint32_t output) const override;
  void skip(const DecodedFrame& frame, std::uint32_t output) override;
  std::expected<void, Refusal> late(const DecodedFrame& frame, std::uint32_t output) override;
  Fate drba_fate(std::uint32_t output);
  // DistilDRBA, instant écrit pour b tenu : c'est l'image source elle-même (left == right), à l'écran
  // ensuite.
  Output held_b(Output made, const DecodedFrame& frame, std::uint32_t output);
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
  // Une paire prise d'emblée à chaque instant (direct_run_) : RIFE à t, sa mesure du premier instant relue à
  // la paire suivante, puis le warp.
  std::expected<void, Refusal> interpolate_direct(float t, const gpu::Motion& other, const gpu::Tensor& a,
                                                  const gpu::Tensor& b, const Target& out);
  // warp_blend et warp_blend_poses vers out (KernelOps, ou leurs variantes _ring).
  std::expected<void, Refusal> blend(const gpu::Tensor& a, const gpu::Tensor& b, const gpu::Motion& motion,
                                     float t, const Target& out) const;
  std::expected<void, Refusal> blend_poses(const gpu::Tensor& a, const gpu::Tensor& b,
                                           const gpu::Motion& motion, const gpu::Motion* previous, float t,
                                           const Target& out) const;
  // Le warp et la composition écrivent l'image intermédiaire directement dans le slot
  // (BackendCaps::ring_targets, étude 1) : après SPAN, ou sans SPAN par la variante mouvement.
  // middle_outs_ et middles_ ne sont alors pas alloués.
  bool ring_direct_ = false;
  // RIFE image à l'instant t, recadrée dans middle().
  std::expected<void, Refusal> interpolate_image(float t);
  // La variante mouvement de RIFE à l'instant t, dans flow et mask.
  // calm : la passe peut prendre rife_motion_calm (P6.17).
  std::expected<void, Refusal> run_motion(float t, const gpu::Tensor& flow, const gpu::Tensor& mask,
                                          bool calm = false);
  bool interpolates() const { return setup_.rife != nullptr || setup_.rife_motion != nullptr || drba(); }
  bool drba() const { return setup_.distil_drba != nullptr; }
  bool drba_motion() const { return drba() && setup_.drba_motion; }
  // DistilDRBA (upscale_drba.cpp) : ses tenseurs ; l'arrivée de l'image b à la place de son
  // ingestion (rgb_[current_] reçoit b, l'image d'avant la paire passe dans drba_.before) ; les
  // mesures de la paire (reprises de l'arrivée précédente si elle les a prises pour sa suivante) ;
  // la suivante ingérée dans drba_.after et mesurée contre b ; puis l'image intermédiaire
  // output + 1 sur outputs_ dans middle().
  std::expected<void, Refusal> allocate_drba();
  std::expected<void, Refusal> drba_arrive(const DecodedFrame& frame, const gpu::IngestSpec& spec);
  // L'image sautée, gardée comme image d'avant (D3).
  void keep_skipped(const DecodedFrame& frame);
  // La paire mesurée entre dans les compteurs (paires, coupes, dessins tenus) ; une coupe oublie le
  // mouvement.
  void count_measured(const PairFacts& facts);
  // Débogage des paires (pair_marks.h) : la paire (previous_, frame) tenue ou coupée notée et écrite au
  // journal, seulement quand le menu Statistiques le demande.
  void mark_pair(const DecodedFrame& frame, const PairFacts& facts);
  // Où DistilDRBA estime : la taille de RIFE.
  kernels::RifeGeometry drba_geometry() const;
  std::expected<gpu::PairScores, Refusal> drba_scores(const DecodedFrame& frame);
  std::expected<void, Refusal> drba_next(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                         bool measure);
  // timed : l'attente hôte de la relecture va au profil de fin (keep_scores_wait).
  std::expected<void, Refusal> settle_after_scores(bool timed = true);
  void keep_scores_wait(std::int64_t waited_us);
  // pts(b) − pts(a), ou 0 si l'une des deux images n'a pas de date.
  static constexpr std::int64_t pts_gap_us(std::int64_t a_us, std::int64_t b_us) {
    return a_us != kNoPts && b_us != kNoPts ? b_us - a_us : 0;
  }
  std::expected<void, Refusal> interpolate_drba(std::uint32_t output);
  // Variante mouvement : la passe rend flot et masque, puis le warp de a et b (agrandies si spanned, de
  // travail sinon) dans out.
  std::expected<void, Refusal> interpolate_drba_motion(std::uint32_t output, bool spanned, const Target& out);
  // Les entrées de la passe pour cet appel (rife_input, ou rife_time quand les images y sont), puis la passe.
  std::expected<void, Refusal> run_drba(const kernels::DrbaCall& call, std::span<gpu::Tensor> results);
  // Copie d'une image de la taille de out (côté tenu) : l'écriture du ring, ou une copie de tenseur.
  std::expected<void, Refusal> copy_out(const gpu::Tensor& image, const Target& out) const;
  // L'instant se rend sans passe : le côté de la paire qu'il regarde est un dessin tenu.
  bool drba_side_held(const kernels::DrbaCall& call) const;
  // Une coupe entre deux images mesurées hors de la paire courante (b et after, le pont, l'avance du Smooth)
  // : le jugement de pair_cut, plans de la pré-analyse et score rapporté à la zone hors bandes noires.
  bool cut_measured(const gpu::PairScores& scores, std::int64_t left_pts_us, std::int64_t right_pts_us) const;
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
    bool debanded = false;  // l'image ingérée en avance, débandée
  };
  Ahead ahead_;
  bool ahead_scores_ = true;  // faux si le backend n'a pas scores_start
  // L'analyse des images sources pour le grade (frame_stats_start), relue sans attente vers
  // LiveSettings::scenes ; faux si le backend ne la sait pas.
  bool frame_stats_ = true;
  double useful_area_ = 1.0;  // P3.32 : la part hors bandes noires de la dernière image analysée
  struct PendingStats {
    std::uint64_t ticket = 0;
    std::int64_t pts_us = 0;
    kernels::FrameTransfer transfer = kernels::FrameTransfer::Sdr;  // le limiteur de flashs (C42-04)
  };
  std::vector<PendingStats> stats_pending_;
  std::uint64_t stats_dropped_ = 0;  // relectures abandonnées (A42-18), journalisées une fois
  void analyze_source(const DecodedFrame& frame);
  bool looks_ahead() const { return interpolates() && !drba(); }
  // Retard mesuré des images de sortie (FrameLatency) ; 0 sans mesure.
  std::int64_t latency_us() const { return setup_.live ? setup_.live->latency().us() : 0; }
  std::expected<void, Refusal> allocate_ahead();
  // L'image déjà ingérée en avance passe dans rgb_[current_] (et small_) ; faux sinon.
  bool take_ahead(const DecodedFrame& frame, const gpu::IngestSpec& spec);
  // Les mesures de la paire (image précédente, image en cours) : lancées en avance si elles l'ont été.
  std::expected<gpu::PairScores, Refusal> pair_scores();
  // La suivante de frame, si elle attend, ingérée et mesurée contre frame. La mesure de la paire n'est lancée
  // que si elle peut servir (next_pair_measured, ou always_scores pour les dessins lissés) ; pair_scores la
  // fait sinon à la demande.
  std::expected<void, Refusal> look_ahead(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                          bool always_scores = false);
  // L'interpolation au repos (×1 sans Rapide, fenêtre cachée) ne lit pas la mesure de la paire suivante
  // (B09-04).
  bool next_pair_measured() const;
  // RIFE mouvement va d'emblée à l'instant t, sans passe au milieu : sans « garder les poses », quand la
  // paire précédente bougeait fort (plusieurs instants), ou quand la paire n'a qu'un instant, hors du milieu
  // (B09-01). La même règle prépare l'entrée de RIFE (prepare_rife_input) et décide la paire.
  bool motion_direct(float t) const;
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
  // La passe de RIFE sur engine depuis rife_in(), après avoir dit au moteur ce que sa passe précédente a déjà
  // encodé (le RIFE fusionné le garde) : les mêmes images dans le même tampon, à un autre instant
  // (next_times_only), ou la paire qui suit, A étant la B de sa passe précédente (next_pair_follows).
  std::expected<void, Refusal> enqueue_rife(gpu::Engine& engine, std::span<gpu::Tensor> results);
  struct RifeEncoded;
  // enqueue_rife depuis input, dont le tampon rife_pair_ tient images ; last : la passe précédente de ce jeu
  // d'entrées, mise à jour.
  std::expected<void, Refusal> enqueue_on(gpu::Engine& engine, const gpu::Tensor& input,
                                          const std::array<std::uint64_t, 2>& images, RifeEncoded& last,
                                          std::span<gpu::Tensor> results);
  // Numéro de l'image ingérée dans rgb_[i] et small_[i], neuf à chaque ingestion.
  std::array<std::uint64_t, 2> source_serial_{};
  // L'autre tampon de rgb_ et small_ devient current_, pour une image neuve : son numéro aussi.
  void next_source() {
    current_ ^= 1U;
    source_serial_[current_] = ++serials_;
  }
  std::uint64_t serials_ = 0;
  // Les images de chaque tampon de rife_ins_ par leur numéro, A puis B ; 0 pour une image qui n'est pas une
  // source telle qu'ingérée (B lissée) ou une écriture en échec.
  std::array<std::array<std::uint64_t, 2>, 2> rife_in_images_{};
  struct RifeEncoded {
    const gpu::Engine* engine = nullptr;  // nul : rien de sûr (début, coupe, saut, passe en échec)
    std::uint32_t buffer = 0;
    std::array<std::uint64_t, 2> images{};
  };
  RifeEncoded rife_encoded_;             // la dernière passe de RIFE menée au bout
  std::uint64_t encodes_kept_ = 0;       // passes annoncées sans encodeur (même paire)
  std::uint64_t encodes_followed_ = 0;   // passes annoncées avec la seule B à encoder
  std::uint64_t encodes_after_gap_ = 0;  // A et B encodées : la paire d'avant n'a pas eu de passe
  std::uint64_t encodes_cold_ = 0;       // A et B encodées : début, coupe, saut, autre moteur, B lissée
  bool rife_in_pair_ = false;            // rife_in() tient les images de la paire en cours
  float rife_in_t_ = 0.0F;               // et cet instant
  bool rife_time_ = true;                // faux si le backend n'a pas rife_time
  bool rife_pair_op_ = true;             // faux si le backend n'a pas rife_pair
  std::array<bool, 2> rife_mesh_{};      // rife_ins_[i] tient le maillage, écrit par un rife_input
  // Le mouvement est estimé à la taille de RIFE puis déformé au warp (variante mouvement), plutôt
  // que l'image de RIFE elle-même.
  bool by_motion() const { return setup_.rife_motion != nullptr; }
  // L'image où RIFE et les mesures de la paire travaillent : la réduite s'il y en a une.
  const gpu::Tensor& rife_source(std::uint32_t index) const {
    return small_[index] ? small_[index]->tensor() : rgb_[index]->tensor();
  }
  // Warp après SPAN : les images sources sont agrandies à l'ingestion, dans spanned_.
  bool warps_spanned() const {
    return setup_.warp_after_span && (by_motion() || drba_motion()) && setup_.engine != nullptr;
  }
  // SPAN de l'image en cours dans son tampon de spanned_, une fois (aucune passe sur un dessin tenu :
  // celui de la précédente).
  std::expected<void, Refusal> span_current();
  // Le tampon de SPAN de l'image arrivée, dès les mesures de sa paire (reuse_span_ connu), et non à son
  // premier SPAN : le warp de la paire prend spanned(current_) avant que span_current enfile SPAN.
  void pick_span_buffer();
  // La paire reprend le SPAN de la précédente (reuse_span_) : dessin tenu, ou copie d'une tenue fondue.
  bool reuses_span(bool held) const;
  const gpu::Tensor& spanned(std::uint32_t index) const { return spanned_[span_of_[index]]->tensor(); }
  bool spanned_now_ = false;
  std::array<std::uint32_t, 2> span_of_{0, 1};  // tampon de spanned_ de chaque image de rgb_
  // Dessin tenu : la sortie de SPAN de la précédente sert (output_ si output_has_source_, ou son
  // tampon de spanned_).
  static constexpr int kMaxSpanReuse = 2;
  bool reuse_span_ = false;
  bool copy_pair_ = false;  // RIFE avec poses après un dessin tenu : pas de passe, b seule est écrite
  bool b_shown_ = false;    // DistilDRBA : b est écrite dans cette paire (côté b tenu)
  // La sortie qui écrit b (b_shown_, copy_written_) : fate() se redemande quand la sortie attend un slot, et
  // doit alors rendre Write de nouveau, pas Same.
  std::uint32_t b_output_ = 0;
  bool held_same_ = false;          // source d'un dessin tenu, déjà à l'écran : Same
  bool shown_source_ = false;       // la dernière image écrite est une source
  bool copy_written_ = false;       // b est écrite dans cette paire
  bool output_has_source_ = false;  // output_ tient SPAN de l'image précédente
  int spans_reused_in_row_ = 0;
  std::uint64_t spans_reused_ = 0;
  // SPAN (s'il est là) puis l'écriture dans le slot.
  std::expected<void, Refusal> finish(const gpu::Tensor& rgb, std::uint32_t slot, const gpu::RingWrite& sync);
  // L'écriture d'une image à la taille du ring.
  std::expected<void, Refusal> write_slot(const gpu::Tensor& image, std::uint32_t slot,
                                          const gpu::RingWrite& sync);

  UpscaleSetup setup_;
  // Voie du backend (gpu/lanes.h) des travaux qui suivent : ingestion et mesures sur Measure, SPAN
  // et écriture du ring sur Upscale (les écritures du ring signalent dans l'ordre sur une seule
  // file), RIFE, DistilDRBA et warp sur Interpolate. Le backend ordonne les voies d'après
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
    std::int64_t gap_us = 0;  // Rapide libre : l'écart planifié depuis l'instant d'avant (RingFrame)
  };
  // Rapide libre : le pts du dernier instant planifié, écrit, sauté en retard ou tenu ; kNoPts hors du mode,
  // au début d'une époque.
  std::int64_t planned_pts_ = kNoPts;
  // Une paire sans passe : les grilles repartent de sa source droite.
  void resync_grids(bool unrun, bool free_mode, bool steady);
  // L'écart planifié de chaque sortie de la paire (instants_), les instants sautés en retard compris ;
  // `step` : la grille de la paire quand elle est interpolée, nullptr pour la source seule. `on` : Rapide
  // libre dans la même époque, sinon rien n'est planifié.
  void plan_gaps(const DecodedFrame& frame, const core::CadenceStep* step, std::size_t late, bool on);
  std::array<Instant, core::kMaxInstantsPerPair> instants_{};
  std::uint32_t outputs_ = 1;
  std::uint32_t intermediates_ = 0;  // celles qui sont interpolées
  bool uninterpolated_ = false;      // la paire courante, laissée sans interpolation à dessein
  // Débandage avant SPAN (KernelOps::deband) : voulu quand le grade tourne, avec SPAN, DistilDRBA compris
  // (A09-07) ; deband_op_ tombe si le backend ne l'a pas, et l'écran garde alors le sien.
  bool deband_op_ = true;
  bool debanded_ = false;  // l'image en cours a été débandée
  bool wants_deband() const {
    return deband_op_ && setup_.engine != nullptr && setup_.live && setup_.live->deband_source();
  }
  // Débande l'image ingérée dans rgb si wants_deband() ; vrai si elle l'est.
  std::expected<bool, Refusal> deband_source(const gpu::Tensor& rgb);
  // ingest() de la source dans rgb puis son débandage (fusionné si le backend sait) ; vrai si débandée.
  std::expected<bool, Refusal> ingest_source(const DecodedFrame& frame, gpu::IngestSpec spec,
                                             const gpu::Tensor& rgb);
  std::optional<core::CadenceAccumulator> cadence_;
  core::SteadyGrid steady_grid_;  // Rapide à pas constant (--fast-steady) : N images par source
  core::FreeGrid free_grid_;      // mode Rapide : la grille des instants à la place de cadence_
  core::FreeInputs free_inputs(std::uint32_t epoch) const;
  MotionEngine engine_kind() const { return drba() ? MotionEngine::DistilDrba : MotionEngine::Rife; }
  Factor cadence_factor_;
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
  // Échelle par paire (upscale_half.cpp, PlaySettings::pair_scale) : les tampons du second RIFE, à
  // pair_scale_side de la taille de RIFE. Les réduites de a et b suivent les tampons de rgb_ par leur numéro
  // (source_serial_), l'entrée suit rife_pair_ comme rife_ins_.
  struct HalfRife {
    std::array<std::unique_ptr<gpu::TensorBuffer>, 2> sources;
    std::array<std::uint64_t, 2> serials{};     // la source réduite dans chaque tampon ; 0 : aucune
    std::unique_ptr<gpu::TensorBuffer> smooth;  // B lissée, réduite à chaque passe
    std::array<std::unique_ptr<gpu::TensorBuffer>, 2> ins;
    std::array<std::array<std::uint64_t, 2>, 2> in_images{};
    std::unique_ptr<gpu::TensorBuffer> flow;
    std::unique_ptr<gpu::TensorBuffer> mask;
    bool in_pair = false;  // ins[rife_pair_] tient la paire en cours
    float in_t = 0.0F;     // à cet instant
    RifeEncoded encoded;   // la dernière passe du second RIFE menée au bout
    std::uint64_t runs = 0;
  };
  HalfRife half_;
  bool halves() const { return setup_.rife_motion_half != nullptr && by_motion(); }
  std::expected<void, Refusal> allocate_half();
  gpu::Motion half_motion() const;
  // La source du tampon index de rgb_ (ou B lissée, smooth) réduite pour le second RIFE.
  std::expected<const gpu::Tensor*, Refusal> half_source(std::uint32_t index, bool smooth);
  // L'entrée du second RIFE à l'instant t pour la paire en cours, comme fill_rife_input.
  std::expected<void, Refusal> fill_half_input(float t);
  // Une passe de RIFE mouvement à l'instant t hors du milieu mesuré : sur le second RIFE quand half et
  // halves(), sinon dans flow_ et mask_ (full). Le mouvement rendu, et le facteur qui met son mouvement
  // maximal en pixels de RIFE (flow_max).
  struct StrongRun {
    gpu::Motion motion;
    float scale = 1.0F;
  };
  std::expected<StrongRun, Refusal> run_strong(float t, const gpu::Motion& full, bool half);
  // Variante mouvement, mouvement fort : celui d'un autre instant que le milieu.
  std::unique_ptr<gpu::TensorBuffer> flow_;
  std::unique_ptr<gpu::TensorBuffer> mask_;
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
    bool after_held = false;  // (b, after) est un dessin tenu : les instants après ½ sont b
    bool continuous = false;  // la paire en cours suit la précédente sans coupe, mesures prises
    std::optional<SourceId> after_id;
    bool after_debanded = false;  // l'image d'après, ingérée en avance, a été débandée (A09-07)
    std::uint32_t after_epoch = 0;
    gpu::Reduction
        after_reduce;  // la réduction de son ingestion : changée en direct, elle se refait (C09-07)
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
  // Mode Rapide : la paire préparée est annoncée en mouvement fort (strong_motion_ relu avant sa grille), son
  // écart se planifie sur le temps de service lourd (core/free_rate.h).
  bool heavy_pair_ = false;
  bool direct_run_ = false;
  bool flow_max_async_ = true;  // faux si le backend n'a pas flow_max_start
  struct PendingMotion {
    std::uint64_t ticket = 0;  // 0 : aucune
    float scale = 1.0F;        // vers le mouvement maximal au milieu
    bool known = false;        // faux après une coupe ou un saut : relue, mais ignorée
  };
  PendingMotion motion_;
  // La paire en cours se décide sur la mesure de la précédente (NL3 : aucune attente de l'hôte) ; faux au
  // début d'un plan, après une coupe ou un saut : la mesure du milieu est alors attendue, comme avant.
  bool motion_measured_ = false;
  void forget_motion();
  // La mesure en vol relue (sans attente : une paire est passée), strong_motion_ et motion_measured_ mis à
  // jour.
  std::expected<void, Refusal> settle_motion();
  // Avec les poses, la première paire d'un plan (motion_measured_ faux) attend la mesure de son milieu pour
  // strong_motion_ (C09-04).
  std::expected<void, Refusal> measure_first_pair(const gpu::Motion& middle, float t);
  std::uint64_t single_pairs_ = 0;
  std::uint64_t multi_pairs_ = 0;
  std::uint64_t direct_pairs_ = 0;  // dont sans passe au milieu
  std::uint64_t rife_runs_ = 0;
  std::uint64_t calm_runs_ = 0;  // passes faites par rife_motion_calm (P6.17)
  std::uint64_t warm_runs_ = 0;  // passes du milieu reprises à chaud (P6.15)
  // La dernière passe est celle du milieu, sur ce moteur : la suivante peut la reprendre (P6.15).
  const gpu::Engine* warm_from_ = nullptr;
  core::MovingTally moving_;  // la zone qui bouge des paires mesurées (P6.6, P6.18)
  // Dessin tenu (ni coupe, ni mesure manquante) ; sinon la zone qui bouge de la paire entre au cumul.
  bool held_pair(const PairFacts& facts) {
    if (!facts.scores || pair_cut(facts)) return false;
    if (kernels::is_held(facts.scores->worst_block)) return true;
    moving_.add(facts.scores->moving);
    return false;
  }
  // La paire précédente était calme et mesurée (P6.17, P6.15).
  bool calm_pair() const { return motion_measured_ && !strong_motion_; }
  std::uint64_t skipped_ = 0;     // images source sautées avant calcul (passed)
  bool fences_mapped_ = false;    // les signaux de l'export écrits par le GPU (KernelOps::map_fences)
  std::uint64_t late_pairs_ = 0;  // paires dont des instants sont sautés pour rattraper l'horloge
  std::uint64_t late_instants_ = 0;
  // Fate::Late (UpscaleSetup::late_fate) : vrai pour la paire en cours ; les instants laissés, et ceux
  // calculés quand même après leur instant (le critère du chantier 7.6 : 0).
  bool late_fate_now_ = false;
  std::uint64_t late_dropped_ = 0;
  std::uint64_t late_computed_ = 0;
  // Les images intermédiaires de la paire déjà calculées, et le rang de celle en cours : la première prépare
  // la paire (mouvement), même quand des instants de tête sont laissés en retard.
  std::uint32_t middles_done_ = 0;
  std::uint32_t middle_rank_ = 0;
  bool first_middle() const { return middle_rank_ == 0; }
  // Smooth (upscale_smooth.cpp) : une tenue de smooth_hold_ sources (rife_plan.h, smooth_hold_u) fondue en un
  // mouvement vers b (smooth_b_id_). Paires de tête (smooth_first_) : la paire va de la copie tenue vers b,
  // lue en avance (ahead_ pour une tenue de deux sources, smooth_far_ au-delà) ; dernière paire
  // (smooth_second_) : la paire elle-même, ses instants pris au rang smooth_rank_.
  bool smooth_possible() const { return setup_.smooth_drawings && by_motion() && !drba() && interpolates(); }
  std::size_t frames_ahead() const override {
    return drba() ? 1 : smooth_possible() ? kSmoothMaxHold - 1 : 0;
  }
  std::expected<void, Refusal> allocate_smooth();
  // La tenue en cours passe à la paire suivante, ou s'arrête (époque, coupe, grille) ; avant smooth_decide.
  void smooth_advance(const DecodedFrame& frame, const PairFacts& facts, bool on_grid);
  std::expected<void, Refusal> smooth_decide(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                             const PairFacts& facts, bool on_grid);
  // La tenue dure plus de deux sources : b cherché derrière la suivante, chaque image mesurée contre la copie
  // en cours ; sa longueur (3 à kSmoothMaxHold), ou 0 : coupe, image absente, tenue plus longue.
  std::expected<std::uint32_t, Refusal> smooth_find_b(const DecodedFrame& frame, const gpu::IngestSpec& spec);
  void smooth_stop();
  // Warp après SPAN : SPAN de b (l'image en avance) dans le tampon libre, une fois par tenue fondue.
  std::expected<void, Refusal> span_smooth_b();
  const gpu::Tensor& smooth_b_rgb() const {
    return smooth_far_ ? smooth_far_rgb_->tensor() : ahead_.rgb->tensor();
  }
  const gpu::Tensor& smooth_b_rife() const {
    if (smooth_far_) return smooth_far_small_ ? smooth_far_small_->tensor() : smooth_far_rgb_->tensor();
    return ahead_.small ? ahead_.small->tensor() : ahead_.rgb->tensor();
  }
  bool smooth_first_ = false;
  bool smooth_second_ = false;
  std::uint32_t smooth_rank_ = 0;  // la paire en cours dans la tenue : 0 à smooth_hold_ - 1
  std::uint32_t smooth_hold_ = 0;  // les sources de la tenue fondue ; 0 : aucune
  bool smooth_far_ = false;        // b dans smooth_far_rgb_ (tenue de trois sources ou plus)
  std::unique_ptr<gpu::TensorBuffer> smooth_far_rgb_;    // b d'une longue tenue, à la taille de travail
  std::unique_ptr<gpu::TensorBuffer> smooth_far_small_;  // et à la taille de RIFE, si small_
  SourceId smooth_b_id_;
  SourceId frame_id_;                                                // l'image en tête (outputs)
  std::optional<std::pair<SourceId, gpu::PairScores>> ahead_taken_;  // (gauche, mesures de la suivante)
  std::optional<SourceId> span_ahead_;  // SPAN de cette image déjà fait (span_smooth_b), dans span_ahead_buf_
  std::uint32_t span_ahead_buf_ = 0;
  bool smooth_spanned_ = false;  // span_smooth_b fait pour la tenue en cours
  std::array<std::uint64_t, kSmoothMaxHold + 1> smoothed_holds_{};  // tenues fondues, par longueur
  // L'instant de la sortie serait passé quand une image soumise maintenant paraîtrait.
  bool instant_late(const DecodedFrame& frame, std::uint32_t output) const;
  // Annonce la paire en cours au compte du mode auto ou du Rapide (`free`), avec ce qu'elle vaut si elle
  // n'a qu'une image (LonePair, A12-12). `runs` : l'interpolation tourne sur cette paire.
  void expect_pair(std::uint32_t id, int written, std::int64_t lead_us, bool free, bool steady, bool runs);
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
  std::uint64_t cut_disagreements_ = 0;
  std::uint64_t held_ = 0;
  bool drba_held_ =
      false;  // l'instant en cours de production est une copie d'un côté tenu (Output::repeated)
  static constexpr std::size_t kScoresWaitKept = 4096;
  std::vector<std::int64_t> scores_wait_us_;  // les dernières attentes hôte de scores_now, pour le profil
  std::size_t scores_wait_next_ = 0;          // case réécrite quand le tampon est plein
};

}  // namespace ajp::app::play
