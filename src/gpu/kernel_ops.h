// Noyaux de la chaîne exécutés par un backend sur ses propres tenseurs : le contrat de
// kernels/spec.h, que la référence CPU de kernels/ref/ fixe pour chaque port.
//
// Chaque opération s'ajoute à la file du backend, dans l'ordre des appels et des enqueue() de
// moteur ; aucune n'attend côté hôte. Les tenseurs d'image sont en NCHW, lot 1 : RGB planaire
// 1 × 3 × h × w, en F16 ou F32 ; un tenseur de sortie décrit la mémoire écrite. Un format ou une taille hors
// contrat rend un refus nommé (FormatUnsupported, ShapeInvalid, FieldOutOfRange) avant toute soumission,
// jamais une conversion silencieuse.
//
// Une opération qui lit une DecodedFrame (ingest, rife_pad, measure) en garde une copie de
// frame.data jusqu'à ce que sa lecture soit achevée côté GPU, constaté sans attente hôte à un
// appel suivant, et toutes à la destruction du backend : l'appelant peut relâcher la sienne
// au retour, l'image ne revient au pool du décodeur qu'après la lecture. Une image Vulkan se
// lit entre begin_vulkan_access et end_vulkan_access (core/frame.h).
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/tensor.h"
#include "kernels/spec.h"
#include "kernels/spec/grade.h"

namespace ajp::gpu {

// Ingestion d'une image décodée : quelle conversion, selon l'étage qui la lit.
// Filtre de la réduction fusionnée à l'ingestion (extension de la tranche verticale) : boîte
// (Area), triangle (bilinéaire étiré), cubique de Mitchell-Netravali (B, C) ou Spline36 (rayon 3),
// étirés à l'échelle de la réduction, en codes de la source.
enum class ReduceKind : std::uint8_t { Area, Triangle, Cubic, Spline36 };
struct Reduction {
  ReduceKind kind = ReduceKind::Area;
  float b = 0.0F;
  float c = 0.0F;
  friend bool operator==(const Reduction&, const Reduction&) = default;
};

struct IngestSpec {
  // Chaîne d'agrandissement : kChainChromaFilter, matrice, plage et siting de la source.
  // RIFE : kRifeChromaFilter et kRifeMatrix, siting gauche (rife_cuda.py), quelle que soit
  // l'étiquette de la source.
  kernels::Filter chroma = kernels::kChainChromaFilter;
  bool rife_conversion = false;
  Reduction reduce;  // tenseur plus petit que l'image (extension de la tranche verticale)
  // Débande à l'ingestion, sans copie : seulement si debands_on_ingest(), sinon ignoré.
  bool deband = false;
};

// Mouvement estimé au milieu d'une paire, sorti du moteur RIFE (kernels/spec/warp.h).
struct Motion {
  Tensor flow;           // 1 × 4 × fh × fw : f0x, f0y, f1x, f1y, en pixels de la fenêtre
  Tensor mask;           // 1 × 1 × fh × fw : poids de a, dans [0, 1]
  kernels::Window area;  // la fenêtre de l'image dans les plans remplis
};

// Le grade exact dans l'anneau (chantier 2c, docs/adr/0009-grade-dans-l-anneau.md) d'une image écrite :
// la version d'un ordre confié au backend par set_grade, et ce qui situe l'image dans sa chaîne (le lissage
// de la zone suit l'écart de pts ; une nouvelle époque repart sans historique). version 0 : l'image n'est
// pas gradée dans l'anneau, l'écran garde le grade en shader.
struct GradeSlot {
  std::uint32_t version = 0;
  std::uint32_t epoch = 0;
  std::int64_t pts_us = kNoPts;
};

// Synchronisation d'une écriture dans le ring (protocole de core/ring.h).
struct RingWrite {
  std::uint64_t wait_value = 0;    // SlotFree::semaphore_value ; 0 au premier tour
  std::uint64_t signal_value = 0;  // RingFrame::ready_value, impaire, > wait_value
  GradeSlot grade;                 // gradée entre l'écriture et le signal (KernelOps::set_grade)
};

// Une version du grade, que KernelOps::set_grade copie : ses paramètres (kernels/spec/grade.h), la pile de
// LUT composée et l'entrée SDR → PQ de libplacebo (render::capture_grade_input). Les vues ne vivent que le
// temps de l'appel.
struct GradeOrder {
  std::uint32_t version = 0;  // > 0, unique pour le lecteur
  kernels::GradeUniforms uniforms;
  int look_size = 0;                    // 0 : aucun look (uniforms.look_mix vaut alors 0)
  std::span<const float> look_rgb;      // look_size³ × 3, R le plus rapide
  std::span<const float> input_nits;    // 3 × kernels::grade::kInputTablePoints, R, V, B bout à bout
  std::array<float, 9> input_matrix{};  // primaires de la source vers BT.2020, ligne par ligne
};

// Ordres de grade qu'un backend garde au plus : les images déjà dans l'anneau gardent le leur.
inline constexpr std::size_t kGradeOrders = 8;

// Une image intermédiaire écrite directement dans un slot du ring, au protocole de write_ring :
// ses noyaux attendent wait_value, le slot est signalé à signal_value après eux.
struct RingTarget {
  const RingSlotDesc* slot = nullptr;
  RingWrite sync;
};

// Une image YUV 4:2:0 à chroma entrelacée en mémoire d'appareil du backend : une image du pool de
// l'encodeur que write_yuv remplit (export par puits CUDA). La luma fait width × height
// échantillons, la chroma (width / 2) × (height / 2) paires CbCr ; adresses et pas en octets.
struct YuvTarget {
  std::uintptr_t luma = 0;
  std::uintptr_t chroma = 0;
  std::size_t luma_pitch = 0;
  std::size_t chroma_pitch = 0;
  int width = 0;  // paires
  int height = 0;
  kernels::Format format = kernels::Format::P010;  // Nv12 ou P010
  kernels::Matrix matrix = kernels::Matrix::Bt709;
  kernels::Range range = kernels::Range::Limited;
  kernels::Siting siting = kernels::kOutputSiting;  // celui que le flux encodé déclare
  kernels::Dither dither = kernels::Dither::Bayer8;
  // L'image finie côté GPU : le backend range done_value dans *done (release) depuis un rappel
  // hôte du flux, après les noyaux ; l'appelant le lit sans bloquer avant de confier l'image.
  // Les valeurs d'un même done croissent ; *done doit vivre jusqu'au rappel.
  std::atomic<std::uint64_t>* done = nullptr;
  std::uint64_t done_value = 0;
};

// Mesures d'une paire de sources (kernels/spec/detect.h), lues sur la luma non remplie.
struct PairScores {
  SourceId left;
  SourceId right;
  double scene_score = 0.0;       // kernels::is_scene_cut(scene_score)
  std::uint32_t worst_block = 0;  // pire bloc 32 × 32, en 1/256 de niveau 8 bits
  bool identical = false;         // luma égale au bit : somme des écarts nulle
};

// Mesures lancées sans attente (scores_start, flow_max_start) et pas encore relues, par sorte.
inline constexpr std::size_t kPendingReadbacks = 2;

// Images décodées qu'un backend garde au plus après ingest, le temps que le GPU finisse de les lire
// en place (TensorRT avec NVDEC) : le décodeur leur réserve autant de surfaces.
inline constexpr int kMaxIngestHeldFrames = 12;

class KernelOps {
 public:
  virtual ~KernelOps() = default;

  // YUV décodé (formats et ColorInfo de core/frame.h) → RGB planaire de même taille, attendu
  // derrière l'événement ready_event / ready_value de l'image (kernels::ref::yuv_to_rgb).
  virtual std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                              const Tensor& rgb) = 0;

  // Rééchantillonnage séparable à la taille de out (kernels::ref::resize). Area réduit
  // seulement : un agrandissement par Area est refusé.
  virtual std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter,
                                                const Tensor& out) = 0;

  // Entrée de RIFE : remplissage mod-64 centré en noir studio, en YUV, puis la conversion de
  // RIFE (ref::rife_pad puis ref::rife_prepare). out mesure rife_geometry(w, h) rempli.
  virtual std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) = 0;

  // Sortie de RIFE : la fenêtre de rife_geometry(w, h), w × h étant la taille de out
  // (ref::rife_crop sur RGB).
  virtual std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) = 0;

  // Warp arrière et mélange de a et b à l'instant t, 0 < t < 1 (ref::warp_blend_at) ;
  // a, b et out ont la taille de sortie, le mouvement celle de la fenêtre de RIFE.
  virtual std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion,
                                                  float t, const Tensor& out) = 0;

  // RGB → RGBA16 UNORM du slot (ref::to_rgba16), rgb ayant la taille du slot. Le slot est
  // importé au premier appel de sa génération ; une autre taille ou un autre format est refusé.
  // Le premier appel d'une génération plus récente relâche les imports des précédentes, une
  // fois leurs écritures achevées côté GPU : un import gardé retiendrait la mémoire du ring.
  virtual std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                                  const RingWrite& sync) = 0;

  // Lance la mesure de coupe de scène et de pire bloc entre deux sources de même format et de
  // même taille. Le résultat est relu sans attente, comme k_flow_max : il sert à la paire
  // suivante, par previous_scores().
  virtual std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) = 0;

  // La dernière mesure terminée, une seule fois ; rien si aucune n'a fini depuis l'appel
  // précédent. N'attend jamais côté hôte.
  virtual std::optional<PairScores> previous_scores() = 0;

  // Extensions de la tranche verticale (pc/vertical-slice), à reprendre dans le contrat à
  // l'intégration ; un backend qui ne les a pas les refuse par son nom.
  //
  // Entrée de RIFE depuis deux images RGB déjà ingérées, de même taille w × h : les 11 plans de
  // kernels/spec/rife.h (7 pour un modèle v2, selon input) dans rife_geometry(w, h), noir de studio
  // autour, instant t (0 < t < 1).
  virtual std::expected<void, Refusal> rife_input(const Tensor& /*a*/, const Tensor& /*b*/, float /*t*/,
                                                  const Tensor& /*input*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "rife_input : absent de ce backend"});
  }
  // L'instant seul de l'entrée de RIFE déjà remplie par rife_input pour la même paire (images et
  // grilles inchangées) : les autres instants d'une paire sans réécrire ses images.
  virtual std::expected<void, Refusal> rife_time(const Tensor& /*input*/, float /*t*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "rife_time : absent de ce backend"});
  }
  // rife_input d'une nouvelle paire dans une entrée déjà remplie une fois par rife_input : les
  // images et l'instant seuls, le maillage (plans 7 à 10, constants à taille donnée) reste.
  virtual std::expected<void, Refusal> rife_pair(const Tensor& /*a*/, const Tensor& /*b*/, float /*t*/,
                                                 const Tensor& /*input*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "rife_pair : absent de ce backend"});
  }
  // Mesures de la paire a, b (RGB de même taille, luma BT.709 tirée du RGB), rendues tout de suite
  // au prix d'une attente hôte, pour décider de RIFE avant de le lancer. left et right sont nuls.
  virtual std::expected<PairScores, Refusal> scores_now(const Tensor& /*a*/, const Tensor& /*b*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "scores_now : absent de ce backend"});
  }
  // scores_now en deux temps : scores_start enfile la mesure sans attendre et rend un ticket ;
  // scores_take attend la fin de cette mesure seulement, pas le travail enfilé derrière elle. Un
  // ticket vaut jusqu'à kPendingReadbacks lancements plus loin ; périmé, il est refusé.
  virtual std::expected<std::uint64_t, Refusal> scores_start(const Tensor& /*a*/, const Tensor& /*b*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "scores_start : absent de ce backend"});
  }
  virtual std::expected<PairScores, Refusal> scores_take(std::uint64_t /*ticket*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "scores_take : absent de ce backend"});
  }
  // Plus grand |déplacement| du mouvement dans sa fenêtre, en pixels de la fenêtre, rendu tout de
  // suite au prix d'une attente hôte (ref::flow_max) : décide si une passe RIFE sert toute la paire.
  virtual std::expected<float, Refusal> flow_max_now(const Motion& /*motion*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "flow_max_now : absent de ce backend"});
  }
  // flow_max_now en deux temps, comme scores_start et scores_take.
  virtual std::expected<std::uint64_t, Refusal> flow_max_start(const Motion& /*motion*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "flow_max_start : absent de ce backend"});
  }
  virtual std::expected<float, Refusal> flow_max_take(std::uint64_t /*ticket*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "flow_max_take : absent de ce backend"});
  }
  // Garder les poses (kernels/spec/poses.h) : warp_blend à l'instant réglé de chaque pixel, d'après
  // le mouvement au milieu de la paire précédente (mêmes plans, même fenêtre) ; previous nul : la
  // paire précédente était tenue. motion est celui du milieu de la paire (ref::warp_blend_poses).
  virtual std::expected<void, Refusal> warp_blend_poses(const Tensor& /*a*/, const Tensor& /*b*/,
                                                        const Motion& /*motion*/, const Motion* /*previous*/,
                                                        float /*t*/, const Tensor& /*out*/) {
    return std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, "warp_blend_poses : absent de ce backend"});
  }
  // GMFSS Fortuna (kernels/spec/gmfss.h). Entrée du graphe « paire » depuis deux images RGB déjà
  // ingérées, de même taille w × h : a puis b (kGmfssFramePlanes plans) dans rife_geometry(w, h),
  // noir de studio autour, comme les plans 0 à 5 de rife_input.
  virtual std::expected<void, Refusal> gmfss_frames(const Tensor& /*a*/, const Tensor& /*b*/,
                                                    const Tensor& /*frames*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "gmfss_frames : absent de ce backend"});
  }
  // Softmax splatting (ref::soft_splat) : source, C plans h × w, poussée à l'instant s de [0, 1]
  // par motion (kGmfssMotionChannels plans fx, fy, z de même taille, donnés pour t = 1), additionnée
  // en FP32 puis normalisée ; out a la forme et le type de source.
  virtual std::expected<void, Refusal> soft_splat(const Tensor& /*source*/, const Tensor& /*motion*/,
                                                  float /*s*/, const Tensor& /*out*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "soft_splat : absent de ce backend"});
  }
  // Les splats d'un même instant (les six de GMFSS : trois niveaux, deux côtés) en un seul
  // passage : une remise à zéro des sommes pour tous, une addition pour tous, une normalisation
  // pour tous, au lieu de trois lancements par splat qui s'attendent l'un l'autre. Chaque travail
  // suit le contrat de soft_splat ; au premier refus, les sorties sont indéfinies. Par défaut, un
  // soft_splat par travail, dans l'ordre.
  // Un travail peut lire son mouvement sur une autre grille que sa source (grid, spec MotionGrid)
  // et sortir un plan de plus que sa source, la couverture (spec) ; ces deux-là ne passent pas par
  // soft_splat, un backend sans soft_splat_all les refuse.
  struct SplatJob {
    Tensor source;
    Tensor motion;
    float s = 0.5F;
    Tensor out;
    kernels::MotionGrid grid;
  };
  virtual std::expected<void, Refusal> soft_splat_all(std::span<const SplatJob> jobs) {
    for (const SplatJob& job : jobs) {
      if (job.grid.scale != 1.0F || job.grid.x != 0.0F || job.grid.y != 0.0F ||
          job.out.shape.dims[1] != job.source.shape.dims[1]) {
        return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                       "soft_splat_all : grille et couverture absentes de ce backend"});
      }
      if (auto ok = soft_splat(job.source, job.motion, job.s, job.out); !ok) return ok;
    }
    return {};
  }
  // Déformation après SPAN (spec gmfss_compose) : high_a et high_b les sources agrandies
  // splattées (3 plans + couverture, taille de sortie), low_a et low_b les mêmes à la taille de
  // travail, fusion la sortie du graphe « fusion » (3 plans, remplie) et window sa fenêtre (la
  // taille de travail) ; out (3 plans, taille de sortie) reçoit H + F − L à l'instant t de ]0, 1[.
  virtual std::expected<void, Refusal> gmfss_compose(const Tensor& /*high_a*/, const Tensor& /*high_b*/,
                                                     const Tensor& /*low_a*/, const Tensor& /*low_b*/,
                                                     const Tensor& /*fusion*/,
                                                     const kernels::Window& /*window*/, float /*t*/,
                                                     const Tensor& /*out*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "gmfss_compose : absent de ce backend"});
  }
  // Les mêmes, l'image écrite directement dans le slot (étude 1, P4) : le code RGBA16 que write_ring
  // écrirait de out, au bit près, sans le tenseur ni sa relecture. a, b et high_a ont la taille du
  // slot. BackendCaps::ring_targets dit que le backend les a ; sinon, refusés.
  virtual std::expected<void, Refusal> warp_blend_ring(const Tensor& /*a*/, const Tensor& /*b*/,
                                                       const Motion& /*motion*/, float /*t*/,
                                                       const RingTarget& /*target*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "warp_blend_ring : absent de ce backend"});
  }
  virtual std::expected<void, Refusal> warp_blend_poses_ring(const Tensor& /*a*/, const Tensor& /*b*/,
                                                             const Motion& /*motion*/,
                                                             const Motion* /*previous*/, float /*t*/,
                                                             const RingTarget& /*target*/) {
    return std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, "warp_blend_poses_ring : absent de ce backend"});
  }
  virtual std::expected<void, Refusal> gmfss_compose_ring(const Tensor& /*high_a*/, const Tensor& /*high_b*/,
                                                          const Tensor& /*low_a*/, const Tensor& /*low_b*/,
                                                          const Tensor& /*fusion*/,
                                                          const kernels::Window& /*window*/, float /*t*/,
                                                          const RingTarget& /*target*/) {
    return std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, "gmfss_compose_ring : absent de ce backend"});
  }
  // Le tenseur RGB, de la taille de la cible, en YUV dans la cible (ref rgb_to_yuv avec le tramage
  // et le siting de target) ; les codes à kernels::tolerance::kCode de la référence. Puis *done
  // reçoit done_value. BackendCaps::yuv_targets dit que le backend l'a ; sinon, refusé.
  virtual std::expected<void, Refusal> write_yuv(const Tensor& /*rgb*/, const YuvTarget& /*target*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "write_yuv : absent de ce backend"});
  }
  // Débande sur place l'image RGB FP16 1 × 3 × h × w d'une source, à sa taille, avant SPAN : le débandage
  // sans grain que l'écran posait sinon avant le grade, sur l'image agrandie. FormatUnsupported : l'écran
  // garde le sien.
  virtual std::expected<void, Refusal> deband(const Tensor& /*rgb*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "deband : absent de ce backend"});
  }
  // Vrai si ingest() débande elle-même quand IngestSpec::deband est posé (même résultat que deband()).
  virtual bool debands_on_ingest() const { return false; }
  // Le grade exact dans l'anneau : le backend sait grader un emplacement avant son signal
  // (RingWrite::grade). set_grade copie l'ordre (gardé avec les kGradeOrders derniers, le plus ancien
  // remplacé) ; une écriture qui nomme une version inconnue est refusée.
  virtual bool grades_ring() const { return false; }
  virtual std::expected<void, Refusal> set_grade(const GradeOrder& /*order*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "set_grade : absent de ce backend"});
  }
  // Chaque élément du tenseur prend value : l'instant t du graphe « fusion » de GMFSS Union.
  virtual std::expected<void, Refusal> fill(const Tensor& /*tensor*/, float /*value*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "fill : absent de ce backend"});
  }
  // soft_splat_all(jobs) puis fill(time, t), que le backend peut fusionner (le remplissage part
  // avec la normalisation des splats). Par défaut, les deux appels dans l'ordre.
  virtual std::expected<void, Refusal> soft_splat_fill(std::span<const SplatJob> jobs, const Tensor& time,
                                                       float t) {
    if (auto ok = soft_splat_all(jobs); !ok) return ok;
    return fill(time, t);
  }
  // Mémoire de l'appareil, utilisée et totale, en octets, pour les statistiques ; appelable depuis
  // n'importe quel thread. Rien si le backend ne sait pas la lire.
  struct DeviceMemory {
    std::uint64_t used = 0;
    std::uint64_t total = 0;
  };
  virtual std::optional<DeviceMemory> device_memory() { return std::nullopt; }
  // L'analyse d'une image RGB pour le grade (kernels/spec/frame_stats.h), lancée sans attente sur la voie
  // courante ; frame_stats_take la relit quand le GPU l'a finie, sans jamais l'attendre (nullopt : pas
  // encore). kPendingReadbacks analyses en vol au plus : un ticket dépassé est périmé (SequenceOrder).
  virtual std::expected<std::uint64_t, Refusal> frame_stats_start(const Tensor& /*rgb*/,
                                                                  kernels::FrameTransfer /*transfer*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "frame_stats : absent de ce backend"});
  }
  virtual std::expected<std::optional<kernels::FrameStats>, Refusal> frame_stats_take(
      std::uint64_t /*ticket*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "frame_stats : absent de ce backend"});
  }
  // L'appareil répond-il encore (étape 11.5) ? Chaque file interrogée sans l'attendre ; DeviceLost quand le
  // contexte est perdu. RingStage le demande quand une image attend un slot depuis plus de 200 ms.
  virtual std::expected<void, Refusal> health() { return {}; }
  // Les signaux YuvTarget::done logés dans block (alignée sur une page, en multiple de pages), écrits par le
  // GPU lui-même au lieu d'un rappel hôte (étape 11.5). Faux : le backend garde le rappel. unmap_fences
  // attend les écritures en vol et rend le bloc ; à appeler avant de le libérer.
  virtual bool map_fences(std::span<std::byte> /*block*/) { return false; }
  virtual void unmap_fences(std::span<std::byte> /*block*/) {}
  // Statistiques de luminance d'une image (0 à 1, pondération BT.709 des canaux) : moyenne, centiles et parts
  // de noir et de clair. La pré-analyse des plans les mesure (media/shot_cutter.h).
  struct SceneStats {
    float mean = 0.0F;
    float p05 = 0.0F;
    float p50 = 0.0F;
    float p95 = 0.0F;
    float p995 = 0.0F;    // le pic de la scène
    float dark = 0.0F;    // part des points sous 10 % de luminance
    float bright = 0.0F;  // part des points au-dessus de 80 %
    float blue = 0.0F;    // moyenne de max(0, B - max(R, V)) : ciel, eau, aplats bleus saturés
    float chroma = 0.0F;  // moyenne de max(R, V, B) - min(R, V, B)
  };
};

}  // namespace ajp::gpu
