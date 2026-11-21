// GMFSS Fortuna Union dans l'étage gpu (upscale_stage.h, upscale_gmfss.cpp) : les tenseurs de ses deux
// graphes et de ses splats, l'image du milieu d'une paire forte (le mélange) et l'entrée de RIFE sur ses
// demi-paires.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/cadence.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/gmfss.h"

namespace kirari::app::play {

// Le choix de l'étage (UpscaleSetup::gmfss) : rien, le mélange (une fusion au milieu des paires qui bougent
// fort, RIFE 4.26 sur chaque demi-paire), ou GMFSS seul (une fusion à chaque instant, sans RIFE).
enum class GmfssUse : std::uint8_t { Off, Middle, Only };

// Le mélange : au plus trois clés par paire, aux instants k / (n + 1) ; RIFE fait les sous-intervalles.
inline constexpr std::uint32_t kGmfssMaxKeys = 3;

struct GmfssSetup {
  GmfssUse use = GmfssUse::Off;
  gpu::Engine* pair = nullptr;    // le graphe « paire », à rife_geometry(taille de RIFE, 64)
  gpu::Engine* fusion = nullptr;  // le graphe « fusion », mêmes plans remplis
  // Le mélange : le mouvement de la paire, en pixels à 720 lignes (mis à la hauteur de RIFE), à partir duquel
  // la fusion fait le milieu.
  float strong_px = 15.0F;
  float floor_px = 0.0F;  // aucune clé sous ce mouvement (PlaySettings::gmfss_key_floor), à 720 lignes
  // Le mélange en travail plus grand que RIFE : M composée à la taille de travail (kernels/spec/gmfss.h),
  // au lieu de la sortie de la fusion agrandie.
  bool compose = false;
  bool direct_splats = false;
  // Le coût d'une paire GMFSS au-delà de RIFE (paire, fusion, splats), en µs de temps réel : le mélange la
  // laisse à RIFE si son premier instant serait alors en retard (upscale_late.cpp).
  std::int64_t cost_us = 0;
  std::int64_t pair_us = 0;  // la part du graphe « paire » dans cost_us
  // Le mélange : le nombre de clés au plus par paire ; la paire en pose une de plus à chaque doublement du
  // mouvement au-delà du seuil (kGmfssKeyStepPx), tant que le crédit la paie.
  std::uint32_t max_keys = kGmfssMaxKeys;
};

// Une clé du mélange : l'image GMFSS à son instant, à la taille de RIFE, de travail, et après SPAN.
struct GmfssKey {
  std::unique_ptr<gpu::TensorBuffer> middle;
  std::unique_ptr<gpu::TensorBuffer> work;
  std::unique_ptr<gpu::TensorBuffer> spanned;
  std::uint64_t serial = 0;  // comme une source (l'encodage de RIFE)
};

struct GmfssBuffers {
  struct FeatureViews {
    using Planes = std::array<gpu::Tensor, kernels::kGmfssLevels>;
    std::array<Planes, 4> banks;
    std::array<std::array<std::size_t, kernels::kGmfssLevels>, 2> outputs{};
    std::array<std::array<std::size_t, 2>, 2> bound{{{0, 1}, {2, 3}}};
    bool present = false;
    bool moved = false;
  };

  // Un splat du lot : le côté (0 depuis a, 1 depuis b), sa source, son mouvement et sa tranche de warped.
  struct Splat {
    int side = 0;
    gpu::Tensor source;
    gpu::Tensor motion;
    gpu::Tensor warped;
  };
  std::unique_ptr<gpu::TensorBuffer> frames;  // 1 × 6 × ph × pw
  // Tenseurs des graphes. Les sorties de la paire sont en double (« half0 » et « half0#1 ») : la paire
  // suivante, sur sa voie (gpu::Lane::Pair), écrit l'autre jeu pendant que les fusions lisent celui-ci ;
  // warped et time sont communs. set : le jeu de la paire en cours.
  std::vector<std::pair<std::string, std::unique_ptr<gpu::TensorBuffer>>> named;
  std::array<std::vector<gpu::Tensor>, 2> pair_outputs;  // ordre du moteur, par jeu
  std::array<std::vector<gpu::Tensor>, 2> fusion_inputs;
  std::array<std::vector<Splat>, 2> splats;
  FeatureViews features;  // quatre banques existantes, liées par côté pour éviter la copie de A
  std::uint32_t set = 0;
  std::vector<gpu::KernelOps::SplatJob> jobs;
  gpu::Tensor time;                        // l'entrée « time » de la fusion
  std::unique_ptr<gpu::TensorBuffer> out;  // sortie de la fusion, 1 × 3 × ph × pw
  // Le mélange : les clés de la paire (work si RIFE est plus petit que le travail, spanned si le warp suit
  // SPAN).
  std::array<GmfssKey, kGmfssMaxKeys> keys;
  // La composition (GmfssSetup::compose) : par côté, la source de travail et celle de RIFE splattées avec
  // leur couverture (4 plans) ; vide sans composition ou après un refus du backend.
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> high;
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> low;
  std::unique_ptr<gpu::TensorBuffer> input;  // l'entrée de RIFE d'une demi-paire
  bool pair_ready = false;                   // la paire en cours est passée par le graphe « paire »
  std::uint32_t frame_id = 0;                // l'image de la paire en cours (LiveSettings::auto_fixed)
  std::uint32_t keys_made = 0;               // les clés faites de la paire en cours (le mélange)
  std::array<std::uint64_t, 2> in_images{};
  bool in_pair = false;  // input tient la demi-paire in_images
  float in_t = 0.0F;
  std::uint64_t pairs = 0;            // passes du graphe « paire »
  std::uint64_t last_pair_right = 0;  // numéro de B à la dernière passe, zéro tant qu'elle est en avance
  std::uint64_t fusions = 0;          // passes de la fusion
  std::uint64_t halves = 0;           // passes de RIFE sur une demi-paire
  std::uint64_t composed = 0;         // M composées à la taille de travail
  // Le mélange sans le temps de GMFSS (upscale_late.cpp, plan_gaps) : la paire forte dont le premier instant
  // serait passé reste à RIFE, en quelques ms, au lieu de figer l'image derrière une paire GMFSS de 50 ms.
  bool behind = false;
  std::uint64_t behind_pairs = 0;  // paires fortes laissées à RIFE faute de temps
  // Le crédit déborde (kGmfssSpareUs au-delà d'une M) : une paire qui bouge sous le seuil strong_px a aussi
  // sa clé GMFSS. Le seuil ne fait plus que donner la priorité aux paires fortes. Décidé à l'arrivée de la
  // paire (plan_gaps), avant que sa M ne prenne sur le crédit.
  bool spare = false;
  std::uint64_t spare_keys = 0;  // clés posées sur une paire sous le seuil
  // Les clés au-delà de la première que le crédit paie pour cette paire (plan_gaps), et celles posées.
  std::uint32_t extra_keys = 0;
  std::uint64_t more_keys = 0;
  // Le crédit de temps de GMFSS : chaque image source lui donne kGmfssShare de sa durée, chaque M lui coûte
  // GmfssSetup::cost_us ; sans crédit, la paire forte reste à RIFE. Le coût moyen tient dans la cadence, quel
  // que soit le retard que mesure l'horloge (elle le voit trop tard : la file du GPU l'a déjà pris). Plafonné
  // à kGmfssCreditMaxUs, l'avance de l'anneau : les dessins tenus en remplissent assez pour les paires
  // fortes.
  std::int64_t credit_us = 0;
  std::int64_t credit_step_us = 0;  // ce que la dernière image a donné : la paire en avance s'y fie
  std::int64_t credit_pts = std::numeric_limits<std::int64_t>::min();
  // La paire suivante en avance (upscale_gmfss.cpp, run_gmfss_ahead) : dès que l'image d'après est ingérée en
  // avance, son graphe « paire » part dans l'autre jeu, derrière le travail de la paire en cours. À son
  // arrivée, la paire n'attend plus ses 12 à 35 ms : le pic tombe pendant la paire d'avant.
  bool ahead_queued = false;
  std::uint32_t ahead_frame = 0;  // SourceId::value de l'image d'après
  std::uint32_t ahead_epoch = 0;
  std::uint64_t ahead_left = 0;   // numéro de l'image de gauche (source_serial_)
  std::uint64_t ahead_pairs = 0;  // paires lancées en avance
  std::uint64_t ahead_used = 0;   // et reprises à l'arrivée de leur image
  // Une paire reprise sans fusion (calme, tenue, ou sans crédit) a coûté sa paire pour rien ; la suivante
  // attend alors une fusion faite à la demande avant de repartir en avance.
  bool ahead_adopted = false;
  std::uint64_t fusions_seen = 0;
  bool ahead_cool = false;
  std::uint64_t ahead_wasted = 0;
  std::uint64_t ahead_adopted_all_late = 0;  // avance reprise, puis tous les intermédiaires en retard
  std::uint64_t ahead_declined_expired = 0;  // avance évitée, source droite déjà derrière l'horloge
  // Passes à l'arrivée (GMFSS seul, upscale_early.cpp) : la fusion de chaque instant de la paire, enfilée dès
  // outputs() dans son tampon (1 × 3 × ph × pw) au lieu d'une au slot, recadrée au slot. early_t[k] < 0 :
  // rien d'enfilé pour la sortie k.
  std::array<std::unique_ptr<gpu::TensorBuffer>, core::kMaxInstantsPerPair> early;
  std::array<float, core::kMaxInstantsPerPair> early_t{};
  std::uint64_t early_fusions = 0;
  // Tri des paires de GMFSS seul (rife_triage.h) : le mouvement de la caméra d'une paire impossible ou d'un
  // panoramique pur (kernels/spec/motion_camera.h), écrit une fois par paire, lu par le warp à chaque
  // instant.
  std::unique_ptr<gpu::TensorBuffer> camera_flow;  // 1 × 4 × h × w, taille de travail
  std::unique_ptr<gpu::TensorBuffer> camera_mask;  // 1 × 1 × h × w
  bool camera_ready = false;                       // écrit pour la paire en cours
  bool camera_refused = false;                     // le backend n'a pas motion_camera : la fusion
  std::uint64_t camera_pairs = 0;                  // paires faites par la caméra
  std::uint64_t camera_images = 0;                 // et leurs images
};

}  // namespace kirari::app::play
