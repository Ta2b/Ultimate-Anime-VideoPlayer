// GMFSS Fortuna Union dans l'étage gpu (upscale_stage.h, upscale_gmfss.cpp) : les tenseurs de ses deux
// graphes et de ses splats, l'image du milieu d'une paire forte (le mélange) et l'entrée de RIFE sur ses
// demi-paires.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/cadence.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"

namespace ajp::app::play {

// Le choix de l'étage (UpscaleSetup::gmfss) : rien, le mélange (une fusion au milieu des paires qui bougent
// fort, RIFE 4.26 sur chaque demi-paire), ou GMFSS seul (une fusion à chaque instant, sans RIFE).
enum class GmfssUse : std::uint8_t { Off, Middle, Only };

struct GmfssSetup {
  GmfssUse use = GmfssUse::Off;
  gpu::Engine* pair = nullptr;    // le graphe « paire », à rife_geometry(taille de RIFE, 64)
  gpu::Engine* fusion = nullptr;  // le graphe « fusion », mêmes plans remplis
  // Le mélange : le mouvement de la paire, en pixels à 720 lignes (mis à la hauteur de RIFE), à partir duquel
  // la fusion fait le milieu.
  float strong_px = 15.0F;
};

struct GmfssBuffers {
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
  std::uint32_t set = 0;
  std::vector<gpu::KernelOps::SplatJob> jobs;
  gpu::Tensor time;                            // l'entrée « time » de la fusion
  std::unique_ptr<gpu::TensorBuffer> out;      // sortie de la fusion, 1 × 3 × ph × pw
  std::unique_ptr<gpu::TensorBuffer> middle;   // le mélange : M à la taille de RIFE
  std::unique_ptr<gpu::TensorBuffer> work;     // M agrandie à la taille de travail, si RIFE est plus petit
  std::unique_ptr<gpu::TensorBuffer> spanned;  // M passée par SPAN, quand le warp suit SPAN
  std::unique_ptr<gpu::TensorBuffer> input;    // l'entrée de RIFE d'une demi-paire
  bool pair_ready = false;                     // la paire en cours est passée par le graphe « paire »
  bool middle_ready = false;                   // et M est faite (le mélange)
  std::uint64_t middle_serial = 0;             // numéro de M, comme une source (l'encodage de RIFE)
  std::array<std::uint64_t, 2> in_images{};
  bool in_pair = false;  // input tient la demi-paire in_images
  float in_t = 0.0F;
  std::uint64_t pairs = 0;    // passes du graphe « paire »
  std::uint64_t fusions = 0;  // passes de la fusion
  std::uint64_t halves = 0;   // passes de RIFE sur une demi-paire
  // Passes à l'arrivée (GMFSS seul, upscale_early.cpp) : la fusion de chaque instant de la paire, enfilée dès
  // outputs() dans son tampon (1 × 3 × ph × pw) au lieu d'une au slot, recadrée au slot. early_t[k] < 0 :
  // rien d'enfilé pour la sortie k.
  std::array<std::unique_ptr<gpu::TensorBuffer>, core::kMaxInstantsPerPair> early;
  std::array<float, core::kMaxInstantsPerPair> early_t{};
  std::uint64_t early_fusions = 0;
};

}  // namespace ajp::app::play
