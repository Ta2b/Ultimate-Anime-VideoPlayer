// Tri des paires de RIFE par la carte de la scène (kernels/spec/scene_flow.h, --pair-triage, plan du flux
// NVIDIA, phase 3). Le flux optique matériel ne sert qu'à décider, jamais à déformer : aucun modèle nouveau.
//   - Dessin tenu (Held) : la carte dit que rien ne bouge, ni le dessin ni la caméra, et qu'aucune tuile ne
//     change hors du grain. RIFE ne tourne pas : b sort seul, comme un dessin tenu de la mesure des blocs
//     (rife_plan.h), que le grain d'une source granuleuse empêche souvent de reconnaître.
//   - Paire impossible (Fade) : le flux ne trouve de correspondance presque nulle part (fumée, éclair, plan
//     qui se transforme) sans que ce soit une coupe. Le warp de RIFE y ferait des fantômes : tout le
//     mouvement est remplacé par un fondu enchaîné de a et b (kernels/spec/motion_doubt.h).
//   - Zone douteuse (Masked) : seules les tuiles où le flux doute (effet, tuile qui change sans
//     correspondance) passent au fondu, avec un bord adouci ; RIFE reste ailleurs.
// Sans carte (AMD, Intel, Turing, NVOF refusé, --scene-map éteint) ou sans --pair-triage : la décision
// d'aujourd'hui, image pour image. Les seuils ne sont pas mesurés : la porte du groupe 3 les règle.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "app/play/rife_plan.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::app::play {

enum class PairSort : std::uint8_t { Today, Held, Fade, Masked };

namespace triage {
// Part des tuiles qui changent (hors grain) au plus d'un dessin tenu par le flux.
inline constexpr double kHeldChanged = 0.02;
// Part des vecteurs sans correspondance d'une paire impossible (la carte dit coupe à 0,6).
inline constexpr double kImpossibleUnmatched = 0.45;
// Doute moyen de l'image à partir duquel toute la paire passe au fondu.
inline constexpr double kFadeShare = 0.5;
// Confiance sous laquelle une tuile qui change est douteuse (proportionnellement).
inline constexpr float kDoubtConfidence = 0.5F;
// Doute d'une case sous lequel elle compte pour rien (bruit des tuiles voisines adoucies).
inline constexpr float kDoubtFloor = 0.04F;
// Part du doute de la paire précédente gardée à la suivante (une zone ne clignote pas d'une paire à l'autre).
inline constexpr float kDoubtKeep = 0.5F;
}  // namespace triage

// Vrai si la carte dit la paire tenue : ni dessin ni caméra qui bougent, ni fondu, presque aucune tuile qui
// change.
bool flow_held(const kernels::SceneMotion& scene);

// Le doute brut de chaque tuile de la carte, dans [0, 1] : 1 pour un effet qui change, 1 − confiance / seuil
// pour une tuile qui change avec une confiance basse, 0 sinon.
std::vector<float> tile_doubt(const kernels::SceneMotion& scene);

// La grille de doute d'un champ de doutes cols × rows (tuiles) : dilatée d'une tuile puis adoucie (3 × 3),
// regroupée par le plus fort doute jusqu'à kDoubtMaxCells de côté, en poids 8 bits.
struct DoubtGrid {
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> weights;
  // Le doute moyen de l'image, dans [0, 1].
  double share = 0.0;
};
DoubtGrid doubt_grid(std::span<const float> doubt, int cols, int rows);

struct TriageCounts {
  std::uint64_t read = 0;     // paires où RIFE aurait tourné et que la carte décrit
  std::uint64_t held = 0;     // tenues par le flux seul
  std::uint64_t faded = 0;    // impossibles, fondues
  std::uint64_t masked = 0;   // fondues par endroits
  std::uint64_t refused = 0;  // le backend n'a pas motion_doubt : RIFE tel quel
};

// L'état du tri dans l'étage : la décision de la paire en cours, la grille de doute qu'elle laisse aux warps
// de ses instants, et le doute de la paire précédente (gardé en partie, oublié au saut et à la coupe).
class RifeTriage {
 public:
  // runs : RIFE tourne sur la paire d'après rife_runs(facts) (le flux tenu compris). Rend la décision.
  PairSort sort(const PairFacts& facts, bool runs);
  void reset();  // saut, nouvelle époque : rien de la paire précédente ne vaut plus
  // Dans l'étage : sort (sauf la première paire de Smooth, une tenue forcée), puis vrai si les instants de la
  // paire manquent à dessein (coupe ou dessin tenu), pas faute de temps (UpscaleStage::uninterpolated_).
  bool settle(const PairFacts& facts, bool on_grid, bool runs, bool smooth_first);

  // Le mouvement à lire au warp d'un instant de la paire : motion lui-même, ou le mouvement fondu par
  // endroits (KernelOps::motion_doubt) dans des tampons à part, l'entrée intacte pour les autres instants.
  // fade : 0,5 pour un mouvement du milieu, 1 − t pour un mouvement lu à l'instant t. Un refus du backend
  // éteint le tri pour la lecture, avec une ligne de journal : RIFE tel quel.
  gpu::Motion doubted(gpu::Backend& backend, const gpu::Motion& motion, float fade);

  const TriageCounts& counts() const { return counts_; }
  // La ligne de fin de lecture, si --pair-triage a été demandé sur une paire mesurée au moins.
  void log() const;

 private:
  PairSort sort_unrun(const PairFacts& facts);  // RIFE ne tourne pas : tenue par le flux, ou rien
  void keep_doubt(std::vector<float>& doubt, const kernels::SceneMotion& scene);
  bool asked_ = false;
  PairSort now_ = PairSort::Today;
  DoubtGrid grid_;
  std::vector<float> kept_;  // le doute de la paire précédente, tuile par tuile
  int kept_cols_ = 0;
  int kept_rows_ = 0;
  bool refused_ = false;
  TriageCounts counts_;
  std::unique_ptr<gpu::TensorBuffer> flow_;
  std::unique_ptr<gpu::TensorBuffer> mask_;
};

}  // namespace kirari::app::play
