// Tri des paires de RIFE par la carte de la scène (kernels/spec/scene_flow.h, --pair-triage, plan du flux
// NVIDIA, phase 3). Le flux optique matériel ne sert qu'à décider, jamais à déformer : aucun modèle nouveau.
//   - Dessin tenu (Held) : la carte dit que rien ne bouge, ni le dessin ni la caméra, qu'aucune tuile ne
//     change hors du grain et qu'aucune tuile de dessin ne se déplace. RIFE ne tourne pas : b sort seul,
//     comme un dessin tenu de la mesure des blocs (rife_plan.h), que le grain d'une source granuleuse empêche
//     souvent de reconnaître.
//   - Paire impossible (Fade) : le flux ne trouve de correspondance presque nulle part, pas même un décor que
//     la caméra explique (fumée ou éclair qui couvre l'image, plan qui se transforme, coupe que RIFE n'a pas
//     vue), ou la moitié des tuiles doute. Le warp de RIFE y ferait des fantômes : tout le mouvement est
//     remplacé par un fondu enchaîné de a et b (kernels/spec/motion_doubt.h).
//   - Zone douteuse (Masked) : seules les tuiles d'effet sûres (sans correspondance, beaucoup d'écart, vues
//     deux paires de suite) et groupées passent au fondu, avec un bord adouci ; RIFE reste ailleurs. Un
//     dessin que le flux suit mal (traits qui s'éparpillent, classe qui bascule) n'en est pas une : RIFE y
//     trouve son mouvement, le fondu y ferait une image double.
// GMFSS seul (for_gmfss) : le dessin tenu copié de même, la paire impossible fondue par le warp de l'étage
// (caméra identité), le panoramique pur (Pan : seule la caméra bouge) fait par le warp de la caméra de la
// carte, sans réseau (kernels/spec/motion_camera.h) ; pas de zone fondue par endroits, où GMFSS fait mieux
// que RIFE. Le mélange garde le tri de RIFE : une paire fondue n'a pas de clé GMFSS (upscale_gmfss.cpp).
// Smooth (dessins lissés) : ses paires restent hors du tri, la dernière de la tenue comprise (b y est un
// dessin nouveau pour la mesure des blocs, upscale_smooth.cpp). Sans carte (AMD, Intel, Turing, NVOF refusé,
// --scene-map éteint) ou sans --pair-triage : la décision d'aujourd'hui, image pour image. Les seuils ne sont
// pas mesurés : la porte du groupe 3 les règle.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "app/play/rife_plan.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::app::play {

enum class PairSort : std::uint8_t { Today, Held, Fade, Masked, Pan };

namespace triage {
// Part des tuiles qui changent (hors grain) au plus d'un dessin tenu par le flux.
inline constexpr double kHeldChanged = 0.02;
// Part des vecteurs sans correspondance d'une paire impossible, sans caméra ajustée (la carte dit coupe à
// 0,6).
inline constexpr double kImpossibleUnmatched = 0.45;
// Doute moyen des tuiles (avant d'être élargi au voisinage) à partir duquel toute la paire passe au fondu.
inline constexpr double kFadeShare = 0.5;
// Confiance d'une tuile d'effet sûre. La carte la donne quand elle voit l'effet deux paires de suite : à 1 −
// part des vecteurs qui correspondent (plus de 0,5), par l'image d'après la part des pixels qui bougent, ou
// pour une lumière à l'accord de ses pixels (0,6 à 1, kernels::scene::kLightAgree) ; à la première, elle
// garde la classe d'avant, et à la fin l'effet tenu n'a plus que 0,4 fois la confiance de la classe nouvelle
// (kernels::scene::kContradicted).
inline constexpr float kEffectSure = 0.5F;
// Doute d'une case sous lequel elle compte pour rien (bruit des tuiles voisines adoucies).
inline constexpr float kDoubtFloor = 0.04F;
// Part du doute propre de la paire précédente gardée à la suivante (une zone ne clignote pas d'une paire à
// l'autre), sans passer plus loin.
inline constexpr float kDoubtKeep = 0.5F;
// Paires sans RIFE de suite (dessin tenu) à travers lesquelles ce doute passe encore : un dessin en quatre en
// a trois entre deux dessins ; après une tenue plus longue, l'effet d'avant ne vaut plus.
inline constexpr int kKeepAcross = 3;
// Panoramique pur (GMFSS) : écart moyen des tuiles décor à la caméra, pixels de travail, au plus.
inline constexpr double kPanResidual = 1.0;
// Et part des vecteurs sans correspondance au plus (bords qui entrent dans l'image compris).
inline constexpr double kPanUnmatched = 0.1;
}  // namespace triage

// Vrai si la carte dit la paire tenue : ni dessin ni caméra qui bougent, ni fondu, presque aucune tuile qui
// change, et pas une seule tuile de dessin qui se déplace (un petit personnage au loin).
bool flow_held(const kernels::SceneMotion& scene);

// Vrai si la carte ne trouve presque rien qui corresponde (coupe probable comprise) sans même un décor que la
// caméra explique : la paire entière est impossible. Une caméra ajustée dit que le décor correspond : ce qui
// ne correspond pas est alors local (un personnage rapide, un effet), au doute des tuiles de le dire.
bool flow_lost(const kernels::SceneMotion& scene);

// Vrai si la carte dit un panoramique pur : la caméra ajustée bouge, aucune tuile de dessin (qui bouge hors
// de la caméra) ni d'effet, ni fondu ni coupe, peu de vecteurs sans correspondance et un décor qui suit le
// modèle.
bool flow_pan(const kernels::SceneMotion& scene);

// Le doute brut de chaque tuile de la carte, 0 ou 1 : 1 pour une tuile d'effet sûre (kEffectSure) qui change
// et qui touche une autre tuile d'effet sûre (8 voisines), 0 sinon. Ni dessin ni décor, quelle que soit leur
// confiance : une confiance basse y dit que NVOF suit mal le trait (vecteurs épars d'un personnage qui
// bouge, classe qui bascule à son bord), pas que RIFE n'y trouvera rien ; par elle, 362 paires sur 368
// fondaient par endroits sur un passage d'action. Ni Uncertain (aplat, peu d'écart sans
// correspondance).
std::vector<float> tile_doubt(const kernels::SceneMotion& scene);

// La grille de doute d'un champ de doutes cols × rows (tuiles) : dilatée d'une tuile puis adoucie (3 × 3),
// regroupée par le plus fort doute jusqu'à kDoubtMaxCells de côté, en poids 8 bits.
struct DoubtGrid {
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> weights;
  // Le doute moyen des tuiles mêmes, avant d'être élargi au voisinage (qui ne sert qu'au bord du fondu),
  // dans [0, 1].
  double share = 0.0;
};
DoubtGrid doubt_grid(std::span<const float> doubt, int cols, int rows);

struct TriageCounts {
  std::uint64_t read = 0;     // paires où RIFE aurait tourné et que la carte décrit
  std::uint64_t held = 0;     // tenues par le flux seul
  std::uint64_t faded = 0;    // impossibles, fondues
  std::uint64_t lost = 0;     // dont : rien qui corresponde ni caméra (flow_lost) ; le reste, par le doute
  std::uint64_t masked = 0;   // fondues par endroits
  double masked_share = 0.0;  // somme de leur doute moyen (DoubtGrid::share), pour la ligne de fin
  std::uint64_t panned = 0;   // panoramiques purs par le warp de la caméra (GMFSS seul)
  std::uint64_t refused = 0;  // le backend n'a pas motion_doubt : RIFE tel quel
};

// L'état du tri dans l'étage : la décision de la paire en cours, la grille de doute qu'elle laisse aux warps
// de ses instants, et le doute de la paire précédente (gardé en partie, oublié au saut et à la coupe).
class RifeTriage {
 public:
  // GMFSS seul : Fade et Pan par le warp de la caméra, jamais Masked (UpscaleStage).
  void for_gmfss(bool only) { gmfss_only_ = only; }
  PairSort now() const { return now_; }
  // La caméra de la paire Fade (identité) ou Pan (celle de la carte), pixels de travail.
  const std::array<float, 6>& camera() const { return camera_; }
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
  // RIFE ne tourne pas : tenue par le flux, ou rien.
  PairSort sort_unrun(const PairFacts& facts, const kernels::SceneMotion& scene);
  void keep_doubt(std::vector<float>& doubt, const kernels::SceneMotion& scene);
  PairSort sort_gmfss(const kernels::SceneMotion& scene, const std::vector<float>& doubt);
  bool asked_ = false;
  bool gmfss_only_ = false;
  std::array<float, 6> camera_{};
  PairSort now_ = PairSort::Today;
  DoubtGrid grid_;
  std::vector<float> kept_;  // le doute de la paire précédente, tuile par tuile
  int kept_cols_ = 0;
  int kept_rows_ = 0;
  int unrun_ = 0;  // paires sans RIFE depuis le dernier doute gardé
  bool refused_ = false;
  TriageCounts counts_;
  // Les tampons du mouvement fondu, un jeu par forme : le milieu et la demi-taille de l'échelle par paire
  // (pair_scale) alternent d'un instant à l'autre d'une paire forte, et une allocation attend la file.
  struct Buffers {
    std::unique_ptr<gpu::TensorBuffer> flow;
    std::unique_ptr<gpu::TensorBuffer> mask;
  };
  std::array<Buffers, 2> buffers_;
  std::size_t last_ = 0;  // le jeu servi en dernier ; une forme nouvelle prend l'autre
};

}  // namespace kirari::app::play
