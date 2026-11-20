// La carte de la scène dans le backend TensorRT (KernelOps::scene_flow_start et scene_flow_take,
// kernels/spec/scene_flow.h) : la session NVOF, sa file, ses événements et ses relectures.
//
// Ordre des files, sans aucune attente hôte :
//   voie (file du backend) : attend que NVOF ait fini de lire le tampon de luma qu'elle va écrire, y écrit
//     la luma de b (et celle de a si la paire ne suit pas la précédente), pose « lumas prêtes » ;
//   file de NVOF : attend « lumas prêtes », calcule le flux, réduit les tuiles, pose « relâché » sur les deux
//     tampons lus, copie la grille dans une mémoire hôte épinglée, pose l'événement de la relecture.
// Trois tampons de luma tournent (NvofSession::kLumas) : une paire qui suit la précédente reprend sa luma de
// b comme a et n'écrit que b, dans le tampon que la paire d'avant-dernière a lu. La voie n'attend donc NVOF
// que deux paires en arrière, déjà passées : si ce n'est pas le cas (NVOF plus lent que la chaîne), la paire
// n'est pas lancée (plan rend nullopt) et rien n'attend. Le bloc NVOFA tourne pendant RIFE et SPAN. Un seul
// thread s'en sert (l'API de NVOF n'est pas sûre entre threads).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "core/types.h"
#include "gpu/cuda/backend/gpu_timing.h"
#include "gpu/cuda/backend/nvof_session.h"
#include "gpu/cuda/backend/scene_roundtrip.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/scene_camera.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::gpu::cuda {

struct SceneRefineState;  // scene_camera_kernels.h

class SceneFlow {
 public:
  explicit SceneFlow(bool roundtrip = false) : roundtrip_wanted_(roundtrip) {}
  ~SceneFlow();
  SceneFlow(const SceneFlow&) = delete;
  SceneFlow& operator=(const SceneFlow&) = delete;

  // Une session pour un flux de cette taille : gardée si la taille n'a pas changé, sinon refaite (les
  // relectures en vol deviennent périmées). Vrai si la session est neuve. Un refus d'ouverture est gardé :
  // les appels suivants le rendent sans rien essayer.
  std::expected<bool, Refusal> prepare(int flow_width, int flow_height, int divisor);
  NvofSession& session() { return *session_; }
  // À la destruction du backend : finit la dernière relecture éventuelle et rend le relevé du retour.
  std::string finish_roundtrip();

  // Les tampons de luma d'une paire : a et b, et si la luma de a est déjà là (b de la paire d'avant).
  struct Lumas {
    int a = 0;
    int b = 1;
    bool keep_a = false;
  };
  // Ceux de la paire à venir. follows : a est l'image b de la paire lancée juste avant. nullopt si NVOF lit
  // encore un tampon à écrire : la paire ne sera pas lancée. Sans attente hôte.
  std::expected<std::optional<Lumas>, Refusal> plan(bool follows) const;
  // Avant que la voie lane écrive les lumas du plan : elle attend que NVOF ait lu ces tampons (déjà fait si
  // plan les a rendus).
  std::expected<void, Refusal> before_lumas(cudaStream_t lane, const Lumas& lumas);
  // Après : le flux, la réduction et la copie sur la file de NVOF ; rend le ticket. Deux mesures : le flux
  // (le bloc NVOFA, à côté des SM : une latence, pas un coût par image) et la réduction (sur les SM).
  std::expected<std::uint64_t, Refusal> run(cudaStream_t lane, const Lumas& lumas, bool restart,
                                            GpuTiming& flow_timing, GpuTiming& tiles_timing);
  // La grille d'un ticket si sa copie est passée, sans attendre ; nullopt sinon. Périmé : SequenceOrder.
  std::expected<std::optional<kernels::SceneFlowTiles>, Refusal> take(std::uint64_t ticket);
  std::expected<std::optional<kernels::SceneRoundtripTiles>, Refusal> take_roundtrip(std::uint64_t ticket);

  // L'affinage de la caméra d'une paire lancée (KernelOps::scene_camera_start, scene_camera.cpp) : sur la
  // file de NVOF, sur les lumas de la paire, puis « relâché » reposé sur leurs tampons. Ticket 0, rien de
  // lancé, si une paire plus récente a réécrit l'un d'eux, si la paire est trop ancienne ou si les relectures
  // sont toutes en vol.
  std::expected<std::uint64_t, Refusal> refine(std::uint64_t flow_ticket,
                                               const kernels::SceneCameraRequest& request, GpuTiming& timing);
  // Le résultat d'un affinage si sa copie est passée, sans attendre ; nullopt sinon. Périmé : SequenceOrder.
  std::expected<std::optional<kernels::SceneCameraRefined>, Refusal> refined(std::uint64_t ticket);

 private:
  struct Read {
    kernels::SceneTileSums* host = nullptr;
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
  };
  std::expected<void, Refusal> allocate_grid();  // la grille de cols_ × rows_ tuiles et ses relectures
  void release_grid();                           // la file de NVOF déjà vide
  // Ce qu'il faut à l'affinage (scene_camera.cpp) : lumas floutées, état, tuiles, relectures épinglées.
  std::expected<void, Refusal> allocate_refine();
  void release_refine();
  // Les lumas floutées de la paire et les pas de Gauss-Newton, puis la vérification, sur la file own.
  std::expected<void, Refusal> enqueue_refine(cudaStream_t own, const Lumas& lumas);

  // Une paire lancée : ses tampons et le compte d'écritures de chacun à son lancement.
  struct Launched {
    std::uint64_t ticket = 0;
    Lumas lumas;
    std::array<std::uint64_t, 2> writes{};
  };
  // Un affinage : sa demande et sa relecture, en mémoire hôte épinglée.
  struct Refine {
    double* model = nullptr;        // 6
    std::uint8_t* decor = nullptr;  // cols × rows
    SceneRefineState* state = nullptr;
    kernels::SceneTileCheck* checks = nullptr;  // cols × rows
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
    bool posted = false;  // done enregistré au moins une fois
  };

  std::optional<Refusal> refused_;
  std::unique_ptr<NvofSession> session_;
  bool roundtrip_wanted_ = false;
  std::unique_ptr<SceneRoundtrip> roundtrip_;
  std::string roundtrip_notice_;
  int divisor_ = 1;
  int cols_ = 0;
  int rows_ = 0;
  kernels::SceneTileSums* device_ = nullptr;
  std::array<Read, kSceneFlowReadbacks> reads_{};
  cudaEvent_t lumas_ready_ = nullptr;
  // Par tampon de luma : NVOF a fini de le lire (flux et réduction de la dernière paire qui l'a lu).
  std::array<cudaEvent_t, NvofSession::kLumas> released_{};
  std::array<bool, NvofSession::kLumas> released_posted_{};
  std::optional<Lumas> last_;  // les tampons de la dernière paire lancée
  std::uint64_t tickets_ = 0;
  std::array<std::uint64_t, NvofSession::kLumas> writes_{};  // écritures de chaque tampon de luma
  std::array<Launched, kSceneFlowReadbacks> launched_{};     // par ticket % kSceneFlowReadbacks
  std::array<Refine, kSceneFlowReadbacks> refines_{};
  std::array<float*, 3> blurred_{};  // a, b, et la somme des lignes
  SceneRefineState* refine_state_ = nullptr;
  std::uint8_t* refine_decor_ = nullptr;
  kernels::SceneTileCheck* refine_checks_ = nullptr;
  std::uint64_t refine_tickets_ = 0;
};

}  // namespace kirari::gpu::cuda
