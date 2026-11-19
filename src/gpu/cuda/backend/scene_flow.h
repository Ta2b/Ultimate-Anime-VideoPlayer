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

#include "core/types.h"
#include "gpu/cuda/backend/gpu_timing.h"
#include "gpu/cuda/backend/nvof_session.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::gpu::cuda {

class SceneFlow {
 public:
  SceneFlow() = default;
  ~SceneFlow();
  SceneFlow(const SceneFlow&) = delete;
  SceneFlow& operator=(const SceneFlow&) = delete;

  // Une session pour un flux de cette taille : gardée si la taille n'a pas changé, sinon refaite (les
  // relectures en vol deviennent périmées). Vrai si la session est neuve. Un refus d'ouverture est gardé :
  // les appels suivants le rendent sans rien essayer.
  std::expected<bool, Refusal> prepare(int flow_width, int flow_height, int divisor);
  NvofSession& session() { return *session_; }

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

 private:
  struct Read {
    kernels::SceneTileSums* host = nullptr;
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
  };
  std::expected<void, Refusal> allocate_grid();  // la grille de cols_ × rows_ tuiles et ses relectures
  void release_grid();                           // la file de NVOF déjà vide

  std::optional<Refusal> refused_;
  std::unique_ptr<NvofSession> session_;
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
};

}  // namespace kirari::gpu::cuda
