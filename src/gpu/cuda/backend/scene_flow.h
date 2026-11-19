// La carte de la scène dans le backend TensorRT (KernelOps::scene_flow_start et scene_flow_take,
// kernels/spec/scene_flow.h) : la session NVOF, sa file, ses événements et ses relectures.
//
// Ordre des files, sans aucune attente hôte :
//   voie (file du backend) : attend « consommées » (NVOF a fini de lire les lumas de la paire précédente),
//     écrit les deux lumas du flux dans les tampons de NVOF, pose « lumas prêtes » ;
//   file de NVOF : attend « lumas prêtes », calcule le flux, réduit les tuiles, copie la grille dans une
//     mémoire hôte épinglée, pose l'événement de la relecture, puis « consommées ».
// Le bloc NVOFA tourne pendant RIFE et SPAN. La voie n'attend jamais NVOF : si « consommées » n'est pas
// encore passé quand la paire suivante arrive (NVOF plus lent que la chaîne), cette paire n'est pas lancée
// (busy), et l'attente de la voie sur « consommées » ne porte que sur un événement déjà passé. Un seul thread
// s'en sert (l'API de NVOF n'est pas sûre entre threads).
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

  // NVOF lit encore les lumas de la paire d'avant : la paire en cours ne sera pas lancée. Sans attente hôte.
  std::expected<bool, Refusal> busy() const;
  // Avant que la voie lane écrive les lumas : elle attend que NVOF ait lu les précédentes (déjà fait si
  // busy() a dit faux).
  std::expected<void, Refusal> before_lumas(cudaStream_t lane);
  // Après : le flux, la réduction et la copie sur la file de NVOF ; rend le ticket.
  std::expected<std::uint64_t, Refusal> run(cudaStream_t lane, bool restart, GpuTiming& timing);
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
  cudaEvent_t consumed_ = nullptr;
  bool consumed_posted_ = false;
  std::uint64_t tickets_ = 0;
};

}  // namespace kirari::gpu::cuda
