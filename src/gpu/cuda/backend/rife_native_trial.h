// L'essai du RIFE natif morceau par morceau (RifeFusedEngine::try_native, docs/rife-natif.md). Un morceau :
// l'encodeur sur l'image A de la paire de test, ou un bloc sur le X_k que la passe de TensorRT a laissé dans
// la zone (les morceaux seuls n'écrivent que l'encodage et les Y_k : les X_k restent ceux de TensorRT).
// Chacun passe par TensorRT puis par chaque réseau natif : sa sortie contre celle de TensorRT, son temps seul
// (un graphe de quelques passes, rejoué en tours intercalés comme span_timing.h), et le temps de chaque
// couche.
#pragma once

#include <cuda_runtime_api.h>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "gpu/cuda/backend/rife_fused_engine.h"

namespace kirari::gpu::cuda {

class RifeNativeTrial {
 public:
  // engine : la chaîne passée une fois par TensorRT sur input (la paire de test), dans la zone shared.
  RifeNativeTrial(RifeFusedEngine& engine, cudaStream_t stream, std::shared_ptr<SharedActivations> shared,
                  const Tensor& input);

  int pieces() const { return engine_.plan_.blocks + 1; }
  static std::string piece_name(int piece);

  // La justesse de chaque réseau, morceau par morceau : [morceau][réseau], en dB contre TensorRT (pic : la
  // plus grande valeur absolue de la sortie de TensorRT, au moins 1 ; -infini sur une valeur non finie).
  std::expected<std::vector<std::vector<double>>, Refusal> quality(
      std::span<const RifeNativeNet* const> nets);
  // Le temps seul de chaque morceau, en millisecondes : [morceau][option], l'option 0 TensorRT, puis nets.
  std::expected<std::vector<std::vector<double>>, Refusal> times(std::span<const RifeNativeNet* const> nets);
  // Le temps de chaque couche de net (rejoué en graphe, le meilleur de trois) : une ligne de journal par
  // couche, name nommant le réseau.
  std::expected<std::vector<std::string>, Refusal> profile(const RifeNativeNet& net, std::string_view name);

 private:
  struct ExecDelete {
    void operator()(cudaGraphExec_t exec) const { static_cast<void>(cudaGraphExecDestroy(exec)); }
  };
  using Exec = std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, ExecDelete>;
  struct Io {
    const void* in = nullptr;
    void* out = nullptr;
  };

  Io io(int piece) const;
  // La sortie du morceau par net (nul : TensorRT), ses canaux utiles copiés vers l'hôte.
  std::expected<std::vector<float>, Refusal> output(int piece, const RifeNativeNet* net);
  // Un graphe de kReps passes du morceau par net.
  std::expected<Exec, Refusal> capture(int piece, const RifeNativeNet* net);
  // Le temps d'une passe du graphe : un lancement de chauffe, puis deux entre deux événements.
  std::expected<double, Refusal> time(cudaGraphExec_t exec);

  RifeFusedEngine& engine_;
  cudaStream_t stream_;
  std::shared_ptr<SharedActivations> shared_;
  Tensor input_;
};

}  // namespace kirari::gpu::cuda
