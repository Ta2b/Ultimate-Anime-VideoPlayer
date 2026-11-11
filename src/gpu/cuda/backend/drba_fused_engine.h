// DistilDRBA fusionné vu comme gpu::Engine (kernels/spec/drba_fused.h) : quatre petits moteurs TensorRT
// (l'encodeur en lot de trois images, et les blocs 0 à 2, des convolutions à basse résolution) reliés par les
// noyaux CUDA de drba_fused_kernels.h, le tout enfilé sur le flux du backend et capturé dans un seul graphe
// CUDA. L'interface est celle du moteur DistilDRBA entier : deux entrées « pair01 » et « pair12 » de
// 1 × 7 × H × W (rife_input) et une sortie « out » de 1 × 3 × H × W. Les tampons intermédiaires vivent à la
// suite des activations des moteurs dans la zone partagée de la voie (SharedActivations) : rien d'alloué de
// plus.
#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/backend.h"
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/fused_keys.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "kernels/spec/drba_fused.h"

namespace nvinfer1 {
class IRuntime;
}

namespace ajp::gpu::cuda {

class DrbaFusedEngine final : public Engine {
 public:
  // Charge les quatre moteurs (chacun leurs activations dans shared), vérifie leurs formes, prend la place
  // des tampons intermédiaires dans shared et fait passer trois fois une paire de zéros par toute la chaîne
  // (une exécution à froid, une capture de graphe, un rejeu) : un noyau ou un moteur qui ne tourne pas sur
  // cet appareil se voit ici, au chargement, pas en pleine lecture. Refus : un moteur illisible ou de forme
  // inattendue, mémoire, erreur CUDA de l'essai.
  static std::expected<std::unique_ptr<DrbaFusedEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const DrbaFusedKey& key,
      const std::shared_ptr<SharedActivations>& shared);
  ~DrbaFusedEngine() override;
  DrbaFusedEngine(const DrbaFusedEngine&) = delete;
  DrbaFusedEngine& operator=(const DrbaFusedEngine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  Shape input_shape() const { return {.dims = {1, kernels::kDrbaFusedInputPlanes, height_, width_}}; }

  // Enfile la chaîne sur stream (arena : la zone de la voie de stream). Les mêmes adresses une troisième fois
  // : le lancement passe par un graphe CUDA capturé à la deuxième. Une sortie : l'image ; deux : le flot et
  // le masque du warp final (variante mouvement, le warp après SPAN).
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  // D6 : le prochain enqueue reprend les mêmes paires, seul le temps a changé.
  void next_times_only() override { times_only_next_ = true; }

  // Le mot du moteur pour le journal du backend (graphes coupés, registre plein), une fois ; vide sinon.
  std::string take_notice() { return graphs_.take_notice(); }

 private:
  DrbaFusedEngine() = default;
  // La chaîne, lancée ou capturée selon l'appelant.
  // outputs : l'image (1 × 3), ou le flot (1 × 4) et le masque (1 × 1) de la variante mouvement.
  // times_only : pre n'écrit que les plans du temps, l'encodeur et post sont sautés (D6).
  std::expected<void, Refusal> run(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                   std::span<const Tensor> inputs, std::span<const Tensor> outputs,
                                   bool times_only);
  std::expected<void, Refusal> enqueue_chain(cudaStream_t stream,
                                             const std::shared_ptr<SharedActivations>& arena,
                                             std::span<const Tensor> inputs, std::span<const Tensor> outputs,
                                             bool times_only);
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);

  std::unique_ptr<TrtEngine> encode_;
  std::array<std::unique_ptr<TrtEngine>, kernels::kDrbaFusedBlocks> blocks_;
  kernels::DrbaFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  kernels::DrbaFusedLayout layout_;
  std::size_t scratch_offset_ =
      0;  // début des tampons intermédiaires dans la zone : après les activations des moteurs
  std::vector<std::string> inputs_{"pair01", "pair12"};
  std::vector<std::string> outputs_{"out"};
  std::shared_ptr<SharedActivations> zone_;

  std::uint64_t zone_generation_ = 0;  // la génération de zone_ à use_zone

  // Graphes par jeu d'adresses (pair01, pair12, sortie ou flot, masque), de zone et de variante : deux par
  // jeu d'adresses, complet et temps seul (D6).
  static constexpr std::size_t kMaxGraphs = 16;
  FusedGraphs graphs_{kMaxGraphs, "DistilDRBA fusionné"};
  // Ce que le dernier appel mené jusqu'au bout a laissé dans la zone : images réduites et encodages de ces
  // paires (D6). Repris seulement si la zone est la même et qu'aucun autre moteur n'y a écrit depuis.
  struct Kept {
    std::uintptr_t pair01 = 0;
    std::uintptr_t pair12 = 0;
    const void* zone = nullptr;
    std::uint64_t generation = 0;
  };
  std::optional<Kept> kept_;
  bool times_only_next_ = false;
};

}  // namespace ajp::gpu::cuda
