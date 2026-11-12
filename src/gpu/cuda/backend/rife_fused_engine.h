// RIFE fusionné vu comme gpu::Engine (kernels/spec/rife_fused.h) : six petits moteurs TensorRT (l'encodeur et
// les blocs, des convolutions à basse résolution) reliés par les noyaux CUDA de rife_fused_kernels.h, le tout
// enfilé sur le flux du backend et capturé dans un seul graphe CUDA. L'interface est celle du moteur RIFE
// entier : une entrée 1 × 11 × H × W (rife_input) et une sortie 1 × 3 × H × W, ou, pour la variante
// mouvement, le flot 1 × 4 × H × W et le masque 1 × 1 × H × W. Les tampons intermédiaires vivent à la suite
// des activations des moteurs dans la zone partagée de la voie (SharedActivations) : rien d'alloué de plus.
// L'encodeur prend une image à la fois : l'encodage d'une paire se garde d'un appel à l'autre, entier pour un
// autre instant de la même paire (next_times_only), celui de B pour A à la paire suivante
// (next_pair_follows), tant que la zone n'a servi à aucun autre moteur.
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
#include <vector>

#include "gpu/backend.h"
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/fused_keys.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "kernels/spec/rife_fused.h"

namespace nvinfer1 {
class IRuntime;
}

namespace ajp::gpu::cuda {

class RifeFusedEngine final : public Engine {
 public:
  // Charge les 1 + blocs moteurs (chacun leurs activations dans shared), vérifie leurs formes, prend la place
  // des tampons intermédiaires dans shared et fait passer trois fois une image de zéros par toute la chaîne
  // (une exécution à froid, une capture de graphe, un rejeu) : un noyau ou un moteur qui ne tourne pas sur
  // cet appareil se voit ici, au chargement, pas en pleine lecture. Refus : un moteur illisible ou de forme
  // inattendue, mémoire, erreur CUDA de l'essai.
  static std::expected<std::unique_ptr<RifeFusedEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const RifeFusedKey& key,
      const std::shared_ptr<SharedActivations>& shared);
  ~RifeFusedEngine() override;
  RifeFusedEngine(const RifeFusedEngine&) = delete;
  RifeFusedEngine& operator=(const RifeFusedEngine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  Shape input_shape() const { return {.dims = {1, kernels::kFusedInputPlanes, height_, width_}}; }
  bool motion() const { return motion_; }

  // Enfile la chaîne sur stream (arena : la zone de la voie de stream). Les mêmes adresses une troisième fois
  // : le lancement passe par un graphe CUDA capturé à la deuxième.
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  // Le prochain enqueue reprend la même entrée, seul l'instant a changé : l'encodeur est sauté.
  void next_times_only() override { hint_ = Hint::TimesOnly; }
  // Le prochain enqueue prend pour A la B de l'appel précédent : son encodage passe à la place de A, seule la
  // nouvelle B est encodée.
  void next_pair_follows() override { hint_ = Hint::Follows; }
  // L'annonce oubliée : le backend a refusé l'appel avant d'enfiler la chaîne.
  void drop_hint() { hint_ = Hint::None; }

  // Le mot du moteur pour le journal du backend (graphes coupés, registre plein), une fois ; vide sinon.
  std::string take_notice() { return graphs_.take_notice(); }

 private:
  RifeFusedEngine() = default;
  // Ce que l'encodeur fait à cet appel : les deux images, B seule (l'encodage de B d'avant copié à la place
  // de A), rien (l'encodage de l'appel précédent sert tel quel).
  enum class Encode : std::uint8_t { Both, Second, None };
  enum class Hint : std::uint8_t { None, TimesOnly, Follows };
  // La chaîne, lancée ou capturée selon l'appelant.
  std::expected<void, Refusal> run(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                   const Tensor& input, std::span<const Tensor> outputs, Encode encode);
  std::expected<void, Refusal> enqueue_chain(cudaStream_t stream,
                                             const std::shared_ptr<SharedActivations>& arena,
                                             std::span<const Tensor> inputs, std::span<const Tensor> outputs,
                                             Encode encode);
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);

  std::unique_ptr<TrtEngine> encode_;
  std::vector<std::unique_ptr<TrtEngine>> blocks_;
  kernels::RifeFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  bool motion_ = false;
  kernels::RifeWarp warp_ = kernels::RifeWarp::Bilinear;
  kernels::RifeFusedLayout layout_;
  std::size_t scratch_offset_ =
      0;  // début des tampons intermédiaires dans la zone : après les activations des moteurs
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_;
  std::shared_ptr<SharedActivations> zone_;

  std::uint64_t zone_generation_ = 0;  // la génération de zone_ à use_zone

  // Graphes par jeu d'adresses (entrée, sortie ou flot, masque), de zone et de travail de l'encodeur : trois
  // par jeu d'adresses (Encode).
  static constexpr std::size_t kMaxGraphs = 24;
  FusedGraphs graphs_{kMaxGraphs, "RIFE fusionné"};
  // L'encodage que le dernier appel mené jusqu'au bout a laissé dans la zone, et l'entrée qu'il a lue.
  // Repris seulement si la zone est la même et qu'aucun autre moteur n'y a écrit depuis.
  struct Kept {
    std::uintptr_t input = 0;
    const void* zone = nullptr;
    std::uint64_t generation = 0;
  };
  std::optional<Kept> kept_;
  Hint hint_ = Hint::None;
};

}  // namespace ajp::gpu::cuda
