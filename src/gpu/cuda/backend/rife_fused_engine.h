// RIFE fusionné vu comme gpu::Engine (kernels/spec/rife_fused.h) : six petits moteurs TensorRT (l'encodeur et
// les blocs, des convolutions à basse résolution) reliés par les noyaux CUDA de rife_fused_kernels.h, le tout
// enfilé sur le flux du backend et capturé dans un seul graphe CUDA. L'interface est celle du moteur RIFE
// entier : une entrée 1 × 11 × H × W (rife_input) et une sortie 1 × 3 × H × W, ou, pour la variante
// mouvement, le flot 1 × 4 × H × W et le masque 1 × 1 × H × W. Les tampons intermédiaires vivent à la suite
// des activations des moteurs dans la zone partagée de la voie (SharedActivations) : rien d'alloué de plus.
#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "gpu/backend.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "kernels/spec/rife_fused.h"

namespace nvinfer1 {
class IRuntime;
}

namespace ajp::gpu::cuda {

// La clé d'un Artifact de ce moteur : « rife-fused:<h>x<w>:<image|motion>:<d0,d1…>:<moteur0>,<moteur1>… »,
// les moteurs étant les fichiers du cache, dans l'ordre encodeur, blocs 0 à N-1. Artifact ne porte que la clé
// et un chemin : la clé dit tout ce que load() doit savoir.
struct RifeFusedKey {
  int height = 0;
  int width = 0;
  bool motion = false;
  kernels::RifeFusedPlan plan;
  std::vector<std::string> engines;
};
std::string rife_fused_artifact_key(const RifeFusedKey& key);
// Rien si la clé est celle d'un autre moteur ou illisible.
std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text);

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

  // Le mot du moteur pour le journal du backend (graphes coupés), une fois ; vide sinon.
  std::string take_notice();

 private:
  RifeFusedEngine() = default;
  // La chaîne, lancée ou capturée selon l'appelant.
  std::expected<void, Refusal> run(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                   const Tensor& input, std::span<const Tensor> outputs);
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);
  void drop_graphs();

  std::unique_ptr<TrtEngine> encode_;
  std::vector<std::unique_ptr<TrtEngine>> blocks_;
  kernels::RifeFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  bool motion_ = false;
  kernels::RifeFusedLayout layout_;
  std::size_t scratch_offset_ =
      0;  // début des tampons intermédiaires dans la zone : après les activations des moteurs
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_;
  std::shared_ptr<SharedActivations> zone_;

  static constexpr std::size_t kMaxGraphs = 8;
  static constexpr int kCaptureAttempts = 3;
  struct Graph {
    std::array<std::uintptr_t, 3> addresses{};  // entrée, sortie (image, ou flot), masque
    std::uintptr_t zone = 0;
    std::uint64_t generation = 0;
    cudaGraphExec_t exec = nullptr;
    bool warmed = false;
  };
  std::vector<Graph> graphs_;
  bool graphs_off_ = false;
  int capture_failures_ = 0;
  std::string notice_;
};

}  // namespace ajp::gpu::cuda
