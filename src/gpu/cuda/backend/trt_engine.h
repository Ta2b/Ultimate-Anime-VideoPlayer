// Un moteur TensorRT désérialisé et son contexte d'exécution, vus comme gpu::Engine.
//
// Provisoire pour la tranche verticale : l'enveloppe complète (profils de formes, journal
// relié à spdlog, refus détaillés) est le chantier pc/tensorrt (src/gpu/cuda/trt). Celle-ci
// ne connaît que ce que SPAN, RIFE et GMFSS demandent : 1 à kMaxInputs entrées et 1 à kMaxOutputs
// sorties FP16 de formes fixes (GMFSS : le graphe « paire » a 14 sorties, « fusion » 8 entrées).
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace ajp::gpu::cuda {

// Mémoire d'activation partagée par les moteurs d'une voie du backend (gpu/lanes.h) : ils
// s'exécutent l'un après l'autre sur sa file, une zone de la taille du plus gros suffit. Sans
// elle, RIFE, SPAN et les graphes de GMFSS gardent chacun la leur : plusieurs centaines de Mo de
// plus sur les 6 Go d'une RTX 4050 mobile. Deux voies de calcul, deux zones : SPAN tourne
// pendant RIFE ou GMFSS sans écrire leur mémoire.
class SharedActivations {
 public:
  SharedActivations() = default;
  SharedActivations(const SharedActivations&) = delete;
  SharedActivations& operator=(const SharedActivations&) = delete;
  SharedActivations(SharedActivations&&) = delete;
  SharedActivations& operator=(SharedActivations&&) = delete;
  ~SharedActivations();

  // Au moins bytes octets ; une zone plus grande remplace l'ancienne après la fin du travail en
  // cours sur l'appareil (generation() change). Refus GpuMemoryExhausted.
  std::expected<void, Refusal> reserve(std::int64_t bytes);
  void* data() const { return data_; }
  std::int64_t bytes() const { return bytes_; }
  std::uint64_t generation() const { return generation_; }

 private:
  void* data_ = nullptr;
  std::int64_t bytes_ = 0;
  std::uint64_t generation_ = 0;
};

class TrtEngine final : public Engine {
 public:
  static constexpr std::size_t kMaxInputs = 16;
  static constexpr std::size_t kMaxOutputs = 16;
  // Lit et désérialise le moteur. Refus : fichier illisible (IoFailed), moteur refusé par
  // ce TensorRT (VendorCallFailed), entrée ou sortie qui n'est pas un tenseur FP16 de rang 4.
  // shared : les activations dans cette zone commune (kUSER_MANAGED), sinon dans celle du
  // contexte.
  static std::expected<std::unique_ptr<TrtEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& path,
      std::shared_ptr<SharedActivations> shared = nullptr);
  ~TrtEngine() override;
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }

  // Forme de l'entrée ou de la sortie i, telle que le moteur la fixe.
  Shape input_shape(std::size_t i) const { return input_shapes_[i]; }
  Shape output_shape(std::size_t i) const { return output_shapes_[i]; }

  // Pose les adresses et lance le moteur sur stream. Les formes doivent être celles du moteur.
  // Les mêmes adresses une troisième fois : le lancement passe par un graphe CUDA capturé à la
  // deuxième (la première prépare le contexte), qui rejoue les noyaux sans les relancer un à un.
  // arena : la zone de la voie de stream ; un moteur chargé avec une zone commune y passe (elle
  // grandit à sa taille au besoin, ses graphes sont refaits). L'appelant ordonne les voies.
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  // Comme enqueue, mais sans graphe CUDA propre : adresses posées, puis enqueueV3 sur stream. Pour un
  // appelant qui capture lui-même une chaîne de moteurs et de noyaux dans un seul graphe (RifeFusedEngine) :
  // le premier lancement d'une série d'adresses se fait hors capture, comme pour enqueue.
  std::expected<void, Refusal> enqueue_plain(cudaStream_t stream,
                                             const std::shared_ptr<SharedActivations>& arena,
                                             std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  // La mémoire d'activation que ce moteur demande à sa zone.
  std::int64_t activation_bytes() const;

  // Le mot du moteur pour le journal du backend (graphes CUDA coupés après kCaptureAttempts
  // captures refusées), une fois ; vide sinon.
  std::string take_notice() { return std::exchange(notice_, std::string()); }

 private:
  TrtEngine() = default;

  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<Shape> input_shapes_;
  std::vector<Shape> output_shapes_;
  // Graphes capturés, par jeu d'adresses (les tampons de l'étage tournent sur quelques-uns).
  // Une capture refusée est réessayée au lancement suivant : un appel CUDA d'un autre fil au
  // mauvais moment peut l'invalider (les noyaux NVENC sur le flux par défaut le faisaient avant
  // aux_streams_) ; graphs_off_ après kCaptureAttempts refus, et les lancements restent directs.
  static constexpr std::size_t kMaxGraphs = 8;
  static constexpr int kCaptureAttempts = 3;
  struct Graph {
    std::vector<std::uintptr_t> addresses;
    cudaGraphExec_t exec = nullptr;
    bool warmed = false;
  };
  std::vector<Graph> graphs_;
  // Files auxiliaires du moteur, non bloquantes, créées ici et données au contexte : celles que
  // TensorRT crée seul sont bloquantes, et un lancement sur le flux par défaut d'un autre fil (les
  // noyaux de l'encodeur NVENC) dépend alors d'elles ; pendant une capture, ce lancement l'invalide
  // (CUDA 906 chez lui, 901 ici).
  std::vector<cudaStream_t> aux_streams_;
  bool graphs_off_ = false;
  int capture_failures_ = 0;
  std::string notice_;
  std::shared_ptr<SharedActivations> shared_;
  std::uint64_t shared_generation_ = 0;  // zone posée sur le contexte ; 0 : aucune
  void drop_graphs();
  // La zone commune posée, les adresses des tenseurs et les files auxiliaires donnés au contexte.
  std::expected<void, Refusal> place(const std::shared_ptr<SharedActivations>& arena,
                                     std::span<const Tensor> inputs, std::span<const Tensor> outputs);
  std::expected<void, Refusal> launch(cudaStream_t stream, Graph* graph);
};

}  // namespace ajp::gpu::cuda
