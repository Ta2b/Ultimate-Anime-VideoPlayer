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
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace ajp::gpu::cuda {

class TrtEngine final : public Engine {
 public:
  static constexpr std::size_t kMaxInputs = 16;
  static constexpr std::size_t kMaxOutputs = 16;
  // Lit et désérialise le moteur. Refus : fichier illisible (IoFailed), moteur refusé par
  // ce TensorRT (VendorCallFailed), entrée ou sortie qui n'est pas un tenseur FP16 de rang 4.
  static std::expected<std::unique_ptr<TrtEngine>, Refusal> load(nvinfer1::IRuntime& runtime,
                                                                 const std::filesystem::path& path);
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
  std::expected<void, Refusal> enqueue(cudaStream_t stream, std::span<const Tensor> inputs,
                                       std::span<const Tensor> outputs);

 private:
  TrtEngine() = default;

  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<Shape> input_shapes_;
  std::vector<Shape> output_shapes_;
  // Graphes capturés, par jeu d'adresses (les tampons de l'étage tournent sur quelques-uns) ;
  // graphs_off_ après un refus de capture : les lancements restent directs.
  static constexpr std::size_t kMaxGraphs = 8;
  struct Graph {
    std::vector<std::uintptr_t> addresses;
    cudaGraphExec_t exec = nullptr;
    bool warmed = false;
  };
  std::vector<Graph> graphs_;
  bool graphs_off_ = false;
  std::expected<void, Refusal> launch(cudaStream_t stream, Graph* graph);
};

}  // namespace ajp::gpu::cuda
