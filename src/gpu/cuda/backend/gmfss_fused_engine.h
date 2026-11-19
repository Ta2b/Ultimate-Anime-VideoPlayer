// La fusion de GMFSS Fortuna Union vue comme un gpu::Engine (--gmfss-native, docs/gmfss-natif.md), avec les
// entrées et la sortie du moteur TensorRT de la fusion entière qu'elle remplace : l'étage
// (upscale_gmfss.cpp) ne voit pas la différence. Natif : l'IFNet (moteur TensorRT extrait,
// prepare/gmfss_fused.h) écrit l'image « rife », puis le GridNet par les noyaux de gmfss_native_net.h. Le
// moteur de la fusion entière reste chargé : référence de l'essai au chargement, et repli quand le natif est
// écarté ou que l'essai n'a pas été demandé.
#pragma once

#include <array>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "gpu/backend.h"
#include "gpu/cuda/backend/fused_keys.h"
#include "gpu/cuda/backend/gmfss_native_net.h"
#include "gpu/cuda/backend/trt_engine.h"

namespace nvinfer1 {
class IRuntime;
}

namespace kirari::gpu::cuda {

class SharedActivations;

class GmfssFusedEngine final : public Engine {
 public:
  // Les deux moteurs du cache, de formes vérifiées (entrées de la fusion à key.height × key.width).
  static std::expected<std::unique_ptr<GmfssFusedEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const GmfssFusedKey& key,
      const std::shared_ptr<SharedActivations>& shared);
  ~GmfssFusedEngine() override;
  GmfssFusedEngine(const GmfssFusedEngine&) = delete;
  GmfssFusedEngine& operator=(const GmfssFusedEngine&) = delete;

  std::span<const std::string> input_names() const override { return full_->input_names(); }
  std::span<const std::string> output_names() const override { return full_->output_names(); }

  // Les entrées dans l'ordre de input_names(), la sortie 1 × 3 × 2h × 2w.
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  // L'essai du natif (gmfss_fused_trial.cpp) : des entrées de test passent par le moteur entier, puis par
  // l'IFNet et le GridNet natif en sommes f16, mixtes et f32 ; le natif n'est gardé que si sa sortie tient
  // min_db contre celle de TensorRT et s'il le bat d'au moins 2 % en temps. La ligne du journal qui dit ce
  // qui a été retenu ; refus : l'essai n'a pas pu tourner (le moteur entier reste).
  std::expected<std::string, Refusal> try_native(const prepare::GmfssGrid& grid,
                                                 const std::shared_ptr<SharedActivations>& shared,
                                                 double min_db);
  bool native() const { return native_on_; }

 private:
  GmfssFusedEngine() = default;
  std::expected<void, Refusal> enqueue_native(cudaStream_t stream,
                                              const std::shared_ptr<SharedActivations>& shared,
                                              std::span<const Tensor> inputs, const Tensor& out);

  // Les entrées de la fusion, par nom.
  enum Input : std::size_t { kHalf0, kHalf1, kTime, kW01, kW02, kW03, kW11, kW12, kW13, kInputs };
  std::unique_ptr<TrtEngine> full_;
  std::unique_ptr<TrtEngine> ifnet_;
  std::unique_ptr<GmfssNativeNet> native_;
  bool native_on_ = false;
  int height_ = 0;
  int width_ = 0;
  std::array<std::size_t, kInputs> at_{};    // la place de chaque entrée dans input_names()
  std::array<std::size_t, 3> ifnet_from_{};  // l'entrée de la fusion de chaque entrée de l'IFNet
  void* rife_ = nullptr;                     // l'image de l'IFNet, 3 plans FP16
};

}  // namespace kirari::gpu::cuda
