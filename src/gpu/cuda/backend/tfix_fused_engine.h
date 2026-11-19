// TemporalFix fusionné vu comme deux gpu::Engine (kernels/spec/tfix_fused.h), avec l'interface des deux
// moteurs du cache de l'encodeur (prepare/onnx_cache.h) qu'ils remplacent : l'étage (upscale_temporal.cpp)
// ne voit pas la différence, sauf le nombre de plans de ses tampons gardés (kernels::tfix_packed_planes).
//
// - L'encodeur : une image de travail (1 × 3) → son tampon gardé (1 × P) : en demi-taille l'image moyennée
//   2 × 2, puis les traits (moteur TensorRT de l'encodeur), puis les trois niveaux de pyramide. Lancé une
//   fois par image ingérée.
// - La fenêtre : sept images de travail (« input_0 » à « input_6 ») et leurs sept tampons gardés
//   (« features_0 » à « features_6 ») → l'image du milieu stabilisée. Les six appuis passent en un lot de six
//   par les trois blocs et la tête (quatre petits moteurs TensorRT), reliés par les noyaux de
//   tfix_fused_kernels.h ; toute la fenêtre est capturée dans un seul graphe CUDA, qui ne dépend pas des
//   tampons choisis (table d'adresses posée avant chaque lancement). Les tampons intermédiaires vivent dans
//   la zone partagée de la voie, dans la région du moteur (comme DistilDRBA fusionné).
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
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/fused_keys.h"
#include "gpu/cuda/backend/tfix_native_net.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "kernels/spec/tfix_fused.h"

namespace nvinfer1 {
class IRuntime;
}

namespace kirari::gpu::cuda {

class TfixEncoderEngine final : public Engine {
 public:
  // Charge le moteur de l'encodeur et vérifie sa forme (1 × 3 × H × W → 1 × 4 × H × W).
  static std::expected<std::unique_ptr<TfixEncoderEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const TfixFusedKey& key,
      const std::shared_ptr<SharedActivations>& shared);

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  // L'image de travail et le tampon gardé.
  Shape input_shape() const;
  Shape output_shape() const;

  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  // --tfix-native (docs/tfix-natif.md) : les convolutions de l'encodeur passent aux noyaux natifs si leurs
  // traits tiennent min_db contre ceux du moteur TensorRT sur une image de test (sommes f16, mixtes, puis
  // f32). pieces : l'encodeur seul (les quatre autres vides). Rend la ligne du journal ; un refus laisse
  // TensorRT.
  std::expected<std::string, Refusal> try_native(const std::vector<RifeNativePiece>& pieces,
                                                 const std::shared_ptr<SharedActivations>& shared,
                                                 double min_db);
  bool native() const { return native_on_; }

 private:
  TfixEncoderEngine() = default;
  std::expected<std::vector<float>, Refusal> trial_traits(cudaStream_t stream,
                                                          const std::shared_ptr<SharedActivations>& shared,
                                                          const Tensor& input, const Tensor& output);
  std::unique_ptr<TrtEngine> encode_;
  std::unique_ptr<TfixNativeNet> native_;  // actif à la place de encode_ (qui reste chargé) si native_on_
  bool native_on_ = false;
  int height_ = 0;  // le réseau
  int width_ = 0;
  bool half_ = false;
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_{"features"};
};

class TfixWindowEngine final : public Engine {
 public:
  // Charge les quatre moteurs (activations dans shared), vérifie leurs formes, prend la place des tampons
  // dans shared et fait passer trois fois une fenêtre de zéros par toute la chaîne (exécution à froid,
  // capture, rejeu) : un noyau ou un moteur qui ne tourne pas sur cet appareil se voit au chargement.
  static std::expected<std::unique_ptr<TfixWindowEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const TfixFusedKey& key,
      const std::shared_ptr<SharedActivations>& shared);
  ~TfixWindowEngine() override;
  TfixWindowEngine(const TfixWindowEngine&) = delete;
  TfixWindowEngine& operator=(const TfixWindowEngine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  Shape image_shape() const;   // une image de travail, et la sortie
  Shape packed_shape() const;  // un tampon gardé

  // Entrées dans l'ordre de input_names : les sept images puis les sept tampons gardés.
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);

  std::string take_notice() { return graphs_.take_notice(); }

  // --tfix-native : les blocs et la tête passent aux noyaux natifs si une fenêtre de test (sept images qui
  // bougent et scintillent) tient min_db contre TensorRT, sur la sortie, Y2 et la sortie de la tête (sommes
  // f16, mixtes, puis f32). pieces : les blocs et la tête (l'encodeur vide). Rend la ligne du journal ; un
  // refus laisse TensorRT.
  std::expected<std::string, Refusal> try_native(const std::vector<RifeNativePiece>& pieces,
                                                 const std::shared_ptr<SharedActivations>& shared,
                                                 double min_db);
  bool native() const { return native_on_; }

 private:
  TfixWindowEngine() = default;
  // Les noyaux natifs (native_ chargé) ou TensorRT : la disposition des tampons et les activations changent.
  std::expected<void, Refusal> use_native(bool on, const std::shared_ptr<SharedActivations>& shared);
  // L'essai : trois passes (directe, capture, rejeu), puis la sortie, Y2 et la tête copiés vers l'hôte.
  std::expected<std::vector<float>, Refusal> trial_output(cudaStream_t stream,
                                                          const std::shared_ptr<SharedActivations>& shared,
                                                          std::span<const Tensor> inputs, const Tensor& out);
  std::expected<void, Refusal> run(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena);
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);
  std::expected<void, Refusal> follow_zone(const std::shared_ptr<SharedActivations>& arena);
  std::size_t scratch_offset_for(SharedActivations& zone);

  std::array<std::unique_ptr<TrtEngine>, kernels::kTfixLevels> blocks_;
  std::unique_ptr<TrtEngine> conf_;
  int height_ = 0;  // le réseau
  int width_ = 0;
  bool half_ = false;
  kernels::TfixLayout layout_;
  std::unique_ptr<TfixNativeNet> native_;  // actif à la place de blocks_ et conf_ (chargés) si native_on_
  bool native_on_ = false;
  std::int64_t trt_activation_ = 0;  // les activations des moteurs TensorRT
  std::size_t scratch_offset_ = 0;
  std::int64_t own_activation_ = 0;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_{"output"};
  std::shared_ptr<SharedActivations> zone_;
  std::uint64_t zone_generation_ = 0;
  // Un graphe par zone : les adresses de la fenêtre passent par la table.
  static constexpr std::size_t kMaxGraphs = 4;
  FusedGraphs graphs_{kMaxGraphs, "TemporalFix fusionné"};
};

}  // namespace kirari::gpu::cuda
