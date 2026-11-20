// La paire de GMFSS Fortuna Union vue comme un gpu::Engine (--gmfss-native, docs/gmfss-natif.md), avec
// l'entrée et les sorties du moteur TensorRT de la paire entière qu'elle remplace : l'étage
// (upscale_gmfss.cpp) ne voit pas la différence. Natif : le moteur du « flot » (GMFlow et ce qui prépare
// l'entrée du MetricNet, prepare/gmfss_pair_split.h) écrit half0, half1, les flots (les deux premiers plans
// de chaque motion{k}_{l}) et l'entrée du MetricNet ; puis le FeatureNet de chaque image et le MetricNet par
// les noyaux de gmfss_pair_native_net.h. La paire entière reste chargée : référence de l'essai au
// chargement, et repli quand le natif est écarté ou que l'essai n'a pas été demandé.
#pragma once

#include <array>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "gpu/backend.h"
#include "gpu/cuda/backend/fused_keys.h"
#include "gpu/cuda/backend/gmfss_pair_native_net.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "prepare/gmfss_pair_split.h"

namespace nvinfer1 {
class IRuntime;
}

namespace kirari::gpu::cuda {

class SharedActivations;
class GmflowPairFlowNet;

class GmfssPairEngine final : public Engine {
 public:
  // Les deux moteurs du cache, de formes vérifiées (la fusion à key.height × key.width, les images au
  // double) ; cut : les noms des sorties du flot, relus dans le manifeste.
  static std::expected<std::unique_ptr<GmfssPairEngine>, Refusal> load(
      nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const GmfssFusedKey& key,
      const prepare::GmfssPairCut& cut, const std::shared_ptr<SharedActivations>& shared, bool feature_reuse);
  ~GmfssPairEngine() override;
  GmfssPairEngine(const GmfssPairEngine&) = delete;
  GmfssPairEngine& operator=(const GmfssPairEngine&) = delete;

  std::span<const std::string> input_names() const override { return full_->input_names(); }
  std::span<const std::string> output_names() const override { return full_->output_names(); }
  bool next_gmfss_pair(bool follows) override {
    reuse_next_ = follows && feature_reuse_ && native_on_ && right_features_.front().handle != 0;
    return reuse_next_;
  }
  std::span<const Tensor> take_right_features();
  bool feature_already_bound(std::size_t output, const Tensor& tensor, std::span<const Tensor> reuse) const;

  // frames (1 × 6 × 2h × 2w), les quatorze sorties dans l'ordre d'output_names().
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs,
                                       std::span<const Tensor> reuse = {});

  // L'essai du natif (gmfss_pair_trial.cpp) : le FeatureNet et le MetricNet natifs réglés au banc, puis deux
  // images de test passent par la paire entière et par le flot et les réseaux natifs, les sommes choisies
  // réseau par réseau (gmfss_trial.h) ; le natif n'est gardé que si chacune des quatorze sorties tient min_db
  // contre TensorRT (rapporté à l'étendue de la sortie de TensorRT) et s'il le bat d'au moins 2 % en temps.
  // Les lignes du journal, la dernière disant ce qui a été retenu ; refus : l'essai n'a pas pu tourner (la
  // paire entière reste).
  std::expected<std::vector<std::string>, Refusal> try_native(
      const prepare::GmfssPairNets& nets, std::span<const std::byte> pair,
      const std::shared_ptr<SharedActivations>& shared, double min_db, bool fast = false);
  bool native() const { return native_on_; }

  // Les sorties de la paire, dans l'ordre de kOutputNames (gmfss_pair_engine.cpp) : half0, half1, puis par
  // côté feature{k}_1, source{k}_2, source{k}_3, enfin motion{k}_{l}, rangés k · 3 + l - 1.
  enum Output : std::size_t {
    kHalf0,
    kHalf1,
    kFeature0,  // feature0_1, source0_2, source0_3
    kFeature1 = kFeature0 + 3,
    kMotion = kFeature1 + 3,  // motion0_1 … motion1_3
    kOutputs = kMotion + 6,
  };

 private:
  GmfssPairEngine() = default;
  std::expected<void, Refusal> enqueue_native(cudaStream_t stream,
                                              const std::shared_ptr<SharedActivations>& shared,
                                              const Tensor& frames, std::span<const Tensor> outputs,
                                              const GmflowPairFlowNet* gmflow,
                                              std::span<const Tensor> reuse = {});
  std::expected<void, Refusal> enqueue_flow(cudaStream_t stream,
                                            const std::shared_ptr<SharedActivations>& shared,
                                            const Tensor& frames, std::span<const Tensor> outputs,
                                            const GmflowPairFlowNet* gmflow, bool follows) const;
  std::expected<std::string, Refusal> try_flow_native(std::span<const std::byte> pair, cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& shared,
                                                      const Tensor& frames, std::span<const Tensor> outputs,
                                                      double min_db, bool fast);
  // Un réseau natif (GmfssPairNativeNet::kFeatureNet, ses deux images ; kMetricNet) sur ce que le flot a
  // écrit : la fin de enqueue_native, et le temps d'un morceau à l'essai.
  std::expected<void, Refusal> enqueue_piece(cudaStream_t stream, int piece, const Tensor& frames,
                                             std::span<const Tensor> outputs,
                                             std::span<const Tensor> reuse = {}) const;

  // Les sorties du flot, dans l'ordre de la coupe : l'entrée du MetricNet, les six flots, half0, half1.
  static constexpr std::size_t kFlowOutputs = 1 + prepare::kGmfssPairFlows + 2;
  std::unique_ptr<TrtEngine> full_;
  std::unique_ptr<TrtEngine> flow_;
  std::unique_ptr<GmfssPairNativeNet> native_;
  std::unique_ptr<GmflowPairFlowNet> gmflow_;
  bool native_on_ = false;
  bool feature_reuse_ = false;
  bool reuse_next_ = false;
  std::array<Tensor, prepare::kGmfssFeatureLevels> right_features_{};
  int height_ = 0;
  int width_ = 0;
  std::array<std::size_t, kOutputs> at_{};           // la place de chaque sortie dans output_names()
  std::array<std::size_t, kFlowOutputs> flow_at_{};  // la place de chaque sortie du flot dans les siennes
  void* metric_in_ = nullptr;                        // l'entrée du MetricNet, 14 plans FP16
};

}  // namespace kirari::gpu::cuda
