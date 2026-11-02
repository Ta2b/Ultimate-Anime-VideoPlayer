// RIFE fusionné sur le backend Vulkan (kernels/spec/rife_fused.h) : les six petits réseaux du découpage
// (l'encodeur et les blocs, des convolutions à basse résolution, chacun un NetEngine) reliés par les noyaux
// de colle GLSL (kernels/glsl/rife_fused_*.comp, portage des noyaux CUDA du backend TensorRT). L'interface
// est celle du RIFE entier : une entrée 1 × 11 × H × W (rife_input) et une sortie 1 × 3 × H × W, ou, pour la
// variante mouvement, le flot 1 × 4 × H × W et le masque 1 × 1 × H × W. Les tampons intermédiaires (encodage,
// X_k et Y_k) vivent dans un tampon du moteur, rangé par rife_fused_layout. Ce qui travaille en pleine
// résolution ne passe plus par les opérateurs du graphe un à un (Resize, Split, Mul, Add, GridSample,
// Concat…) : trois noyaux par paire au lieu d'une vingtaine de dispatchs.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/tensor.h"
#include "gpu/vulkan/backend/queue.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/recording.h"
#include "kernels/spec/rife_fused.h"

namespace ajp::gpu::vk {

// La clé d'un Artifact de ce moteur : «
// rife-fused-vk:<h>x<w>:<image|motion>:<d0,d1…>:<artefact0>|<artefact1>… », les artefacts des morceaux
// (chemins complets) dans l'ordre encodeur, blocs 0 à N-1.
struct RifeFusedKey {
  int height = 0;
  int width = 0;
  bool motion = false;
  kernels::RifeFusedPlan plan;
  std::vector<std::string> artifacts;
  friend bool operator==(const RifeFusedKey&, const RifeFusedKey&) = default;
};
std::string rife_fused_artifact_key(const RifeFusedKey& key);
// Rien si la clé est celle d'un autre moteur ou illisible.
std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text);

// Les trois noyaux de colle, enregistrés tels quels (le moteur ; les tests contre kernels/ref/rife_fused.h).
// Tenseurs FP16 aux formes du découpage : entrée 11 plans, encodage 8 plans, X_k, Y_k (y : Y_0 à Y_{k-1}).
// first : l'image (3 plans), ou le flot (4 plans) avec mask (1 plan) si motion.
Done record_rife_fused_entry(PipelineCache& pipelines, Recorder& recorder, const kernels::RifeFusedPlan& plan,
                             int height, int width, const BufferView& input, const BufferView& encode,
                             const BufferView& x0);
Done record_rife_fused_glue(PipelineCache& pipelines, Recorder& recorder, const kernels::RifeFusedPlan& plan,
                            int height, int width, int k, const BufferView& input, const BufferView& encode,
                            const BufferView& xk, std::span<const BufferView> y);
Done record_rife_fused_finish(PipelineCache& pipelines, Recorder& recorder,
                              const kernels::RifeFusedPlan& plan, int height, int width, bool motion,
                              const BufferView& input, const BufferView& x_last, const BufferView& y_last,
                              const BufferView& first, const BufferView& mask);

class RifeFusedEngine final : public Engine {
 public:
  // Les morceaux chargés (encodeur, puis un par bloc) : vérifie leurs entrées et sorties contre le
  // découpage et alloue les tampons intermédiaires. Refus : InputMismatch (morceau de forme inattendue),
  // ceux de l'allocation.
  static std::expected<std::unique_ptr<RifeFusedEngine>, Refusal> create(
      VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, const RifeFusedKey& key,
      std::vector<std::unique_ptr<VulkanEngine>> pieces);
  ~RifeFusedEngine() override;
  RifeFusedEngine(const RifeFusedEngine&) = delete;
  RifeFusedEngine& operator=(const RifeFusedEngine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  bool motion() const { return motion_; }
  int height() const { return height_; }
  int width() const { return width_; }

  // Soumet la chaîne sur queue, dans l'ordre : encodeur, entrée du bloc 0, bloc 0, colle, bloc 1… puis la
  // fin. Les enregistrements de la colle sont faits au premier passage de ces tenseurs, puis rejoués.
  // Refus : InputMismatch (tenseur de type ou de forme inattendus), ceux des enregistrements et de la file.
  std::expected<void, Refusal> enqueue(BackendQueue& queue, std::span<const Tensor> inputs,
                                       std::span<const Tensor> outputs);

 private:
  RifeFusedEngine(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines)
      : context_(context), memory_(memory), pipelines_(pipelines) {}
  // Les enregistrements de colle pour ces vues : entrée du bloc 0, colle des blocs 1 à N-1, fin.
  struct Glue {
    std::array<BufferView, 3> io{};  // entrée, sortie (image ou flot), masque
    std::vector<std::unique_ptr<Recording>> steps;
  };
  std::expected<Glue*, Refusal> glue_for(const BufferView& input, std::span<const BufferView> outputs);
  std::expected<std::unique_ptr<Recording>, Refusal> record_entry(const BufferView& input);
  std::expected<std::unique_ptr<Recording>, Refusal> record_glue(const BufferView& input, int k);
  std::expected<std::unique_ptr<Recording>, Refusal> record_finish(const BufferView& input,
                                                                   std::span<const BufferView> outputs);
  BufferView scratch(std::size_t offset, std::int64_t elements) const;

  VkContext& context_;
  MemoryAllocator& memory_;
  PipelineCache& pipelines_;
  std::vector<std::unique_ptr<VulkanEngine>> pieces_;
  kernels::RifeFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  bool motion_ = false;
  kernels::RifeFusedLayout layout_;
  std::optional<Buffer> scratch_;
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_;
  static constexpr std::size_t kMaxGlues = 16;
  std::vector<std::unique_ptr<Glue>> glues_;
  RetiredRecordings retired_;
};

}  // namespace ajp::gpu::vk
