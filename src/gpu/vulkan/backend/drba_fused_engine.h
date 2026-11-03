// DistilDRBA fusionné sur le backend Vulkan (kernels/spec/drba_fused.h, D7 de l'audit) : les quatre petits
// réseaux du découpage (l'encodeur en lot de trois images, et les blocs 0 à 2, chacun un NetEngine) reliés
// par les noyaux de colle GLSL (kernels/glsl/drba_fused_*.comp, portage des noyaux CUDA du backend TensorRT).
// L'interface est celle du moteur DistilDRBA entier : deux entrées « pair01 » et « pair12 » de 1 × 7 × H × W
// et une sortie de 1 × 3 × H × W ; ou, variante mouvement, deux sorties : le flot 1 × 4 et le masque 1 × 1 du
// warp final. Les tampons intermédiaires (X_k, Y_k) vivent dans un tampon du moteur (drba_fused_layout).
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
#include "kernels/spec/drba_fused.h"

namespace ajp::gpu::vk {

// La clé d'un Artifact de ce moteur : « drba-fused-vk:<h>x<w>:<t>:<artefact0>|<artefact1>… », les artefacts
// des morceaux (chemins complets) dans l'ordre encodeur, blocs 0 à 2.
struct DrbaFusedKey {
  int height = 0;
  int width = 0;
  kernels::DrbaFusedPlan plan;
  std::vector<std::string> artifacts;
  friend bool operator==(const DrbaFusedKey&, const DrbaFusedKey&) = default;
};
std::string drba_fused_artifact_key(const DrbaFusedKey& key);
// Rien si la clé est celle d'un autre moteur ou illisible.
std::optional<DrbaFusedKey> drba_fused_artifact_key_parse(std::string_view text);

class DrbaFusedEngine final : public Engine {
 public:
  // Les morceaux chargés (encodeur, puis un par bloc) : vérifie leurs entrées et sorties contre le
  // découpage et alloue les tampons intermédiaires. Refus : InputMismatch, ceux de l'allocation.
  static std::expected<std::unique_ptr<DrbaFusedEngine>, Refusal> create(
      VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, const DrbaFusedKey& key,
      std::vector<std::unique_ptr<VulkanEngine>> pieces);
  ~DrbaFusedEngine() override;
  DrbaFusedEngine(const DrbaFusedEngine&) = delete;
  DrbaFusedEngine& operator=(const DrbaFusedEngine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  int height() const { return height_; }
  int width() const { return width_; }

  // Soumet la chaîne sur queue : pre, encodeur, post, bloc 0, colle, bloc 1, colle, bloc 2, fin. Une sortie
  // : l'image ; deux : le flot et le masque. Les enregistrements de la colle sont faits au premier passage
  // de ces tenseurs, puis rejoués. Refus : InputMismatch, ceux des enregistrements et de la file.
  std::expected<void, Refusal> enqueue(BackendQueue& queue, std::span<const Tensor> inputs,
                                       std::span<const Tensor> outputs);

 private:
  DrbaFusedEngine(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines)
      : context_(context), memory_(memory), pipelines_(pipelines) {}
  // Les enregistrements de colle pour ces vues : pre, post, colles des blocs 1 et 2, fin.
  struct Glue {
    std::array<BufferView, 4> io{};  // pair01, pair12, sortie (ou flot), masque
    std::vector<std::unique_ptr<Recording>> steps;
  };
  std::expected<Glue*, Refusal> glue_for(std::span<const BufferView, 4> io, bool motion);
  BufferView scratch(std::size_t offset, std::int64_t elements) const;

  VkContext& context_;
  MemoryAllocator& memory_;
  PipelineCache& pipelines_;
  std::vector<std::unique_ptr<VulkanEngine>> pieces_;
  kernels::DrbaFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  kernels::DrbaFusedLayout layout_;
  std::optional<Buffer> scratch_;
  std::vector<std::string> inputs_{"pair01", "pair12"};
  std::vector<std::string> outputs_{"out"};
  static constexpr std::size_t kMaxGlues = 16;
  std::vector<std::unique_ptr<Glue>> glues_;
  RetiredRecordings retired_;
};

}  // namespace ajp::gpu::vk
