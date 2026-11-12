// RIFE fusionné sur le backend Vulkan (kernels/spec/rife_fused.h) : les six petits réseaux du découpage
// (l'encodeur et les blocs, des convolutions à basse résolution, chacun un NetEngine) reliés par les noyaux
// de colle GLSL (kernels/glsl/rife_fused_*.comp, portage des noyaux CUDA du backend TensorRT). L'interface
// est celle du RIFE entier : une entrée 1 × 11 × H × W (rife_input) et une sortie 1 × 3 × H × W, ou, pour la
// variante mouvement, le flot 1 × 4 × H × W et le masque 1 × 1 × H × W. Les tampons intermédiaires (encodage,
// X_k et Y_k) vivent dans un tampon du moteur, rangé par rife_fused_layout. Ce qui travaille en pleine
// résolution ne passe plus par les opérateurs du graphe un à un (Resize, Split, Mul, Add, GridSample,
// Concat…) : trois noyaux par paire au lieu d'une vingtaine de dispatchs. L'encodeur prend une image à la
// fois : l'encodage de la paire, gardé dans le tampon du moteur, sert encore à un autre instant de la même
// paire (next_times_only), et celui de B devient celui de A à la paire suivante (next_pair_follows).
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
// rife-fused-vk:<h>x<w>:<image|motion|image-cubic>:<d0,d1…>:<artefact0>|<artefact1>… », les artefacts des
// morceaux (chemins complets) dans l'ordre encodeur, blocs 0 à N-1.
struct RifeFusedKey {
  int height = 0;
  int width = 0;
  bool motion = false;
  kernels::RifeWarp warp = kernels::RifeWarp::Bilinear;  // le warp de l'image (sortie « image-cubic »)
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
// warp : celui de l'image (ignoré si motion).
Done record_rife_fused_finish(PipelineCache& pipelines, Recorder& recorder,
                              const kernels::RifeFusedPlan& plan, int height, int width, bool motion,
                              const BufferView& input, const BufferView& x_last, const BufferView& y_last,
                              const BufferView& first, const BufferView& mask,
                              kernels::RifeWarp warp = kernels::RifeWarp::Bilinear);

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
  // Les blocs de la réduction la plus forte (8 : deux pour 4.25, un pour 4.22 lite) sont repris de l'appel
  // précédent, leurs sorties Y étant encore dans le tampon du moteur ; seulement après un appel complet ou
  // chaud.
  bool next_warm() override {
    warm_next_ = warm_ready_ && warm_blocks_ > 0;
    return warm_next_;
  }
  // Le prochain enqueue reprend la même entrée, seul l'instant a changé : l'encodeur est sauté.
  void next_times_only() override { hint_ = Hint::TimesOnly; }
  // Le prochain enqueue prend pour A la B de l'appel précédent : son encodage est copié à la place de A,
  // seule la nouvelle B passe dans l'encodeur.
  void next_pair_follows() override { hint_ = Hint::Follows; }
  // L'annonce oubliée : le backend a refusé l'appel avant d'enfiler la chaîne.
  void drop_hint() { hint_ = Hint::None; }
  int height() const { return height_; }
  int width() const { return width_; }
  // Les passes de l'encodeur depuis la création, une image chacune (tests : deux par paire à froid, une quand
  // la paire suit, aucune pour un autre instant).
  std::uint64_t encodes() const { return encodes_; }

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
  // L'encodage de B copié à la place de celui de A, enregistré une fois.
  std::expected<Recording*, Refusal> shift_encoding();
  enum class Hint : std::uint8_t { None, TimesOnly, Follows };
  // L'encodage de la paire de input selon l'annonce et ce que le dernier appel mené au bout a encodé
  // (encoded_from, son entrée) : rien, B seule, ou A et B.
  Done encode_pair(BackendQueue& queue, const BufferView& input, Hint hint,
                   const std::optional<BufferView>& encoded_from);
  static Done submit_net(BackendQueue& queue, VulkanEngine& piece, const BufferView& in,
                         const BufferView& out);

  VkContext& context_;
  MemoryAllocator& memory_;
  PipelineCache& pipelines_;
  std::vector<std::unique_ptr<VulkanEngine>> pieces_;
  kernels::RifeFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  bool motion_ = false;
  kernels::RifeWarp warp_ = kernels::RifeWarp::Bilinear;
  int warm_blocks_ = 0;      // blocs repris à chaud (next_warm)
  bool warm_ready_ = false;  // les Y de ces blocs sont ceux de l'appel précédent
  bool warm_next_ = false;
  Hint hint_ = Hint::None;
  // L'entrée du dernier appel mené au bout, dont l'encodage est dans le tampon du moteur ; nul sinon.
  std::optional<BufferView> encoded_from_;
  std::uint64_t encodes_ = 0;  // passes de l'encodeur, une image chacune
  kernels::RifeFusedLayout layout_;
  std::optional<Buffer> scratch_;
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_;
  static constexpr std::size_t kMaxGlues = 16;
  std::vector<std::unique_ptr<Glue>> glues_;
  RetiredRecordings retired_;
  std::unique_ptr<Recording> shift_;  // détruit le premier : il attend sa fin avant que le tampon parte
};

}  // namespace ajp::gpu::vk
