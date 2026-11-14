// RIFE fusionné vu comme gpu::Engine (kernels/spec/rife_fused.h) : six petits moteurs TensorRT (l'encodeur et
// les blocs, des convolutions à basse résolution) reliés par les noyaux CUDA de rife_fused_kernels.h, le tout
// enfilé sur le flux du backend et capturé dans un seul graphe CUDA. L'interface est celle du moteur RIFE
// entier : une entrée 1 × 11 × H × W (rife_input) et une sortie 1 × 3 × H × W, ou, pour la variante
// mouvement, le flot 1 × 4 × H × W et le masque 1 × 1 × H × W. Les tampons intermédiaires vivent dans la zone
// partagée de la voie (SharedActivations), au-dessus des activations de tous les moteurs entiers qui la
// partagent (C03-01 : la zone grandit de leur taille quand SPAN y a les plus grandes activations).
// L'encodeur prend une image à la fois : l'encodage d'une paire se garde d'un appel à l'autre, entier pour un
// autre instant de la même paire (next_times_only), celui de B pour A à la paire suivante
// (next_pair_follows), tant qu'aucun autre moteur n'a écrit sur ces tampons : un SPAN qui passe entre deux
// appels sur la même voie n'écrit que sous eux.
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
#include "kernels/spec/warp.h"

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
  // P6.15 : le prochain enqueue reprend les Y des blocs grossiers (ceux à la réduction du premier) de l'appel
  // précédent, si cet appel est allé au bout et que ses tampons sont intacts (comme le backend Vulkan).
  bool next_warm() override {
    warm_next_ = warm_ready_ && warm_blocks_ > 0;
    return warm_next_;
  }
  // L'annonce oubliée : le backend a refusé l'appel avant d'enfiler la chaîne.
  void drop_hint() {
    hint_ = Hint::None;
    warm_next_ = false;
  }
  // La fenêtre de l'image source dans le tenseur rempli (kernels::rife_window), que le backend connaît par
  // rife_input : le warp final de la variante image n'en sort pas (C03-03). Hors du tenseur : le tenseur.
  void set_window(const kernels::Window& window);
  // Les appels menés jusqu'à l'encodeur depuis le chargement : les deux images encodées, B seule, aucune (les
  // tests de l'encodage gardé, C03-09).
  struct EncodeCounts {
    std::uint64_t both = 0;
    std::uint64_t second = 0;
    std::uint64_t none = 0;
  };
  EncodeCounts encode_counts() const {
    return {.both = encodes_[0], .second = encodes_[1], .none = encodes_[2]};
  }

  // --l2-keep (essai, éteint par défaut) : les deux emplacements de l'encodage gardé sont tenus en L2 le
  // temps de la chaîne (fenêtre d'accès persistante sur le flux, capturée dans les graphes avec les noyaux) ;
  // persisting : la part de L2 réservée (cudaLimitPersistingL2CacheSize), max_window : la fenêtre la plus
  // grande que l'appareil accepte. 0 : rien n'est posé. Les graphes la portent dans leur clé.
  void keep_in_l2(std::size_t persisting, std::size_t max_window) {
    l2_persisting_ = persisting;
    l2_max_window_ = max_window;
  }

  // Le mot du moteur pour le journal du backend (graphes coupés, registre plein), une fois ; vide sinon.
  std::string take_notice() { return graphs_.take_notice(); }

 private:
  RifeFusedEngine() = default;
  // Ce que l'encodeur fait à cet appel : les deux images, B seule (l'encodage de B d'avant devient celui de A
  // en changeant d'emplacement), rien (l'encodage de l'appel précédent sert tel quel).
  enum class Encode : std::uint8_t { Both, Second, None };
  enum class Hint : std::uint8_t { None, TimesOnly, Follows };
  // Ce que la chaîne refait à cet appel : le travail de l'encodeur, l'emplacement de A, le premier bloc
  // calculé (0, ou warm_blocks_ à chaud).
  struct Pass {
    Encode encode = Encode::Both;
    unsigned a_slot = 0;
    int first_block = 0;
  };
  // La chaîne, lancée ou capturée selon l'appelant.
  std::expected<void, Refusal> run(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                   const Tensor& input, std::span<const Tensor> outputs, const Pass& pass);
  std::expected<void, Refusal> launch_glue(cudaStream_t stream, const void* in, int block, std::byte* base,
                                           std::size_t encode_a, std::size_t encode_b);
  std::expected<void, Refusal> enqueue_chain(cudaStream_t stream,
                                             const std::shared_ptr<SharedActivations>& arena,
                                             std::span<const Tensor> inputs, std::span<const Tensor> outputs,
                                             const Pass& pass);
  // Le travail de l'encodeur à cet appel, d'après l'encodage gardé (kept_) et l'indice de l'appelant.
  Encode choose_encode(const SharedActivations& arena, Hint hint, std::uintptr_t input) const;
  // L'encodage et les Y du dernier appel sont encore dans la zone : même zone, même place, aucun autre moteur
  // n'y a écrit depuis.
  bool kept_intact(const SharedActivations& arena) const;
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);
  std::expected<std::int64_t, Refusal> load_blocks(nvinfer1::IRuntime& runtime,
                                                   const std::filesystem::path& cache_dir,
                                                   const RifeFusedKey& key,
                                                   const std::shared_ptr<SharedActivations>& shared);
  std::size_t scratch_offset_for(SharedActivations& zone);

  std::unique_ptr<TrtEngine> encode_;
  std::vector<std::unique_ptr<TrtEngine>> blocks_;
  kernels::RifeFusedPlan plan_;
  int height_ = 0;
  int width_ = 0;
  bool motion_ = false;
  kernels::RifeWarp warp_ = kernels::RifeWarp::Bilinear;
  kernels::Window window_;  // set_window
  kernels::RifeFusedLayout layout_;
  // Les blocs lisent X et écrivent Y au format HWC8 (TrtEngine::input_pitch) : les noyaux de colle aussi.
  bool hwc8_ = false;
  // Le pas par pixel d'un tenseur de channels canaux pour les noyaux : 0 en plans.
  int pitch(int channels) const { return hwc8_ ? kernels::rife_fused_hwc8_pitch(channels) : 0; }
  // Début des tampons intermédiaires dans la zone : au-dessus des activations de ses morceaux et de celles
  // des moteurs entiers de la même zone (SharedActivations::activation_peak, C03-01).
  std::size_t scratch_offset_ = 0;
  std::int64_t own_activation_ = 0;  // les plus grandes activations de ses morceaux
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_;
  std::shared_ptr<SharedActivations> zone_;

  std::uint64_t zone_generation_ = 0;  // la génération de zone_ à use_zone

  // Graphes par jeu d'adresses (entrée, sortie ou flot, masque), de zone, de travail de l'encodeur,
  // d'emplacement de A, de premier bloc et de fenêtre L2 : jusqu'à cinq par jeu d'adresses à froid (Both,
  // puis Second et None sur chaque emplacement), autant à chaud.
  static constexpr std::size_t kMaxGraphs = 64;
  FusedGraphs graphs_{kMaxGraphs, "RIFE fusionné"};
  // L'encodage que le dernier appel mené jusqu'au bout a laissé dans la zone, et l'entrée qu'il a lue.
  // Repris seulement si la zone est la même et qu'aucun autre moteur n'y a écrit depuis.
  struct Kept {
    std::uintptr_t input = 0;
    const void* zone = nullptr;
    std::uint64_t generation = 0;
    std::size_t offset = 0;  // scratch_offset_ de l'appel
  };
  std::optional<Kept> kept_;
  Hint hint_ = Hint::None;
  unsigned a_slot_ = 0;      // l'emplacement de l'encodage de A gardé (kept_)
  int warm_blocks_ = 0;      // blocs repris à chaud (next_warm)
  bool warm_ready_ = false;  // les Y de ces blocs sont ceux de l'appel précédent, mené au bout
  bool warm_next_ = false;
  std::size_t l2_persisting_ = 0;  // keep_in_l2
  std::size_t l2_max_window_ = 0;
  std::array<std::uint64_t, 3> encodes_{};  // appels par travail de l'encodeur (Encode), pour les tests
};

}  // namespace ajp::gpu::cuda
