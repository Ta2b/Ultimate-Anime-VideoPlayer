// Le SPAN entier en FP8 (noyaux CUTLASS, span_fp8_net.h) vu comme gpu::Engine : le backend TensorRT le charge
// à la place d'un moteur TensorRT quand la variante du modèle est un paquet « .span8 »
// (docs/span-fp8-cutlass.md). Les noyaux tournent sur le flux du backend ; leurs tampons intermédiaires
// vivent dans la zone d'activation de la voie (SharedActivations), comme ceux d'un moteur TensorRT. Étape 3.3
// : le corps du réseau est capturé en graphe CUDA par adresse d'entrée (kMaxGraphs au plus), la queue
// (to_pixel, DepthToSpace fondu depuis l'étape 3.4a) lancée en direct avec l'adresse de sortie : un
// cudaGraphLaunch et un lancement par image.
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
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/backend.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_fp8_view.h"
#include "gpu/cuda/backend/trt_engine.h"
#include "gpu/graph_book.h"
#include "gpu/span_fp8/package.h"

namespace kirari::gpu::cuda {

// La clé d'un Artifact de ce moteur : « span-fp8:<CRC-32 du paquet>:<hauteur>x<largeur> ». La taille voyage
// dans la clé parce qu'Artifact ne porte que la clé et le chemin, et que le moteur n'existe qu'à une taille.
std::string span_fp8_artifact_key(std::uint32_t crc, int height, int width);
// Hauteur et largeur d'une telle clé ; rien si la clé est celle d'un autre moteur.
std::optional<Shape> span_fp8_artifact_shape(std::string_view key);
// Le CRC-32 du paquet que la clé couvre ; rien si la clé est celle d'un autre moteur.
std::optional<std::uint32_t> span_fp8_artifact_crc(std::string_view key);
// Le fichier d'un paquet, entier. Refus FileUnreadable.
std::expected<std::vector<std::byte>, Refusal> read_span_fp8_package(const std::filesystem::path& path);
// L'extension d'un paquet : « .span8 ».
bool is_span_fp8_package(const std::filesystem::path& path);

class SpanFp8Engine final : public Engine {
 public:
  // Lit le paquet, vérifie qu'il est celui de la clé (crc, B02-06), envoie les poids, fait passer une image
  // de zéros par les noyaux (un noyau qui ne tourne pas sur cet appareil se voit ici, au chargement, pas en
  // pleine lecture), puis prend sa zone de travail dans shared. Refus : fichier ou paquet illisible, paquet
  // réécrit depuis la construction (GraphInvalid), taille ou appareil que les noyaux ne prennent pas,
  // mémoire. native_edges : les bords du FP8 par les noyaux du FP16 natif (SpanFp8Weights::native_edges).
  static std::expected<std::unique_ptr<SpanFp8Engine>, Refusal> load(
      const std::filesystem::path& package, std::uint32_t crc, int height, int width,
      const std::shared_ptr<SharedActivations>& shared, bool native_edges = false);
  ~SpanFp8Engine() override;
  SpanFp8Engine(const SpanFp8Engine&) = delete;
  SpanFp8Engine& operator=(const SpanFp8Engine&) = delete;

  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  Shape input_shape() const { return input_; }
  Shape output_shape() const { return output_; }

  // Pose les adresses et lance les noyaux sur stream. arena : la zone de la voie de stream.
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
                                       std::span<const Tensor> inputs, std::span<const Tensor> outputs);
  // enqueue en deux temps (étape 3.4b) : enqueue_body vérifie les tenseurs, pose la zone et lance le corps ;
  // enqueue_tail lance to_pixel, qui écrit outputs[0] et, si ring porte une surface, l'emplacement de
  // l'anneau. Entre les deux, l'appelant peut attendre l'emplacement sur stream. ring : la queue écrira
  // l'anneau, les tuiles demandées (next_span_tiles) sont abandonnées et l'image se calcule entière.
  std::expected<void, Refusal> enqueue_body(cudaStream_t stream,
                                            const std::shared_ptr<SharedActivations>& arena,
                                            std::span<const Tensor> inputs, std::span<const Tensor> outputs,
                                            bool ring = false);

  // Tuiles qui bougent (gpu::Engine, span_fp8_tiles.cpp) : seul le SPAN FP16 natif les calcule.
  kernels::SpanTileGrid span_tile_grid() const override;
  bool next_span_tiles(std::span<const std::uint8_t> core, int shift_x, int shift_y,
                       const Tensor* previous) override;
  std::expected<void, Refusal> enqueue_tail(cudaStream_t stream, std::span<const Tensor> outputs,
                                            const SpanFp8Ring& ring);

  // Le mot du moteur pour le journal (accumulateur f16 débordé, graphes coupés, plus de place), une fois ;
  // vide sinon.
  std::string take_notice() {
    return overflow_notice_.empty() ? book_.take_notice() : std::exchange(overflow_notice_, {});
  }
  const GraphCounters& graph_counters() const { return book_.counters(); }
  // SPAN FP16 natif : le PSNR de la carte contre le calcul du processeur au chargement (span_fp8_check.h) ;
  // vide pour un paquet FP8, ou une image trop petite pour en juger.
  std::optional<double> native_check_db() const { return native_check_db_; }
  // Le paquet chargé est le SPAN FP16 natif (drapeau fp16), pas le FP8 ; native_edges : les bords du FP8 par
  // les noyaux natifs (--span-native). Pour le journal.
  bool fp16() const { return fp16_; }
  bool native_edges() const { return native_edges_; }

 private:
  SpanFp8Engine() = default;
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);
  // Le corps sur stream : rejoué, capturé ou lancé en direct selon le registre.
  std::expected<void, Refusal> body(cudaStream_t stream, const void* source);
  void drop_graphs();
  // Tuiles qui bougent : la demande prise pour cet appel, décalage de la sortie d'avant et masques envoyés.
  // Vrai : le corps se lance masqué, en direct (un graphe capturé est celui de l'image entière).
  std::expected<bool, Refusal> start_tiles(cudaStream_t stream, std::uintptr_t output);
  bool allocate_tiles();
  void end_tiles();
  void release_tiles();

  struct TileJob {
    std::vector<std::uint8_t> core;
    int shift_x = 0;  // pixels d'entrée
    int shift_y = 0;
    std::uintptr_t previous = 0;  // adresse de la sortie d'avant ; 0 : déjà dans la sortie
  };
  static constexpr int kTileStages = 4;  // envois des masques en vol au plus
  std::optional<TileJob> tile_job_;
  std::uint32_t* tile_masks_ = nullptr;  // appareil : région puis cœur, kSpanTileMaskWords mots chacun
  std::array<std::uint32_t*, kTileStages> tile_staging_{};  // hôte, épinglée
  std::array<cudaEvent_t, kTileStages> tile_sent_{};
  int tile_stage_ = 0;

  static constexpr std::size_t kMaxGraphs = 4;
  static constexpr int kCaptureAttempts = 3;
  GraphBook<std::uintptr_t> book_{kMaxGraphs, kCaptureAttempts};
  std::vector<cudaGraphExec_t> execs_;  // rangés par l'indice du registre
  std::optional<double> native_check_db_;
  bool fp16_ = false;
  bool native_edges_ = false;
  std::string overflow_notice_;

  std::unique_ptr<SpanFp8Net> net_;
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_{"output"};
  Shape input_;
  Shape output_;
  std::shared_ptr<SharedActivations> zone_;  // la zone où les noyaux sont posés
  std::uint64_t generation_ = 0;
};

}  // namespace kirari::gpu::cuda
