// Le SPAN entier en FP8 (noyaux CUTLASS, span_fp8_net.h) vu comme gpu::Engine : le backend TensorRT le charge
// à la place d'un moteur TensorRT quand la variante du modèle est un paquet « .span8 »
// (docs/span-fp8-cutlass.md). Les noyaux tournent sur le flux du backend, sans capture de graphe propre (rien
// d'alloué à l'exécution : la capture d'un appelant les prend telle quelle) ; leurs tampons intermédiaires
// vivent dans la zone d'activation de la voie (SharedActivations), comme ceux d'un moteur TensorRT.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/backend.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/trt_engine.h"

namespace ajp::gpu::cuda {

// La clé d'un Artifact de ce moteur : « span-fp8:<CRC-32 du paquet>:<hauteur>x<largeur> ». La taille voyage
// dans la clé parce qu'Artifact ne porte que la clé et le chemin, et que le moteur n'existe qu'à une taille.
std::string span_fp8_artifact_key(std::uint32_t crc, int height, int width);
// Hauteur et largeur d'une telle clé ; rien si la clé est celle d'un autre moteur.
std::optional<Shape> span_fp8_artifact_shape(std::string_view key);
// Le fichier d'un paquet, entier. Refus FileUnreadable.
std::expected<std::vector<std::byte>, Refusal> read_span_fp8_package(const std::filesystem::path& path);
// L'extension d'un paquet : « .span8 ».
bool is_span_fp8_package(const std::filesystem::path& path);

class SpanFp8Engine final : public Engine {
 public:
  // Lit le paquet, envoie les poids, prend sa zone de travail dans shared et fait passer une image de zéros
  // par les noyaux (un noyau qui ne tourne pas sur cet appareil se voit ici, au chargement, pas en pleine
  // lecture). Refus : fichier ou paquet illisible, taille ou appareil que les noyaux ne prennent pas,
  // mémoire.
  static std::expected<std::unique_ptr<SpanFp8Engine>, Refusal> load(
      const std::filesystem::path& package, int height, int width, std::shared_ptr<SharedActivations> shared);
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

 private:
  SpanFp8Engine() = default;
  std::expected<void, Refusal> use_zone(const std::shared_ptr<SharedActivations>& zone);

  std::unique_ptr<SpanFp8Net> net_;
  std::vector<std::string> inputs_{"input"};
  std::vector<std::string> outputs_{"output"};
  Shape input_;
  Shape output_;
  std::shared_ptr<SharedActivations> zone_;  // la zone où les noyaux sont posés
  std::uint64_t generation_ = 0;
};

}  // namespace ajp::gpu::cuda
