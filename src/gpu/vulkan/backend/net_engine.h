// Réseau chargé sur l'appareil : pipelines créés, poids envoyés une fois, arène des activations
// allouée selon le plan mémoire de gpu_ir ; un tenseur rangé dans un autre (net_alias.h) est lié
// par une vue dans sa racine. Un enregistrement par jeu de tenseurs d'entrée et
// de sortie, fait au premier passage puis rejoué tel quel : rien n'est alloué ni recalculé par
// image. Au-delà de kMaxRecordings jeux, le plus ancien sort du cache et se libère à la fin de sa
// dernière exécution, sans attente hôte (RetiredRecordings).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/ir/memory_plan.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::gpu::vk::net {

class NetEngine {
 public:
  // SPAN en régime : deux entrées RGB × deux sorties en alternance, plus la sortie finale et
  // l'image du milieu : jusqu'à 7 jeux ; le double pour la lecture anticipée et l'export.
  static constexpr std::size_t kMaxRecordings = 16;

  // Refus : FeatureUnavailable (shader absent), ceux du plan mémoire, de l'allocateur et du
  // cache de pipelines.
  static std::expected<std::unique_ptr<NetEngine>, Refusal> create(VkContext& context,
                                                                   MemoryAllocator& memory,
                                                                   PipelineCache& pipelines, ir::Graph graph,
                                                                   Program program);

  NetEngine(const NetEngine&) = delete;
  NetEngine& operator=(const NetEngine&) = delete;
  NetEngine(NetEngine&&) = delete;
  NetEngine& operator=(NetEngine&&) = delete;
  ~NetEngine() = default;

  const ir::Graph& graph() const { return graph_; }
  // Le programme sans ses blobs, envoyés à l'appareil au chargement.
  const Program& program() const { return program_; }
  std::span<const std::string> input_names() const { return input_names_; }
  std::span<const std::string> output_names() const { return output_names_; }
  // Octets d'un tenseur d'entrée ou de sortie dans sa disposition (Nchw, son type).
  std::uint64_t input_bytes(std::size_t k) const;
  std::uint64_t output_bytes(std::size_t k) const;
  std::uint64_t arena_bytes() const { return plan_.arena_bytes; }

  // L'enregistrement de ces tenseurs, dans l'ordre de input_names() et output_names(), créé à
  // la première demande. Refus InputMismatch : nombre de vues ou vue trop petite.
  std::expected<Recording*, Refusal> recording(std::span<const BufferView> inputs,
                                               std::span<const BufferView> outputs);
  std::size_t recordings() const { return recordings_.size(); }  // jeux de tenseurs enregistrés

 private:
  struct Cached {
    std::vector<BufferView> views;
    std::unique_ptr<Recording> recording;
    std::uint64_t used = 0;
  };

  NetEngine(VkContext& context, MemoryAllocator& memory, ir::Graph graph, Program program)
      : context_(context),
        memory_(memory),
        graph_(std::move(graph)),
        program_(std::move(program)),
        destroyed_mark_(context.destroyed().mark()) {}
  std::expected<void, Refusal> prepare(PipelineCache& pipelines);
  std::expected<BufferView, Refusal> resolve(const Binding& binding, std::span<const BufferView> inputs,
                                             std::span<const BufferView> outputs) const;
  // La place d'un tenseur qui n'est rangé dans aucun autre.
  std::expected<BufferView, Refusal> root_view(ir::TensorId id, std::span<const BufferView> inputs,
                                               std::span<const BufferView> outputs) const;
  std::expected<std::unique_ptr<Recording>, Refusal> record(std::span<const BufferView> inputs,
                                                            std::span<const BufferView> outputs);
  // Retire les jeux dont un tampon a été détruit depuis la dernière recherche : sa poignée peut
  // déjà désigner un autre tampon (destroyed_log.h).
  void forget_destroyed();

  VkContext& context_;
  MemoryAllocator& memory_;
  ir::Graph graph_;
  Program program_;
  ir::MemoryPlan plan_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const Pipeline*> pipelines_;             // un par dispatch
  std::vector<std::optional<ir::Allocation>> places_;  // par tenseur : sa place dans l'arène
  std::vector<VkDeviceSize> blob_offsets_;
  std::vector<VkDeviceSize> blob_sizes_;
  std::optional<Buffer> arena_;
  std::optional<Buffer> blobs_;
  std::optional<Buffer> dummy_;
  std::vector<Cached> recordings_;
  RetiredRecordings retired_;  // sortis du cache, libérés à leur arrêt (avant l'arène)
  std::uint64_t uses_ = 0;
  std::uint64_t destroyed_mark_ = 0;  // DestroyedLog::mark à la dernière purge
  std::vector<std::uint64_t> gone_;
};

}  // namespace ajp::gpu::vk::net
