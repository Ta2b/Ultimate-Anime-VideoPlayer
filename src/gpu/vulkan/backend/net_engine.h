// Réseau chargé sur l'appareil : pipelines créés, poids envoyés une fois, arène des activations
// allouée selon le plan mémoire de gpu_ir. Un enregistrement par jeu de tenseurs d'entrée et
// de sortie, fait au premier passage puis rejoué tel quel : rien n'est alloué ni recalculé par
// image. Au-delà de kMaxRecordings jeux, le plus ancien est libéré (après la fin de sa dernière
// exécution).
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
  static constexpr std::size_t kMaxRecordings = 8;

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

 private:
  struct Cached {
    std::vector<BufferView> views;
    std::unique_ptr<Recording> recording;
    std::uint64_t used = 0;
  };

  NetEngine(VkContext& context, MemoryAllocator& memory, ir::Graph graph, Program program)
      : context_(context), memory_(memory), graph_(std::move(graph)), program_(std::move(program)) {}
  std::expected<void, Refusal> prepare(PipelineCache& pipelines);
  std::expected<BufferView, Refusal> resolve(const Binding& binding, std::span<const BufferView> inputs,
                                             std::span<const BufferView> outputs) const;
  std::expected<std::unique_ptr<Recording>, Refusal> record(std::span<const BufferView> inputs,
                                                            std::span<const BufferView> outputs);

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
  std::uint64_t uses_ = 0;
};

}  // namespace ajp::gpu::vk::net
