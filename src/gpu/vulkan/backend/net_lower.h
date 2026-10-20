// Abaissement d'un graphe gpu_ir optimisé vers les noyaux GLSL du moteur Vulkan : un choix de
// noyau par nœud, puis les dispatchs et les blobs du programme (net_program.h). Un opérateur
// sans noyau, une forme hors des noyaux ou une précision que l'appareil ne tient pas sont
// refusés ici, à la construction, jamais à l'exécution.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_params.h"
#include "gpu/vulkan/backend/net_program.h"

namespace ajp::gpu::vk::net {

// Version de l'abaissement : change à chaque changement des shaders, des constantes ou du
// rangement des poids ; un artefact d'une autre version est refusé au chargement.
inline constexpr std::uint32_t kLoweringVersion = 1;

// Variantes candidates d'un nœud, la première étant celle par défaut ; une seule pour les
// opérations sans réglage. Refus OperatorUnknown : opérateur sans noyau Vulkan.
std::expected<std::vector<KernelChoice>, Refusal> candidates(const ir::Graph& graph, std::size_t node,
                                                             const Target& target);

// Construit un programme nœud par nœud ; sert à lower() et à l'autotuner, qui chronomètre un
// nœud seul.
class Lowerer {
 public:
  Lowerer(const ir::Graph& graph, const Target& target) : graph_(graph), target_(target) {}

  // Ajoute les dispatchs et les blobs du nœud. Refus : OperatorUnknown (opérateur sans
  // noyau), InputMismatch (choix inconnu pour cet opérateur), ShapeInvalid (forme ou
  // paramètre hors des noyaux), FeatureUnavailable (float16 sans stockage 16 bits, fp16 de
  // calcul ou matrices coopératives absents).
  std::expected<void, Refusal> node(std::size_t index, const KernelChoice& choice);

  Program take() { return std::move(program_); }
  const Program& program() const { return program_; }

 private:
  // Un tenseur lié : sa liaison, sa vue de rang 4 et son type (0 float, 1 float16).
  struct Bound {
    Binding binding;
    Desc4 desc;
    std::uint32_t type = 0;
    const ir::Tensor* tensor = nullptr;
  };

  std::expected<Bound, Refusal> bind(ir::TensorId id);
  std::uint32_t add_blob(std::vector<std::byte> bytes);
  std::expected<std::string, Refusal> shader(std::string_view base, std::span<const Bound> tensors) const;
  void emit(std::size_t node, std::string shader, std::vector<std::uint32_t> constants,
            std::vector<Binding> bindings, std::vector<std::uint32_t> push,
            std::array<std::uint32_t, 3> groups);
  void emit_linear(std::size_t node, std::string shader, std::vector<std::uint32_t> constants,
                   std::vector<Binding> bindings, std::vector<std::uint32_t> push, std::uint64_t count);

  std::expected<void, Refusal> unary(std::size_t index);
  std::expected<void, Refusal> binary(std::size_t index);
  std::expected<void, Refusal> concat(std::size_t index);
  std::expected<void, Refusal> split(std::size_t index);
  std::expected<void, Refusal> slice(std::size_t index);
  std::expected<void, Refusal> transpose(std::size_t index);
  std::expected<void, Refusal> depth_space(std::size_t index);
  std::expected<void, Refusal> pad(std::size_t index);
  std::expected<void, Refusal> resize(std::size_t index);
  std::expected<void, Refusal> grid_sample(std::size_t index);
  std::expected<void, Refusal> reduce_mean(std::size_t index);
  std::expected<void, Refusal> average_pool(std::size_t index);
  std::expected<void, Refusal> conv(std::size_t index, const KernelChoice& choice);
  std::expected<void, Refusal> conv_matrix(std::size_t index, const KernelChoice& choice);
  std::expected<void, Refusal> conv_transpose(std::size_t index, const KernelChoice& choice);

  const ir::Graph& graph_;
  Target target_;
  Program program_;
  std::map<ir::TensorId, std::uint32_t> constants_;  // constantes déjà rangées en blob
};

// Le programme entier : un choix par nœud, dans l'ordre du graphe.
std::expected<Program, Refusal> lower(const ir::Graph& graph, std::span<const KernelChoice> choices,
                                      const Target& target);

// Les choix par défaut (premier candidat de chaque nœud).
std::expected<std::vector<KernelChoice>, Refusal> default_choices(const ir::Graph& graph,
                                                                  const Target& target);

}  // namespace ajp::gpu::vk::net
