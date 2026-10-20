// Construction et rechargement des réseaux du moteur Vulkan. build_network importe l'ONNX,
// fixe la forme d'entrée, applique les passes de gpu_ir, choisit un noyau par nœud (autotuner
// ou choix par défaut) et abaisse le graphe : tout refus arrive ici, jamais à l'exécution.
// verify_artifact rend le programme d'un artefact relu, après avoir vérifié qu'il a été fait
// pour cet appareil, ces shaders et cette version de l'abaissement.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <stop_token>
#include <string>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_program.h"

namespace ajp::gpu::vk::net {

struct BuildRequest {
  std::filesystem::path onnx;
  std::string model_id;
  std::string model_sha256;
  ir::Dims input_shape;  // NCHW de l'entrée d'un modèle à une entrée ; ceux à plusieurs sont figés
  bool fp16 = true;      // false : tous les tenseurs en float
};

using BuildReport = std::function<void(float fraction, const std::string& stage)>;

// Refus : ceux de gpu_ir (import, formes, passes), InputMismatch (modèle à plusieurs entrées),
// FeatureUnavailable (float16 sans stockage 16 bits), OperatorUnknown et ShapeInvalid de
// l'abaissement, Cancelled (stop demandé), ceux de l'autotuner. tuner nul : choix par défaut.
std::expected<ArtifactData, Refusal> build_network(const BuildRequest& request, const DeviceProfile& profile,
                                                   Autotuner* tuner, const BuildReport& report,
                                                   const std::stop_token& stop);

// Le graphe d'un modèle importé, à cette forme, optimisé, dans la précision demandée.
std::expected<ir::Graph, Refusal> prepare_graph(const BuildRequest& request, const Target& target);

// Hachages SPIR-V des shaders que le programme utilise, un par nom.
std::vector<ShaderHash> shader_hashes(std::span<const Dispatch> dispatches);

// Refus : DeviceMismatch (artefact d'un autre appareil ou pilote), FormatVersion (autre
// abaissement, autres shaders, ou programme qui ne se refait pas à l'identique), ceux de
// ir::validate et de l'abaissement.
std::expected<Program, Refusal> verify_artifact(const ArtifactData& data, const DeviceProfile& profile);

// Clé de cache de l'artefact : modèle, forme, précision, appareil et versions.
std::uint64_t artifact_key(const BuildRequest& request, const DeviceProfile& profile);

// Écriture atomique (fichier temporaire puis renommage) et lecture. Refus CacheIo.
std::expected<void, Refusal> write_file(const std::filesystem::path& path, std::span<const std::byte> bytes);
std::expected<std::vector<std::byte>, Refusal> read_file(const std::filesystem::path& path);

}  // namespace ajp::gpu::vk::net
