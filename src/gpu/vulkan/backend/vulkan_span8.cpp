// Ce que la dérivation du SPAN FP8 sous Vulkan (vulkan_span_derive.cpp) demande au backend : le graphe d'un
// paquet construit en moteur, la file attendue sur l'hôte, une copie entre tampons.
#include <chrono>
#include <optional>
#include <utility>

#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/backend/net_span8.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"

namespace ajp::gpu::vk {
namespace {

using detail::refuse_backend;

// Une calibration ou une mesure attend toute la file : bien plus qu'une image de lecture.
constexpr auto kHostWait = std::chrono::seconds(120);

}  // namespace

std::expected<void, Refusal> VulkanBackend::finish_on_host() {
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  auto point = queue_->joined();
  if (!point) return std::unexpected(point.error());
  if (auto ok = queue_->flush(); !ok) return std::unexpected(ok.error());
  return context_->wait_on_host(*point, kHostWait);
}

std::expected<void, Refusal> VulkanBackend::copy_on_host(VkBuffer from, VkDeviceSize from_offset, VkBuffer to,
                                                         VkDeviceSize to_offset, VkDeviceSize size) {
  if (auto ok = finish_on_host(); !ok) return ok;
  return context_->run_once([&](VkCommandBuffer cb) {
    const VkBufferCopy region{.srcOffset = from_offset, .dstOffset = to_offset, .size = size};
    context_->fn().vkCmdCopyBuffer(cb, from, to, 1, &region);
  });
}

std::expected<std::unique_ptr<Engine>, Refusal> VulkanBackend::span8_engine(
    const span_fp8::Package& package, const span_fp8::ModelOptions& options, int width, int height,
    bool tuned, Progress& progress) {
  auto graph = net::span8_graph(package, options);
  if (!graph) return std::unexpected(graph.error());
  const net::BuildRequest request{.onnx = {},
                                  .model_id = options.quantize ? "span8-fp8" : "span8-fp16",
                                  .model_sha256 = {},
                                  .input_shape = {1, 3, height, width},
                                  .fp16 = true,
                                  .strict = options_.strict_precision};
  std::optional<net::Autotuner> tuner;
  if (tuned && options_.autotune)
    tuner.emplace(*context_, *memory_, *pipelines_, profile_, cache_dir_ / "vulkan" / "autotune.tsv",
                  options_.tuner);
  auto data = net::build_imported(std::move(*graph), request, profile_, tuner ? &*tuner : nullptr, nullptr,
                                  progress.stop);
  if (!data) return std::unexpected(data.error());
  // Le graphe FP8 qui sera joué passe la porte de bout en bout de la lecture (gate), comme build ; la
  // référence garde les choix par défaut.
  if (tuned) {
    // L'arrêt seulement : la barre de la porte (95 à 100 %) n'est pas celle de la dérivation.
    const Progress gating{.report = {}, .stop = progress.stop};
    if (auto ok = gate(ModelSpec{.id = request.model_id}, *data, tuner ? &*tuner : nullptr, &gating); !ok)
      return std::unexpected(ok.error());
  }
  auto program = net::lower(data->graph, data->choices, profile_.target);
  if (!program) return std::unexpected(program.error());
  auto engine =
      net::NetEngine::create(*context_, *memory_, *pipelines_, std::move(data->graph), std::move(*program));
  if (!engine) return std::unexpected(engine.error());
  return std::make_unique<VulkanEngine>(std::move(*engine));
}

}  // namespace ajp::gpu::vk
