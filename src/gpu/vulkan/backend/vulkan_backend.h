// Backend Vulkan (gpu/backend.h, ABI 3) : l'appareil de rendu adopté tel quel (mêmes VkDevice et
// VkImage que libplacebo, aucune interop) ou, sans appareil fourni, un appareil créé (outils,
// banc, tests) ; les noyaux de la chaîne (kernel_ops.h) ; les réseaux de gpu_ir construits en
// artefacts dans le dossier de cache, chargés, puis rejoués sur la file du backend.
//
// Bibliothèque statique : ajp_backend_vulkan (entry.cpp) l'enveloppe dans les points d'entrée
// extern "C" ; les tests l'utilisent directement.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_engine.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/backend/queue.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"

namespace ajp::gpu::vk {

struct HostViews;  // vulkan_backend_detail.h

// Réglages hors contrat, pour les tests et le banc.
struct BackendOptions {
  bool autotune = true;  // false : le premier candidat de chaque nœud, sans mesure
  net::TunerOptions tuner;
  // Les réseaux par un autre runtime (net_runtime.h), créé à l'ouverture de l'appareil :
  // ajp_backend_migraphx, ajp_backend_openvino.
  net::RuntimeFactory runtime;
};

// Un réseau chargé : son moteur et les noms de ses entrées et sorties.
class VulkanEngine final : public Engine {
 public:
  explicit VulkanEngine(std::unique_ptr<net::NetEngine> net) : net_(std::move(net)) {}
  std::span<const std::string> input_names() const override { return net_->input_names(); }
  std::span<const std::string> output_names() const override { return net_->output_names(); }
  net::NetEngine& net() { return *net_; }

 private:
  std::unique_ptr<net::NetEngine> net_;
};

class VulkanBackend final : public Backend {
 public:
  // Refus : DeviceMismatch (appareil de rendu qui n'est pas Vulkan), ceux de l'adoption ou de la
  // création de l'appareil (LibraryUnavailable, DeviceUnavailable, FeatureUnavailable…).
  static std::expected<std::unique_ptr<VulkanBackend>, Refusal> create(const BackendInit& init,
                                                                       BackendOptions options = {});

  VulkanBackend(const VulkanBackend&) = delete;
  VulkanBackend& operator=(const VulkanBackend&) = delete;
  VulkanBackend(VulkanBackend&&) = delete;
  VulkanBackend& operator=(VulkanBackend&&) = delete;
  ~VulkanBackend() override;

  BackendCaps capabilities() const override;
  DeviceMatch match(const DeviceUuid& render_device) const override;
  std::expected<Artifact, Refusal> build(const ModelSpec& model, const ShapeSet& shapes,
                                         Progress& progress) override;
  std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact& artifact) override;
  // Le backend lit les images décodées par KernelOps::ingest et écrit le ring par write_ring :
  // aucune image ne devient tenseur sans copie. Refus FormatUnsupported, toujours.
  std::expected<Tensor, Refusal> import(const ExternalImage& image) override;
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                     const Shape& shape) override;
  KernelOps* kernels() override { return kernels_.get(); }
  std::expected<void, Refusal> enqueue(Engine& engine, std::span<const Tensor> inputs,
                                       std::span<Tensor> outputs) override;
  std::expected<Fence, Refusal> signal() override;
  std::expected<void, Refusal> wait(const Fence& fence) override;

  VkContext& context() { return *context_; }
  MemoryAllocator& memory() { return *memory_; }
  const DeviceProfile& profile() const { return profile_; }
  BackendQueue& queue() { return *queue_; }
  // Chemin de l'artefact de ce modèle à cette forme, construit ou non.
  std::filesystem::path artifact_path(const ModelSpec& model, const ShapeSet& shapes) const;

 private:
  explicit VulkanBackend(const BackendInit& init, BackendOptions options);
  std::expected<void, Refusal> open(const BackendInit& init);
  void log(BackendLogLevel level, std::string_view message) const;
  // Le chemin d'un runtime (vulkan_backend_runtime.cpp).
  std::expected<Artifact, Refusal> build_runtime(const ModelSpec& model, const ShapeSet& shapes,
                                                 Progress& progress);
  std::expected<std::unique_ptr<Engine>, Refusal> load_runtime(const Artifact& artifact);
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_runtime_tensor(TensorType type,
                                                                             const Shape& shape);
  std::expected<void, Refusal> enqueue_runtime(Engine& engine, std::span<const Tensor> inputs,
                                               std::span<Tensor> outputs);

  BackendLog log_;
  std::filesystem::path cache_dir_;
  BackendOptions options_;
  std::unique_ptr<VkContext> context_;
  std::unique_ptr<MemoryAllocator> memory_;
  std::unique_ptr<PipelineCache> pipelines_;
  DeviceProfile profile_;
  std::optional<BackendQueue> queue_;
  std::unique_ptr<VulkanKernelOps> kernels_;
  std::shared_ptr<net::Runtime> runtime_;
  std::shared_ptr<HostViews> host_views_;  // projections des tenseurs d'un runtime
};

// Le contexte d'un appareil de rendu Vulkan partagé : sa file compute, son verrou de file.
// Refus DeviceMismatch : appareil sans description Vulkan.
std::expected<AdoptedDevice, Refusal> adopted_device(const GpuDevice& render);

}  // namespace ajp::gpu::vk
