// Pipelines de calcul depuis le SPIR-V embarqué (ajp_add_spirv), avec leurs constantes de
// spécialisation. Disposition fixe : un ensemble de descripteurs, des tampons de stockage aux
// bindings 0..n−1 puis des images de stockage aux suivants, et un bloc de push constants. Le
// cache garde en mémoire chaque pipeline créé, par shader et constantes, et un VkPipelineCache.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

struct PipelineDesc {
  std::string_view shader;                   // nom embarqué, variante comprise (« resample_h_f16 »)
  std::span<const std::uint32_t> constants;  // constantes de spécialisation 0..n−1, 32 bits chacune
  std::uint32_t buffers = 0;                 // tampons de stockage, bindings 0..buffers−1
  std::uint32_t images = 0;                  // images de stockage, bindings buffers..buffers+images−1
  std::uint32_t push_bytes = 0;              // taille du bloc push_constant, multiple de 4
};

class Pipeline {
 public:
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;
  ~Pipeline();

  VkPipeline handle() const { return pipeline_; }
  VkPipelineLayout layout() const { return layout_; }
  VkDescriptorSetLayout set_layout() const { return set_layout_; }
  const std::string& name() const { return name_; }
  std::uint32_t buffers() const { return buffers_; }
  std::uint32_t images() const { return images_; }
  std::uint32_t push_bytes() const { return push_bytes_; }

 private:
  friend class PipelineCache;
  explicit Pipeline(const VkContext& context) : context_(context) {}

  const VkContext& context_;
  std::string name_;
  std::uint32_t buffers_ = 0;
  std::uint32_t images_ = 0;
  std::uint32_t push_bytes_ = 0;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class PipelineCache {
 public:
  static std::expected<std::unique_ptr<PipelineCache>, Refusal> create(const VkContext& context);

  PipelineCache(const PipelineCache&) = delete;
  PipelineCache& operator=(const PipelineCache&) = delete;
  PipelineCache(PipelineCache&&) = delete;
  PipelineCache& operator=(PipelineCache&&) = delete;
  ~PipelineCache();

  // Le pipeline de ce shader et de ces constantes, créé à la première demande. Valide tant que
  // le cache vit. Refus : FeatureUnavailable (shader absent, push constants au-delà de la
  // limite), InputMismatch (disposition différente de celle d'une demande précédente ou push
  // non multiple de 4), VulkanCallFailed.
  std::expected<const Pipeline*, Refusal> get(const PipelineDesc& desc);

  std::size_t size() const { return pipelines_.size(); }
  std::size_t shader_modules() const { return modules_.size(); }

 private:
  explicit PipelineCache(const VkContext& context) : context_(context) {}
  std::expected<VkShaderModule, Refusal> module(std::string_view shader);
  std::expected<std::unique_ptr<Pipeline>, Refusal> build(const PipelineDesc& desc, VkShaderModule module);

  using Key = std::pair<std::string, std::vector<std::uint32_t>>;
  const VkContext& context_;
  VkPipelineCache cache_ = VK_NULL_HANDLE;
  std::map<std::string, VkShaderModule, std::less<>> modules_;
  std::map<Key, std::unique_ptr<Pipeline>> pipelines_;
};

}  // namespace ajp::gpu::vk
