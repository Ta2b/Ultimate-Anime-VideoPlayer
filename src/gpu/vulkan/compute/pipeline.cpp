#include "gpu/vulkan/compute/pipeline.h"

#include <format>

#include <spdlog/logger.h>

#include "ajp_vk_compute_spirv.h"
#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {

Pipeline::~Pipeline() {
  const DeviceFunctions& fn = context_.fn();
  const VkDevice device = context_.device();
  if (pipeline_ != VK_NULL_HANDLE) fn.vkDestroyPipeline(device, pipeline_, nullptr);
  if (layout_ != VK_NULL_HANDLE) fn.vkDestroyPipelineLayout(device, layout_, nullptr);
  if (set_layout_ != VK_NULL_HANDLE) fn.vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
}

std::expected<std::unique_ptr<PipelineCache>, Refusal> PipelineCache::create(const VkContext& context) {
  std::unique_ptr<PipelineCache> cache(new PipelineCache(context));
  const VkPipelineCacheCreateInfo info{.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  if (auto ok = check(context.fn().vkCreatePipelineCache(context.device(), &info, nullptr, &cache->cache_),
                      "vkCreatePipelineCache");
      !ok) {
    return std::unexpected(ok.error());
  }
  return cache;
}

PipelineCache::~PipelineCache() {
  pipelines_.clear();
  const DeviceFunctions& fn = context_.fn();
  for (const auto& [name, module] : modules_) fn.vkDestroyShaderModule(context_.device(), module, nullptr);
  if (cache_ != VK_NULL_HANDLE) fn.vkDestroyPipelineCache(context_.device(), cache_, nullptr);
}

std::expected<VkShaderModule, Refusal> PipelineCache::module(std::string_view shader) {
  if (const auto found = modules_.find(shader); found != modules_.end()) return found->second;
  const std::span<const std::uint32_t> words = spirv::words(shader);
  if (words.empty()) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("shader {} absent du SPIR-V embarqué", shader));
  }
  const VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = words.size_bytes(),
                                      .pCode = words.data()};
  VkShaderModule module = VK_NULL_HANDLE;
  if (auto ok = check(context_.fn().vkCreateShaderModule(context_.device(), &info, nullptr, &module),
                      std::format("vkCreateShaderModule({})", shader));
      !ok) {
    return std::unexpected(ok.error());
  }
  modules_.emplace(std::string(shader), module);
  return module;
}

std::expected<std::unique_ptr<Pipeline>, Refusal> PipelineCache::build(const PipelineDesc& desc,
                                                                       VkShaderModule module) {
  const DeviceFunctions& fn = context_.fn();
  const VkDevice device = context_.device();
  std::unique_ptr<Pipeline> pipeline(new Pipeline(context_));
  pipeline->name_ = std::string(desc.shader);
  pipeline->buffers_ = desc.buffers;
  pipeline->images_ = desc.images;
  pipeline->push_bytes_ = desc.push_bytes;

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(std::size_t{desc.buffers} + desc.images);
  for (std::uint32_t i = 0; i < desc.buffers + desc.images; ++i) {
    bindings.push_back({.binding = i,
                        .descriptorType = i < desc.buffers ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                           : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
  }
  const VkDescriptorSetLayoutCreateInfo set_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                 .bindingCount = static_cast<std::uint32_t>(bindings.size()),
                                                 .pBindings = bindings.data()};
  if (auto ok = check(fn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &pipeline->set_layout_),
                      "vkCreateDescriptorSetLayout");
      !ok) {
    return std::unexpected(ok.error());
  }
  const VkPushConstantRange push{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = desc.push_bytes};
  const VkPipelineLayoutCreateInfo layout_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                               .setLayoutCount = 1,
                                               .pSetLayouts = &pipeline->set_layout_,
                                               .pushConstantRangeCount = desc.push_bytes > 0 ? 1U : 0U,
                                               .pPushConstantRanges = &push};
  if (auto ok = check(fn.vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline->layout_),
                      "vkCreatePipelineLayout");
      !ok) {
    return std::unexpected(ok.error());
  }

  std::vector<VkSpecializationMapEntry> entries;
  entries.reserve(desc.constants.size());
  for (std::uint32_t i = 0; i < desc.constants.size(); ++i) {
    entries.push_back({.constantID = i, .offset = i * 4, .size = sizeof(std::uint32_t)});
  }
  const VkSpecializationInfo specialization{.mapEntryCount = static_cast<std::uint32_t>(entries.size()),
                                            .pMapEntries = entries.data(),
                                            .dataSize = desc.constants.size_bytes(),
                                            .pData = desc.constants.data()};
  const VkComputePipelineCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
                .pSpecializationInfo = entries.empty() ? nullptr : &specialization},
      .layout = pipeline->layout_};
  if (auto ok = check(fn.vkCreateComputePipelines(device, cache_, 1, &info, nullptr, &pipeline->pipeline_),
                      std::format("vkCreateComputePipelines({})", desc.shader));
      !ok) {
    return std::unexpected(ok.error());
  }
  return pipeline;
}

std::expected<const Pipeline*, Refusal> PipelineCache::get(const PipelineDesc& desc) {
  Key key{std::string(desc.shader), std::vector<std::uint32_t>(desc.constants.begin(), desc.constants.end())};
  if (const auto found = pipelines_.find(key); found != pipelines_.end()) {
    const Pipeline& p = *found->second;
    if (p.buffers() != desc.buffers || p.images() != desc.images || p.push_bytes() != desc.push_bytes) {
      return refuse(
          RefusalCode::InputMismatch,
          std::format("{} : {} tampons, {} images et {} o de push demandés, {}, {} et {} déjà créés",
                      desc.shader, desc.buffers, desc.images, desc.push_bytes, p.buffers(), p.images(),
                      p.push_bytes()));
    }
    return &p;
  }
  if (desc.push_bytes % 4 != 0) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : push de {} o, multiple de 4 attendu", desc.shader, desc.push_bytes));
  }
  if (desc.push_bytes > context_.caps().max_push_constants) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} : push de {} o, l'appareil en permet {}", desc.shader, desc.push_bytes,
                              context_.caps().max_push_constants));
  }
  auto shader = module(desc.shader);
  if (!shader) return std::unexpected(shader.error());
  auto pipeline = build(desc, *shader);
  if (!pipeline) return std::unexpected(pipeline.error());
  const Pipeline* created = pipeline->get();
  pipelines_.emplace(std::move(key), std::move(*pipeline));
  context_.log().debug("pipeline {} créé ({} constantes)", desc.shader, desc.constants.size());
  return created;
}

}  // namespace ajp::gpu::vk
