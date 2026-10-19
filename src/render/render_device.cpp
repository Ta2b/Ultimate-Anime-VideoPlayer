#include "render/render_device.h"

#include <array>
#include <cstddef>
#include <ranges>
#include <utility>

#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/vulkan.h>

#include "render/device_description.h"
#include "render/log.h"
#include "render/placebo_log.h"
#include "render/vk_util.h"

namespace ajp::render {
namespace {

// Extensions d'appareil facultatives, activées si l'appareil les a. D'abord celles que le
// décodage Vulkan de FFmpeg exploite, dans l'ordre de mpv (fork 5afefa7,
// video/out/vulkan/context.c:198-249 ; libavutil >= 60.26 se passe de descriptor_buffer), puis
// celles de notre moteur. Les mémoires et sémaphores externes (fd, win32) sont activés par
// libplacebo lui-même (pl_vulkan_recommended_extensions).
constexpr std::array kOptionalDeviceExtensions = {
    VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
    VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
    VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME,
    VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME,
    VK_KHR_SHADER_EXPECT_ASSUME_EXTENSION_NAME,
    VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME,
    VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME,
    VK_KHR_VIDEO_MAINTENANCE_2_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_9_EXTENSION_NAME,
    // Moteur : FP16 et stockage 16 bits (cœur depuis 1.1 et 1.2), unités matricielles.
    VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
    VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
    VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME,
};

// Fonctionnalités demandées, chaînées ; libplacebo n'active que celles que l'appareil a
// (pl_vulkan_params::features) et range chacune dans la structure de sa version.
struct FeatureRequest {
  VkPhysicalDeviceCooperativeMatrix2FeaturesNV matrix2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
      .cooperativeMatrixWorkgroupScope = VK_TRUE,
      .cooperativeMatrixFlexibleDimensions = VK_TRUE,
      .cooperativeMatrixReductions = VK_TRUE,
      .cooperativeMatrixConversions = VK_TRUE,
      .cooperativeMatrixPerElementOperations = VK_TRUE,
      .cooperativeMatrixTensorAddressing = VK_TRUE,
      .cooperativeMatrixBlockLoads = VK_TRUE};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      .cooperativeMatrix = VK_TRUE};
  VkPhysicalDeviceMaintenance9FeaturesKHR maintenance9{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR, .maintenance9 = VK_TRUE};
  VkPhysicalDeviceVideoDecodeVP9FeaturesKHR vp9{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR, .videoDecodeVP9 = VK_TRUE};
  VkPhysicalDeviceVideoMaintenance2FeaturesKHR video2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR,
      .videoMaintenance2 = VK_TRUE};
  VkPhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR relaxed{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES_KHR,
      .shaderRelaxedExtendedInstruction = VK_TRUE};
  VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR rotate{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR,
      .shaderSubgroupRotate = VK_TRUE};
  VkPhysicalDeviceShaderExpectAssumeFeaturesKHR expect{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES_KHR,
      .shaderExpectAssume = VK_TRUE};
  VkPhysicalDeviceVideoMaintenance1FeaturesKHR video1{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
      .videoMaintenance1 = VK_TRUE};
  VkPhysicalDeviceShaderObjectFeaturesEXT shader_object{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT, .shaderObject = VK_TRUE};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
      .shaderBufferFloat32Atomics = VK_TRUE,
      .shaderBufferFloat32AtomicAdd = VK_TRUE};
  VkPhysicalDeviceVulkan13Features vk13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                        .subgroupSizeControl = VK_TRUE,
                                        .computeFullSubgroups = VK_TRUE,
                                        .synchronization2 = VK_TRUE,
                                        .shaderZeroInitializeWorkgroupMemory = VK_TRUE,
                                        .dynamicRendering = VK_TRUE,
                                        .shaderIntegerDotProduct = VK_TRUE,
                                        .maintenance4 = VK_TRUE};
  VkPhysicalDeviceVulkan12Features vk12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                        .storageBuffer8BitAccess = VK_TRUE,
                                        .uniformAndStorageBuffer8BitAccess = VK_TRUE,
                                        .shaderBufferInt64Atomics = VK_TRUE,
                                        .shaderFloat16 = VK_TRUE,
                                        .shaderInt8 = VK_TRUE,
                                        .shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
                                        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
                                        .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
                                        .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
                                        .runtimeDescriptorArray = VK_TRUE,
                                        .scalarBlockLayout = VK_TRUE,
                                        .uniformBufferStandardLayout = VK_TRUE,
                                        .shaderSubgroupExtendedTypes = VK_TRUE,
                                        .hostQueryReset = VK_TRUE,
                                        .timelineSemaphore = VK_TRUE,
                                        .bufferDeviceAddress = VK_TRUE,
                                        .vulkanMemoryModel = VK_TRUE,
                                        .vulkanMemoryModelDeviceScope = VK_TRUE};
  VkPhysicalDeviceVulkan11Features vk11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                        .storageBuffer16BitAccess = VK_TRUE,
                                        .uniformAndStorageBuffer16BitAccess = VK_TRUE,
                                        .storagePushConstant16 = VK_TRUE,
                                        .samplerYcbcrConversion = VK_TRUE};
  VkPhysicalDeviceFeatures2 base{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                 .features = {.fragmentStoresAndAtomics = VK_TRUE,
                                              .shaderStorageImageReadWithoutFormat = VK_TRUE,
                                              .shaderStorageImageWriteWithoutFormat = VK_TRUE,
                                              .shaderFloat64 = VK_TRUE,
                                              .shaderInt64 = VK_TRUE,
                                              .shaderInt16 = VK_TRUE}};

  FeatureRequest() { link(); }
  FeatureRequest(const FeatureRequest&) = delete;
  FeatureRequest& operator=(const FeatureRequest&) = delete;
  FeatureRequest(FeatureRequest&&) = delete;
  FeatureRequest& operator=(FeatureRequest&&) = delete;
  ~FeatureRequest() = default;

 private:
  void link() {
    base.pNext = &vk11;
    vk11.pNext = &vk12;
    vk12.pNext = &vk13;
    vk13.pNext = &atomic_float;
    atomic_float.pNext = &shader_object;
    shader_object.pNext = &video1;
    video1.pNext = &expect;
    expect.pNext = &rotate;
    rotate.pNext = &relaxed;
    relaxed.pNext = &video2;
    video2.pNext = &vp9;
    vp9.pNext = &maintenance9;
    maintenance9.pNext = &matrix;
    matrix.pNext = &matrix2;
  }
};

}  // namespace

struct RenderDevice::Impl {
  std::shared_ptr<spdlog::logger> logger = make_logger();
  pl_log log = nullptr;
  pl_vk_inst instance = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  pl_vulkan vk = nullptr;
  GpuDevice description;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    pl_vulkan_destroy(&vk);
    if (surface != VK_NULL_HANDLE) {
      const auto destroy = detail::instance_function<PFN_vkDestroySurfaceKHR>(
          instance->get_proc_addr, instance->instance, "vkDestroySurfaceKHR");
      if (destroy != nullptr) destroy(instance->instance, surface, nullptr);
    }
    pl_vk_inst_destroy(&instance);
    pl_log_destroy(&log);
  }

  // Le verrou de GpuDevice : celui de pl_vulkan, qui couvre toutes les files de l'appareil.
  static void lock_queue(void* context, std::uint32_t family, std::uint32_t index) {
    const pl_vulkan vk = static_cast<const Impl*>(context)->vk;
    vk->lock_queue(vk, family, index);
  }
  static void unlock_queue(void* context, std::uint32_t family, std::uint32_t index) {
    const pl_vulkan vk = static_cast<const Impl*>(context)->vk;
    vk->unlock_queue(vk, family, index);
  }

  std::expected<void, Refusal> create_instance(const RenderDeviceOptions& options) {
    std::vector<const char*> extensions;
    extensions.reserve(options.instance_extensions.size());
    for (const std::string& name : options.instance_extensions) extensions.push_back(name.c_str());
    pl_vk_inst_params params = pl_vk_inst_default_params;
    params.get_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(options.get_instance_proc_addr);
    params.extensions = extensions.data();
    params.num_extensions = static_cast<int>(extensions.size());
    instance = pl_vk_inst_create(log, &params);
    if (instance == nullptr) {
      return std::unexpected(
          Refusal{RefusalCode::VulkanCallFailed, "pl_vk_inst_create (voir le journal render)"});
    }
    return {};
  }

  std::expected<void, Refusal> create_surface(const RenderDeviceOptions& options) {
    if (!options.make_surface) return {};
    const std::uint64_t handle = options.make_surface(instance->instance);
    if (handle == 0) {
      return std::unexpected(Refusal{RefusalCode::VulkanCallFailed, "création de la surface de la fenêtre"});
    }
    surface = detail::handle_from_u64<VkSurfaceKHR>(handle);
    return {};
  }

  std::expected<void, Refusal> create_device(const RenderDeviceOptions& options) {
    const FeatureRequest features;
    pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = instance->instance;
    params.get_proc_addr = instance->get_proc_addr;
    params.surface = surface;
    params.allow_software = options.allow_software;
    params.extra_queues = VK_QUEUE_VIDEO_DECODE_BIT_KHR;
    params.opt_extensions = kOptionalDeviceExtensions.data();
    params.num_opt_extensions = static_cast<int>(kOptionalDeviceExtensions.size());
    params.features = &features.base;
    if (options.device_uuid) std::ranges::copy(*options.device_uuid, std::begin(params.device_uuid));
    vk = pl_vulkan_create(log, &params);
    if (vk == nullptr) {
      return std::unexpected(Refusal{RefusalCode::DeviceUnavailable,
                                     options.device_uuid
                                         ? "pl_vulkan_create : aucun appareil n'a l'UUID demandé"
                                         : "pl_vulkan_create : aucun appareil Vulkan ne convient"});
    }
    // Les hooks du grade (passes COMPUTE, tampon AJG_STATE) exigent le calcul et le stockage.
    if (!vk->gpu->glsl.compute || vk->gpu->limits.max_ssbo_size == 0) {
      return std::unexpected(
          Refusal{RefusalCode::FeatureUnavailable, "shaders de calcul ou tampons de stockage"});
    }
    return {};
  }
};

std::expected<RenderDevice, Refusal> RenderDevice::create(const RenderDeviceOptions& options) {
  auto impl = std::make_unique<Impl>();
  impl->log = make_placebo_log(*impl->logger);
  if (auto ok = impl->create_instance(options); !ok) return std::unexpected(ok.error());
  if (auto ok = impl->create_surface(options); !ok) return std::unexpected(ok.error());
  if (auto ok = impl->create_device(options); !ok) return std::unexpected(ok.error());
  const detail::QueueLocks locks{
      .lock = &Impl::lock_queue, .unlock = &Impl::unlock_queue, .context = impl.get()};
  auto description = detail::describe_device(impl->instance, impl->vk, locks);
  if (!description) return std::unexpected(description.error());
  impl->logger->info("appareil de rendu : {}, pilote {}, {} extensions d'appareil", description->name,
                     description->driver_version, impl->vk->num_extensions);
  impl->description = std::move(*description);
  return RenderDevice(std::move(impl));
}

RenderDevice::RenderDevice(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RenderDevice::RenderDevice(RenderDevice&&) noexcept = default;
RenderDevice& RenderDevice::operator=(RenderDevice&&) noexcept = default;
RenderDevice::~RenderDevice() = default;

const GpuDevice& RenderDevice::description() const {
  return impl_->description;
}
const pl_gpu_t* RenderDevice::gpu() const {
  return impl_->vk->gpu;
}
const pl_vulkan_t* RenderDevice::vulkan() const {
  return impl_->vk;
}
const pl_log_t* RenderDevice::log() const {
  return impl_->log;
}
std::uint64_t RenderDevice::surface() const {
  return detail::handle_to_u64(impl_->surface);
}

}  // namespace ajp::render
