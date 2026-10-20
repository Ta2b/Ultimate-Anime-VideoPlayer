// Ouverture et fermeture de l'écran libplacebo : fenêtre SDL, instance et appareil Vulkan (les
// fonctionnalités de calcul du backend, Vulkan Video pour FFmpeg), swapchain, renderer, calques,
// et ce que l'appareil sait faire (device_facts). Présentation dans placebo_screen.cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>

#include "app/play/osd_text.h"
#include "app/play/placebo_impl.h"
#include "app/play/placebo_screen.h"
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

void placebo_log(void* priv, pl_log_level level, const char* message) {
  auto* log = static_cast<spdlog::logger*>(priv);
  if (level <= PL_LOG_ERR) {
    log->error("libplacebo : {}", message);
  } else if (level == PL_LOG_WARN) {
    log->warn("libplacebo : {}", message);
  }
}

// Les files de libplacebo se verrouillent par ses fonctions : le backend Vulkan, qui adopte
// l'appareil, soumet sur la même file de calcul.
void lock_placebo_queue(void* context, std::uint32_t family, std::uint32_t index) {
  const auto vk = static_cast<pl_vulkan>(context);
  vk->lock_queue(vk, family, index);
}
void unlock_placebo_queue(void* context, std::uint32_t family, std::uint32_t index) {
  const auto vk = static_cast<pl_vulkan>(context);
  vk->unlock_queue(vk, family, index);
}

// L'appareil de libplacebo tel que gpu/backend.h le décrit : un backend Vulkan l'adopte au lieu
// d'en créer un autre, et lit ainsi les images du ring sans les exporter.
VulkanDevice vulkan_description(pl_vulkan vk) {
  VulkanDevice d{.instance = vk->instance,
                 .physical_device = vk->phys_device,
                 .device = vk->device,
                 .get_instance_proc_addr = reinterpret_cast<VoidFunction>(vk->get_proc_addr),
                 .api_version = vk->api_version,
                 .enabled_features = vk->features,
                 .graphics = {.family = static_cast<std::uint32_t>(vk->queue_graphics.index), .index = 0},
                 .compute = {.family = static_cast<std::uint32_t>(vk->queue_compute.index), .index = 0},
                 .lock_queue = lock_placebo_queue,
                 .unlock_queue = unlock_placebo_queue,
                 .queue_context = const_cast<pl_vulkan_t*>(vk)};
  for (int i = 0; i < vk->num_extensions; ++i) d.device_extensions.emplace_back(vk->extensions[i]);
  return d;
}

// Ce que le backend Vulkan demande à l'appareil qu'il adopte (tenseurs float16, matrices
// coopératives, additions atomiques de float) et ce que FFmpeg attend du même appareil pour Vulkan
// Video (video maintenance 1 et 2), activé par libplacebo seulement là où le pilote le propose.
struct ComputeFeatures {
  VkPhysicalDeviceVideoMaintenance2FeaturesKHR video2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR,
      .videoMaintenance2 = VK_TRUE};
  VkPhysicalDeviceVideoMaintenance1FeaturesKHR video1{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
      .pNext = &video2,
      .videoMaintenance1 = VK_TRUE};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
      .pNext = &video1,
      .shaderBufferFloat32AtomicAdd = VK_TRUE};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      .pNext = &atomic,
      .cooperativeMatrix = VK_TRUE};
  VkPhysicalDeviceVulkan13Features v13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                       .pNext = &matrix,
                                       .subgroupSizeControl = VK_TRUE,
                                       .computeFullSubgroups = VK_TRUE};
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       .pNext = &v13,
                                       .shaderFloat16 = VK_TRUE,
                                       .vulkanMemoryModel = VK_TRUE};
  VkPhysicalDeviceVulkan11Features v11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                       .pNext = &v12,
                                       .storageBuffer16BitAccess = VK_TRUE};
  VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v11};

  ComputeFeatures() { features.features.shaderInt16 = VK_TRUE; }
  ComputeFeatures(const ComputeFeatures&) = delete;
  ComputeFeatures& operator=(const ComputeFeatures&) = delete;
  ComputeFeatures(ComputeFeatures&&) = delete;
  ComputeFeatures& operator=(ComputeFeatures&&) = delete;
  ~ComputeFeatures() = default;
};

// Les extensions demandées si l'appareil les a : celles du backend Vulkan, puis Vulkan Video, pour
// que FFmpeg décode et encode sur cet appareil (vulkan_share.h) ; ses files vidéo sont créées en
// plus (extra_queues).
constexpr std::array<const char*, 15> kOptionalExtensions{
    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,      VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME, VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,  VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME,
    VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME,   VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME,  VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME,
    VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME,   VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME,
    VK_KHR_VIDEO_MAINTENANCE_2_EXTENSION_NAME};

}  // namespace

PlaceboScreen::Impl::~Impl() {
  if (vk != nullptr) pl_gpu_finish(vk->gpu);
  share.reset();  // avant l'appareil qu'il prête
  if (!grade_gpu_us.empty()) {
    std::ranges::sort(grade_gpu_us);
    const auto at = [&](double q) {
      return grade_gpu_us[static_cast<std::size_t>(q * static_cast<double>(grade_gpu_us.size() - 1))];
    };
    channel->info("profil GPU · grade HDR : médiane {:.2f} ms, p95 {:.2f} ms ({} images)", at(0.5) / 1000.0,
                  at(0.95) / 1000.0, grade_gpu_us.size());
  }
  ui.reset();
  grade.reset();
  ramp_grade.reset();
  pl_renderer_destroy(&ramp_renderer);
  osd.reset();
  overlays.reset();
  ring.clear(vk != nullptr ? vk->gpu : nullptr);
  pl_renderer_destroy(&renderer);
  pl_swapchain_destroy(&swapchain);
  pl_vulkan_destroy(&vk);
  if (surface != VK_NULL_HANDLE) SDL_Vulkan_DestroySurface(inst->instance, surface, nullptr);
  pl_vk_inst_destroy(&inst);
  pl_log_destroy(&log);
  if (window != nullptr) SDL_DestroyWindow(window);
}

std::expected<std::unique_ptr<PlaceboScreen>, Refusal> PlaceboScreen::open(const ScreenOptions& options) {
  auto impl = std::make_unique<Impl>();
  impl->channel = log_channel("app");
#ifdef __APPLE__
  // Paquet .app : le chargeur Vulkan est dans Contents/Frameworks, où dlopen ne cherche pas un
  // nom seul ; SDL le reçoit par son chemin (SDL_GetBasePath rend Contents/Resources).
  if (const char* base = SDL_GetBasePath(); base != nullptr) {
    const std::filesystem::path loader =
        std::filesystem::path(base) / ".." / "Frameworks" / "libvulkan.1.dylib";
    std::error_code missing;
    if (std::filesystem::exists(loader, missing)) (void)SDL_Vulkan_LoadLibrary(loader.string().c_str());
  }
#endif
  SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  if (options.fullscreen) flags |= SDL_WINDOW_FULLSCREEN;
  impl->window = SDL_CreateWindow(options.title.c_str(), options.width, options.height, flags);
  if (impl->window == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("SDL_CreateWindow : {}", SDL_GetError()));
  }
  impl->fullscreen = options.fullscreen;

  pl_log_params log_params{};
  log_params.log_cb = placebo_log;
  log_params.log_priv = impl->channel.get();
  log_params.log_level = PL_LOG_WARN;
  impl->log = pl_log_create(PL_API_VER, &log_params);

  Uint32 count = 0;
  const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&count);
  if (sdl_extensions == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("SDL_Vulkan_GetInstanceExtensions : {}", SDL_GetError()));
  }
  if (options.smooth_motion) {
    // Le chargeur Vulkan lit la variable à la création de l'instance ; le thread main est encore
    // seul à toucher l'environnement à ce moment.
    SDL_setenv_unsafe("NVPRESENT_ENABLE_SMOOTH_MOTION", "1", 1);
    impl->channel->info(
        "Smooth Motion : couche du pilote NVIDIA demandée (NVPRESENT_ENABLE_SMOOTH_MOTION=1)");
  }
  pl_vk_inst_params inst_params{};
  inst_params.extensions = sdl_extensions;
  inst_params.num_extensions = static_cast<int>(count);
  impl->inst = pl_vk_inst_create(impl->log, &inst_params);
  if (impl->inst == nullptr) return refuse(RefusalCode::LibraryUnavailable, "pl_vk_inst_create a échoué");
  if (!SDL_Vulkan_CreateSurface(impl->window, impl->inst->instance, nullptr, &impl->surface)) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("SDL_Vulkan_CreateSurface : {}", SDL_GetError()));
  }

  pl_vulkan_params vk_params = pl_vulkan_default_params;
  vk_params.instance = impl->inst->instance;
  vk_params.get_proc_addr = impl->inst->get_proc_addr;
  vk_params.surface = impl->surface;
  vk_params.allow_software = options.allow_software;
  const ComputeFeatures compute;
  vk_params.features = &compute.features;
  vk_params.opt_extensions = kOptionalExtensions.data();
  vk_params.num_opt_extensions = static_cast<int>(kOptionalExtensions.size());
  vk_params.extra_queues = VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_VIDEO_ENCODE_BIT_KHR;
  impl->vk = pl_vulkan_create(impl->log, &vk_params);
  if (impl->vk == nullptr)
    return refuse(RefusalCode::DeviceMismatch, "pl_vulkan_create : aucun appareil ne convient");
  const pl_gpu gpu = impl->vk->gpu;
  if (auto share = VulkanShare::create(impl->inst, impl->vk)) {
    impl->share = std::move(*share);
  } else {
    impl->channel->info("appareil Vulkan non partagé avec FFmpeg : {} ; ses images passeront par l'hôte",
                        share.error().detail);
  }
  if ((gpu->export_caps.tex & PL_HANDLE_FD) == 0 || (gpu->export_caps.sync & PL_HANDLE_FD) == 0) {
    // Sans export, le backend TensorRT ne peut pas importer le ring ; le backend Vulkan s'en passe.
    impl->channel->info("l'appareil n'exporte pas ses images ni ses sémaphores (fd) : ring sans poignées");
  }

  pl_vulkan_swapchain_params sw_params{};
  sw_params.surface = impl->surface;
  sw_params.present_mode = VK_PRESENT_MODE_FIFO_KHR;
  sw_params.swapchain_depth = kSwapchainDepth;
  impl->swapchain = pl_vulkan_create_swapchain(impl->vk, &sw_params);
  if (impl->swapchain == nullptr)
    return refuse(RefusalCode::LibraryUnavailable, "pl_vulkan_create_swapchain a échoué");
  if (options.hdr) pl_swapchain_colorspace_hint(impl->swapchain, &pl_color_space_hdr10);
  impl->renderer = pl_renderer_create(impl->log, gpu);

  impl->device = GpuDevice{.api = GpuApi::Vulkan, .name = "Vulkan (libplacebo)"};
  for (std::size_t i = 0; i < impl->device.uuid.size(); ++i) impl->device.uuid[i] = gpu->uuid[i];
  impl->device.vulkan = vulkan_description(impl->vk);
  auto osd = OsdText::create();
  if (!osd) return std::unexpected(std::move(osd.error()));
  impl->osd = std::move(*osd);
  impl->overlays = std::make_unique<Overlays>(gpu);
  impl->ui = std::make_unique<ImGuiLayer>(gpu, impl->log);
  if (auto grade = render::Grade::create(gpu)) {
    impl->grade.emplace(std::move(*grade));
  } else {
    impl->channel->warn("grade HDR indisponible : {}", grade.error().detail);
  }
  impl->source = pl_color_space_bt709;
  return std::unique_ptr<PlaceboScreen>(new PlaceboScreen(std::move(impl)));
}

PlaceboScreen::DeviceFacts PlaceboScreen::device_facts() const {
  DeviceFacts facts;
  const pl_vulkan vk = impl_->vk;
  const auto properties_of = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceProperties"));
  if (properties_of != nullptr) {
    VkPhysicalDeviceProperties properties{};
    properties_of(vk->phys_device, &properties);
    facts.name = properties.deviceName;
    facts.pci_vendor = properties.vendorID;
  }
  const auto memory = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceMemoryProperties"));
  if (memory != nullptr) {
    VkPhysicalDeviceMemoryProperties properties{};
    memory(vk->phys_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryHeapCount; ++i) {
      const VkMemoryHeap& heap = properties.memoryHeaps[i];
      if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0U)
        facts.vram_bytes = std::max<std::uint64_t>(facts.vram_bytes, heap.size);
    }
  }
  const auto extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
      vk->get_proc_addr(vk->instance, "vkEnumerateDeviceExtensionProperties"));
  std::uint32_t count = 0;
  if (extensions != nullptr && extensions(vk->phys_device, nullptr, &count, nullptr) == VK_SUCCESS) {
    std::vector<VkExtensionProperties> list(count);
    if (extensions(vk->phys_device, nullptr, &count, list.data()) == VK_SUCCESS) {
      for (const VkExtensionProperties& extension : list) {
        const std::string_view name(extension.extensionName);
        facts.cooperative_matrix |= name == "VK_KHR_cooperative_matrix" || name == "VK_NV_cooperative_matrix";
      }
    }
  }
  return facts;
}

}  // namespace ajp::app::play
