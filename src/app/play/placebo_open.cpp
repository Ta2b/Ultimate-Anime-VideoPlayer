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
#include "app/play/window_size.h"
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

// La fenêtre en points de l'écran sous Windows et X11 aussi (window_size.h), centrée sur son
// écran.
void fit_display_scale(spdlog::logger& log, SDL_Window* window, int width, int height) {
  const float density = SDL_GetWindowPixelDensity(window);
  const float scale = SDL_GetWindowDisplayScale(window);
  if (!(density > 0.0F) || !(scale > 0.0F)) return;
  const SDL_DisplayID display = SDL_GetDisplayForWindow(window);
  SDL_Rect usable{};
  if (display == 0 || !SDL_GetDisplayUsableBounds(display, &usable)) usable = SDL_Rect{};
  const WindowSize size = window_size_for(width, height, scale / density, usable.w, usable.h);
  if (size.width == width && size.height == height) return;
  if (!SDL_SetWindowSize(window, size.width, size.height)) return;
  SDL_SetWindowPosition(window, static_cast<int>(SDL_WINDOWPOS_CENTERED_DISPLAY(display)),
                        static_cast<int>(SDL_WINDOWPOS_CENTERED_DISPLAY(display)));
  log.info("fenêtre {} × {} : échelle du bureau {:.0f} %", size.width, size.height, scale / density * 100.0F);
}

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
// l'appareil, soumet sous ce verrou quand il n'a pas de file à lui (engine_queue).
void lock_placebo_queue(void* context, std::uint32_t family, std::uint32_t index) {
  const auto vk = static_cast<pl_vulkan>(context);
  vk->lock_queue(vk, family, index);
}
void unlock_placebo_queue(void* context, std::uint32_t family, std::uint32_t index) {
  const auto vk = static_cast<pl_vulkan>(context);
  vk->unlock_queue(vk, family, index);
}

// Une file Vulkan exécute ses soumissions dans l'ordre : sur la file de l'affichage, chaque image
// attendrait derrière les passes des moteurs, et derrière chaque réseau d'un runtime (une attente
// de la timeline que l'hôte signale à la fin du calcul, que les pilotes Mesa retiennent même sur
// l'hôte). libplacebo crée toutes les files de ses familles et n'emploie que la première de
// chacune (queue_count = 1, son défaut) : le backend prend la deuxième de sa famille de calcul,
// sinon de sa famille graphique, que le GPU ordonnance à côté de l'affichage (préemption, tranches
// de temps). Sous MoltenVK, une file par famille, chacune file de commandes Metal, toutes
// générales : libplacebo les crée toutes (extra_queues) et le backend prend une autre famille ;
// un slot du ring passe de l'une à l'autre sans transfert de propriété, sans effet sous MoltenVK.
// Ailleurs, une autre famille demanderait ce transfert à chaque slot : le backend reste dans
// celles de libplacebo, où ses images sont partagées (VK_SHARING_MODE_CONCURRENT).
#ifdef __APPLE__
constexpr VkQueueFlags kBackendQueueFlags = VK_QUEUE_COMPUTE_BIT;
#else
constexpr VkQueueFlags kBackendQueueFlags = 0;
#endif
constexpr std::uint32_t kPlaceboQueues = 1;  // pl_vulkan_params.queue_count

struct EngineQueue {
  QueueRef queue;
  bool shared = true;  // la file de calcul de libplacebo, sous son verrou
};

// Le LUID de l'adaptateur (VkPhysicalDeviceIDProperties), que D3D11VA retrouve par DXGI pour
// décoder sur le GPU du rendu ; rien hors de Windows, ou si le pilote ne le donne pas.
std::optional<DeviceLuid> device_luid(pl_vulkan vk) {
#if defined(_WIN32)
  const auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceProperties2"));
  if (get_properties == nullptr) return std::nullopt;
  VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &ids};
  get_properties(vk->phys_device, &properties);
  if (ids.deviceLUIDValid == VK_FALSE) return std::nullopt;
  DeviceLuid luid{};
  std::copy_n(ids.deviceLUID, luid.size(), luid.begin());
  return luid;
#else
  static_cast<void>(vk);
  return std::nullopt;
#endif
}

EngineQueue engine_queue(pl_vulkan vk) {
  const auto compute = static_cast<std::uint32_t>(vk->queue_compute.index);
  const auto graphics = static_cast<std::uint32_t>(vk->queue_graphics.index);
  const EngineQueue shared{.queue = {.family = compute, .index = 0}};
  const auto get_families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  if (get_families == nullptr) return shared;
  std::uint32_t count = 0;
  get_families(vk->phys_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  get_families(vk->phys_device, &count, families.data());
  const auto computes = [&](std::uint32_t family) {
    return family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
  };
  for (const std::uint32_t family : {compute, graphics}) {
    if (computes(family) && families[family].queueCount > kPlaceboQueues)
      return EngineQueue{.queue = {.family = family, .index = kPlaceboQueues}, .shared = false};
  }
  if constexpr (kBackendQueueFlags != 0) {
    const auto transfer = static_cast<std::uint32_t>(vk->queue_transfer.index);
    for (std::uint32_t family = 0; family < count; ++family) {
      const bool placebo = family == graphics || family == compute || family == transfer;
      if (!placebo && families[family].queueCount > 0 &&
          (families[family].queueFlags & kBackendQueueFlags) == kBackendQueueFlags) {
        return EngineQueue{.queue = {.family = family, .index = 0}, .shared = false};
      }
    }
  }
  return shared;
}

// L'appareil de libplacebo tel que gpu/backend.h le décrit : un backend Vulkan l'adopte au lieu
// d'en créer un autre, et lit ainsi les images du ring sans les exporter. Le verrou seulement sur
// une file partagée : sur la sienne, le backend soumet seul (personne d'autre n'y touche, FFmpeg
// ne reçoit que les files de libplacebo).
VulkanDevice vulkan_description(pl_vulkan vk) {
  const EngineQueue engines = engine_queue(vk);
  VulkanDevice d{.instance = vk->instance,
                 .physical_device = vk->phys_device,
                 .device = vk->device,
                 .get_instance_proc_addr = reinterpret_cast<VoidFunction>(vk->get_proc_addr),
                 .api_version = vk->api_version,
                 .enabled_features = vk->features,
                 .graphics = {.family = static_cast<std::uint32_t>(vk->queue_graphics.index), .index = 0},
                 .compute = engines.queue};
  if (engines.shared) {
    d.lock_queue = lock_placebo_queue;
    d.unlock_queue = unlock_placebo_queue;
    d.queue_context = const_cast<pl_vulkan_t*>(vk);
  }
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
  if (!options.fullscreen) fit_display_scale(*impl->channel, impl->window, options.width, options.height);

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
#if defined(__linux__)
  // Le pilote Intel de Mesa (ANV) n'expose qu'une file graphique, que l'affichage et les moteurs
  // partageraient : deux, chacune son contexte sur le moteur de rendu, que le GPU alterne par
  // tranches de temps. Une valeur déjà posée par l'utilisateur prime ; les autres pilotes
  // l'ignorent. Lue à l'énumération des appareils de l'instance, plus bas.
  SDL_setenv_unsafe("ANV_QUEUE_OVERRIDE", "gc=2", 0);
#endif
  // Le chargeur Vulkan que SDL a ouvert pour la fenêtre : libplacebo n'est pas toujours lié au
  // chargeur (le port vcpkg de Windows le construit sans, -Dvk-proc-addr=disabled), et le Mac
  // charge le sien par son chemin, plus haut.
  const auto get_proc_addr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
  if (get_proc_addr == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("SDL_Vulkan_GetVkGetInstanceProcAddr : {}", SDL_GetError()));
  }
  pl_vk_inst_params inst_params{};
  inst_params.get_proc_addr = get_proc_addr;
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
  vk_params.extra_queues = kVideoQueueFlags | kBackendQueueFlags;
  vk_params.queue_count = static_cast<int>(kPlaceboQueues);  // la file d'après reste aux moteurs
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
  impl->device.luid = device_luid(impl->vk);
  // Le fabricant se déduit du vendorID, jamais du nom : sans lui la télémétrie NVML, l'avertissement de
  // pilote et le choix de décodeur voient un fabricant inconnu.
  if (const auto properties_of = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
          impl->vk->get_proc_addr(impl->vk->instance, "vkGetPhysicalDeviceProperties"));
      properties_of != nullptr) {
    VkPhysicalDeviceProperties properties{};
    properties_of(impl->vk->phys_device, &properties);
    impl->device.pci_vendor = properties.vendorID;
    impl->device.pci_device = properties.deviceID;
    impl->device.name = properties.deviceName;
    if (impl->device.vendor() == GpuVendor::Nvidia) {
      // NVIDIA code sa version en 10.8.8.6 : « 615.78.08 ».
      const std::uint32_t v = properties.driverVersion;
      impl->device.driver_version = std::format("{}.{}.{:02}", v >> 22, (v >> 14) & 0xFF, (v >> 6) & 0xFF);
    }
  }
  const QueueRef engines = impl->device.vulkan->compute;
  if (impl->device.vulkan->lock_queue == nullptr) {
    impl->channel->info("moteurs sur leur propre file (famille {}, file {}), à côté de l'affichage",
                        engines.family, engines.index);
  } else {
    impl->channel->warn(
        "moteurs sur la file de l'affichage (famille {}) : l'appareil n'en a pas d'autre, chaque "
        "passe des moteurs retarde l'image affichée",
        engines.family);
  }
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
    constexpr VkMemoryPropertyFlags kVisibleLocal =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      const VkMemoryType& type = properties.memoryTypes[i];
      if ((type.propertyFlags & kVisibleLocal) == kVisibleLocal)
        facts.host_visible_vram_bytes = std::max<std::uint64_t>(facts.host_visible_vram_bytes,
                                                                properties.memoryHeaps[type.heapIndex].size);
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
