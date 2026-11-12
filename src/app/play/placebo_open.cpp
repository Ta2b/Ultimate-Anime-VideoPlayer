// Ouverture et fermeture de l'écran libplacebo : fenêtre SDL, instance et appareil Vulkan (les
// fonctionnalités de calcul du backend, Vulkan Video pour FFmpeg), swapchain, renderer, calques,
// et ce que l'appareil sait faire (device_facts). Présentation dans placebo_screen.cpp.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <libplacebo/cache.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>

#include "app/play/close_watch.h"
#include "app/play/host_facts.h"
#include "app/play/osd_text.h"
#include "app/play/placebo_impl.h"
#include "app/play/placebo_present.h"
#include "app/play/placebo_screen.h"
#include "app/play/queue_lanes.h"
#include "app/play/relaunch_target.h"
#include "app/play/window_size.h"
#include "core/clock.h"
#include "core/file_sync.h"
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

// Les espaces colorimétriques que la surface offre : dit, au journal, si un écran HDR peut recevoir du HDR10
// (A2B10G10R10 + HDR10_ST2084) ou si le pilote ne propose que du sRGB.
void log_surface_color_spaces(const pl_vulkan vk, VkSurfaceKHR surface) {
  const auto formats_of = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
  if (formats_of == nullptr || surface == VK_NULL_HANDLE) return;
  std::uint32_t count = 0;
  if (formats_of(vk->phys_device, surface, &count, nullptr) != VK_SUCCESS || count == 0) return;
  std::vector<VkSurfaceFormatKHR> formats(count);
  if (formats_of(vk->phys_device, surface, &count, formats.data()) != VK_SUCCESS) return;
  std::string text;
  bool hdr10 = false;
  for (const VkSurfaceFormatKHR& format : formats) {
    hdr10 = hdr10 || format.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT;
    text += std::format(" [format {} espace {}]", static_cast<int>(format.format),
                        static_cast<int>(format.colorSpace));
  }
  log_channel("app")->info("surface Vulkan : {} configurations, HDR10 (ST2084) {}{}",
                           formats.size(),  // i18n-ok : journal
                           hdr10 ? "offert" : "absent", text);
}

// A10-01 : un même message redit sans fin (une acquisition qui échoue à chaque tour) s'écrit une fois, puis
// son compte toutes les kRepeatSummaryUs ou au message suivant ; le journal ne se remplit plus de mille
// lignes par seconde.
constexpr std::int64_t kRepeatSummaryUs = 10'000'000;
struct RepeatedLog {
  std::mutex mutex;
  std::string last;
  std::int64_t repeats = 0;
  std::int64_t since_us = 0;
};

void placebo_log(void* priv, pl_log_level level, const char* message) {
  if (level > PL_LOG_WARN) return;
  auto* log = static_cast<spdlog::logger*>(priv);
  static RepeatedLog repeated;
  {
    const std::scoped_lock lock(repeated.mutex);
    const std::int64_t now = host_now_us();
    if (repeated.last == message && now - repeated.since_us < kRepeatSummaryUs) {
      ++repeated.repeats;
      return;
    }
    if (repeated.repeats > 0)
      log->warn("libplacebo : message précédent redit {} fois en {} s", repeated.repeats,
                (now - repeated.since_us) / 1'000'000);
    repeated.last = message;
    repeated.repeats = 0;
    repeated.since_us = now;
  }
  if (level <= PL_LOG_ERR) {
    log->error("libplacebo : {}", message);
  } else {
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
  bool shared = true;               // la file de calcul de libplacebo, sous son verrou
  std::uint32_t family_queues = 0;  // files de sa famille, toutes créées par libplacebo ; 0 : aucune voie
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
      return EngineQueue{.queue = {.family = family, .index = kPlaceboQueues},
                         .shared = false,
                         .family_queues = families[family].queueCount};
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
  // Les files qui suivent celle du backend, créées par libplacebo (toute la famille) et qu'il
  // n'emploie pas : des voies de calcul en plus (au plus deux), seulement hors d'une file partagée.
  // Comptées sur la famille, pas sur pl_vulkan_queue::count, qui ne compte que le pool de libplacebo
  // (B10-04). FFmpeg ne reçoit que la file 0 des familles de libplacebo (vulkan_share.cpp) : ces files sont
  // libres.
  if (!engines.shared) d.compute_lanes = backend_lanes(engines.family_queues, d.compute.index);
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

// Le cache des shaders de libplacebo : au plus 64 Mo, relu d'un fichier à l'ouverture. Un fichier abîmé ou
// d'une autre version de libplacebo est simplement ignoré (pl_cache_load le refuse) et sera réécrit.

// Les « <cache>.<n>.part » d'une fermeture interrompue (save_shader_cache) : effacés passé une heure, le
// temps qu'un autre lecteur finisse d'écrire le sien (A10-11).
void purge_cache_parts(const std::filesystem::path& file) {
  std::error_code error;
  const std::u8string prefix = file.filename().u8string() + u8".";
  const auto now = std::filesystem::file_time_type::clock::now();
  for (std::filesystem::directory_iterator it(file.parent_path(), error), end; !error && it != end;
       it.increment(error)) {
    const std::u8string name = it->path().filename().u8string();
    if (!name.starts_with(prefix) || !name.ends_with(u8".part")) continue;
    std::error_code ignored;
    const auto written = std::filesystem::last_write_time(it->path(), ignored);
    if (!ignored && now - written > std::chrono::hours(1)) std::filesystem::remove(it->path(), ignored);
  }
}

// B37-01 : sur un poste à deux cartes (portable RTX + iGPU : Optimus, PRIME), libplacebo prend la première
// que le système lui donne, souvent l'iGPU, et TensorRT refuse alors la carte. Sans carte demandée, la carte
// NVIDIA RTX discrète est préférée ; seulement quand une autre carte existe, un poste à une carte ne change
// pas. Rien si elle ne peut pas présenter : pl_vulkan_create échoue et le choix automatique reprend.
std::string preferred_rtx_name(pl_vk_inst inst) {
  constexpr std::uint32_t kNvidiaPci = 0x10DE;
  const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      inst->get_proc_addr(inst->instance, "vkEnumeratePhysicalDevices"));
  const auto properties_of = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      inst->get_proc_addr(inst->instance, "vkGetPhysicalDeviceProperties"));
  if (enumerate == nullptr || properties_of == nullptr) return {};
  std::uint32_t count = 0;
  if (enumerate(inst->instance, &count, nullptr) != VK_SUCCESS || count < 2) return {};
  std::vector<VkPhysicalDevice> devices(count);
  if (enumerate(inst->instance, &count, devices.data()) != VK_SUCCESS) return {};
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties properties{};
    properties_of(device, &properties);
    if (properties.vendorID == kNvidiaPci && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        std::string_view(properties.deviceName).contains("RTX"))
      return properties.deviceName;
  }
  return {};
}

// La carte à demander à libplacebo : celle des options, sinon la RTX préférée (nul : choix automatique). Son
// nom reste dans l'Impl, vivant jusqu'à pl_vulkan_create.
template <class ScreenImpl>  // PlaceboScreen::Impl est privé : le gabarit évite de le nommer
const char* pick_device(const ScreenOptions& options, ScreenImpl& impl) {
  if (!options.device_name.empty()) return options.device_name.c_str();
  if (options.allow_software) return nullptr;
  impl.device_choice = preferred_rtx_name(impl.inst);
  if (impl.device_choice.empty()) return nullptr;
  impl.channel->info("plusieurs cartes : la carte NVIDIA RTX « {} » est préférée", impl.device_choice);
  return impl.device_choice.c_str();
}

}  // namespace

void PlaceboScreen::Impl::open_shader_cache(const std::filesystem::path& file) {
  Impl& impl = *this;
  if (file.empty()) return;
  pl_cache_params params = pl_cache_default_params;
  params.log = impl.log;
  params.max_total_size = 64u << 20u;
  impl.shader_cache = pl_cache_create(&params);
  if (impl.shader_cache == nullptr) return;
  impl.shader_cache_file = file;
  purge_cache_parts(file);
  std::error_code error;
  const std::uintmax_t on_disk = std::filesystem::file_size(file, error);
  if (!error && on_disk > params.max_total_size) {
    // Jamais écrit par nous à cette taille : abîmé ou d'ailleurs, il n'est pas lu en mémoire (A10-11).
    impl.channel->warn("cache des shaders : {} Mo sur le disque, plus que le plafond, ignoré",
                       on_disk >> 20u);
  } else if (std::ifstream in(file, std::ios::binary); !error && in) {
    // D'un seul bloc, à la taille du fichier : jusqu'à 64 Mo sur le chemin de l'ouverture de la fenêtre.
    std::vector<char> bytes(static_cast<std::size_t>(on_disk));
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    bytes.resize(static_cast<std::size_t>(in.gcount()));
    const int loaded =
        pl_cache_load(impl.shader_cache, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    impl.channel->info("cache des shaders : {} objets relus ({} Ko)", std::max(loaded, 0),
                       bytes.size() >> 10U);
  }
  impl.shader_cache_loaded = pl_cache_signature(impl.shader_cache);
  pl_gpu_set_cache(impl.vk->gpu, impl.shader_cache);
}

// Gardé à la fermeture, par l'écriture atomique du projet (core::write_atomic : à côté, sur le disque, puis
// renommé) : une fermeture brutale ne laisse jamais un fichier à moitié écrit. Un cache que la lecture n'a
// pas changé n'est pas réécrit (B36-05 : jusqu'à 64 Mo à chaque fermeture).
void PlaceboScreen::Impl::save_shader_cache() {
  Impl& impl = *this;
  if (impl.shader_cache == nullptr) return;
  if (impl.vk != nullptr) pl_gpu_set_cache(impl.vk->gpu, nullptr);
  const std::size_t size = pl_cache_save(impl.shader_cache, nullptr, 0);
  if (size > 0 && pl_cache_signature(impl.shader_cache) != impl.shader_cache_loaded) {
    std::vector<std::byte> bytes(size);
    pl_cache_save(impl.shader_cache, reinterpret_cast<std::uint8_t*>(bytes.data()), bytes.size());
    if (auto written = core::write_atomic(impl.shader_cache_file, bytes); !written)
      impl.channel->warn("cache des shaders non écrit : {}", written.error().detail);
  }
  pl_cache_destroy(&impl.shader_cache);
}

namespace {
// Smooth Motion : le chargeur Vulkan lit la variable à la création de l'instance ; le thread main est encore
// seul à toucher l'environnement à ce moment. B34-06 : c'est la voie du pilote Linux ; sous Windows, Smooth
// Motion se règle par profil dans l'application NVIDIA, et le réglage reste lu (play.toml) sans rien
// demander.
void request_smooth_motion(spdlog::logger& log) {
#if defined(__linux__)
  SDL_setenv_unsafe("NVPRESENT_ENABLE_SMOOTH_MOTION", "1", 1);
  log.info("Smooth Motion : couche du pilote NVIDIA demandée (NVPRESENT_ENABLE_SMOOTH_MOTION=1)");
#else
  log.info("Smooth Motion : réglé par l'application NVIDIA sous Windows, rien n'est demandé ici");
#endif
}

// Le démontage d'un pilote sain se compte en secondes (moteurs TensorRT compris) ; au-delà, il ne finira pas.
constexpr std::chrono::seconds kCloseLimit{60};
}  // namespace

PlaceboScreen::Impl::~Impl() {
  // La fermeture : la fenêtre est d'abord cachée et le compositeur attendu (SDL_SyncWindow), avant le
  // démontage du GPU, qui peut durer (pilote NVIDIA, moteurs TensorRT). Sous Wayland, une fenêtre que le
  // processus garde encore pendant ce démontage reste affichée avec son dernier tampon (contours, ombre des
  // décorations côté client) ; cachée d'abord, elle disparaît tout de suite. Le temps du démontage est au
  // journal.
  const auto closing_since = std::chrono::steady_clock::now();
  if (window != nullptr) {
    SDL_HideWindow(window);
    SDL_SyncWindow(window);
  }
  // A10-04 : tout ce démontage attend le GPU sans délai. Sur un GPU bloqué, le lecteur sort au lieu de
  // rester sans fenêtre pour toujours ; le journal dit pourquoi.
  const CloseWatch stuck(kCloseLimit, [] {
    exit_stuck(std::format("fermeture de l'écran bloquée depuis {} s", kCloseLimit.count()));
  });
  // Le veilleur d'écriture attend sur un sémaphore de l'appareil : arrêté avant tout le reste, quel que soit
  // le chemin qui ferme l'écran (sortie anticipée, exception), jamais après pl_vulkan_destroy.
  ring.set_watcher(nullptr);
  watcher.reset();
  if (vk != nullptr) pl_gpu_finish(vk->gpu);
  for (pl_tex& tex : offscreen_tex)
    if (tex != nullptr) pl_tex_destroy(vk->gpu, &tex);
  if (composed != nullptr) pl_tex_destroy(vk->gpu, &composed);
  share.reset();  // avant l'appareil qu'il prête
  if (!grade_gpu_us.empty()) {
    std::vector<std::uint32_t>& kept = grade_gpu_us.kept();
    std::ranges::sort(kept);
    const auto at = [&](double q) {
      return kept[static_cast<std::size_t>(q * static_cast<double>(kept.size() - 1))];
    };
    channel->info("profil GPU · grade HDR : médiane {:.2f} ms, p95 {:.2f} ms ({} images, centiles sur {})",
                  at(0.5) / 1000.0, at(0.95) / 1000.0, grade_gpu_us.seen(), kept.size());
  }
  if (recompositions + reuses > 0)
    channel->info("composition gardee : {} recompositions, {} recopies sans recomposition", recompositions,
                  reuses);
  ui.reset();
  grade.reset();
  ramp_grade.reset();
  pl_renderer_destroy(&ramp_renderer);
  osd.reset();
  overlays.reset();
  ring.clear(vk != nullptr ? vk->gpu : nullptr);
  pl_renderer_destroy(&renderer);
  save_shader_cache();
  feedback.reset();
  pl_swapchain_destroy(&swapchain);
  pl_vulkan_destroy(&vk);
  if (surface != VK_NULL_HANDLE) SDL_Vulkan_DestroySurface(inst->instance, surface, nullptr);
  pl_vk_inst_destroy(&inst);
  pl_log_destroy(&log);
  if (window != nullptr) SDL_DestroyWindow(window);
  // La destruction de la surface part au compositeur avant que le processus ne rende la main à SDL_Quit.
  SDL_PumpEvents();
  channel->info("fermeture : {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - closing_since)
                                         .count());
}

// NOLINTNEXTLINE(readability-function-size)
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
  // Essais silencieux : une fenêtre minuscule, sans bordure ni focus. Une fenêtre cachée (SDL_WINDOW_HIDDEN)
  // ne reçoit plus d'appels de rafraîchissement du compositeur : ses présentations tombent à ~20 par seconde,
  // ce qui fausse toute mesure.
  if (options.hidden) flags |= SDL_WINDOW_BORDERLESS | SDL_WINDOW_NOT_FOCUSABLE | SDL_WINDOW_UTILITY;
  // Hors écran : rien n'est jamais présenté, la fenêtre (qui porte la surface Vulkan) reste cachée.
  if (options.offscreen) flags = (flags & ~SDL_WINDOW_FULLSCREEN) | SDL_WINDOW_HIDDEN;
#ifdef _WIN32
  // Windows montre la fenêtre dès sa création : pendant l'ouverture de Vulkan (une à deux secondes), on
  // voyait une fenêtre vide, puis, après la mise à l'échelle du bureau (fit_display_scale), l'ancienne image
  // dans un coin. Elle reste cachée jusqu'à ce que la swapchain ait sa taille, à la fin de l'ouverture.
  const bool show_when_ready = !options.offscreen && !options.hidden;
  if (show_when_ready) flags |= SDL_WINDOW_HIDDEN;
#endif
  impl->window = SDL_CreateWindow(options.title.c_str(), options.width, options.height, flags);
  if (impl->window == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("SDL_CreateWindow : {}", SDL_GetError()));
  }
  impl->fullscreen = options.fullscreen && !options.offscreen;
  impl->offscreen = options.offscreen;
  impl->hdr_wanted = options.hdr;
  impl->hdr_allowed = options.hdr;
  impl->kde_sdr_nits = options.kde_sdr_nits;
  impl->kde_sdr_auto = options.kde_sdr_auto;
  impl->offscreen_size = {options.offscreen_width, options.offscreen_height};
  if (const double hz = display_refresh_hz(impl->window); hz > 1.0)
    impl->period_us = static_cast<std::int64_t>(1e6 / hz);
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
  if (options.smooth_motion) request_smooth_motion(*impl->channel);
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
  ComputeFeatures compute;
  // Retour de présentation (chantier 6.2) : rien de plus demandé à --present=open.
  PresentFeatures present_features;
  if (options.present != PresentWanted::Open) compute.video2.pNext = present_features.head(nullptr);
  vk_params.features = &compute.features;
  std::vector<const char*> extensions(kOptionalExtensions.begin(), kOptionalExtensions.end());
  for (const char* extension : present_extensions(options.present)) extensions.push_back(extension);
  vk_params.opt_extensions = extensions.data();
  vk_params.num_opt_extensions = static_cast<int>(extensions.size());
  vk_params.extra_queues = kVideoQueueFlags | kBackendQueueFlags;
  vk_params.queue_count = static_cast<int>(kPlaceboQueues);  // la file d'après reste aux moteurs
  vk_params.device_name = pick_device(options, *impl);
  impl->vk = pl_vulkan_create(impl->log, &vk_params);
  if (impl->vk == nullptr && vk_params.device_name != nullptr) {
    // La carte choisie n'est plus là (eGPU débranché, pilote changé) : libplacebo choisit (P3.16).
    impl->channel->warn("carte « {} » introuvable : choix automatique", vk_params.device_name);
    vk_params.device_name = nullptr;
    impl->vk = pl_vulkan_create(impl->log, &vk_params);
  }
  if (impl->vk == nullptr)
    return refuse(RefusalCode::DeviceMismatch, "pl_vulkan_create : aucun appareil ne convient");
  const pl_gpu gpu = impl->vk->gpu;
  if (auto share = VulkanShare::create(impl->inst, impl->vk)) {
    impl->share = std::move(*share);
  } else {
    impl->channel->info("appareil Vulkan non partagé avec FFmpeg : {} ; ses images passeront par l'hôte",
                        share.error().detail);
  }
  if ((gpu->export_caps.tex & kRingHandle) == 0 || (gpu->export_caps.sync & kRingHandle) == 0) {
    // Sans export, le backend TensorRT ne peut pas importer le ring ; le backend Vulkan s'en passe. La
    // poignée est celle du système (kRingHandle) : sous Windows, un HANDLE NT, jamais un descripteur.
    impl->channel->info("l'appareil n'exporte pas ses images ni ses sémaphores ({}) : ring sans poignées",
                        kRingHandle == PL_HANDLE_FD ? "fd" : "HANDLE NT");
  }

  // Le niveau de retour de présentation (present_level.h) : à --present=open, la swapchain d'aujourd'hui.
  impl->present_caps =
      options.present == PresentWanted::Open ? PresentCaps{} : read_present_caps(impl->vk, impl->surface);
  // wp_presentation (placebo_wayland.h), seulement s'il peut servir.
  std::unique_ptr<WaylandPresentation> wayland;
  if (wants_wayland(options.present)) wayland = WaylandPresentation::open(impl->window, impl->channel);
  impl->present_caps.wayland_presentation = wayland != nullptr;
  impl->present_level = choose_present_level(impl->present_caps, options.present);
  if (options.present != PresentWanted::Open) {
    const PresentCaps& c = impl->present_caps;
    impl->channel->info(
        "présentation : niveau {} (present_id2 {}, present_wait2 {}, present timing {}, date "
        "absolue {}, horodatages calibrés {}, wp_presentation {})",
        name(impl->present_level), c.present_id2, c.present_wait2, c.timing, c.timing_absolute, c.calibrated,
        c.wayland_presentation);
  }
  const SwapchainShape shape = swapchain_shape(impl->present_level);
  pl_vulkan_swapchain_params sw_params{};
  sw_params.surface = impl->surface;
  sw_params.present_mode = VK_PRESENT_MODE_FIFO_KHR;
  sw_params.swapchain_depth = shape.depth > 0 ? shape.depth : kSwapchainDepth;
  sw_params.min_images = shape.min_images;
  sw_params.present_flags = present_flags(impl->present_level, impl->present_caps);
  impl->swapchain = pl_vulkan_create_swapchain(impl->vk, &sw_params);
  if (impl->swapchain == nullptr)
    return refuse(RefusalCode::LibraryUnavailable, "pl_vulkan_create_swapchain a échoué");
  if (impl->present_level != PresentLevel::Open)
    impl->feedback = std::make_unique<PresentFeedback>(impl->vk, impl->present_level, impl->present_caps,
                                                       impl->channel, std::move(wayland));
  if (impl->feedback) impl->feedback->record_stamps(options.screen_stamps);  // 0.7 : <stats>.screen.tsv
  log_surface_color_spaces(impl->vk, impl->surface);
  if (options.hdr) pl_swapchain_colorspace_hint(impl->swapchain, &pl_color_space_hdr10);
  impl->open_shader_cache(options.shader_cache);
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
  if (auto sharpen = render::Sharpen::create(gpu)) {
    impl->sharpen.emplace(std::move(*sharpen));
  } else {
    impl->channel->warn("netteté indisponible : {}", sharpen.error().detail);
  }
  impl->source = pl_color_space_bt709;
#ifdef _WIN32
  if (show_when_ready) {
    int width = 0;
    int height = 0;
    SDL_GetWindowSizeInPixels(impl->window, &width, &height);
    if (width > 0 && height > 0) pl_swapchain_resize(impl->swapchain, &width, &height);
    SDL_ShowWindow(impl->window);
    SDL_RaiseWindow(impl->window);
  }
#endif
  return std::unique_ptr<PlaceboScreen>(new PlaceboScreen(std::move(impl)));
}

std::vector<std::string> PlaceboScreen::device_names() const {
  std::vector<std::string> names;
  const pl_vulkan vk = impl_->vk;
  const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      vk->get_proc_addr(vk->instance, "vkEnumeratePhysicalDevices"));
  const auto properties_of = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceProperties"));
  if (enumerate == nullptr || properties_of == nullptr) return names;
  std::uint32_t count = 0;
  if (enumerate(vk->instance, &count, nullptr) != VK_SUCCESS) return names;
  std::vector<VkPhysicalDevice> devices(count);
  if (enumerate(vk->instance, &count, devices.data()) != VK_SUCCESS) return names;
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties properties{};
    properties_of(device, &properties);
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;  // lavapipe n'est pas une carte
    names.emplace_back(properties.deviceName);
  }
  return names;
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
    facts.driver = std::to_string(properties.driverVersion);
  }
  const auto properties2_of = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceProperties2"));
  if (properties2_of != nullptr) {
    VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &driver};
    properties2_of(vk->phys_device, &properties);
    facts.mesa_nvk = driver.driverID == VK_DRIVER_ID_MESA_NVK;
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
