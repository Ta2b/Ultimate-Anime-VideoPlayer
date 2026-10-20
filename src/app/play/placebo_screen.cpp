#include "app/play/placebo_screen.h"

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
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

// Blanc de l'interface dans une sortie HDR : le blanc de référence (BT.2408), comme le SDR.
constexpr float kUiWhiteNits = 203.0F;
// Nits d'un code PQ (ST 2084).
double pq_nits(double e) {
  const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / 78.84375);
  return 10000.0 * std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

// L'exposant du transfert d'une cible SDR BT.709 ; rien si le grade ne sait pas y travailler.
std::optional<double> sdr_gamma(const pl_color_space& color) {
  if (color.primaries != PL_COLOR_PRIM_BT_709) return std::nullopt;
  switch (color.transfer) {
    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_GAMMA22:
      return 2.2;
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_GAMMA24:
      return 2.4;
    default:
      return std::nullopt;
  }
}

// Le grade d'une cible : celui du PQ, ou celui du SDR à l'exposant de la cible.
std::optional<render::GradeParams> grade_for(const render::GradeChoice& choice, const pl_color_space& color) {
  if (color.transfer == PL_COLOR_TRC_PQ) return choice.pq;
  const std::optional<double> gamma = sdr_gamma(color);
  if (!gamma || !choice.sdr) return std::nullopt;
  render::GradeParams params = *choice.sdr;
  params.sdr_gamma = *gamma;
  return params;
}

// Images gardées pour le profil GPU du grade (une heure à 165 Hz tient en dessous).
constexpr std::size_t kMaxGradeSamples = 1 << 20;

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

pl_color_space color_space_of(const ColorInfo& color) {
  pl_color_space space{};
  switch (color.primaries) {
    case Primaries::Bt709:
      space.primaries = PL_COLOR_PRIM_BT_709;
      break;
    case Primaries::Bt601_625:
      space.primaries = PL_COLOR_PRIM_BT_601_625;
      break;
    case Primaries::Bt601_525:
      space.primaries = PL_COLOR_PRIM_BT_601_525;
      break;
    case Primaries::Bt2020:
      space.primaries = PL_COLOR_PRIM_BT_2020;
      break;
    case Primaries::DciP3:
      space.primaries = PL_COLOR_PRIM_DCI_P3;
      break;
  }
  switch (color.transfer) {
    case Transfer::Bt1886:
      space.transfer = PL_COLOR_TRC_BT_1886;
      break;
    case Transfer::Srgb:
      space.transfer = PL_COLOR_TRC_SRGB;
      break;
    case Transfer::Linear:
      space.transfer = PL_COLOR_TRC_LINEAR;
      break;
    case Transfer::Pq:
      space.transfer = PL_COLOR_TRC_PQ;
      break;
    case Transfer::Hlg:
      space.transfer = PL_COLOR_TRC_HLG;
      break;
  }
  if (color.hdr) {  // crête et lumière de la source (TrueHDR : le pic qu'on lui a donné)
    space.hdr.max_luma = color.hdr->max_luminance;
    space.hdr.min_luma = color.hdr->min_luminance;
    space.hdr.max_cll = color.hdr->max_cll;
    space.hdr.max_fall = color.hdr->max_fall;
  }
  return space;
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
// coopératives, additions atomiques de float), activé par libplacebo seulement là où le pilote le
// propose.
struct ComputeFeatures {
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
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

constexpr std::array<const char*, 2> kComputeExtensions{VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
                                                        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME};

}  // namespace

PlaceboScreen::PlaceboScreen(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PlaceboScreen::~PlaceboScreen() = default;

PlaceboScreen::Impl::~Impl() {
  if (vk != nullptr) pl_gpu_finish(vk->gpu);
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
  vk_params.opt_extensions = kComputeExtensions.data();
  vk_params.num_opt_extensions = static_cast<int>(kComputeExtensions.size());
  impl->vk = pl_vulkan_create(impl->log, &vk_params);
  if (impl->vk == nullptr)
    return refuse(RefusalCode::DeviceMismatch, "pl_vulkan_create : aucun appareil ne convient");
  const pl_gpu gpu = impl->vk->gpu;
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

const GpuDevice& PlaceboScreen::device() const {
  return impl_->device;
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

SDL_Window* PlaceboScreen::window() const {
  return impl_->window;
}

std::string PlaceboScreen::output_description() const {
  return impl_->hdr_output ? "sortie HDR10 (PQ BT.2020)" : "sortie SDR";
}

std::expected<std::vector<RingSlotDesc>, Refusal> PlaceboScreen::create_ring(int width, int height,
                                                                             std::uint32_t count) {
  return impl_->ring.create(impl_->vk->gpu, width, height, count);
}

void PlaceboScreen::set_source_color(const ColorInfo& color) {
  impl_->source = color_space_of(color);
}

void PlaceboScreen::set_grade(const render::GradeChoice& choice) {
  impl_->grade_choice = choice;
  impl_->select_grade(impl_->target_color);
}

void PlaceboScreen::Impl::select_grade(const pl_color_space& target) {
  const std::optional<render::GradeParams> params = grade_for(grade_choice, target);
  grade_loaded_pq.reset();
  if (!grade || !params) return;
  if (*params != grade->params()) grade->set_params(*params);
  grade_loaded_pq = target.transfer == PL_COLOR_TRC_PQ;
}

bool PlaceboScreen::Impl::graded(const pl_color_space& target) const {
  const bool pq = target.transfer == PL_COLOR_TRC_PQ;
  return grade && grade_loaded_pq == pq && (pq || sdr_gamma(target).has_value());
}

std::expected<std::vector<double>, Refusal> PlaceboScreen::grade_ramp(std::span<const float> levels) {
  Impl& s = *impl_;
  if (!s.grade || !s.graded(s.target_color))
    return refuse(RefusalCode::HdrUnavailable, "rampe du vrai noir : pas de grade sur cette sortie");
  const pl_gpu gpu = s.vk->gpu;
  if (s.ramp_renderer == nullptr) s.ramp_renderer = pl_renderer_create(s.log, gpu);
  if (!s.ramp_grade) {
    auto grade = render::Grade::create(gpu);
    if (!grade) return std::unexpected(std::move(grade.error()));
    s.ramp_grade.emplace(std::move(*grade));
  }
  render::GradeParams params = s.grade->params();
  params.black_lo = 0.0;  // le vrai noir se cale sur ce qui sort sans lui
  params.black_hi = 0.0;
  s.ramp_grade->set_params(params);
  // Une bande de 16 px par marche, 128 lignes : assez pour les cartes 1/8 et 1/32 du grade.
  constexpr int kBand = 16;
  constexpr int kRows = 128;
  const int width = kBand * static_cast<int>(levels.size());
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(width) * kRows * 4);
  for (int y = 0; y < kRows; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto value = static_cast<std::uint16_t>(
          std::lround(std::clamp(levels[static_cast<std::size_t>(x / kBand)], 0.0F, 1.0F) * 65535.0F));
      const std::size_t at =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4;
      pixels[at] = pixels[at + 1] = pixels[at + 2] = value;
      pixels[at + 3] = 65535;
    }
  }
  pl_tex_params tex{};
  tex.w = width;
  tex.h = kRows;
  tex.format = pl_find_named_fmt(gpu, "rgba16");
  tex.sampleable = true;
  tex.host_writable = true;
  tex.initial_data = pixels.data();
  pl_tex in = pl_tex_create(gpu, &tex);
  tex.initial_data = nullptr;
  tex.sampleable = false;
  tex.host_writable = false;
  tex.renderable = true;
  tex.host_readable = true;
  pl_tex out = pl_tex_create(gpu, &tex);
  const auto frame_of = [&](pl_tex texture, const pl_color_space& color) {
    pl_frame frame{};
    frame.num_planes = 1;
    frame.planes[0].texture = texture;
    frame.planes[0].components = 4;
    for (int c = 0; c < 4; ++c) frame.planes[0].component_mapping[c] = c;
    frame.repr = pl_color_repr_rgb;
    frame.color = color;
    frame.crop = {0, 0, static_cast<float>(width), static_cast<float>(kRows)};
    return frame;
  };
  bool done = in != nullptr && out != nullptr;
  if (done) {
    const pl_frame image = frame_of(in, s.source);  // comme une image du ring
    const pl_frame target = frame_of(out, s.target_color);
    pl_render_params render = pl_render_default_params;
    render.dither_params = nullptr;
    const std::span<const pl_hook* const> hooks = s.ramp_grade->hooks();
    render.hooks = hooks.data();
    render.num_hooks = static_cast<int>(hooks.size());
    done = pl_render_image(s.ramp_renderer, &image, &target, &render);
    pl_tex_transfer_params download{};
    download.tex = out;
    download.ptr = pixels.data();
    done = done && pl_tex_download(gpu, &download);
  }
  pl_tex_destroy(gpu, &in);
  pl_tex_destroy(gpu, &out);
  if (!done) return refuse(RefusalCode::VendorCallFailed, "rampe du vrai noir : rendu ou relecture");
  std::vector<double> nits;
  nits.reserve(levels.size());
  const std::size_t middle = static_cast<std::size_t>(kRows / 2) * static_cast<std::size_t>(width);
  for (std::size_t i = 0; i < levels.size(); ++i) {
    const std::size_t at = (middle + i * kBand + kBand / 2) * 4;
    const std::uint16_t peak = std::max({pixels[at], pixels[at + 1], pixels[at + 2]});
    const double code = peak / 65535.0;
    nits.push_back(s.hdr_output ? pq_nits(code) : std::pow(code, params.sdr_gamma) * params.peak);
  }
  return nits;
}

bool PlaceboScreen::hdr_output() const {
  return impl_->hdr_output;
}

void PlaceboScreen::flush() {
  pl_gpu_flush(impl_->vk->gpu);
}

void PlaceboScreen::snapshot_next(std::filesystem::path path, int count) {
  impl_->snapshot = std::move(path);
  impl_->snapshot_count = std::max(count, 1);
  impl_->snapshot_index = 0;
}

void PlaceboScreen::release_ui(ImGuiPlatformIO& platform) {
  if (impl_->ui) impl_->ui->release_textures(platform);
}

void PlaceboScreen::Impl::draw_ui(const ImDrawData* data, const pl_frame& target) {
  if (data == nullptr || ui_failed) return;
  if (auto drawn = ui->draw(*data, target, kUiWhiteNits); !drawn) {
    channel->warn("interface : {} {}", name(drawn.error().code), drawn.error().detail);
    ui_failed = true;
  }
}

std::expected<void, Refusal> PlaceboScreen::Impl::compose(const pl_frame* image,
                                                          const pl_frame& target) const {
  pl_render_params params = pl_render_default_params;
  // Le grade chargé est celui du PQ ou du SDR : sur une autre cible (instantané SDR d'une sortie
  // HDR10), il est sauté.
  const bool graded_now = grade.has_value() && graded(target.color);
  if (graded_now) {
    const std::span<const pl_hook* const> hooks = grade->hooks();
    params.hooks = hooks.data();
    params.num_hooks = static_cast<int>(hooks.size());
    params.info_callback = [](void* priv, const pl_render_info* info) {
      const char* what = info->pass->shader->description;
      if (what != nullptr && std::string_view(what).find("AnimeJaNai") != std::string_view::npos)
        static_cast<const Impl*>(priv)->grade_frame_ns += info->pass->last;
    };
    params.info_priv = const_cast<Impl*>(this);
    grade_frame_ns = 0;
  }
  if (image == nullptr) pl_frame_clear(vk->gpu, &target, std::array<float, 3>{0.0F, 0.0F, 0.0F}.data());
  if (!pl_render_image(renderer, image, &target, &params)) {
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image a échoué"});
  }
  if (graded_now && image != nullptr && grade_frame_ns > 0 && grade_gpu_us.size() < kMaxGradeSamples)
    grade_gpu_us.push_back(static_cast<std::uint32_t>(grade_frame_ns / 1000));
  return {};
}

OutputSize PlaceboScreen::size() const {
  int w = 0;
  int h = 0;
  SDL_GetWindowSizeInPixels(impl_->window, &w, &h);
  return {w, h};
}

void PlaceboScreen::set_fullscreen(bool on) {
  if (SDL_SetWindowFullscreen(impl_->window, on)) impl_->fullscreen = on;
}

bool PlaceboScreen::fullscreen() const {
  return impl_->fullscreen;
}

SlotFree PlaceboScreen::retire(const RingFrame& frame) {
  return impl_->ring.retire(impl_->vk->gpu, frame);
}

std::expected<ScreenTiming, Refusal> PlaceboScreen::show(const ShowRequest& request) {
  Impl& s = *impl_;
  const pl_gpu gpu = s.vk->gpu;
  const std::int64_t start = host_now_us();
  OutputSize out = size();
  if (out != s.swapchain_size) {
    int w = out.width;
    int h = out.height;
    if (!pl_swapchain_resize(s.swapchain, &w, &h)) return ScreenTiming{.presented = false};
    s.swapchain_size = out;
  }
  pl_swapchain_frame swframe{};
  if (!pl_swapchain_start_frame(s.swapchain, &swframe))
    return ScreenTiming{.presented = false};  // fenêtre cachée
  s.hdr_output = swframe.color_space.transfer == PL_COLOR_TRC_PQ;
  s.target_color = swframe.color_space;
  s.select_grade(s.target_color);
  pl_frame target{};
  pl_frame_from_swapchain(&target, &swframe);

  pl_frame image{};
  if (request.frame != nullptr) {
    auto tex = s.ring.acquire(gpu, *request.frame);
    if (!tex) return std::unexpected(std::move(tex.error()));
    image.num_planes = 1;
    image.planes[0].texture = *tex;
    image.planes[0].components = 3;
    image.planes[0].component_mapping[0] = 0;
    image.planes[0].component_mapping[1] = 1;
    image.planes[0].component_mapping[2] = 2;
    image.repr = pl_color_repr_rgb;
    image.color = s.source;
    image.crop = {0, 0, static_cast<float>((*tex)->params.w), static_cast<float>((*tex)->params.h)};
    pl_rect2df_aspect_copy(&target.crop, &image.crop, 0.0F);
  }

  auto overlays = s.overlays->build(request.subtitles, request.osd.empty() ? nullptr : &request.osd, *s.osd,
                                    out, request.ui != nullptr ? request.ui_bottom : 0,
                                    request.ui != nullptr ? request.ui_top : 0);
  if (!overlays) return std::unexpected(std::move(overlays.error()));
  target.overlays = overlays->data();
  target.num_overlays = static_cast<int>(overlays->size());

  if (auto drawn = s.compose(request.frame != nullptr ? &image : nullptr, target); !drawn) {
    return std::unexpected(std::move(drawn.error()));
  }
  s.draw_ui(request.ui, target);
  if (s.snapshot) {
    std::filesystem::path path = *s.snapshot;
    if (s.snapshot_count > 1) {
      // « nom.pq.ppm » → « nom-001.pq.ppm », « nom.ppm » → « nom-001.ppm ».
      const std::string file = path.filename().string();
      const std::size_t dot = file.ends_with(".pq.ppm") ? file.size() - 7 : file.rfind('.');
      path.replace_filename(std::format("{}-{:03}{}", file.substr(0, dot), s.snapshot_index + 1,
                                        dot == std::string::npos ? std::string() : file.substr(dot)));
    }
    if (++s.snapshot_index >= s.snapshot_count) s.snapshot.reset();
    if (auto saved =
            s.save_snapshot(request.frame != nullptr ? &image : nullptr, target, out, request.ui, path);
        !saved) {
      s.channel->warn("instantané {} : {}", path.string(), saved.error().detail);
    }
  }
  if (!pl_swapchain_submit_frame(s.swapchain))
    return refuse(RefusalCode::VendorCallFailed, "pl_swapchain_submit_frame");
  const std::int64_t submitted = host_now_us();
  pl_swapchain_swap_buffers(s.swapchain);  // FIFO : attend une place en file quand elle est pleine
  const std::int64_t presented = host_now_us();
  return ScreenTiming{.render_us = submitted - start,
                      .present_us = presented,
                      .queue_wait_us = presented - submitted,
                      .presented = true};
}

}  // namespace ajp::app::play
