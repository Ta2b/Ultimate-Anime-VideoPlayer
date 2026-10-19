// Test de fumée sur un vrai écran : fenêtre SDL3 petite et brève (jamais en plein écran),
// surface sur une instance libplacebo, swapchain FIFO, quelques images effacées présentées,
// redimensionnement, période de vsync mesurée, état HDR lu. Se SKIP avec la raison quand il n'y
// a ni écran, ni Vulkan, ni appareil qui présente sur la surface.
#include "present/window.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>
#include <catch2/catch_test_macros.hpp>
#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "present/display_info.h"
#include "present/display_system.h"
#include "present/surface.h"
#include "present/swapchain.h"
#include "present/vk_handle.h"

using ajp::RefusalCode;
using ajp::present::PixelSize;
using ajp::present::Surface;
using ajp::present::Swapchain;
using ajp::present::Window;

namespace {

// Ce que libplacebo crée pour le test, détruit dans l'ordre inverse de la création.
struct Placebo {
  pl_log log = nullptr;
  pl_vk_inst instance = nullptr;
  pl_vulkan vulkan = nullptr;
  Placebo() = default;
  Placebo(const Placebo&) = delete;
  Placebo& operator=(const Placebo&) = delete;
  ~Placebo() {
    pl_vulkan_destroy(&vulkan);
    pl_vk_inst_destroy(&instance);
    pl_log_destroy(&log);
  }
};

bool software_device(const Placebo& placebo) {
  const auto properties_of = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      placebo.instance->get_proc_addr(placebo.instance->instance, "vkGetPhysicalDeviceProperties"));
  VkPhysicalDeviceProperties properties{};
  properties_of(placebo.vulkan->phys_device, &properties);
  return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
}

void pump_events() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
  }
}

struct Presented {
  PixelSize size;
  ajp::present::VsyncTick tick;  // celle de la dernière image
};

// n images effacées puis présentées, à 10 Hz au moins ; rend le nombre d'images obtenues.
int present_frames(Swapchain& swapchain, pl_gpu gpu, int n, Presented& last) {
  int presented = 0;
  const std::uint64_t deadline = SDL_GetTicks() + 2000 + 100 * static_cast<std::uint64_t>(n);
  while (presented < n && SDL_GetTicks() < deadline) {
    pump_events();
    const auto frame = swapchain.start_frame();
    if (!frame) {
      SDL_Delay(2);
      continue;
    }
    const float level = presented % 2 == 0 ? 0.25F : 0.5F;
    const float color[4] = {level, 0.0F, 1.0F - level, 1.0F};
    pl_tex_clear(gpu, frame->target->planes[0].texture, color);
    REQUIRE(swapchain.submit_frame());
    if (presented > 0) CHECK(frame->tick.index > last.tick.index);
    last = {.size = frame->size, .tick = frame->tick};
    ++presented;
  }
  return presented;
}

PixelSize wait_for_size(const Window& window, PixelSize previous) {
  for (int i = 0; i < 200; ++i) {
    pump_events();
    if (window.pixel_size() != previous) break;
    SDL_Delay(5);
  }
  return window.pixel_size();
}

}  // namespace

TEST_CASE("fenêtre : taille hors bornes refusée avant d'ouvrir quoi que ce soit") {
  CHECK(Window::create({.width = 0, .height = 180}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(Window::create({.width = 320, .height = 20000}).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("fenêtre réelle : swapchain FIFO, images, redimensionnement, vsync et HDR", "[display][vulkan]") {
  auto window = Window::create({.title = "ajp present (test)", .width = 320, .height = 180});
  if (!window) SKIP("pas d'écran : " << window.error().detail);
  const auto extensions = window->vulkan_instance_extensions();
  if (!extensions) SKIP("pas de Vulkan pour cette fenêtre : " << extensions.error().detail);

  Placebo placebo;
  const pl_log_params log_params{.log_cb = pl_log_color, .log_level = PL_LOG_WARN};
  placebo.log = pl_log_create(PL_API_VER, &log_params);
  std::vector<const char*> names;
  for (const std::string& name : *extensions) names.push_back(name.c_str());
  const pl_vk_inst_params instance_params{.extensions = names.data(),
                                          .num_extensions = static_cast<int>(names.size())};
  placebo.instance = pl_vk_inst_create(placebo.log, &instance_params);
  if (placebo.instance == nullptr) SKIP("instance Vulkan refusée avec les extensions de SDL3");

  auto surface = Surface::create(*window, placebo.instance->instance);
  REQUIRE(surface);
  const pl_vulkan_params device_params{
      .instance = placebo.instance->instance,
      .get_proc_addr = placebo.instance->get_proc_addr,
      .surface = ajp::present::from_u64<VkSurfaceKHR>(surface->handle()),
      .allow_software = true,  // lavapipe en CI
  };
  placebo.vulkan = pl_vulkan_create(placebo.log, &device_params);
  if (placebo.vulkan == nullptr) SKIP("aucun appareil Vulkan ne présente sur cette surface");

  std::optional<Swapchain> swapchain;
  {
    auto created = Swapchain::create(*window, *surface, placebo.vulkan, {});
    REQUIRE(created);
    swapchain.emplace(std::move(*created));
  }
  const pl_gpu gpu = placebo.vulkan->gpu;

  // Quelques images d'abord : la fenêtre n'a d'écran qu'une fois affichée.
  Presented last;
  const int first = present_frames(*swapchain, gpu, 30, last);
  // Écran en veille ou fenêtre masquée : FIFO ne présenterait rien, la swapchain n'attend pas.
  if (first < 30 && !window->presentable()) SKIP("fenêtre masquée ou écran en veille : rien n'est présenté");
  REQUIRE(first == 30);
  const auto system = ajp::present::read_system_display(*window);
  REQUIRE(system);
  const auto detected = ajp::present::detect_display(*system, {}, std::nullopt);
  REQUIRE(detected);
  const auto& info = detected->info;
  REQUIRE(swapchain->request_output(info, info.hdr ? info.peak_nits : 203.0));
  REQUIRE(present_frames(*swapchain, gpu, 150, last) == 150);
  CHECK(last.size == window->pixel_size());

  // Redimensionnement : la swapchain suit la fenêtre.
  const PixelSize before = window->pixel_size();
  REQUIRE(window->set_size(400, 240));
  const PixelSize after = wait_for_size(*window, before);
  CHECK(after != before);
  const auto kept = swapchain->resize(after);
  REQUIRE(kept);
  REQUIRE(present_frames(*swapchain, gpu, 60, last) == 60);
  CHECK(last.size == *kept);

  const auto estimate = swapchain->vsync().estimate();
  if (software_device(placebo)) {
    // lavapipe : la cadence FIFO d'un appareil logiciel dépend du serveur d'affichage (Xvfb en
    // CI n'a pas de vsync), elle n'est que notée.
    WARN("appareil logiciel : vsync non vérifiée, mesure "
         << (estimate ? std::to_string(estimate->refresh_hz()) + " Hz" : std::string("aucune")));
  } else if (estimate && info.refresh_hz > 0.0 && estimate->refresh_hz() < info.refresh_hz / 2.0) {
    // KWin, écran en veille (DPMS) : il ne rappelle plus la fenêtre qu'à 20 Hz environ, sans la
    // dire masquée. La vsync de l'écran n'est pas mesurable ainsi.
    SKIP("le compositeur ne présente qu'à " << estimate->refresh_hz() << " Hz pour un mode à "
                                            << info.refresh_hz << " Hz : écran en veille ?");
  } else {
    REQUIRE(estimate);
    // La vsync de chaque image vient de la période mesurée : rang croissant, instant futur.
    CHECK(last.tick.period_us == estimate->period_us);
    CHECK(static_cast<double>(last.tick.host_us) > estimate->anchor_host_us);
    CHECK(estimate->period_us > 1'000.0);
    CHECK(estimate->period_us < 100'000.0);
    if (info.refresh_hz > 0.0 && !info.vrr) {
      // FIFO : une image par vsync une fois la file pleine.
      CHECK(estimate->refresh_hz() > info.refresh_hz * 0.95);
      CHECK(estimate->refresh_hz() < info.refresh_hz * 1.05);
    }
  }
  const auto output = swapchain->output();
  WARN("écran " << system->name << " : " << info.width << "x" << info.height << ", mode "
                << (system->refresh
                        ? std::to_string(system->refresh->num) + "/" + std::to_string(system->refresh->den)
                        : std::string("inconnu"))
                << " Hz, nominal " << info.refresh_hz << " Hz, vsync mesurée "
                << (estimate ? estimate->refresh_hz() : 0.0) << " Hz (gigue "
                << (estimate ? estimate->jitter_us : 0.0) << " µs), HDR " << (info.hdr ? "actif" : "inactif")
                << ", crête " << info.peak_nits << " nits, blanc SDR proposé "
                << detected->proposed_sdr_white_nits.value_or(0.0) << " nits, sortie "
                << static_cast<int>(output.space) << (output.hdr_lost ? " (HDR perdu)" : "") << ", grade "
                << (output.grade ? "actif" : "coupé") << ", latence " << swapchain->latency() << ", fenêtre "
                << before.width << "x" << before.height << " puis " << kept->width << "x" << kept->height);
  for (const std::string& unknown : detected->unknown) WARN("inconnu : " << unknown);

  swapchain.reset();
}
