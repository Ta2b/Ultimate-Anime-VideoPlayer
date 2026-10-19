// Détails de PlaceboScreen, partagés par ses fichiers : le ring exporté vers CUDA et les
// calques (sous-titres, OSD). Interne à placebo_*.cpp.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>
#include <spdlog/logger.h>

#include "app/play/links.h"
#include "app/play/osd_text.h"
#include "app/play/placebo_screen.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "subs/subtitle_frame.h"

namespace ajp::app::play {

// Les slots du ring : textures RGBA16 exportées et leurs sémaphores timeline. Un slot est soit
// tenu par CUDA (gpu), soit rendu à libplacebo le temps que main le lise (core/ring.h).
class PlaceboRing {
 public:
  std::expected<std::vector<RingSlotDesc>, Refusal> create(pl_gpu gpu, int width, int height,
                                                           std::uint32_t count);
  // La texture d'un RingFrame : rendue à libplacebo, qui attend ready_value, à sa première lecture.
  std::expected<pl_tex, Refusal> acquire(pl_gpu gpu, const RingFrame& frame);
  // Cède le slot à CUDA après le dernier rendu qui l'a lu (signal de ready_value + 1).
  SlotFree retire(pl_gpu gpu, const RingFrame& frame);
  void clear(pl_gpu gpu);

 private:
  struct Slot {
    pl_tex tex = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    int semaphore_fd = -1;            // exporté ; fermé avec le ring
    std::uint64_t placebo_value = 0;  // ready_value que libplacebo attend ; 0 : tenu par CUDA
  };
  std::expected<pl_tex, Refusal> slot_of(const RingFrame& frame);

  std::vector<Slot> slots_;
  std::uint32_t generation_ = 0;
};

// Sous-titres et OSD en calques monochromes, au-dessus de l'image, en pixels de la sortie.
class Overlays {
 public:
  explicit Overlays(pl_gpu gpu) : gpu_(gpu) {}
  ~Overlays();
  Overlays(const Overlays&) = delete;
  Overlays& operator=(const Overlays&) = delete;

  std::expected<std::span<const pl_overlay>, Refusal> build(const subs::SubtitleFrame* subtitles,
                                                            const std::span<const std::string>* osd,
                                                            OsdText& text, OutputSize out);

 private:
  struct Layer {
    pl_tex tex = nullptr;
    const subs::SubtitleAtlas* atlas = nullptr;  // celui que tex contient
    std::uint64_t generation = 0;
    std::vector<pl_overlay_part> parts;
  };
  std::expected<bool, Refusal> add(Layer& layer, const subs::SubtitleFrame& frame, OutputSize out);

  pl_gpu gpu_;
  Layer subtitles_;
  Layer osd_;
  std::array<pl_overlay, 2> overlays_{};
  int count_ = 0;
};

struct PlaceboScreen::Impl {
  ~Impl();
  std::expected<void, Refusal> compose(const pl_frame* image, const pl_frame& target);
  std::expected<void, Refusal> save_snapshot(const pl_frame* image, const pl_frame& target, OutputSize out);
  std::shared_ptr<spdlog::logger> channel;
  SDL_Window* window = nullptr;
  pl_log log = nullptr;
  pl_vk_inst inst = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  pl_vulkan vk = nullptr;
  pl_swapchain swapchain = nullptr;
  pl_renderer renderer = nullptr;
  GpuDevice device;
  PlaceboRing ring;
  std::unique_ptr<Overlays> overlays;
  std::unique_ptr<OsdText> osd;
  pl_color_space source{};
  OutputSize swapchain_size;
  bool fullscreen = false;
  bool hdr_output = false;
  std::optional<std::filesystem::path> snapshot;
};

}  // namespace ajp::app::play
