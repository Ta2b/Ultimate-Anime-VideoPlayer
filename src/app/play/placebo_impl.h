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
#include "app/play/placebo_imgui.h"
#include "app/play/placebo_screen.h"
#include "app/play/vulkan_share.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "render/grade.h"
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
  // Le sémaphore du slot a atteint ready_value : l'image est écrite. Lu sans attendre.
  bool ready(const RingFrame& frame) const;
  void clear(pl_gpu gpu);

 private:
  struct Slot {
    pl_tex tex = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    // Poignée exportée du sémaphore (descripteur, ou HANDLE sous Windows) ; fermée avec le ring.
    std::optional<std::uintptr_t> semaphore_export;
    std::uint64_t placebo_value = 0;  // ready_value que libplacebo attend ; 0 : tenu par CUDA
  };
  std::expected<pl_tex, Refusal> slot_of(const RingFrame& frame);

  std::vector<Slot> slots_;
  std::uint32_t generation_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;
  PFN_vkGetSemaphoreCounterValue counter_ = nullptr;
};

// Sous-titres et OSD en calques monochromes, au-dessus de l'image, en pixels de la sortie.
class Overlays {
 public:
  explicit Overlays(pl_gpu gpu) : gpu_(gpu) {}
  ~Overlays();
  Overlays(const Overlays&) = delete;
  Overlays& operator=(const Overlays&) = delete;

  // ui_bottom, ui_top : pixels pris en bas et en haut par l'interface ; les sous-titres du bas
  // montent au-dessus, l'OSD descend dessous.
  std::expected<std::span<const pl_overlay>, Refusal> build(const subs::SubtitleFrame* subtitles,
                                                            const std::span<const std::string>* osd,
                                                            OsdText& text, OutputSize out, int ui_bottom,
                                                            int ui_top);

 private:
  struct Layer {
    pl_tex tex = nullptr;
    const subs::SubtitleAtlas* atlas = nullptr;  // celui que tex contient
    std::uint64_t generation = 0;
    std::vector<pl_overlay_part> parts;
  };
  std::expected<bool, Refusal> add(Layer& layer, const subs::SubtitleFrame& frame, OutputSize out,
                                   int ui_bottom = 0);

  pl_gpu gpu_;
  Layer subtitles_;
  Layer osd_;
  std::array<pl_overlay, 2> overlays_{};
  int count_ = 0;
};

struct PlaceboScreen::Impl {
  ~Impl();
  std::expected<void, Refusal> compose(const pl_frame* image, const pl_frame& target) const;
  // L'interface par-dessus la composition ; un refus est écrit au journal une fois, la lecture
  // continue sans elle.
  void draw_ui(const ImDrawData* ui, const pl_frame& target);
  std::expected<void, Refusal> save_snapshot(const pl_frame* image, const pl_frame& target, OutputSize out,
                                             const ImDrawData* ui, const std::filesystem::path& path);
  std::shared_ptr<spdlog::logger> channel;
  SDL_Window* window = nullptr;
  pl_log log = nullptr;
  pl_vk_inst inst = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  pl_vulkan vk = nullptr;
  std::unique_ptr<VulkanShare> share;  // vk prêté à FFmpeg ; détruit avant lui
  pl_swapchain swapchain = nullptr;
  pl_renderer renderer = nullptr;
  GpuDevice device;
  PlaceboRing ring;
  std::unique_ptr<Overlays> overlays;
  std::unique_ptr<OsdText> osd;
  std::unique_ptr<ImGuiLayer> ui;
  static constexpr int kMaxUiFailures = 5;
  int ui_failures = 0;   // échecs de suite du dessin de l'interface
  bool ui_failed = false;  // abandonnée après kMaxUiFailures échecs de suite
  pl_color_space source{};
  OutputSize swapchain_size;
  bool fullscreen = false;
  bool hdr_output = false;
  std::optional<std::filesystem::path> snapshot;
  int snapshot_count = 1;  // présentations à écrire, numérotées s'il y en a plusieurs
  int snapshot_index = 0;
  std::optional<render::Grade> grade;  // absent si les hooks ne se lisent pas (dit au journal)
  render::GradeChoice grade_choice;    // posé par set_grade
  // Ce que le grade porte : rien, le grade PQ (vrai) ou le grade SDR (faux) de grade_choice.
  std::optional<bool> grade_loaded_pq;
  // Charge dans le grade le choix qui va à cette cible ; graded() dit s'il s'applique à une cible.
  void select_grade(const pl_color_space& target);
  bool graded(const pl_color_space& target) const;
  // Temps GPU des passes du grade (minuteries de libplacebo), par image composée avec lui ; le
  // résumé va au journal à la fermeture de l'écran.
  pl_color_space target_color{};  // celle de la dernière présentation
  // Mesure de la rampe du vrai noir : un renderer et un grade à part, pour ne toucher ni la
  // détection de pic ni l'état temporel du grade de la lecture.
  pl_renderer ramp_renderer = nullptr;
  std::optional<render::Grade> ramp_grade;
  mutable std::uint64_t grade_frame_ns = 0;
  mutable std::vector<std::uint32_t> grade_gpu_us;
};

}  // namespace ajp::app::play
