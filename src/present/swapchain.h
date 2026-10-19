// Swapchain libplacebo de la fenêtre, en FIFO. present fournit à chaque image une cible
// pl_frame (pl_swapchain_start_frame) puis la soumet ; il ne compose jamais l'image : render
// y fait pl_render_image, app relie les deux.
//
// L'espace de sortie suit DisplayInfo : HDR10 PQ BT.2020 avec max_luma égal au pic du grade
// quand l'écran est en HDR, repli scRGB (grade coupé) puis SDR sinon (present/output.h).
// Chaque image acceptée nourrit l'estimation de la période de vsync (VsyncEstimator).
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>

#include "core/resolve.h"
#include "core/types.h"
#include "present/output.h"
#include "present/scheduler.h"
#include "present/surface.h"
#include "present/vsync_estimator.h"
#include "present/window.h"

// Types de libplacebo, déclarés seulement : libplacebo reste une dépendance privée.
struct pl_vulkan_t;
struct pl_frame;

namespace ajp::present {

struct SwapchainOptions {
  int depth = 3;  // images en vol (pl_vulkan_swapchain_params::swapchain_depth)
};

namespace swapchain_limits {
inline constexpr core::Range<int> kDepth{1, 8};
}  // namespace swapchain_limits

// Une image de la swapchain, à composer par render puis à soumettre.
struct SwapFrame {
  const pl_frame* target = nullptr;  // cible de pl_render_image, valable jusqu'à submit_frame()
  PixelSize size;
  bool flipped = false;
  OutputState output;                 // l'espace que cette image aura vraiment
  std::int64_t acquired_host_us = 0;  // retour de pl_swapchain_start_frame (host_now_us)
  // La vsync où l'image paraîtra, à donner à PresentScheduler::on_vsync : rang et instant
  // prévus par la période mesurée, latence de la swapchain comprise. Tant que la période n'est
  // pas connue, une vsync par image et l'instant d'acquisition.
  VsyncTick tick;
};

class Swapchain {
 public:
  // vk : le pl_vulkan de render, créé avec cette surface (VK_KHR_swapchain activée). La taille
  // de départ est celle de la fenêtre : sous Wayland, la surface n'en a aucune à elle.
  static std::expected<Swapchain, Refusal> create(const Window& window, const Surface& surface,
                                                  const pl_vulkan_t* vk, const SwapchainOptions& options);

  Swapchain(Swapchain&& other) noexcept;
  Swapchain& operator=(Swapchain&& other) noexcept;
  Swapchain(const Swapchain&) = delete;
  Swapchain& operator=(const Swapchain&) = delete;
  ~Swapchain();

  // L'espace à demander pour cet écran ; grade_peak_nits devient max_luma (et MaxCLL) des
  // métadonnées HDR10. Refuse un pic hors de [1, 10000] nits.
  std::expected<void, Refusal> request_output(const core::DisplayInfo& display, double grade_peak_nits);

  // Nouvelle taille en pixels (SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) ; rend la taille retenue.
  std::expected<PixelSize, Refusal> resize(PixelSize size);

  // L'image suivante, ou rien pour l'instant : fenêtre cachée, réduite ou masquée (écran en
  // veille), où FIFO attendrait sans fin un affichage qui ne vient pas, ou swapchain recréée.
  // Réessayer plus tard. Bloque en FIFO tant qu'aucune image n'est libre, une vsync au plus.
  // La fenêtre de create() doit vivre plus longtemps que la swapchain.
  std::optional<SwapFrame> start_frame();

  // Soumet l'image commencée puis attend la place de la suivante (pl_swapchain_swap_buffers).
  // PresentFailed si l'appareil ou la surface sont perdus.
  std::expected<void, Refusal> submit_frame();

  const VsyncEstimator& vsync() const;
  OutputState output() const;  // celui de la dernière image commencée
  int latency() const;         // images en vol (pl_swapchain_latency)

 private:
  struct Impl;
  explicit Swapchain(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::present
