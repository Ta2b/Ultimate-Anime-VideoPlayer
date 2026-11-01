// Le rendu hors écran des essais du bench (ScreenOptions::offscreen) : la composition entière (grade,
// sous-titres, mise à l'échelle vers la taille de l'écran) dans deux images qui tiennent lieu de swapchain,
// et la cadence de l'écran simulée. La charge du GPU est celle de la lecture ; seul l'envoi au compositeur
// manque, et aucune fenêtre ne se montre.
#include <thread>

#include "app/play/placebo_impl.h"
#include "core/clock.h"

namespace ajp::app::play {

std::expected<pl_frame, Refusal> PlaceboScreen::Impl::offscreen_target() {
  const pl_gpu gpu = vk->gpu;
  pl_tex& tex = offscreen_tex[offscreen_index];
  offscreen_index = (offscreen_index + 1) % offscreen_tex.size();
  // Deux images en file, comme la swapchain (kSwapchainDepth) : l'image d'il y a deux tours doit être finie
  // avant d'être réécrite, sans quoi le GPU prendrait du retard sans que rien ne le dise.
  if (tex != nullptr) {
    while (pl_tex_poll(gpu, tex, 50'000'000)) {
    }
  }
  pl_tex_params params{};
  params.w = offscreen_size.width;
  params.h = offscreen_size.height;
  params.format = pl_find_named_fmt(gpu, "rgb10a2");
  if (params.format == nullptr) params.format = pl_find_named_fmt(gpu, "rgba16");
  params.renderable = true;
  params.blit_dst = true;
  if (!pl_tex_recreate(gpu, &tex, &params))
    return std::unexpected(
        Refusal{RefusalCode::VendorCallFailed, "pl_tex_recreate (hors écran)"});  // i18n-ok : journal
  pl_frame target{};
  target.num_planes = 1;
  target.planes[0].texture = tex;
  target.planes[0].components = 4;
  for (int c = 0; c < 4; ++c) target.planes[0].component_mapping[c] = c;
  target.repr = pl_color_repr_rgb;
  // La sortie que la lecture aurait : HDR10 quand l'écran est en HDR (sans --sdr), sRGB sinon.
  target.color = hdr_wanted ? pl_color_space_hdr10 : pl_color_space_monitor;
  target.crop = {0, 0, static_cast<float>(params.w), static_cast<float>(params.h)};
  return target;
}

std::int64_t PlaceboScreen::Impl::offscreen_present() {
  // La prochaine synchronisation verticale simulée : une période après la précédente, ou la première qui
  // suit maintenant si le rendu l'a manquée.
  const std::int64_t now = host_now_us();
  if (next_tick_us == 0) next_tick_us = now;
  next_tick_us += period_us;
  if (next_tick_us < now) next_tick_us += ((now - next_tick_us) / period_us + 1) * period_us;
  std::this_thread::sleep_for(std::chrono::microseconds(next_tick_us - now));
  return next_tick_us;
}

}  // namespace ajp::app::play
