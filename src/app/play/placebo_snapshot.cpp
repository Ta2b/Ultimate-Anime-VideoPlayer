// Instantané de PlaceboScreen : la composition de la fenêtre rendue une fois de plus, en
// SDR sRGB, dans une texture hors écran relue sur l'hôte et écrite en PPM. Outil de
// vérification (--snapshot), hors du chemin de lecture.
#include <cstdint>
#include <format>
#include <fstream>
#include <vector>

#include "app/play/placebo_impl.h"

namespace ajp::app::play {

std::expected<void, Refusal> PlaceboScreen::Impl::save_snapshot(const pl_frame* image, const pl_frame& target,
                                                                OutputSize out) {
  const pl_gpu gpu = vk->gpu;
  pl_tex_params params{};
  params.w = out.width;
  params.h = out.height;
  params.format = pl_find_named_fmt(gpu, "rgba8");
  params.renderable = true;
  params.blit_dst = true;
  params.host_readable = true;
  pl_tex tex = pl_tex_create(gpu, &params);
  if (tex == nullptr)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_tex_create (instantané)"});
  pl_frame sdr = target;
  sdr.num_planes = 1;
  sdr.planes[0] = {};
  sdr.planes[0].texture = tex;
  sdr.planes[0].components = 4;
  for (int c = 0; c < 4; ++c) sdr.planes[0].component_mapping[c] = c;
  sdr.repr = pl_color_repr_rgb;
  sdr.color = pl_color_space_srgb;
  sdr.crop = {0, 0, static_cast<float>(out.width), static_cast<float>(out.height)};
  if (image != nullptr) pl_rect2df_aspect_copy(&sdr.crop, &image->crop, 0.0F);
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(out.width) *
                                   static_cast<std::size_t>(out.height) * 4);
  auto rendered = compose(image, sdr);
  pl_tex_transfer_params download{};
  download.tex = tex;
  download.ptr = pixels.data();
  const bool read = rendered && pl_tex_download(gpu, &download);
  pl_tex_destroy(gpu, &tex);
  if (!rendered) return rendered;
  if (!read) return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_tex_download (instantané)"});
  std::ofstream file(*snapshot, std::ios::binary);
  file << std::format("P6\n{} {}\n255\n", out.width, out.height);
  for (std::size_t i = 0; i < pixels.size(); i += 4) file.write(reinterpret_cast<const char*>(&pixels[i]), 3);
  if (!file) return std::unexpected(Refusal{RefusalCode::IoFailed, "écriture de l'instantané"});
  channel->info("instantané écrit : {}", snapshot->string());
  return {};
}

}  // namespace ajp::app::play
