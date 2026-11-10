// Instantané de PlaceboScreen : la composition de la fenêtre rendue une fois de plus, dans l'espace SDR de
// la fenêtre (sRGB si elle est en HDR), dans une texture hors écran relue sur l'hôte et écrite en PPM. Outil
// de vérification (--snapshot), hors du chemin de lecture. Un fichier en « .pq.ppm » garde la sortie HDR
// telle que l'écran la reçoit (PQ BT.2020, grade compris), en PPM 16 bits.
#include <array>
#include <cstdint>
#include <format>
#include <fstream>
#include <vector>

#include "app/play/placebo_impl.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

std::expected<void, Refusal> PlaceboScreen::Impl::save_snapshot(const pl_frame* image, const pl_frame& target,
                                                                OutputSize out, const ImDrawData* ui_data,
                                                                const std::filesystem::path& path) {
  const pl_gpu gpu = vk->gpu;
  pl_tex_params params{};
  params.w = out.width;
  params.h = out.height;
  // PQ seulement si l'écran le reçoit ; sinon l'instantané reste en SDR.
  const bool pq = core::to_utf8(path).ends_with(".pq.ppm") && target.color.transfer == PL_COLOR_TRC_PQ;
  // Constat 6 de l'audit du grade : le grade chargé est celui du PQ ; rendu en sRGB, l'instantané sort sans
  // lui (coché ou décoché, le même fichier). Seul .pq.ppm est fidèle : le journal le dit.
  if (!pq && target.color.transfer == PL_COLOR_TRC_PQ && grade && channel)
    channel->warn(
        "instantané {} : sortie HDR10, l'image SDR est sans grade ; un nom en .pq.ppm garde "
        "l'image gradée telle que l'écran la reçoit",
        core::to_utf8(path.filename()));
  params.format = pl_find_named_fmt(gpu, pq ? "rgba16" : "rgba8");
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
  // A10-18 : une fenêtre SDR donne l'instantané dans son propre espace (primaires et courbe de la surface),
  // ce que l'écran reçoit ; avant, toujours sRGB, donc une image qui n'était pas celle de la fenêtre dès que
  // la surface n'était pas en sRGB. Une fenêtre HDR sans « .pq.ppm » reste rendue en sRGB.
  sdr.color = snapshot_keeps_target(pq, pl_color_transfer_is_hdr(target.color.transfer))
                  ? target.color
                  : pl_color_space_srgb;
  sdr.crop = {0, 0, static_cast<float>(out.width), static_cast<float>(out.height)};
  if (image != nullptr)
    pl_rect2df_aspect_set(&sdr.crop, pl_rect2df_aspect(&image->crop) * static_cast<float>(sample_aspect),
                          0.0F);
  const std::size_t bytes = pq ? 2 : 1;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(out.width) *
                                   static_cast<std::size_t>(out.height) * 4 * bytes);
  auto rendered = compose(image, sdr);
  if (rendered) draw_ui(ui_data, sdr);
  pl_tex_transfer_params download{};
  download.tex = tex;
  download.ptr = pixels.data();
  const bool read = rendered && pl_tex_download(gpu, &download);
  pl_tex_destroy(gpu, &tex);
  if (!rendered) return rendered;
  if (!read) return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_tex_download (instantané)"});
  std::ofstream file(path, std::ios::binary);
  file << std::format("P6\n{} {}\n{}\n", out.width, out.height, pq ? 65535 : 255);
  for (std::size_t i = 0; i < pixels.size(); i += 4 * bytes) {
    if (!pq) {
      file.write(reinterpret_cast<const char*>(&pixels[i]), 3);
      continue;
    }
    for (std::size_t c = 0; c < 3; ++c) {  // PPM 16 bits : poids fort d'abord
      const std::array<char, 2> sample{static_cast<char>(pixels[i + 2 * c + 1]),
                                       static_cast<char>(pixels[i + 2 * c])};
      file.write(sample.data(), 2);
    }
  }
  if (!file) return std::unexpected(Refusal{RefusalCode::IoFailed, "écriture de l'instantané"});
  channel->info("instantané écrit : {}", core::to_utf8(path.filename()));
  return {};
}

void PlaceboScreen::Impl::take_snapshot(const pl_frame* image, const pl_frame& target, OutputSize out,
                                        const ImDrawData* drawn_ui) {
  if (!snapshot) return;
  std::filesystem::path path = *snapshot;
  if (snapshot_count > 1) {
    // « nom.pq.ppm » → « nom-001.pq.ppm », « nom.ppm » → « nom-001.ppm ».
    const std::string file = core::to_utf8(path.filename());
    const std::size_t dot = file.ends_with(".pq.ppm") ? file.size() - 7 : file.rfind('.');
    path.replace_filename(
        core::path_from_utf8(std::format("{}-{:03}{}", file.substr(0, dot), snapshot_index + 1,
                                         dot == std::string::npos ? std::string() : file.substr(dot))));
  }
  if (++snapshot_index >= snapshot_count) snapshot.reset();
  if (auto saved = save_snapshot(image, target, out, drawn_ui, path); !saved) {
    channel->warn("instantané {} : {}", core::to_utf8(path.filename()), saved.error().detail);
  }
}

}  // namespace ajp::app::play
