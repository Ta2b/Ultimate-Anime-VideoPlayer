#include "render/grade_input.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ajp::render {

namespace {

double pq_to_nits(double e) {
  const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / 78.84375);
  return 10000.0 * std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

}  // namespace

std::expected<GradeInputTable, Refusal> capture_grade_input(pl_renderer renderer, pl_gpu gpu,
                                                            const pl_color_space& source,
                                                            const pl_color_space& target) {
  if (target.transfer != PL_COLOR_TRC_PQ)
    return std::unexpected(
        Refusal{RefusalCode::HdrUnavailable, "entrée du grade : la cible n'est pas en PQ"});
  constexpr int kWidth = kGradeInputPoints;
  constexpr int kRows = 2;
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kWidth) * kRows * 4);
  for (int y = 0; y < kRows; ++y)
    for (int x = 0; x < kWidth; ++x) {
      const auto value = static_cast<std::uint16_t>(
          std::lround(static_cast<double>(x) / static_cast<double>(kWidth - 1) * 65535.0));
      const std::size_t at = (static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)) * 4;
      pixels[at] = pixels[at + 1] = pixels[at + 2] = value;
      pixels[at + 3] = 65535;
    }
  pl_tex_params tex{};
  tex.w = kWidth;
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
  const auto frame_of = [](pl_tex texture, const pl_color_space& color) {
    pl_frame frame{};
    frame.num_planes = 1;
    frame.planes[0].texture = texture;
    frame.planes[0].components = 4;
    for (int c = 0; c < 4; ++c) frame.planes[0].component_mapping[c] = c;
    frame.repr = pl_color_repr_rgb;
    frame.color = color;
    frame.crop = {0, 0, static_cast<float>(kWidth), static_cast<float>(kRows)};
    return frame;
  };
  bool done = in != nullptr && out != nullptr;
  if (done) {
    const pl_frame image = frame_of(in, source);
    const pl_frame dest = frame_of(out, target);
    pl_render_params render = pl_render_default_params;
    render.dither_params = nullptr;
    done = pl_render_image(renderer, &image, &dest, &render);
    pl_tex_transfer_params download{};
    download.tex = out;
    download.ptr = pixels.data();
    done = done && pl_tex_download(gpu, &download);
  }
  pl_tex_destroy(gpu, &in);
  pl_tex_destroy(gpu, &out);
  if (!done)
    return std::unexpected(
        Refusal{RefusalCode::VendorCallFailed, "entrée du grade : rendu ou relecture de la rampe"});
  GradeInputTable table;
  for (auto& channel : table.nits) channel.resize(kWidth);
  for (int x = 0; x < kWidth; ++x)
    for (std::size_t c = 0; c < 3; ++c)
      table.nits[c][static_cast<std::size_t>(x)] =
          static_cast<float>(pq_to_nits(pixels[static_cast<std::size_t>(x) * 4 + c] / 65535.0));
  const pl_matrix3x3 m = pl_get_color_mapping_matrix(pl_raw_primaries_get(source.primaries),
                                                     pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020),
                                                     PL_INTENT_RELATIVE_COLORIMETRIC);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) table.matrix[static_cast<std::size_t>(r * 3 + c)] = m.m[r][c];
  return table;
}

}  // namespace ajp::render
