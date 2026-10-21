// Parité avec libass : la SubtitleFrame (atlas, rectangles, couleurs) dessinée
// comme render le fera doit donner exactement l'image que donne la liste
// ASS_Image de libass composée directement, sur un second moteur réglé pareil.
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include <ass/ass.h>
#include <catch2/catch_test_macros.hpp>

#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"
#include "subs/timebase.h"
#include "support.h"

using ajp::subs::RenderRequest;
using ajp::subs::RenderTarget;
using ajp::subs::SubtitleAtlas;
using ajp::subs::SubtitleFrame;
using ajp::subs::test::Canvas;

namespace {

struct RendererFree {
  void operator()(ASS_Renderer* renderer) const { ass_renderer_done(renderer); }
};

// Moteur libass de référence, réglé comme SubtitleRenderer avec ses options par défaut.
std::unique_ptr<ASS_Renderer, RendererFree> reference_renderer(ASS_Library* library,
                                                               const RenderTarget& target) {
  std::unique_ptr<ASS_Renderer, RendererFree> renderer(ass_renderer_init(library));
  REQUIRE(renderer != nullptr);
  ass_set_fonts(renderer.get(), nullptr, "sans-serif", ASS_FONTPROVIDER_AUTODETECT, nullptr, 1);
  ass_set_frame_size(renderer.get(), target.width, target.height);
  const bool video_known = target.video_width != 0;
  ass_set_storage_size(renderer.get(), video_known ? target.video_width : target.width,
                       video_known ? target.video_height : target.height);
  return renderer;
}

// La liste de libass composée sur fond noir, avec les mêmes opérations que test::draw.
Canvas compose(const ASS_Image* images, int width, int height) {
  Canvas canvas{.width = width, .height = height, .opacity = {}, .rgb = {}};
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  canvas.opacity.assign(pixels, 0.0F);
  canvas.rgb.assign(pixels, {0.0F, 0.0F, 0.0F});
  for (const ASS_Image* image = images; image != nullptr; image = image->next) {
    const std::array<float, 3> color{static_cast<float>(image->color >> 24) / 255.0F,
                                     static_cast<float>((image->color >> 16) & 0xFFU) / 255.0F,
                                     static_cast<float>((image->color >> 8) & 0xFFU) / 255.0F};
    const auto alpha = static_cast<float>(255U - (image->color & 0xFFU));
    for (int y = 0; y < image->h; ++y) {
      for (int x = 0; x < image->w; ++x) {
        const auto at = static_cast<std::size_t>(image->dst_y + y) * static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(image->dst_x + x);
        const auto source = static_cast<std::size_t>(y) * static_cast<std::size_t>(image->stride) +
                            static_cast<std::size_t>(x);
        const float o = (static_cast<float>(image->bitmap[source]) / 255.0F) * (alpha / 255.0F);
        canvas.opacity[at] += o * (1.0F - canvas.opacity[at]);
        for (std::size_t c = 0; c < 3; ++c) canvas.rgb[at][c] = canvas.rgb[at][c] * (1.0F - o) + color[c] * o;
      }
    }
  }
  return canvas;
}

void check_same(const Canvas& ours, const Canvas& reference) {
  REQUIRE(ours.width == reference.width);
  REQUIRE(ours.height == reference.height);
  std::size_t differing = 0;
  for (std::size_t i = 0; i < ours.opacity.size(); ++i) {
    if (ours.opacity[i] != reference.opacity[i] || ours.rgb[i] != reference.rgb[i]) ++differing;
  }
  CHECK(differing == 0);
}

struct Parity {
  ajp::subs::AssLibrary library = ajp::subs::AssLibrary::create().value();
  ajp::subs::SubtitleTrack track =
      ajp::subs::SubtitleTrack::from_ass(library, ajp::subs::test::reference_script()).value();
  std::vector<std::unique_ptr<SubtitleAtlas>> returned;
  ajp::subs::AtlasPool pool{
      [this](std::unique_ptr<SubtitleAtlas> atlas) { returned.push_back(std::move(atlas)); }};
  ajp::subs::SubtitleRenderer renderer = ajp::subs::SubtitleRenderer::create(library, {}).value();

  // Compare notre image et celle de libass à ce pts ; rend notre image.
  SubtitleFrame check(double seconds, const RenderTarget& target) {
    for (auto& atlas : returned) pool.give_back(std::move(atlas));
    returned.clear();
    const RenderRequest request{
        .pts_us = static_cast<std::int64_t>(seconds * 1e6), .epoch = 0, .target = target};
    auto outcome = renderer.render(track, request, pool);
    REQUIRE(outcome);
    REQUIRE(std::holds_alternative<SubtitleFrame>(*outcome));
    SubtitleFrame frame = std::get<SubtitleFrame>(std::move(*outcome));

    auto reference = reference_renderer(library.get(), target);
    const ASS_Image* images =
        ass_render_frame(reference.get(), track.get(), ajp::subs::to_ms(request.pts_us), nullptr);
    check_same(ajp::subs::test::draw(frame), compose(images, target.width, target.height));
    return frame;
  }
};

}  // namespace

TEST_CASE("Parité libass : chaque événement du script de référence, pixel pour pixel", "[parity]") {
  Parity parity;
  const RenderTarget target{.width = ajp::subs::test::kScriptWidth, .height = ajp::subs::test::kScriptHeight};
  for (const double seconds : {0.5, 1.5, 2.5, 3.5, 4.1, 4.5, 4.9}) {
    INFO("pts " << seconds << " s");
    const SubtitleFrame frame = parity.check(seconds, target);
    ajp::subs::test::check_atlas(frame);
  }
}

TEST_CASE("Parité libass : sortie agrandie, vidéo source plus petite, changement de cible", "[parity]") {
  Parity parity;
  const RenderTarget small{.width = 640, .height = 360, .video_width = 640, .video_height = 360};
  const RenderTarget large{.width = 1920, .height = 1080, .video_width = 640, .video_height = 360};
  const SubtitleFrame before = parity.check(0.5, small);
  const SubtitleFrame after = parity.check(0.5, large);  // même pts, même moteur, autre cible
  REQUIRE(before.atlas != nullptr);
  REQUIRE(after.atlas != nullptr);
  CHECK(after.atlas->generation > before.atlas->generation);
  CHECK(after.width == 1920);
  parity.check(4.5, large);  // \move à la nouvelle échelle
}
