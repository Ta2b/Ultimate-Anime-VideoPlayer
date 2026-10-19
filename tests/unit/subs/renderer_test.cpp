#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"
#include "support.h"

using ajp::RefusalCode;
using ajp::subs::AtlasPool;
using ajp::subs::RenderRequest;
using ajp::subs::SubtitleAtlas;
using ajp::subs::SubtitleFrame;
using ajp::subs::SubtitleRenderer;
using ajp::subs::WaitForAtlas;
using ajp::subs::test::Box;
using ajp::subs::test::draw;
using ajp::subs::test::kScriptHeight;
using ajp::subs::test::kScriptWidth;

namespace {

// Bibliothèque, moteur, piste et réserve d'un test ; les atlas relâchés
// attendent dans returned, comme dans la boîte du thread.
struct Bench {
  ajp::subs::AssLibrary library = ajp::subs::AssLibrary::create().value();
  std::vector<std::unique_ptr<SubtitleAtlas>> returned;
  AtlasPool pool{[this](std::unique_ptr<SubtitleAtlas> atlas) { returned.push_back(std::move(atlas)); }};
  SubtitleRenderer renderer = SubtitleRenderer::create(library, {}).value();
  ajp::subs::SubtitleTrack track;

  enum class Format : std::uint8_t { Ass, Srt };

  explicit Bench(std::string_view script, Format format = Format::Ass)
      : track(format == Format::Ass ? ajp::subs::SubtitleTrack::from_ass(library, script).value()
                                    : ajp::subs::SubtitleTrack::from_srt(library, script).value()) {}

  void recycle() {
    for (auto& atlas : returned) pool.give_back(std::move(atlas));
    returned.clear();
  }

  SubtitleFrame render(double seconds, int width = kScriptWidth, int height = kScriptHeight,
                       float scale = 1.0F) {
    recycle();
    const RenderRequest request{.pts_us = static_cast<std::int64_t>(seconds * 1e6),
                                .epoch = 0,
                                .target = {.width = width, .height = height, .font_scale = scale}};
    auto outcome = renderer.render(track, request, pool);
    REQUIRE(outcome);
    REQUIRE(std::holds_alternative<SubtitleFrame>(*outcome));
    return std::get<SubtitleFrame>(std::move(*outcome));
  }
};

}  // namespace

TEST_CASE("Rendu : le style Default se place en bas au centre, bordé de noir", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame frame = bench.render(0.5);
  CHECK(frame.pts_us == 500'000);
  CHECK(frame.width == kScriptWidth);
  ajp::subs::test::check_atlas(frame);

  const Box box = draw(frame).covered();
  REQUIRE_FALSE(box.empty());
  CHECK(box.y0 > kScriptHeight / 2);        // moitié basse
  CHECK(box.y1 <= kScriptHeight - 20 + 4);  // au-dessus de la marge verticale (bordure comprise)
  CHECK(std::abs((box.x0 + box.x1) / 2 - kScriptWidth / 2) < 8);  // centré
  // Bordure noire et remplissage blanc : deux couleurs parmi les rectangles.
  bool white = false;
  bool black = false;
  for (const auto& rect : frame.rects) {
    white = white || (rect.color == ajp::subs::Rgba8{255, 255, 255, 255});
    black = black || (rect.color == ajp::subs::Rgba8{0, 0, 0, 255});
  }
  CHECK(white);
  CHECK(black);
}

TEST_CASE("Rendu : le style Top se place en haut, en jaune", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame frame = bench.render(1.5);
  ajp::subs::test::check_atlas(frame);
  const auto canvas = draw(frame);
  const Box box = canvas.covered();
  REQUIRE_FALSE(box.empty());
  CHECK(box.y0 >= 20 - 2);
  CHECK(box.y1 < kScriptHeight / 2);
  REQUIRE(frame.rects.size() >= 1);
  CHECK(frame.rects.front().color == ajp::subs::Rgba8{255, 255, 0, 255});
}

TEST_CASE("Rendu : \\pos place le texte, \\1c le colore", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame frame = bench.render(2.5);
  ajp::subs::test::check_atlas(frame);
  const auto canvas = draw(frame);
  const Box box = canvas.covered();
  REQUIRE_FALSE(box.empty());
  CHECK(box.x0 >= 40);
  CHECK(box.x0 < 40 + 8);
  CHECK(box.y0 >= 60);
  CHECK(box.y0 < 60 + 16);
  CHECK(canvas.opacity_at(20, 20) == 0.0F);    // rien en haut à gauche
  CHECK(canvas.opacity_at(600, 300) == 0.0F);  // rien en bas à droite
  for (const auto& rect : frame.rects) CHECK(rect.color == ajp::subs::Rgba8{255, 0, 0, 255});
}

TEST_CASE("Rendu : \\alpha devient l'opacité des rectangles", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame frame = bench.render(3.5);
  ajp::subs::test::check_atlas(frame);
  for (const auto& rect : frame.rects) CHECK(rect.color.a == 255 - 0x80);
  const auto canvas = draw(frame);
  float peak = 0.0F;
  for (const float o : canvas.opacity) peak = std::max(peak, o);
  CHECK(peak > 0.45F);
  CHECK(peak < 0.55F);
}

TEST_CASE("Rendu : à la résolution de sortie, sans recadrage du script", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame frame = bench.render(2.5, 2 * kScriptWidth, 2 * kScriptHeight);
  CHECK(frame.width == 2 * kScriptWidth);
  const Box box = draw(frame).covered();
  REQUIRE_FALSE(box.empty());
  CHECK(box.x0 >= 80);
  CHECK(box.x0 < 80 + 16);
  CHECK(box.y0 >= 120);
}

TEST_CASE("Rendu : rien à dessiner hors des événements", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame frame = bench.render(6.0);
  CHECK(frame.atlas == nullptr);
  CHECK(frame.rects.empty());
}

TEST_CASE("Rendu : un contenu inchangé réutilise l'atlas sans nouveau téléversement", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame first = bench.render(0.2);
  const SubtitleFrame second = bench.render(0.6);
  REQUIRE(first.atlas != nullptr);
  CHECK(second.atlas == first.atlas);
  CHECK(second.atlas->generation == first.atlas->generation);
  CHECK(second.rects == first.rects);
  CHECK(bench.pool.in_flight() == 1);

  const SubtitleFrame third = bench.render(1.5);  // autre événement
  REQUIRE(third.atlas != nullptr);
  CHECK(third.atlas->generation > first.atlas->generation);
}

TEST_CASE("Rendu : un déplacement seul garde l'atlas et change les positions", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const SubtitleFrame start = bench.render(4.2);
  const SubtitleFrame later = bench.render(4.6);
  REQUIRE(start.atlas != nullptr);
  CHECK(later.atlas == start.atlas);
  REQUIRE(later.rects.size() == start.rects.size());
  CHECK(later.rects.front().src == start.rects.front().src);
  CHECK(later.rects.front().dst_x > start.rects.front().dst_x + 100);  // 400 px en 1 s, 0,4 s plus tard
  CHECK(draw(later).covered().x0 > draw(start).covered().x0 + 100);
}

TEST_CASE("Rendu : le 9e contenu neuf attend qu'un atlas revienne", "[render]") {
  Bench bench(ajp::subs::test::numbered_script(10));
  std::vector<SubtitleFrame> held;
  for (int i = 0; i < 8; ++i) held.push_back(bench.render(i + 0.5));
  CHECK(bench.pool.in_flight() == ajp::subs::kMaxAtlasesInFlight);

  const RenderRequest ninth{.pts_us = 8'500'000, .epoch = 0, .target = {.width = 640, .height = 360}};
  auto waiting = bench.renderer.render(bench.track, ninth, bench.pool);
  REQUIRE(waiting);
  CHECK(std::holds_alternative<WaitForAtlas>(*waiting));

  held.erase(held.begin());  // render relâche la plus ancienne image
  REQUIRE(bench.returned.size() == 1);
  bench.recycle();
  auto served = bench.renderer.render(bench.track, ninth, bench.pool);
  REQUIRE(served);
  REQUIRE(std::holds_alternative<SubtitleFrame>(*served));
  ajp::subs::test::check_atlas(std::get<SubtitleFrame>(*served));
  CHECK(bench.pool.in_flight() == ajp::subs::kMaxAtlasesInFlight);
}

TEST_CASE("Rendu : l'échelle des polices agrandit le texte, en direct", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const Box normal = draw(bench.render(0.5)).covered();
  const Box bigger = draw(bench.render(0.5, kScriptWidth, kScriptHeight, 1.5F)).covered();
  REQUIRE_FALSE(normal.empty());
  REQUIRE_FALSE(bigger.empty());
  CHECK(bigger.y1 - bigger.y0 > (normal.y1 - normal.y0) * 5 / 4);
  CHECK(bigger.x1 - bigger.x0 > (normal.x1 - normal.x0) * 5 / 4);
  const Box back = draw(bench.render(0.5)).covered();  // retour à 1 : la taille d'origine
  CHECK(back.y1 - back.y0 == normal.y1 - normal.y0);
}

TEST_CASE("Rendu : refuse une cible hors plage", "[render]") {
  Bench bench(ajp::subs::test::reference_script());
  const auto refused = [&](ajp::subs::RenderTarget target) {
    auto outcome =
        bench.renderer.render(bench.track, {.pts_us = 0, .epoch = 0, .target = target}, bench.pool);
    return !outcome && outcome.error().code == RefusalCode::FieldOutOfRange;
  };
  CHECK(refused({.width = 0, .height = 360}));
  CHECK(refused({.width = 640, .height = ajp::subs::kMaxTargetSide + 1}));
  CHECK(refused({.width = 640, .height = 360, .video_width = 1920, .video_height = 0}));
  CHECK(refused({.width = 640, .height = 360, .font_scale = 0.1F}));
  CHECK(refused({.width = 640, .height = 360, .font_scale = 5.0F}));
  CHECK(ajp::subs::SubtitleRenderer::create(bench.library, {.render_threads = -1}).error().code ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("Rendu : un fichier SRT converti s'affiche à ses horodatages, en bas, coloré", "[render][srt]") {
  Bench bench(
      "1\n00:00:01,000 --> 00:00:02,000\n<font color=\"#00FF00\">Vert</font>\n\n"
      "2\n00:00:03,000 --> 00:00:04,000\n<i>Penché</i>\nsur deux lignes\n",
      Bench::Format::Srt);
  CHECK(bench.render(0.5).atlas == nullptr);
  CHECK(bench.render(2.5).atlas == nullptr);

  const SubtitleFrame green = bench.render(1.5);
  ajp::subs::test::check_atlas(green);
  bool fill = false;
  for (const auto& rect : green.rects) fill = fill || rect.color == ajp::subs::Rgba8{0, 255, 0, 255};
  CHECK(fill);
  const Box one = draw(green).covered();
  CHECK(one.y0 > kScriptHeight / 2);  // style Default : en bas

  // Deux lignes : nettement plus haut qu'une seule.
  const Box two = draw(bench.render(3.5)).covered();
  CHECK(two.y1 - two.y0 > 3 * (one.y1 - one.y0) / 2);
}
