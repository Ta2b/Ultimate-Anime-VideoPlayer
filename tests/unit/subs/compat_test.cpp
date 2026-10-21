// Compatibilité avec le lecteur actuel (mpv) : couleurs VSFilter, polices des
// sections [Fonts], et refus unique d'un contenu trop grand pour l'atlas.
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subs_worker.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"
#include "subs/vsfilter_color.h"
#include "support.h"

using ajp::ColorMatrix;
using ajp::ColorRange;
using ajp::RefusalCode;
using ajp::subs::Rgba8;
using ajp::subs::ScriptMatrix;
using ajp::subs::SubtitleFrame;
using ajp::subs::VideoMatrix;
using ajp::subs::vsfilter_bt601_on_bt709;
using ajp::subs::vsfilter_converts;

namespace {

constexpr VideoMatrix kHd{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited};

// Une piste, un moteur et sa réserve ; les atlas relâchés reviennent à chaque rendu.
struct Scene {
  ajp::subs::AssLibrary library = ajp::subs::AssLibrary::create().value();
  std::vector<std::unique_ptr<ajp::subs::SubtitleAtlas>> returned;
  ajp::subs::AtlasPool pool{
      [this](std::unique_ptr<ajp::subs::SubtitleAtlas> atlas) { returned.push_back(std::move(atlas)); }};
  ajp::subs::SubtitleRenderer renderer;
  ajp::subs::SubtitleTrack track;

  explicit Scene(std::string_view script, ajp::subs::RendererOptions options = {})
      : renderer(ajp::subs::SubtitleRenderer::create(library, options).value()),
        track(ajp::subs::SubtitleTrack::from_ass(library, script).value()) {}

  auto render(double seconds, ajp::subs::RenderTarget target) {
    for (auto& atlas : returned) pool.give_back(std::move(atlas));
    returned.clear();
    return renderer.render(
        track, {.pts_us = static_cast<std::int64_t>(seconds * 1e6), .epoch = 0, .target = target}, pool);
  }

  SubtitleFrame frame(double seconds, ajp::subs::RenderTarget target) {
    auto outcome = render(seconds, target);
    REQUIRE(outcome);
    REQUIRE(std::holds_alternative<SubtitleFrame>(*outcome));
    return std::get<SubtitleFrame>(std::move(*outcome));
  }

  // Couleur de remplissage du texte rouge \1c&H0000FF& à 2,5 s.
  Rgba8 red_text(std::optional<VideoMatrix> video) {
    const SubtitleFrame red = frame(2.5, {.width = 640, .height = 360, .video_matrix = video});
    REQUIRE_FALSE(red.rects.empty());
    for (const auto& rect : red.rects) CHECK(rect.color == red.rects.front().color);
    return red.rects.front().color;
  }
};

std::string with_matrix(std::string_view matrix) {
  return ajp::subs::test::with_info_line(ajp::subs::test::reference_script(), matrix);
}

}  // namespace

TEST_CASE("vsfilter_bt601_on_bt709 : RGB, YCbCr BT.601 limité sur 8 bits, puis RGB par BT.709", "[compat]") {
  // Valeurs de la conversion de mpv (mp_get_csp_matrix puis mp_map_fixp_color), recalculées à part.
  CHECK(vsfilter_bt601_on_bt709({255, 0, 0, 255}) == Rgba8{255, 24, 0, 255});
  CHECK(vsfilter_bt601_on_bt709({0, 255, 0, 255}) == Rgba8{0, 216, 0, 255});
  CHECK(vsfilter_bt601_on_bt709({0, 0, 255, 255}) == Rgba8{0, 15, 255, 255});
  CHECK(vsfilter_bt601_on_bt709({255, 255, 0, 255}) == Rgba8{255, 240, 0, 255});
  CHECK(vsfilter_bt601_on_bt709({255, 128, 0, 255}) == Rgba8{255, 133, 0, 255});
  CHECK(vsfilter_bt601_on_bt709({200, 100, 50, 255}) == Rgba8{209, 107, 46, 255});
  // Les gris ne changent pas : les deux matrices ont la même luminance de blanc.
  CHECK(vsfilter_bt601_on_bt709({255, 255, 255, 255}) == Rgba8{255, 255, 255, 255});
  CHECK(vsfilter_bt601_on_bt709({128, 128, 128, 255}) == Rgba8{128, 128, 128, 255});
  CHECK(vsfilter_bt601_on_bt709({0, 0, 0, 255}) == Rgba8{0, 0, 0, 255});
  CHECK(vsfilter_bt601_on_bt709({255, 0, 0, 0x37}).a == 0x37);  // l'alpha passe tel quel
}

TEST_CASE("vsfilter_converts : seulement TV.601 ou sans en-tête, sur BT.709 limité", "[compat]") {
  CHECK(vsfilter_converts(ScriptMatrix::Bt601Tv, kHd));
  CHECK(vsfilter_converts(ScriptMatrix::Absent, kHd));
  CHECK_FALSE(vsfilter_converts(ScriptMatrix::None, kHd));
  CHECK_FALSE(vsfilter_converts(ScriptMatrix::Other, kHd));
  CHECK_FALSE(vsfilter_converts(ScriptMatrix::Bt601Tv, std::nullopt));
  CHECK_FALSE(vsfilter_converts(ScriptMatrix::Bt601Tv, VideoMatrix{.matrix = ColorMatrix::Bt601}));
  CHECK_FALSE(vsfilter_converts(ScriptMatrix::Bt601Tv, VideoMatrix{.range = ColorRange::Full}));
  CHECK_FALSE(vsfilter_converts(ScriptMatrix::Bt601Tv, VideoMatrix{.matrix = ColorMatrix::Bt2020Ncl}));
}

TEST_CASE("Rendu : un script TV.601 sur une vidéo BT.709 prend les couleurs de VSFilter",
          "[compat][render]") {
  Scene tv601(with_matrix("YCbCr Matrix: TV.601"));
  CHECK(tv601.track.script_matrix() == ScriptMatrix::Bt601Tv);
  CHECK(tv601.red_text(kHd) == Rgba8{255, 24, 0, 255});
  CHECK(tv601.red_text(std::nullopt) == Rgba8{255, 0, 0, 255});  // vidéo inconnue : telles quelles
  CHECK(tv601.red_text(VideoMatrix{.matrix = ColorMatrix::Bt601}) == Rgba8{255, 0, 0, 255});
  CHECK(tv601.red_text(kHd) == Rgba8{255, 24, 0, 255});  // et retour, sur le même moteur

  Scene absent(ajp::subs::test::reference_script());
  CHECK(absent.track.script_matrix() == ScriptMatrix::Absent);
  CHECK(absent.red_text(kHd) == Rgba8{255, 24, 0, 255});

  Scene tv709(with_matrix("YCbCr Matrix: TV.709"));
  CHECK(tv709.track.script_matrix() == ScriptMatrix::Other);
  CHECK(tv709.red_text(kHd) == Rgba8{255, 0, 0, 255});

  Scene none(with_matrix("YCbCr Matrix: None"));
  CHECK(none.red_text(kHd) == Rgba8{255, 0, 0, 255});
}

TEST_CASE("Rendu : même image, autre en-tête de piste, couleurs refaites", "[compat][render]") {
  // libass voit les mêmes couvertures (change 0) ; seules les couleurs diffèrent.
  Scene scene(with_matrix("YCbCr Matrix: TV.601"));
  const auto plain = ajp::subs::SubtitleTrack::from_ass(scene.library, with_matrix("YCbCr Matrix: None"));
  REQUIRE(plain);
  CHECK(scene.red_text(kHd) == Rgba8{255, 24, 0, 255});
  const ajp::subs::RenderTarget target{.width = 640, .height = 360, .video_matrix = kHd};
  auto outcome =
      scene.renderer.render(*plain, {.pts_us = 2'500'000, .epoch = 0, .target = target}, scene.pool);
  REQUIRE(outcome);
  REQUIRE(std::holds_alternative<SubtitleFrame>(*outcome));
  const auto& frame = std::get<SubtitleFrame>(*outcome);
  REQUIRE_FALSE(frame.rects.empty());
  CHECK(frame.rects.front().color == Rgba8{255, 0, 0, 255});
}

TEST_CASE("Piste texte : en-tête None, couleurs jamais converties", "[compat][track]") {
  auto library = ajp::subs::AssLibrary::create();
  REQUIRE(library);
  auto track = ajp::subs::SubtitleTrack::text(*library);
  REQUIRE(track);
  CHECK(track->script_matrix() == ScriptMatrix::None);
  CHECK_FALSE(track->embeds_fonts());
}

TEST_CASE("Rendu : les polices d'une section [Fonts] servent sans police système", "[compat][render]") {
  const std::vector<std::byte> font = ajp::subs::test::system_font_bytes();
  if (font.empty()) SKIP("DejaVu Sans absente du système");
  const ajp::subs::RendererOptions no_system{.fonts = ajp::subs::FontProvider::None};
  const ajp::subs::RenderTarget target{.width = 640, .height = 360};

  Scene without(ajp::subs::test::reference_script(), no_system);
  CHECK_FALSE(without.track.embeds_fonts());
  CHECK(without.frame(0.5, target).atlas == nullptr);  // témoin : aucune police connue

  const std::string script = std::string(ajp::subs::test::reference_script()) +
                             ajp::subs::test::ass_font_section("DejaVuSans_0.ttf", font);
  Scene with(script, no_system);
  CHECK(with.track.embeds_fonts());
  ajp::subs::test::check_atlas(with.frame(0.5, target));
}

TEST_CASE("SubsWorker : les polices d'un script choisi après un premier rendu servent", "[compat][worker]") {
  const std::vector<std::byte> font = ajp::subs::test::system_font_bytes();
  if (font.empty()) SKIP("DejaVu Sans absente du système");
  ajp::subs::test::RecordingSink sink;
  const auto discard = [](std::unique_ptr<ajp::subs::SubtitleAtlas>) {};
  auto worker = ajp::subs::SubsWorker::create({.fonts = ajp::subs::FontProvider::None}, sink, discard);
  REQUIRE(worker);
  const auto select = [&](std::uint32_t id, std::string_view script) {
    const ajp::subs::test::TempFile file("piste.ass", script);
    (*worker)->handle(ajp::subs::SelectTrack{.id = id, .source = ajp::subs::ExternalFile{file.path()}});
  };
  const ajp::subs::RenderRequest request{
      .pts_us = 500'000, .epoch = 0, .target = {.width = 640, .height = 360}};

  select(1, ajp::subs::test::reference_script());
  (*worker)->handle(request);  // le moteur cherche DejaVu Sans, en vain, et s'en souvient
  select(2, std::string(ajp::subs::test::reference_script()) +
                ajp::subs::test::ass_font_section("DejaVuSans_0.ttf", font));
  (*worker)->handle(request);
  auto frames = sink.take_frames();
  REQUIRE(frames.size() == 2);
  CHECK(frames[0].atlas == nullptr);
  ajp::subs::test::check_atlas(frames[1]);
}

TEST_CASE("Rendu : un contenu trop grand pour l'atlas est refusé une fois par apparition",
          "[compat][render]") {
  // Un carré plein sur toute l'image, bordé : remplissage et bordure font chacun
  // 2050 × 2050, et deux tels rectangles ne tiennent pas dans 4096 × 4096.
  const std::string script =
      "[Script Info]\nScriptType: v4.00+\nPlayResX: 100\nPlayResY: 100\n\n"
      "[V4+ Styles]\n"
      "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
      "Italic, "
      "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
      "MarginL, "
      "MarginR, MarginV, Encoding\n"
      "Style: Default,DejaVu "
      "Sans,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,0,0,0,"
      "1\n\n"
      "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
      "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\pos(0,0)\\p1}m 0 0 l 100 0 100 100 0 100\n"
      "Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,{\\pos(0,0)\\p1}m 0 0 l 100 0 100 100 0 100\n";
  Scene scene(script);
  const ajp::subs::RenderTarget target{.width = 2050, .height = 2050};

  auto first = scene.render(0.2, target);
  REQUIRE_FALSE(first);
  CHECK(first.error().code == RefusalCode::SubtitleAtlasFull);
  // Même contenu : image vide, sans nouveau refus.
  CHECK(scene.frame(0.6, target).atlas == nullptr);
  CHECK(scene.frame(0.9, target).atlas == nullptr);
  CHECK(scene.frame(1.5, target).atlas == nullptr);  // entre les deux événements

  auto again = scene.render(2.5, target);  // nouvelle apparition : nouveau refus
  REQUIRE_FALSE(again);
  CHECK(again.error().code == RefusalCode::SubtitleAtlasFull);
}
