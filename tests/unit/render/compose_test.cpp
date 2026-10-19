// Composition hors écran : SDR à l'identique, blanc SDR en PQ, sous-titres, grade.
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "render/compositor.h"
#include "support.h"

namespace ajp::render {
namespace {

#define REQUIRE_DEVICE(device)                  \
  std::string skip_reason;                      \
  auto device = test::make_device(skip_reason); \
  if (!device) SKIP("pas d'appareil Vulkan : " << skip_reason)

constexpr SlotColor kSrgb{.primaries = Primaries::Bt709, .transfer = Transfer::Srgb};

// Dégradé 2D connu : chaque pixel a ses trois valeurs propres.
std::vector<std::uint16_t> gradient(int w, int h) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(w * h * 4));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const auto i = static_cast<std::size_t>((y * w + x) * 4);
      out[i] = static_cast<std::uint16_t>(x * 65535 / (w - 1));
      out[i + 1] = static_cast<std::uint16_t>(y * 65535 / (h - 1));
      out[i + 2] = static_cast<std::uint16_t>((x + y) * 65535 / (w + h - 2));
      out[i + 3] = 65535;
    }
  }
  return out;
}

struct Scene {
  pl_gpu gpu;
  pl_tex slot = nullptr;
  ~Scene() { pl_tex_destroy(gpu, &slot); }
};

std::vector<std::uint16_t> compose_once(const RenderDevice& device, pl_tex slot, const SlotColor& color,
                                        const CompositionSettings& settings, int w, int h, bool grade = true,
                                        const subs::SubtitleFrame* subtitles = nullptr) {
  auto compositor = Compositor::create(device);
  auto target = OffscreenTarget::create(device, {.width = w, .height = h, .display = settings.display});
  if (!compositor || !target || !compositor->configure(settings)) return {};
  compositor->set_grade_enabled(grade);
  if (!compositor->compose(slot, color, target->frame(), {.subtitles = subtitles})) return {};
  return test::read_back(*target);
}

}  // namespace

TEST_CASE("un slot RGB connu rendu hors écran en SDR redonne ses pixels", "[gpu]") {
  REQUIRE_DEVICE(device);
  constexpr int w = 48;
  constexpr int h = 32;
  const auto source = gradient(w, h);
  Scene scene{.gpu = device->gpu(), .slot = test::make_texture(device->gpu(), w, h, source)};
  REQUIRE(scene.slot != nullptr);
  const auto pixels = compose_once(*device, scene.slot, kSrgb, {.display = test::sdr_display()}, w, h);
  REQUIRE(pixels.size() == source.size());
  float worst = 0.0F;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        worst =
            std::max(worst, std::abs(test::channel(pixels, w, x, y, c) - test::channel(source, w, x, y, c)));
      }
    }
  }
  INFO("écart maximal " << worst);
  CHECK(worst < 1.0F / 1024.0F);
}

TEST_CASE("l'image garde son rapport d'aspect, avec des bandes noires", "[gpu]") {
  REQUIRE_DEVICE(device);
  const auto white = test::solid(16, 16, 1.0F, 1.0F, 1.0F);
  Scene scene{.gpu = device->gpu(), .slot = test::make_texture(device->gpu(), 16, 16, white)};
  const auto pixels = compose_once(*device, scene.slot, kSrgb, {.display = test::sdr_display()}, 64, 32);
  REQUIRE(pixels.size() == 64U * 32U * 4U);
  CHECK(test::channel(pixels, 64, 2, 16, 0) < 1e-3F);   // bande de gauche
  CHECK(test::channel(pixels, 64, 61, 16, 0) < 1e-3F);  // bande de droite
  CHECK(test::channel(pixels, 64, 32, 16, 0) > 0.999F);
  CHECK(test::channel(pixels, 64, 20, 2, 1) > 0.999F);
}

TEST_CASE("une cible PQ code le blanc SDR à 203 nits", "[gpu]") {
  REQUIRE_DEVICE(device);
  const auto white = test::solid(16, 16, 1.0F, 1.0F, 1.0F);
  Scene scene{.gpu = device->gpu(), .slot = test::make_texture(device->gpu(), 16, 16, white)};
  const CompositionSettings settings{.display = test::pq_display(1000.0), .route = core::HdrMode::Native};
  const auto pixels = compose_once(*device, scene.slot, kSrgb, settings, 16, 16, /*grade=*/false);
  REQUIRE_FALSE(pixels.empty());
  const double expected = test::pq_encode(203.0);
  for (int c = 0; c < 3; ++c) {
    INFO("canal " << c << " : " << test::channel(pixels, 16, 8, 8, c) << ", attendu " << expected);
    CHECK(std::abs(test::channel(pixels, 16, 8, 8, c) - expected) < 2e-3);
  }
}

TEST_CASE("les sous-titres se posent au bon endroit, de la bonne couleur et au bon niveau", "[gpu]") {
  REQUIRE_DEVICE(device);
  constexpr int w = 64;
  constexpr int h = 32;
  const auto black = test::solid(w, h, 0.0F, 0.0F, 0.0F);
  Scene scene{.gpu = device->gpu(), .slot = test::make_texture(device->gpu(), w, h, black)};
  auto atlas = std::make_shared<subs::SubtitleAtlas>();
  atlas->generation = 1;
  atlas->width = 16;
  atlas->height = 8;
  atlas->pixels.assign(16U * 8U, 0);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) atlas->pixels[static_cast<std::size_t>(y * 16 + x)] = 255;
  }
  subs::SubtitleFrame frame{.width = w, .height = h, .atlas = atlas};
  frame.rects.push_back(subs::SubtitleRect{
      .src = {.x = 0, .y = 0, .width = 6, .height = 4}, .dst_x = 20, .dst_y = 10, .color = {255, 0, 0, 255}});

  SECTION("cible SDR : rouge plein sur le rectangle, rien ailleurs") {
    const auto pixels =
        compose_once(*device, scene.slot, kSrgb, {.display = test::sdr_display()}, w, h, true, &frame);
    REQUIRE_FALSE(pixels.empty());
    CHECK(test::channel(pixels, w, 22, 12, 0) > 0.995F);
    CHECK(test::channel(pixels, w, 22, 12, 1) < 0.005F);
    CHECK(test::channel(pixels, w, 25, 13, 0) > 0.995F);
    CHECK(test::channel(pixels, w, 19, 12, 0) < 0.005F);  // à gauche du rectangle
    CHECK(test::channel(pixels, w, 26, 12, 0) < 0.005F);  // à droite
    CHECK(test::channel(pixels, w, 22, 14, 0) < 0.005F);  // dessous
  }
  SECTION("cible PQ : le blanc des sous-titres vaut subtitle_white") {
    frame.rects[0].color = {255, 255, 255, 255};
    CompositionSettings settings{.display = test::pq_display(1000.0), .route = core::HdrMode::Native};
    settings.grade.subtitle_white = 200.0;
    const auto pixels = compose_once(*device, scene.slot, kSrgb, settings, w, h, true, &frame);
    REQUIRE_FALSE(pixels.empty());
    const double expected = test::pq_encode(200.0);
    INFO("sous-titre " << test::channel(pixels, w, 22, 12, 1) << ", attendu " << expected);
    CHECK(std::abs(test::channel(pixels, w, 22, 12, 1) - expected) < 3e-3);
    CHECK(test::channel(pixels, w, 30, 20, 1) < 0.01F);
  }
  SECTION("un rectangle hors de l'atlas est refusé") {
    frame.rects[0].src = {.x = 12, .y = 0, .width = 6, .height = 4};
    auto compositor = Compositor::create(*device);
    auto target = OffscreenTarget::create(*device, {.width = w, .height = h, .display = test::sdr_display()});
    REQUIRE(compositor);
    REQUIRE(target);
    const auto refused = compositor->compose(scene.slot, kSrgb, target->frame(), {.subtitles = &frame});
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::InputMismatch);
  }
}

TEST_CASE("le grade change l'image et reste déterministe", "[gpu]") {
  REQUIRE_DEVICE(device);
  constexpr int w = 64;
  constexpr int h = 64;
  const auto source = gradient(w, h);
  Scene scene{.gpu = device->gpu(), .slot = test::make_texture(device->gpu(), w, h, source)};
  const CompositionSettings settings{.display = test::pq_display(1000.0), .route = core::HdrMode::Native};
  const auto graded = compose_once(*device, scene.slot, kSrgb, settings, w, h, true);
  const auto again = compose_once(*device, scene.slot, kSrgb, settings, w, h, true);
  const auto neutral = compose_once(*device, scene.slot, kSrgb, settings, w, h, false);
  REQUIRE_FALSE(graded.empty());
  REQUIRE(graded.size() == neutral.size());
  CHECK(graded == again);
  std::size_t changed = 0;
  for (std::size_t i = 0; i < graded.size(); ++i) changed += graded[i] != neutral[i] ? 1U : 0U;
  INFO(changed << " valeurs changées sur " << graded.size());
  CHECK(changed > graded.size() / 4);
}

TEST_CASE("OLED : le vrai noir met les quasi-noirs à 0 et laisse le reste", "[gpu]") {
  REQUIRE_DEVICE(device);
  constexpr int w = 32;
  constexpr int h = 16;
  // Moitié gauche presque noire, moitié droite gris moyen.
  auto pixels_in = test::solid(w, h, 0.5F, 0.5F, 0.5F);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w / 2; ++x) {
      for (int c = 0; c < 3; ++c) pixels_in[static_cast<std::size_t>((y * w + x) * 4 + c)] = 300;
    }
  }
  Scene scene{.gpu = device->gpu(), .slot = test::make_texture(device->gpu(), w, h, pixels_in)};
  CompositionSettings settings{.display = test::pq_display(1000.0), .route = core::HdrMode::Native};
  settings.grade.oled = true;
  const auto without = compose_once(*device, scene.slot, kSrgb, settings, w, h);
  settings.grade.black_lo = 1.0;
  settings.grade.black_hi = 2.0;
  const auto with = compose_once(*device, scene.slot, kSrgb, settings, w, h);
  REQUIRE_FALSE(without.empty());
  REQUIRE(with.size() == without.size());
  CHECK(test::channel(without, w, 2, 8, 1) > 0.0F);  // le quasi-noir existe sans la passe
  CHECK(test::channel(with, w, 2, 8, 1) == 0.0F);
  CHECK(test::channel(with, w, 28, 8, 1) == test::channel(without, w, 28, 8, 1));
}

}  // namespace ajp::render
