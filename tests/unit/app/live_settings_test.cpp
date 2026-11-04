#include "app/play/live_settings.h"

#include <cmath>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

// Une paire de 4 images écrites : `copies` d'entre elles sont des copies à 100 µs, les autres du calcul à 1
// ms.
std::int64_t service_after_pairs(int copies) {
  LiveSettings live;
  live.set_free(true, 1000, 0);
  std::int64_t now = 20'000'000;
  for (std::uint32_t pair = 1; pair <= 40; ++pair) {
    live.auto_expect(pair, 4, 0);
    for (int image = 0; image < 4; ++image) {
      const bool copy = image >= 4 - copies;
      const std::int64_t submitted = now;
      now += copy ? 100 : 1000;
      live.auto_written(image == 3 ? pair : pair - 1, pair, submitted, now, copy);
    }
  }
  return live.free_info(now).service_us;
}

}  // namespace

TEST_CASE("Mode Rapide : les copies ne comptent pas dans le temps de service", "[app][free]") {
  const std::int64_t with_copies = service_after_pairs(2);  // 2 images calculées à 1 ms + 2 copies
  const std::int64_t all_computed = service_after_pairs(0);
  CHECK(with_copies > 900);  // ≈ 1 000 µs : une image calculée, pas (2 000 + 200) / 4 = 550
  CHECK(with_copies < 1100);
  CHECK(all_computed > 900);
}

}  // namespace ajp::app::play

namespace ajp::app::play {
namespace {

GradeVersion posted_at(std::int64_t from_pts_us, std::uint32_t epoch, bool off = false) {
  return GradeVersion{.epoch = epoch, .from_pts_us = from_pts_us, .off = off};
}

}  // namespace

TEST_CASE("Grade de l'anneau : la version d'une image choisie par pts et époque", "[app][grade_ring]") {
  LiveSettings live;
  CHECK(live.grade_at(0, 0) == nullptr);  // rien posté : l'écran grade
  const std::uint32_t v1 = live.post_grade(posted_at(100'000, 1));
  const std::uint32_t v2 = live.post_grade(posted_at(300'000, 1));
  CHECK(v1 == 1);
  CHECK(v2 == 2);
  // Époque 1 : avant 100 ms, aucune ; de 100 à 300 ms, v1 ; dès 300 ms, v2.
  CHECK(live.grade_at(99'999, 1) == nullptr);
  REQUIRE(live.grade_at(200'000, 1) != nullptr);
  CHECK(live.grade_at(200'000, 1)->version == v1);
  CHECK(live.grade_at(300'000, 1)->version == v2);
  // Une époque d'avant ne connaît pas les ordres d'une époque plus récente.
  CHECK(live.grade_at(500'000, 0) == nullptr);
  // Les époques suivantes prennent la dernière, quel que soit le pts (seek en arrière).
  CHECK(live.grade_at(0, 2)->version == v2);
  // Un pts inconnu dans l'époque de l'ordre : pas couvert.
  CHECK(live.grade_at(kNoPts, 1) == nullptr);
  // Un ordre « off » posté après : les images qu'il couvre ne sont plus gradées dans l'anneau.
  const std::uint32_t v3 = live.post_grade(posted_at(400'000, 1, true));
  CHECK(live.grade_at(400'000, 1)->version == v3);
  CHECK(live.grade_at(400'000, 1)->off);
  CHECK(live.grade_at(350'000, 1)->version == v2);
}

TEST_CASE("Grade de l'anneau : un ordre posté à P s'applique dès la première image de pts ≥ P",
          "[app][grade_ring]") {
  LiveSettings live;
  live.post_grade(posted_at(0, 0));
  const std::uint32_t at = live.post_grade(posted_at(1'000'000, 0));
  for (std::int64_t pts = 960'000; pts <= 1'040'000; pts += 40'000) {
    INFO("pts " << pts);
    CHECK((live.grade_at(pts, 0)->version == at) == (pts >= 1'000'000));
  }
}

TEST_CASE("Grade de l'anneau : les kGradeOrders derniers ordres gardés", "[app][grade_ring]") {
  LiveSettings live;
  for (std::uint32_t i = 0; i < gpu::kGradeOrders + 2; ++i)
    live.post_grade(posted_at(static_cast<std::int64_t>(i) * 1000, 0));
  // Les deux premiers (pts 0 et 1000) sont partis : avant 2000, aucune version.
  CHECK(live.grade_at(1'500, 0) == nullptr);
  CHECK(live.grade_at(2'000, 0)->version == 3);
  CHECK(live.grade_at(1'000'000, 0)->version == gpu::kGradeOrders + 2);
}

TEST_CASE("Grade de l'anneau : l'ordre confié au backend pointe dans la version", "[app][grade_ring]") {
  GradeVersion v{.look_size = 2};
  v.look_rgb = std::make_shared<const std::vector<float>>(24, 0.5F);
  v.input_nits = std::make_shared<const std::vector<float>>(3 * kernels::grade::kInputTablePoints, 1.0F);
  v.input_matrix[0] = 1.0F;
  v.uniforms.look_mix = 0.25F;
  v.version = 7;
  const gpu::GradeOrder order = v.order();
  CHECK(order.version == 7);
  CHECK(order.look_size == 2);
  CHECK(order.look_rgb.data() == v.look_rgb->data());
  CHECK(order.input_nits.size() == 3 * static_cast<std::size_t>(kernels::grade::kInputTablePoints));
  CHECK(order.input_matrix[0] == 1.0F);
  CHECK(order.uniforms.look_mix == 0.25F);
  v.look_rgb.reset();
  CHECK(v.order().look_size == 0);  // sans pile, aucun look
}

TEST_CASE("Mode Rapide : la grille des vblanks mesurée prime sur la phase estimée", "[app][free][present]") {
  LiveSettings live;
  live.set_free(true, 6061, 0);
  live.set_vsync(6061, 3'000);  // 165 Hz annoncés, 3 ms entre présentation et affichage
  live.vsync_note(1'000'000);
  const LiveSettings::VsyncGrid estimated = live.vsync_grid(1'000'500);
  CHECK(estimated.period_us == 6061);
  // 164,78 Hz mesurés, vblanks affichés à 2 000 000 + k × 6068,7 µs.
  const core::VblankGrid measured{.phase_us = 2'000'000.0, .period_us = 1e6 / 164.78};
  live.vsync_note(1'006'061, measured);
  const LiveSettings::VsyncGrid grid = live.vsync_grid(2'000'000 + 10 * 6069 + 1'000);
  CHECK(grid.period_us == 6069);
  CHECK(grid.anchor_host_us == std::llround(measured.at(10)) + 6069 / 2);
  // Sans grille mesurée (niveau Open), la phase estimée revient.
  live.vsync_note(1'012'122);
  CHECK(live.vsync_grid(1'000'500).period_us == 6061);
  // VRR : aucune grille, mesurée ou non.
  live.set_vsync(0, 0);
  live.vsync_note(1'018'183, measured);
  CHECK(live.vsync_grid(1'000'500).period_us == 0);
}

}  // namespace ajp::app::play
