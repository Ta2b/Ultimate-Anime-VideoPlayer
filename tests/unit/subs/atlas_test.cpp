#include <cstddef>
#include <format>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "subs/atlas_packer.h"
#include "subs/atlas_pool.h"

namespace {

using ajp::RefusalCode;
using ajp::subs::kAtlasPadding;
using ajp::subs::Packing;
using ajp::subs::PackSize;

// Propriétés d'un empaquetage réussi : tout est dans l'atlas, et deux
// rectangles, agrandis de l'écart vers la droite et le bas, ne se touchent pas.
void check_packing(const std::vector<PackSize>& sizes, const Packing& packing, int max_side) {
  REQUIRE(packing.positions.size() == sizes.size());
  REQUIRE(packing.width <= max_side);
  REQUIRE(packing.height <= max_side);
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const auto [x, y] = packing.positions[i];
    REQUIRE(x >= 0);
    REQUIRE(y >= 0);
    REQUIRE(x + sizes[i].width <= packing.width);
    REQUIRE(y + sizes[i].height <= packing.height);
  }
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    for (std::size_t j = 0; j < sizes.size(); ++j) {
      if (i == j) continue;
      const auto a = packing.positions[i];
      const auto b = packing.positions[j];
      const bool apart = a.x + sizes[i].width + kAtlasPadding <= b.x || b.x + sizes[j].width <= a.x ||
                         a.y + sizes[i].height + kAtlasPadding <= b.y || b.y + sizes[j].height <= a.y;
      if (!apart) FAIL(std::format("rectangles {} et {} trop proches", i, j));
    }
  }
}

std::vector<PackSize> random_sizes(std::mt19937& random, int count, int largest) {
  std::uniform_int_distribution<int> side(1, largest);
  std::uniform_int_distribution<int> shape(0, 3);
  std::vector<PackSize> sizes;
  for (int i = 0; i < count; ++i) {
    PackSize size{side(random), side(random)};
    if (shape(random) == 0) size.height = 1 + size.height / 16;  // lignes de texte étirées
    sizes.push_back(size);
  }
  return sizes;
}

}  // namespace

TEST_CASE("pack : propriété, aucun recouvrement et tout dans l'atlas", "[atlas]") {
  for (unsigned seed = 1; seed <= 200; ++seed) {
    std::mt19937 random(seed);
    const int count = std::uniform_int_distribution<int>(1, 120)(random);
    const auto sizes = random_sizes(random, count, 300);
    const auto packing = ajp::subs::pack(sizes, 4096);
    REQUIRE(packing);
    check_packing(sizes, *packing, 4096);
  }
}

TEST_CASE("pack : propriété sous un côté serré, refus seulement par manque de place", "[atlas]") {
  constexpr int kSide = 256;
  for (unsigned seed = 1; seed <= 200; ++seed) {
    std::mt19937 random(seed);
    const int count = std::uniform_int_distribution<int>(1, 60)(random);
    const auto sizes = random_sizes(random, count, 64);
    const auto packing = ajp::subs::pack(sizes, kSide);
    if (packing) {
      check_packing(sizes, *packing, kSide);
    } else {
      CHECK(packing.error().code == RefusalCode::SubtitleAtlasFull);
    }
    long long area = 0;
    for (const PackSize& size : sizes) area += static_cast<long long>(size.width) * size.height;
    if (area > static_cast<long long>(kSide) * kSide) CHECK_FALSE(packing);
  }
}

TEST_CASE("pack : déterministe et tassé", "[atlas]") {
  std::mt19937 random(7);
  const auto sizes = random_sizes(random, 80, 120);
  const auto first = ajp::subs::pack(sizes, 4096);
  const auto second = ajp::subs::pack(sizes, 4096);
  REQUIRE(first);
  REQUIRE(second);
  CHECK(first->positions == second->positions);
  // Une étagère sur un carré de l'aire : l'atlas ne gaspille pas un facteur 4.
  long long area = 0;
  for (const PackSize& size : sizes) area += static_cast<long long>(size.width) * size.height;
  CHECK(static_cast<long long>(first->width) * first->height < 4 * area);
}

TEST_CASE("pack : bornes et refus", "[atlas]") {
  CHECK(ajp::subs::pack({}, 64)->positions.empty());
  const std::vector<PackSize> exact{{64, 64}};
  const auto full = ajp::subs::pack(exact, 64);
  REQUIRE(full);
  CHECK(full->width == 64);
  CHECK(full->height == 64);

  const std::vector<PackSize> too_wide{{65, 1}};
  CHECK(ajp::subs::pack(too_wide, 64).error().code == RefusalCode::SubtitleAtlasFull);
  const std::vector<PackSize> two_full{{64, 64}, {1, 1}};
  CHECK(ajp::subs::pack(two_full, 64).error().code == RefusalCode::SubtitleAtlasFull);
  const std::vector<PackSize> empty_rect{{0, 5}};
  CHECK(ajp::subs::pack(empty_rect, 64).error().code == RefusalCode::FieldOutOfRange);
  CHECK(ajp::subs::pack(exact, 0).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("AtlasPool : au plus kMaxAtlasesInFlight, recyclés par la route", "[atlas]") {
  std::vector<std::unique_ptr<ajp::subs::SubtitleAtlas>> returned;
  ajp::subs::AtlasPool pool(
      [&](std::unique_ptr<ajp::subs::SubtitleAtlas> atlas) { returned.push_back(std::move(atlas)); });

  std::vector<std::shared_ptr<const ajp::subs::SubtitleAtlas>> shared;
  for (std::size_t i = 0; i < ajp::subs::kMaxAtlasesInFlight; ++i) {
    auto atlas = pool.take();
    REQUIRE(atlas != nullptr);
    atlas->pixels.assign(16, 1);
    shared.push_back(pool.share(std::move(atlas)));
  }
  CHECK(pool.in_flight() == ajp::subs::kMaxAtlasesInFlight);
  CHECK(pool.take() == nullptr);

  // Une copie tenue ailleurs retarde le retour jusqu'à la dernière référence.
  auto copy = shared.front();
  shared.erase(shared.begin());
  CHECK(returned.empty());
  const auto* address = copy.get();
  copy.reset();
  REQUIRE(returned.size() == 1);
  CHECK(returned.front().get() == address);

  pool.give_back(std::move(returned.front()));
  returned.clear();
  CHECK(pool.in_flight() == ajp::subs::kMaxAtlasesInFlight - 1);
  auto recycled = pool.take();
  CHECK(recycled.get() == address);
  CHECK(recycled->pixels.capacity() >= 16);  // la mémoire est réutilisée

  // Tout rentre ; un atlas de trop est détruit, la borne tient.
  pool.give_back(std::move(recycled));
  shared.clear();
  REQUIRE(returned.size() == ajp::subs::kMaxAtlasesInFlight - 1);
  for (auto& atlas : returned) pool.give_back(std::move(atlas));
  CHECK(pool.in_flight() == 0);
  pool.give_back(std::make_unique<ajp::subs::SubtitleAtlas>());
  CHECK(pool.in_flight() == 0);
  std::vector<std::unique_ptr<ajp::subs::SubtitleAtlas>> taken;
  while (auto atlas = pool.take()) taken.push_back(std::move(atlas));
  CHECK(taken.size() == ajp::subs::kMaxAtlasesInFlight);
}
