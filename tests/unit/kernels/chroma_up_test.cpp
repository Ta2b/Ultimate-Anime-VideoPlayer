// La chroma Spline36 de l'ingestion CUDA (spec/chroma_up.h) contre la référence (ref/resample.h) : les deux
// phases sont la table de compute_weights, et leur application rend la même image, au bit près (A03-01).
#include "kernels/spec/chroma_up.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/resample.h"

using namespace ajp::kernels;

namespace {

// La chroma portée à width × height par les deux passes de chroma_up_at, comme le noyau CUDA : lignes
// d'abord (codes bruts), colonnes ensuite, intermédiaire float.
PlaneF up_by_phases(const PlaneF& chroma, int width, int height, ChromaShift shift) {
  const ChromaUpPhase wx[2] = {chroma_up_phase(0, shift.x), chroma_up_phase(1, shift.x)};
  const ChromaUpPhase wy[2] = {chroma_up_phase(0, shift.y), chroma_up_phase(1, shift.y)};
  PlaneF rows(width, chroma.height);
  for (int y = 0; y < chroma.height; ++y)
    for (int x = 0; x < width; ++x)
      rows.at(x, y) = chroma_up_at(wx[x & 1], x, chroma.width, [&](int k) { return chroma.at(k, y); });
  PlaneF out(width, height);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      out.at(x, y) = chroma_up_at(wy[y & 1], y, chroma.height, [&](int k) { return rows.at(x, k); });
  return out;
}

// Des codes bruts de 10 bits, sans motif régulier (un générateur congruentiel suffit).
PlaneF noise(int width, int height, std::uint32_t seed) {
  PlaneF plane(width, height);
  std::uint32_t state = seed;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      state = state * 1664525U + 1013904223U;
      plane.at(x, y) = static_cast<float>(64U + (state >> 16U) % 896U);
    }
  }
  return plane;
}

}  // namespace

TEST_CASE("chroma Spline36 : les deux phases sont la table de compute_weights", "[kernels][chroma_up]") {
  for (const Siting siting : {Siting::Left, Siting::Center, Siting::TopLeft}) {
    const ChromaShift shift = chroma_shift_up(siting);
    for (const double axis_shift : {shift.x, shift.y}) {
      const ChromaUpPhase phases[2] = {chroma_up_phase(0, axis_shift), chroma_up_phase(1, axis_shift)};
      for (const int n : {2, 4, 6, 10, 48, 1080, 1920, 3840}) {
        const auto table = ref::compute_weights(n / 2, n, axis_shift, Filter::Spline36);
        REQUIRE(table);
        REQUIRE(table->taps == kChromaUpTaps);
        for (int i = 0; i < n; ++i) {
          const ChromaUpPhase& phase = phases[i & 1];
          REQUIRE(table->start[static_cast<std::size_t>(i)] == i / 2 + phase.offset);
          for (int j = 0; j < kChromaUpTaps; ++j)
            REQUIRE(table->row(i)[static_cast<std::size_t>(j)] == phase.w[j]);  // au bit près
        }
      }
    }
  }
}

TEST_CASE("chroma Spline36 : le miroir du bord est celui de la référence", "[kernels][chroma_up]") {
  for (const int n : {1, 2, 3, 7})
    for (int i = -8; i < n + 8; ++i) CHECK(chroma_up_mirror(i, n) == mirror_index(i, n));
}

TEST_CASE("chroma Spline36 : l'application rend l'image de la référence, bords compris",
          "[kernels][chroma_up]") {
  std::uint32_t seed = 7;
  for (const Siting siting : {Siting::Left, Siting::Center, Siting::TopLeft}) {
    const ChromaShift shift = chroma_shift_up(siting);
    for (const auto& [width, height] :
         std::vector<std::pair<int, int>>{{2, 2}, {4, 6}, {48, 32}, {130, 74}}) {
      const PlaneF chroma = noise(width / 2, height / 2, ++seed);
      const auto wx = ref::compute_weights(chroma.width, width, shift.x, Filter::Spline36);
      const auto wy = ref::compute_weights(chroma.height, height, shift.y, Filter::Spline36);
      REQUIRE(wx);
      REQUIRE(wy);
      const auto rows = ref::resample_rows(chroma, *wx);
      REQUIRE(rows);
      const auto want = ref::resample_columns(*rows, *wy);
      REQUIRE(want);
      const PlaneF got = up_by_phases(chroma, width, height, shift);
      bool same = true;
      for (int y = 0; y < height && same; ++y)
        for (int x = 0; x < width && same; ++x) same = got.at(x, y) == want->at(x, y);
      CAPTURE(width, height, static_cast<int>(siting));
      CHECK(same);
    }
  }
}

TEST_CASE("chroma Spline36 : un aplat reste un aplat, un échelon dépasse comme un Spline36",
          "[kernels][chroma_up]") {
  PlaneF flat(8, 8);
  for (float& v : flat.data) v = 512.0F;
  const PlaneF up = up_by_phases(flat, 16, 16, chroma_shift_up(Siting::Left));
  // Les six poids float ne somment à 1 qu'à un ulp près.
  for (const float v : up.data) CHECK(std::fabs(v - 512.0F) < 1e-3F);
  // Un bord franc : le bilinéaire reste entre les deux valeurs, le Spline36 dépasse un peu de part et
  // d'autre.
  PlaneF step(8, 2);
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 8; ++x) step.at(x, y) = x < 4 ? 200.0F : 800.0F;
  const PlaneF edge = up_by_phases(step, 16, 4, chroma_shift_up(Siting::Left));
  float low = 800.0F;
  float high = 200.0F;
  for (const float v : edge.data) {
    low = v < low ? v : low;
    high = v > high ? v : high;
  }
  CHECK(low < 200.0F);
  CHECK(high > 800.0F);
}
