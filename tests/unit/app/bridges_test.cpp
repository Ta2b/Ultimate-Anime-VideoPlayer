// Les sous-titres texte d'un MP4 (mov_text) et d'un WebVTT passent au module de sous-titres (A18-15).
#include "app/play/bridges.h"

#include <array>
#include <cstddef>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

std::string_view text_of(std::span<const std::byte> bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

TEST_CASE("mov_text : le texte derrière sa longueur, sans les boîtes de style", "[app][subs]") {
  // « Salut » (5 octets), puis une boîte de style « styl » que le module ne lit pas.
  constexpr std::array kPacket{std::byte{0},   std::byte{5},   std::byte{'S'}, std::byte{'a'},
                               std::byte{'l'}, std::byte{'u'}, std::byte{'t'}, std::byte{0},
                               std::byte{0},   std::byte{0},   std::byte{10},  std::byte{'s'},
                               std::byte{'t'}, std::byte{'y'}, std::byte{'l'}};
  CHECK(text_of(mov_text_payload(kPacket)) == "Salut");
  // Une longueur qui dépasse le paquet (paquet coupé) : ce qui reste ; trop court : rien.
  constexpr std::array kCut{std::byte{0}, std::byte{9}, std::byte{'O'}, std::byte{'k'}};
  CHECK(text_of(mov_text_payload(kCut)) == "Ok");
  constexpr std::array kTiny{std::byte{0}};
  CHECK(mov_text_payload(kTiny).empty());
}

}  // namespace
}  // namespace ajp::app::play
