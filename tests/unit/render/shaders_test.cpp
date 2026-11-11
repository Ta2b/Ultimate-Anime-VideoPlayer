// Les hooks embarqués (render/shaders.h) : les octets que libplacebo lit, quelle que soit l'extraction.
#include "render/shaders.h"

#include <string_view>

#include <catch2/catch_test_macros.hpp>

namespace {

using namespace ajp::render;

TEST_CASE("les hooks embarqués n'ont aucun retour chariot", "[render]") {
  // Une extraction Windows mettait les .hook en CRLF : la marque des textures du look restait dans
  // l'en-tête, le corps du hook disparaissait, et la pile de LUT ne compilait plus (rc.1 à rc.3).
  for (const std::string_view text : {shaders::anime_hdr_params, shaders::anime_hdr_anime,
                                      shaders::anime_hdr_look, shaders::anime_hdr_black, shaders::sharpen}) {
    REQUIRE_FALSE(text.empty());
    CHECK(text.find('\r') == std::string_view::npos);
  }
}

TEST_CASE("les marques du look sont celles que grade.cpp remplace", "[render]") {
  // grade.cpp retire la ligne entière des textures (marque et « \n ») : une marque laissée dans l'en-tête
  // coupe le hook.
  const std::string_view look = shaders::anime_hdr_look;
  for (const std::string_view mark : {"//@LOOK_PARAMS@", "//@LOOK_BINDS@\n", "//@LOOK_CHAIN@"}) {
    const std::size_t at = look.find(mark);
    REQUIRE(at != std::string_view::npos);
    CHECK(look.find(mark, at + 1) == std::string_view::npos);
  }
}

}  // namespace
