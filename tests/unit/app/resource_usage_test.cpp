// B33-05 : la ligne des ressources du journal dit la dérive et la mémoire vidéo de la chaîne ouverte.
#include "app/play/resource_usage.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("B33-05 : la dérive de la mémoire ou des descripteurs se signale au-delà de la moitié de plus",
          "[resources]") {
  const ProcessResources first{.resident_bytes = 400ULL << 20U, .threads = 40, .handles = 100};
  CHECK(resource_drift(first, first).empty());
  CHECK(resource_drift(first, {.resident_bytes = 590ULL << 20U, .threads = 40, .handles = 149}).empty());
  const std::string memory =
      resource_drift(first, {.resident_bytes = 700ULL << 20U, .threads = 40, .handles = 100});
  CHECK(memory.find("700") != std::string::npos);
  const std::string handles =
      resource_drift(first, {.resident_bytes = 400ULL << 20U, .threads = 40, .handles = 151});
  CHECK(handles.find("151") != std::string::npos);
  CHECK(resource_drift(ProcessResources{}, first).empty());  // rien de connu au premier relevé
  // C33-02 : les fils comptent aussi, le signal d'un fil abandonné.
  const std::string threads =
      resource_drift(first, {.resident_bytes = 400ULL << 20U, .threads = 61, .handles = 100});
  CHECK(threads.find("61 threads") != std::string::npos);
}

}  // namespace ajp::app::play
