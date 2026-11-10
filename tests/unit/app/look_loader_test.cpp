// A30-07 : la durée de vie de LookLoader (A30-02). Son fil est joint à la destruction, jamais détaché : un
// fil détaché journalisait après close_log et appelait SDL après SDL_Quit. Sous TSan (job « Cœur (tsan) »),
// une course entre le fil et la destruction se voit ici.
#include <chrono>
#include <filesystem>
#include <optional>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

TEST_CASE("LookLoader détruit juste après une demande, puis pendant une rafale (A30-07)", "[app][looks]") {
  const std::filesystem::path luts = std::filesystem::temp_directory_path() / "ajp-look-loader-vide";
  for (int round = 0; round < 20; ++round) {
    LookLoader loader;
    loader.request(PlaySettings{}, luts);
  }
  LookLoader loader;
  for (int i = 0; i < 50; ++i) loader.request(PlaySettings{}, luts);
  // La dernière demande seule rend un résultat, une fois.
  std::optional<LoadedLooks> looks;
  for (int i = 0; i < 400 && !looks; ++i) {
    looks = loader.take();
    if (!looks) std::this_thread::sleep_for(5ms);
  }
  CHECK(looks.has_value());
  CHECK_FALSE(loader.take().has_value());
}

}  // namespace
}  // namespace ajp::app::play
