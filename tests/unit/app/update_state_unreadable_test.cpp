// C36-03 : un update.toml présent qui ne se lit pas n'est ni pris pour un état neuf, ni réécrit (les clés
// révoquées qu'il garde resteraient sinon perdues).
#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "app/play/updater.h"

using namespace ajp::app::play;
namespace fs = std::filesystem;

TEST_CASE("mises à jour : un update.toml illisible est un refus, rien n'est réécrit (C36-03)", "[updater]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-update-unreadable";
  fs::remove_all(dir);
  const fs::path file = dir / "update.toml";
  fs::create_directories(file);  // présent, illisible
  CHECK_FALSE(read_update_state(file));
  bool changed = false;
  CHECK_FALSE(change_update_state(file, [&](UpdateState& state) {
    changed = true;
    state.last_check_unix = 42;
  }));
  CHECK_FALSE(changed);
  CHECK(fs::is_directory(file));
  // Absent : l'état neuf, comme avant.
  fs::remove_all(file);
  const auto fresh = read_update_state(file);
  REQUIRE(fresh);
  CHECK(fresh->revoked_keys.empty());
  fs::remove_all(dir);
}
