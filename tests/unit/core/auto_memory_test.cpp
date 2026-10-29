#include "core/auto_memory.h"

#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::core;

TEST_CASE("mémoire auto : aller-retour du fichier", "[core][auto_memory]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp_auto_memory_test";
  std::filesystem::create_directories(dir);
  const auto file = dir / "auto_memory.toml";
  std::filesystem::remove(file);
  const std::string key = auto_memory_key("/v/ep18.mkv", 123456, "rife-4.25", 720, "RTX 4070 Ti");
  CHECK_FALSE(load_auto_step(file, key));  // absent : vide
  REQUIRE(store_auto_step(file, key, 250));
  REQUIRE(store_auto_step(file, "autre", 400));
  CHECK(load_auto_step(file, key) == 250);
  CHECK(load_auto_step(file, "autre") == 400);
  CHECK_FALSE(load_auto_step(file, "inconnue"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("mémoire auto : LRU borné", "[core][auto_memory]") {
  AutoMemory memory;
  for (std::size_t i = 0; i < kAutoMemoryEntries + 50; ++i) memory.put("k" + std::to_string(i), 200);
  CHECK(memory.size() == kAutoMemoryEntries);
  CHECK_FALSE(memory.find("k0"));
  CHECK(memory.find("k249"));
  memory.put("k50", 300);  // la plus récente : survit à la prochaine éviction
  memory.put("neuve", 200);
  CHECK(memory.find("k50") == 300);
  CHECK_FALSE(memory.find("k51"));
  const AutoMemory again = AutoMemory::parse(memory.dump());
  CHECK(again.size() == memory.size());
  CHECK(again.find("k50") == 300);
}

TEST_CASE("mémoire auto : fichier corrompu toléré", "[core][auto_memory]") {
  const AutoMemory memory = AutoMemory::parse(
      "n'importe quoi\n[\"bon\"]\nstep = 250\n[\"sans valeur\"]\n[\"mauvais\"]\nstep = abc\n[\"hors\"]\nstep "
      "= 99999\n"
      "[\"\"]\nstep = 200\n\x01\x02[\"x\"\nstep = 3\n[\"fin\"]\r\nstep = 300\r\n");
  CHECK(memory.find("bon") == 250);
  CHECK(memory.find("fin") == 300);
  CHECK_FALSE(memory.find("mauvais"));
  CHECK_FALSE(memory.find("hors"));
  CHECK(memory.size() == 2);
  CHECK(AutoMemory::parse("").size() == 0);
}

TEST_CASE("mémoire auto : la clé ne porte ni guillemet ni saut de ligne", "[core][auto_memory]") {
  const std::string key = auto_memory_key("/v/a\"b\nc.mkv", 1, "m", 720, "gpu");
  CHECK(key.find('"') == std::string::npos);
  CHECK(key.find('\n') == std::string::npos);
}
