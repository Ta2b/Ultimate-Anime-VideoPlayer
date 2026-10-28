#include "core/settings.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;
namespace fs = std::filesystem;

namespace {

fs::path scratch(const std::string& name) {
  const fs::path dir = fs::current_path() / "test-tmp" / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

void write(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

Settings full() {
  Settings settings;
  settings.first_run_done = true;
  settings.chosen_engines = {EngineId::Vulkan, EngineId::TensorRt, EngineId::RtxVideoSdk};
  settings.installed_engines = {EngineId::Vulkan, EngineId::TensorRt, EngineId::RtxVideoSdk};
  settings.preferred = {EngineId::TensorRt, EngineId::Vulkan, EngineId::RtxVideoSdk};
  settings.audio_languages = {"ja", "en"};
  settings.subtitle_languages = {"fr", "pt-BR", "en"};
  settings.resume_position = false;
  settings.audio_delay_ms = -120;
  return settings;
}

RefusalCode refusal_of(const std::string& text) {
  auto settings = parse_settings("version = 1\n" + text);
  REQUIRE_FALSE(settings);
  return settings.error().code;
}

RefusalCode invalid(const Settings& settings) {
  auto valid = validate_settings(settings);
  REQUIRE_FALSE(valid);
  return valid.error().code;
}

}  // namespace

TEST_CASE("Réglages : sans settings.toml, les défauts du premier lancement", "[settings]") {
  const fs::path dir = scratch("settings-defaults");
  auto settings = load_settings(dir / "settings.toml");
  REQUIRE(settings);
  CHECK(*settings == Settings{});
  CHECK_FALSE(settings->first_run_done);
  CHECK(settings->installed_engines == std::vector{EngineId::Vulkan});
  CHECK(settings->preferred[static_cast<std::size_t>(Usage::Upscale)] == EngineId::Vulkan);
  CHECK_FALSE(settings->preferred[static_cast<std::size_t>(Usage::TrueHdr)]);
  CHECK(validate_settings(Settings{}));
  CHECK(usage_name(Usage::TrueHdr) == "truehdr");
}

TEST_CASE("Réglages : écrire puis relire rend les mêmes valeurs", "[settings]") {
  auto text = format_settings(full());
  REQUIRE(text);
  auto settings = parse_settings(*text);
  REQUIRE(settings);
  CHECK(*settings == full());

  Settings none = full();
  none.preferred = {};
  auto none_text = format_settings(none);
  REQUIRE(none_text);
  auto reread = parse_settings(*none_text);
  REQUIRE(reread);
  CHECK(*reread == none);
}

TEST_CASE("Réglages : une valeur hors plage ou incohérente est refusée", "[settings]") {
  CHECK(refusal_of("first_run_done = \"yes\"") == RefusalCode::FieldType);
  CHECK(refusal_of("[engines]\nchosen = [\"cuda\"]") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[engines]\ninstalled = \"vulkan\"") == RefusalCode::FieldType);
  CHECK(refusal_of("[engines]\npreferred = { upscale = \"tensorrt\" }") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[languages]\naudio = [\"english\"]") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[languages]\naudio = [1]") == RefusalCode::FieldType);
  CHECK(refusal_of("[playback]\naudio_delay_ms = 20000") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[playback]\naudio_delay_ms = 1.5") == RefusalCode::FieldType);
  CHECK(refusal_of("playback = 3") == RefusalCode::FieldType);

  auto missing = parse_settings("first_run_done = true\n");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::FieldMissing);
  auto newer = parse_settings("version = 2\n");
  REQUIRE_FALSE(newer);
  CHECK(newer.error().code == RefusalCode::FormatVersion);
  auto broken = parse_settings("version = \n");
  REQUIRE_FALSE(broken);
  CHECK(broken.error().code == RefusalCode::FileUnreadable);

  Settings settings = full();
  settings.installed_engines.push_back(EngineId::Vulkan);
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
  // Vulkan est livré avec le lecteur : il ne manque ni aux moteurs choisis ni aux installés.
  for (const char* engines : {"chosen = [\"tensorrt\"]", "installed = []\npreferred = {}"}) {
    auto without_vulkan = parse_settings(std::format("version = 1\n[engines]\n{}\n", engines));
    INFO(engines);
    REQUIRE_FALSE(without_vulkan);
    CHECK(without_vulkan.error().code == RefusalCode::FieldOutOfRange);
    CHECK(without_vulkan.error().detail.find("vulkan") != std::string::npos);
  }
  settings = full();
  settings.preferred[static_cast<std::size_t>(Usage::TrueHdr)] = EngineId::TensorRt;
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
  settings = full();
  settings.preferred[static_cast<std::size_t>(Usage::Upscale)] = EngineId::RtxVideoSdk;
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
  settings = full();
  settings.audio_languages = {"ja", "ja"};
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
  settings = full();
  settings.subtitle_languages = std::vector<std::string>(9, "fr");
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
  for (const char* tag : {"zh-Hant", "pt-BR", "jpn", "es-419"}) {
    settings = full();
    settings.audio_languages = {tag};
    CHECK(validate_settings(settings));
  }
  for (const char* tag : {"", "e", "EN", "en-", "en_US", "english", "en-x"}) {
    settings = full();
    settings.audio_languages = {tag};
    INFO(tag);
    CHECK_FALSE(validate_settings(settings));
  }
}

TEST_CASE("Réglages : l'écriture est atomique et garde les clés inconnues", "[settings]") {
  const fs::path dir = scratch("settings-save");
  const fs::path path = dir / "settings.toml";
  write(path, "version = 1\nfuture_option = 42\n[ui]\ntheme = \"sombre\"\n[playback]\nspeed = 1.25\n");

  REQUIRE(save_settings(path, full()));
  const std::string text = read(path);
  CHECK(text.find("future_option = 42") != std::string::npos);
  CHECK(text.find("theme = \"sombre\"") != std::string::npos);
  CHECK(text.find("speed = 1.25") != std::string::npos);
  auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(*reread == full());

  // Rien d'autre que settings.toml dans le dossier : le fichier temporaire a été renommé.
  CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 1);
}

TEST_CASE("Réglages : un refus laisse le fichier en place", "[settings]") {
  const fs::path dir = scratch("settings-refused");
  const fs::path path = dir / "settings.toml";
  write(path, "version = 1\nfirst_run_done = true\n");

  Settings wrong = full();
  wrong.audio_delay_ms = 99'999;
  auto refused = save_settings(path, wrong);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(read(path) == "version = 1\nfirst_run_done = true\n");

  write(path, "version = [\n");
  auto corrupt = save_settings(path, full());
  REQUIRE_FALSE(corrupt);
  CHECK(corrupt.error().code == RefusalCode::FileUnreadable);
  CHECK(read(path) == "version = [\n");
  auto load = load_settings(path);
  REQUIRE_FALSE(load);
  CHECK(load.error().code == RefusalCode::FileUnreadable);

  // Un dossier à la place du fichier : refusé, jamais lu comme un fichier vide ni remplacé par les défauts.
  const fs::path folder = dir / "dossier.toml";
  fs::create_directories(folder);
  auto from_folder = load_settings(folder);
  REQUIRE_FALSE(from_folder);
  CHECK(from_folder.error().code == RefusalCode::FileUnreadable);
  CHECK_FALSE(save_settings(folder, full()));
  CHECK(fs::is_directory(folder));
}

TEST_CASE("Réglages : le dossier de données se crée à la première écriture", "[settings]") {
  const fs::path dir = scratch("settings-mkdir");
  const fs::path path = dir / "sous" / "dossier" / "settings.toml";
  REQUIRE(save_settings(path, Settings{}));
  auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(*reread == Settings{});
}

TEST_CASE("Réglages : assistant et inscription font l'aller-retour", "[settings]") {
  Settings settings;
  settings.first_run_done = true;
  settings.first_run_step = 5;
  settings.open_with_registered = true;
  const auto text = format_settings(settings);
  REQUIRE(text);
  const auto read = parse_settings(*text);
  REQUIRE(read);
  CHECK(*read == settings);
}
