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
  const fs::path dir = fs::temp_directory_path() / "ajp-test-tmp" / name;
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
  settings.chosen_engines = {EngineId::Vulkan, EngineId::TensorRt, EngineId::Metal};
  settings.installed_engines = {EngineId::Vulkan, EngineId::TensorRt, EngineId::Metal};
  settings.preferred = {EngineId::TensorRt, EngineId::Vulkan};
  settings.resume_position = false;
  return settings;
}

// Le fichier lu malgré la valeur fausse : les clés écartées, une seule ici.
std::vector<std::string> skipped_of(const std::string& text) {
  std::vector<std::string> skipped;
  auto settings = parse_settings("version = 1\nfirst_run_step = 3\n" + text, &skipped);
  REQUIRE(settings);
  CHECK(settings->first_run_step == 3);  // le reste du fichier vaut
  return skipped;
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
  CHECK(validate_settings(Settings{}));
  CHECK(usage_name(Usage::Interpolate) == "interpolate");
}

TEST_CASE("Réglages : les SDK de RTX HDR d'un ancien settings.toml sont oubliés", "[settings]") {
  auto settings = parse_settings(
      "version = 1\n[engines]\nchosen = [\"vulkan\", \"tensorrt\", \"maxine_vfx\"]\n"
      "installed = [\"vulkan\", \"tensorrt\", \"rtx_video_sdk\", \"maxine_vfx\"]\n"
      "preferred = { upscale = \"tensorrt\", interpolate = \"tensorrt\", truehdr = \"maxine_vfx\" }\n");
  REQUIRE(settings);
  CHECK(settings->chosen_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});
  CHECK(settings->installed_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});
  auto text = format_settings(*settings, "version = 1\n[engines.preferred]\ntruehdr = \"maxine_vfx\"\n");
  REQUIRE(text);
  CHECK(text->find("truehdr") == std::string::npos);
}

TEST_CASE("Réglages : MIGraphX, OpenVINO et DirectML d'un ancien settings.toml sont oubliés sans bruit",
          "[settings]") {
  std::vector<std::string> skipped;
  auto settings = parse_settings(
      "version = 1\n[engines]\nchosen = [\"vulkan\", \"openvino\", \"migraphx\"]\n"
      "installed = [\"vulkan\", \"tensorrt\", \"openvino\", \"directml\"]\n"
      "preferred = { upscale = \"openvino\", interpolate = \"tensorrt\" }\n",
      &skipped);
  REQUIRE(settings);
  CHECK(settings->chosen_engines == std::vector{EngineId::Vulkan});
  CHECK(settings->installed_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});
  CHECK_FALSE(settings->preferred[static_cast<std::size_t>(Usage::Upscale)].has_value());
  CHECK(settings->preferred[static_cast<std::size_t>(Usage::Interpolate)] == EngineId::TensorRt);
  CHECK(skipped.empty());
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

TEST_CASE("Réglages : une valeur fausse est écartée seule, le fichier vaut (P2.21)", "[settings]") {
  for (const char* text :
       {"first_run_done = \"yes\"", "[engines]\nchosen = [\"cuda\"]", "[engines]\ninstalled = \"vulkan\"",
        "[engines]\npreferred = { upscale = \"tensorrt\" }", "playback = 3"}) {
    INFO(text);
    CHECK(skipped_of(text).size() == 1);
  }
  // Vulkan, livré avec le lecteur, revient de lui-même dans une liste qui l'oublie.
  auto without_vulkan = parse_settings("version = 1\n[engines]\nchosen = [\"tensorrt\"]\n");
  REQUIRE(without_vulkan);
  CHECK(without_vulkan->chosen_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});

  auto missing = parse_settings("first_run_done = true\n");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::FieldMissing);
  auto older = parse_settings("version = 0\n");
  REQUIRE_FALSE(older);
  CHECK(older.error().code == RefusalCode::FormatVersion);
  auto broken = parse_settings("version = \n");
  REQUIRE_FALSE(broken);
  CHECK(broken.error().code == RefusalCode::FileUnreadable);

  Settings settings = full();
  settings.installed_engines.push_back(EngineId::Vulkan);
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
  settings = full();
  settings.installed_engines = {EngineId::Vulkan, EngineId::TensorRt};
  settings.preferred[static_cast<std::size_t>(Usage::Upscale)] = EngineId::Metal;
  CHECK(invalid(settings) == RefusalCode::FieldOutOfRange);
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
  wrong.installed_engines.push_back(EngineId::Vulkan);  // un doublon
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

TEST_CASE("Réglages : une version plus récente se lit, sans être abaissée (A22-07)", "[settings]") {
  std::vector<std::string> skipped;
  auto newer =
      parse_settings("version = 7\nfirst_run_done = true\n[engines]\nchosen = [\"tensorrt\"]\n", &skipped);
  REQUIRE(newer);
  CHECK(newer->first_run_done);
  CHECK(newer->chosen_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});
  REQUIRE(skipped.size() == 1);
  CHECK(skipped[0].find("version = 7") != std::string::npos);

  // Réécrit par ce lecteur : la version et les clés inconnues restent.
  auto text = format_settings(*newer, "version = 7\nfuture_option = 42\n");
  REQUIRE(text);
  CHECK(text->find("version = 7") != std::string::npos);
  CHECK(text->find("future_option = 42") != std::string::npos);
  auto fresh = format_settings(*newer, "");
  REQUIRE(fresh);
  CHECK(fresh->find("version = 1") != std::string::npos);
}

TEST_CASE("Réglages : un ancien settings.toml avec langues et délai se relit (A22-08)", "[settings]") {
  const std::string old =
      "version = 1\nfirst_run_done = true\n[languages]\naudio = [\"ja\", \"en\"]\nsubtitles = [\"fr\"]\n"
      "[playback]\nresume_position = false\naudio_delay_ms = -120\n";
  std::vector<std::string> skipped;
  auto settings = parse_settings(old, &skipped);
  REQUIRE(settings);
  CHECK(settings->first_run_done);
  CHECK_FALSE(settings->resume_position);
  CHECK(skipped.empty());
  // La réécriture garde les clés que ce lecteur ne connaît plus.
  auto text = format_settings(*settings, old);
  REQUIRE(text);
  CHECK(text->find("audio_delay_ms = -120") != std::string::npos);
  CHECK(text->find("subtitles") != std::string::npos);
}

TEST_CASE("Réglages : un moteur d'une version plus récente reste à la réécriture (B22-07)", "[settings]") {
  const std::string newer =
      "version = 2\n[engines]\nchosen = [\"vulkan\", \"nouveau\"]\ninstalled = [\"vulkan\", \"nouveau\"]\n"
      "[engines.preferred]\nupscale = \"nouveau\"\ninterpolate = \"vulkan\"\n";
  std::vector<std::string> skipped;
  auto settings = parse_settings(newer, &skipped);
  REQUIRE(settings);
  CHECK_FALSE(skipped.empty());  // le moteur inconnu est écarté à la lecture
  CHECK_FALSE(settings->preferred[0]);
  auto text = format_settings(*settings, newer);
  REQUIRE(text);
  CHECK(text->find("nouveau") != std::string::npos);
  // La préférence inconnue reste ; la connue se réécrit comme avant.
  CHECK(text->find("upscale = \"nouveau\"") != std::string::npos);
  CHECK(text->find("interpolate = \"vulkan\"") != std::string::npos);
  // « nouveau » reste dans les deux listes et dans la préférence : trois fois.
  std::size_t kept = 0;
  for (auto at = text->find("nouveau"); at != std::string::npos; at = text->find("nouveau", at + 1)) ++kept;
  CHECK(kept == 3);
  // Une préférence que l'utilisateur retire (moteur connu) s'efface.
  Settings cleared = *settings;
  cleared.preferred[1].reset();
  auto without = format_settings(cleared, *text);
  REQUIRE(without);
  CHECK(without->find("interpolate") == std::string::npos);
  CHECK(without->find("nouveau") != std::string::npos);
}
