#include "app/play/default_player.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/data_dirs.h"
#include "app/play/episodes.h"
#include "app/play/player.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

std::string line_of(const std::string& text, std::string_view key) {
  const std::size_t at = text.find(std::string("\n") + std::string(key) + "=");
  if (at == std::string::npos) return {};
  const std::size_t start = at + 1 + key.size() + 1;
  return text.substr(start, text.find('\n', start) - start);
}

bool has(const std::vector<RegistryValue>& values, std::string_view key, std::string_view name,
         std::string_view data) {
  return std::ranges::find(values, RegistryValue{std::string(key), std::string(name), std::string(data)}) !=
         values.end();
}

TEST_CASE("Lecteur par défaut : le .desktop nomme le lecteur et les types vidéo", "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  const auto entry = desktop_entry("/opt/ajp-linux-x64/ajp");
  REQUIRE(entry);
  CHECK(entry->starts_with("[Desktop Entry]\n"));
  CHECK(line_of(*entry, "Type") == "Application");
  CHECK(line_of(*entry, "Name") == kAppName);
  CHECK(line_of(*entry, "Exec") == "\"/opt/ajp-linux-x64/ajp\" %f");
  CHECK(line_of(*entry, "TryExec") == "/opt/ajp-linux-x64/ajp");
  CHECK(line_of(*entry, "Terminal") == "false");
  const std::string mimes = line_of(*entry, "MimeType");
  for (const std::string_view mime : video_mime_types()) CHECK(mimes.contains(std::string(mime) + ";"));
  CHECK(mimes.contains("video/x-matroska;"));
}

TEST_CASE("Lecteur par défaut : Exec échappe le chemin selon la spécification XDG", "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  // Entre guillemets, « " », « ` », « $ » et « \ » prennent une barre oblique inverse et « % » se
  // double ; la règle des chaînes double ensuite chaque barre oblique inverse.
  const auto entry = desktop_entry("/home/t/Mes vidéos/a\"b`c$d\\e%f/ajp");
  REQUIRE(entry);
  CHECK(line_of(*entry, "Exec") == R"("/home/t/Mes vidéos/a\\"b\\`c\\$d\\\\e%%f/ajp" %f)");
  CHECK(line_of(*entry, "TryExec") == R"(/home/t/Mes vidéos/a"b`c$d\\e%f/ajp)");
}

TEST_CASE("Lecteur par défaut : un chemin qu'une ligne ne porte pas est refusé", "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  std::vector<std::string_view> paths{"/opt/a\nb/ajp", "/opt/a\tb/ajp", ""};
#ifndef _WIN32
  // Un chemin POSIX est fait d'octets, pas forcément d'UTF-8 ; sous Windows il est en UTF-16, et
  // fs::path refuse de lire cet octet seul.
  paths.emplace_back("/opt/\xC3/ajp");
#endif
  for (const std::string_view path : paths) {
    const auto entry = desktop_entry(fs::path(std::string(path)));
    REQUIRE_FALSE(entry);
    CHECK(entry.error().code == RefusalCode::FieldOutOfRange);
  }
}

TEST_CASE("Lecteur par défaut : les clés du registre de l'utilisateur sous Windows",
          "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  constexpr std::string_view exe = R"(C:\Program Files\AJP\ajp.exe)";
  const std::vector<RegistryValue> values = registry_values(exe);
  const std::string command = R"("C:\Program Files\AJP\ajp.exe" "%1")";
  CHECK(has(values, R"(Software\Classes\AJP.Video\shell\open\command)", "", command));
  CHECK(has(values, R"(Software\Classes\AJP.Video\DefaultIcon)", "", R"("C:\Program Files\AJP\ajp.exe",0)"));
  CHECK(has(values, R"(Software\Classes\Applications\ajp.exe\shell\open\command)", "", command));
  CHECK(has(values, R"(Software\AJP\Capabilities)", "ApplicationName", kAppName));
  CHECK(has(values, R"(Software\RegisteredApplications)", "AJP", R"(Software\AJP\Capabilities)"));
  for (const std::string_view extension : video_extensions()) {
    INFO(extension);
    CHECK(has(values, std::string(R"(Software\AJP\Capabilities\FileAssociations)"), extension, "AJP.Video"));
    CHECK(has(values, std::string(R"(Software\Classes\)") + std::string(extension) + R"(\OpenWithProgids)",
              "AJP.Video", ""));
    CHECK(has(values, R"(Software\Classes\Applications\ajp.exe\SupportedTypes)", extension, ""));
  }
}

TEST_CASE("Lecteur par défaut : le retrait couvre tout ce que l'inscription écrit", "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  constexpr std::string_view exe = R"(D:\ajp-windows-x64\ajp.exe)";
  const std::vector<std::string> trees = registry_trees(exe);
  const std::vector<RegistryValue> shared = registry_shared_values();
  for (const RegistryValue& value : registry_values(exe)) {
    INFO(value.key << " : " << value.name);
    const bool in_tree = std::ranges::any_of(trees, [&](const std::string& tree) {
      return value.key == tree || value.key.starts_with(tree + "\\");
    });
    const bool shared_value = std::ranges::any_of(
        shared, [&](const RegistryValue& s) { return s.key == value.key && s.name == value.name; });
    CHECK(in_tree != shared_value);  // l'un ou l'autre, jamais les deux
  }
}

#if !defined(_WIN32) && !defined(__APPLE__)
TEST_CASE("Lecteur par défaut : inscrit puis retiré dans le dossier de l'utilisateur",
          "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  const fs::path home = fs::temp_directory_path() / "ajp-default-player-test";
  std::error_code error;
  fs::remove_all(home, error);
  const PlayerPlace place{.executable = "/opt/ajp-linux-x64/ajp", .home = home};

  const auto registered = register_player(place);
  REQUIRE(registered);
  CHECK(registered->contains("xdg-mime default io.github.ta2b.ajp.desktop"));
  const fs::path file = desktop_file(place);
  CHECK(file == home / ".local/share/applications/io.github.ta2b.ajp.desktop");
  std::ifstream in(file, std::ios::binary);
  const std::string written{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  CHECK(written == desktop_entry(place.executable).value());
  CHECK_FALSE(fs::exists(fs::path(file.string() + ".tmp")));

  CHECK(register_player(place));  // une seconde inscription remplace la première
  const auto removed = unregister_player(place);
  REQUIRE(removed);
  CHECK_FALSE(fs::exists(file));
  const auto again = unregister_player(place);
  REQUIRE(again);
  CHECK(again->contains("rien à retirer"));
  fs::remove_all(home, error);

  const auto nowhere = register_player(PlayerPlace{.executable = place.executable});
  REQUIRE_FALSE(nowhere);
  CHECK(nowhere.error().code == RefusalCode::FieldMissing);
}

TEST_CASE("Lecteur par défaut : dans les données de l'utilisateur ($XDG_DATA_HOME)",
          "[app][default_player]") {
  ajp::app::play::FrenchUi french;
  const fs::path root = fs::temp_directory_path() / "ajp-default-player-data-test";
  std::error_code error;
  fs::remove_all(root, error);
  // SDL_GetPrefPath("ajp", "ajp") : <données>/ajp/ajp/, barre finale comprise.
  CHECK(user_data(root / "donnees" / "ajp" / "ajp" / "") == root / "donnees");
  CHECK(user_data(root / "donnees" / "ajp" / "ajp") == root / "donnees");
  CHECK(user_data({}).empty());
  const PlayerPlace place{.executable = "/opt/ajp-linux-x64/ajp",
                          .home = root / "maison",
                          .data = user_data(root / "donnees/ajp/ajp/")};
  const fs::path file = desktop_file(place);
  CHECK(file == root / "donnees/applications/io.github.ta2b.ajp.desktop");
  const auto registered = register_player(place);
  REQUIRE(registered);
  CHECK(registered->contains(file.string()));
  CHECK(fs::exists(file));
  CHECK_FALSE(fs::exists(place.home / ".local"));
  REQUIRE(unregister_player(place));
  CHECK_FALSE(fs::exists(file));
  fs::remove_all(root, error);
}
#endif

}  // namespace
}  // namespace ajp::app::play

TEST_CASE("xdg-mime query default : reconnaît le .desktop d'AJP", "[default_player]") {
  ajp::app::play::FrenchUi french;
  using namespace ajp::app::play;
  CHECK(default_handler_is_ajp("io.github.ta2b.ajp.desktop\n"));
  CHECK(default_handler_is_ajp("io.github.ta2b.ajp.desktop"));
  CHECK_FALSE(default_handler_is_ajp("mpv.desktop\n"));
  CHECK_FALSE(default_handler_is_ajp(""));
}

TEST_CASE("Lecteur par défaut : la mise à jour et l'inscription ne lisent pas le même dossier",
          "[app][default_player]") {
  // Le dossier du lecteur garde update.toml (vérification, préparation et application au lancement) ;
  // l'inscription part de son grand-parent. Avant, la vérification écrivait dans le grand-parent.
  const ajp::app::play::PlayerPaths paths{
      .home = "/maison", .data = "/donnees/ajp/ajp", .executable = "/opt/ajp/ajp"};
  using namespace ajp::app::play;
  const PlayerPlace place = ajp::app::play::place_of(paths);
  CHECK(place.app_data == std::filesystem::path("/donnees/ajp/ajp"));
  CHECK(place.data == std::filesystem::path("/donnees"));
  CHECK(place.executable == std::filesystem::path("/opt/ajp/ajp"));
}

TEST_CASE("Les moteurs et caches d'une version d'avant passent dans le dossier local, les réglages restent",
          "[app]") {
  namespace fs = std::filesystem;
  using ajp::app::play::move_heavy_data;
  const fs::path root = fs::temp_directory_path() / "ajp-local-data";
  fs::remove_all(root);
  const fs::path roaming = root / "Roaming" / "ajp" / "ajp";
  const fs::path local = root / "Local" / "ajp" / "ajp";
  fs::create_directories(roaming / "engines");
  fs::create_directories(roaming / "backends" / "tensorrt");
  std::ofstream(roaming / "engines" / "span.plan") << "plan";
  std::ofstream(roaming / "play.toml") << "[playlist]";
  fs::create_directories(local / "backends");  // déjà là : gardé tel quel, l'ancien n'écrase rien
  std::vector<std::string> said;
  CHECK(move_heavy_data(roaming, local, [&](const std::string& line) { said.push_back(line); }) == 1);
  CHECK(fs::is_regular_file(local / "engines" / "span.plan"));
  CHECK_FALSE(fs::exists(roaming / "engines"));
  CHECK(fs::exists(roaming / "backends" / "tensorrt"));
  CHECK(fs::is_regular_file(roaming / "play.toml"));
  CHECK(said.size() == 1);
  CHECK(move_heavy_data(roaming, roaming) == 0);
#ifndef _WIN32
  CHECK(ajp::app::play::local_data(roaming) == roaming);  // hors Windows, le même dossier
#endif
  fs::remove_all(root);
}

TEST_CASE("Les fichiers d'une version d'avant partent du dossier partagé, ceux des autres restent (P3.70)",
          "[app]") {
  namespace fs = std::filesystem;
  const fs::path shared = fs::temp_directory_path() / "ajp-purge" / "AppData";
  fs::remove_all(shared.parent_path());
  fs::create_directories(shared / "plans");
  fs::create_directories(shared / "updates" / "1.2.0");
  std::ofstream(shared / "update.toml") << "last_check_unix = 12\n";
  std::ofstream(shared / "shaders.cache") << "pl_cache....";
  std::ofstream(shared / "plans" / "a.shots") << "x";
  std::ofstream(shared / "engines-relaunched") << "";
  CHECK(ajp::app::play::purge_misplaced(shared) == 6);
  CHECK(fs::is_empty(shared));
  // Ce que le lecteur ne reconnaît pas reste : un update.toml étranger, un cache d'un autre, un plans/
  // habité.
  fs::create_directories(shared / "plans");
  fs::create_directories(shared / "updates" / "Other");
  std::ofstream(shared / "update.toml") << "channel = 1\n";
  std::ofstream(shared / "shaders.cache") << "other";
  std::ofstream(shared / "plans" / "notes.txt") << "x";
  CHECK(ajp::app::play::purge_misplaced(shared) == 0);
  CHECK(fs::exists(shared / "update.toml"));
  CHECK(fs::exists(shared / "plans" / "notes.txt"));
  CHECK(fs::exists(shared / "updates" / "Other"));
  fs::remove_all(shared.parent_path());
}

TEST_CASE("un fichier encore en téléchargement n'est pas fini (P3.27)", "[episodes]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-growing";
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / "ep01.mkv";
  {
    std::ofstream(file, std::ios::binary) << "1234";
  }
  const ajp::app::play::GrowingFile opened = ajp::app::play::GrowingFile::of(file);
  CHECK_FALSE(opened.grew(file, std::chrono::milliseconds(200)));  // écrit à l'instant, mais n'a pas grandi
  {
    std::ofstream(file, std::ios::binary | std::ios::app) << "5678";
  }
  CHECK(opened.grew(file));
  std::filesystem::remove_all(dir);
}
