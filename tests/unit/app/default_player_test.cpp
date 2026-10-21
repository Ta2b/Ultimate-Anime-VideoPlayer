#include "app/play/default_player.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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
  // Entre guillemets, « " », « ` », « $ » et « \ » prennent une barre oblique inverse et « % » se
  // double ; la règle des chaînes double ensuite chaque barre oblique inverse.
  const auto entry = desktop_entry("/home/t/Mes vidéos/a\"b`c$d\\e%f/ajp");
  REQUIRE(entry);
  CHECK(line_of(*entry, "Exec") == R"("/home/t/Mes vidéos/a\\"b\\`c\\$d\\\\e%%f/ajp" %f)");
  CHECK(line_of(*entry, "TryExec") == R"(/home/t/Mes vidéos/a"b`c$d\\e%f/ajp)");
}

TEST_CASE("Lecteur par défaut : un chemin qu'une ligne ne porte pas est refusé", "[app][default_player]") {
  for (const std::string_view path : {std::string_view("/opt/a\nb/ajp"), std::string_view("/opt/a\tb/ajp"),
                                      std::string_view("/opt/\xC3/ajp"), std::string_view()}) {
    const auto entry = desktop_entry(fs::path(std::string(path)));
    REQUIRE_FALSE(entry);
    CHECK(entry.error().code == RefusalCode::FieldOutOfRange);
  }
}

TEST_CASE("Lecteur par défaut : les clés du registre de l'utilisateur sous Windows",
          "[app][default_player]") {
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
  const fs::path home = fs::temp_directory_path() / "ajp-default-player-test";
  std::error_code error;
  fs::remove_all(home, error);
  const PlayerPlace place{.executable = "/opt/ajp-linux-x64/ajp", .home = home};

  const auto registered = register_player(place);
  REQUIRE(registered);
  CHECK(registered->contains("xdg-mime default io.github.ta2b.ajp.desktop"));
  const fs::path file = desktop_file(home);
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
#endif

}  // namespace
}  // namespace ajp::app::play
