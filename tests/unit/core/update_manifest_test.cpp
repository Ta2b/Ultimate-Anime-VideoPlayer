#include "core/update_manifest.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::core;

namespace {

constexpr const char* kManifest = R"(version = "0.5.0"
channel = "stable"
date = "2026-10-30"
expires = "2027-02-27"
notes_url = "https://github.com/Ta2b/ajp-releases/releases/tag/v0.5.0"
[linux-x64]
url = "https://github.com/Ta2b/ajp-releases/releases/download/v0.5.0/ajp-linux-x64.tar.gz"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
size = 123456
)";

}  // namespace

TEST_CASE("manifeste : lecture", "[update]") {
  const auto manifest = parse_update_manifest(kManifest);
  REQUIRE(manifest);
  CHECK(manifest->version == "0.5.0");
  CHECK(manifest->notes_url == "https://github.com/Ta2b/ajp-releases/releases/tag/v0.5.0");
  CHECK(manifest->revoked_keys.empty());
}

TEST_CASE("manifeste : les sections d'archive, le canal et la date n'empêchent pas la notification (B26-07)",
          "[update]") {
  const auto with = [](std::string_view from, std::string_view to) {
    std::string text = kManifest;
    text.replace(text.find(from), from.size(), to);
    return text;
  };
  // Une archive d'une autre plateforme vide ou en http, un canal inconnu, une date absurde : ignorés.
  CHECK(parse_update_manifest(with("size = 123456", "size = 0")));
  CHECK(parse_update_manifest(
      with("https://github.com/Ta2b/ajp-releases/releases/download", "http://x/releases/download")));
  CHECK(
      parse_update_manifest(with("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "xyz")));
  CHECK(parse_update_manifest(with("stable", "nightly")));
  CHECK(parse_update_manifest(with("date = \"2026-10-30\"", "date = \"hier\"")));
  CHECK(parse_update_manifest(std::string(kManifest) + "[macos-arm64]\nurl = 3\n"));
}

TEST_CASE("manifeste : les clés révoquées se lisent en minuscules", "[update]") {
  const std::string key(64, 'A');
  const auto manifest =
      parse_update_manifest(std::string(kManifest).insert(0, "revoked_keys = [\"" + key + "\"]\n"));
  REQUIRE(manifest);
  CHECK(manifest->revoked_keys == std::vector<std::string>{std::string(64, 'a')});
  CHECK_FALSE(parse_update_manifest("revoked_keys = [\"ab\"]\n" + std::string(kManifest)));
}

TEST_CASE("manifeste : refus nets", "[update]") {
  const auto with = [](std::string_view from, std::string_view to) {
    std::string text = kManifest;
    text.replace(text.find(from), from.size(), to);
    return text;
  };
  CHECK_FALSE(parse_update_manifest("version = ["));
  CHECK_FALSE(parse_update_manifest(with("0.5.0\"\nchannel", "cinq\"\nchannel")));
  CHECK_FALSE(
      parse_update_manifest(with("https://github.com/Ta2b/ajp-releases/releases/tag", "http://x/tag")));
  CHECK_FALSE(parse_update_manifest(with("expires = \"2027-02-27\"\n", "")));
  CHECK_FALSE(parse_update_manifest(with("2027-02-27", "2027-02-30")));
}

TEST_CASE("manifeste : refusé après son expiration", "[update]") {
  using namespace std::chrono;
  const auto manifest = parse_update_manifest(kManifest);
  REQUIRE(manifest);
  CHECK_FALSE(manifest_expired(*manifest, sys_days{2027y / February / 27}));
  CHECK(manifest_expired(*manifest, sys_days{2027y / February / 28}));
}

TEST_CASE("versions : ordre, préversions avant la finale", "[update]") {
  CHECK(compare_versions("0.5.0", "0.4.9") == 1);
  CHECK(compare_versions("0.5.0", "0.5.0") == 0);
  CHECK(compare_versions("0.5.0-preview.2", "0.5.0") == -1);
  CHECK(compare_versions("0.5.0-preview.10", "0.5.0-preview.2") == 1);
  // B26-11 : le nombre final se compare comme un nombre, avec ou sans point.
  CHECK(compare_versions("0.5.0-rc10", "0.5.0-rc2") == 1);
  CHECK(compare_versions("0.5.0-rc.2", "0.5.0-rc.10") == -1);
  CHECK(compare_versions("0.5.0-rc2", "0.5.0-rc2") == 0);
  CHECK(compare_versions("0.5.0-beta1", "0.5.0-rc1") == -1);
  CHECK(compare_versions("1.0.0", "0.99.99") == 1);
  CHECK_FALSE(compare_versions("0.5", "0.5.0"));
  CHECK_FALSE(compare_versions("a.b.c", "0.5.0"));
  // A35-03 : une préversion sur plusieurs lignes ou sans fin n'est pas une version.
  CHECK_FALSE(compare_versions("0.5.0-preview\nVisitez http://x", "0.5.0"));
  CHECK_FALSE(compare_versions("0.5.0-" + std::string(65, 'a'), "0.5.0"));
  CHECK(compare_versions("0.5.0-rc-1.2", "0.5.0"));
  CHECK_FALSE(parse_update_manifest(std::string(kManifest).replace(11, 5, "0.5.0-a b")));
}
