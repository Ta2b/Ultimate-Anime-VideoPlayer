#include <format>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "vendor_hdr/sdk_manifest.h"

using namespace ajp::vendor_hdr;

namespace {

Manifest sample() {
  return Manifest{.kind = SdkKind::MaxineVfx,
                  .version = SdkVersion{.parts = {1, 3, 0, 0}},
                  .copy = "versions/1.3.0.0-18f2a",
                  .files = {{.path = "lib/libVideoFX.so", .bytes = 1234},
                            {.path = "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so", .bytes = 99}}};
}

std::string manifest_with(std::string_view copy, std::string_view kind = "maxine-vfx", int format = 1) {
  return std::format("format = {}\nkind = \"{}\"\nversion = \"1.3.0.0\"\ncopy = \"{}\"\n", format, kind,
                     copy);
}

}  // namespace

TEST_CASE("manifeste : aller-retour", "[manifest]") {
  const Manifest written = sample();
  const std::string text = format_manifest(written);
  CHECK(text.contains("format = 1"));
  CHECK((text.contains("kind = 'maxine-vfx'") || text.contains("kind = \"maxine-vfx\"")));
  const auto read = parse_manifest(text, SdkKind::MaxineVfx);
  REQUIRE(read);
  CHECK(*read == written);
}

TEST_CASE("manifeste : sans fichiers", "[manifest]") {
  const auto read = parse_manifest(manifest_with("versions/1.3.0.0-1"), SdkKind::MaxineVfx);
  REQUIRE(read);
  CHECK(read->files.empty());
  CHECK(read->version.text() == "1.3.0.0");
}

TEST_CASE("manifeste : refus", "[manifest]") {
  CHECK_FALSE(parse_manifest("ceci n'est pas du TOML = = =", SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("versions/a", "rtx-video"), SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("versions/../../etc"), SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("/tmp/versions/a"), SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("ailleurs/a"), SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("versions/a", "maxine-vfx", 0), SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest("format = 1\nkind = \"maxine-vfx\"\nversion = \"un\"\ncopy = \"versions/a\"\n",
                             SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("versions/a") + "[[file]]\npath = \"../x\"\nbytes = 1\n",
                             SdkKind::MaxineVfx));
  CHECK_FALSE(parse_manifest(manifest_with("versions/a") + "[[file]]\npath = \"lib/x\"\nbytes = -1\n",
                             SdkKind::MaxineVfx));
  CHECK(parse_manifest(manifest_with("versions/a"), SdkKind::MaxineVfx));  // le témoin passe
}

TEST_CASE("manifeste : un format plus récent est signalé à part", "[manifest]") {
  const auto newer = parse_manifest(manifest_with("versions/a", "maxine-vfx", 2), SdkKind::MaxineVfx);
  REQUIRE_FALSE(newer);
  CHECK(newer.error().newer_format);
  CHECK(newer.error().reason.contains("lecteur plus récent"));

  const auto broken = parse_manifest(manifest_with("../a"), SdkKind::MaxineVfx);
  REQUIRE_FALSE(broken);
  CHECK_FALSE(broken.error().newer_format);
}
