#include "core/update_manifest.h"

#include <catch2/catch_test_macros.hpp>

using namespace ajp::core;

namespace {

constexpr const char* kManifest = R"(version = "0.5.0"
channel = "stable"
date = "2026-10-30"
notes_url = "https://github.com/Ta2b/ajp-releases/releases/tag/v0.5.0"
[linux-x64]
url = "https://github.com/Ta2b/ajp-releases/releases/download/v0.5.0/ajp-linux-x64.tar.gz"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
size = 123456
)";

}  // namespace

TEST_CASE("manifeste : lecture et archive du système", "[update]") {
  const auto manifest = parse_update_manifest(kManifest);
  REQUIRE(manifest);
  CHECK(manifest->version == "0.5.0");
  CHECK(manifest->channel == "stable");
  REQUIRE(artifact_for(*manifest, Os::Linux));
  CHECK(artifact_for(*manifest, Os::Linux)->size == 123456);
  CHECK(artifact_for(*manifest, Os::Windows) == nullptr);
}

TEST_CASE("manifeste : refus nets", "[update]") {
  const auto with = [](std::string_view from, std::string_view to) {
    std::string text = kManifest;
    text.replace(text.find(from), from.size(), to);
    return text;
  };
  CHECK_FALSE(parse_update_manifest("version = ["));
  CHECK_FALSE(parse_update_manifest(with("0.5.0\"\nchannel", "cinq\"\nchannel")));
  CHECK_FALSE(parse_update_manifest(with("stable", "nightly")));
  CHECK_FALSE(parse_update_manifest(with("https://github.com/Ta2b/ajp-releases/releases/download", "http://x/releases/download")));
  CHECK_FALSE(parse_update_manifest(with("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "xyz")));
  CHECK_FALSE(parse_update_manifest(with("size = 123456", "size = 0")));
}

TEST_CASE("versions : ordre, préversions avant la finale", "[update]") {
  CHECK(compare_versions("0.5.0", "0.4.9") == 1);
  CHECK(compare_versions("0.5.0", "0.5.0") == 0);
  CHECK(compare_versions("0.5.0-preview.2", "0.5.0") == -1);
  CHECK(compare_versions("0.5.0-preview.10", "0.5.0-preview.2") == 1);
  CHECK(compare_versions("1.0.0", "0.99.99") == 1);
  CHECK_FALSE(compare_versions("0.5", "0.5.0"));
  CHECK_FALSE(compare_versions("a.b.c", "0.5.0"));
}
