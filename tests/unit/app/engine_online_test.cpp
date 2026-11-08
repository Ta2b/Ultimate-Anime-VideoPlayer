// Les moteurs joints à la release après elle (engines-nvidia.lock) : pris pour cet OS et sous la release de
// cette version seulement, un moteur déjà livré remplacé avec ses fichiers NVIDIA, le reste écarté.
#include "app/play/engine_online.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/model_index.h"

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

constexpr std::string_view kPrefix = "https://github.com/o/r/releases/download/v1.2.3/";

std::string engine(std::string_view platform, std::string_view url, std::string_view sha) {
  return std::format(
      "\n[[engine]]\nid = \"tensorrt\"\nplatform = \"{}\"\nurl = \"{}\"\nsha256 = \"{}\"\n"
      "size = 10\nversion = \"11\"\n",
      platform, url, sha);
}

std::string vendor(std::string_view platform, std::string_view url, std::string_view name) {
  return std::format(
      "\n[[vendor_file]]\nid = \"tensorrt\"\nplatform = \"{}\"\nurl = \"{}\"\nname = \"{}\"\n"
      "data_offset = 1\npacked_size = 2\nsize = 3\nmethod = \"deflate\"\nsha256 = \"{}\"\n",
      platform, url, name, std::string(64, 'c'));
}

std::string_view os_name() {
  return core::kHostOs == core::Os::Windows ? "windows"
         : core::kHostOs == core::Os::Linux ? "linux"
                                            : "macos";
}

EngineCatalog empty_catalog() {
  return EngineCatalog{.layout = core::install_layout(host_os(), fs::temp_directory_path() / "ajp-online")};
}

}  // namespace

TEST_CASE("Moteurs de la release : un paquet joint après coup rejoint le catalogue", "[app][install]") {
  const std::string url = std::string(kPrefix) + "ajp-engine-tensorrt.tar.gz";
  const std::string text = "version = 1\n" + engine(os_name(), url, std::string(64, 'a')) +
                           vendor(os_name(), "https://pypi.nvidia.com/x.whl", "nvinfer.dll");
  EngineCatalog catalog = empty_catalog();
  CHECK(merge_release_index(catalog, text, kPrefix) == 1);
  REQUIRE(catalog.packages.size() == 1);
  CHECK(catalog.packages[0].url == url);
  CHECK(catalog.vendor_files.size() == 1);

  // Un paquet plus récent du même moteur remplace le précédent et ses fichiers.
  const std::string newer = "version = 1\n" + engine(os_name(), url, std::string(64, 'b')) +
                            vendor(os_name(), "https://pypi.nvidia.com/y.whl", "nvinfer2.dll");
  CHECK(merge_release_index(catalog, newer, kPrefix) == 1);
  REQUIRE(catalog.packages.size() == 1);
  CHECK(catalog.packages[0].sha256 == std::string(64, 'b'));
  REQUIRE(catalog.vendor_files.size() == 1);
  CHECK(catalog.vendor_files[0].name == "nvinfer2.dll");
}

TEST_CASE("Moteurs de la release : hors de la release ou de pypi.nvidia.com, écarté", "[app][install]") {
  EngineCatalog catalog = empty_catalog();
  const std::string elsewhere =
      "version = 1\n" + engine(os_name(), "https://example.com/m.tar.gz", std::string(64, 'a'));
  CHECK(merge_release_index(catalog, elsewhere, kPrefix) == 0);
  CHECK(catalog.packages.empty());
  CHECK_FALSE(catalog.problem.empty());

  const std::string url = std::string(kPrefix) + "m.tar.gz";
  const std::string bad_vendor = "version = 1\n" + engine(os_name(), url, std::string(64, 'a')) +
                                 vendor(os_name(), "https://example.com/x.whl", "nvinfer.dll");
  CHECK(merge_release_index(catalog, bad_vendor, kPrefix) == 0);
  CHECK(catalog.packages.empty());

  // Une autre plateforme ne compte pas.
  const std::string other =
      "version = 1\n" + engine(os_name() == "linux" ? "windows" : "linux", url, std::string(64, 'a'));
  CHECK(merge_release_index(catalog, other, kPrefix) == 0);
}

TEST_CASE("Moteurs de la release : sans réseau, la copie gardée sert", "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-cache";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  const std::string url = std::string(kPrefix) + "m.tar.gz";
  std::ofstream(cache) << "version = 1\n" << engine(os_name(), url, std::string(64, 'a'));
  EngineCatalog catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", UpdateFetch());
  CHECK(catalog.packages.size() == 1);
  CHECK(release_download_prefix("o/r", "1.2.3") == kPrefix);
  fs::remove_all(dir);
}
