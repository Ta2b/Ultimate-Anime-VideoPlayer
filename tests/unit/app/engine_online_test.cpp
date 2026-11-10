// Les moteurs joints à la release après elle (engines-nvidia.lock) : pris pour cet OS et sous la release de
// cette version seulement, un moteur déjà livré remplacé avec ses fichiers NVIDIA, le reste écarté.
#include "app/play/engine_online.h"

#include <chrono>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <stop_token>
#include <string>
#include <vector>

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

// L'index des trois plateformes signé par la clé de test kTestKey (signature kTestSignature, openssl
// pkeyutl -sign -rawin) : l'archive vit dans la release fixe trt-abc du dépôt, comme en CI.
std::string signed_index() {
  std::string text = "version = 1\n";
  for (const std::string_view platform : {"linux", "windows", "macos"})
    text +=
        engine(platform, "https://github.com/o/r/releases/download/trt-abc/m.tar.gz", std::string(64, 'a'));
  return text;
}
constexpr std::string_view kTestKey = "adf8860afcb60ac5a6de8c2d2ee667bd04e4e88f85570c5fe73b2c75f4c3358f";
constexpr std::string_view kTestSignature =
    "b0c2c0c4053091bf05d9dabcb7c0797a85e5d7079e487d9f3f18c4ffaa4c2059dc668021a86a5200026fa6d73efd4f99b69a80ff"
    "1158"
    "4d6c3fca3a95a3528e0f";

void write(const fs::path& file, std::string_view text) {
  std::ofstream(file, std::ios::binary) << text;
}

TEST_CASE("Moteurs de la release : sans réseau, la copie gardée et signée sert", "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-cache";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  write(cache, signed_index());
  write(dir / "engines-nvidia.lock.sig", std::string(kTestSignature) + "\n");
  const std::vector<std::string> keys{std::string(kTestKey)};
  EngineCatalog catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, UpdateFetch());
  CHECK(catalog.packages.size() == 1);
  CHECK(catalog.problem.empty());
  CHECK(release_download_prefix("o/r", "1.2.3") == kPrefix);
  CHECK(release_repo_prefix("o/r") == "https://github.com/o/r/releases/download/");
  fs::remove_all(dir);
}

TEST_CASE("Moteurs de la release : sans signature valide, l'index n'est pas cru (A26-02)", "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-unsigned";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  const std::vector<std::string> keys{std::string(kTestKey)};

  // Pas de signature du tout.
  write(cache, signed_index());
  EngineCatalog catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, UpdateFetch());
  CHECK(catalog.packages.empty());
  CHECK(catalog.problem.contains("signature"));

  // L'index changé après signature (un autre SHA-256).
  std::string changed = signed_index();
  changed[changed.find("aaaa")] = 'b';
  write(cache, changed);
  write(dir / "engines-nvidia.lock.sig", kTestSignature);
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, UpdateFetch());
  CHECK(catalog.packages.empty());

  // Signé, mais par une clé que le lecteur ne croit pas.
  write(cache, signed_index());
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", {std::string(64, '0')}, UpdateFetch());
  CHECK(catalog.packages.empty());
  fs::remove_all(dir);
}

TEST_CASE("Moteurs de la release : l'index et sa signature se téléchargent ensemble", "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-fetch";
  fs::remove_all(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  const std::vector<std::string> keys{std::string(kTestKey)};
  std::vector<std::string> asked;
  const UpdateFetch fetch = [&](std::string_view url, const fs::path& file, int,
                                const std::stop_token&) -> std::expected<void, Refusal> {
    asked.emplace_back(url);
    write(file, url.ends_with(".sig") ? std::string(kTestSignature) : signed_index());
    return {};
  };
  EngineCatalog catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, fetch);
  CHECK(catalog.packages.size() == 1);
  REQUIRE(asked.size() == 2);
  CHECK(asked[0] == std::string(kPrefix) + "engines-nvidia.lock");
  CHECK(asked[1] == std::string(kPrefix) + "engines-nvidia.lock.sig");
  CHECK(fs::exists(dir / "engines-nvidia.lock.sig"));

  // La release n'a plus d'index (404) : les copies gardées s'effacent.
  fs::last_write_time(cache, fs::file_time_type::clock::now() - std::chrono::hours(48));
  const UpdateFetch gone = [](std::string_view, const fs::path&, int,
                              const std::stop_token&) -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "404"});
  };
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, gone);
  CHECK(catalog.packages.empty());
  CHECK_FALSE(fs::exists(cache));
  CHECK_FALSE(fs::exists(dir / "engines-nvidia.lock.sig"));

  // Le 404 est retenu un jour : le lancement suivant ne refait pas la requête.
  int asked_again = 0;
  const UpdateFetch counting = [&](std::string_view, const fs::path&, int,
                                   const std::stop_token&) -> std::expected<void, Refusal> {
    ++asked_again;
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "404"});
  };
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, counting);
  CHECK(asked_again == 0);

  // Un jour plus tard, on redemande ; et un téléchargement réussi retire la marque.
  fs::last_write_time(dir / "engines-nvidia.lock.absent",
                      fs::file_time_type::clock::now() - std::chrono::hours(48));
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, counting);
  CHECK(asked_again == 1);
  fs::last_write_time(dir / "engines-nvidia.lock.absent",
                      fs::file_time_type::clock::now() - std::chrono::hours(48));
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, fetch);
  CHECK(catalog.packages.size() == 1);
  CHECK_FALSE(fs::exists(dir / "engines-nvidia.lock.absent"));
  fs::remove_all(dir);
}
