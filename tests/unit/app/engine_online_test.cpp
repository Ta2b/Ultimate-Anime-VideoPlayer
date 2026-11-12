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
#include "core/update_manifest.h"

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
  std::string text = "version = 1\ntag = \"v1.2.3\"\n";
  for (const std::string_view platform : {"linux", "windows", "macos"})
    text +=
        engine(platform, "https://github.com/o/r/releases/download/trt-abc/m.tar.gz", std::string(64, 'a'));
  return text;
}
constexpr std::string_view kTestKey = "6e9a2010904180bcac497c94a59aa5ebf48a01578501b17a2493a8782ffb3a3f";
constexpr std::string_view kTestSignature =
    "57c0ffe2b77eb8d212e7e4c78339aed31b2c85e8a87181035432ec25098210ecf3188fdce24eada703e2f9074a94a948e3449739"
    "e47e74046da339ff45105b0b";
// Le même index avec « tag = "v9.9.9" », signé par la même clé : celui d'une autre release.
constexpr std::string_view kOtherTagSignature =
    "a97b27690df97e7d270f16041e1004072caf2cde98d54ba2631f1af4c70bd352a1cb48246b4b3ced7a571e4062adb06c000206c9"
    "b11bfe8f4c1bf70a9a50ca02";
// L'index d'avant l'étiquette (sans « tag »), signé par une ancienne clé de test : la signature est bonne,
// mais l'index ne dit pas de quelle release il est.
constexpr std::string_view kTaglessKey = "adf8860afcb60ac5a6de8c2d2ee667bd04e4e88f85570c5fe73b2c75f4c3358f";
constexpr std::string_view kTaglessSignature =
    "b0c2c0c4053091bf05d9dabcb7c0797a85e5d7079e487d9f3f18c4ffaa4c2059dc668021a86a5200026fa6d73efd4f99b69a80ff"
    "11584d6c3fca3a95a3528e0f";

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

TEST_CASE("Moteurs de la release : une panne réseau se retient quelques heures, la copie reste (A18-02)",
          "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-offline";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  write(cache, signed_index());
  write(dir / "engines-nvidia.lock.sig", std::string(kTestSignature) + "\n");
  fs::last_write_time(cache, fs::file_time_type::clock::now() - std::chrono::hours(48));
  CHECK(release_index_due(cache, "1.2.3"));
  int asked = 0;
  const UpdateFetch offline = [&](std::string_view, const fs::path&, int,
                                  const std::stop_token&) -> std::expected<void, Refusal> {
    ++asked;
    return std::unexpected(Refusal{RefusalCode::IoFailed, "hors ligne"});
  };
  refresh_release_index(cache, "o/r", "1.2.3", offline, std::stop_token{});
  CHECK(asked == 1);
  CHECK(fs::exists(dir / "engines-nvidia.lock.tried"));
  CHECK_FALSE(release_index_due(cache, "1.2.3"));
  // La copie gardée et signée sert toujours.
  EngineCatalog catalog = empty_catalog();
  merge_release_copy(catalog, read_release_index(cache), "o/r", {std::string(kTestKey)}, "1.2.3");
  CHECK(catalog.packages.size() == 1);
  // Pas de nouvel essai tout de suite ; quelques heures après, si.
  refresh_release_index(cache, "o/r", "1.2.3", offline, std::stop_token{});
  CHECK(asked == 1);
  fs::last_write_time(dir / "engines-nvidia.lock.tried",
                      fs::file_time_type::clock::now() - std::chrono::hours(7));
  CHECK(release_index_due(cache, "1.2.3"));
  refresh_release_index(cache, "o/r", "1.2.3", offline, std::stop_token{});
  CHECK(asked == 2);

  // Le lecteur fermé pendant la requête : ni marque ni copie effacée.
  fs::remove(dir / "engines-nvidia.lock.tried");
  std::stop_source closing;
  closing.request_stop();
  const UpdateFetch cancelled = [](std::string_view, const fs::path&, int,
                                   const std::stop_token&) -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::Cancelled, "annulé"});
  };
  refresh_release_index(cache, "o/r", "1.2.3", cancelled, closing.get_token());
  CHECK_FALSE(fs::exists(dir / "engines-nvidia.lock.tried"));
  CHECK(fs::exists(cache));
  fs::remove_all(dir);
}

TEST_CASE(
    "Moteurs de la release : un index signé d'une autre release, ou sans étiquette, est écarté (B51-06)",
    "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-tag";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  const std::vector<std::string> keys{std::string(kTestKey), std::string(kTaglessKey)};
  CHECK(release_index_tag(signed_index()) == "v1.2.3");

  // Le bon index, signé, sous la bonne étiquette : pris.
  write(cache, signed_index());
  write(dir / "engines-nvidia.lock.sig", std::string(kTestSignature) + "\n");
  EngineCatalog catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, UpdateFetch());
  CHECK(catalog.packages.size() == 1);

  // Le même index recopié dans la release d'une autre version : écarté, malgré sa signature valide.
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.4", keys, UpdateFetch());
  CHECK(catalog.packages.empty());
  CHECK(catalog.problem.contains("v1.2.4"));

  // Un index qui dit une autre étiquette (signé lui aussi).
  std::string other = signed_index();
  other.replace(other.find("v1.2.3"), 6, "v9.9.9");
  write(cache, other);
  write(dir / "engines-nvidia.lock.sig", std::string(kOtherTagSignature) + "\n");
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, UpdateFetch());
  CHECK(catalog.packages.empty());

  // Un index d'avant l'étiquette : sa signature est bonne, mais il ne dit pas de quelle release il est.
  std::string tagless = signed_index();
  tagless.erase(tagless.find("tag = "), std::string("tag = \"v1.2.3\"\n").size());
  write(cache, tagless);
  write(dir / "engines-nvidia.lock.sig", std::string(kTaglessSignature) + "\n");
  catalog = empty_catalog();
  add_release_engines(catalog, cache, "o/r", "1.2.3", keys, UpdateFetch());
  CHECK(catalog.packages.empty());
  CHECK_FALSE(catalog.problem.empty());
  fs::remove_all(dir);
}

// La même clé signe le manifeste des mises à jour et l'index des moteurs, sans préfixe de domaine dans le
// message signé (B26-12) : la séparation tient au format. Un fichier signé pour l'un ne se lit pas comme
// l'autre.
TEST_CASE("Moteurs de la release : un index ne se lit pas comme un manifeste, ni l'inverse (B26-12)",
          "[app][install]") {
  CHECK_FALSE(core::parse_update_manifest(signed_index()));
  const std::string manifest =
      "version = \"99.0.0\"\nexpires = \"2099-12-31\"\nnotes_url = \"https://example.invalid/n\"\n";
  REQUIRE(core::parse_update_manifest(manifest));
  const auto as_index = core::read_engine_index(manifest);
  CHECK((!as_index || as_index->packages.empty()));
  EngineCatalog catalog = empty_catalog();
  CHECK(merge_release_index(catalog, manifest, kPrefix) == 0);
  CHECK(catalog.packages.empty());
}

// C06-01 : après une mise à jour du lecteur, la copie gardée (étiquette de la version d'avant), encore
// fraîche, ne retient plus la relecture : sans cela, le catalogue n'a plus de paquet TensorRT jusqu'au
// lendemain.
TEST_CASE("Moteurs de la release : une copie fraîche d'une autre version est relue (C06-01)",
          "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-newer";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  write(cache, signed_index());  // étiquette v1.2.3
  write(dir / "engines-nvidia.lock.sig", std::string(kTestSignature) + "\n");
  CHECK_FALSE(release_index_due(cache, "1.2.3"));  // la bonne version, fraîche : rien à demander
  CHECK(release_index_due(cache, "1.2.4"));        // le lecteur a été mis à jour : à relire
  int asked = 0;
  const UpdateFetch offline = [&](std::string_view, const fs::path&, int,
                                  const std::stop_token&) -> std::expected<void, Refusal> {
    ++asked;
    return std::unexpected(Refusal{RefusalCode::IoFailed, "hors ligne"});
  };
  refresh_release_index(cache, "o/r", "1.2.4", offline, std::stop_token{});
  CHECK(asked == 1);
  fs::remove_all(dir);
}

// C06-04 : un 404 se retient deux heures, pas un jour, et seulement pour la version qui l'a vu.
TEST_CASE("Moteurs de la release : un 404 se retient deux heures et pour sa version (C06-04)",
          "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-404";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  const UpdateFetch missing = [](std::string_view, const fs::path&, int,
                                 const std::stop_token&) -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "404"});
  };
  refresh_release_index(cache, "o/r", "1.2.3", missing, std::stop_token{});
  REQUIRE(fs::exists(dir / "engines-nvidia.lock.absent"));
  CHECK_FALSE(release_index_due(cache, "1.2.3"));
  CHECK(release_index_due(cache, "1.2.4"));  // le 404 de l'ancienne version ne retient rien
  fs::last_write_time(dir / "engines-nvidia.lock.absent",
                      fs::file_time_type::clock::now() - std::chrono::hours(3));
  CHECK(release_index_due(cache, "1.2.3"));
  fs::remove_all(dir);
}

// C06-10 : une relecture réussie laisse le texte et la signature de la même relecture, jamais l'un des deux
// d'une autre ; l'ancienne copie est retirée avant les renommages.
TEST_CASE("Moteurs de la release : la copie relue remplace l'ancienne en entier (C06-10)", "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-online-pair";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path cache = dir / "engines-nvidia.lock";
  write(cache, "ancien texte\n");
  write(dir / "engines-nvidia.lock.sig", "ancienne signature\n");
  fs::last_write_time(cache, fs::file_time_type::clock::now() - std::chrono::hours(48));
  const UpdateFetch fetch = [](std::string_view url, const fs::path& to, int,
                               const std::stop_token&) -> std::expected<void, Refusal> {
    write(to, url.ends_with(".sig") ? "nouvelle signature\n" : "nouveau texte\n");
    return {};
  };
  refresh_release_index(cache, "o/r", "1.2.3", fetch, std::stop_token{});
  const ReleaseIndexCopy copy = read_release_index(cache);
  CHECK(copy.text == "nouveau texte\n");
  CHECK(copy.signature == "nouvelle signature\n");
  // Une copie sans texte (arrêt entre les renommages) est à relire, pas fraîche.
  fs::remove(cache);
  CHECK(release_index_due(cache, "1.2.3"));
  fs::remove_all(dir);
}
