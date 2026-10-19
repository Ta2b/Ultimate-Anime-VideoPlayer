// Magasin des SDK sur de faux dossiers. Fige le comportement repris de manager-v2
// Services/TrueHdrSdkStore.cs : un refus ne touche jamais le SDK déjà choisi.
#include "vendor_hdr/sdk_store.h"

#include <filesystem>
#include <format>
#include <iterator>
#include <stop_token>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "support.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;
namespace fs = std::filesystem;

namespace {

struct Fixture {
  TempDir dir;
  fs::path source = dir.path() / "VideoFX";
  fs::path data = dir.path() / "data";
  fs::path store_root = data / "optional-sdks" / "maxine-vfx";
  SdkStore store{data, SdkKind::MaxineVfx};
};

std::size_t count_entries(const fs::path& directory) {
  std::error_code error;
  if (!fs::is_directory(directory, error)) return 0;
  return static_cast<std::size_t>(std::distance(fs::directory_iterator(directory), fs::directory_iterator()));
}

std::string import_refusal(Fixture& fixture) {
  auto imported = fixture.store.import(fixture.source);
  REQUIRE_FALSE(imported);
  CHECK(imported.error().code == RefusalCode::SdkRejected);
  CHECK(fixture.store.status().state == SdkState::NotInstalled);
  CHECK(count_entries(fixture.store_root / "versions") == 0);
  return imported.error().detail;
}

}  // namespace

TEST_CASE("magasin : rien d'importé", "[store]") {
  Fixture fixture;
  const SdkStatus status = fixture.store.status();
  CHECK(status.kind == SdkKind::MaxineVfx);
  CHECK(status.state == SdkState::NotInstalled);
  CHECK(status.reason ==
        "Maxine VFX 1.3 (TrueHDR) non importé : choisir son dossier dans Paramètres > Moteurs.");
  CHECK_FALSE(status.installed);
}

TEST_CASE("magasin : la bonne version s'importe et se vérifie", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  auto imported = fixture.store.import(fixture.source);
  REQUIRE(imported);
  CHECK(imported->state == SdkState::Ready);
  CHECK(imported->reason.empty());
  REQUIRE(imported->installed);
  const InstalledSdk& sdk = *imported->installed;
  CHECK(sdk.kind() == SdkKind::MaxineVfx);
  CHECK(sdk.version().text() == "1.3.0.0");
  CHECK(sdk.root().parent_path() == fixture.store_root / "versions");
  CHECK(sdk.root().filename().string().starts_with("1.3.0.0-"));
  CHECK(fs::exists(sdk.root() / "share/license.txt"));
  CHECK(fs::is_directory(sdk.root() / "lib/models"));
  CHECK(fs::exists(fixture.store_root / "current.toml"));

  // Une autre instance du magasin (au démarrage suivant) relit la même chose.
  const SdkStatus again = SdkStore(fixture.data, SdkKind::MaxineVfx).status();
  CHECK(again.state == SdkState::Ready);
  CHECK(again.installed->root() == sdk.root());
}

TEST_CASE("magasin : un dossier de données relatif donne des chemins absolus", "[store]") {
  // Les chargeurs refusent un chemin relatif ; le magasin l'absolutise, comme Path.GetFullPath
  // du Manager (TrueHdrSdkStore.cs:30).
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  const fs::path relative = fs::relative(fixture.data, fs::current_path());
  REQUIRE(relative.is_relative());
  SdkStore store(relative, SdkKind::MaxineVfx);
  auto imported = store.import(fixture.source);
  REQUIRE(imported);
  CHECK(imported->installed->root().is_absolute());
  CHECK(fs::equivalent(imported->installed->root().parent_path(), fixture.store_root / "versions"));
}

TEST_CASE("magasin : fichiers vides refusés", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source, {.libraries = LibraryContent::Empty});
  CHECK(import_refusal(fixture) == "external/cuda/lib/libcudart.so.12 est vide");
}

TEST_CASE("magasin : mauvaise version refusée", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source, {.version_header = "1.2.0.0", .feature_version = "1.2.0.0"});
  CHECK(import_refusal(fixture) ==
        "version 1.2.0.0 : ce lecteur prend en charge le Maxine VFX 1.3 (TrueHDR) (1.3)");
}

TEST_CASE("magasin : version introuvable refusée", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source, {.version_header = std::nullopt});
  CHECK(import_refusal(fixture).starts_with("version du SDK introuvable"));
}

TEST_CASE("magasin : version lue dans le nom réel de libVideoFX", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source, {.version_header = std::nullopt});
  fs::rename(fixture.source / "lib/libVideoFX.so", fixture.source / "lib/libVideoFX.so.1.3.0");
  fs::create_symlink("libVideoFX.so.1.3.0", fixture.source / "lib/libVideoFX.so");
  auto imported = fixture.store.import(fixture.source);
  REQUIRE(imported);
  CHECK(imported->state == SdkState::Ready);
  CHECK(imported->installed->version().text() == "1.3.0.0");
  // Le lien interne est recréé tel quel dans la copie.
  const fs::path copied = imported->installed->root() / "lib/libVideoFX.so";
  CHECK(fs::is_symlink(copied));
  CHECK(fs::read_symlink(copied) == "libVideoFX.so.1.3.0");
}

TEST_CASE("magasin : marqueurs de version qui se contredisent", "[store]") {
  Fixture fixture;
  SECTION("en-tête et nom réel de libVideoFX") {
    make_maxine_sdk(fixture.source);
    fs::rename(fixture.source / "lib/libVideoFX.so", fixture.source / "lib/libVideoFX.so.1.4.0");
    fs::create_symlink("libVideoFX.so.1.4.0", fixture.source / "lib/libVideoFX.so");
    CHECK(import_refusal(fixture) ==
          "share/videofx_version.h annonce 1.3.0.0, mais lib/libVideoFX.so est en 1.4.0.0 : SDK mélangé");
  }
  SECTION("fonction TrueHDR d'une autre version") {
    make_maxine_sdk(fixture.source, {.feature_version = "1.2.0.0"});
    CHECK(
        import_refusal(fixture).starts_with("la fonction TrueHDR 1.2.0.0 ne correspond pas au SDK 1.3.0.0"));
  }
  SECTION("fonction TrueHDR sans version") {
    make_maxine_sdk(fixture.source);
    write_file(fixture.source / "features/nvvfxtruehdr/include/nvVFXTrueHDR.h", abi_header_text());
    CHECK(import_refusal(fixture).starts_with("version de la fonction TrueHDR introuvable"));
  }
}

TEST_CASE("magasin : SDK incomplet refusé avec la marche à suivre", "[store]") {
  Fixture fixture;
  SECTION("fonction TrueHDR absente") {
    make_maxine_sdk(fixture.source, {.feature = false});
    CHECK(
        import_refusal(fixture) ==
        "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so manque : la fonction TrueHDR manque : l'installer avec "
        "install_feature.sh --feature nvvfxtruehdr");
  }
  SECTION("en-têtes absents") {
    make_maxine_sdk(fixture.source, {.headers = false});
    CHECK(import_refusal(fixture).starts_with("include/nvCVImage.h manque : les en-têtes du SDK manquent"));
  }
  SECTION("dépendance CUDA absente") {
    make_maxine_sdk(fixture.source);
    fs::remove(fixture.source / "external/cuda/lib/libnppig.so.12");
    CHECK(import_refusal(fixture).starts_with("external/cuda/lib/libnppig.so.12 manque"));
  }
  SECTION("pas un dossier") {
    write_file(fixture.source, "archive.zip");
    CHECK(import_refusal(fixture).ends_with("n'est pas un dossier"));
  }
}

TEST_CASE("magasin : archive d'une autre architecture refusée", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  write_file(fixture.source / "lib/libVideoFXLocal.so", elf_header(183));
  CHECK(import_refusal(fixture).starts_with("lib/libVideoFXLocal.so n'est pas une bibliothèque Linux x64"));
}

TEST_CASE("magasin : chaque bibliothèque chargeable est vérifiée, pas seulement les exigées", "[store]") {
  // Celles de lib/ et de la fonction, que le SDK peut ouvrir par leur nom (sdk_layout.h).
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  SECTION("une bibliothèque facultative ARM est refusée") {
    write_file(fixture.source / "lib/libVideoFXTriton.so", elf_header(183));
    CHECK(
        import_refusal(fixture).starts_with("lib/libVideoFXTriton.so n'est pas une bibliothèque Linux x64"));
  }
  SECTION("ni un lien pendant ni un autre fichier ne comptent") {
    fs::create_symlink("libVideoFXTriton.so.1.3.0", fixture.source / "lib/libVideoFXTriton.so.1");
    write_file(fixture.source / "lib/notes.txt", "");
    write_file(fixture.source / "lib/libVideoFX.so.debug", "");
    CHECK(fixture.store.import(fixture.source));
  }
  SECTION("ce que TrueHDR ne charge pas n'est ni vérifié ni copié") {
    write_file(fixture.source / "external/cuda/lib/libcublas.so.12", elf_header(183));
    write_file(fixture.source / "external/tensorrt/lib/libnvinfer.so.10", elf_header());
    auto imported = fixture.store.import(fixture.source);
    REQUIRE(imported);
    const fs::path copy = imported->installed->root();
    CHECK(fs::exists(copy / "external/cuda/lib/libcudart.so.12"));
    CHECK_FALSE(fs::exists(copy / "external/cuda/lib/libcublas.so.12"));
    CHECK_FALSE(fs::exists(copy / "external/tensorrt"));
  }
}

TEST_CASE("magasin : une bibliothèque externe nommée par son lien apporte sa cible", "[store]") {
  // Disposition réelle : libcudart.so.12 -> libcudart.so.12.8.90 dans external/cuda/lib.
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  const fs::path cuda = fixture.source / "external/cuda/lib";
  fs::rename(cuda / "libcudart.so.12", cuda / "libcudart.so.12.8.90");
  fs::create_symlink("libcudart.so.12.8.90", cuda / "libcudart.so.12");
  fs::create_symlink("libcudart.so.12", cuda / "libcudart.so");
  auto imported = fixture.store.import(fixture.source);
  REQUIRE(imported);
  const fs::path copied = imported->installed->root() / "external/cuda/lib";
  CHECK(fs::read_symlink(copied / "libcudart.so.12") == "libcudart.so.12.8.90");
  CHECK(fs::is_regular_file(fs::symlink_status(copied / "libcudart.so.12.8.90")));
  CHECK_FALSE(fs::exists(fs::symlink_status(copied / "libcudart.so")));
  CHECK(fixture.store.status().state == SdkState::Ready);
}

TEST_CASE("magasin : en-têtes d'une autre ABI refusés", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  std::string header = abi_header_text();
  header.replace(header.find("NVCV_RGB10A2 13"), 15, "NVCV_RGB10A2 15");
  write_file(fixture.source / "include/nvCVImage.h", header);
  CHECK(import_refusal(fixture).starts_with("NVCV_RGB10A2 vaut 15 dans les en-têtes du SDK"));
}

TEST_CASE("magasin : un lien qui sort du SDK est refusé", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  write_file(fixture.dir.path() / "secret.txt", "hors du SDK");
  fs::create_symlink("../../secret.txt", fixture.source / "share/escape.txt");
  auto imported = fixture.store.import(fixture.source);
  REQUIRE_FALSE(imported);
  CHECK(imported.error().detail.starts_with("share/escape.txt est un lien qui sort du SDK"));
  CHECK(fixture.store.status().state == SdkState::NotInstalled);
  CHECK(count_entries(fixture.store_root) == 1);  // versions/ seul : la copie a été retirée
}

TEST_CASE("magasin : import annulé", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  std::stop_source stop;
  stop.request_stop();
  auto imported = fixture.store.import(fixture.source, stop.get_token());
  REQUIRE_FALSE(imported);
  CHECK(imported.error().code == RefusalCode::Cancelled);
  CHECK(fixture.store.status().state == SdkState::NotInstalled);
  CHECK(count_entries(fixture.store_root) == 1);
}

TEST_CASE("magasin : un réimport refusé garde le SDK choisi", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  auto first = fixture.store.import(fixture.source);
  REQUIRE(first);
  const fs::path chosen = first->installed->root();

  const fs::path other = fixture.dir.path() / "older";
  make_maxine_sdk(other, {.version_header = "1.2.0.0", .feature_version = "1.2.0.0"});
  REQUIRE_FALSE(fixture.store.import(other));
  SdkStatus status = fixture.store.status();
  CHECK(status.state == SdkState::Ready);
  CHECK(status.installed->root() == chosen);

  // Un réimport accepté choisit une nouvelle copie, sans toucher l'ancienne.
  auto second = fixture.store.import(fixture.source);
  REQUIRE(second);
  CHECK(second->installed->root() != chosen);
  CHECK(fs::exists(chosen / "lib/libVideoFX.so"));
}

TEST_CASE("magasin : une copie interrompue est nettoyée au prochain import", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  write_file(fixture.store_root / ".import-dead/lib/partial.so", "moitié");
  REQUIRE(fixture.store.import(fixture.source));
  CHECK_FALSE(fs::exists(fixture.store_root / ".import-dead"));
}

TEST_CASE("magasin : copie modifiée après l'import", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  auto imported = fixture.store.import(fixture.source);
  REQUIRE(imported);
  const fs::path copy = imported->installed->root();

  SECTION("taille changée") {
    write_file(copy / "lib/libVideoFX.so", elf_header() + "plus long");
    const SdkStatus status = fixture.store.status();
    CHECK(status.state == SdkState::Damaged);
    CHECK(status.reason ==
          "lib/libVideoFX.so a changé depuis l'import : réimporter le Maxine VFX 1.3 (TrueHDR) (Paramètres > "
          "Moteurs).");
    CHECK_FALSE(status.installed);
  }
  SECTION("fichier supprimé") {
    fs::remove(copy / "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so");
    CHECK(fixture.store.status().state == SdkState::Damaged);
  }
  SECTION("même taille, contenu revérifié") {
    std::string header = abi_header_text();
    header.replace(header.find("NVCV_RGB10A2 13"), 15, "NVCV_RGB10A2 14");
    write_file(copy / "include/nvCVImage.h", header);
    const SdkStatus status = fixture.store.status();
    CHECK(status.state == SdkState::Damaged);
    CHECK(status.reason.starts_with("NVCV_RGB10A2 vaut 14"));
  }
  SECTION("en-tête de version changé") {
    write_file(copy / "share/videofx_version.h", version_header_text(SdkVersion{.parts = {1, 3, 9, 0}}));
    CHECK(fixture.store.status().reason.starts_with("version changée depuis l'import"));
  }
}

TEST_CASE("magasin : manifeste illisible, ancien ou plus récent", "[store]") {
  Fixture fixture;
  make_maxine_sdk(fixture.source);
  auto imported = fixture.store.import(fixture.source);
  REQUIRE(imported);
  const std::string copy = imported->installed->root().filename().string();
  const fs::path manifest = fixture.store_root / "current.toml";

  SECTION("illisible") {
    write_file(manifest, "pas du TOML = =");
    CHECK(fixture.store.status().state == SdkState::Damaged);
  }
  SECTION("format plus récent") {
    write_file(manifest, std::format("format = 2\nkind = \"maxine-vfx\"\nversion = \"1.3.0.0\"\n"
                                     "copy = \"versions/{}\"\n",
                                     copy));
    const SdkStatus status = fixture.store.status();
    CHECK(status.state == SdkState::WrongVersion);
    CHECK(status.reason.contains("lecteur plus récent"));
  }
  SECTION("SDK d'une version que ce lecteur ne prend plus en charge") {
    write_file(manifest, std::format("format = 1\nkind = \"maxine-vfx\"\nversion = \"1.2.0.0\"\n"
                                     "copy = \"versions/{}\"\n",
                                     copy));
    const SdkStatus status = fixture.store.status();
    CHECK(status.state == SdkState::WrongVersion);
    CHECK(status.reason.starts_with("SDK 1.2.0.0 importé, ce lecteur prend en charge 1.3"));
  }
  SECTION("copie disparue") {
    fs::remove_all(imported->installed->root());
    CHECK(fixture.store.status().state == SdkState::Damaged);
  }
}

TEST_CASE("magasin : RTX Video SDK", "[store]") {
  TempDir dir;
  const fs::path source = dir.path() / "rtx_video_sdk";
  SdkStore store(dir.path() / "data", SdkKind::RtxVideo);
  CHECK(store.status().reason ==
        "RTX Video SDK 1.1 non importé : choisir son dossier dans Paramètres > Moteurs.");

  SECTION("1.1 accepté") {
    make_rtx_sdk(source, SdkVersion{.parts = {1, 1, 0, 0}});
    write_file(source / "cudart64_12.dll", pe_dll(std::nullopt));
    write_file(source / "readme.txt", "non copié");
    auto imported = store.import(source);
    REQUIRE(imported);
    CHECK(imported->state == SdkState::Ready);
    const fs::path root = imported->installed->root();
    CHECK(root.parent_path() == dir.path() / "data/optional-sdks/rtx-video/versions");
    CHECK(fs::exists(root / "cudart64_12.dll"));
    CHECK_FALSE(fs::exists(root / "readme.txt"));
  }
  SECTION("1.0 refusé") {
    make_rtx_sdk(source, SdkVersion{.parts = {1, 0, 0, 0}});
    auto imported = store.import(source);
    REQUIRE_FALSE(imported);
    CHECK(imported.error().detail ==
          "version 1.0.0.0 : ce lecteur prend en charge le RTX Video SDK 1.1 (1.1)");
  }
  SECTION("DLL ARM64 refusée") {
    make_rtx_sdk(source, SdkVersion{.parts = {1, 1, 0, 0}});
    write_file(source / "nvngx_truehdr.dll", pe_dll(std::nullopt, 0xAA64));
    CHECK(store.import(source).error().detail == "nvngx_truehdr.dll n'est pas une DLL Windows x64");
  }
  SECTION("DLL copiée d'une autre architecture refusée") {
    make_rtx_sdk(source, SdkVersion{.parts = {1, 1, 0, 0}});
    write_file(source / "cudart64_12.dll", elf_header());
    CHECK(store.import(source).error().detail == "cudart64_12.dll n'est pas une DLL Windows x64");
  }
  SECTION("bibliothèques Linux refusées") {
    write_file(source / "rtx_video_api.dll", elf_header());
    write_file(source / "nvngx_truehdr.dll", elf_header());
    CHECK(store.import(source).error().detail == "rtx_video_api.dll n'est pas une DLL Windows x64");
  }
  CHECK(store.status().state != SdkState::Damaged);
}
