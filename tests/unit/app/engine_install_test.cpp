// Installation des moteurs et de RTX HDR depuis une archive locale (url file://, que curl sert
// comme un https) : le paquet complet, le SHA-256 faux, l'archive sans bibliothèque, la reprise
// d'un .part et le .part complet mais faux, puis le paquet RTX HDR importé dans le magasin des
// SDK. Sans curl ni tar sur le système, les cas sont sautés.
#include "app/play/engine_install.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../vendor_hdr/support.h"
#include "app/play/engine_tools.h"
#include "app/play/maxine_install.h"
#include "core/model_index.h"
#include "french_ui.h"
#include "vendor_hdr/sdk_store.h"

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

// Une variable d'environnement, vide si absente (getenv est refusé par MSVC sous /WX).
std::string env_text(const char* name) {
#ifdef _WIN32
  char* value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) return {};
  std::string text(value);
  std::free(value);
  return text;
#else
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
#endif
}

std::string read_all(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_all(const fs::path& file, std::string_view bytes) {
  fs::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << bytes;
  REQUIRE(out.good());
}

// L'archive tar.gz du dossier source, par le tar du système ; sauté s'il manque.
fs::path pack(const fs::path& source, const fs::path& archive) {
  const auto tar = system_tool("tar");
  if (!tar) SKIP("tar absent du système");
  auto made = run_tool({.program = *tar,
                        .args = {"-czf", archive.string(), "-C", source.string(), "."},
                        .timeout = std::chrono::minutes(1)},
                       {}, {}, "tar");
  if (!made) FAIL(made.error().detail);
  return archive;
}

struct Fixture {
  explicit Fixture(std::string_view test)
      : root(fs::temp_directory_path() / std::format("ajp-engine-install-{}", test)) {
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root / "data");
    catalog.layout = core::install_layout(host_os(), root / "data");
  }
  ~Fixture() {
    std::error_code error;
    fs::remove_all(root, error);
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  // Un paquet OpenVINO dont l'archive tient la bibliothèque du backend et un fichier à côté.
  core::EnginePackage openvino_package(bool with_library = true) {
    const fs::path source = root / "source";
    const fs::path library =
        core::backend_library(catalog.layout, core::EnginePackage{.engine = core::EngineId::OpenVino});
    if (with_library) write_all(source / library.filename(), "bibliothèque factice\n");
    write_all(source / "LICENCE.txt", "licence factice\n");
    return package_of(core::EngineId::OpenVino, pack(source, root / "openvino.tar.gz"));
  }

  core::EnginePackage package_of(core::EngineId engine, const fs::path& archive) {
    std::error_code error;
    return core::EnginePackage{.engine = engine,
                               .platform = catalog.layout.os,
                               .url = "file://" + archive.generic_string(),
                               .sha256 = core::sha256_file(archive).value(),
                               .size = fs::file_size(archive, error),
                               .version = "1.0"};
  }

  InstallProgress progress() {
    return InstallProgress{.report = [this](float fraction, const std::string& stage) {
      fractions.push_back(fraction);
      stages.push_back(stage);
    }};
  }

  fs::path root;
  EngineCatalog catalog;
  std::vector<float> fractions;
  std::vector<std::string> stages;
};

// Sauté quand le système n'a pas les outils ; échec sur tout autre refus.
template <class T>
const T& require_installed(const std::expected<T, Refusal>& result) {
  if (!result && result.error().code == RefusalCode::LibraryUnavailable) SKIP(result.error().detail);
  if (!result) FAIL(result.error().detail);
  return *result;
}

void require_done(const std::expected<void, Refusal>& result) {
  if (!result && result.error().code == RefusalCode::LibraryUnavailable) SKIP(result.error().detail);
  if (!result) FAIL(result.error().detail);
}

template <class T>
void require_refused(const std::expected<T, Refusal>& result, RefusalCode code) {
  if (!result && result.error().code == RefusalCode::LibraryUnavailable) SKIP(result.error().detail);
  REQUIRE_FALSE(result);
  INFO(result.error().detail);
  CHECK(result.error().code == code);
}

}  // namespace

namespace {

// Une « roue » factice : un préfixe de remplissage, puis les octets du membre ; curl en lit la plage.
struct Wheel {
  explicit Wheel(Fixture& f, std::string_view member, bool deflated) : fixture(f) {
    path = f.root / (deflated ? "roue-deflate.whl" : "roue-stockee.whl");
    constexpr std::string_view kPrefix = "PK-entete-";
    write_all(path, std::string(kPrefix) + std::string(member) + "-fin-de-la-roue");
    file.engine = core::EngineId::OpenVino;
    file.platform = f.catalog.layout.os;
    file.url = "file://" + path.generic_string();
    file.data_offset = kPrefix.size();
    file.packed_size = member.size();
    file.deflated = deflated;
  }
  Fixture& fixture;
  fs::path path;
  core::VendorFile file;
};

constexpr std::string_view kDeflated{"\x4b\x4c\x4a\x4e\x44\x45\x00", 7};  // « abcabcabcabcabcabc »
constexpr std::string_view kPlain = "abcabcabcabcabcabc";

}  // namespace

TEST_CASE("Fichier tiers : la plage d'une roue, décompressée et vérifiée", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("tiers");
  Wheel wheel(f, kDeflated, true);
  wheel.file.name = "libtiers.so.1";
  wheel.file.size = kPlain.size();
  wheel.file.sha256 = core::sha256_hex(kPlain);
  const fs::path dest = f.root / "sortie" / "libtiers.so.1";
  fs::create_directories(dest.parent_path());
  require_done(fetch_vendor_file(wheel.file, dest, f.root / "tmp", f.progress()));
  CHECK(read_all(dest) == kPlain);
  CHECK_FALSE(fs::exists(fs::path(dest.string() + ".partiel")));

  // Un membre stocké se recopie tel quel.
  Wheel stored(f, kPlain, false);
  stored.file.name = "stocke.so";
  stored.file.size = kPlain.size();
  stored.file.sha256 = core::sha256_hex(kPlain);
  require_done(fetch_vendor_file(stored.file, f.root / "sortie" / "stocke.so", f.root / "tmp", f.progress()));
  CHECK(read_all(f.root / "sortie" / "stocke.so") == kPlain);

  // Un SHA-256 faux ne laisse rien à l'arrivée.
  core::VendorFile wrong = wheel.file;
  wrong.sha256 = core::sha256_hex("autre");
  const fs::path refused = f.root / "sortie" / "faux.so";
  require_refused(fetch_vendor_file(wrong, refused, f.root / "tmp", f.progress()), RefusalCode::HashMismatch);
  CHECK_FALSE(fs::exists(refused));
  CHECK_FALSE(fs::exists(fs::path(refused.string() + ".partiel")));

  // Une plage qui dépasse la roue est refusée, et rien n'arrive à destination.
  core::VendorFile beyond = wheel.file;
  beyond.data_offset = fs::file_size(wheel.path) + 10;
  const auto outside = fetch_vendor_file(beyond, f.root / "sortie" / "loin.so", f.root / "tmp", f.progress());
  if (!outside && outside.error().code == RefusalCode::LibraryUnavailable) SKIP(outside.error().detail);
  CHECK_FALSE(outside);
  CHECK_FALSE(fs::exists(f.root / "sortie" / "loin.so"));
}

TEST_CASE("Installation : les fichiers tiers de l'architecture du GPU rejoignent le module",
          "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("tiers-installation");
  const core::EnginePackage package = f.openvino_package();
  Wheel wheel(f, kPlain, false);
  const auto vendor = [&](std::string name, std::string arch) {
    core::VendorFile file = wheel.file;
    file.name = std::move(name);
    file.arch = std::move(arch);
    file.size = kPlain.size();
    file.sha256 = core::sha256_hex(kPlain);
    return file;
  };
  f.catalog.vendor_files = {vendor("commun.so", ""), vendor("ada.so", "sm89"),
                            vendor("blackwell.so", "sm120")};
  InstallProgress progress = f.progress();
  progress.cuda_arch = "sm89";
  const fs::path library = require_installed(install_package(f.catalog, package, progress));
  const fs::path target = core::engine_dir(f.catalog.layout, package);
  CHECK(fs::exists(library));
  CHECK(read_all(target / "commun.so") == kPlain);
  CHECK(read_all(target / "ada.so") == kPlain);
  CHECK_FALSE(fs::exists(target / "blackwell.so"));

  // Un fichier tiers refusé fait échouer l'installation, sans dossier à moitié rempli.
  Fixture g("tiers-refuse");
  const core::EnginePackage other = g.openvino_package();
  Wheel bad(g, kPlain, false);
  bad.file.name = "faux.so";
  bad.file.size = kPlain.size();
  bad.file.sha256 = core::sha256_hex("autre");
  g.catalog.vendor_files = {bad.file};
  require_refused(install_package(g.catalog, other, g.progress()), RefusalCode::HashMismatch);
  CHECK_FALSE(installed(g.catalog, other));
}

TEST_CASE("Installation : un paquet complet depuis une archive locale", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("complet");
  const core::EnginePackage package = f.openvino_package();
  // Une version déjà extraite, à moitié, est remplacée entière.
  const fs::path target = core::engine_dir(f.catalog.layout, package);
  write_all(target / "reste.txt", "vieux\n");

  const fs::path library = require_installed(install_package(f.catalog, package, f.progress()));
  CHECK(library == core::backend_library(f.catalog.layout, package));
  CHECK(read_all(library) == "bibliothèque factice\n");
  CHECK(fs::exists(target / "LICENCE.txt"));
  CHECK_FALSE(fs::exists(target / "reste.txt"));
  CHECK(installed(f.catalog, package));
  CHECK_FALSE(fs::exists(core::download_path(f.catalog.layout, package.sha256)));
  CHECK_FALSE(fs::exists(fs::path(target.string() + ".partiel")));
  REQUIRE_FALSE(f.fractions.empty());
  CHECK(f.fractions.front() == 0.0F);
  CHECK(f.fractions.back() == 1.0F);
  CHECK(std::ranges::is_sorted(f.fractions));
  CHECK(f.stages.back() == "openvino installé");
}

TEST_CASE("Installation : le paquet installé et sa licence restent notés à côté du moteur",
          "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("note");
  core::EnginePackage package = f.openvino_package();
  CHECK_FALSE(installed_package(f.catalog, package));
  package.license = "https://exemple.invalid/licence";
  require_installed(install_package(f.catalog, package, f.progress()));
  CHECK(installed_package(f.catalog, package) ==
        InstalledPackage{.sha256 = package.sha256, .license = "https://exemple.invalid/licence"});
}

TEST_CASE("Installation : un SHA-256 faux refuse le paquet et jette le fichier", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("sha");
  core::EnginePackage package = f.openvino_package();
  package.sha256 = std::string(64, 'a');
  require_refused(install_package(f.catalog, package, f.progress()), RefusalCode::HashMismatch);
  // Un fichier faux ne se reprend pas : effacé, le prochain essai repart de zéro.
  CHECK_FALSE(fs::exists(core::download_path(f.catalog.layout, package.sha256)));
  CHECK_FALSE(fs::exists(core::engine_dir(f.catalog.layout, package)));
  CHECK_FALSE(installed(f.catalog, package));
}

TEST_CASE("Installation : une taille fausse refuse le paquet", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("taille");
  core::EnginePackage package = f.openvino_package();
  package.size += 1;
  require_refused(install_package(f.catalog, package, f.progress()), RefusalCode::HashMismatch);
  CHECK_FALSE(fs::exists(core::download_path(f.catalog.layout, package.sha256)));
}

TEST_CASE("Installation : une archive sans la bibliothèque du backend est refusée", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("sans-bibliotheque");
  const core::EnginePackage package = f.openvino_package(false);
  const auto result = install_package(f.catalog, package, f.progress());
  require_refused(result, RefusalCode::HashMismatch);
  CHECK(result.error().detail.contains("ajp_backend_openvino"));
  const fs::path target = core::engine_dir(f.catalog.layout, package);
  CHECK_FALSE(fs::exists(target));
  CHECK_FALSE(fs::exists(fs::path(target.string() + ".partiel")));
  CHECK_FALSE(installed(f.catalog, package));
}

TEST_CASE("Installation : un .part interrompu reprend où il en était", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("reprise");
  const core::EnginePackage package = f.openvino_package();
  const std::string archive = read_all(f.root / "openvino.tar.gz");
  const std::size_t head = archive.size() * 2 / 5;
  // Le .part tient le début de l'archive ; le serveur en sert une copie dont ce début est faux :
  // seule une reprise à la bonne position rend un fichier dont le SHA-256 est le bon.
  write_all(core::download_path(f.catalog.layout, package.sha256), std::string_view(archive).substr(0, head));
  write_all(f.root / "serveur.bin", std::string(head, '\0') + archive.substr(head));
  core::EnginePackage served = package;
  served.url = "file://" + (f.root / "serveur.bin").generic_string();
  const fs::path library = require_installed(install_package(f.catalog, served, f.progress()));
  CHECK(read_all(library) == "bibliothèque factice\n");
  CHECK_FALSE(fs::exists(core::download_path(f.catalog.layout, package.sha256)));
}

TEST_CASE("Installation : un .part complet mais faux repart de zéro", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("part-faux");
  const core::EnginePackage package = f.openvino_package();
  write_all(core::download_path(f.catalog.layout, package.sha256), std::string(package.size + 7, 'x'));
  const fs::path library = require_installed(install_package(f.catalog, package, f.progress()));
  CHECK(read_all(library) == "bibliothèque factice\n");
  CHECK_FALSE(fs::exists(core::download_path(f.catalog.layout, package.sha256)));
}

TEST_CASE("Installation : un .part déjà complet et juste s'installe sans transfert", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("part-juste");
  const core::EnginePackage package = f.openvino_package();
  write_all(core::download_path(f.catalog.layout, package.sha256), read_all(f.root / "openvino.tar.gz"));
  core::EnginePackage unreachable = package;
  unreachable.url = "file://" + (f.root / "absent.bin").generic_string();
  const fs::path library = require_installed(install_package(f.catalog, unreachable, f.progress()));
  CHECK(read_all(library) == "bibliothèque factice\n");
  CHECK(std::ranges::none_of(f.stages, [](const std::string& s) { return s.starts_with("téléchargement"); }));
}

TEST_CASE("Installation : une archive inaccessible est un refus de transfert", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("inaccessible");
  core::EnginePackage package = f.openvino_package();
  package.url = "file://" + (f.root / "absent.bin").generic_string();
  require_refused(install_package(f.catalog, package, f.progress()), RefusalCode::DecodeFailed);
  CHECK_FALSE(installed(f.catalog, package));
}

TEST_CASE("Installation : le paquet RTX HDR s'importe dans le magasin des SDK", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("maxine");
  const fs::path source = f.root / "VideoFX";
  vendor_hdr::test::make_maxine_sdk(source);
  const core::EnginePackage package =
      f.package_of(core::EngineId::MaxineVfx, pack(source, f.root / "maxine.tar.gz"));
  const fs::path sdk_dir = f.root / "sdks";
  const auto result = install_maxine_package(f.catalog, package, sdk_dir, f.progress());
  require_done(result);
  const vendor_hdr::SdkStatus status = vendor_hdr::SdkStore(sdk_dir, vendor_hdr::SdkKind::MaxineVfx).status();
  CHECK(status.state == vendor_hdr::SdkState::Ready);
  REQUIRE(status.installed);
  CHECK(status.installed->version().text() == "1.3.0.0");
  // La copie du magasin suffit : le dossier extrait est effacé, et le paquet n'est pas « installé »
  // comme un backend (il n'a pas de module).
  CHECK_FALSE(fs::exists(core::engine_dir(f.catalog.layout, package)));
  CHECK_FALSE(installed(f.catalog, package));
  CHECK(f.fractions.back() == 1.0F);
  CHECK(f.stages.back() == "RTX HDR installé");
}

TEST_CASE("Installation : un paquet RTX HDR incomplet est refusé par le magasin", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("maxine-incomplet");
  const fs::path source = f.root / "VideoFX";
  vendor_hdr::test::make_maxine_sdk(source, {.feature = false});
  const core::EnginePackage package =
      f.package_of(core::EngineId::MaxineVfx, pack(source, f.root / "maxine.tar.gz"));
  const fs::path sdk_dir = f.root / "sdks";
  require_refused(install_maxine_package(f.catalog, package, sdk_dir, f.progress()),
                  RefusalCode::SdkRejected);
  CHECK(vendor_hdr::SdkStore(sdk_dir, vendor_hdr::SdkKind::MaxineVfx).status().state ==
        vendor_hdr::SdkState::NotInstalled);
  CHECK_FALSE(fs::exists(core::engine_dir(f.catalog.layout, package)));
}

TEST_CASE("Installation : la clé NGC vide ou mal collée est refusée avant tout transfert", "[app][install]") {
  ajp::app::play::FrenchUi french;
  Fixture f("ngc");
  const core::NgcSdkSource source{};
  for (const std::string key : {"", "clé avec espace", "clé\navec retour"}) {
    const auto result = install_maxine(f.catalog, source, key, f.root / "sdks", f.progress());
    require_refused(result, RefusalCode::FieldOutOfRange);
  }
  CHECK(f.stages.empty());
}

// Les index de tools/release/nvidia-vendor contre pypi.nvidia.com (réseau, ~0,8 Go pour une carte
// sm89) : caché, à lancer à la main (« [.vendor] »). Chaque fichier de la carte est tiré par sa
// plage, décompressé, et son SHA-256 est comparé à l'index.
TEST_CASE("Fichiers tiers : les index de NVIDIA se téléchargent et se vérifient", "[.vendor][app][install]") {
  ajp::app::play::FrenchUi french;
  // Lancé depuis le dossier de construction (build/<préréglage>) ; AJP_VENDOR_DIR en prend un autre.
  const std::string custom = env_text("AJP_VENDOR_DIR");
  const fs::path dir = !custom.empty() ? fs::path(custom) : fs::path("../../tools/release/nvidia-vendor");
  REQUIRE(fs::exists(dir / "tensorrt-linux-x64.lock"));
  Fixture f("vendor-reel");
  std::uint64_t downloaded = 0;
  for (const auto& [os, lock, arch] : {std::tuple{core::Os::Linux, "tensorrt-linux-x64.lock", "sm89"},
                                       std::tuple{core::Os::Windows, "tensorrt-windows-x64.lock", "sm89"}}) {
    const auto index = core::read_engine_index("version = 1\n" + read_all(dir / lock));
    REQUIRE(index);
    for (const auto& skipped : index->skipped) WARN("écartée : " << skipped.detail);
    const auto files = core::vendor_files_for(index->vendor_files, core::EngineId::TensorRt, os, arch);
    CAPTURE(lock);
    REQUIRE(files.size() == 7);  // libnvinfer, plugin, onnxparser, ptx, sm89, nvrtc, nvrtc-builtins
    std::uint64_t packed = 0;
    for (const auto& file : files) packed += file.packed_size;
    INFO(std::format("{} : {} fichiers, {} Mo à télécharger", lock, files.size(), packed >> 20));
    if (os != host_os()) continue;  // le téléchargement réel : ceux de cet OS seulement
    // AJP_VENDOR_OUT : garder les fichiers tirés (pour essayer le moteur depuis un dossier composé ainsi).
    const std::string keep = env_text("AJP_VENDOR_OUT");
    const fs::path out = !keep.empty() ? fs::path(keep) : f.root / "tiers";
    fs::create_directories(out);
    for (const auto& file : files) {
      const auto start = std::chrono::steady_clock::now();
      require_done(fetch_vendor_file(file, out / file.name, f.root / "tmp", f.progress()));
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      WARN(std::format("{} : {} Mo (paquet {} Mo) en {:.0f} s", file.name, file.size >> 20,
                       file.packed_size >> 20, seconds));
      CHECK(fs::file_size(out / file.name) == file.size);
      CHECK(core::sha256_file(out / file.name).value() == file.sha256);
      downloaded += file.packed_size;
    }
  }
  WARN(std::format("total transféré : {} Mo", downloaded >> 20));
}

// Le paquet TensorRT construit par package-linux.sh, installé par install_package : l'archive locale
// (file://) pour notre module, les fichiers de NVIDIA tirés pour de bon de pypi.nvidia.com. Caché :
// AJP_PACKAGE_DIR = le dossier de sortie du script (engines.lock et archives).
TEST_CASE("Paquet TensorRT du script : installation réelle en lien direct", "[.package][app][install]") {
  ajp::app::play::FrenchUi french;
  const std::string given = env_text("AJP_PACKAGE_DIR");
  if (given.empty()) SKIP("AJP_PACKAGE_DIR non défini");
  const fs::path dir(given);
  Fixture f("paquet-reel");
  const auto index = core::read_engine_index(read_all(dir / "engines.lock"));
  REQUIRE(index);
  const core::EnginePackage* found =
      core::find_package(index->packages, core::EngineId::TensorRt, f.catalog.layout.os);
  REQUIRE(found != nullptr);
  core::EnginePackage package = *found;
  package.url = "file://" + (dir / "ajp-engine-tensorrt-linux-x64.tar.gz").generic_string();
  f.catalog.packages = {package};
  f.catalog.vendor_files = index->vendor_files;
  InstallProgress progress = f.progress();
  progress.cuda_arch = "sm89";
  WARN(std::format("à télécharger pour sm89 : {} Mo", download_bytes(f.catalog, package, "sm89") >> 20));
  const auto installed_library = install_package(f.catalog, package, progress);
  REQUIRE(installed_library);
  const fs::path target = core::engine_dir(f.catalog.layout, package);
  std::string listing;
  for (const auto& entry : fs::directory_iterator(target))
    listing += std::format("{} ({} Mo) ; ", entry.path().filename().string(), fs::file_size(entry) >> 20);
  WARN("dossier installé : " + target.string());
  WARN(listing);
  CHECK(fs::exists(target / "ajp_backend_tensorrt.so"));
  CHECK(fs::exists(target / "libnvinfer.so.11"));
  CHECK(fs::exists(target / "libnvinfer_plugin.so.11"));
  CHECK(fs::exists(target / "libnvonnxparser.so.11"));
  CHECK(fs::exists(target / "libnvinfer_builder_resource_sm89.so.11.3.0"));
  CHECK_FALSE(fs::exists(target / "libnvinfer_builder_resource_sm120.so.11.3.0"));
  CHECK(fs::exists(target / "libnvrtc.so.13"));
  CHECK(fs::exists(target / "trtexec"));
  if (const std::string keep = env_text("AJP_PACKAGE_KEEP"); !keep.empty())
    fs::copy(target, keep, fs::copy_options::recursive);
}

TEST_CASE("Espace : le pic compte le décompressé et l'ancienne version, pas seulement le téléchargement",
          "[app][install][space]") {
  ajp::app::play::FrenchUi french;
  EngineCatalog catalog;
  catalog.layout = core::install_layout(host_os(), "/d");
  const auto vendor = [&](std::string name, std::uint64_t packed, std::uint64_t size, std::string arch = "") {
    core::VendorFile file;
    file.engine = core::EngineId::TensorRt;
    file.platform = host_os();
    file.name = std::move(name);
    file.packed_size = packed;
    file.size = size;
    file.arch = std::move(arch);
    catalog.vendor_files.push_back(std::move(file));
  };
  constexpr std::uint64_t kMo = 1U << 20;
  // Les chiffres de TensorRT 11.3 pour une carte sm89 : 739 Mo tirés, 1250 Mo une fois décompressés.
  vendor("libnvinfer.so.11", 306 * kMo, 663 * kMo);
  vendor("builder_resource_ptx", 206 * kMo, 230 * kMo);
  vendor("builder_resource_sm89", 157 * kMo, 180 * kMo, "sm89");
  vendor("builder_resource_sm120", 230 * kMo, 257 * kMo, "sm120");  // pas pour cette carte
  core::EnginePackage package;
  package.engine = core::EngineId::TensorRt;
  package.size = 2 * kMo;

  const InstallSpace fresh = install_space(catalog, package, "sm89");
  CHECK(fresh.download == (2 + 306 + 206 + 157) * kMo);  // sm120 ignoré
  CHECK(fresh.installed >= (663 + 230 + 180) * kMo);     // le décompressé, pas le téléchargé
  CHECK(fresh.peak > fresh.installed);                   // le fichier en cours de copie s'ajoute
  CHECK(fresh.peak > fresh.download * 15 / 10);          // bien plus que ce que l'écran annonçait
  // Une version déjà installée est gardée jusqu'à la fin : elle compte.
  const InstallSpace again = install_space(catalog, package, "sm89", fresh.installed);
  CHECK(again.peak > 2 * fresh.installed);
}

TEST_CASE("Espace : refus net avec les chiffres, avant tout téléchargement", "[app][install][space]") {
  ajp::app::play::FrenchUi french;
  const InstallSpace space{.download = 742ULL << 20, .installed = 1250ULL << 20, .peak = 1400ULL << 20};
  CHECK(check_space(space, 2ULL << 30, "/d"));  // 2 Go : assez
  const auto refused = check_space(space, 300ULL << 20, "/d");
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::InsufficientSpace);
  CHECK(refused.error().detail.contains("1.4 Go"));
  CHECK(refused.error().detail.contains("300 Mo"));
  CHECK(refused.error().detail.contains("742 Mo"));
  CHECK(format_size(742ULL << 20) == "742 Mo");
  CHECK(format_size(1536ULL << 20) == "1.5 Go");
}

TEST_CASE("Installation : sans place, rien n'est tiré et le moteur reste absent", "[app][install][space]") {
  ajp::app::play::FrenchUi french;
  Fixture f("sans-place");
  core::EnginePackage package = f.openvino_package();
  package.size = std::numeric_limits<std::uint64_t>::max() / 16;  // un paquet qu'aucun disque ne tient
  InstallProgress progress = f.progress();
  const auto installed = install_package(f.catalog, package, progress);
  REQUIRE_FALSE(installed);
  CHECK(installed.error().code == RefusalCode::InsufficientSpace);
  CHECK_FALSE(fs::exists(core::engine_dir(f.catalog.layout, package)));
}

TEST_CASE("Dossier d'installation des moteurs : le catalogue installe là, les modèles restent",
          "[app][install][space]") {
  ajp::app::play::FrenchUi french;
  const fs::path data = fs::temp_directory_path() / "ajp-catalogue-data";
  const fs::path moved = fs::temp_directory_path() / "ajp-catalogue-moteurs";
  const EngineCatalog catalog = load_catalog(data / "absent.lock", data, moved);
  const EngineCatalog plain = load_catalog(data / "absent.lock", data);
  CHECK(catalog.layout.engines.string().starts_with(moved.string()));
  CHECK(catalog.layout.downloads.string().starts_with(moved.string()));
  CHECK(catalog.layout.models == plain.layout.models);
  CHECK(plain.layout.engines.string().starts_with(data.string()));
}

TEST_CASE("Architecture : lue, sinon déduite du nom, sinon ptx — jamais toutes", "[app][install]") {
  ajp::app::play::FrenchUi french;
  CHECK(ajp::app::play::cuda_arch_from_name("NVIDIA GeForce RTX 4050 Laptop GPU") == "sm89");
  CHECK(ajp::app::play::cuda_arch_from_name("NVIDIA GeForce RTX 4070 Ti") == "sm89");
  CHECK(ajp::app::play::cuda_arch_from_name("NVIDIA GeForce RTX 3060") == "sm86");
  CHECK(ajp::app::play::cuda_arch_from_name("NVIDIA GeForce RTX 2070") == "sm75");
  CHECK(ajp::app::play::cuda_arch_from_name("NVIDIA GeForce RTX 5080") == "sm120");
  CHECK(ajp::app::play::cuda_arch_from_name("AMD Radeon RX 7900").empty());
  CHECK(ajp::app::play::effective_cuda_arch("sm90", "RTX 4050") == "sm90");
  CHECK(ajp::app::play::effective_cuda_arch("", "NVIDIA GeForce RTX 4050 Laptop GPU") == "sm89");
  CHECK(ajp::app::play::effective_cuda_arch("", "inconnue") == "ptx");
}
