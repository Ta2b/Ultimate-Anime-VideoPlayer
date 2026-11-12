// Les versions installées d'un moteur : paquet remplacé à version égale (A26-01), nouvelle version de
// TensorRT (A06-01), anciennes versions retirées, installation coupée entre ses deux renommages (A06-08),
// remplacement remis au lancement suivant (A06-09).
#include "app/play/engine_versions.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "core/model_index.h"

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

struct Fixture {
  explicit Fixture(std::string_view test)
      : root(fs::temp_directory_path() / std::format("ajp-versions-{}", test)) {
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);
    catalog.layout = core::install_layout(host_os(), root);
  }
  ~Fixture() {
    std::error_code error;
    fs::remove_all(root, error);
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  core::EnginePackage package(std::string_view version, char sha) const {
    return core::EnginePackage{.engine = core::EngineId::TensorRt,
                               .platform = catalog.layout.os,
                               .sha256 = std::string(64, sha),
                               .version = std::string(version)};
  }

  // Un dossier de version comme install_package le laisse : le module, puis la note.
  fs::path install(const core::EnginePackage& of, const fs::path& dir, bool note = true) const {
    fs::create_directories(dir);
    std::ofstream(dir / core::backend_library(catalog.layout, of).filename()) << "module\n";
    if (note) std::ofstream(dir / std::string(kPackageNote)) << "sha256=" << of.sha256 << "\nlicence=L\n";
    return dir;
  }
  fs::path install(const core::EnginePackage& of) const {
    return install(of, core::engine_dir(catalog.layout, of));
  }

  fs::path root;
  EngineCatalog catalog;
};

}  // namespace

TEST_CASE("Versions : absent, à jour, remplacé ou nouvelle version", "[app][install]") {
  Fixture f("etats");
  const core::EnginePackage v11 = f.package("11", 'a');
  CHECK(installed_state(f.catalog, v11).state == PackageState::Absent);

  f.install(v11);
  const InstalledState current = installed_state(f.catalog, v11);
  CHECK(current.state == PackageState::Current);
  REQUIRE(current.note);
  CHECK(current.note->license == "L");

  // Même version, autre archive dans la release : à réinstaller.
  CHECK(installed_state(f.catalog, f.package("11", 'b')).state == PackageState::Replaced);

  // L'index demande la 12, seule la 11 est là.
  const InstalledState next = installed_state(f.catalog, f.package("12", 'c'));
  CHECK(next.state == PackageState::NewVersion);
  CHECK(next.dir == core::engine_dir(f.catalog.layout, v11));
}

TEST_CASE("Versions : un module sans note est un paquet remplacé", "[app][install]") {
  Fixture f("sans-note");
  const core::EnginePackage v11 = f.package("11", 'a');
  f.install(v11, core::engine_dir(f.catalog.layout, v11), false);
  const InstalledState found = installed_state(f.catalog, v11);
  CHECK(found.state == PackageState::Replaced);
  CHECK_FALSE(found.note);
}

TEST_CASE("Versions : un fichier tiers manquant ou tronqué rend le paquet à réinstaller (B06-11)",
          "[app][install]") {
  Fixture f("fichiers-tiers");
  const core::EnginePackage v11 = f.package("11", 'a');
  const fs::path dir = f.install(v11);
  std::ofstream(dir / "tiers.bin") << "12345";
  std::ofstream(dir / std::string(kPackageNote), std::ios::app) << "fichier=5 tiers.bin\n";

  const InstalledState whole = installed_state(f.catalog, v11);
  CHECK(whole.state == PackageState::Current);
  REQUIRE(whole.note);
  REQUIRE(whole.note->files.size() == 1);
  CHECK(whole.note->files.front().first == "tiers.bin");
  CHECK(whole.note->files.front().second == 5);

  // Un antivirus l'a mis en quarantaine : le dossier a son module et sa note, mais ne se chargerait pas.
  fs::remove(dir / "tiers.bin");
  const InstalledState missing = installed_state(f.catalog, v11);
  CHECK(missing.state == PackageState::Replaced);
  CHECK(missing.missing == "tiers.bin");

  // Tronqué : même verdict.
  std::ofstream(dir / "tiers.bin") << "123";
  CHECK(installed_state(f.catalog, v11).state == PackageState::Replaced);

  // Une note d'avant (sans ligne « fichier ») n'a rien à vérifier.
  std::ofstream(dir / std::string(kPackageNote)) << "sha256=" << v11.sha256 << "\nlicence=L\n";
  CHECK(installed_state(f.catalog, v11).state == PackageState::Current);
}

TEST_CASE("Versions : la ressource de construction de la carte d'aujourd'hui manque (C37-02)",
          "[app][install]") {
  Fixture f("architecture");
  const core::EnginePackage v11 = f.package("11", 'a');
  const auto vendor = [&](std::string name, std::string arch) {
    return core::VendorFile{.engine = core::EngineId::TensorRt,
                            .platform = f.catalog.layout.os,
                            .name = std::move(name),
                            .size = 4,
                            .arch = std::move(arch)};
  };
  f.catalog.vendor_files = {vendor("libnvinfer.so", ""), vendor("builder_resource_sm86.so", "sm86"),
                            vendor("builder_resource_sm89.so", "sm89")};
  // Installé sur une RTX 3070 : la ressource sm86 seulement.
  const fs::path dir = f.install(v11);
  for (const char* name : {"libnvinfer.so", "builder_resource_sm86.so"}) {
    std::ofstream(dir / name) << "1234";
    std::ofstream(dir / std::string(kPackageNote), std::ios::app) << "fichier=4 " << name << '\n';
  }
  CHECK(installed_state(f.catalog, v11, "sm86").state == PackageState::Current);
  // La carte est maintenant une RTX 4070 : avant, « à jour », et trtexec n'avait que la ressource générique.
  const InstalledState moved = installed_state(f.catalog, v11, "sm89");
  CHECK(moved.state == PackageState::Replaced);
  CHECK(moved.missing == "builder_resource_sm89.so");
  // Sans architecture connue, rien de plus n'est vérifié.
  CHECK(installed_state(f.catalog, v11).state == PackageState::Current);
}

TEST_CASE("Versions : la version de l'index en place, les autres se retirent", "[app][install]") {
  Fixture f("retrait");
  const core::EnginePackage v10 = f.package("10", 'a');
  const core::EnginePackage v11 = f.package("11", 'b');
  const core::EnginePackage v12 = f.package("12", 'c');
  f.install(v10);
  f.install(v11);
  f.install(v12);
  const fs::path unrelated = core::engine_dir(f.catalog.layout, v12).parent_path() / "notes";
  fs::create_directories(unrelated);
  CHECK(remove_other_versions(f.catalog, v12) == 2);
  CHECK(fs::exists(core::engine_dir(f.catalog.layout, v12)));
  CHECK_FALSE(fs::exists(core::engine_dir(f.catalog.layout, v10)));
  CHECK_FALSE(fs::exists(core::engine_dir(f.catalog.layout, v11)));
  CHECK(fs::exists(unrelated));  // ni module ni note : pas une version
  CHECK_FALSE(other_installed_version(f.catalog, v12));
}

TEST_CASE("Versions : une installation coupée reprend sa copie complète (A06-08)", "[app][install]") {
  Fixture f("coupure");
  const core::EnginePackage v11 = f.package("11", 'b');
  const fs::path target = core::engine_dir(f.catalog.layout, v11);
  fs::path staging = target;
  staging += ".partiel";
  fs::path aside = target;
  aside += ".ancien";

  SECTION("le « .partiel » sans note est incomplet : la version d'avant revient") {
    f.install(v11, staging, false);
    f.install(f.package("11", 'a'), aside);
    CHECK(recover_interrupted_install(f.catalog, v11));
    CHECK(installed_state(f.catalog, v11).state == PackageState::Replaced);
    CHECK(fs::exists(staging));  // reste à jeter par la prochaine installation
  }
  SECTION("le « .partiel » avec sa note est complet : il prend la place, la copie d'avant part") {
    f.install(v11, staging);
    f.install(f.package("11", 'a'), aside);
    CHECK(recover_interrupted_install(f.catalog, v11));
    CHECK(installed_state(f.catalog, v11).state == PackageState::Current);
    CHECK_FALSE(fs::exists(aside));
  }
  SECTION("la version en place et un « .partiel » sans note : rien ne bouge") {
    f.install(v11);
    f.install(v11, staging, false);
    CHECK_FALSE(recover_interrupted_install(f.catalog, v11));
    CHECK(fs::exists(staging));
  }
  SECTION("un remplacement que Windows a refusé se fait au lancement suivant (A06-09)") {
    f.install(f.package("11", 'a'));  // le paquet d'avant, dont une DLL était chargée
    f.install(v11, staging);          // le nouveau, entier et noté
    CHECK(recover_interrupted_install(f.catalog, v11));
    CHECK(installed_state(f.catalog, v11).state == PackageState::Current);
    CHECK_FALSE(fs::exists(staging));
    CHECK_FALSE(fs::exists(aside));
  }
  SECTION("les copies de remplacement ne sont jamais une autre version") {
    f.install(v11, staging);
    CHECK_FALSE(other_installed_version(f.catalog, f.package("12", 'c')));
  }
}
