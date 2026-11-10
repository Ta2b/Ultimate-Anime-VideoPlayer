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
