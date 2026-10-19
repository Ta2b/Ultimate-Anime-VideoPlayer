// Installation des modèles et des moteurs : téléchargement, SHA-256 de l'index, renommage
// atomique, contre le serveur HTTP local du test.
#include "app/install.h"

#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include "http_server.h"
#include "support.h"

using namespace ajp;
using namespace ajp::app;
using namespace ajp::app::test;
using namespace std::chrono_literals;

namespace {

DownloadOptions local_options() {
  return DownloadOptions{.connect_timeout = 2'000ms, .attempts = 2, .retry_delay = 10ms, .protocols = "http"};
}

const Downloader& downloader() {
  static const Downloader instance = *Downloader::create();
  return instance;
}

struct Fixture {
  TempDir dir;
  HttpServer server;
  core::InstallLayout layout = core::install_layout(core::Os::Linux, dir.path());
  std::string model_body = test_bytes(70'000, 7);
  std::string engine_body = test_bytes(50'000, 8);
  core::ModelEntry model{.id = "2x_AnimeJaNai_V3_Compact",
                         .sha256 = core::sha256_hex(model_body),
                         .size = model_body.size(),
                         .license = "CC-BY-NC-SA-4.0",
                         .profiles = {1, 2}};
  core::EnginePackage engine{.engine = core::EngineId::TensorRt,
                             .platform = core::Os::Linux,
                             .sha256 = core::sha256_hex(engine_body),
                             .size = engine_body.size(),
                             .version = "11.3.0"};

  Fixture() {
    server.serve("/model.onnx", Route{.body = model_body});
    server.serve("/engine.so", Route{.body = engine_body});
    model.url = server.url("/model.onnx");
    engine.url = server.url("/engine.so");
  }
  std::vector<InstallItem> items() const { return {engine_item(layout, engine), model_item(layout, model)}; }
};

}  // namespace

TEST_CASE("Installation : modèle et moteur vérifiés puis mis en place", "[app][install]") {
  Fixture f;
  std::vector<InstallProgress> seen;
  const auto record = [&](const InstallProgress& p) { seen.push_back(p); };
  const std::vector<InstallItem> items = f.items();
  REQUIRE(install_all(downloader(), f.layout, items, local_options(), record));

  const fs::path library = f.dir / "engines" / "tensorrt" / "11.3.0" / "ajp_backend_tensorrt.so";
  CHECK(read_file(library) == f.engine_body);
  CHECK(read_file(f.dir / "models" / "2x_AnimeJaNai_V3_Compact.onnx") == f.model_body);
  CHECK(fs::is_empty(f.dir / "downloads"));  // les .part sont devenus les fichiers installés
  CHECK_FALSE(fs::exists(f.dir / "engines" / "tensorrt" / "11.3.0.staging"));
  for (const InstallItem& item : items) CHECK(is_installed(item));

  REQUIRE_FALSE(seen.empty());
  CHECK(seen.front().item == 0);
  CHECK(seen.front().label == "moteur tensorrt 11.3.0");
  CHECK(seen.back().item == 1);
  CHECK(seen.back().done == seen.back().total);
  CHECK(seen.back().total == f.engine_body.size() + f.model_body.size());

  // Déjà installés : rien ne se retélécharge.
  REQUIRE(install_all(downloader(), f.layout, items, local_options()));
  CHECK(f.server.hits("/engine.so") == 1);
  CHECK(f.server.hits("/model.onnx") == 1);
}

TEST_CASE("Installation : un SHA-256 différent de l'index est refusé et rien n'est installé",
          "[app][install]") {
  Fixture f;
  f.model.sha256 = std::string(64, 'a');
  const auto result = install_all(downloader(), f.layout, f.items(), local_options());
  REQUIRE_FALSE(result);
  CHECK(result.error().code == RefusalCode::HashMismatch);
  CHECK(result.error().detail.starts_with("modèle 2x_AnimeJaNai_V3_Compact"));
  CHECK_FALSE(fs::exists(core::model_path(f.layout, f.model)));
  CHECK_FALSE(fs::exists(core::download_path(f.layout, f.model.sha256)));  // un .part faux ne se reprend pas
  CHECK(is_installed(engine_item(f.layout, f.engine)));                    // ce qui précède reste installé
}

TEST_CASE("Installation : un dossier de version incomplet est remplacé d'un coup", "[app][install]") {
  Fixture f;
  const InstallItem item = engine_item(f.layout, f.engine);
  write_file(item.target, "tronqué");
  write_file(item.target.parent_path() / "reste.txt", "ancien");
  CHECK_FALSE(is_installed(item));
  REQUIRE(install_all(downloader(), f.layout, std::vector{item}, local_options()));
  CHECK(read_file(item.target) == f.engine_body);
  CHECK_FALSE(fs::exists(item.target.parent_path() / "reste.txt"));

  REQUIRE(uninstall_engine(f.layout, core::EngineId::TensorRt));
  CHECK_FALSE(fs::exists(f.dir / "engines" / "tensorrt"));
  CHECK(uninstall_engine(f.layout, core::EngineId::OpenVino));  // absent : rien à faire
}

TEST_CASE("Modèles manquants : ceux des profils, pas encore installés", "[app][install]") {
  Fixture f;
  core::ModelEntry unused = f.model;
  unused.id = "modele_sans_profil";
  unused.profiles.clear();
  const std::vector<core::ModelEntry> lock{f.model, unused};
  CHECK(missing_models(f.layout, lock) == std::vector{f.model});
  write_file(core::model_path(f.layout, f.model), f.model_body);
  CHECK(missing_models(f.layout, lock).empty());
}
