// Exécution des commandes sur un système factice et un dossier de données temporaire.
#include "app/commands.h"

#include <algorithm>
#include <chrono>
#include <sstream>

#include <catch2/catch_test_macros.hpp>

#include "http_server.h"
#include "support.h"

using namespace ajp;
using namespace ajp::app;
using namespace ajp::app::test;
using namespace std::chrono_literals;

namespace {

struct Fixture {
  TempDir dir;
  FakeLoader loader;
  FakeVulkan vulkan{.devices = {rtx_4070_ti()}};
  FakeVulkanScope scope{loader, vulkan};
  std::ostringstream out;
  std::ostringstream err;
  AppContext context;

  Fixture() {
    add_fake_cuda(loader, FakeCuda::Cuda13_4);
    context = AppContext{.os = core::Os::Linux,
                         .paths = *data_paths_in(dir / "data", dir / "res"),
                         .loader = &loader,
                         .out = &out,
                         .err = &err,
                         .download = DownloadOptions{.attempts = 1, .protocols = "http"}};
  }
  std::expected<void, Refusal> run_args(std::vector<std::string_view> args) {
    auto invocation = parse_invocation(args);
    if (!invocation) return std::unexpected(invocation.error());
    return run_command(*invocation, context);
  }
};

const Downloader& downloader() {
  static const Downloader instance = *Downloader::create();
  return instance;
}

core::SystemInfo rtx_system() {
  return core::SystemInfo{.os = core::Os::Linux,
                          .gpus = {core::GpuInfo{.vendor = core::GpuVendor::Nvidia, .tensor_cores = true}}};
}

}  // namespace

TEST_CASE("Commandes : aide, version et refus nommés des commandes à venir", "[app][commands]") {
  Fixture f;
  REQUIRE(f.run_args({"--help"}));
  CHECK(f.out.str().find("Commandes :") != std::string::npos);
  CHECK(f.out.str().find(utf8(f.dir / "data")) != std::string::npos);  // défaut de --data-dir

  f.out.str("");
  REQUIRE(f.run_args({"--version"}));
  CHECK(f.out.str().starts_with("ajp "));

  const auto missing = f.run_args({"play", (f.dir / "absent.mkv").string()});
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::MediaUnreadable);

  write_file(f.dir / "episode.mkv", "pas une vraie vidéo");
  const auto play = f.run_args({(f.dir / "episode.mkv").string()});
  REQUIRE_FALSE(play);
  CHECK(play.error().code == RefusalCode::CommandUnavailable);

  write_file(f.dir / "sortie.mkv", "existe");
  const auto exported =
      f.run_args({"export", (f.dir / "episode.mkv").string(), (f.dir / "sortie.mkv").string()});
  CHECK(exported.error().code == RefusalCode::FileUnwritable);
  const auto overwrite = f.run_args(
      {"export", "--overwrite", (f.dir / "episode.mkv").string(), (f.dir / "sortie.mkv").string()});
  CHECK(overwrite.error().code == RefusalCode::CommandUnavailable);

  // run() affiche le refus sur err et rend le code de sortie.
  const auto invocation = parse_invocation(std::vector<std::string_view>{"bench", "absent.mkv"});
  REQUIRE(invocation);
  CHECK(run(*invocation, f.context) == kExitRefused);
  CHECK(f.err.str().starts_with("ajp : MediaUnreadable : absent.mkv"));
}

TEST_CASE("ajp engines : le GPU détecté, puis chaque moteur avec badges, état et taille", "[app][commands]") {
  Fixture f;
  write_file(f.context.paths.engines_lock(),
             "version = 1\n[[engine]]\nid = \"tensorrt\"\nplatform = \"linux\"\n"
             "url = \"https://engines.example/trt.so\"\nsha256 = \"" +
                 std::string(64, 'a') + "\"\nsize = 1288490189\nversion = \"11.3.0\"\n");
  REQUIRE(f.run_args({"engines"}));
  const std::string text = f.out.str();
  INFO(text);
  CHECK(text.find("GPU 0 (rendu) : NVIDIA GeForce RTX 4070 Ti") != std::string::npos);
  CHECK(text.find("Pilote CUDA : 13.4 (libcuda.so.1)") != std::string::npos);
  CHECK(text.find("Moteurs (présélection pour ce GPU)") != std::string::npos);
  CHECK(text.find("[x] TensorRT") != std::string::npos);
  CHECK(text.find("Recommandé, Seulement NVIDIA") != std::string::npos);
  CHECK(text.find("à télécharger, 1,2 Gio") != std::string::npos);
  CHECK(text.find("[ ] MIGraphX") != std::string::npos);
  CHECK(text.find("ajp import-sdk") != std::string::npos);
  CHECK_FALSE(fs::exists(f.context.paths.settings()));  // lister n'écrit rien
}

TEST_CASE("ajp engines install sans index : la sélection est enregistrée, rien ne se télécharge",
          "[app][commands]") {
  Fixture f;
  REQUIRE(f.run_args({"engines", "install"}));
  CHECK(f.out.str().find("Rien à télécharger") != std::string::npos);
  CHECK(f.out.str().find("Maxine VFX (TrueHDR) : téléchargez le SDK") != std::string::npos);
  const auto settings = core::load_settings(f.context.paths.settings());
  REQUIRE(settings);
  CHECK(settings->first_run_done);
  CHECK(settings->chosen_engines == std::vector{core::EngineId::Vulkan, core::EngineId::MaxineVfx});

  // Un moteur nommé sans paquet est refusé, sans toucher aux réglages.
  const auto named = f.run_args({"engines", "install", "tensorrt"});
  REQUIRE_FALSE(named);
  CHECK(named.error().code == RefusalCode::FieldOutOfRange);
  write_file(f.context.paths.models_lock(), "version = 2\n");
  CHECK(f.run_args({"engines"}).error().code == RefusalCode::FormatVersion);
}

TEST_CASE("Plan validé : téléchargé, installé, puis noté et préféré dans settings.toml", "[app][commands]") {
  Fixture f;
  HttpServer server;
  const std::string library = test_bytes(40'000, 3);
  server.serve("/trt.so", Route{.body = library});
  const core::EnginePackage trt{.engine = core::EngineId::TensorRt,
                                .platform = core::Os::Linux,
                                .url = server.url("/trt.so"),
                                .sha256 = core::sha256_hex(library),
                                .size = library.size(),
                                .version = "11.3.0"};
  const EngineChooser chooser(ChooserInput{
      .os = core::Os::Linux, .system = rtx_system(), .packages = {trt}, .free_bytes = 1ULL << 40});
  const auto plan = chooser.confirm();
  REQUIRE(plan);

  SECTION("tout réussit") {
    std::vector<InstallProgress> seen;
    const auto settings = apply_plan(*plan, rtx_system(), f.context, downloader(),
                                     [&](const InstallProgress& p) { seen.push_back(p); });
    REQUIRE(settings);
    CHECK(settings->installed_engines == std::vector{core::EngineId::Vulkan, core::EngineId::TensorRt});
    CHECK(settings->preferred[0] == core::EngineId::TensorRt);
    CHECK(core::load_settings(f.context.paths.settings()) == *settings);
    CHECK(fs::exists(core::backend_library(f.context.paths.layout(core::Os::Linux), trt)));
    REQUIRE_FALSE(seen.empty());
    CHECK(seen.back().done == library.size());
  }
  SECTION("un modèle introuvable : refus, mais le moteur installé est noté") {
    ChooserPlan with_model = *plan;
    server.serve("/absent.onnx", Route{.status = 404});
    with_model.models.push_back(core::ModelEntry{.id = "absent",
                                                 .url = server.url("/absent.onnx"),
                                                 .sha256 = std::string(64, 'b'),
                                                 .size = 10,
                                                 .profiles = {1}});
    const auto settings = apply_plan(with_model, rtx_system(), f.context, downloader(), {});
    REQUIRE_FALSE(settings);
    CHECK(settings.error().code == RefusalCode::DownloadFailed);
    CHECK(settings.error().detail.starts_with("modèle absent"));
    const auto saved = core::load_settings(f.context.paths.settings());
    REQUIRE(saved);
    CHECK(std::ranges::contains(saved->installed_engines, core::EngineId::TensorRt));
  }
}

TEST_CASE("ajp import-sdk : un dossier qui n'est pas un SDK est refusé, rien n'est noté", "[app][commands]") {
  Fixture f;
  write_file(f.dir / "faux-sdk" / "LISEZMOI.txt", "rien");
  const auto result = f.run_args({"import-sdk", (f.dir / "faux-sdk").string()});
  REQUIRE_FALSE(result);
  CHECK(result.error().code == RefusalCode::SdkRejected);
  CHECK_FALSE(fs::exists(f.context.paths.settings()));

  f.context.os = core::Os::MacOs;
  CHECK(f.run_args({"import-sdk", (f.dir / "faux-sdk").string()}).error().code ==
        RefusalCode::HdrUnavailable);
}
