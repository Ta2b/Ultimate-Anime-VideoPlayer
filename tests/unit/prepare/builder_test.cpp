// Construction par un faux trtexec (helper.cpp copié en « trtexec-<mode> ») :
// ligne de commande réelle, progression, cache, et chaque façon d'échouer.
#include "prepare/builder.h"

#include <algorithm>
#include <array>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "prepare/key_lock.h"
#include "prepare/sha256.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;
using ajp::gpu::ShapeSet;
using ajp::gpu::TensorType;
using namespace std::chrono_literals;

namespace {

ShapeSet fixed(std::int64_t channels, std::int64_t h, std::int64_t w) {
  const ajp::gpu::Shape shape{{1, channels, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

struct Steps {
  std::vector<std::pair<float, std::string>> seen;
  ajp::gpu::Progress progress() {
    return {.report = [this](float f, const std::string& stage) { seen.emplace_back(f, stage); }, .stop = {}};
  }
};

class Bench {
 public:
  explicit Bench(std::string_view mode, const test::FakeModel& graph = {},
                 TensorType precision = TensorType::F16) {
    const auto fake =
        dir_ / "bin" / ("trtexec-" + std::string(mode) + test::helper_path().extension().string());
    test::fs::create_directories(fake.parent_path());
    test::fs::copy_file(test::helper_path(), fake);
    test::fs::permissions(fake, test::fs::perms::owner_all, test::fs::perm_options::add);
    const std::string onnx = test::onnx(graph);
    test::write_file(dir_ / "modeles/modele.onnx", onnx);
    model = {.id = "modele",
             .onnx_path = dir_ / "modeles/modele.onnx",
             .sha256 = sha256_hex(onnx),
             .precision = precision};
    config = {.trtexec = fake,
              .cache_dir = dir_ / "moteurs",
              .runtime = {.major = 11,
                          .minor = 3,
                          .patch = 0,
                          .headers_build = 99,
                          .cuda_runtime = 13000,
                          .cuda_driver = 13000},
              .device = {.name = "RTX de test", .sm_major = 8, .sm_minor = 9, .uuid = {}},
              .timeout = 10s};
  }

  // Ce que la construction de shapes produirait, calculé sans la lancer.
  BuildPlan plan(const ShapeSet& shapes) const {
    return plan_build(config, model, read_model(model, "input").value(), shapes).value();
  }
  // Lancements de trtexec, comptés par le faux trtexec à côté du modèle.
  std::size_t builder_calls() const { return test::read_lines(model.onnx_path.string() + ".calls").size(); }
  std::vector<std::string> timing_cache() const { return test::read_lines(device_timing_cache(config)); }
  std::vector<std::string> logged_args(const ShapeSet& shapes) const {
    std::vector<std::string> args;
    for (const std::string& line : test::read_lines(plan(shapes).log_file)) {
      if (line.starts_with("[I] arg ")) args.push_back(line.substr(8));
    }
    return args;
  }
  const test::fs::path& cache() const { return config.cache_dir; }

  test::TempDir dir_;
  ajp::gpu::ModelSpec model;
  TrtexecConfig config;
};

const ShapeSet kHd = fixed(3, 1080, 1920);

}  // namespace

TEST_CASE("TrtexecBuilder : construit, publie, puis réutilise le moteur", "[prepare][builder]") {
  Bench bench("ok");
  const TrtexecBuilder builder(bench.config);
  Steps steps;
  auto progress = steps.progress();
  const auto artifact = builder.build(bench.model, kHd, progress);
  REQUIRE(artifact);
  CHECK(test::read_file(artifact->path) == "moteur factice pour " + bench.model.onnx_path.string());
  CHECK(artifact->path.parent_path() == bench.cache());
  CHECK(test::stage_count(bench.cache()) == 0);

  // Clé du fork : contenu ONNX, réglages exacts, runtime, appareil.
  const auto content = onnx_source_key(test::as_bytes(test::read_file(bench.model.onnx_path)),
                                       bench.model.onnx_path.parent_path(), {});
  const auto expected =
      engine_name({.content = content.value(),
                   .settings = "--builderOptimizationLevel=5 --optShapes=input:1x3x1080x1920 --skipInference",
                   .runtime = bench.config.runtime.token(),
                   .device = bench.config.device.token()});
  CHECK(artifact->cache_key == expected.value());

  const auto args = bench.logged_args(kHd);
  REQUIRE(args.size() == 8);
  CHECK(args[0] == "--onnx=" + bench.model.onnx_path.string());
  CHECK(args[1].starts_with("--saveEngine=" + (bench.cache() / ".aji-build-").string()));
  CHECK(std::vector(args.begin() + 2, args.end() - 3) ==
        std::vector<std::string>{"--builderOptimizationLevel=5", "--optShapes=input:1x3x1080x1920",
                                 "--skipInference"});
  // Le rapport des couches, écrit dans le dossier de travail puis publié à côté du moteur.
  CHECK(args[5] == "--profilingVerbosity=detailed");
  CHECK(args[6].starts_with("--exportLayerInfo=" + (bench.cache() / ".aji-build-").string()));
  CHECK(layer_counts(test::read_file(bench.plan(kHd).layer_info)).reformats == 2);
  // trtexec écrit dans une copie du timing cache, publiée après la construction.
  CHECK(args.back().starts_with("--timingCacheFile=" + (bench.cache() / ".aji-build-").string()));
  CHECK(args.back().ends_with("timing.cache"));
  CHECK(bench.timing_cache() == std::vector<std::string>{"passe"});

  REQUIRE_FALSE(steps.seen.empty());
  CHECK(std::ranges::is_sorted(steps.seen, {}, &std::pair<float, std::string>::first));
  CHECK(steps.seen.back() == std::pair<float, std::string>{1.0F, "Terminé"});

  Steps again;
  auto second = again.progress();
  CHECK(builder.build(bench.model, kHd, second).value().path == artifact->path);
  CHECK(bench.builder_calls() == 1);
  CHECK(again.seen.back().second == "Moteur en cache");
}

TEST_CASE("TrtexecBuilder : sans rapport des couches, le moteur est publié quand même",
          "[prepare][builder]") {
  Bench bench("no-layers");
  Steps steps;
  auto progress = steps.progress();
  REQUIRE(TrtexecBuilder(bench.config).build(bench.model, kHd, progress));
  CHECK(test::fs::exists(bench.plan(kHd).engine));
  CHECK_FALSE(test::fs::exists(bench.plan(kHd).layer_info));
  CHECK(test::stage_count(bench.cache()) == 0);
}

TEST_CASE("TrtexecBuilder : RIFE garde ses îlots FP32, drapeaux TRT 11 retirés", "[prepare][builder]") {
  const test::FakeModel rife{.inputs = {{.dims = {1, 11, -1, -1}}}, .ops = {"Conv", "GridSample"}};
  Bench bench("ok", rife);
  const ShapeSet shapes = fixed(11, 1088, 1920);
  Steps steps;
  auto progress = steps.progress();
  REQUIRE(TrtexecBuilder(bench.config).build(bench.model, shapes, progress));
  const auto args = bench.logged_args(shapes);
  // Petit graphe : niveau 5 (kTunedMaxNodes) ; les drapeaux d'E/S et de tactiques retirés.
  CHECK(std::vector(args.begin() + 2, args.end() - 3) ==
        std::vector<std::string>{"--builderOptimizationLevel=5", "--stronglyTyped",
                                 "--optShapes=input:1x11x1088x1920", "--skipInference"});
  CHECK(bench.plan(shapes).command.removed.size() == 3);
}

TEST_CASE("TrtexecBuilder : chaque échec de trtexec devient un refus, rien n'est publié",
          "[prepare][builder]") {
  const auto [mode, code, timeout] =
      GENERATE(table<std::string_view, RefusalCode, std::chrono::milliseconds>({
          {"fail", RefusalCode::BuilderFailed, 10s},
          {"crash", RefusalCode::BuilderCrashed, 10s},
          {"hang", RefusalCode::BuilderTimeout, 300ms},
          {"empty", RefusalCode::BuilderFailed, 10s},
      }));
  Bench bench(mode);
  bench.config.timeout = timeout;
  test::write_file(device_timing_cache(bench.config), "ancien\n");
  Steps steps;
  auto progress = steps.progress();
  const auto result = TrtexecBuilder(bench.config).build(bench.model, kHd, progress);
  CHECK(test::refusal(result) == code);
  CHECK_FALSE(test::fs::exists(bench.plan(kHd).engine));
  CHECK(test::stage_count(bench.cache()) == 0);
  // Un trtexec qui plante en écrivant son timing cache ne touche pas au cache persistant.
  CHECK(bench.timing_cache() == std::vector<std::string>{"ancien"});
  if (mode == "fail") {
    CHECK(result.error().detail.find("fausse erreur de construction") != std::string::npos);
    CHECK(test::read_lines(bench.plan(kHd).log_file).back().starts_with("[ajp] "));
  }
}

TEST_CASE("TrtexecBuilder : annulation pendant la construction", "[prepare][builder]") {
  Bench bench("hang");
  std::stop_source stop;
  ajp::gpu::Progress progress{.report =
                                  [&](float, const std::string& stage) {
                                    if (stage == "Choix des tactiques") stop.request_stop();
                                  },
                              .stop = stop.get_token()};
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::Cancelled);
  CHECK(test::stage_count(bench.cache()) == 0);
}

TEST_CASE("TrtexecBuilder : le modèle est vérifié avant de lancer trtexec", "[prepare][builder]") {
  Steps steps;
  auto progress = steps.progress();
  SECTION("SHA-256 différent") {
    Bench bench("ok");
    bench.model.sha256 = sha256_hex("autre chose");
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
          RefusalCode::ModelMismatch);
  }
  SECTION("précision annoncée différente du graphe") {
    Bench bench("ok", {}, TensorType::F32);
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
          RefusalCode::ModelMismatch);
  }
  SECTION("RIFE en FP32") {
    Bench bench("ok", {.inputs = {{.elem_type = 1, .dims = {1, 11, -1, -1}}}, .ops = {"GridSample"}},
                TensorType::F32);
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, fixed(11, 64, 64), progress)) ==
          RefusalCode::ModelMismatch);
  }
  SECTION("canaux fixés par le graphe") {
    Bench bench("ok");
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, fixed(4, 64, 64), progress)) ==
          RefusalCode::ModelMismatch);
  }
  SECTION("TensorRT 10") {
    Bench bench("ok");
    bench.config.runtime.major = 10;
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
          RefusalCode::FieldOutOfRange);
  }
  CHECK(steps.seen.empty());
}

TEST_CASE("TrtexecBuilder : trtexec absent", "[prepare][builder]") {
  Bench bench("ok");
  bench.config.trtexec = bench.dir_ / "absent";
  Steps steps;
  auto progress = steps.progress();
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::BuilderFailed);
  CHECK(test::stage_count(bench.cache()) == 0);
}

TEST_CASE("TrtexecBuilder : ONNX statique, aucun profil de formes", "[prepare][builder]") {
  Bench bench("ok", {.inputs = {{.dims = {1, 3, 720, 1280}}}});
  const ShapeSet shapes = fixed(3, 720, 1280);
  Steps steps;
  auto progress = steps.progress();
  REQUIRE(TrtexecBuilder(bench.config).build(bench.model, shapes, progress));
  const auto args = bench.logged_args(shapes);
  CHECK(std::vector(args.begin() + 2, args.end() - 3) ==
        std::vector<std::string>{"--builderOptimizationLevel=5", "--skipInference"});
}

TEST_CASE("TrtexecBuilder : graphe statique à plusieurs entrées nommées autrement", "[prepare][builder]") {
  // Le graphe « fusion » de GMFSS, ses tailles fixées : half0, half1, warped…, aucun « input ».
  Steps steps;
  auto progress = steps.progress();
  const ShapeSet shapes = fixed(3, 96, 128);
  SECTION("statique : sa première entrée sert aux formes, aucun profil") {
    Bench bench("ok", {.inputs = {{.name = "half0", .dims = {1, 3, 96, 128}},
                                  {.name = "half1", .dims = {1, 3, 96, 128}}}});
    REQUIRE(TrtexecBuilder(bench.config).build(bench.model, shapes, progress));
    const auto args = bench.logged_args(shapes);
    CHECK(std::vector(args.begin() + 2, args.end() - 3) ==
          std::vector<std::string>{"--builderOptimizationLevel=5", "--skipInference"});
  }
  SECTION("une entrée d'une autre précision est refusée") {
    Bench bench("ok", {.inputs = {{.name = "half0", .dims = {1, 3, 96, 128}},
                                  {.name = "half1", .elem_type = 1, .dims = {1, 3, 96, 128}}}});
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, shapes, progress)) ==
          RefusalCode::ModelMismatch);
  }
  SECTION("dynamique : l'entrée doit porter le nom attendu") {
    Bench bench("ok", {.inputs = {{.name = "frames", .dims = {1, 6, -1, -1}}}});
    CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, fixed(6, 64, 64), progress)) ==
          RefusalCode::ModelMismatch);
  }
}

#ifndef _WIN32
TEST_CASE("TrtexecBuilder : un poids externe qui sort du dossier par un lien symbolique est refusé",
          "[prepare][builder]") {
  Bench bench("ok", {.external = {"poids.bin"}});
  Steps steps;
  auto progress = steps.progress();
  test::TempDir outside;
  test::write_file(outside.path() / "secret.bin", "hors du dossier");
  test::fs::create_symlink(outside.path() / "secret.bin", bench.model.onnx_path.parent_path() / "poids.bin");
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::ModelMismatch);
}
#endif

TEST_CASE("TrtexecBuilder : la vérification du moteur peut refuser sa publication", "[prepare][builder]") {
  Bench bench("ok");
  const TrtexecBuilder builder(bench.config, [](const test::fs::path&) -> std::expected<void, ajp::Refusal> {
    return std::unexpected(ajp::Refusal{RefusalCode::BuilderFailed, "désérialisation impossible"});
  });
  Steps steps;
  auto progress = steps.progress();
  CHECK(test::refusal(builder.build(bench.model, kHd, progress)) == RefusalCode::BuilderFailed);
  CHECK_FALSE(test::fs::exists(bench.plan(kHd).engine));
}

TEST_CASE("TrtexecBuilder : un modèle modifié pendant la construction n'est pas publié",
          "[prepare][builder]") {
  Bench bench("mutate");
  Steps steps;
  auto progress = steps.progress();
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::ModelMismatch);
  CHECK(test::stage_count(bench.cache()) == 0);
  CHECK(std::ranges::none_of(test::fs::directory_iterator(bench.cache()),
                             [](const auto& entry) { return entry.path().extension() == ".engine"; }));
}

TEST_CASE("TrtexecBuilder : un ONNX qui n'est pas un fichier est refusé sans exception",
          "[prepare][builder]") {
  Bench bench("ok");
  test::fs::remove(bench.model.onnx_path);
  test::fs::create_directories(bench.model.onnx_path);  // un dossier à la place du modèle
  Steps steps;
  auto progress = steps.progress();
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::ModelUnreadable);
  test::fs::remove(bench.model.onnx_path);
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::ModelUnreadable);
}

TEST_CASE("TrtexecBuilder : rebuild remplace le moteur refusé par load(), pas un autre",
          "[prepare][builder]") {
  Bench bench("ok");
  const TrtexecBuilder builder(bench.config);
  Steps steps;
  auto progress = steps.progress();
  const auto first = builder.build(bench.model, kHd, progress);
  REQUIRE(first);
  test::write_file(first->path, "moteur tronqué par une coupure");
  const std::string rejected = sha256_file(first->path).value();

  // Un autre lecteur a refusé d'autres octets : ceux en place se reprennent.
  REQUIRE(builder.rebuild(bench.model, kHd, progress, sha256_hex("autres octets")));
  CHECK(bench.builder_calls() == 1);
  CHECK(test::read_file(first->path) == "moteur tronqué par une coupure");

  const auto second = builder.rebuild(bench.model, kHd, progress, rejected);
  REQUIRE(second);
  CHECK(second->path == first->path);
  CHECK(bench.builder_calls() == 2);
  CHECK(test::read_file(second->path) == "moteur factice pour " + bench.model.onnx_path.string());
  CHECK(test::stage_count(bench.cache()) == 0);
  CHECK(bench.timing_cache() == std::vector<std::string>{"passe", "passe"});  // reçu, puis enrichi

  // Un échec de la reconstruction garde le moteur en place.
  test::write_file(first->path, "encore tronqué");
  const TrtexecBuilder refusing(bench.config, [](const test::fs::path&) -> std::expected<void, ajp::Refusal> {
    return std::unexpected(ajp::Refusal{RefusalCode::BuilderFailed, "désérialisation impossible"});
  });
  CHECK(test::refusal(refusing.rebuild(bench.model, kHd, progress, sha256_file(first->path).value())) ==
        RefusalCode::BuilderFailed);
  CHECK(test::read_file(first->path) == "encore tronqué");
  CHECK(test::refusal(builder.rebuild(bench.model, kHd, progress, "pas un sha")) ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("TrtexecBuilder : deux demandes simultanées, un seul trtexec", "[prepare][builder]") {
  Bench bench("slow");
  const TrtexecBuilder builder(bench.config);
  std::vector<test::fs::path> paths(2);
  {
    std::vector<std::jthread> threads;
    for (std::size_t i = 0; i < paths.size(); ++i) {
      threads.emplace_back([&, i] {
        Steps steps;
        auto progress = steps.progress();
        if (auto artifact = builder.build(bench.model, kHd, progress)) paths[i] = artifact->path;
      });
    }
  }
  CHECK_FALSE(paths[0].empty());
  CHECK(paths[0] == paths[1]);
  CHECK(bench.builder_calls() == 1);
}

TEST_CASE("TrtexecBuilder : un timing cache par GPU, partagé par les tailles et les modèles",
          "[prepare][builder]") {
  Bench bench("ok");
  const TrtexecBuilder builder(bench.config);
  Steps steps;
  auto progress = steps.progress();
  REQUIRE(builder.build(bench.model, kHd, progress));
  REQUIRE(builder.build(bench.model, fixed(3, 720, 1280), progress));
  // La seconde taille a reçu les temps de la première, puis ajouté les siens.
  CHECK(bench.timing_cache() == std::vector<std::string>{"passe", "passe"});
  const auto cache = device_timing_cache(bench.config);
  CHECK(cache.parent_path() == bench.cache());
  CHECK(cache.filename().string().starts_with("aji-timing-"));
  CHECK(std::ranges::none_of(test::fs::directory_iterator(bench.cache()), [](const auto& entry) {
    return entry.path().string().ends_with(".engine.timing.cache");
  }));

  // Un autre GPU, ou une autre version de TensorRT : un autre cache.
  TrtexecConfig other = bench.config;
  other.device.uuid[0] = 1;
  CHECK(device_timing_cache(other) != cache);
  other = bench.config;
  other.runtime.minor = 4;
  CHECK(device_timing_cache(other) != cache);
}

TEST_CASE("TrtexecBuilder : un trtexec à la fois par GPU, les temps de chacun gardés", "[prepare][builder]") {
  Bench bench("slow");
  test::write_file(bench.dir_ / "modeles/autre.onnx", test::onnx({.ops = {"Conv", "PRelu"}}));
  const ajp::gpu::ModelSpec other{.id = "autre",
                                  .onnx_path = bench.dir_ / "modeles/autre.onnx",
                                  .sha256 = sha256_file(bench.dir_ / "modeles/autre.onnx").value(),
                                  .precision = TensorType::F16};
  const TrtexecBuilder builder(bench.config);
  // Pas de std::vector<bool> : ses éléments partagent un mot, que les deux threads écriraient.
  std::array<bool, 2> built{};
  {
    std::vector<std::jthread> threads;
    for (std::size_t i = 0; i < built.size(); ++i) {
      threads.emplace_back([&, i] {
        Steps steps;
        auto progress = steps.progress();
        built[i] = builder.build(i == 0 ? bench.model : other, kHd, progress).has_value();
      });
    }
  }
  CHECK(built == std::array<bool, 2>{true, true});
  CHECK(test::read_lines(bench.cache() / "trtexec.runs") ==
        std::vector<std::string>{"début", "fin", "début", "fin"});
  CHECK(bench.timing_cache() == std::vector<std::string>{"passe", "passe"});
}

TEST_CASE("TrtexecBuilder : l'attente du GPU se dit et s'annule", "[prepare][builder]") {
  Bench bench("ok");
  test::fs::create_directories(bench.cache());
  test::fs::path lock_file = device_timing_cache(bench.config);
  lock_file += ".build-lock";
  std::stop_source stop;
  std::vector<std::string> stages;
  ajp::gpu::Progress progress{.report =
                                  [&](float, const std::string& stage) {
                                    stages.push_back(stage);
                                    if (stage == "En attente d'une autre construction sur ce GPU")
                                      stop.request_stop();
                                  },
                              .stop = stop.get_token()};
  auto held = KeyLock::acquire(lock_file);
  REQUIRE(held);
  CHECK(test::refusal(TrtexecBuilder(bench.config).build(bench.model, kHd, progress)) ==
        RefusalCode::Cancelled);
  CHECK(std::ranges::find(stages, "En attente d'une autre construction sur ce GPU") != stages.end());
  CHECK(bench.builder_calls() == 0);
  CHECK(test::stage_count(bench.cache()) == 0);
}
