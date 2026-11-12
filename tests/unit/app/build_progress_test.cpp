// Pourcentage de la préparation d'un moteur : au temps écoulé, planchers des repères de trtexec,
// durée attendue lue dans les journaux du cache.
#include "app/play/build_progress.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace ajp::app::play {
namespace {

using Catch::Matchers::WithinAbs;
constexpr std::int64_t kS = 1'000'000;

TEST_CASE("La barre avance au temps écoulé et n'atteint 100 % qu'à la fin", "[app][build]") {
  BuildProgress progress(60.0, 10 * kS);
  CHECK(progress.fraction(10 * kS) == 0.0F);
  CHECK_THAT(progress.fraction(40 * kS), WithinAbs(0.5, 1e-4));
  CHECK_THAT(progress.fraction(64 * kS), WithinAbs(0.9, 1e-4));
  const float late = progress.fraction(130 * kS);  // bien au-delà de la durée attendue
  CHECK(late > 0.95F);
  CHECK(late < 0.99F);
  CHECK(progress.fraction(1000 * kS) <= 0.99F);
  CHECK(progress.fraction(131 * kS) >= late);  // ne recule pas
  progress.finish();
  CHECK(progress.fraction(131 * kS) == 1.0F);
}

TEST_CASE("Une construction bien plus longue que prévu passe en barre sans pourcentage (B24-08)",
          "[app][build]") {
  BuildProgress progress(60.0, 10 * kS);
  CHECK_FALSE(progress.overdue(40 * kS));
  CHECK_FALSE(progress.overdue(100 * kS));  // 90 s écoulées : exactement une fois et demie
  CHECK(progress.overdue(101 * kS));
  progress.finish();
  CHECK_FALSE(progress.overdue(1000 * kS));
}

TEST_CASE("Seuls les repères sûrs de trtexec servent de plancher", "[app][build]") {
  BuildProgress progress(100.0, 0);
  progress.marker(0.10F);  // fin de la lecture du modèle
  CHECK(progress.fraction(kS) == 0.10F);
  progress.marker(0.85F);  // plan mémoire : revient à chaque sous-moteur, ignoré
  CHECK(progress.fraction(kS) == 0.10F);
  progress.marker(1.0F);  // PASSED : 99 % au plus avant le résultat
  CHECK(progress.fraction(kS) == 0.99F);
}

TEST_CASE("Un journal de construction donne le modèle, les pixels et la durée", "[app][build]") {
  const auto past = read_build_log(
      "&&&& RUNNING TensorRT.trtexec [TensorRT v110300] [b99] # /x/trtexec --onnx=/m/SPAN fp16.onnx "
      "--saveEngine=/e/model.engine --builderOptimizationLevel=5 --optShapes=input:1x3x720x1280 "
      "--skipInference\n[I] ...\n[10/19/2026-20:57:42] [I] Engine built in 60.2205 sec.\n&&&& PASSED\n");
  REQUIRE(past);
  CHECK(past->onnx == "/m/SPAN fp16.onnx");
  CHECK(past->pixels == 720 * 1280);
  CHECK_THAT(past->seconds, WithinAbs(60.2205, 1e-6));
  CHECK_FALSE(
      read_build_log("&&&& RUNNING trtexec --onnx=/m/a.onnx --optShapes=input:1x3x720x1280\n"
                     "&&&& FAILED\n"));  // pas de durée
  CHECK_FALSE(read_build_log("pas un journal"));
  // Trouvé par tests/fuzz : from_chars lit « Inf » au début d'un mot.
  CHECK_FALSE(
      read_build_log("# trtexec --onnx=/m/a.onnx --optShapes=input:1x3x720x1280\n"
                     "[I] Engine built in Inference\n"));
  CHECK_FALSE(
      read_build_log("# trtexec --onnx=/m/a.onnx --optShapes=input:1x3x720x1280\n"
                     "[I] Engine built in nan sec.\n"));
}

TEST_CASE("La durée attendue vient de la construction passée la plus proche en pixels", "[app][build]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-build-progress-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto write = [&](const char* name, const char* shape, const char* seconds) {
    std::ofstream(dir / name) << "&&&& RUNNING # trtexec --onnx=/m/span.onnx --optShapes=input:1x3x" << shape
                              << " --skipInference\nEngine built in " << seconds << " sec.\n";
  };
  write("a.engine.build.log", "720x1280", "60");
  write("b.engine.build.log", "1080x1920", "116");
  CHECK_THAT(expected_build_seconds(dir, "/m/span.onnx", 720 * 1280), WithinAbs(60.0, 1e-9));
  CHECK_THAT(expected_build_seconds(dir, "/m/span.onnx", 1080 * 1920), WithinAbs(116.0, 1e-9));
  // 960x540 : la construction de 720p, pas plus courte.
  CHECK_THAT(expected_build_seconds(dir, "/m/span.onnx", 540 * 960), WithinAbs(60.0, 1e-9));
  // 1440p : depuis 1080p, allongée en pixels^0,8.
  CHECK_THAT(expected_build_seconds(dir, "/m/span.onnx", 1440 * 2560), WithinAbs(116.0 * 1.5845, 0.1));
  // Modèle sans historique : la durée par défaut à l'échelle de 720p.
  CHECK_THAT(expected_build_seconds(dir, "/m/other.onnx", 720 * 1280), WithinAbs(kDefaultBuildSeconds, 1e-9));
  CHECK_THAT(expected_build_seconds(dir / "absent", "/m/span.onnx", 720 * 1280),
             WithinAbs(kDefaultBuildSeconds, 1e-9));
  std::filesystem::remove_all(dir);
}

TEST_CASE("Un long journal se lit par ses deux bouts", "[app][build]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-build-progress-long";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  {
    std::ofstream log(dir / "c.engine.build.log");
    log << "&&&& RUNNING # trtexec --onnx=/m/rife.onnx --optShapes=input:1x8x720x1280 --skipInference\n";
    // Un journal verbeux de 2 Mo entre la commande et la fin.
    const std::string line(1023, 'v');
    for (int i = 0; i < 2048; ++i) log << line << '\n';
    log << "Engine built in 42.5 sec.\nCreated engine with size: 12 MiB\n";
  }
  const std::vector<PastBuild> builds = past_builds(dir);
  REQUIRE(builds.size() == 1);
  CHECK(builds[0].onnx == std::filesystem::path("/m/rife.onnx"));
  CHECK(builds[0].pixels == std::int64_t{720} * 1280);
  CHECK_THAT(builds[0].seconds, WithinAbs(42.5, 1e-9));
  std::filesystem::remove_all(dir);
}

TEST_CASE("C24-06 : le journal récent du modèle est lu même dans un cache de mille journaux",
          "[app][build]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-build-progress-many";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto write = [&](const std::string& name, const char* model, const char* seconds) {
    std::ofstream(dir / name) << "&&&& RUNNING # trtexec --onnx=/m/" << model
                              << ".onnx --optShapes=input:1x3x720x1280 --skipInference\nEngine built in "
                              << seconds << " sec.\n";
    return dir / name;
  };
  const auto old = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24);
  for (int i = 0; i < 1000; ++i) {
    std::filesystem::last_write_time(write("old" + std::to_string(i) + ".build.log", "autre", "10"), old);
  }
  write("current.build.log", "span", "180");
  CHECK_THAT(expected_build_seconds(dir, "/m/span.onnx", 720 * 1280), WithinAbs(180.0, 1e-9));
  std::filesystem::remove_all(dir);
}

}  // namespace
TEST_CASE("B08-12 : le moteur RIFE chargé se reconnaît à son graphe dérivé ou à ses morceaux fusionnés",
          "[app][build]") {
  const PastBuild motion{.onnx = "/cache/rife422_lite-mouvement-t2-0123456789abcdef.onnx",
                         .width = 1920,
                         .height = 1088,
                         .pixels = 1920 * 1088,
                         .seconds = 30.0};
  // Le modèle tel que livré n'est pas ce qui a été construit : la variante mouvement l'est.
  CHECK_FALSE(built_from({.onnx = "/models/rife422_lite.onnx", .width = 1920, .height = 1088}, motion));
  CHECK(built_from(
      {.onnx = "/derived/rife422_lite-mouvement-t2-0123456789abcdef.onnx", .width = 1920, .height = 1088},
      motion));
  CHECK_FALSE(built_from(
      {.onnx = "/derived/rife422_lite-mouvement-t2-0123456789abcdef.onnx", .width = 1280, .height = 736},
      motion));
  // Le fusionné : le manifeste, et des morceaux nommés d'après lui.
  const LoadedEngine fused{.onnx = "/derived/rife422_lite-fusion1-1088x1920-0123456789abcdef-image.rifefused",
                           .width = 1920,
                           .height = 1088};
  CHECK(
      built_from(fused, PastBuild{.onnx = "/cache/rife422_lite-fusion1-1088x1920-0123456789abcdef-head.onnx",
                                  .width = 960,
                                  .height = 544}));
  CHECK_FALSE(
      built_from(fused, PastBuild{.onnx = "/cache/rife422_lite-fusion1-736x1280-0123456789abcdef-head.onnx",
                                  .width = 640,
                                  .height = 368}));
}

}  // namespace ajp::app::play
