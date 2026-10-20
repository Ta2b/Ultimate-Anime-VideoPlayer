// Réglages et ligne de commande de trtexec. Les chaînes attendues de
// sanitize_trt11 sortent de sanitize_settings_trt11 (aji_trt.cpp:815-860 du
// fork, commit 118c272) compilée telle quelle sur les mêmes entrées.
#include "prepare/trtexec_command.h"

#include <catch2/catch_test_macros.hpp>

#include "prepare/trtexec_progress.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;

namespace {

ajp::gpu::ShapeSet fixed(std::int64_t channels, std::int64_t h, std::int64_t w) {
  const ajp::gpu::Shape shape{{1, channels, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

}  // namespace

TEST_CASE("engine_settings : chaînes du fork pour SPAN et RIFE", "[prepare][trtexec]") {
  // DEFAULT_TRT_ENGINE_SETTINGS (aji_trt.cpp:206-208), %video_resolution% remplacé.
  CHECK(engine_settings(Recipe::Upscaler, "input", fixed(3, 1080, 1920)) ==
        "--builderOptimizationLevel=5 --optShapes=input:1x3x1080x1920 --skipInference");
  // setup_rife (aji_trt.cpp:1317-1320).
  CHECK(
      engine_settings(Recipe::Fp32Islands, "input", fixed(11, 1088, 1920)) ==
      "--stronglyTyped --optShapes=input:1x11x1088x1920 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw "
      "--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference");
  // Petit RIFE : le niveau 5 devant, le reste inchangé.
  CHECK(engine_settings(Recipe::Fp32IslandsTuned, "input", fixed(11, 1088, 1920)) ==
        "--builderOptimizationLevel=5 --stronglyTyped --optShapes=input:1x11x1088x1920 "
        "--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw "
        "--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference");
}

TEST_CASE("engine_settings : profil dynamique et ONNX statique", "[prepare][trtexec]") {
  const ajp::gpu::ShapeSet dynamic{
      .min = {{1, 3, 360, 640}}, .opt = {{1, 3, 720, 1280}}, .max = {{1, 3, 1080, 1920}}};
  CHECK(engine_settings(Recipe::Upscaler, "input", dynamic) ==
        "--builderOptimizationLevel=5 --skipInference --minShapes=input:1x3x360x640 "
        "--optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920");
  // Préréglage « Dynamic » du Manager (MainWindowViewModel.cs:1606-1610) appliqué
  // au réglage par défaut, octet pour octet : la clé du moteur du fork est reprise.
  const ajp::gpu::ShapeSet manager_dynamic{
      .min = {{1, 3, 8, 8}}, .opt = {{1, 3, 1080, 1920}}, .max = {{1, 3, 1080, 1920}}};
  CHECK(engine_settings(Recipe::Upscaler, "input", manager_dynamic) ==
        "--builderOptimizationLevel=5 --skipInference --minShapes=input:1x3x8x8 "
        "--optShapes=input:1x3x1080x1920 --maxShapes=input:1x3x1080x1920");
  // Préréglage « Static ONNX » du Manager : aucun profil (RemoveShapeArgs).
  CHECK(engine_settings(Recipe::Upscaler, "input", std::nullopt) ==
        "--builderOptimizationLevel=5 --skipInference");
  CHECK(shapes_label(dynamic) == "640x360-1920x1080");
  CHECK(shapes_label(fixed(3, 1080, 1920)) == "1920x1080");
}

TEST_CASE("sanitize_trt11 : parité avec le fork", "[prepare][trtexec]") {
  CHECK(sanitize_trt11("--stronglyTyped --optShapes=input:1x3x1080x1920 --inputIOFormats=fp16:chw "
                       "--outputIOFormats=fp16:chw --builderOptimizationLevel=5 "
                       "--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference")
            .kept ==
        "--stronglyTyped --optShapes=input:1x3x1080x1920 --builderOptimizationLevel=5 --skipInference");
  CHECK(sanitize_trt11("--stronglyTyped --optShapes=input:1x11x1088x1920 --inputIOFormats=fp16:chw "
                       "--outputIOFormats=fp16:chw --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference")
            .kept == "--stronglyTyped --optShapes=input:1x11x1088x1920 --skipInference");
  CHECK(sanitize_trt11("--fp16 --workspace=4096 --best --fp16x --int8").kept ==
        "--memPoolSize=workspace:4096M --fp16x ");
  CHECK(sanitize_trt11(
            "--optShapes=input:1x3x720x1280 --fp16 --tacticSources=-CUDNN --workspace=2048 --buildOnly")
            .kept == "--optShapes=input:1x3x720x1280 --memPoolSize=workspace:2048M ");
  CHECK(sanitize_trt11("--bf16  --int8   --skipInference --outputIOFormats=fp16:chw").kept ==
        "--skipInference ");
  // Bords du fork, gardés tels quels : seul le bord droit d'un drapeau est vérifié,
  // et seul le premier --workspace est réécrit.
  CHECK(sanitize_trt11("x--fp16 --int8").kept == "x");
  CHECK(sanitize_trt11("--workspace=1 --workspace=2").kept == "--memPoolSize=workspace:1M --workspace=2");
  CHECK(sanitize_trt11("--tacticSources=-CUDNN").kept.empty());
}

TEST_CASE("sanitize_trt11 : chaque drapeau retiré est nommé", "[prepare][trtexec]") {
  const auto result = sanitize_trt11("--fp16 --inputIOFormats=fp16:chw --workspace=64 --skipInference");
  CHECK(result.removed == std::vector<std::string>{"--fp16", "--inputIOFormats=fp16:chw",
                                                   "--workspace=64 → --memPoolSize=workspace:64M"});
  CHECK(sanitize_trt11("--builderOptimizationLevel=5 --skipInference").removed.empty());
}

TEST_CASE("trtexec_arguments : chemins sans guillemets, timing cache en dernier", "[prepare][trtexec]") {
  const TrtexecPaths paths{.onnx = "/modèles/span.onnx",
                           .engine = "/cache/.aji-build-1/model.engine",
                           .timing_cache = timing_cache_for("/cache/aji-v3-x.engine")};
  CHECK(trtexec_arguments(paths, "--stronglyTyped  --skipInference ") ==
        std::vector<std::string>{"--onnx=/modèles/span.onnx", "--saveEngine=/cache/.aji-build-1/model.engine",
                                 "--stronglyTyped", "--skipInference",
                                 "--timingCacheFile=/cache/aji-v3-x.engine.timing.cache"});
}

TEST_CASE("check_shapes : dimensions positives et ordonnées", "[prepare][trtexec]") {
  CHECK(check_shapes(fixed(3, 720, 1280)));
  CHECK(test::refusal(check_shapes(fixed(3, 0, 1280))) == RefusalCode::FieldOutOfRange);
  const ajp::gpu::ShapeSet reversed{
      .min = {{1, 3, 1080, 1920}}, .opt = {{1, 3, 720, 1280}}, .max = {{1, 3, 1080, 1920}}};
  CHECK(test::refusal(check_shapes(reversed)) == RefusalCode::FieldOutOfRange);
}

TEST_CASE("TrtexecProgress : repères dans l'ordre, jamais de recul", "[prepare][trtexec]") {
  TrtexecProgress progress;
  CHECK(progress.feed("&&&& RUNNING TensorRT.trtexec [TensorRT v110300]")->fraction == 0.02F);
  CHECK_FALSE(progress.feed("[I] === Model Options ==="));
  const auto parsing =
      progress.feed("[10/18/2026-15:29:26] [I] Finished parsing network model. Parse time: 0.02");
  REQUIRE(parsing);
  CHECK(parsing->stage == "Choix des tactiques");
  CHECK_FALSE(progress.feed("[I] Start parsing network model."));  // repère antérieur : ignoré
  CHECK_FALSE(progress.passed());
  CHECK(progress.feed("&&&& PASSED TensorRT.trtexec")->fraction == 1.0F);
  CHECK(progress.passed());
}

TEST_CASE("TrtexecProgress : dernière erreur et échec", "[prepare][trtexec]") {
  TrtexecProgress progress;
  progress.feed("[E] Error[1]: première");
  progress.feed("  [10/18/2026] [E] Error[4]: seconde  ");
  progress.feed("&&&& FAILED TensorRT.trtexec");
  CHECK(progress.failed());
  CHECK(progress.last_error() == "[10/18/2026] [E] Error[4]: seconde");
}
