// Program : les étapes de chaque ordonnancement, et les plans qu'il refuse d'exécuter.
#include "gpu/program.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cpu/cpu_backend.h"
#include "gpu_test_support.h"

using ajp::RefusalCode;
using namespace ajp::gpu;
namespace core = ajp::core;

namespace {

struct Models {
  cpu::CpuBackend backend{BackendInit{}};
  test::TempDir dir{"program"};
  std::shared_ptr<Engine> up2 =
      test::load_engine(backend, test::write_model(dir.path(), "up2", test::upscale_x2_model()), "up2");
  std::shared_ptr<Engine> rife =
      test::load_engine(backend, test::write_model(dir.path(), "rife", test::rife_blend_model()), "rife");
  std::shared_ptr<Engine> motion =
      test::load_engine(backend, test::write_model(dir.path(), "motion", test::motion_model()), "motion");

  ProgramParts parts(const core::Plan& plan) {
    return {.plan = plan,
            .backend = &backend,
            .rife = backend.rife_input(),
            .models = {{"up2", up2}, {"rife", rife}, {"motion", motion}},
            .precision = TensorType::F32};
  }
};

std::vector<StageKind> kinds(const Program& program) {
  std::vector<StageKind> result;
  for (const Stage& stage : program.stages()) result.push_back(stage.kind);
  return result;
}

core::Plan s1() {
  return test::with_level(test::upscale_plan(), {2, 1}, core::Policy::Quality, core::Schedule::BeforeUpscale);
}

core::Plan s3(core::Plan plan) {
  return test::with_level(std::move(plan), {2, 1}, core::Policy::Fast, core::Schedule::OutputRes,
                          core::Pacing::Source, "motion");
}

}  // namespace

TEST_CASE("Program : les étapes suivent les ordonnancements", "[program]") {
  Models models;
  REQUIRE(models.up2);
  REQUIRE(models.rife);
  REQUIRE(models.motion);

  SECTION("S0 : agrandissement seul") {
    auto program = Program::create(models.parts(test::upscale_plan()));
    REQUIRE(program);
    CHECK(kinds(**program) == std::vector{StageKind::Ingest, StageKind::Upscale, StageKind::WriteRing});
    CHECK((*program)->plan_hash() == test::upscale_plan().hash);
    CHECK((*program)->upscale_factor() == 2);
    CHECK(describe(**program) == "ingestion · agrandissement x2 up2 · ring");
  }
  SECTION("S1 : RIFE à la source, puis l'agrandissement de chaque image") {
    auto program = Program::create(models.parts(s1()));
    REQUIRE(program);
    CHECK(kinds(**program) == std::vector{StageKind::Ingest, StageKind::RifePad, StageKind::Interpolate,
                                          StageKind::Upscale, StageKind::WriteRing});
    CHECK(&(*program)->level_engine(0) == models.rife.get());
  }
  SECTION("S1 puis S3 : le mouvement après l'agrandissement, à la sortie") {
    auto program = Program::create(models.parts(s3(s1())));
    REQUIRE(program);
    CHECK(kinds(**program) == std::vector{StageKind::Ingest, StageKind::RifePad, StageKind::Interpolate,
                                          StageKind::Upscale, StageKind::Motion, StageKind::WarpBlend,
                                          StageKind::WriteRing});
    CHECK((*program)->stages()[4].level == 1);
    CHECK(describe(**program).find("S3 x2/1 motion") != std::string::npos);
  }
  SECTION("réduction avant l'agrandissement") {
    core::Plan plan = test::upscale_plan();
    plan.downscale = core::DownscaleStep{.filter = core::DownscaleFilter::Area, .height = 18};
    plan.output_width = 64;
    plan.output_height = 36;
    auto program = Program::create(models.parts(plan));
    REQUIRE(program);
    CHECK(kinds(**program) ==
          std::vector{StageKind::Ingest, StageKind::Downscale, StageKind::Upscale, StageKind::WriteRing});
  }
}

TEST_CASE("Program : refus des plans qu'il n'exécute pas", "[program]") {
  Models models;
  SECTION("ordonnancements") {
    const core::Plan s2 =
        test::with_level(test::upscale_plan(), {2, 1}, core::Policy::Quality, core::Schedule::AfterUpscale);
    CHECK(Program::create(models.parts(s2)).error().code == RefusalCode::StepOrder);
    const core::Plan s3_then_s1 = test::with_level(s3(test::upscale_plan()), {2, 1}, core::Policy::Quality,
                                                   core::Schedule::BeforeUpscale);
    CHECK(Program::create(models.parts(s3_then_s1)).error().code == RefusalCode::StepOrder);
    CHECK(Program::create(models.parts(s3(s3(test::upscale_plan())))).error().code == RefusalCode::StepOrder);
  }
  SECTION("moteurs") {
    ProgramParts missing = models.parts(s1());
    missing.models.erase(missing.models.begin() + 1);
    CHECK(Program::create(missing).error().code == RefusalCode::ModelUnavailable);
    const core::Plan wrong =
        test::with_level(test::upscale_plan(), {2, 1}, core::Policy::Fast, core::Schedule::OutputRes);
    CHECK(Program::create(models.parts(wrong)).error().code == RefusalCode::ModelMismatch);  // RIFE sans flot
    ProgramParts without_rife = models.parts(s1());
    without_rife.rife = nullptr;
    CHECK(Program::create(without_rife).error().code == RefusalCode::FormatUnsupported);
  }
  SECTION("format du ring, sortie, HDR et backend") {
    core::Plan yuv = test::upscale_plan();
    yuv.ring = ajp::RingFormat::Yuv444p16;
    CHECK(Program::create(models.parts(yuv)).error().code == RefusalCode::FormatUnsupported);
    core::Plan hdr = test::upscale_plan();
    hdr.hdr.kind = core::HdrMode::TrueHdr;
    CHECK(Program::create(models.parts(hdr)).error().code == RefusalCode::HdrUnavailable);
    core::Plan odd = test::upscale_plan();
    odd.output_width = 130;  // 65 × 2 : la source agrandie serait impaire
    ProgramParts odd_parts = models.parts(odd);
    CHECK(Program::create(odd_parts).error().code == RefusalCode::FieldOutOfRange);
    ProgramParts no_backend = models.parts(test::upscale_plan());
    no_backend.backend = nullptr;
    CHECK(Program::create(no_backend).error().code == RefusalCode::FieldMissing);
  }
}

TEST_CASE("Program : gpu charge les moteurs d'une ProgramRequest dans son backend", "[program]") {
  cpu::CpuBackend backend{BackendInit{}};
  test::TempDir dir("request");
  const std::vector<ModelArtifact> artifacts{
      {.id = "up2",
       .artifact = test::build_artifact(
           backend, test::write_model(dir.path(), "up2", test::upscale_x2_model()), "up2")}};
  const ProgramRequest request(test::upscale_plan(), artifacts);
  CHECK(request.plan_hash() == test::upscale_plan().hash);
  auto program = load_program(request, backend, backend.rife_input());
  REQUIRE(program);
  CHECK((*program)->precision() == TensorType::F16);
  CHECK((*program)->upscale().size() == 1);

  const ProgramRequest broken(test::upscale_plan(),
                              {{.id = "up2", .artifact = {.path = dir.path() / "absent.onnx"}}});
  auto refused = load_program(broken, backend, backend.rife_input());
  REQUIRE_FALSE(refused);
  CHECK(refused.error().detail.find("up2") != std::string::npos);  // le refus cite le modèle
  const ProgramRequest missing(test::upscale_plan(), {});
  CHECK(load_program(missing, backend, nullptr).error().code == RefusalCode::ModelUnavailable);
}
