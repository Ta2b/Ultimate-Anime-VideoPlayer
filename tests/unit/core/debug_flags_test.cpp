#include "core/debug_flags.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;

namespace {

std::expected<CommandLine, ajp::Refusal> parse(std::vector<std::string_view> args) {
  return parse_command_line(args);
}

RefusalCode refusal_of(std::vector<std::string_view> args) {
  auto line = parse(std::move(args));
  REQUIRE_FALSE(line);
  return line.error().code;
}

}  // namespace

TEST_CASE("DebugFlags : sans option, les valeurs par défaut", "[debug_flags]") {
  auto line = parse({});
  REQUIRE(line);
  CHECK(line->debug.log_level == LogLevel::Info);
  CHECK_FALSE(line->debug.vulkan_validation);
  CHECK_FALSE(line->debug.force_engine);
  CHECK_FALSE(line->debug.display);
  CHECK(line->debug.max_frames == 0);
  CHECK(line->inputs.empty());
  CHECK_FALSE(line->help);
}

TEST_CASE("DebugFlags : chaque option remplit son champ", "[debug_flags]") {
  auto line = parse({"--log-level=trace", "--vulkan-validation", "--no-cuda-graphs", "--trace-pairs",
                     "--no-exact-reuse", "--dump-plan", "--engine=tensorrt", "--ring-format=yuv444p16",
                     "--display=2560x1440@59.94:hdr", "--max-frames=240", "episode.mkv"});
  REQUIRE(line);
  const DebugFlags& flags = line->debug;
  CHECK(flags.log_level == LogLevel::Trace);
  CHECK(flags.vulkan_validation);
  CHECK(flags.no_cuda_graphs);
  CHECK(flags.trace_pairs);
  CHECK(flags.no_exact_reuse);
  CHECK(flags.dump_plan);
  CHECK(flags.force_engine == EngineId::TensorRt);
  CHECK(flags.ring_format == RingFormat::Yuv444p16);
  REQUIRE(flags.display);
  CHECK(flags.display->width == 2560);
  CHECK(flags.display->height == 1440);
  CHECK(flags.display->refresh_hz == 59.94);
  CHECK(flags.display->hdr);
  CHECK(flags.max_frames == 240);
  CHECK(line->inputs == std::vector<std::string>{"episode.mkv"});

  auto sdr = parse({"--display=1920x1080@60"});
  REQUIRE(sdr);
  CHECK_FALSE(sdr->debug.display->hdr);
}

TEST_CASE("DebugFlags : fichiers, « - » et « -- »", "[debug_flags]") {
  auto line = parse({"a.mkv", "-", "--dump-plan", "--", "--vulkan-validation", "-b.mkv"});
  REQUIRE(line);
  CHECK(line->debug.dump_plan);
  CHECK_FALSE(line->debug.vulkan_validation);
  CHECK(line->inputs == std::vector<std::string>{"a.mkv", "-", "--vulkan-validation", "-b.mkv"});

  auto help = parse({"--help"});
  REQUIRE(help);
  CHECK(help->help);
}

TEST_CASE("DebugFlags : une option inconnue ou une valeur fausse est refusée", "[debug_flags]") {
  CHECK(refusal_of({"--verbose"}) == RefusalCode::OptionUnknown);
  CHECK(refusal_of({"-v"}) == RefusalCode::OptionUnknown);
  CHECK(refusal_of({"--dump-plan=yes"}) == RefusalCode::FieldType);
  CHECK(refusal_of({"--engine"}) == RefusalCode::FieldMissing);
  CHECK(refusal_of({"--help=1"}) == RefusalCode::OptionUnknown);
  CHECK(refusal_of({"--log-level=loud"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--engine=cuda"}) == RefusalCode::FieldOutOfRange);
  // Les SDK TrueHDR ne sont pas des moteurs d'inférence.
  CHECK(refusal_of({"--engine=rtx_video_sdk"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--engine=maxine_vfx"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--ring-format=nv12"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--display=1920x1080"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--display=0x1080@60"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--display=1920x1080@0"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--display=1920x1080@60:sdr"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--max-frames=-1"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of({"--max-frames=12abc"}) == RefusalCode::FieldOutOfRange);
}

TEST_CASE("DebugFlags : argv saute le nom du programme ; l'aide liste chaque option", "[debug_flags]") {
  const char* argv[] = {"ajp", "--dump-plan", "film.mkv"};
  auto line = parse_command_line(3, argv);
  REQUIRE(line);
  CHECK(line->debug.dump_plan);
  CHECK(line->inputs == std::vector<std::string>{"film.mkv"});

  const std::string help = command_line_help();
  for (const char* option :
       {"--log-level=", "--vulkan-validation", "--no-cuda-graphs", "--trace-pairs", "--no-exact-reuse",
        "--dump-plan", "--engine=", "--ring-format=", "--display=", "--max-frames=", "--help"}) {
    INFO(option);
    CHECK(help.find(option) != std::string::npos);
  }
}
