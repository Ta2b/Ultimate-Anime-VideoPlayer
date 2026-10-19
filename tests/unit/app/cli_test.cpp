// Ligne de commande : commandes, options, options de débogage de core, refus nommés, aide.
#include "app/cli.h"

#include <variant>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using namespace ajp::app;

namespace {

std::expected<Invocation, Refusal> parse(std::vector<std::string_view> args) {
  return parse_invocation(args);
}

RefusalCode refusal(std::vector<std::string_view> args) {
  const auto result = parse(std::move(args));
  REQUIRE_FALSE(result);
  return result.error().code;
}

template <class Args>
Args command(std::vector<std::string_view> args) {
  const auto result = parse(std::move(args));
  REQUIRE(result);
  REQUIRE(std::holds_alternative<Args>(result->command));
  return std::get<Args>(result->command);
}

}  // namespace

TEST_CASE("Ligne de commande : chaque commande et ses arguments", "[app][cli]") {
  CHECK(command<PlayArgs>({"play", "episode01.mkv"}).file == "episode01.mkv");
  CHECK(command<PlayArgs>({"episode01.mkv"}).file == "episode01.mkv");  // raccourci de play
  CHECK(command<PlayArgs>({"--", "--bizarre.mkv"}).file == "--bizarre.mkv");
  CHECK(command<PlayArgs>({"play", "--profile=3", "a.mkv"}).profile == 3);

  const ExportArgs exported = command<ExportArgs>(
      {"export", "a.mkv", "b.mkv", "--encoder=svt-av1", "--burn-subtitles", "--overwrite"});
  CHECK(exported.input == "a.mkv");
  CHECK(exported.output == "b.mkv");
  CHECK(exported.encoder == ExportEncoder::SvtAv1);
  CHECK(exported.burn_subtitles);
  CHECK(exported.overwrite);

  CHECK(command<BenchArgs>({"bench", "a.mkv", "--dump=sortie.framemd5"}).dump == "sortie.framemd5");
  CHECK(command<ImportSdkArgs>({"import-sdk", "/usr/local/VideoFX"}).folder == "/usr/local/VideoFX");

  CHECK_FALSE(command<EnginesArgs>({"engines"}).install);
  const EnginesArgs install = command<EnginesArgs>({"engines", "install", "tensorrt", "openvino"});
  CHECK(install.install);
  CHECK(install.engines == std::vector{core::EngineId::TensorRt, core::EngineId::OpenVino});

  CHECK_FALSE(command<HelpArgs>({}).topic);
  CHECK(command<HelpArgs>({"help", "export"}).topic == CommandKind::Export);
  CHECK(command<HelpArgs>({"export", "--help"}).topic == CommandKind::Export);
  CHECK(std::holds_alternative<VersionArgs>(parse({"--version"})->command));
}

TEST_CASE("Ligne de commande : options générales et de débogage n'importe où", "[app][cli]") {
  const auto result = parse({"--log-level=debug", "play", "a.mkv", "--max-frames=50", "--data-dir=/tmp/ajp"});
  REQUIRE(result);
  CHECK(result->debug.log_level == core::LogLevel::Debug);
  CHECK(result->debug.max_frames == 50);
  CHECK(result->data_dir == "/tmp/ajp");
  CHECK(command<PlayArgs>({"a.mkv", "--engine=tensorrt"}).file == "a.mkv");
  CHECK(parse({"a.mkv", "--engine=tensorrt"})->debug.force_engine == core::EngineId::TensorRt);
}

TEST_CASE("Ligne de commande : refus nommés, avec ce qu'il faut taper", "[app][cli]") {
  CHECK(refusal({"--profile=0", "play", "a.mkv"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal({"play", "a.mkv", "--profile=10"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal({"play"}) == RefusalCode::FieldMissing);
  CHECK(refusal({"play", "a.mkv", "b.mkv"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"play", "a.mkv", "--encoder=svt-av1"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"export", "a.mkv", "b.mkv", "--encoder=x264"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal({"export", "a.mkv", "b.mkv", "--overwrite=oui"}) == RefusalCode::FieldType);
  CHECK(refusal({"bench", "a.mkv", "--dump"}) == RefusalCode::FieldMissing);
  CHECK(refusal({"--data-dir=", "engines"}) == RefusalCode::FieldMissing);
  CHECK(refusal({"-x", "a.mkv"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"--inconnue", "a.mkv"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"--profile=2"}) == RefusalCode::OptionUnknown);  // option de commande sans commande
  CHECK(refusal({"engines", "installer"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"engines", "install", "vulkan"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal({"engines", "install", "maxine_vfx"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal({"engines", "install", "tensorrt", "tensorrt"}) == RefusalCode::FieldOutOfRange);
  CHECK(refusal({"engines", "install", "cuda"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"help", "lire"}) == RefusalCode::OptionUnknown);
  CHECK(refusal({"version", "en-trop"}) == RefusalCode::OptionUnknown);

  const auto typo = parse({"exprot", "a.mkv"});
  REQUIRE_FALSE(typo);
  CHECK(typo.error().detail == "commande « exprot » inconnue ; vouliez-vous dire « export » ?");
  const auto word = parse({"film"});
  REQUIRE_FALSE(word);
  CHECK(word.error().detail.find("ajp play film") != std::string::npos);
  const auto vulkan = parse({"engines", "install", "vulkan"});
  CHECK(vulkan.error().detail == "vulkan est livré avec le lecteur");
}

TEST_CASE("Aide : chaque commande, chaque option et les options de débogage de core", "[app][cli]") {
  const std::string general = help_text("/home/u/.local/share/ajp");
  for (const std::string_view needle :
       {"play <fichier>", "export <fichier> <sortie>", "bench <fichier>", "import-sdk <dossier>",
        "engines [install [moteur…]]", "--data-dir=DOSSIER", "/home/u/.local/share/ajp", "--profile=1..9",
        "--encoder=nvenc-hevc|nvenc-av1|svt-av1|videotoolbox-hevc", "--burn-subtitles", "--dump=FICHIER",
        "--log-level", "--vulkan-validation", "--max-frames"}) {
    INFO(needle);
    CHECK(general.find(needle) != std::string::npos);
  }
  const std::string engines = help_text(CommandKind::Engines);
  CHECK(engines.starts_with("Usage : ajp engines [install [moteur…]]"));
  CHECK(engines.find("ajp engines install tensorrt") != std::string::npos);
  const std::string exported = help_text(CommandKind::Export);
  CHECK(exported.find("--overwrite") != std::string::npos);
  CHECK(exported.find("--dump") == std::string::npos);  // option d'une autre commande
  CHECK(command_name(CommandKind::ImportSdk) == "import-sdk");
  CHECK(encoder_name(ExportEncoder::NvencAv1) == "nvenc-av1");
}
