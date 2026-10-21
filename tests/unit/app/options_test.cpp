#include "app/play/options.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/frame_queue.h"
#include "app/play/frame_stats.h"
#include "doubles.h"

namespace ajp::app::play {
namespace {

std::expected<PlayOptions, Refusal> parse(std::vector<std::string_view> args) {
  return parse_play_options(args);
}

TEST_CASE("ajp play : fichier et options", "[app][options]") {
  const auto options = parse({"--model", "/m/span.onnx", "épisode.mkv", "--window=960x540", "--seconds=4.5",
                              "--fullscreen-after=2", "--start=90", "--stats=/tmp/s.tsv"});
  REQUIRE(options);
  CHECK(options->file == "épisode.mkv");
  CHECK(options->model == std::filesystem::path("/m/span.onnx"));
  CHECK(options->window_width == 960);
  CHECK(options->window_height == 540);
  CHECK(options->stop_after_us == 4'500'000);
  CHECK(options->fullscreen_after_us == 2'000'000);
  CHECK(options->start_us == 90'000'000);
  CHECK(options->stats == std::filesystem::path("/tmp/s.tsv"));
  CHECK(options->upscale);
  CHECK_FALSE(options->fullscreen);
  CHECK_FALSE(options->sdr);
  const auto sdr = parse({"épisode.mkv", "--sdr"});
  REQUIRE(sdr);
  CHECK(sdr->sdr);
  CHECK_FALSE(options->one_lane);
  const auto one = parse({"épisode.mkv", "--one-lane"});
  REQUIRE(one);
  CHECK(one->one_lane);
}

TEST_CASE("ajp play : refus nommés", "[app][options]") {
  CHECK(parse({}).error().code == RefusalCode::FieldMissing);
  CHECK(parse({"a.mkv", "--vitesse=2"}).error().code == RefusalCode::OptionUnknown);
  CHECK(parse({"a.mkv", "b.mkv"}).error().code == RefusalCode::OptionUnknown);
  CHECK(parse({"a.mkv", "--window=10x10"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--window=1280"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--seconds=-1"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--model"}).error().code == RefusalCode::FieldMissing);
  CHECK(parse({"--", "--fichier.mkv"})->file == "--fichier.mkv");
  CHECK(parse({"--help"})->help);
}

TEST_CASE("ajp play : taille de travail, 720 lignes par défaut comme le profil actuel", "[app][options]") {
  CHECK(parse({"a.mkv"})->work_height == 720);
  CHECK(parse({"a.mkv", "--work-height=0"})->work_height == 0);
  CHECK(parse({"a.mkv", "--work-height=540"})->work_height == 540);
  CHECK(parse({"a.mkv", "--work-height=-1"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--work-height=haut"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(work_size(1920, 1080, 720) == WorkSize{1280, 720});
  CHECK(work_size(1280, 720, 720) == WorkSize{1280, 720});  // déjà à la taille : rien
  CHECK(work_size(854, 480, 720) == WorkSize{854, 480});    // plus petite : jamais agrandie
  CHECK(work_size(1920, 1080, 0) == WorkSize{1920, 1080});  // 0 : jamais réduite
  CHECK(work_size(1920, 800, 720) == WorkSize{1728, 720});  // 2,40:1
  CHECK(work_size(1440, 1080, 720) == WorkSize{960, 720});  // 4:3
  CHECK(work_size(1918, 1080, 720).width % 2 == 0);         // largeur paire
}

TEST_CASE("ajp play : RIFE ×3 par défaut, --rife et --rife-model", "[app][options]") {
  CHECK(parse({"a.mkv"})->rife_factor == kDefaultRifeFactor);
  CHECK(parse({"a.mkv", "--rife=4"})->rife_factor == 4);
  CHECK(parse({"a.mkv", "--rife=0"})->rife_factor == 0);
  CHECK(parse({"a.mkv", "--rife-model=/m/rife.onnx"})->rife_model == std::filesystem::path("/m/rife.onnx"));
  CHECK(parse({"a.mkv", "--rife=11"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--rife=deux"}).error().code == RefusalCode::FieldOutOfRange);
  for (const char* half : {"--rife=1.5", "--rife=1,5"}) {
    const auto opts = parse({"a.mkv", half});
    REQUIRE(opts);
    CHECK(opts->rife_factor == 3);
    CHECK(opts->rife_factor_den == 2);
  }
  CHECK(parse({"a.mkv", "--rife=1.25"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(factor_text({.num = 3, .den = 2}) == "×1,5");
  CHECK(factor_text({.num = 3, .den = 1}) == "×3");
  CHECK(default_rife_candidates("/home/x").front() ==
        std::filesystem::path("/home/x/.local/share/AnimeJaNai-HDR-linux/animejanai/rife/rife_v4.25.onnx"));
}

TEST_CASE("ajp play : RTX HDR et ses réglages, refus nommés hors plage", "[app][options]") {
  const auto off = parse({"a.mkv"});
  CHECK_FALSE(off->true_hdr);
  CHECK(off->true_hdr_peak == 0);  // la crête de l'écran
  CHECK(off->true_hdr_sdk == std::filesystem::path("/usr/local/VideoFX"));
  const auto on = parse({"a.mkv", "--truehdr", "--truehdr-peak=766", "--truehdr-contrast=150",
                         "--truehdr-saturation=120", "--truehdr-gray=40", "--truehdr-sdk=/opt/vfx"});
  REQUIRE(on);
  CHECK(on->true_hdr);
  CHECK(on->true_hdr_peak == 766);
  CHECK(on->true_hdr_contrast == 150);
  CHECK(on->true_hdr_saturation == 120);
  CHECK(on->true_hdr_gray == 40);
  CHECK(on->true_hdr_sdk == std::filesystem::path("/opt/vfx"));
  CHECK(parse({"a.mkv", "--truehdr-peak=0"})->true_hdr_peak == 0);
  CHECK(parse({"a.mkv", "--truehdr-peak=300"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--truehdr-peak=2001"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--truehdr-gray=5"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--truehdr-contrast=201"}).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Modèle : --model doit exister, sinon le premier candidat présent", "[app][options]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-app-model-test";
  std::filesystem::create_directories(dir);
  const auto present = dir / "present.onnx";
  std::ofstream(present) << "x";
  const std::array candidates{dir / "absent.onnx", present};

  PlayOptions options;
  CHECK(find_model(options, candidates) == present);
  options.model = dir / "autre.onnx";
  CHECK(find_model(options, candidates).error().code == RefusalCode::ModelUnreadable);
  CHECK(find_model(PlayOptions{}, std::span(candidates).first(1)).error().code ==
        RefusalCode::ModelUnreadable);
  // Les candidats par défaut sont l'installation d'AnimeJaNai, puis le dépôt mpv.
  const auto defaults = default_model_candidates("/home/x");
  REQUIRE(defaults.size() == 2);
  CHECK(defaults[0] == std::filesystem::path("/home/x/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx") /
                           kDefaultModelFile);
  // Les modèles livrés avec le lecteur viennent en dernier, SPAN comme RIFE.
  const auto bundled = default_model_candidates("/home/x", "/opt/ajp/models");
  REQUIRE(bundled.size() == 3);
  CHECK(bundled.back() == std::filesystem::path("/opt/ajp/models/onnx") / kDefaultModelFile);
  CHECK(default_rife_candidates("/home/x", "/opt/ajp/models").back() ==
        std::filesystem::path("/opt/ajp/models/rife/rife_v4.25.onnx"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("File d'images : choix par l'horloge, retards, époques", "[app][frames]") {
  FrameQueue queue;
  for (std::uint32_t i = 0; i < 4; ++i) queue.push(test::ring_frame(i, i * 40'000));
  std::vector<RingFrame> late;
  CHECK_FALSE(queue.pick(-10'000, 0, late));  // la tête n'est pas atteinte
  const auto picked = queue.pick(85'000, 0, late);
  REQUIRE(picked);
  CHECK(picked->pts_us == 80'000);
  CHECK(late.size() == 2);
  CHECK(queue.size() == 1);

  queue.push(test::ring_frame(5, kNoPts, 1));
  queue.push(test::ring_frame(6, 0, 1));
  CHECK(queue.first_of(1)->slot == 5);
  const auto stale = queue.take_stale(1);
  REQUIRE(stale.size() == 1);
  CHECK(stale[0].slot == 3);
  late.clear();
  // Une image sans pts passe dès qu'elle est en tête, et seule.
  CHECK(queue.pick(-1'000'000, 0, late)->slot == 5);
  CHECK(queue.take_all().size() == 1);
}

TEST_CASE("File d'images : l'image sans pts arrête le choix, bloc de deque après bloc", "[app][frames]") {
  // Assez d'images pour que pop_front libère plusieurs blocs de la deque : le choix ne relit
  // jamais l'image qu'il vient de retirer (ASan le verrait).
  FrameQueue queue;
  constexpr std::uint32_t kCount = 999;
  for (std::uint32_t i = 0; i < kCount; ++i)
    queue.push(test::ring_frame(i, i % 3 == 2 ? kNoPts : std::int64_t{i} * 1'000));
  std::vector<RingFrame> late;
  for (std::uint32_t i = 2; i < kCount; i += 3) {
    late.clear();
    const auto picked = queue.pick(std::int64_t{kCount} * 1'000, 0, late);
    REQUIRE(picked);
    CHECK(picked->slot == i);
    CHECK(late.size() == 2);
  }
  CHECK(queue.empty());
}

TEST_CASE("Mesures : médiane, écart type, format", "[app][stats]") {
  FrameStats stats;
  for (int i = 0; i < 10; ++i) stats.presented(1000, (i + 1) * 41'700, true);
  CHECK(stats.frame_interval.median() == 41'700);
  CHECK(stats.frame_interval.stddev() == 0.0);
  CHECK(stats.new_frames == 10);
  CHECK(format_ms(41'708) == "41,7 ms");
  CHECK(format_position(3'723'456'000) == "01:02:03.456");
  CHECK(format_position(kNoPts) == "--:--:--.---");
}

}  // namespace
}  // namespace ajp::app::play
