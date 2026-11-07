#include "app/play/options.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/frame_queue.h"
#include "app/play/frame_stats.h"
#include "doubles.h"

namespace ajp::app::play {
namespace {

// Avec --debug : les options d'essai y sont acceptées (P2.35).
std::expected<PlayOptions, Refusal> parse(std::vector<std::string_view> args) {
  args.insert(args.begin(), "--debug");
  return parse_play_options(args);
}

TEST_CASE("ajp --register et --unregister : sans fichier, l'un ou l'autre", "[app][options]") {
  const auto registering = parse({"--register"});
  REQUIRE(registering);
  CHECK(registering->registration == Registration::Register);
  CHECK(registering->file.empty());
  const auto unregistering = parse({"--unregister", "--unregister"});
  REQUIRE(unregistering);
  CHECK(unregistering->registration == Registration::Unregister);
  CHECK(parse({"épisode.mkv"})->registration == Registration::None);
  for (const auto& args : {std::vector<std::string_view>{"--register", "--unregister"},
                           std::vector<std::string_view>{"--register", "épisode.mkv"}}) {
    const auto refused = parse(args);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::OptionUnknown);
  }
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
  CHECK(options->ring_rgb10a2);
  const auto ten = parse({"épisode.mkv", "--ring-rgb10a2"});
  REQUIRE(ten);
  CHECK(ten->ring_rgb10a2);
  CHECK_FALSE(parse({"épisode.mkv", "--no-ring-rgb10a2"})->ring_rgb10a2);
  CHECK_FALSE(options->fast_steady);
  const auto steady = parse({"épisode.mkv", "--fast-steady"});
  REQUIRE(steady);
  CHECK(steady->fast_steady);
  CHECK_FALSE(options->smooth_drawings);
  const auto smooth = parse({"épisode.mkv", "--smooth-drawings"});
  REQUIRE(smooth);
  CHECK(smooth->smooth_drawings);
  CHECK(options->ring_watcher);
  const auto watcher = parse({"épisode.mkv", "--ring-watcher"});
  REQUIRE(watcher);
  CHECK(watcher->ring_watcher);
  CHECK_FALSE(parse({"épisode.mkv", "--no-ring-watcher"})->ring_watcher);
  CHECK(options->precise_wait);
  const auto precise = parse({"épisode.mkv", "--precise-wait"});
  REQUIRE(precise);
  CHECK(precise->precise_wait);
  CHECK_FALSE(parse({"épisode.mkv", "--no-precise-wait"})->precise_wait);
  CHECK(options->audio_pll);
  const auto pll = parse({"épisode.mkv", "--audio-pll"});
  REQUIRE(pll);
  CHECK(pll->audio_pll);
  CHECK_FALSE(parse({"épisode.mkv", "--no-audio-pll"})->audio_pll);
  CHECK(options->vsync_index);
  const auto index = parse({"épisode.mkv", "--vsync-index"});
  REQUIRE(index);
  CHECK(index->vsync_index);
  CHECK_FALSE(parse({"épisode.mkv", "--no-vsync-index"})->vsync_index);
  CHECK(options->probe_nvdec);
  const auto nvdec = parse({"épisode.mkv", "--probe-nvdec"});
  REQUIRE(nvdec);
  CHECK(nvdec->probe_nvdec);
  CHECK_FALSE(parse({"épisode.mkv", "--no-probe-nvdec"})->probe_nvdec);
  CHECK(options->shot_cuts);  // défaut
  const auto cuts = parse({"épisode.mkv", "--no-shot-cuts"});
  REQUIRE(cuts);
  CHECK_FALSE(cuts->shot_cuts);
  CHECK(parse({"épisode.mkv", "--shot-cuts"})->shot_cuts);
  CHECK(options->anime_shots);  // défaut ; --no-anime-shots le coupe seul (audit F3)
  CHECK_FALSE(parse({"épisode.mkv", "--no-anime-shots"})->anime_shots);
  CHECK(options->present == PresentWanted::Wait);
  CHECK(parse({"épisode.mkv", "--no-present"})->present == PresentWanted::Open);
  const auto present = parse({"épisode.mkv", "--present=timing"});
  REQUIRE(present);
  CHECK(present->present == PresentWanted::Timing);
  CHECK_FALSE(parse({"épisode.mkv", "--present=vite"}).has_value());
  CHECK_FALSE(options->kde_sdr_nits);
  const auto metadata = parse({"épisode.mkv", "--hdr-metadata"});
  REQUIRE(metadata);
  CHECK(metadata->kde_sdr_nits == kDefaultKdeSdrNits);
  CHECK(metadata->kde_sdr_auto);  // sans valeur : la luminosité SDR lue dans KDE
#if defined(__linux__) || defined(_WIN32)
  CHECK(options->kde_sdr_auto);  // défaut sous Linux (KDE) et Windows : métadonnées et compression du grade
#else
  CHECK_FALSE(options->kde_sdr_auto);
#endif
  const auto kwin = parse({"épisode.mkv", "--no-hdr-metadata"});
  REQUIRE(kwin);
  CHECK_FALSE(kwin->kde_sdr_auto);
  CHECK_FALSE(kwin->kde_sdr_nits);
  const auto brighter = parse({"épisode.mkv", "--hdr-metadata=250"});
  REQUIRE(brighter);
  CHECK(brighter->kde_sdr_nits == 250.0);
  CHECK_FALSE(brighter->kde_sdr_auto);
  CHECK_FALSE(parse({"épisode.mkv", "--hdr-metadata=20"}));
  CHECK_FALSE(parse({"épisode.mkv", "--hdr-metadata=beaucoup"}));
  CHECK_FALSE(options->one_lane);
  const auto one = parse({"épisode.mkv", "--one-lane"});
  REQUIRE(one);
  CHECK(one->one_lane);
}

TEST_CASE("ajp play : refus nommés", "[app][options]") {
  CHECK(parse({})->file.empty());  // l'écran d'accueil
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
  // « Restaurer les traits » coupé (P3.19) : une 1080p au-dessus de 720 n'est ni réduite ni agrandie.
  CHECK(span_runs(true, true, 1080, 720));
  CHECK_FALSE(span_runs(true, false, 1080, 720));
  CHECK(span_runs(true, false, 480, 720));
  CHECK_FALSE(span_runs(false, true, 480, 720));
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
  CHECK(factor_text({.num = 3, .den = 2}) == "×1.5");  // en anglais, la langue par défaut
  CHECK(factor_text({.num = 3, .den = 1}) == "×3");
  CHECK(default_rife_candidates("/home/x").front() ==
        std::filesystem::path(
            "/home/x/.local/share/AnimeJaNai-HDR-linux/animejanai/rife/rife_v4.22_lite.onnx"));
}

TEST_CASE("ajp play : le pic du grade HDR, refus nommés hors plage", "[app][options]") {
  const auto off = parse({"a.mkv"});
  CHECK(off->true_hdr_peak == 0);  // la crête de l'écran
  const auto on = parse({"a.mkv", "--truehdr-peak=766"});
  REQUIRE(on);
  CHECK(on->true_hdr_peak == 766);
  // RTX HDR et l'émulation TrueHDR sont retirés : leurs options ne sont plus connues.
  CHECK_FALSE(parse({"a.mkv", "--truehdr-contrast=150"}));
  CHECK_FALSE(parse({"a.mkv", "--truehdr-gray=40"}));
  CHECK_FALSE(parse({"a.mkv", "--truehdr"}));
  CHECK_FALSE(parse({"a.mkv", "--truehdr-sdk=/opt/vfx"}));
  CHECK(parse({"a.mkv", "--truehdr-peak=0"})->true_hdr_peak == 0);
  CHECK(parse({"a.mkv", "--truehdr-peak=300"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--truehdr-peak=2001"}).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("ajp play : le peps de --hdr-anime en %, 100 par défaut, refus hors de 0 à 200", "[app][options]") {
  const auto plain = parse({"a.mkv"});
  REQUIRE(plain);
  CHECK(plain->anime_contrast == 100);
  CHECK(plain->anime_saturation == 100);
  CHECK(plain->anime_highlights == 100);
  const auto peps =
      parse({"a.mkv", "--anime-contrast=140", "--anime-saturation=125", "--anime-highlights=0"});
  REQUIRE(peps);
  CHECK(peps->anime_contrast == 140);
  CHECK(peps->anime_saturation == 125);
  CHECK(peps->anime_highlights == 0);
  CHECK(parse({"a.mkv", "--anime-contrast=201"}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(parse({"a.mkv", "--anime-saturation=fort"}).error().code == RefusalCode::FieldOutOfRange);
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
        std::filesystem::path("/opt/ajp/models/rife/rife_v4.22_lite.onnx"));
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

namespace ajp::app::play {

// P2.35 : chaque option que l'aide cite, l'analyseur la connaît (seule ou avec une valeur).
TEST_CASE("ajp play : l'aide ne cite aucune option inconnue de l'analyseur", "[app][options]") {
  const std::string usage = play_usage(true);
  std::vector<std::string> unknown;
  for (std::size_t at = usage.find("--"); at != std::string::npos; at = usage.find("--", at + 2)) {
    std::size_t end = at + 2;
    while (end < usage.size() && ((usage[end] >= 'a' && usage[end] <= 'z') ||
                                  (usage[end] >= '0' && usage[end] <= '9') || usage[end] == '-'))
      ++end;
    const std::string name = usage.substr(at, end - at);
    if (name.size() <= 2) continue;
    const auto known = [](const std::expected<PlayOptions, Refusal>& parsed) {
      return parsed || !parsed.error().detail.starts_with("unknown option");
    };
    const std::string valued = name + "=1";
    if (!known(parse_play_options(std::vector<std::string_view>{name, "video.mkv"})) &&
        !known(parse_play_options(std::vector<std::string_view>{valued, "video.mkv"})))
      unknown.push_back(name);
  }
  INFO(std::accumulate(unknown.begin(), unknown.end(), std::string(),
                       [](std::string a, const std::string& b) { return std::move(a) + " " + b; }));
  CHECK(unknown.empty());
  // Sans --debug, une option d'essai est refusée, et l'aide simple ne la cite pas.
  const auto refused = parse_play_options(std::vector<std::string_view>{"video.mkv", "--one-lane"});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::OptionUnknown);
  CHECK(play_usage().find("--one-lane") == std::string::npos);
}

}  // namespace ajp::app::play
