#include "app/play/auto_bench.h"

#include <algorithm>
#include <filesystem>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/auto_bench_run.h"
#include "app/play/options.h"

using namespace ajp;
using namespace ajp::app::play;

namespace {

BenchRun run(Tier level, std::int64_t lost = 0, std::int64_t presents = 2000, std::int64_t lag_us = 3000) {
  return BenchRun{.level = level,
                  .presents = presents,
                  .dropped = lost / 2,
                  .missing = lost - lost / 2,
                  .lag_p95_us = lag_us,
                  .period_us = 6060};
}

}  // namespace

TEST_CASE("un essai tient quand peu d'images sont perdues et le retard est court", "[auto_bench]") {
  CHECK(holds(run(Tier::Fast)));
  CHECK(holds(run(Tier::Fast, 40)));        // 2 %
  CHECK_FALSE(holds(run(Tier::Fast, 41)));  // juste au-dessus
  CHECK_FALSE(holds(run(Tier::Fast, 0, 2000, 6060 * 2 + 1)));
  CHECK(holds(run(Tier::Fast, 0, 2000, 6060 * 2)));
}

TEST_CASE("un essai trop court ne tient pas", "[auto_bench]") {
  CHECK_FALSE(holds(run(Tier::Light, 0, 50)));
}

TEST_CASE("la recherche monte tant que ça tient puis s'arrête", "[auto_bench]") {
  std::vector<BenchRun> runs;
  CHECK(next_level(runs, Tier::Normal) == Tier::Normal);
  runs.push_back(run(Tier::Normal));
  CHECK(next_level(runs, Tier::Normal) == Tier::Fast);
  runs.push_back(run(Tier::Fast));
  CHECK(next_level(runs, Tier::Normal) == Tier::Ultra);
  runs.push_back(run(Tier::Ultra, 500));
  CHECK_FALSE(next_level(runs, Tier::Normal));  // Ultra ne tient pas, Rapide tient : fini
  CHECK(best_level(runs) == Tier::Fast);
}

TEST_CASE("la recherche descend quand le niveau estimé ne tient pas", "[auto_bench]") {
  std::vector<BenchRun> runs{run(Tier::Fast, 500)};
  CHECK(next_level(runs, Tier::Fast) == Tier::Normal);
  runs.push_back(run(Tier::Normal, 500));
  CHECK(next_level(runs, Tier::Fast) == Tier::Light);
  runs.push_back(run(Tier::Light));
  CHECK_FALSE(next_level(runs, Tier::Fast));
  CHECK(best_level(runs) == Tier::Light);
}

TEST_CASE("rien ne tient : Économe, et la recherche s'arrête en bas", "[auto_bench]") {
  const std::vector<BenchRun> runs{run(Tier::Light, 500)};
  CHECK(next_level(runs, Tier::Light) == Tier::Economy);
  std::vector<BenchRun> all = runs;
  all.push_back(run(Tier::Economy, 500));
  CHECK_FALSE(next_level(all, Tier::Light));
  CHECK(best_level(all) == Tier::Economy);
}

TEST_CASE("la recherche s'arrête en haut", "[auto_bench]") {
  CHECK_FALSE(next_level({run(Tier::Ultra)}, Tier::Ultra));
  CHECK(best_level({run(Tier::Ultra)}) == Tier::Ultra);
}

TEST_CASE("bench.toml : aller-retour, refus, fichier absent", "[auto_bench]") {
  const BenchResult result{.gpu = "NVIDIA GeForce RTX 4070 Ti", .level = Tier::Ultra, .date = "2026-10-28"};
  const auto read = parse_bench(format_bench(result));
  REQUIRE(read);
  CHECK(read->gpu == result.gpu);
  CHECK(read->level == Tier::Ultra);
  CHECK(read->date == "2026-10-28");
  CHECK_FALSE(parse_bench("level = 9\ngpu = \"x\""));
  CHECK_FALSE(parse_bench("level = 1"));
  CHECK_FALSE(parse_bench("gpu = ["));
  CHECK_FALSE(load_bench(std::filesystem::temp_directory_path() / "ajp-no-such-bench.toml"));

  const auto file = std::filesystem::temp_directory_path() / "ajp-bench-test.toml";
  REQUIRE(save_bench(file, result));
  const auto loaded = load_bench(file);
  REQUIRE(loaded);
  CHECK(loaded->level == Tier::Ultra);
  std::filesystem::remove(file);
}

TEST_CASE("la mesure remplace l'estimation pour la même carte seulement", "[auto_bench]") {
  const GpuFacts gpu{.name = "NVIDIA GeForce RTX 4070 Ti", .vendor = GpuVendor::Nvidia, .tensor_cores = true};
  CHECK(effective_tier(gpu, std::nullopt) == Tier::Fast);  // l'estimation
  CHECK(effective_tier(gpu, BenchResult{.gpu = gpu.name, .level = Tier::Ultra}) == Tier::Ultra);
  // Une autre carte (changement de matériel) : la mesure ne vaut plus.
  CHECK(effective_tier(gpu, BenchResult{.gpu = "NVIDIA GeForce GTX 1060", .level = Tier::Economy}) ==
        Tier::Fast);
}

TEST_CASE("la mesure est refaite quand le pilote change", "[auto_bench]") {
  const GpuFacts gpu{.name = "NVIDIA GeForce RTX 4070 Ti",
                     .driver = "2400",
                     .vendor = GpuVendor::Nvidia,
                     .tensor_cores = true};
  CHECK(effective_tier(gpu, BenchResult{.gpu = gpu.name, .driver = "2400", .level = Tier::Ultra}) ==
        Tier::Ultra);
  CHECK(effective_tier(gpu, BenchResult{.gpu = gpu.name, .driver = "2300", .level = Tier::Ultra}) ==
        Tier::Fast);
  const auto read =
      parse_bench(format_bench(BenchResult{.gpu = "g", .driver = "2400", .level = Tier::Light}));
  REQUIRE(read);
  CHECK(read->driver == "2400");
}

TEST_CASE("la lecture d'un essai : aller-retour et refus", "[auto_bench]") {
  const BenchReading reading{.gpu = "g", .driver = "7", .run = run(Tier::Fast, 10)};
  const auto read = parse_reading(format_reading(reading));
  REQUIRE(read);
  CHECK(read->gpu == "g");
  CHECK(read->driver == "7");
  CHECK(read->run.level == Tier::Fast);
  CHECK(read->run.presents == 2000);
  CHECK(read->run.dropped + read->run.missing == 10);
  CHECK(read->run.lag_p95_us == 3000);
  CHECK(read->run.period_us == 6060);
  CHECK_FALSE(parse_reading("level = 12"));
  CHECK_FALSE(parse_reading("gpu = ["));
}

TEST_CASE("la ligne de commande d'un essai : son coupé, dossier jetable, niveau", "[auto_bench]") {
  const BenchPlan plan{.executable = "/x/ajp", .video = "/v/a.mkv", .work_dir = "/tmp/w", .seconds = 9};
  const auto args = bench_child_args(plan, Tier::Light, "/tmp/w/d", "/tmp/w/o.toml");
  const auto has = [&](std::string_view arg) { return std::ranges::find(args, arg) != args.end(); };
  CHECK(args.front() == "play");
  CHECK(has("--mute"));
  CHECK(has("--seconds=9"));
  CHECK(has("--no-resume"));
  CHECK(has("--fullscreen"));
  CHECK(has("--bench-level=1"));
  CHECK(has("--data-dir=/tmp/w/d"));
  CHECK(has("--bench-out=/tmp/w/o.toml"));
  const auto automatic = bench_child_args(plan, std::nullopt, "/d", "/o");
  CHECK(std::ranges::find(automatic, "--bench-level=auto") != automatic.end());
  // Ce que l'enfant reçoit se lit.
  std::vector<std::string_view> views(args.begin() + 1, args.end());
  const auto parsed = parse_play_options(views);
  REQUIRE(parsed);
  CHECK(parsed->mute);
  CHECK(parsed->bench_level == 1);
  CHECK(parsed->data_dir == std::filesystem::path("/tmp/w/d"));
  CHECK(parsed->bench_out == std::filesystem::path("/tmp/w/o.toml"));
}

TEST_CASE("ajp bench : options et refus", "[auto_bench]") {
  const std::vector<std::string_view> good{"a.mkv", "--seconds=20", "--report=r.txt", "--windowed"};
  const auto options = parse_bench_options(good);
  REQUIRE(options);
  CHECK(options->seconds == 20);
  CHECK(options->report == std::filesystem::path("r.txt"));
  CHECK_FALSE(options->fullscreen);
  CHECK_FALSE(parse_bench_options(std::vector<std::string_view>{}));
  CHECK_FALSE(parse_bench_options(std::vector<std::string_view>{"a.mkv", "--seconds=2"}));
  CHECK_FALSE(parse_bench_options(std::vector<std::string_view>{"a.mkv", "--nope"}));
  CHECK_FALSE(parse_bench_options(std::vector<std::string_view>{"a.mkv", "b.mkv"}));
  const std::vector<std::string_view> bad_level{"--bench-level=9", "a.mkv"};
  CHECK_FALSE(parse_play_options(bad_level));
}

TEST_CASE("le rapport nomme chaque essai et le niveau gardé", "[auto_bench]") {
  BenchOutcome outcome;
  outcome.estimated = Tier::Fast;
  outcome.runs = {run(Tier::Fast), run(Tier::Ultra, 600)};
  outcome.result = BenchResult{.gpu = "g", .driver = "", .level = Tier::Fast};
  const std::string report = format_bench_report(outcome);
  CHECK(report.contains("tient"));
  CHECK(report.contains("ne tient pas"));
  CHECK(report.contains("Niveau gardé : Rapide"));
}

TEST_CASE("ajp compare : options, ligne de commande d'une variante, rapport", "[auto_bench]") {
  const std::vector<std::string_view> args{"a.mkv",    "--a=--rife=3  --work-height=540", "--b",
                                           "--rife=4", "--b-settings=poses.toml",         "--seconds=8"};
  const auto options = parse_compare_options(args);
  REQUIRE(options);
  CHECK(options->a == std::vector<std::string>{"--rife=3", "--work-height=540"});
  CHECK(options->b == std::vector<std::string>{"--rife=4"});
  CHECK(options->b_settings == std::filesystem::path("poses.toml"));
  CHECK(options->seconds == 8);
  CHECK_FALSE(parse_compare_options(std::vector<std::string_view>{}));

  const BenchPlan plan{.executable = "/x/ajp", .video = "/v/a.mkv", .seconds = 8};
  const auto child = variant_child_args(plan, Variant{.name = "A", .args = options->a}, "/d", "/o");
  CHECK(std::ranges::find(child, "--rife=3") != child.end());
  CHECK(std::ranges::find(child, "--mute") != child.end());
  CHECK(std::ranges::find(child, "--bench-level=1") == child.end());

  CompareOutcome outcome{.a = Variant{.name = "A"}, .b = Variant{.name = "B"}};
  outcome.reading_a = BenchReading{.gpu = "g", .run = run(Tier::Normal), .interval_p95_us = 20000};
  outcome.reading_b = BenchReading{.gpu = "g", .run = run(Tier::Normal, 600), .interval_p95_us = 30000};
  CHECK(format_compare_report(outcome).contains("Seule « A » tient"));
  outcome.reading_b.run = run(Tier::Normal);
  CHECK(format_compare_report(outcome).contains("« A » est plus légère de 10.0 ms"));
}
