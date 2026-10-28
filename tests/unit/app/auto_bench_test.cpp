#include "app/play/auto_bench.h"

#include <filesystem>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using namespace ajp::app::play;

namespace {

BenchRun run(Tier level, std::int64_t lost = 0, std::int64_t presents = 2000, std::int64_t lag_us = 3000) {
  return BenchRun{.level = level, .presents = presents, .dropped = lost / 2, .missing = lost - lost / 2,
                  .lag_p95_us = lag_us, .period_us = 6060};
}

}  // namespace

TEST_CASE("un essai tient quand peu d'images sont perdues et le retard est court", "[auto_bench]") {
  CHECK(holds(run(Tier::Fast)));
  CHECK(holds(run(Tier::Fast, 40)));          // 2 %
  CHECK_FALSE(holds(run(Tier::Fast, 41)));    // juste au-dessus
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
  CHECK(effective_tier(gpu, BenchResult{.gpu = "NVIDIA GeForce GTX 1060", .level = Tier::Economy}) == Tier::Fast);
}
