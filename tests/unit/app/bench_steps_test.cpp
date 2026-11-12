#include "app/play/bench_steps.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/auto_bench_run.h"
#include "app/play/bench_events.h"
#include "french_ui.h"

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

[[maybe_unused]] fs::path scratch(const std::string& name) {
  static std::atomic<int> counter{0};
  const fs::path dir = fs::temp_directory_path() / std::format("ajp-bench-test-{}-{}", name, counter++);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

}  // namespace

TEST_CASE("les étapes se lisent « Étape n sur 3 » et la barre globale ne recule jamais", "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  CHECK(step_line(BenchStep::Levels, "test des réglages « Rapide »") ==
        "Étape 2 sur 3 : test des réglages « Rapide »");
  float last = -1.0F;
  for (const BenchStep step : {BenchStep::Prepare, BenchStep::Levels, BenchStep::Explore}) {
    for (const float within : {0.0F, 0.5F, 1.0F}) {
      const float now = overall_fraction(step, within);
      CHECK(now >= last);
      last = now;
    }
  }
  CHECK(overall_fraction(BenchStep::Prepare, 0.0F) == 0.0F);
  CHECK(overall_fraction(BenchStep::Explore, 1.0F) == 1.0F);
  CHECK(overall_fraction(BenchStep::Levels, 7.0F) == overall_fraction(BenchStep::Levels, 1.0F));  // bornée
}

TEST_CASE("l'avancement partagé rend l'étape, la phrase et la note posées", "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  BenchProgress progress;
  progress.set(BenchStep::Levels, "test des réglages « Normal »", 0.5F, "patience");
  auto view = progress.get();
  CHECK(view.step == BenchStep::Levels);
  CHECK(view.what == "test des réglages « Normal »");
  CHECK(view.note == "patience");
  CHECK(view.fraction == overall_fraction(BenchStep::Levels, 0.5F));
  progress.set_note("autre");
  CHECK(progress.get().note == "autre");
}

TEST_CASE("la dernière ligne d'erreur de l'enfant perd son horodatage et son canal", "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  CHECK(strip_log_prefix("[2026-10-29 12:53:57.048] [app] [error] refus : GPU saturé") ==
        "refus : GPU saturé");
  CHECK(strip_log_prefix(
            "[2026-10-29 12:00:00.000] [app] [info] essai : [2026-10-29 12:00:00.000] [gpu] [warn] x") ==
        "essai : [2026-10-29 12:00:00.000] [gpu] [warn] x");
  CHECK(strip_log_prefix("sans prefixe") == "sans prefixe");
  CHECK(strip_log_prefix(std::string(500, 'a')).size() < 210);

  ChildTail tail(3);
  CHECK(tail.problem().empty());
  tail.add("[t] [app] [info] démarrage");
  tail.add("[t] [app] [error] CUDA : mémoire insuffisante");
  tail.add("[t] [app] [info] fermeture");
  CHECK(tail.problem() == "CUDA : mémoire insuffisante");  // l'erreur, pas la dernière ligne
  tail.add("[t] [app] [info] encore");
  tail.add("[t] [app] [info] et encore");  // l'erreur sort des 3 lignes gardées
  CHECK(tail.problem() == "et encore");    // sans erreur : la dernière ligne
  CHECK(tail.text().contains("encore"));
  CHECK_FALSE(tail.text().contains("CUDA"));
}

TEST_CASE("un essai qui n'a rien rendu est toujours expliqué en une phrase", "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  using prepare::ExitKind;
  const auto exit = [](ExitKind kind, std::int64_t code) {
    return prepare::ProcessExit{.kind = kind, .code = code, .elapsed = std::chrono::seconds(3)};
  };
  const std::string what = "des réglages « Rapide »";
  const std::string coded =
      explain_trial(what, exit(ExitKind::Exited, 3), "refus : GPU saturé", std::chrono::seconds(190));
  CHECK(coded.starts_with("L'essai des réglages « Rapide » n'a rien rendu : refus : GPU saturé"));
  CHECK(coded.contains("code 3"));
  const std::string crash = explain_trial(what, exit(ExitKind::Crashed, 11), "", std::chrono::seconds(190));
  CHECK(crash.contains("n'a rien rendu"));
  CHECK(crash.contains("brutalement"));
  const std::string slow =
      explain_trial("de préparation", exit(ExitKind::TimedOut, 0), "", std::chrono::seconds(1210));
  CHECK(slow.contains("1210 s"));
  CHECK(slow.contains("moteur"));  // dit la cause probable : la construction d'un moteur
  CHECK(explain_trial(what, exit(ExitKind::Cancelled, 0), "x", std::chrono::seconds(1)).empty());
  CHECK(explain_trial(what, exit(ExitKind::Exited, 0), "x", std::chrono::seconds(1)).empty());
  CHECK(explain_no_reading(what, "fin").contains("sans mesures lisibles"));
}

#ifndef _WIN32

namespace {

// Un faux « ajp » : un script qui rend la lecture du niveau demandé, ou échoue au-dessus d'un niveau.
struct FakePlayer {
  fs::path dir = scratch("fake");
  fs::path exe = dir / "ajp";

  // fail_above : les niveaux plus hauts s'arrêtent avec une erreur ; fail_all : tous échouent ; sleepy :
  // dort.
  FakePlayer(int fail_above, bool fail_all = false, bool sleepy = false) {
    for (int level = 0; level < kTierCount; ++level) {
      const BenchReading reading{.gpu = "RTX 4050 Laptop",
                                 .driver = "580",
                                 .run = {.level = static_cast<Tier>(level),
                                         .presents = 1500,
                                         .dropped = 0,
                                         .missing = 0,
                                         .lag_p95_us = 1000,
                                         .period_us = 16'667},
                                 // RIFE ×2 à 48 i/s sur l'écran à 60 Hz : le verdict de la calibration le
                                 // dit tenu (C07-06).
                                 .interval_mean_us = 20'833,
                                 .new_frames = 1200};
      std::ofstream(dir / std::format("lecture-{}.toml", level)) << format_reading(reading);
    }
    std::ofstream script(exe);
    script << "#!/bin/sh\n"
              "for a in \"$@\"; do case \"$a\" in\n"
              "  --bench-out=*) out=\"${a#--bench-out=}\";;\n"
              "  --bench-level=*) level=\"${a#--bench-level=}\";;\n"
              "esac; done\n"
              "[ \"$level\" = auto ] && level=2\n";
    if (sleepy) script << "sleep 30\n";
    if (fail_all) {
      script << "echo '[2026-10-29 12:00:00.000] [app] [error] SDL_CreateWindow : aucun écran' >&2\nexit 1\n";
    } else {
      script << "if [ \"$level\" -gt " << fail_above
             << " ]; then\n  echo '[2026-10-29 12:00:00.000] [app] [error] refus : GPU saturé' >&2\n  exit "
                "3\nfi\n";
    }
    script << "cp \"" << dir.string() << "/lecture-$level.toml\" \"$out\"\nexit 0\n";
    script.close();
    fs::permissions(exe, fs::perms::owner_all);
  }
  ~FakePlayer() { fs::remove_all(dir); }
  FakePlayer(const FakePlayer&) = delete;
  FakePlayer& operator=(const FakePlayer&) = delete;
};

}  // namespace

TEST_CASE("l'évaluation monte jusqu'au dernier niveau qui tient et explique l'essai qui échoue",
          "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  const FakePlayer fake(2);
  BenchProgress shown;
  const BenchPlan plan{
      .executable = fake.exe, .video = "/v/a.mkv", .work_dir = fake.dir / "w", .shown = &shown};
  std::vector<std::string> lines;
  const auto outcome =
      run_bench(plan, std::stop_token{}, [&](const std::string& line) { lines.push_back(line); });
  REQUIRE(outcome);
  CHECK(outcome->result.level == Tier::Normal);
  REQUIRE(lines.size() >= 2);
  CHECK(lines.front() == "Étape 1 sur 3 : préparation des moteurs et première mesure");
  CHECK(lines[1] == "Étape 2 sur 3 : test des réglages « Qualité »");
  REQUIRE(outcome->notes.size() == 1);
  CHECK(outcome->notes.front().starts_with(
      "L'essai des réglages « Qualité » n'a rien rendu : refus : GPU saturé"));
  CHECK(outcome->notes.front().contains("code 3"));
  CHECK(shown.get().step == BenchStep::Levels);
}

TEST_CASE("un premier essai qui échoue donne un refus qui cite la dernière ligne d'erreur", "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  const FakePlayer fake(0, true);
  const BenchPlan plan{.executable = fake.exe, .video = "/v/a.mkv", .work_dir = fake.dir / "w"};
  const auto outcome = run_bench(plan, std::stop_token{});
  REQUIRE_FALSE(outcome);
  CHECK(outcome.error().detail.starts_with(
      "L'essai de préparation n'a rien rendu : SDL_CreateWindow : aucun écran"));
}

TEST_CASE("annuler arrête l'évaluation vite, sans attendre l'essai en cours", "[bench_steps]") {
  ajp::app::play::FrenchUi french;
  const FakePlayer fake(4, false, true);
  const BenchPlan plan{.executable = fake.exe, .video = "/v/a.mkv", .work_dir = fake.dir / "w"};
  std::stop_source stop;
  std::jthread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    stop.request_stop();
  });
  const auto begun = std::chrono::steady_clock::now();
  const auto outcome = run_bench(plan, stop.get_token());
  REQUIRE_FALSE(outcome);
  CHECK(outcome.error().code == RefusalCode::Cancelled);
  CHECK(std::chrono::steady_clock::now() - begun < std::chrono::seconds(15));
}

#endif

TEST_CASE("Banc : l'enfant dit ses événements en JSON, le parent les relit champ par champ (P2.11)",
          "[app][bench]") {
  using ajp::app::play::ChildBenchEvent;
  const ChildBenchEvent refusal{
      .event = "refusal", .code = "DeviceLost", .detail = "« échec » \"cité\" \\ fin"};
  const std::string line = ajp::app::play::bench_event_line(refusal);
  CHECK(line.starts_with("ajp-bench {\"event\":\"refusal\""));
  CHECK(ajp::app::play::parse_bench_event(line) == refusal);
  CHECK(ajp::app::play::parse_bench_event("ajp-bench {\"event\":\"build\",\"futur\":\"x\"}") ==
        ChildBenchEvent{.event = "build"});
  // Une ligne du journal qui parle de construction ou d'échec n'est pas un événement.
  CHECK_FALSE(ajp::app::play::parse_bench_event("[app] [info] construction de x.engine"));
  CHECK_FALSE(ajp::app::play::parse_bench_event("ajp-bench {\"event\":\"build\""));
  CHECK_FALSE(ajp::app::play::parse_bench_event("ajp-bench {}"));
}
