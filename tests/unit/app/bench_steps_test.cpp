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

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

fs::path scratch(const std::string& name) {
  static std::atomic<int> counter{0};
  const fs::path dir = fs::temp_directory_path() / std::format("ajp-bench-test-{}-{}", name, counter++);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

std::string slurp(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

}  // namespace

TEST_CASE("les étapes se lisent « Étape n sur 4 » et la barre globale ne recule jamais", "[bench_steps]") {
  CHECK(step_line(BenchStep::Levels, "test des réglages « Rapide »") ==
        "Étape 2 sur 4 : test des réglages « Rapide »");
  float last = -1.0F;
  for (const BenchStep step : {BenchStep::Prepare, BenchStep::Levels, BenchStep::Explore, BenchStep::Span}) {
    for (const float within : {0.0F, 0.5F, 1.0F}) {
      const float now = overall_fraction(step, within);
      CHECK(now >= last);
      last = now;
    }
  }
  CHECK(overall_fraction(BenchStep::Prepare, 0.0F) == 0.0F);
  CHECK(overall_fraction(BenchStep::Span, 1.0F) == 1.0F);
  CHECK(overall_fraction(BenchStep::Levels, 7.0F) == overall_fraction(BenchStep::Levels, 1.0F));  // bornée
}

TEST_CASE("l'avancement partagé rend l'étape, la phrase et la note posées", "[bench_steps]") {
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

TEST_CASE("le résultat tient en une phrase : chaîne gardée et mode Performance", "[bench_steps]") {
  const BenchResult fast{.gpu = "g", .driver = "d", .level = Tier::Normal};
  const ChainContext context{
      .display_height = 1440, .display_hz = 72.0, .source_height = 1080, .source_fps = 24.0};
  const std::string ready = result_sentence(fast, context, {SpanFp8Verdict::State::Ready, {}});
  CHECK(ready.starts_with("Votre PC tient : agrandissement ×2 depuis 720p"));
  CHECK(ready.contains("interpolation jusqu'à ×3"));
  CHECK(ready.contains("RIFE"));
  CHECK(ready.contains("Performance FP8 pour SPAN : prête"));
  CHECK(ready.find('\n') == std::string::npos);

  const std::string no = result_sentence(
      fast, context,
      {SpanFp8Verdict::State::Unavailable,
       "moteur TensorRT non installé : installez-le dans Paramètres > Export et moteurs > Moteurs"});
  CHECK(no.contains("Performance FP8 pour SPAN : non — moteur TensorRT non installé"));
  CHECK_FALSE(result_sentence(fast, context, {}).contains("FP8"));  // inconnu : la phrase n'en parle pas

  const std::string economy = result_sentence(BenchResult{.level = Tier::Economy}, context, {});
  CHECK(economy.contains("sans agrandissement"));
  CHECK(economy.contains("sans interpolation"));

  // L'exploration trouvée est dite : une hauteur plus haute que celle du niveau.
  BenchResult explored = fast;
  explored.work_height = 900;
  CHECK(result_sentence(explored, context, {}).contains("depuis 900p"));
}

TEST_CASE("défaire l'optimisation remet bench.toml comme il était, ou le retire", "[bench_steps]") {
  const fs::path dir = scratch("undo");
  const fs::path file = dir / "bench.toml";
  PlaySettings before;
  before.preset = Preset::Quality;

  // Pas de bench.toml avant : l'annulation le retire.
  BenchUndo none = capture_undo(file, before);
  CHECK(none.valid);
  CHECK_FALSE(none.bench_toml.has_value());
  CHECK(none.settings.preset == Preset::Quality);
  REQUIRE(
      save_bench(file, BenchResult{.gpu = "g", .driver = "d", .level = Tier::Fast, .date = "2026-10-29"}));
  REQUIRE(restore_bench_file(none, file));
  CHECK_FALSE(fs::exists(file));

  // Un bench.toml d'avant : il revient octet pour octet.
  REQUIRE(
      save_bench(file, BenchResult{.gpu = "g", .driver = "d", .level = Tier::Light, .date = "2026-10-01"}));
  const std::string old_text = slurp(file);
  BenchUndo some = capture_undo(file, before);
  REQUIRE(some.bench_toml.has_value());
  REQUIRE(
      save_bench(file, BenchResult{.gpu = "g", .driver = "d", .level = Tier::Ultra, .date = "2026-10-29"}));
  CHECK(slurp(file) != old_text);
  REQUIRE(restore_bench_file(some, file));
  CHECK(slurp(file) == old_text);
  fs::remove_all(dir);
}

TEST_CASE("la fenêtre des essais reste visible sous GNOME / Wayland, masquée ailleurs", "[bench_steps]") {
  CHECK_FALSE(hide_is_reliable("wayland", "GNOME"));
  CHECK_FALSE(hide_is_reliable("wayland", "ubuntu:GNOME"));
  CHECK_FALSE(hide_is_reliable("Wayland", "gnome"));
  CHECK(hide_is_reliable("x11", "GNOME"));
  CHECK(hide_is_reliable("wayland", "KDE"));
  CHECK(hide_is_reliable("", ""));
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
                                         .period_us = 16'667}};
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
  CHECK(lines.front() == "Étape 1 sur 4 : préparation des moteurs et première mesure");
  CHECK(lines[1] == "Étape 2 sur 4 : test des réglages « Rapide »");
  REQUIRE(outcome->notes.size() == 1);
  CHECK(outcome->notes.front().starts_with(
      "L'essai des réglages « Rapide » n'a rien rendu : refus : GPU saturé"));
  CHECK(outcome->notes.front().contains("code 3"));
  CHECK(shown.get().step == BenchStep::Levels);
}

TEST_CASE("un premier essai qui échoue donne un refus qui cite la dernière ligne d'erreur", "[bench_steps]") {
  const FakePlayer fake(0, true);
  const BenchPlan plan{.executable = fake.exe, .video = "/v/a.mkv", .work_dir = fake.dir / "w"};
  const auto outcome = run_bench(plan, std::stop_token{});
  REQUIRE_FALSE(outcome);
  CHECK(outcome.error().detail.starts_with(
      "L'essai de préparation n'a rien rendu : SDL_CreateWindow : aucun écran"));
}

TEST_CASE("annuler arrête l'évaluation vite, sans attendre l'essai en cours", "[bench_steps]") {
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
