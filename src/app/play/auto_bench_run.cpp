#include "app/play/auto_bench_run.h"

#include <algorithm>
#include <format>
#include <system_error>

#include "core/log.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Un essai : l'enfant joue, rend sa lecture ; une lecture absente est un essai vide (ne tient pas).
std::optional<BenchReading> play_one(const BenchPlan& plan, const std::optional<Tier>& level, int index,
                                     const std::stop_token& stop, bool& cancelled) {
  const std::filesystem::path dir = plan.work_dir / std::format("essai-{}", index);
  std::error_code error;
  std::filesystem::remove_all(dir, error);
  std::filesystem::create_directories(dir, error);
  const std::filesystem::path out = dir / "mesures.toml";
  prepare::ProcessSpec spec{.program = plan.executable,
                            .args = bench_child_args(plan, level, dir / "reglages", out),
                            .timeout = std::chrono::seconds(plan.seconds) + plan.build_allowance};
  const auto log = log_channel("app");
  const auto exit = prepare::run_process(spec, [&](std::string_view line) { log->info("essai : {}", line); }, stop);
  std::optional<BenchReading> reading;
  if (!exit) {
    log->warn("essai : {}", exit.error().detail);
  } else if (exit->kind == prepare::ExitKind::Cancelled) {
    cancelled = true;
  } else if (exit->kind != prepare::ExitKind::Exited || exit->code != 0) {
    log->warn("essai : {}", prepare::describe(*exit));
  } else if (auto read = load_reading(out)) {
    reading = std::move(*read);
  } else {
    log->warn("essai : {}", read.error().detail);
  }
  std::filesystem::remove_all(dir, error);
  return reading;
}

}  // namespace

std::vector<std::string> bench_child_args(const BenchPlan& plan, const std::optional<Tier>& level,
                                          const std::filesystem::path& data_dir,
                                          const std::filesystem::path& out_file) {
  std::vector<std::string> args{"play",
                                prepare::utf8_path(plan.video),
                                std::format("--seconds={}", plan.seconds),
                                "--mute",
                                "--no-resume",
                                std::format("--data-dir={}", prepare::utf8_path(data_dir)),
                                std::format("--bench-out={}", prepare::utf8_path(out_file)),
                                std::format("--bench-level={}", level ? std::to_string(static_cast<int>(*level)) : "auto")};
  if (plan.fullscreen) args.emplace_back("--fullscreen");
  return args;
}

std::expected<BenchOutcome, Refusal> run_bench(const BenchPlan& plan, const std::stop_token& stop,
                                               const std::function<void(const std::string&)>& progress) {
  BenchOutcome outcome;
  std::string gpu;
  std::string driver;
  std::optional<Tier> level;  // absent : « auto », le premier essai
  for (int index = 0; index <= kTierCount; ++index) {
    if (progress) progress(level ? std::format("Essai : niveau {}", tier_label(*level)) : "Essai : niveau estimé");
    bool cancelled = false;
    const auto reading = play_one(plan, level, index, stop, cancelled);
    if (cancelled) return refuse(RefusalCode::Cancelled, "évaluation annulée");
    if (!level) {
      if (!reading) return refuse(RefusalCode::DecodeFailed, "le premier essai n'a rien rendu : la machine n'a pas pu jouer");
      outcome.estimated = reading->run.level;
      gpu = reading->gpu;
      driver = reading->driver;
    }
    outcome.runs.push_back(reading ? reading->run : BenchRun{.level = *level});  // vide : ne tient pas
    level = next_level(outcome.runs, outcome.estimated);
    if (!level) break;
  }
  outcome.result = BenchResult{.gpu = gpu, .driver = driver, .level = best_level(outcome.runs)};
  return outcome;
}

std::expected<std::string, Refusal> evaluate_machine(const BenchPlan& plan, const std::filesystem::path& data_dir,
                                                     const std::stop_token& stop,
                                                     const std::function<void(const std::string&)>& progress) {
  auto outcome = run_bench(plan, stop, progress);
  if (!outcome) return std::unexpected(std::move(outcome.error()));
  BenchResult result = outcome->result;
  result.date = std::format("{:%F}", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()));
  std::string report = format_bench_report(*outcome);
  if (const auto saved = save_bench(data_dir / "bench.toml", result); !saved)
    report += "bench.toml : " + saved.error().detail + "\n";
  return report;
}

std::string format_bench_report(const BenchOutcome& outcome) {
  std::string out = std::format("Carte : {} (pilote {})\nNiveau estimé : {}\n", outcome.result.gpu,
                                outcome.result.driver.empty() ? "inconnu" : outcome.result.driver,
                                tier_label(outcome.estimated));
  for (const BenchRun& run : outcome.runs) {
    const double lost = run.presents > 0
                            ? 100.0 * static_cast<double>(run.dropped + run.missing) / static_cast<double>(run.presents)
                            : 100.0;
    out += std::format("  {:<8} {:>5} présentations, {:>5.1f} % perdues, retard p95 {:>6.1f} ms : {}\n",
                       tier_label(run.level), run.presents, lost, static_cast<double>(run.lag_p95_us) / 1000.0,
                       holds(run) ? "tient" : "ne tient pas");
  }
  out += std::format("Niveau gardé : {}\n", tier_label(outcome.result.level));
  return out;
}

}  // namespace ajp::app::play
