#include "app/play/auto_bench_run.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <system_error>

#include "app/play/i18n.h"
#include "core/log.h"
#include "media/bench_clip.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string lowered(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

// Ce qu'un essai a donné : sa lecture, ou pourquoi il n'en a pas donné.
struct Trial {
  std::optional<BenchReading> reading;
  bool cancelled = false;
  bool timed_out = false;
  std::chrono::milliseconds elapsed{};
  std::string why;  // une phrase pour l'utilisateur quand il n'y a pas de lecture
};

bool mentions_build(std::string_view line) {
  const std::string text = lowered(line);
  return text.contains("construction") || text.contains("construit") || text.contains("engine built");
}

// Les options de fenêtre d'un essai : masquée quand le bureau le permet, sinon petite et visible, jamais en
// plein écran (une fenêtre plein écran qui apparaît à chaque essai serait pire que le masquage manqué).
void window_args(const BenchPlan& plan, std::vector<std::string>& args) {
  if (plan.silent) {
    args.emplace_back(hide_is_reliable_here() ? "--hidden" : "--window=320x180");
    return;
  }
  if (plan.fullscreen) args.emplace_back("--fullscreen");
}

}  // namespace

bool hide_is_reliable(std::string_view session_type, std::string_view current_desktop) {
  return !(lowered(session_type) == "wayland" && lowered(current_desktop).contains("gnome"));
}

bool hide_is_reliable_here() {
  // Le bureau ne se lit pas dans l'environnement (règle « aucune bascule cachée ») : sous Linux, où GNOME
  // Wayland ne se distingue pas des autres, l'essai prend la petite fenêtre visible, sûre partout.
#if defined(__linux__)
  return false;
#else
  return true;
#endif
}

std::vector<std::string> bench_child_args(const BenchPlan& plan, const std::optional<Tier>& level,
                                          const std::filesystem::path& data_dir,
                                          const std::filesystem::path& out_file,
                                          const std::vector<std::string>& extra) {
  std::vector<std::string> args{
      "play",
      prepare::utf8_path(plan.video),
      std::format("--seconds={}", plan.seconds),
      "--mute",
      "--no-resume",
      std::format("--data-dir={}", prepare::utf8_path(data_dir)),
      std::format("--bench-out={}", prepare::utf8_path(out_file)),
      std::format("--bench-level={}", level ? std::to_string(static_cast<int>(*level)) : "auto")};
  window_args(plan, args);
  args.insert(args.end(), extra.begin(), extra.end());
  return args;
}

std::vector<std::string> variant_child_args(const BenchPlan& plan, const Variant& variant,
                                            const std::filesystem::path& data_dir,
                                            const std::filesystem::path& out_file) {
  std::vector<std::string> args{"play",
                                prepare::utf8_path(plan.video),
                                std::format("--seconds={}", plan.seconds),
                                "--mute",
                                "--no-resume",
                                std::format("--data-dir={}", prepare::utf8_path(data_dir)),
                                std::format("--bench-out={}", prepare::utf8_path(out_file))};
  window_args(plan, args);
  args.insert(args.end(), variant.args.begin(), variant.args.end());
  return args;
}

namespace {

// Un essai : l'enfant joue, rend sa lecture ; une lecture absente est un essai vide (ne tient pas), dit en
// une phrase (what : « des réglages « Rapide » »). allowance : le délai de plus que la durée de l'essai.
Trial play_one(const BenchPlan& plan, const std::optional<Tier>& level, const Variant* variant, int index,
               const std::stop_token& stop, std::chrono::seconds allowance, const std::string& what,
               const std::vector<std::string>& extra = {}) {
  Trial trial;
  const std::filesystem::path dir = plan.work_dir / std::format("essai-{}", index);
  std::error_code error;
  std::filesystem::remove_all(dir, error);
  std::filesystem::create_directories(dir, error);
  const std::filesystem::path out = dir / "mesures.toml";
  // Le dossier de réglages jetable de l'essai : les moteurs choisis par l'utilisateur et l'assistant déjà vu
  // (sinon chaque essai rouvrirait l'écran de bienvenue et n'avancerait pas), puis le play.toml de la
  // variante.
  std::filesystem::create_directories(dir / "reglages", error);
  if (!plan.data.empty() && std::filesystem::exists(plan.data / "settings.toml", error)) {
    std::filesystem::copy_file(plan.data / "settings.toml", dir / "reglages" / "settings.toml", error);
  } else {
    std::ofstream(dir / "reglages" / "settings.toml")
        << "version = 1\nfirst_run_done = true\nfirst_run_step = 8\n";
  }
  if (variant != nullptr && !variant->settings.empty())
    std::filesystem::copy_file(variant->settings, dir / "reglages" / "play.toml", error);
  const std::chrono::seconds limit = std::chrono::seconds(plan.seconds) + allowance;
  prepare::ProcessSpec spec{.program = plan.executable,
                            .args = variant != nullptr
                                        ? variant_child_args(plan, *variant, dir / "reglages", out)
                                        : bench_child_args(plan, level, dir / "reglages", out, extra),
                            .timeout = limit};
  const auto log = log_channel("app");
  ChildTail tail;
  const auto exit = prepare::run_process(
      spec,
      [&](std::string_view line) {
        tail.add(line);
        log->info("essai : {}", line);
        if (plan.shown != nullptr && mentions_build(line))
          plan.shown->set_note(
              i18n::tr("Building a compute engine: first time only, probably several minutes on a laptop."));
      },
      stop);
  if (!exit) {
    trial.why = i18n::tr_format("Test {} could not start: {}", what, exit.error().detail);
  } else {
    trial.elapsed = exit->elapsed;
    if (exit->kind == prepare::ExitKind::Cancelled) {
      trial.cancelled = true;
    } else if (exit->kind != prepare::ExitKind::Exited || exit->code != 0) {
      trial.timed_out = exit->kind == prepare::ExitKind::TimedOut;
      trial.why = explain_trial(what, *exit, tail.problem(), limit);
      // Le code de sortie et les dernières lignes de l'enfant : de quoi conclure sans refaire l'essai.
      log->warn("essai {} : {} ; dernières lignes de l'enfant :\n{}", what, prepare::describe(*exit),
                tail.text());
    } else if (auto read = load_reading(out)) {
      trial.reading = std::move(*read);
    } else {
      trial.why = explain_no_reading(what, tail.problem());
      log->warn("essai {} : {} ; dernières lignes de l'enfant :\n{}", what, read.error().detail, tail.text());
    }
  }
  if (!trial.why.empty()) log->warn("{}", trial.why);
  std::filesystem::remove_all(dir, error);
  return trial;
}

std::string level_what(const std::optional<Tier>& level) {
  return level ? i18n::tr_format("of the \"{}\" settings", tier_label(*level))
               : std::string(i18n::tr("of preparation"));
}

// Le pire de deux passages : plus de pertes, plus de retard, plus long.
BenchReading worse(const BenchReading& x, const BenchReading& y) {
  BenchReading out = x;
  out.run.dropped = std::max(x.run.dropped, y.run.dropped);
  out.run.missing = std::max(x.run.missing, y.run.missing);
  out.run.lag_p95_us = std::max(x.run.lag_p95_us, y.run.lag_p95_us);
  out.run.presents = std::min(x.run.presents, y.run.presents);
  out.interval_median_us = std::max(x.interval_median_us, y.interval_median_us);
  out.interval_p95_us = std::max(x.interval_p95_us, y.interval_p95_us);
  out.render_median_us = std::max(x.render_median_us, y.render_median_us);
  return out;
}

}  // namespace

std::expected<CompareOutcome, Refusal> run_compare(const BenchPlan& plan, const Variant& a, const Variant& b,
                                                   const std::stop_token& stop,
                                                   const std::function<void(const std::string&)>& progress) {
  const Variant* order[4] = {&a, &b, &b, &a};
  std::optional<BenchReading> got[2];
  for (int index = 0; index < 4; ++index) {
    const Variant& variant = *order[index];
    if (progress) progress(i18n::tr_format("Test {}/4: {}", index + 1, variant.name));
    Trial trial = play_one(plan, std::nullopt, &variant, index, stop, plan.first_allowance,
                           i18n::tr_format("of the \"{}\" variant", variant.name));
    if (trial.cancelled) return refuse(RefusalCode::Cancelled, i18n::tr("comparison cancelled"));
    if (!trial.reading)
      return refuse(RefusalCode::DecodeFailed,
                    trial.why.empty() ? i18n::tr_format("the \"{}\" variant produced nothing", variant.name)
                                      : trial.why);
    std::optional<BenchReading>& slot = got[&variant == &a ? 0 : 1];
    slot = slot ? worse(*slot, *trial.reading) : std::move(trial.reading);
  }
  const std::optional<BenchReading>& first = got[0];
  const std::optional<BenchReading>& second = got[1];
  if (!first || !second)
    return refuse(RefusalCode::DecodeFailed, i18n::tr("comparison: a variant produced nothing"));
  return CompareOutcome{.a = a, .b = b, .reading_a = *first, .reading_b = *second};
}

std::string format_compare_report(const CompareOutcome& outcome) {
  const auto line = [](const Variant& variant, const BenchReading& r) {
    const double lost = r.run.presents > 0 ? 100.0 * static_cast<double>(r.run.dropped + r.run.missing) /
                                                 static_cast<double>(r.run.presents)
                                           : 100.0;
    return i18n::tr_format(
        "  {:<24} {:>5.1f} % lost, p95 delay {:>6.1f} ms, frame {:>6.1f} ms (p95 {:>6.1f}), compositing "
        "{:>5.1f} ms: {}\n",
        variant.name, lost, static_cast<double>(r.run.lag_p95_us) / 1000.0,
        static_cast<double>(r.interval_median_us) / 1000.0, static_cast<double>(r.interval_p95_us) / 1000.0,
        static_cast<double>(r.render_median_us) / 1000.0, holds(r.run) ? "tient" : i18n::tr("does not hold"));
  };
  std::string out = i18n::tr_format("Card: {}\n", outcome.reading_a.gpu) +
                    line(outcome.a, outcome.reading_a) + line(outcome.b, outcome.reading_b);
  const bool a_holds = holds(outcome.reading_a.run);
  const bool b_holds = holds(outcome.reading_b.run);
  if (a_holds != b_holds) {
    out += i18n::tr_format("Only \"{}\" holds.\n", a_holds ? outcome.a.name : outcome.b.name);
  } else {
    const auto delta = outcome.reading_b.interval_p95_us - outcome.reading_a.interval_p95_us;
    out += i18n::tr_format("At p95, \"{}\" is lighter by {:.1f} ms per frame.\n",
                           delta >= 0 ? outcome.a.name : outcome.b.name,
                           static_cast<double>(delta >= 0 ? delta : -delta) / 1000.0);
  }
  return out;
}

std::expected<BenchOutcome, Refusal> run_bench(const BenchPlan& plan, const std::stop_token& stop,
                                               const std::function<void(const std::string&)>& progress) {
  const auto say = [&](BenchStep step, const std::string& what, float within, const std::string& note = {}) {
    if (plan.shown != nullptr) plan.shown->set(step, what, within, note);
    if (progress) progress(step_line(step, what));
  };
  const auto log = log_channel("app");
  BenchOutcome outcome;
  std::string gpu;
  std::string driver;
  std::optional<Tier> level;                          // absent : « auto », le premier essai
  std::vector<std::optional<BenchReading>> readings;  // une par essai, comme outcome.runs
  int index = 0;
  bool any_held = false;
  std::chrono::milliseconds first_elapsed{};
  for (; index <= kTierCount; ++index) {
    const bool first = outcome.runs.empty();
    const Tier current = level.value_or(outcome.estimated);  // le niveau joué (le premier essai : l'estimé)
    if (first) {
      say(BenchStep::Prepare, i18n::tr("preparing engines and first measurement"), 0.1F,
          i18n::tr("If this is the first time on this PC, building the engines can take several minutes."));
    } else {
      say(BenchStep::Levels, i18n::tr_format("testing the \"{}\" settings", tier_label(current)),
          std::min(0.9F, static_cast<float>(outcome.runs.size() - 1) / 3.0F));
    }
    // Tant qu'aucun niveau n'a tenu, le résultat dépend de cet essai : il a le temps qu'il lui faut. Après,
    // un niveau dont les moteurs sont trop longs à construire est dit non testé.
    const std::chrono::seconds allowance = any_held ? plan.build_allowance : plan.first_allowance;
    Trial trial = play_one(plan, level, nullptr, index, stop, allowance, level_what(level));
    if (trial.cancelled) return refuse(RefusalCode::Cancelled, i18n::tr("evaluation cancelled"));
    if (first) {
      if (!trial.reading) return refuse(RefusalCode::DecodeFailed, trial.why);
      outcome.estimated = trial.reading->run.level;
      gpu = trial.reading->gpu;
      driver = trial.reading->driver;
      first_elapsed = trial.elapsed;
    }
    if (!first && !trial.reading && trial.timed_out && any_held) {
      outcome.notes.push_back(
          i18n::tr_format("The \"{}\" settings were not tested: their engines take long to build. Run the "
                          "optimization again to continue.",
                          tier_label(current)));
      log->warn("{}", outcome.notes.back());
      break;
    }
    if (!trial.reading) outcome.notes.push_back(trial.why);
    outcome.runs.push_back(trial.reading ? trial.reading->run
                                         : BenchRun{.level = current});  // vide : ne tient pas
    readings.push_back(trial.reading);
    if (holds(outcome.runs.back())) any_held = true;
    level = next_level(outcome.runs, outcome.estimated);
    if (!level) break;
  }
  outcome.result = BenchResult{.gpu = gpu, .driver = driver, .level = best_level(outcome.runs)};

  // L'exploration, une fois le niveau connu : la hauteur de travail au-dessus de celle du niveau tant que la
  // machine tient, puis le modèle RIFE au-dessus du modèle léger. Le multiplicateur reste au mode Auto, qui
  // le règle en direct. Un essai qui ne rend rien ou ne tient pas arrête la piste. Elle ne construit pas de
  // moteur : si les moteurs viennent d'être construits (le premier essai a été long), elle est remise à la
  // prochaine optimisation, où ils sont en cache.
  std::optional<BenchReading> base;
  for (std::size_t i = 0; i < readings.size(); ++i) {
    if (readings[i] && outcome.runs[i].level == outcome.result.level && holds(outcome.runs[i]))
      base = readings[i];
  }
  if (base && outcome.result.level != Tier::Economy && base->work_height > 0) {
    if (first_elapsed > std::chrono::seconds(plan.seconds) + std::chrono::seconds(90)) {
      outcome.notes.push_back(i18n::tr(
          "The search for a finer picture is left to the next optimization: the engines were just built."));
      log->info("{}", outcome.notes.back());
    } else {
      bool cancelled = false;
      int tried = 0;
      const auto explore = [&](const std::string& what, const std::vector<std::string>& extra) {
        say(BenchStep::Explore, i18n::tr_format("searching for a finer picture ({})", what),
            std::min(0.9F, static_cast<float>(tried) / 3.0F));
        Trial trial = play_one(plan, outcome.result.level, nullptr, ++index + tried++, stop,
                               plan.explore_allowance, i18n::tr_format("of search ({})", what), extra);
        if (trial.cancelled) {
          cancelled = true;
          return false;
        }
        return trial.reading && holds(trial.reading->run);
      };
      int height = base->work_height;
      while (const auto next = next_work_height(height, kExplorationSourceHeight)) {
        if (!explore(std::format("{}p", *next), {std::format("--work-height={}", *next)})) break;
        height = *next;
        outcome.result.work_height = height;
      }
      if (!cancelled) {
        if (const auto better = next_rife_model(base->rife_model)) {
          std::vector<std::string> extra{std::format("--rife-model={}", *better)};
          if (outcome.result.work_height > 0)
            extra.push_back(std::format("--work-height={}", outcome.result.work_height));
          if (explore(*better, extra)) outcome.result.rife_model = *better;
        }
      }
      if (cancelled || stop.stop_requested())
        return refuse(RefusalCode::Cancelled, i18n::tr("evaluation cancelled"));
    }
  }
  return outcome;
}

std::expected<Evaluation, Refusal> evaluate_machine(const BenchPlan& plan,
                                                    const std::filesystem::path& data_dir,
                                                    const std::stop_token& stop,
                                                    const std::function<void(const std::string&)>& progress) {
  // Sans vidéo, le lecteur écrit lui-même la séquence de l'évaluation, une fois pour toutes
  // (media/bench_clip.h).
  BenchPlan own = plan;
  if (own.video.empty()) {
    own.video = data_dir / "bench-clip-v2.mkv";
    std::error_code error;
    if (!std::filesystem::is_regular_file(own.video, error)) {
      if (plan.shown != nullptr)
        plan.shown->set(BenchStep::Prepare, i18n::tr("writing the test sequence"), 0.0F);
      if (auto written = media::write_bench_clip(own.video); !written)
        return std::unexpected(std::move(written.error()));
    }
  }
  auto outcome = run_bench(own, stop, progress);
  if (!outcome) return std::unexpected(std::move(outcome.error()));
  Evaluation done;
  done.result = outcome->result;
  done.result.date =
      std::format("{:%F}", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()));
  done.report = format_bench_report(*outcome);
  if (const auto saved = save_bench(data_dir / "bench.toml", done.result); !saved)
    done.report += i18n::tr("bench.toml: ") + saved.error().detail + "\n";
  for (const std::string& note : outcome->notes) done.report += note + "\n";
  done.outcome = std::move(*outcome);
  log_channel("app")->info("évaluation de la machine :\n{}", done.report);  // i18n-ok : journal
  return done;
}

std::string format_bench_report(const BenchOutcome& outcome) {
  std::string out = i18n::tr_format("Card: {} (driver {})\nEstimated level: {}\n", outcome.result.gpu,
                                    outcome.result.driver.empty() ? "inconnu" : outcome.result.driver,
                                    tier_label(outcome.estimated));
  for (const BenchRun& run : outcome.runs) {
    const double lost = run.presents > 0 ? 100.0 * static_cast<double>(run.dropped + run.missing) /
                                               static_cast<double>(run.presents)
                                         : 100.0;
    out += i18n::tr_format("  {:<8} {:>5} presentations, {:>5.1f} % lost, p95 delay {:>6.1f} ms: {}\n",
                           tier_label(run.level), run.presents, lost,
                           static_cast<double>(run.lag_p95_us) / 1000.0,
                           holds(run) ? "tient" : i18n::tr("does not hold"));
  }
  out += i18n::tr_format("Level kept: {}\n", tier_label(outcome.result.level));
  if (outcome.result.work_height > 0)
    out += i18n::tr_format("Exploration: the machine holds a working height of {}p.\n",
                           outcome.result.work_height);
  if (!outcome.result.rife_model.empty())
    out += i18n::tr_format("Exploration: the machine holds the {} model.\n", outcome.result.rife_model);
  return out;
}

}  // namespace ajp::app::play
