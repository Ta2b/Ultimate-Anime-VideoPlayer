#include "app/play/bench_steps.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <fstream>
#include <sstream>

#include "app/play/auto_config.h"

namespace ajp::app::play {
namespace {

// Le poids de chaque étape dans la barre globale : la préparation et les essais pèsent le plus.
constexpr std::array<float, kBenchSteps> kWeights{0.25F, 0.35F, 0.25F, 0.15F};

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

bool looks_like_error(std::string_view line) {
  const std::string text = lower(line);
  for (const std::string_view word : {"[error]", "[critical]", "erreur", "refus", "échec", "failed", "error"})
    if (text.contains(word)) return true;
  return false;
}

}  // namespace

std::string step_line(BenchStep step, std::string_view what) {
  return std::format("Étape {} sur {} : {}", static_cast<int>(step), kBenchSteps, what);
}

float overall_fraction(BenchStep step, float within) {
  const auto at = static_cast<std::size_t>(std::clamp(static_cast<int>(step), 1, kBenchSteps) - 1);
  float before = 0.0F;
  for (std::size_t i = 0; i < at; ++i) before += kWeights[i];
  return std::clamp(before + kWeights[at] * std::clamp(within, 0.0F, 1.0F), 0.0F, 1.0F);
}

void BenchProgress::set(BenchStep step, std::string what, float within, std::string note) {
  const std::scoped_lock lock(mutex_);
  step_ = step;
  what_ = std::move(what);
  within_ = within;
  note_ = std::move(note);
}

void BenchProgress::set_within(float within) {
  const std::scoped_lock lock(mutex_);
  within_ = within;
}

void BenchProgress::set_note(std::string note) {
  const std::scoped_lock lock(mutex_);
  note_ = std::move(note);
}

BenchProgress::View BenchProgress::get() const {
  const std::scoped_lock lock(mutex_);
  return View{.step = step_, .what = what_, .fraction = overall_fraction(step_, within_), .note = note_};
}

std::string strip_log_prefix(std::string_view line) {
  std::size_t at = 0;
  // Les groupes « [...] » du début : horodatage, canal, niveau. Un essai en imbrique plusieurs (sa ligne est
  // reprise par le journal du parent).
  while (at < line.size() && line[at] == '[') {
    const std::size_t close = line.find(']', at);
    if (close == std::string_view::npos) break;
    at = close + 1;
    while (at < line.size() && line[at] == ' ') ++at;
  }
  std::string out(line.substr(at));
  constexpr std::size_t kMax = 200;
  if (out.size() > kMax) {
    out.resize(kMax);
    while (!out.empty() && (static_cast<unsigned char>(out.back()) & 0xC0U) == 0x80U) out.pop_back();
    out += "…";
  }
  return out;
}

void ChildTail::add(std::string_view line) {
  if (line.empty()) return;
  lines_.emplace_back(line);
  while (lines_.size() > keep_) lines_.pop_front();
}

std::string ChildTail::problem() const {
  // Une ligne que le journal a marquée [error] ou [critical] d'abord, puis une qui parle d'erreur.
  for (auto it = lines_.rbegin(); it != lines_.rend(); ++it) {
    const std::string text = lower(*it);
    if (text.contains("[error]") || text.contains("[critical]")) return strip_log_prefix(*it);
  }
  for (auto it = lines_.rbegin(); it != lines_.rend(); ++it)
    if (looks_like_error(*it)) return strip_log_prefix(*it);
  return lines_.empty() ? std::string() : strip_log_prefix(lines_.back());
}

std::string ChildTail::text() const {
  std::string out;
  for (const std::string& line : lines_) out += line + "\n";
  return out;
}

std::string explain_trial(std::string_view what, const prepare::ProcessExit& exit, std::string_view problem,
                          std::chrono::seconds limit) {
  std::string why;
  switch (exit.kind) {
    case prepare::ExitKind::Cancelled:
      return {};
    case prepare::ExitKind::TimedOut:
      why = std::format(
          "il a dépassé {} s sans finir, sans doute en construisant un moteur de calcul (c'est long la "
          "première "
          "fois)",
          limit.count());
      break;
    case prepare::ExitKind::Crashed:
      why = std::format("le lecteur de l'essai s'est arrêté brutalement ({})", prepare::describe(exit));
      break;
    case prepare::ExitKind::Exited:
      if (exit.code == 0) return {};
      why = std::format("le lecteur de l'essai s'est arrêté ({})", prepare::describe(exit));
      break;
  }
  // Un délai dépassé n'a pas de ligne d'erreur : la dernière ligne n'est qu'un message de progression.
  if (!problem.empty() && exit.kind != prepare::ExitKind::TimedOut)
    why = std::format("{} — {}", problem, why);
  return std::format("L'essai {} n'a rien rendu : {}", what, why);
}

std::string explain_no_reading(std::string_view what, std::string_view detail) {
  return std::format("L'essai {} n'a rien rendu : il s'est terminé sans mesures lisibles{}{}", what,
                     detail.empty() ? "" : " — ", detail);
}

std::string result_sentence(const BenchResult& result, const ChainContext& context,
                            const SpanFp8Verdict& fp8) {
  PlaySettings base;
  base.preset = Preset::Auto;
  base.priority = Priority::Balance;
  const PlaySettings chain = apply_preset(base,
                                          Machine{.tier = result.level,
                                                  .display_height = context.display_height,
                                                  .display_hz = context.display_hz,
                                                  .explored_work_height = result.work_height,
                                                  .explored_rife_model = result.rife_model},
                                          core::SourceInfo{.width = context.source_height * 16 / 9,
                                                           .height = context.source_height,
                                                           .fps = context.source_fps});
  std::vector<std::string> parts;
  if (chain.upscale)
    parts.push_back(std::format("agrandissement ×2 depuis {}p", chain.work_height));
  else
    parts.push_back("image d'origine sans agrandissement");
  if (chain.rife_factor >= 2)
    parts.push_back(
        std::format("interpolation jusqu'à ×{}, {}", chain.rife_factor, label(chain.motion_engine)));
  else
    parts.push_back("sans interpolation");
  std::string joined;
  for (const std::string& part : parts) joined += (joined.empty() ? "" : ", ") + part;
  std::string out = std::format("Votre PC tient : {}.", joined);
  switch (fp8.state) {
    case SpanFp8Verdict::State::Ready:
      out += " Performance FP8 pour SPAN : prête.";
      break;
    case SpanFp8Verdict::State::Unavailable:
      out +=
          std::format(" Performance FP8 pour SPAN : non{}{}.", fp8.reason.empty() ? "" : " — ", fp8.reason);
      break;
    case SpanFp8Verdict::State::Unknown:
      break;
  }
  return out;
}

BenchUndo capture_undo(const std::filesystem::path& bench_file, const PlaySettings& settings) {
  BenchUndo undo{.valid = true, .settings = settings};
  std::error_code error;
  if (std::filesystem::is_regular_file(bench_file, error)) {
    std::ifstream in(bench_file, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    undo.bench_toml = text.str();
  }
  return undo;
}

std::expected<void, Refusal> restore_bench_file(const BenchUndo& undo,
                                                const std::filesystem::path& bench_file) {
  std::error_code error;
  if (!undo.bench_toml) {
    std::filesystem::remove(bench_file, error);
    if (error) return refuse(RefusalCode::IoFailed, "bench.toml : " + error.message());
    return {};
  }
  std::ofstream out(bench_file, std::ios::binary | std::ios::trunc);
  out << *undo.bench_toml;
  out.close();
  if (!out) return refuse(RefusalCode::IoFailed, "bench.toml : écriture impossible");
  return {};
}

}  // namespace ajp::app::play
