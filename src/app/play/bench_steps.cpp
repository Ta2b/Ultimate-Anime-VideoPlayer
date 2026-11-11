#include "app/play/bench_steps.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>

#include "app/play/i18n.h"
#include "core/text.h"

namespace ajp::app::play {
namespace {

// Le poids de chaque étape dans la barre globale : la préparation et les essais pèsent le plus.
constexpr std::array<float, kBenchSteps> kWeights{0.3F, 0.4F, 0.3F};

bool looks_like_error(std::string_view line) {
  const std::string text = core::ascii_lower(line);
  // « échec » s'écrit en échappement : ce mot est lu dans le journal, pas affiché.
  for (const std::string_view word :
       {"[error]", "[critical]", "erreur", "refus", "\u00e9chec", "failed", "error"})
    if (text.contains(word)) return true;
  return false;
}

}  // namespace

std::string step_line(BenchStep step, std::string_view what) {
  return i18n::tr_format("Step {} of {}: {}", static_cast<int>(step), kBenchSteps, what);
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
    const std::string text = core::ascii_lower(*it);
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
      why = i18n::tr_format(
          "it ran over {} s without finishing, probably building an engine (slow the first time)",
          limit.count());
      break;
    case prepare::ExitKind::Crashed:
      why = i18n::tr_format("the test player stopped abruptly ({})", prepare::describe(exit));
      break;
    case prepare::ExitKind::Exited:
      if (exit.code == 0) return {};
      why = i18n::tr_format("the test player stopped ({})", prepare::describe(exit));
      break;
  }
  // Un délai dépassé n'a pas de ligne d'erreur : la dernière ligne n'est qu'un message de progression.
  if (!problem.empty() && exit.kind != prepare::ExitKind::TimedOut)
    why = std::format("{} — {}", problem, why);
  return i18n::tr_format("Test {} produced nothing: {}", what, why);
}

std::string explain_no_reading(std::string_view what, std::string_view detail) {
  return i18n::tr_format("Test {} produced nothing: it ended without readable measurements{}{}", what,
                         detail.empty() ? "" : " — ", detail);
}

}  // namespace ajp::app::play
