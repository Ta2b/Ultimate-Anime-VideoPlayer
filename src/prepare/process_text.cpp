// Parties portables du lanceur : découpe en lignes, citation Windows, descriptions.
#include <format>
#include <utility>

#include "prepare/process.h"

namespace ajp::prepare {

LineSplitter::LineSplitter(LineSink sink, std::size_t max_line)
    : sink_(std::move(sink)), max_line_(max_line) {}

void LineSplitter::feed(std::string_view bytes) {
  for (const char c : bytes) {
    if (c == '\n' || c == '\r') {
      emit();
      continue;
    }
    pending_ += c;
    if (pending_.size() >= max_line_) emit();
  }
}

void LineSplitter::finish() {
  emit();
}

void LineSplitter::emit() {
  if (!pending_.empty() && sink_) sink_(pending_);
  pending_.clear();
}

std::string windows_command_line(const std::vector<std::string>& argv) {
  // Règles de CommandLineToArgvW : 2n barres obliques suivies d'un guillemet
  // donnent n barres et ouvrent ou ferment ; 2n+1 donnent n barres et un guillemet.
  std::string line;
  for (const std::string& arg : argv) {
    if (!line.empty()) line += ' ';
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
      line += arg;
      continue;
    }
    line += '"';
    std::size_t backslashes = 0;
    for (const char c : arg) {
      if (c == '\\') {
        ++backslashes;
        continue;
      }
      if (c == '"') {
        line.append(backslashes * 2 + 1, '\\');
      } else {
        line.append(backslashes, '\\');
      }
      backslashes = 0;
      line += c;
    }
    line.append(backslashes * 2, '\\');
    line += '"';
  }
  return line;
}

std::string describe(const ProcessExit& exit) {
  const auto seconds = std::chrono::duration<double>(exit.elapsed).count();
  switch (exit.kind) {
    case ExitKind::Exited:
      return std::format("code {} après {:.1f} s", exit.code, seconds);
    case ExitKind::Crashed:
#ifdef _WIN32
      return std::format("plantage (exception {:#x}) après {:.1f} s", exit.code, seconds);
#else
      return std::format("plantage (signal {}) après {:.1f} s", exit.code, seconds);
#endif
    case ExitKind::TimedOut:
      return std::format("délai dépassé après {:.1f} s", seconds);
    case ExitKind::Cancelled:
      return std::format("annulé après {:.1f} s", seconds);
  }
  return "fin inconnue";
}

}  // namespace ajp::prepare
