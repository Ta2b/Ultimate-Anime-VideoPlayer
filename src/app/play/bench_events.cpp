#include "app/play/bench_events.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <format>
#include <mutex>
#include <utility>

namespace ajp::app::play {
namespace {

std::atomic<bool> g_child{false};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void put_string(std::string& out, std::string_view text) {
  out += '"';
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
    } else {
      out += c;
    }
  }
  out += '"';
}

// Une chaîne JSON qui commence à text[at] ; at passe après elle. Les échappements \uXXXX au-dessus de 0x7f ne
// viennent pas de bench_event_line, qui garde l'UTF-8 tel quel : refusés.
std::optional<std::string> take_string(std::string_view text, std::size_t& at) {
  if (at >= text.size() || text[at] != '"') return std::nullopt;
  std::string out;
  for (++at; at < text.size(); ++at) {
    const char c = text[at];
    if (c == '"') {
      ++at;
      return out;
    }
    if (c != '\\') {
      out += c;
      continue;
    }
    if (++at >= text.size()) return std::nullopt;
    const char escaped = text[at];
    if (escaped == 'u') {
      if (at + 4 >= text.size()) return std::nullopt;
      unsigned value = 0;
      for (int i = 1; i <= 4; ++i) {
        const char h = text[at + static_cast<std::size_t>(i)];
        const int digit = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : -1;
        if (digit < 0) return std::nullopt;
        value = value * 16 + static_cast<unsigned>(digit);
      }
      if (value > 0x7f) return std::nullopt;
      out += static_cast<char>(value);
      at += 4;
    } else if (escaped == '"' || escaped == '\\' || escaped == '/') {
      out += escaped;
    } else if (escaped == 'n') {
      out += '\n';
    } else if (escaped == 't') {
      out += '\t';
    } else {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

void skip_spaces(std::string_view text, std::size_t& at) {
  while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\r')) ++at;
}

}  // namespace

std::string bench_event_line(const ChildBenchEvent& event) {
  std::string out(kBenchEventTag);
  out += "{";
  bool first = true;
  for (const auto& [key, value] : std::array<std::pair<std::string_view, const std::string*>, 3>{
           {{"event", &event.event}, {"code", &event.code}, {"detail", &event.detail}}}) {
    if (value->empty() && key != "event") continue;
    if (!first) out += ',';
    first = false;
    put_string(out, key);
    out += ':';
    put_string(out, *value);
  }
  out += "}";
  return out;
}

std::optional<ChildBenchEvent> parse_bench_event(std::string_view line) {
  if (!line.starts_with(kBenchEventTag)) return std::nullopt;
  line.remove_prefix(kBenchEventTag.size());
  std::size_t at = 0;
  skip_spaces(line, at);
  if (at >= line.size() || line[at] != '{') return std::nullopt;
  ++at;
  ChildBenchEvent event;
  skip_spaces(line, at);
  if (at < line.size() && line[at] == '}') return std::nullopt;  // sans « event »
  while (true) {
    skip_spaces(line, at);
    auto key = take_string(line, at);
    skip_spaces(line, at);
    if (!key || at >= line.size() || line[at] != ':') return std::nullopt;
    ++at;
    skip_spaces(line, at);
    auto value = take_string(line, at);
    if (!value) return std::nullopt;
    if (*key == "event") {
      event.event = std::move(*value);
    } else if (*key == "code") {
      event.code = std::move(*value);
    } else if (*key == "detail") {
      event.detail = std::move(*value);
    }  // une clé d'une version plus récente est ignorée
    skip_spaces(line, at);
    if (at < line.size() && line[at] == ',') {
      ++at;
      continue;
    }
    if (at < line.size() && line[at] == '}') break;
    return std::nullopt;
  }
  if (event.event.empty()) return std::nullopt;
  return event;
}

void set_bench_child(bool child) {
  g_child.store(child, std::memory_order_release);
}

void tell_bench_parent(const ChildBenchEvent& event) {
  if (!g_child.load(std::memory_order_acquire)) return;
  static std::mutex out;  // une ligne entière à la fois
  const std::string line = bench_event_line(event) + "\n";
  const std::scoped_lock lock(out);
  std::fputs(line.c_str(), stdout);
  std::fflush(stdout);
}

}  // namespace ajp::app::play
