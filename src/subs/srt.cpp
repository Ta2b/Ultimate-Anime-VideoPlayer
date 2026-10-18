#include "subs/srt.h"

#include <format>
#include <optional>

namespace ajp::subs {
namespace {

std::unexpected<Refusal> refuse_line(std::size_t line, std::string_view what, std::string_view text) {
  return std::unexpected(
      Refusal{RefusalCode::SubtitleFormat, std::format("SRT, ligne {} : {} « {} »", line, what, text)});
}

bool is_space(char c) {
  return c == ' ' || c == '\t';
}
bool is_digit(char c) {
  return c >= '0' && c <= '9';
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

bool is_blank(std::string_view line) {
  return trim(line).empty();
}

bool is_index(std::string_view line) {
  line = trim(line);
  if (line.empty()) return false;
  for (const char c : line) {
    if (!is_digit(c)) return false;
  }
  return true;
}

// Lignes sans leur fin ('\n' ou "\r\n").
std::vector<std::string_view> split_lines(std::string_view text) {
  std::vector<std::string_view> lines;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    if (line.ends_with('\r')) line.remove_suffix(1);
    lines.push_back(line);
    if (end == std::string_view::npos) break;
    text.remove_prefix(end + 1);
  }
  return lines;
}

// Entre min_digits et max_digits chiffres décimaux, consommés en tête de s.
std::optional<int> take_number(std::string_view& s, std::size_t min_digits, std::size_t max_digits) {
  std::size_t count = 0;
  int value = 0;
  while (count < s.size() && count < max_digits && is_digit(s[count])) {
    value = value * 10 + (s[count] - '0');
    ++count;
  }
  if (count < min_digits) return std::nullopt;
  s.remove_prefix(count);
  return value;
}

bool take(std::string_view& s, std::string_view token) {
  if (!s.starts_with(token)) return false;
  s.remove_prefix(token.size());
  return true;
}

// « 01:02:03,456 » (ou « .456 ») en microsecondes.
std::optional<std::int64_t> take_timestamp(std::string_view& s) {
  const auto hours = take_number(s, 1, 2);
  if (!hours || !take(s, ":")) return std::nullopt;
  const auto minutes = take_number(s, 2, 2);
  if (!minutes || !take(s, ":")) return std::nullopt;
  const auto seconds = take_number(s, 2, 2);
  if (!seconds || !(take(s, ",") || take(s, "."))) return std::nullopt;
  const auto millis = take_number(s, 3, 3);
  if (!millis || *hours > kSrtMaxHours || *minutes > 59 || *seconds > 59) return std::nullopt;
  const std::int64_t total_seconds = (std::int64_t{*hours} * 60 + *minutes) * 60 + *seconds;
  return total_seconds * 1'000'000 + std::int64_t{*millis} * 1000;
}

struct Timing {
  std::int64_t start_us = 0;
  std::int64_t end_us = 0;
};

// « début --> fin », suivi éventuellement de coordonnées, ignorées.
std::optional<Timing> parse_timing(std::string_view line) {
  line = trim(line);
  const auto start = take_timestamp(line);
  if (!start) return std::nullopt;
  line = trim(line);
  if (!take(line, "-->")) return std::nullopt;
  line = trim(line);
  const auto end = take_timestamp(line);
  if (!end || !(line.empty() || is_space(line.front()))) return std::nullopt;
  return Timing{*start, *end};
}

}  // namespace

std::string_view text_track_header() {
  return "[Script Info]\n"
         "ScriptType: v4.00+\n"
         "PlayResX: 384\n"
         "PlayResY: 288\n"
         "ScaledBorderAndShadow: yes\n"
         "YCbCr Matrix: None\n"
         "\n"
         "[V4+ Styles]\n"
         "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
         "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
         "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
         "Style: Default,sans-serif,16,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
         "0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n"
         "\n"
         "[Events]\n"
         "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";
}

std::expected<std::vector<SrtCue>, Refusal> parse_srt(std::string_view file) {
  if (file.starts_with("\xEF\xBB\xBF")) file.remove_prefix(3);
  const std::vector<std::string_view> lines = split_lines(file);

  std::vector<SrtCue> cues;
  std::size_t i = 0;
  while (i < lines.size()) {
    if (is_blank(lines[i])) {
      ++i;
      continue;
    }
    if (is_index(lines[i]) && i + 1 < lines.size()) ++i;  // numéro de cue, facultatif
    const auto timing = parse_timing(lines[i]);
    if (!timing) return refuse_line(i + 1, "horodatage attendu", lines[i]);
    if (timing->end_us < timing->start_us) return refuse_line(i + 1, "fin avant début", lines[i]);
    ++i;

    std::string text;
    for (; i < lines.size() && !is_blank(lines[i]); ++i) {
      if (!text.empty()) text += '\n';
      text += lines[i];
    }
    cues.push_back({.start_us = timing->start_us, .end_us = timing->end_us, .text = std::move(text)});
  }
  return cues;
}

}  // namespace ajp::subs
