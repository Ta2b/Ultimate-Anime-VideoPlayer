#include "app/play/tracks.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <utility>

namespace ajp::app::play {
namespace {

struct Language {
  std::string_view code;  // ISO 639-1 ou 639-2 (B et T)
  std::string_view name;
};

constexpr std::array<Language, 44> kLanguages{{
    {"ja", "Japonais"},     {"jpn", "Japonais"},    {"fr", "Français"},  {"fre", "Français"},
    {"fra", "Français"},    {"en", "Anglais"},      {"eng", "Anglais"},  {"de", "Allemand"},
    {"ger", "Allemand"},    {"deu", "Allemand"},    {"es", "Espagnol"},  {"spa", "Espagnol"},
    {"it", "Italien"},      {"ita", "Italien"},     {"pt", "Portugais"}, {"por", "Portugais"},
    {"ru", "Russe"},        {"rus", "Russe"},       {"ar", "Arabe"},     {"ara", "Arabe"},
    {"zh", "Chinois"},      {"chi", "Chinois"},     {"zho", "Chinois"},  {"ko", "Coréen"},
    {"kor", "Coréen"},      {"pl", "Polonais"},     {"pol", "Polonais"}, {"nl", "Néerlandais"},
    {"dut", "Néerlandais"}, {"nld", "Néerlandais"}, {"tr", "Turc"},      {"tur", "Turc"},
    {"id", "Indonésien"},   {"ind", "Indonésien"},  {"th", "Thaï"},      {"tha", "Thaï"},
    {"vi", "Vietnamien"},   {"vie", "Vietnamien"},  {"ms", "Malais"},    {"may", "Malais"},
    {"msa", "Malais"},      {"hi", "Hindi"},        {"hin", "Hindi"},    {"uk", "Ukrainien"},
}};

std::string lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

}  // namespace

std::string language_name(std::string_view tag) {
  const std::string code = lower(tag.substr(0, tag.find_first_of("-_")));  // « en-US » : « en »
  if (code.empty() || code == "und" || code == "mul" || code == "zxx") return {};
  for (const Language& language : kLanguages) {
    if (language.code == code) return std::string(language.name);
  }
  return std::string(tag);
}

std::string track_label(const media::TrackInfo& track) {
  const std::string language = language_name(track.language);
  std::string label = language;
  const bool repeats = !language.empty() && lower(track.title).find(lower(language)) != std::string::npos;
  if (!track.title.empty() && !repeats)
    label = label.empty() ? track.title : std::format("{} · {}", label, track.title);
  if (!track.title.empty() && repeats) label = track.title;
  if (label.empty()) label = std::format("Piste {}", track.index);
  if (track.kind == media::TrackKind::Subtitle && track.codec != "ass" && track.codec != "ssa")
    label += std::format(" ({})", track.codec == "subrip" ? std::string("SRT") : track.codec);
  if (track.forced) label += " · forcés";
  return label;
}

std::optional<int> next_subtitle(const TrackMenu& menu) {
  if (menu.subtitles.empty()) return std::nullopt;
  if (!menu.subtitle_selected) return menu.subtitles.front().index;
  const auto here = std::ranges::find(menu.subtitles, *menu.subtitle_selected, &TrackEntry::index);
  if (here == menu.subtitles.end() || std::next(here) == menu.subtitles.end()) return std::nullopt;
  return std::next(here)->index;
}

std::optional<int> next_audio(const TrackMenu& menu) {
  if (menu.audio.empty()) return std::nullopt;
  if (!menu.audio_selected) return menu.audio.front().index;
  const auto here = std::ranges::find(menu.audio, *menu.audio_selected, &TrackEntry::index);
  if (here == menu.audio.end() || std::next(here) == menu.audio.end()) return menu.audio.front().index;
  return std::next(here)->index;
}

}  // namespace ajp::app::play
