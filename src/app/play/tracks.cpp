#include "app/play/tracks.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <utility>

#include "media/language.h"

namespace ajp::app::play {
namespace {

// La seule table des noms de langue : une ligne par langue, les autres formes (« ja », « fra »,
// « zho ») s'y ramènent par media::canonical_language.
constexpr std::array<Language, 20> kLanguages{{
    {.code = "jpn", .name = "Japonais", .menu = true},
    {.code = "fre", .name = "Français", .menu = true},
    {.code = "eng", .name = "Anglais", .menu = true},
    {.code = "ger", .name = "Allemand", .menu = true},
    {.code = "spa", .name = "Espagnol", .menu = true},
    {.code = "ita", .name = "Italien", .menu = true},
    {.code = "por", .name = "Portugais", .menu = true},
    {.code = "rus", .name = "Russe", .menu = true},
    {.code = "chi", .name = "Chinois", .menu = true},
    {.code = "kor", .name = "Coréen", .menu = true},
    {.code = "ara", .name = "Arabe", .menu = true},
    {.code = "pol", .name = "Polonais"},
    {.code = "dut", .name = "Néerlandais"},
    {.code = "tur", .name = "Turc"},
    {.code = "ind", .name = "Indonésien"},
    {.code = "tha", .name = "Thaï"},
    {.code = "vie", .name = "Vietnamien"},
    {.code = "may", .name = "Malais"},
    {.code = "hin", .name = "Hindi"},
    {.code = "ukr", .name = "Ukrainien"},
}};

constexpr auto kMenuLanguages = [] {
  std::array<Language, std::ranges::count(kLanguages, true, &Language::menu)> menu{};
  std::ranges::copy_if(kLanguages, menu.begin(), &Language::menu);
  return menu;
}();

std::string lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

// La langue primaire en minuscules (« en-US » : « en »).
std::string primary(std::string_view tag) {
  return lower(tag.substr(0, tag.find_first_of("-_")));
}

}  // namespace

std::string language_name(std::string_view tag) {
  const std::string code = primary(tag);
  if (code.empty() || code == "und" || code == "mul" || code == "zxx") return {};
  const std::string_view canonical = media::canonical_language(code);
  for (const Language& language : kLanguages) {
    if (language.code == canonical) return std::string(language.name);
  }
  return std::string(tag);
}

std::span<const Language> menu_languages() {
  return kMenuLanguages;
}

bool same_language(std::string_view a, std::string_view b) {
  return media::canonical_language(primary(a)) == media::canonical_language(primary(b));
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
