// La banque des noms de langue que portent les titres de pistes : codes de trois lettres, noms en
// français, en anglais et dans la langue même, et les sigles des sorties françaises (VF, VOSTFR).
// Comportement fixé par tests/unit/media/track_language_test.cpp.
#include "media/track_language.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

namespace ajp::media {
namespace {

struct Name {
  std::string_view word;  // en minuscules ; un mot entier, ou une suite d'octets hors ASCII
  std::string_view language;
};

// clang-format off
constexpr std::array<Name, 86> kNames{{
    {"fre", "fre"}, {"fra", "fre"}, {"fr", "fre"}, {"french", "fre"}, {"français", "fre"},
    {"francais", "fre"}, {"truefrench", "fre"}, {"vf", "fre"}, {"vff", "fre"}, {"vfq", "fre"},
    {"vfi", "fre"}, {"vostfr", "fre"}, {"québécois", "fre"}, {"quebecois", "fre"},
    {"eng", "eng"}, {"english", "eng"}, {"anglais", "eng"}, {"englisch", "eng"}, {"inglés", "eng"},
    {"ingles", "eng"}, {"vosta", "eng"},
    {"jpn", "jpn"}, {"jap", "jpn"}, {"japanese", "jpn"}, {"japonais", "jpn"}, {"japanisch", "jpn"},
    {"japonés", "jpn"}, {"japones", "jpn"}, {"日本語", "jpn"},
    {"ger", "ger"}, {"deu", "ger"}, {"german", "ger"}, {"deutsch", "ger"}, {"allemand", "ger"},
    {"alemán", "ger"}, {"aleman", "ger"},
    {"spa", "spa"}, {"spanish", "spa"}, {"español", "spa"}, {"espanol", "spa"}, {"espagnol", "spa"},
    {"castellano", "spa"}, {"castilian", "spa"}, {"latino", "spa"},
    {"ita", "ita"}, {"italian", "ita"}, {"italiano", "ita"}, {"italien", "ita"},
    {"portuguese", "por"}, {"português", "por"}, {"portugues", "por"}, {"portugais", "por"},
    {"brasileiro", "por"}, {"brazilian", "por"},
    {"rus", "rus"}, {"russian", "rus"}, {"russe", "rus"}, {"русский", "rus"}, {"Русский", "rus"},
    {"chi", "chi"}, {"zho", "chi"}, {"chs", "chi"}, {"cht", "chi"}, {"chinese", "chi"}, {"chinois", "chi"},
    {"mandarin", "chi"}, {"中文", "chi"}, {"简体", "chi"}, {"繁體", "chi"}, {"繁体", "chi"},
    {"kor", "kor"}, {"korean", "kor"}, {"coréen", "kor"}, {"coreen", "kor"}, {"한국어", "kor"},
    {"ara", "ara"}, {"arabic", "ara"}, {"arabe", "ara"}, {"العربية", "ara"},
    {"pol", "pol"}, {"polish", "pol"}, {"polski", "pol"}, {"polonais", "pol"},
    {"dut", "dut"}, {"dutch", "dut"}, {"nederlands", "dut"},
}};
// clang-format on

bool ascii_word_char(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Minuscules ASCII, et les majuscules latines accentuées (« É », « Ç », « Ñ » : U+00C0 à U+00DE
// sauf ×) ; le reste tel quel.
std::string folded(std::string_view text) {
  std::string out(text);
  for (std::size_t i = 0; i < out.size(); ++i) {
    const auto c = static_cast<unsigned char>(out[i]);
    if (c >= 'A' && c <= 'Z') {
      out[i] = static_cast<char>(c + ('a' - 'A'));
    } else if (c == 0xC3 && i + 1 < out.size()) {
      const auto next = static_cast<unsigned char>(out[i + 1]);
      if (next >= 0x80 && next <= 0x9E && next != 0x97) out[i + 1] = static_cast<char>(next + 0x20);
      ++i;
    }
  }
  return out;
}

bool is_ascii(std::string_view word) {
  return std::ranges::all_of(word, [](char c) { return static_cast<unsigned char>(c) < 0x80; });
}

// Un mot du titre : une suite de lettres et chiffres, les octets hors ASCII compris (« français »).
struct Word {
  std::size_t at = 0;
  std::string_view text;
};

std::optional<Word> next_word(std::string_view text, std::size_t& from) {
  const auto word_char = [](char c) {
    const auto u = static_cast<unsigned char>(c);
    return u >= 0x80 || ascii_word_char(u);
  };
  while (from < text.size() && !word_char(text[from])) ++from;
  if (from >= text.size()) return std::nullopt;
  const std::size_t start = from;
  while (from < text.size() && word_char(text[from])) ++from;
  return Word{.at = start, .text = text.substr(start, from - start)};
}

}  // namespace

std::string_view language_from_title(std::string_view title) {
  const std::string text = folded(title);
  std::optional<std::size_t> best_at;
  std::string_view best;
  const auto keep = [&](std::size_t at, std::string_view language) {
    if (!best_at || at < *best_at) {
      best_at = at;
      best = language;
    }
  };
  // Les noms ASCII comptent en mot entier ; les autres (日本語, русский) n'importe où, souvent
  // collés à d'autres caractères (« 日本語字幕 »).
  std::size_t from = 0;
  while (const std::optional<Word> word = next_word(text, from)) {
    for (const Name& name : kNames) {
      if (word->text == name.word) keep(word->at, name.language);
    }
  }
  for (const Name& name : kNames) {
    if (is_ascii(name.word)) continue;
    if (const std::size_t at = text.find(name.word); at != std::string::npos) keep(at, name.language);
  }
  return best;
}

std::string track_language(const TrackInfo& track) {
  const std::string tag = folded(track.language);
  if (!tag.empty() && tag != "und" && tag != "mul" && tag != "zxx") return track.language;
  return std::string(language_from_title(track.title));
}

}  // namespace ajp::media
