#include "app/play/episodes.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "app/play/default_player.h"
#include "core/text.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// B35-07 : entrées lues au plus dans le dossier de la vidéo, pour trouver l'épisode voisin.
constexpr std::size_t kMaxScannedEntries = 5000;

bool digit(char c) {
  return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

char folded(char c) {
  return core::ascii_lower(c);
}

// Les mots d'un nom (lettres et chiffres), en majuscules.
std::vector<std::string> words_of(std::string_view name) {
  std::vector<std::string> words;
  std::string word;
  for (const char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
      word.push_back(core::ascii_upper(c));
    } else if (!word.empty()) {
      words.push_back(std::exchange(word, {}));
    }
  }
  if (!word.empty()) words.push_back(word);
  return words;
}

bool pure_number(const std::string& word) {
  return !word.empty() && std::ranges::all_of(word, digit);
}

// Un numéro d'épisode : un nombre seul qui n'est ni une année (« (2008) »), ni le rang d'un volume ou d'un
// disque (« Vol 1 Menu », « BD 2 Menu »).
bool episode_number(const std::vector<std::string>& words, std::size_t i) {
  const std::string& word = words[i];
  if (!pure_number(word)) return false;
  if (word.size() == 4 && (word.starts_with("19") || word.starts_with("20"))) return false;
  if (i == 0) return true;
  const std::string& before = words[i - 1];
  return before != "VOL" && before != "VOLUME" && before != "DISC" && before != "DISK" && before != "BD" &&
         before != "DVD";
}

// Un bonus et pas un épisode (P3.81) : générique sans texte (NCOP, NCED), bande-annonce, publicité, menu.
// NCOP, NCED et CREDITLESS le disent partout. Les autres mots sont aussi des mots de titre (« Soma - 05 - The
// Menu ») : ils ne comptent que si aucun numéro d'épisode ne les précède (A13-03).
// Le mot de bonus, seul ou collé à son numéro (« CM01 », « PV2 », « Menu01 » : words_of garde ensemble
// lettres et chiffres).
bool extra_word(std::string_view w) {
  for (const std::string_view tag : {"PV", "CM", "MENU", "TRAILER", "TEASER", "PREVIEW"}) {
    if (w.starts_with(tag) && std::ranges::all_of(w.substr(tag.size()), digit)) return true;
  }
  return false;
}

bool extra(std::string_view stem) {
  const std::vector<std::string> words = words_of(stem);
  bool numbered = false;
  for (std::size_t i = 0; i < words.size(); ++i) {
    const std::string& w = words[i];
    if (w.starts_with("NCOP") || w.starts_with("NCED") || w == "CREDITLESS") return true;
    if (!numbered && extra_word(w)) return true;
    numbered = numbered || episode_number(words, i);
  }
  return false;
}

// Le nom sans sa version (« 01v2 » donne « 01 ») et la version (1 sans marque) : deux versions d'un même
// épisode ne se lisent pas l'une après l'autre (P3.81).
std::pair<std::string, int> without_version(std::string_view stem) {
  std::string key;
  int version = 1;
  for (std::size_t i = 0; i < stem.size(); ++i) {
    const bool after_digit = i > 0 && digit(stem[i - 1]);
    if (after_digit && folded(stem[i]) == 'v' && i + 1 < stem.size() && digit(stem[i + 1])) {
      std::size_t end = i + 1;
      int value = 0;
      while (end < stem.size() && digit(stem[end])) value = (value * 10) + (stem[end++] - '0');
      if (end == stem.size() || std::isalnum(static_cast<unsigned char>(stem[end])) == 0) {
        version = value;
        i = end - 1;
        continue;
      }
    }
    key.push_back(stem[i]);
  }
  return {key, version};
}

}  // namespace

namespace {

// Le chiffre qui commence en `i` (ASCII, ou pleine chasse U+FF10 à U+FF19 des noms japonais, EF BC 90..99
// en UTF-8) et sa longueur en octets, ou -1 (A13-04).
int digit_at(std::string_view s, std::size_t i, std::size_t& length) {
  if (digit(s[i])) {
    length = 1;
    return s[i] - '0';
  }
  if (i + 2 < s.size() && static_cast<unsigned char>(s[i]) == 0xEF &&
      static_cast<unsigned char>(s[i + 1]) == 0xBC) {
    const auto third = static_cast<unsigned char>(s[i + 2]);
    if (third >= 0x90 && third <= 0x99) {
      length = 3;
      return third - 0x90;
    }
  }
  return -1;
}

// Les chiffres qui commencent en `i`, sans les zéros de tête ; `i` avance après eux.
std::string number_at(std::string_view s, std::size_t& i) {
  std::string value;
  std::size_t length = 0;
  for (int d = digit_at(s, i, length); d >= 0; d = i < s.size() ? digit_at(s, i, length) : -1) {
    if (!value.empty() || d != 0)
      value.push_back(static_cast<char>('0' + d));  // les zéros de tête ne comptent pas
    i += length;
  }
  return value;
}

}  // namespace

int natural_compare(std::string_view a, std::string_view b) {
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t length = 0;
  while (i < a.size() && j < b.size()) {
    if (digit_at(a, i, length) >= 0 && digit_at(b, j, length) >= 0) {
      const std::string na = number_at(a, i);
      const std::string nb = number_at(b, j);
      if (na.size() != nb.size()) return na.size() < nb.size() ? -1 : 1;
      if (const int order = na.compare(nb); order != 0) return order < 0 ? -1 : 1;
      continue;
    }
    if (folded(a[i]) != folded(b[j])) return folded(a[i]) < folded(b[j]) ? -1 : 1;
    ++i;
    ++j;
  }
  if (i < a.size()) return 1;
  if (j < b.size()) return -1;
  return 0;
}

std::optional<std::filesystem::path> sibling_episode(const std::filesystem::path& current, int direction) {
  std::error_code error;
  std::filesystem::path dir = current.parent_path();
  if (dir.empty()) dir = ".";
  // Les bonus restent hors de la liste, sauf si l'on en lit un : on passe alors d'un bonus à l'autre.
  const bool among_extras = extra(prepare::utf8_path(current.stem()));
  struct Video {
    std::filesystem::path path;
    std::string key;  // le nom sans extension ni version : « 12 » avant « 12.5 » (P3.81)
    int version = 1;
  };
  std::vector<Video> videos;
  std::size_t scanned = 0;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    // B35-07 : plafond de lecture d'un dossier (partage réseau), l'extension avant tout stat.
    if (++scanned > kMaxScannedEntries) break;
    std::string extension = prepare::utf8_path(it->path().extension());
    std::ranges::transform(extension, extension.begin(), folded);
    if (std::ranges::find(video_extensions(), extension) == video_extensions().end()) continue;
    if (!it->is_regular_file(error)) continue;  // un lien cassé n'est pas un fichier : sauté
    const std::string stem = prepare::utf8_path(it->path().stem());
    if (extra(stem) != among_extras) continue;
    auto [key, version] = without_version(stem);
    videos.push_back({it->path(), std::move(key), version});
  }
  if (error) return std::nullopt;
  std::ranges::sort(videos, [](const Video& a, const Video& b) {
    const int order = natural_compare(a.key, b.key);
    return order != 0 ? order < 0 : a.version > b.version;  // la plus haute version d'abord
  });
  auto here =
      std::ranges::find_if(videos, [&](const Video& v) { return v.path.filename() == current.filename(); });
  // Ouvert sous une autre casse (« show - 01.mkv » pour « Show - 01.mkv », que NTFS confond) : le même
  // fichier.
  if (here == videos.end()) {
    here = std::ranges::find_if(videos, [&](const Video& v) {
      std::error_code same;
      return std::filesystem::equivalent(v.path, current, same);
    });
  }
  if (here == videos.end()) return std::nullopt;
  // Le voisin dont le nom diffère (une autre version du même épisode n'en est pas un) et qui se lit (P3.81).
  const auto readable = [](const Video& v) { return std::ifstream(v.path, std::ios::binary).good(); };
  if (direction > 0) {
    for (auto it = here + 1; it != videos.end(); ++it)
      if (it->key != here->key && readable(*it)) return it->path;  // la plus haute version vient d'abord
    return std::nullopt;
  }
  for (auto it = here; it != videos.begin();) {
    --it;
    if (it->key == here->key) continue;
    // La plus haute version de cet épisode : la première de son groupe.
    auto first = it;
    while (first != videos.begin() && (first - 1)->key == it->key) --first;
    if (readable(*first)) return first->path;
    it = first;
  }
  return std::nullopt;
}

GrowingFile GrowingFile::of(const std::filesystem::path& file) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  return GrowingFile{error ? 0 : size};
}

bool GrowingFile::grew(const std::filesystem::path& file, std::chrono::milliseconds wait) const {
  if (size == 0) return false;
  if (of(file).size > size) return true;
  std::error_code error;
  const auto written = std::filesystem::last_write_time(file, error);
  if (error || std::filesystem::file_time_type::clock::now() - written > std::chrono::seconds(10))
    return false;
  const auto until = std::chrono::steady_clock::now() + wait;
  while (std::chrono::steady_clock::now() < until) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (of(file).size > size) return true;
  }
  return false;
}

}  // namespace ajp::app::play
