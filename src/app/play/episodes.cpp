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

// Un bonus et pas un épisode (P3.81) : générique sans texte (NCOP, NCED), bande-annonce, publicité, menu.
bool extra(std::string_view stem) {
  return std::ranges::any_of(words_of(stem), [](const std::string& w) {
    return w.starts_with("NCOP") || w.starts_with("NCED") || w == "CREDITLESS" || w == "PV" || w == "CM" ||
           w == "MENU" || w == "TRAILER" || w == "TEASER" || w == "PREVIEW";
  });
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

int natural_compare(std::string_view a, std::string_view b) {
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (digit(a[i]) && digit(b[j])) {
      std::size_t ia = i;
      std::size_t jb = j;
      while (ia < a.size() && a[ia] == '0') ++ia;  // les zéros de tête ne comptent pas
      while (jb < b.size() && b[jb] == '0') ++jb;
      std::size_t ea = ia;
      std::size_t eb = jb;
      while (ea < a.size() && digit(a[ea])) ++ea;
      while (eb < b.size() && digit(b[eb])) ++eb;
      if (ea - ia != eb - jb) return ea - ia < eb - jb ? -1 : 1;
      if (const int order = a.substr(ia, ea - ia).compare(b.substr(jb, eb - jb)); order != 0) return order;
      i = ea;
      j = eb;
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
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (!it->is_regular_file(error)) continue;  // un lien cassé n'est pas un fichier : sauté
    std::string extension = it->path().extension().string();
    std::ranges::transform(extension, extension.begin(), folded);
    if (std::ranges::find(video_extensions(), extension) == video_extensions().end()) continue;
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
  const auto here =
      std::ranges::find_if(videos, [&](const Video& v) { return v.path.filename() == current.filename(); });
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
