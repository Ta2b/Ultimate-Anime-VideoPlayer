#include "app/play/episodes.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <system_error>
#include <vector>

#include "app/play/default_player.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

bool digit(char c) {
  return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

char folded(char c) {
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
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
  std::vector<std::filesystem::path> videos;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (!it->is_regular_file(error)) continue;
    std::string extension = it->path().extension().string();
    std::ranges::transform(extension, extension.begin(), folded);
    if (std::ranges::find(video_extensions(), extension) != video_extensions().end())
      videos.push_back(it->path());
  }
  if (error) return std::nullopt;
  const auto order = [](const std::filesystem::path& a, const std::filesystem::path& b) {
    return natural_compare(prepare::utf8_path(a.filename()), prepare::utf8_path(b.filename())) < 0;
  };
  std::ranges::sort(videos, order);
  const auto here = std::ranges::find_if(videos, [&](const std::filesystem::path& v) {
    return v.filename() == current.filename();
  });
  if (here == videos.end()) return std::nullopt;
  if (direction > 0) return here + 1 != videos.end() ? std::optional(*(here + 1)) : std::nullopt;
  return here != videos.begin() ? std::optional(*(here - 1)) : std::nullopt;
}

}  // namespace ajp::app::play
