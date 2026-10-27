#include "app/play/external_subtitles.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <string_view>
#include <system_error>

#include "app/play/tracks.h"

namespace ajp::app::play {
namespace {

std::string lower(std::string text) {
  std::ranges::transform(text, text.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

// Les premiers jetons après le nom de la vidéo qui sont une langue connue : « fr », « eng ».
std::string language_of(std::string_view tags) {
  std::size_t at = 0;
  while (at < tags.size()) {
    std::size_t end = tags.find('.', at);
    if (end == std::string_view::npos) end = tags.size();
    const std::string tag(tags.substr(at, end - at));
    if (!tag.empty()) {
      const std::string name = language_name(tag);
      if (!name.empty() && name != tag) return name;
    }
    at = end + 1;
  }
  return {};
}

}  // namespace

std::vector<ExternalSubtitle> find_external_subtitles(const std::filesystem::path& video) {
  std::vector<ExternalSubtitle> found;
  const std::filesystem::path dir =
      video.has_parent_path() ? video.parent_path() : std::filesystem::path(".");
  const std::string stem = lower(video.stem().string());
  if (stem.empty()) return found;
  std::error_code error;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (!it->is_regular_file(error)) continue;
    const std::filesystem::path& path = it->path();
    const std::string extension = lower(path.extension().string());
    if (extension != ".ass" && extension != ".ssa" && extension != ".srt") continue;
    const std::string name = lower(path.stem().string());
    if (!name.starts_with(stem)) continue;
    const std::string_view rest = std::string_view(name).substr(stem.size());
    if (!rest.empty() && rest.front() != '.')
      continue;  // « Épisode 010.srt » n'est pas celui de « Épisode 01 »
    const std::string language = language_of(rest);
    found.push_back({.path = path,
                     .label = language.empty()
                                  ? std::format("Externe · {}", path.filename().string())
                                  : std::format("Externe · {} · {}", language, path.filename().string())});
  }
  std::ranges::sort(found, {}, [](const ExternalSubtitle& s) { return s.path.filename().string(); });
  if (found.size() > kMaxExternalSubtitles) found.resize(kMaxExternalSubtitles);
  return found;
}

}  // namespace ajp::app::play
