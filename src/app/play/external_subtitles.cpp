#include "app/play/external_subtitles.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include "app/play/i18n.h"
#include "app/play/tracks.h"
#include "core/text.h"

namespace ajp::app::play {
namespace {

// Le premier jeton après le nom de la vidéo qui est une langue connue : sa balise (« fr », « eng ») et son
// nom. Les jetons se séparent par un point, un tiret bas ou une espace (« 2_English », « Subs/fr »).
std::pair<std::string, std::string> language_of(std::string_view tags) {
  std::size_t at = 0;
  while (at < tags.size()) {
    std::size_t end = tags.find_first_of("._ ", at);
    if (end == std::string_view::npos) end = tags.size();
    const std::string tag(tags.substr(at, end - at));
    if (!tag.empty()) {
      const std::string name = language_name(tag);
      if (!name.empty() && name != tag) return {tag, name};
      for (const Language& known : menu_languages()) {  // « English », « French » en toutes lettres
        if (core::ascii_lower(std::string(known.name)) == tag)
          return {std::string(known.code), language_name(known.code)};
      }
    }
    at = end + 1;
  }
  return {};
}

bool is_subtitle(const std::filesystem::path& path) {
  const std::string extension = core::ascii_lower(path.extension().string());
  return extension == ".ass" || extension == ".ssa" || extension == ".srt";
}

// Le nom sans les balises entre crochets ou parenthèses (« [Groupe] », « [1080p] », « (BD) »), les tirets
// bas en espaces, les espaces fusionnées (P3.80) : « [G] Show_-_01 [1080p] » et « Show - 01 » se valent.
std::string loose(std::string_view name) {
  std::string out;
  int depth = 0;
  for (const char c : name) {
    if (c == '[' || c == '(') {
      ++depth;
    } else if ((c == ']' || c == ')') && depth > 0) {
      --depth;
    } else if (depth == 0) {
      const char k = c == '_' ? ' ' : c;
      if (k != ' ' || (!out.empty() && out.back() != ' ')) out.push_back(k);
    }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

// Ce qui suit le nom de la vidéo dans le nom du sous-titre (« .fr »), ou rien si ce n'est pas le sien.
std::optional<std::string> tags_after(const std::string& name, const std::string& stem) {
  for (const auto& [n, v] : {std::pair{name, stem}, std::pair{loose(name), loose(stem)}}) {
    if (v.empty() || !n.starts_with(v)) continue;
    std::string rest = n.substr(v.size());
    if (rest.empty() || rest.front() == '.') return rest;  // « Épisode 010.srt » n'est pas « Épisode 01 »
  }
  return std::nullopt;
}

void add(std::vector<ExternalSubtitle>& found, const std::filesystem::path& path, std::string_view tags) {
  const auto [tag, language] = language_of(tags);
  found.push_back({.path = path,
                   .label = language.empty()
                                ? i18n::tr_format("External · {}", path.filename().string())
                                : i18n::tr_format("External · {} · {}", language, path.filename().string()),
                   .language = tag});
}

bool is_subs_dir(const std::string& name) {
  return name == "subs" || name == "sub" || name == "subtitles" || name == "subtitle";
}

}  // namespace

std::vector<ExternalSubtitle> find_external_subtitles(const std::filesystem::path& video) {
  std::vector<ExternalSubtitle> found;
  const std::filesystem::path dir =
      video.has_parent_path() ? video.parent_path() : std::filesystem::path(".");
  const std::string stem = core::ascii_lower(video.stem().string());
  if (stem.empty()) return found;
  // Le dossier de la vidéo, puis ses dossiers Subs/ (P3.80) : un fichier au nom de la vidéo, ou tout fichier
  // d'un sous-dossier Subs/<nom de la vidéo>/ (« Subs/Episode 01/2_English.srt »).
  std::vector<std::filesystem::path> dirs{dir};
  std::error_code error;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (it->is_directory(error) && is_subs_dir(core::ascii_lower(it->path().filename().string())))
      dirs.push_back(it->path());
  }
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    error.clear();
    for (std::filesystem::directory_iterator it(dirs[d], error), end; !error && it != end;
         it.increment(error)) {
      const std::filesystem::path& path = it->path();
      const std::string name = core::ascii_lower(path.stem().string());
      if (d > 0 && it->is_directory(error)) {
        if (tags_after(core::ascii_lower(path.filename().string()), stem)) {
          std::error_code inner;
          for (std::filesystem::directory_iterator sub(path, inner), stop; !inner && sub != stop;
               sub.increment(inner)) {
            if (sub->is_regular_file(inner) && is_subtitle(sub->path()))
              add(found, sub->path(), core::ascii_lower(sub->path().stem().string()));
          }
        }
        continue;
      }
      if (!it->is_regular_file(error) || !is_subtitle(path)) continue;
      if (const auto rest = tags_after(name, stem)) add(found, path, *rest);
    }
  }
  std::ranges::sort(found, {}, [](const ExternalSubtitle& s) { return s.path.filename().string(); });
  if (found.size() > kMaxExternalSubtitles) found.resize(kMaxExternalSubtitles);
  return found;
}

}  // namespace ajp::app::play
