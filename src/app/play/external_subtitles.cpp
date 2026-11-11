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
#include "core/log.h"
#include "core/text.h"
#include "core/utf8_path.h"
#include "media/language.h"

namespace ajp::app::play {
namespace {

// B35-07 : un dossier de 20 000 fichiers sur un partage réseau ne retarde pas la première image.
constexpr std::size_t kMaxScannedEntries = 5000;

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
  const std::string extension = core::ascii_lower(core::to_utf8(path.extension()));
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
  found.push_back(
      {.path = path,
       .label = language.empty()
                    ? i18n::tr_format("External · {}", core::to_utf8(path.filename()))
                    : i18n::tr_format("External · {} · {}", language, core::to_utf8(path.filename())),
       .language = tag});
}

bool is_subs_dir(const std::string& name) {
  return name == "subs" || name == "sub" || name == "subtitles" || name == "subtitle";
}

}  // namespace

std::vector<ExternalSubtitle> find_external_subtitles(const std::filesystem::path& video,
                                                      std::span<const std::string> wanted) {
  std::vector<ExternalSubtitle> found;
  const std::filesystem::path dir =
      video.has_parent_path() ? video.parent_path() : std::filesystem::path(".");
  const std::string stem = core::ascii_lower(core::to_utf8(video.stem()));
  if (stem.empty()) return found;
  // Le dossier de la vidéo, puis ses dossiers Subs/ (P3.80) : un fichier au nom de la vidéo, ou tout fichier
  // d'un sous-dossier Subs/<nom de la vidéo>/ (« Subs/Episode 01/2_English.srt »).
  std::vector<std::filesystem::path> dirs{dir};
  std::error_code error;
  std::size_t scanned = 0;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (++scanned > kMaxScannedEntries) break;
    if (it->is_directory(error) && is_subs_dir(core::ascii_lower(core::to_utf8(it->path().filename()))))
      dirs.push_back(it->path());
  }
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    error.clear();
    scanned = 0;
    for (std::filesystem::directory_iterator it(dirs[d], error), end; !error && it != end;
         it.increment(error)) {
      if (++scanned > kMaxScannedEntries) {
        const auto log = log_channel("subs");
        log->warn("recherche des sous-titres voisins : {} entrées lues, la suite est ignorée",
                  kMaxScannedEntries);
        break;
      }
      const std::filesystem::path& path = it->path();
      const std::string name = core::ascii_lower(core::to_utf8(path.stem()));
      if (d > 0 && it->is_directory(error)) {
        if (tags_after(core::ascii_lower(core::to_utf8(path.filename())), stem)) {
          std::error_code inner;
          for (std::filesystem::directory_iterator sub(path, inner), stop; !inner && sub != stop;
               sub.increment(inner)) {
            if (sub->is_regular_file(inner) && is_subtitle(sub->path()))
              add(found, sub->path(), core::ascii_lower(core::to_utf8(sub->path().stem())));
          }
        }
        continue;
      }
      // B35-07 : le nom d'abord, un stat par entrée seulement pour ce qui commence comme la vidéo.
      if (!is_subtitle(path)) continue;
      const auto rest = tags_after(name, stem);
      if (rest && it->is_regular_file(error)) add(found, path, *rest);
    }
  }
  // La langue voulue d'abord : la coupe à kMaxExternalSubtitles ne doit pas la laisser de côté.
  std::ranges::sort(found, {}, [&](const ExternalSubtitle& s) {
    return std::pair(-media::language_score(wanted, s.language), core::to_utf8(s.path.filename()));
  });
  if (found.size() > kMaxExternalSubtitles) found.resize(kMaxExternalSubtitles);
  return found;
}

}  // namespace ajp::app::play
