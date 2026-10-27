#include "app/play/chapters.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <string_view>
#include <vector>

namespace ajp::app::play {

std::optional<std::size_t> chapter_at(std::span<const ChapterMark> chapters, std::int64_t position_us) {
  std::optional<std::size_t> found;
  for (std::size_t k = 0; k < chapters.size(); ++k) {
    if (chapters[k].start_us <= position_us) found = k;
  }
  return found;
}

std::optional<std::size_t> chapter_step(std::span<const ChapterMark> chapters, std::int64_t position_us,
                                        int direction) {
  if (chapters.empty()) return std::nullopt;
  const std::optional<std::size_t> here = chapter_at(chapters, position_us);
  if (direction > 0) {
    const std::size_t next = here ? *here + 1 : 0;
    return next < chapters.size() ? std::optional(next) : std::nullopt;
  }
  if (!here) return std::size_t{0};
  if (position_us - chapters[*here].start_us > kChapterRestartUs || *here == 0) return here;
  return *here - 1;
}

bool is_credits_chapter(const ChapterMark& chapter) {
  std::string title = chapter.title;
  std::ranges::transform(title, title.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  // Les mots du titre : lettres seules, « OP2 » donne « op ».
  std::vector<std::string_view> words;
  std::size_t at = 0;
  while (at < title.size()) {
    while (at < title.size() && std::isalpha(static_cast<unsigned char>(title[at])) == 0) ++at;
    std::size_t end = at;
    while (end < title.size() && std::isalpha(static_cast<unsigned char>(title[end])) != 0) ++end;
    if (end > at) words.emplace_back(title.data() + at, end - at);
    at = end;
  }
  // « Avant-générique », « Pre-credits », « Preview » : ce qui précède un générique n'en est pas un.
  constexpr std::string_view kBefore[] = {"avant", "pre", "preview", "recap"};
  if (std::ranges::any_of(words, [&](std::string_view word) {
        return std::ranges::find(kBefore, word) != std::end(kBefore);
      }))
    return false;
  constexpr std::string_view kWords[] = {"op",     "ed",      "opening", "ending", "intro",
                                         "outro",  "credits", "credit",  "generique"};
  return std::ranges::any_of(words, [&](std::string_view word) {
    return std::ranges::find(kWords, word) != std::end(kWords);
  }) || title.contains("générique");
}

std::optional<std::int64_t> credits_skip_target(std::span<const ChapterMark> chapters,
                                                std::int64_t position_us, std::int64_t duration_us) {
  const std::optional<std::size_t> here = chapter_at(chapters, position_us);
  if (!here || !is_credits_chapter(chapters[*here])) return std::nullopt;
  if (*here + 1 < chapters.size()) return chapters[*here + 1].start_us;
  return duration_us > 0 ? std::optional(duration_us) : std::nullopt;
}

std::string chapter_label(std::span<const ChapterMark> chapters, std::size_t index) {
  std::string label = std::format("Chapitre {}/{}", index + 1, chapters.size());
  if (index < chapters.size() && !chapters[index].title.empty()) label += " · " + chapters[index].title;
  return label;
}

}  // namespace ajp::app::play
