#include "app/play/chapters.h"

#include <format>

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

std::string chapter_label(std::span<const ChapterMark> chapters, std::size_t index) {
  std::string label = std::format("Chapitre {}/{}", index + 1, chapters.size());
  if (index < chapters.size() && !chapters[index].title.empty()) label += " · " + chapters[index].title;
  return label;
}

}  // namespace ajp::app::play
