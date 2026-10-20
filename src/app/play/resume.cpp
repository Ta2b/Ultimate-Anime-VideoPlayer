#include "app/play/resume.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace ajp::app::play {
namespace {

struct Entry {
  std::int64_t position_us = 0;
  std::uintmax_t size = 0;
  std::string path;
};

// Le fichier tel qu'il est retenu : chemin absolu sans . ni .., et sa taille.
std::optional<Entry> identify(const std::filesystem::path& file) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::weakly_canonical(file, error);
  if (error) return std::nullopt;
  const std::uintmax_t size = std::filesystem::file_size(absolute, error);
  if (error) return std::nullopt;
  std::string path = absolute.string();
  if (path.find_first_of("\t\n\r") != std::string::npos) return std::nullopt;
  return Entry{.size = size, .path = std::move(path)};
}

template <class T>
bool parse(std::string_view text, T& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

// Les lignes lisibles, de la plus ancienne à la plus récente ; les autres sont ignorées.
std::vector<Entry> load(const std::filesystem::path& store) {
  std::vector<Entry> entries;
  std::ifstream in(store);
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t a = line.find('\t');
    const std::size_t b = a == std::string::npos ? a : line.find('\t', a + 1);
    if (b == std::string::npos) continue;
    Entry e;
    if (!parse(std::string_view(line).substr(0, a), e.position_us) ||
        !parse(std::string_view(line).substr(a + 1, b - a - 1), e.size) || b + 1 >= line.size()) {
      continue;
    }
    e.path = line.substr(b + 1);
    entries.push_back(std::move(e));
  }
  return entries;
}

std::expected<void, Refusal> save(const std::filesystem::path& store, const std::vector<Entry>& entries) {
  const auto refused = [&](std::string_view what) {
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", store.string(), what)});
  };
  std::error_code error;
  std::filesystem::create_directories(store.parent_path(), error);
  // Écrit à côté puis renommé : une coupure ne laisse jamais un fichier à moitié écrit.
  std::filesystem::path temporary = store;
  temporary += ".tmp";
  {
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) return refused("écriture impossible");
    for (const Entry& e : entries) out << e.position_us << '\t' << e.size << '\t' << e.path << '\n';
    if (!out.flush()) return refused("écriture interrompue");
  }
  std::filesystem::rename(temporary, store, error);
  if (error) return refused(error.message());
  return {};
}

}  // namespace

std::optional<std::int64_t> resume_position(const std::filesystem::path& store,
                                            const std::filesystem::path& file) {
  const std::optional<Entry> wanted = identify(file);
  if (!wanted) return std::nullopt;
  const std::vector<Entry> entries = load(store);
  const auto found = std::ranges::find(entries, wanted->path, &Entry::path);
  if (found == entries.end() || found->size != wanted->size || found->position_us < kResumeMinUs)
    return std::nullopt;
  return found->position_us;
}

std::expected<void, Refusal> remember_position(const std::filesystem::path& store,
                                               const std::filesystem::path& file, std::int64_t position_us,
                                               std::int64_t duration_us) {
  std::optional<Entry> entry = identify(file);
  if (!entry) return {};  // fichier disparu, ou chemin qu'une ligne ne sait pas porter
  std::vector<Entry> entries = load(store);
  const std::size_t before = entries.size();
  std::erase_if(entries, [&](const Entry& e) { return e.path == entry->path; });
  const bool near_end =
      duration_us > 0 && (position_us >= duration_us - kResumeTailUs || position_us >= duration_us / 20 * 19);
  if (!near_end && position_us >= kResumeMinUs) {
    entry->position_us = position_us;
    entries.push_back(std::move(*entry));
    if (entries.size() > kResumeEntries)
      entries.erase(entries.begin(), entries.end() - static_cast<std::ptrdiff_t>(kResumeEntries));
  } else if (entries.size() == before) {
    return {};  // rien à oublier
  }
  return save(store, entries);
}

}  // namespace ajp::app::play
