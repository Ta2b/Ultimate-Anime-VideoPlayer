#include "app/play/resume.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "core/user_overlay.h"

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

// Deux chemins retenus désignent-ils le même fichier ? Sous Windows et macOS, sans égard à la casse.
bool same_path(std::string_view a, std::string_view b) {
#if defined(_WIN32) || defined(__APPLE__)
  return std::ranges::equal(a, b, [](char x, char y) {
    return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
  });
#else
  return a == b;
#endif
}

// Le nom du fichier dans un chemin retenu (après le dernier séparateur).
std::string_view file_name(std::string_view path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string_view::npos ? path : path.substr(slash + 1);
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
  // Écrit à côté, sur le disque, puis renommé : une coupure ne laisse jamais un fichier à moitié
  // écrit ni vide.
  std::string text;
  for (const Entry& e : entries) text += std::format("{}\t{}\t{}\n", e.position_us, e.size, e.path);
  return core::write_text_atomically(store, text);
}

}  // namespace

std::optional<std::int64_t> resume_position(const std::filesystem::path& store,
                                            const std::filesystem::path& file) {
  const std::optional<Entry> wanted = identify(file);
  if (!wanted) return std::nullopt;
  const std::vector<Entry> entries = load(store);
  auto found = std::ranges::find_if(entries, [&](const Entry& e) { return same_path(e.path, wanted->path); });
  // Déplacée ou renommée de dossier : le même nom et la même taille (la plus récente).
  if (found == entries.end()) {
    const auto moved = std::ranges::find_if(entries.rbegin(), entries.rend(), [&](const Entry& e) {
      return e.size == wanted->size && same_path(file_name(e.path), file_name(wanted->path));
    });
    if (moved != entries.rend()) found = std::prev(moved.base());
  }
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
  std::erase_if(entries, [&](const Entry& e) {
    return same_path(e.path, entry->path) ||
           (e.size == entry->size && same_path(file_name(e.path), file_name(entry->path)));  // déplacée
  });
  const std::int64_t tail = std::clamp(duration_us / 10, kResumeTailMinUs, kResumeTailUs);
  const bool near_end = duration_us > 0 && position_us >= duration_us - tail;
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
