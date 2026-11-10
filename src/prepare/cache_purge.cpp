#include "prepare/cache_purge.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

namespace fs = std::filesystem;

constexpr std::array<std::string_view, 3> kSiblings{".sha256", ".build.log", ".layers.json"};

struct Entry {
  fs::path engine;
  fs::file_time_type last_use;
  std::uint64_t bytes = 0;
};

fs::path with_suffix(const fs::path& file, std::string_view suffix) {
  fs::path out = file;
  out += suffix;
  return out;
}

std::uint64_t size_of(const fs::path& file) {
  std::error_code error;
  const auto size = fs::file_size(file, error);
  return error ? 0 : static_cast<std::uint64_t>(size);
}

// La dernière fois qu'il a servi : la date de son empreinte, rafraîchie à chaque reprise, ou celle du
// moteur.
fs::file_time_type last_use_of(const fs::path& engine) {
  std::error_code error;
  fs::file_time_type last = fs::last_write_time(engine, error);
  if (error) last = fs::file_time_type::min();
  const auto digest = fs::last_write_time(with_suffix(engine, ".sha256"), error);
  if (!error) last = std::max(last, digest);
  return last;
}

bool remove_entry(const Entry& entry) {
  std::error_code error;
  if (!fs::remove(entry.engine, error) || error) return false;
  for (const std::string_view suffix : kSiblings) fs::remove(with_suffix(entry.engine, suffix), error);
  return true;
}

}  // namespace

CachePurgeResult purge_engine_cache(const fs::path& dir, const CachePurgeLimits& limits,
                                    fs::file_time_type now) {
  CachePurgeResult result;
  std::vector<Entry> entries;
  std::error_code error;
  fs::directory_iterator it(dir, error);
  for (; !error && it != fs::directory_iterator(); it.increment(error)) {
    std::error_code type_error;
    if (!it->is_regular_file(type_error)) continue;
    const fs::path& file = it->path();
    const std::string name = utf8_path(file.filename());
    if (name.starts_with("aji-timing-") && name.ends_with(".cache")) {
      std::error_code time_error;
      const auto written = fs::last_write_time(file, time_error);
      if (!time_error && now - written > limits.idle) fs::remove(file, time_error);
      continue;
    }
    if (!name.starts_with("aji-v3-") || !name.ends_with(".engine")) continue;
    Entry entry{.engine = file, .last_use = last_use_of(file)};
    entry.bytes = size_of(file);
    for (const std::string_view suffix : kSiblings) entry.bytes += size_of(with_suffix(file, suffix));
    entries.push_back(std::move(entry));
  }
  // Les plus récents d'abord : ceux-là restent tant que la taille le permet.
  std::ranges::sort(entries, [](const Entry& a, const Entry& b) { return a.last_use > b.last_use; });
  for (const Entry& entry : entries) {
    const bool idle = now - entry.last_use > limits.idle;
    const bool recent = now - entry.last_use < limits.protected_recent;
    const bool over = result.kept_bytes + entry.bytes > limits.max_bytes;
    if ((idle || (over && !recent)) && remove_entry(entry)) {
      ++result.removed;
      result.removed_bytes += entry.bytes;
      continue;
    }
    ++result.kept;
    result.kept_bytes += entry.bytes;
  }
  return result;
}

}  // namespace ajp::prepare
