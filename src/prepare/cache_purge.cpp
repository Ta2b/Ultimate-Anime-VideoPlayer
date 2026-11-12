#include "prepare/cache_purge.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "prepare/cache.h"
#include "prepare/cache_io.h"
#include "prepare/key_lock.h"
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

// Le reste d'une construction tuée : dossier de travail marqué, plus vieux que stale. Les octets rendus
// comptent dans removed_bytes.
bool remove_stale_stage(const fs::path& folder, const CachePurgeLimits& limits, fs::file_time_type now,
                        CachePurgeResult& result) {
  if (!utf8_path(folder.filename()).starts_with(kStagePrefix) || !is_stage_folder(folder)) return false;
  std::error_code error;
  const auto written = fs::last_write_time(folder, error);
  if (error || now - written <= limits.stale_build) return true;  // trop récent : peut-être vivant
  std::uint64_t bytes = 0;
  for (fs::recursive_directory_iterator walk(folder, fs::directory_options::skip_permission_denied, error),
       end;
       !error && walk != end; walk.increment(error)) {
    std::error_code size_error;
    if (walk->is_regular_file(size_error)) bytes += size_of(walk->path());
  }
  error.clear();
  fs::remove_all(folder, error);
  if (!error) {
    ++result.removed_stages;
    result.removed_bytes += bytes;
  }
  return true;
}

// Un verrou sans usage depuis limits.idle : pris sans attendre (jamais un verrou tenu), puis retiré. Sous
// Windows, retirer un fichier ouvert échoue : il reste, sans dommage.
void remove_idle_lock(const fs::path& lock, const CachePurgeLimits& limits, fs::file_time_type now,
                      CachePurgeResult& result) {
  std::error_code error;
  const auto written = fs::last_write_time(lock, error);
  if (error || now - written <= limits.idle) return;
  {
    auto held = KeyLock::acquire(lock, {}, LockWait::TryOnly);
    if (!held) return;
#ifndef _WIN32
    if (fs::remove(lock, error) && !error) ++result.removed_locks;
    return;
#endif
  }
#ifdef _WIN32
  // Le verrou est ouvert sans FILE_SHARE_DELETE : il se retire une fois relâché. Si un autre lancement le
  // prend entre-temps, la suppression échoue, sans dommage.
  if (fs::remove(lock, error) && !error) ++result.removed_locks;
#endif
}

// Le verrou d'un moteur disparu.
void remove_orphan_lock(const fs::path& lock, const CachePurgeLimits& limits, fs::file_time_type now,
                        CachePurgeResult& result) {
  const std::string name = utf8_path(lock.filename());
  if (!name.starts_with("aji-v3-") || !name.ends_with(".engine.key-lock")) return;
  fs::path engine = lock;
  engine.replace_extension();  // « .key-lock » retiré
  std::error_code error;
  if (fs::exists(engine, error)) return;
  remove_idle_lock(lock, limits, now, result);
}

// Le verrou du GPU (« aji-timing-<id>.cache.build-lock ») dont le timing cache n'existe plus (C05-11).
void remove_orphan_device_lock(const fs::path& lock, const CachePurgeLimits& limits, fs::file_time_type now,
                               CachePurgeResult& result) {
  fs::path timing = lock;
  timing.replace_extension();  // « .build-lock » retiré
  std::error_code error;
  if (fs::exists(timing, error)) return;
  remove_idle_lock(lock, limits, now, result);
}

// Un fichier « aji-timing-* » : un timing cache ou le verrou de son appareil, rangés pour plus tard ; un
// cache écarté ou le marqueur d'un plantage, retiré sans usage depuis idle (C05-11).
void sort_timing_file(const fs::path& file, const std::string& name, const CachePurgeLimits& limits,
                      fs::file_time_type now, std::vector<fs::path>& timing_caches,
                      std::vector<fs::path>& device_locks) {
  if (name.ends_with(".cache")) {
    timing_caches.push_back(file);
  } else if (name.ends_with(".cache.build-lock")) {
    device_locks.push_back(file);
  } else if (name.ends_with(".cache.bad") || name.ends_with(".cache.suspect")) {
    std::error_code time_error;
    const auto written = fs::last_write_time(file, time_error);
    if (!time_error && now - written > limits.idle) fs::remove(file, time_error);
  }
}

}  // namespace

CachePurgeResult purge_engine_cache(const fs::path& dir, const CachePurgeLimits& limits,
                                    fs::file_time_type now) {
  CachePurgeResult result;
  std::vector<Entry> entries;
  std::error_code error;
  fs::directory_iterator it(dir, error);
  std::vector<fs::path> locks;
  std::vector<fs::path> device_locks;
  std::vector<fs::path> timing_caches;
  for (; !error && it != fs::directory_iterator(); it.increment(error)) {
    std::error_code type_error;
    const fs::path& file = it->path();
    if (!it->is_regular_file(type_error)) {
      if (it->is_directory(type_error)) remove_stale_stage(file, limits, now, result);
      continue;
    }
    const std::string name = utf8_path(file.filename());
    if (name.ends_with(".key-lock")) {
      locks.push_back(file);
      continue;
    }
    if (name.starts_with("aji-timing-")) {
      sort_timing_file(file, name, limits, now, timing_caches, device_locks);
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
  // Un timing cache ne bouge qu'après une construction réussie, pas quand les moteurs de son GPU servent :
  // sa date seule le ferait partir au 46e jour d'un usage quotidien, et la prochaine taille remesurerait
  // toutes les tactiques. Il reste donc tant qu'un moteur est gardé, c'est-à-dire servi depuis idle (C05-06).
  if (result.kept == 0) {
    for (const fs::path& timing : timing_caches) {
      std::error_code time_error;
      const auto written = fs::last_write_time(timing, time_error);
      if (!time_error && now - written > limits.idle) fs::remove(timing, time_error);
    }
  }
  for (const fs::path& lock : locks)
    remove_orphan_lock(lock, limits, now, result);  // après les moteurs retirés
  for (const fs::path& lock : device_locks) remove_orphan_device_lock(lock, limits, now, result);
  return result;
}

}  // namespace ajp::prepare
