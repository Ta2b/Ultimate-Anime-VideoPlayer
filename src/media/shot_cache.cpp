#include "media/shot_cache.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

#include "core/dir_entries.h"

namespace ajp::media {
namespace {

// 2 : la lumière du plan (ShotMeta) ; 3 : son histogramme de I (lum_hist, 8.6). Un cache d'avant se refait.
constexpr std::array<char, 8> kMagic{'A', 'J', 'P', 'S', 'H', 'O', 'T', '3'};
constexpr std::size_t kHeadBytes = 1 << 20;
constexpr std::uint32_t kMaxShots = 200'000;
constexpr const char* kExtension = ".shots";

std::uint64_t fnv(std::uint64_t hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <class T>
void put(std::string& out, const T& value) {
  out.append(reinterpret_cast<const char*>(&value), sizeof(T));  // NOLINT(*-reinterpret-cast)
}

template <class T>
bool take(const std::string& in, std::size_t& at, T& value) {
  if (in.size() - at < sizeof(T)) return false;
  std::memcpy(&value, in.data() + at, sizeof(T));
  at += sizeof(T);
  return true;
}

void put_stats(std::string& out, const ShotStats& s) {
  for (const float v :
       {s.mean, s.p05, s.p50, s.p95, s.p995, s.dark, s.bright, s.blue, s.chroma, s.lit, s.apl_nits})
    put(out, v);
}

bool take_stats(const std::string& in, std::size_t& at, ShotStats& s) {
  return take(in, at, s.mean) && take(in, at, s.p05) && take(in, at, s.p50) && take(in, at, s.p95) &&
         take(in, at, s.p995) && take(in, at, s.dark) && take(in, at, s.bright) && take(in, at, s.blue) &&
         take(in, at, s.chroma) && take(in, at, s.lit) && take(in, at, s.apl_nits);
}

void put_meta(std::string& out, const ShotMeta& m) {
  for (const float v : {m.apl_min_nits, m.apl_mean_nits, m.apl_max_nits, m.max_scl_nits}) put(out, v);
  for (const float v : m.mean.maxrgb_nits) put(out, v);
  for (const float v :
       {m.mean.apl_nits, m.mean.maxrgb_mean_nits, m.mean.bright, m.mean.highlights, m.mean.skin})
    put(out, v);
  for (const float v : m.mean.lum_hist) put(out, v);
}

bool take_meta(const std::string& in, std::size_t& at, ShotMeta& m) {
  bool ok = take(in, at, m.apl_min_nits) && take(in, at, m.apl_mean_nits) && take(in, at, m.apl_max_nits) &&
            take(in, at, m.max_scl_nits);
  for (float& v : m.mean.maxrgb_nits) ok = ok && take(in, at, v);
  ok = ok && take(in, at, m.mean.apl_nits) && take(in, at, m.mean.maxrgb_mean_nits) &&
       take(in, at, m.mean.bright) && take(in, at, m.mean.highlights) && take(in, at, m.mean.skin);
  for (float& v : m.mean.lum_hist) ok = ok && take(in, at, v);
  return ok;
}

std::filesystem::path file_of(const std::filesystem::path& dir, const std::string& key) {
  return dir / (key + kExtension);
}

// Les fichiers du cache au-delà de keep, les plus anciens d'abord ; celui qui vient d'être écrit reste
// toujours (deux fichiers écrits dans le même tic d'horloge, sous Windows, auraient sinon la même date).
void trim(const std::filesystem::path& dir, std::size_t keep, const std::filesystem::path& written) {
  std::error_code ec;
  std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
  for (const auto& entry : core::dir_entries(dir, ec)) {
    if (entry.path().extension() != kExtension || entry.path() == written) continue;
    files.emplace_back(entry.last_write_time(ec), entry.path());
  }
  if (keep > 0) --keep;  // la place du fichier écrit
  if (files.size() <= keep) return;
  std::ranges::sort(files);
  for (std::size_t i = 0; i + keep < files.size(); ++i) std::filesystem::remove(files.at(i).second, ec);
}

}  // namespace

std::string shot_cache_key(const std::filesystem::path& video) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(video, ec);
  if (ec) return {};
  const auto modified = std::filesystem::last_write_time(video, ec).time_since_epoch().count();
  if (ec) return {};
  std::ifstream in(video, std::ios::binary);
  if (!in) return {};
  std::string head(std::min<std::uintmax_t>(size, kHeadBytes), '\0');
  in.read(head.data(), static_cast<std::streamsize>(head.size()));
  if (in.gcount() != static_cast<std::streamsize>(head.size())) return {};
  std::uint64_t hash = 14695981039346656037ULL;
  hash = fnv(hash, &size, sizeof size);
  hash = fnv(hash, &modified, sizeof modified);
  hash = fnv(hash, head.data(), head.size());
  return std::format("{:016x}", hash);
}

bool save_shot_cache(const std::filesystem::path& dir, const std::string& key, std::span<const Shot> shots,
                     bool complete, std::size_t keep) {
  if (key.empty() || shots.empty() || shots.size() > kMaxShots) return false;
  std::string out(kMagic.begin(), kMagic.end());
  put(out, static_cast<std::uint32_t>(shots.size()));
  put(out, static_cast<std::uint8_t>(complete ? 1 : 0));
  for (const Shot& shot : shots) {
    put(out, shot.start_us);
    put(out, shot.end_us);
    put_stats(out, shot.first);
    put_stats(out, shot.last);
    put_meta(out, shot.meta);
    put(out, static_cast<std::uint8_t>((shot.fade ? 1 : 0) | (shot.cut_before ? 2 : 0)));
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::filesystem::path file = file_of(dir, key);
  const std::filesystem::path temporary = file.string() + ".part";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(out.data(), static_cast<std::streamsize>(out.size()));
    if (!stream.flush()) return false;
  }
  std::filesystem::rename(temporary, file, ec);
  if (ec) {
    std::filesystem::remove(temporary, ec);
    return false;
  }
  trim(dir, keep, file);
  return true;
}

std::optional<CachedShots> load_shot_cache(const std::filesystem::path& dir, const std::string& key) {
  if (key.empty()) return std::nullopt;
  std::ifstream stream(file_of(dir, key), std::ios::binary);
  if (!stream) return std::nullopt;
  const std::string in((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (in.size() < kMagic.size() || !std::equal(kMagic.begin(), kMagic.end(), in.begin())) return std::nullopt;
  std::size_t at = kMagic.size();
  std::uint32_t count = 0;
  std::uint8_t complete = 0;
  if (!take(in, at, count) || count == 0 || count > kMaxShots || !take(in, at, complete)) return std::nullopt;
  std::vector<Shot> shots(count);
  std::int64_t previous_end = 0;
  for (Shot& shot : shots) {
    std::uint8_t flags = 0;
    if (!take(in, at, shot.start_us) || !take(in, at, shot.end_us) || !take_stats(in, at, shot.first) ||
        !take_stats(in, at, shot.last) || !take_meta(in, at, shot.meta) || !take(in, at, flags))
      return std::nullopt;
    if (shot.end_us <= shot.start_us || shot.start_us < previous_end) return std::nullopt;
    previous_end = shot.end_us;
    shot.fade = (flags & 1) != 0;
    shot.cut_before = (flags & 2) != 0;
  }
  if (at != in.size()) return std::nullopt;
  return CachedShots{.shots = std::move(shots), .complete = complete == 1};
}

}  // namespace ajp::media
