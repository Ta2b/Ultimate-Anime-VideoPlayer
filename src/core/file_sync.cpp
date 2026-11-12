#include "core/file_sync.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
#include <string>
#include <thread>

#include "core/utf8_path.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ajp::core {

std::error_code sync_file(const std::filesystem::path& file) {
#ifdef _WIN32
  // FlushFileBuffers demande une poignée ouverte en écriture. Tout partage, suppression comprise : un
  // antivirus qui a ouvert le temporaire avec FILE_SHARE_DELETE ne fait pas refuser l'ouverture (B31-17).
  HANDLE handle =
      CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) return {static_cast<int>(GetLastError()), std::system_category()};
  const bool flushed = FlushFileBuffers(handle) != 0;
  const std::error_code error{flushed ? 0 : static_cast<int>(GetLastError()), std::system_category()};
  CloseHandle(handle);
  return error;
#else
  const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {errno, std::generic_category()};
  const bool flushed = ::fsync(fd) == 0;
  int code = flushed ? 0 : errno;
  // Un système de fichiers qui ne sait pas vider (FUSE, fichier spécial) : les octets sont écrits, rien de
  // plus à faire. Seuls le disque plein et une erreur d'écriture refusent (B20-01).
  if (code == EINVAL || code == ENOTSUP || code == EROFS) code = 0;
  ::close(fd);
  return {code, std::generic_category()};
#endif
}

void sync_directory([[maybe_unused]] const std::filesystem::path& folder) {
#ifndef _WIN32
  const int fd = ::open(folder.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  if (fd < 0) return;
  ::fsync(fd);
  ::close(fd);
#endif
}

namespace {

// Sous Windows, le renommage, et l'ouverture du temporaire pour le vider, échouent tant qu'un autre processus
// (antivirus, indexeur, sauvegarde) tient la destination ou le temporaire ouvert sans le partage voulu :
// refus passagers, réessayés pendant un peu plus d'une seconde (A20-01, B20-01, B36-03). Ailleurs, rename(2)
// remplace même un fichier ouvert : un refus est définitif.
bool transient_refusal([[maybe_unused]] const std::error_code& error) {
#ifdef _WIN32
  if (error.category() != std::system_category()) return false;
  const int code = error.value();
  return code == ERROR_ACCESS_DENIED || code == ERROR_SHARING_VIOLATION || code == ERROR_LOCK_VIOLATION;
#else
  return false;
#endif
}

std::string temp_suffix() {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::size_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::format("{:x}-{:x}", ticks, thread);
}

}  // namespace

namespace {

// attempt tant qu'il rend un refus passager, pendant patience au plus : 10, 20, 40, 80, 160 puis 250 ms
// entre deux essais.
std::error_code retrying(const std::function<std::error_code()>& attempt,
                         std::chrono::milliseconds patience = kRenamePatience) {
  const auto deadline = std::chrono::steady_clock::now() + patience;
  auto pause = std::chrono::milliseconds(10);
  for (;;) {
    const std::error_code error = attempt();
    if (!error || !transient_refusal(error)) return error;
    const auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (left <= std::chrono::milliseconds::zero()) return error;
    std::this_thread::sleep_for(std::min(pause, left));
    pause = std::min(pause * 2, std::chrono::milliseconds(250));
  }
}

}  // namespace

std::error_code rename_retrying(const std::filesystem::path& from, const std::filesystem::path& to,
                                std::chrono::milliseconds patience) {
  return retrying(
      [&] {
        std::error_code error;
        std::filesystem::rename(from, to, error);
        return error;
      },
      patience);
}

std::error_code sync_file_retrying(const std::filesystem::path& file, std::chrono::milliseconds patience) {
  return retrying([&] { return sync_file(file); }, patience);
}

namespace {

// Un temporaire plus vieux que ceci n'a plus d'écrivain : un arrêt brutal l'a laissé (A20-08). L'âge laisse
// tranquille celui qu'un autre processus écrit en ce moment.
constexpr auto kOrphanAge = std::chrono::hours(1);

// La destination réelle : un lien symbolique (play.toml mis dans un dépôt de fichiers de configuration, par
// exemple) reste un lien, c'est le fichier qu'il nomme qui est remplacé (A20-08). Les liens en chaîne sont
// suivis ; un lien cassé désigne le fichier qu'il nommerait.
std::filesystem::path real_destination(const std::filesystem::path& path) {
  std::filesystem::path target = path;
  std::error_code error;
  for (int hops = 0; hops < 40 && std::filesystem::is_symlink(target, error); ++hops) {
    std::filesystem::path next = std::filesystem::read_symlink(target, error);
    if (error) return path;
    target = next.is_absolute() ? next : target.parent_path() / next;
  }
  return target;
}

// Les « .<nom>.*.tmp » abandonnés à côté du fichier, de ses écritures d'avant.
void remove_orphan_temps(const std::filesystem::path& path) {
  const std::filesystem::path folder =
      path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  const std::string prefix = "." + to_utf8(path.filename()) + ".";
  std::error_code error;
  std::filesystem::directory_iterator it(folder, error);
  if (error) return;
  const auto now = std::filesystem::file_time_type::clock::now();
  for (const std::filesystem::directory_iterator end; it != end; it.increment(error)) {
    if (error) return;
    const std::string name = to_utf8(it->path().filename());
    if (!name.starts_with(prefix) || !name.ends_with(".tmp")) continue;
    std::error_code stale;
    const auto written = it->last_write_time(stale);
    if (!stale && now - written > kOrphanAge) std::filesystem::remove(it->path(), stale);
  }
}

}  // namespace

std::optional<Refusal> lacking_space(const std::filesystem::path& dir, std::uintmax_t needed) {
  std::error_code error;
  std::filesystem::path at = dir.empty() ? std::filesystem::path(".") : dir;
  while (!std::filesystem::exists(at, error) && at.has_parent_path() && at.parent_path() != at)
    at = at.parent_path();
  const std::filesystem::space_info space = std::filesystem::space(at, error);
  if (error || space.available >= needed) return std::nullopt;
  constexpr double kMiB = 1024.0 * 1024.0;
  return Refusal{RefusalCode::InsufficientSpace,
                 std::format("disque plein : {:.0f} Mo libres sur {}, {:.0f} Mo demandés",
                             static_cast<double>(space.available) / kMiB, to_utf8(at),
                             static_cast<double>(needed) / kMiB)};
}

std::expected<void, Refusal> write_atomic(const std::filesystem::path& given,
                                          std::span<const std::byte> bytes) {
  const std::filesystem::path path = real_destination(given);
  std::error_code error;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
  std::filesystem::path name = ".";
  name += path.filename();
  name += "." + temp_suffix() + ".tmp";
  const std::filesystem::path temp = path.parent_path() / name;
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();  // la fermeture vide le tampon : son échec compte aussi
    if (!out) {
      std::filesystem::remove(temp, error);
      if (auto full = lacking_space(path.parent_path(), bytes.size()))
        return std::unexpected(std::move(*full));
      return std::unexpected(Refusal{RefusalCode::FileUnwritable, to_utf8(temp)});
    }
  }
  // Sur le disque avant le renommage : une coupure juste après ne laisse pas un fichier vide.
  if (const std::error_code synced = retrying([&] { return sync_file(temp); })) {
    std::filesystem::remove(temp, error);
    if (synced == std::errc::no_space_on_device)
      return std::unexpected(Refusal{RefusalCode::InsufficientSpace,
                                     std::format("disque plein : {} ({})", to_utf8(path), synced.message())});
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", to_utf8(temp), synced.message())});
  }
  // Les droits de l'ancien fichier (un play.toml mis en 0600) restent les siens (C20-08).
  if (const auto kept = std::filesystem::status(path, error); !error && std::filesystem::exists(kept))
    std::filesystem::permissions(temp, kept.permissions(), error);
  if (const std::error_code renamed = rename_retrying(temp, path)) {
    std::filesystem::remove(temp, error);
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", to_utf8(path), renamed.message())});
  }
  sync_directory(path.has_parent_path() ? path.parent_path() : std::filesystem::path("."));
  remove_orphan_temps(path);
  return {};
}

std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text) {
  return write_atomic(path, std::as_bytes(std::span(text)));
}

}  // namespace ajp::core
