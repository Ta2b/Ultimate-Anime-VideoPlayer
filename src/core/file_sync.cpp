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
  // FlushFileBuffers demande une poignée ouverte en écriture.
  HANDLE handle = CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) return {static_cast<int>(GetLastError()), std::system_category()};
  const bool flushed = FlushFileBuffers(handle) != 0;
  const std::error_code error{flushed ? 0 : static_cast<int>(GetLastError()), std::system_category()};
  CloseHandle(handle);
  return error;
#else
  const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {errno, std::generic_category()};
  const bool flushed = ::fsync(fd) == 0;
  const std::error_code error{flushed ? 0 : errno, std::generic_category()};
  ::close(fd);
  return error;
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

// 10, 20, 40, 80, 160 puis 250 ms : environ 1,5 s en tout avant le refus.
constexpr int kRenameAttempts = 10;

// Sous Windows, le renommage échoue tant qu'un autre processus (antivirus, indexeur, sauvegarde) tient la
// destination ou le temporaire ouvert sans partage de suppression : refus passagers, réessayés pendant un peu
// plus d'une seconde (A20-01). Ailleurs, rename(2) remplace même un fichier ouvert : un refus est définitif.
bool transient_rename_refusal([[maybe_unused]] const std::error_code& error) {
#ifdef _WIN32
  if (error.category() != std::system_category()) return false;
  const int code = error.value();
  return code == ERROR_ACCESS_DENIED || code == ERROR_SHARING_VIOLATION || code == ERROR_LOCK_VIOLATION;
#else
  return false;
#endif
}

std::error_code rename_retrying(const std::filesystem::path& from, const std::filesystem::path& to) {
  std::error_code error;
  auto pause = std::chrono::milliseconds(10);
  for (int attempt = 0;; ++attempt) {
    error.clear();
    std::filesystem::rename(from, to, error);
    if (!error || attempt == kRenameAttempts - 1 || !transient_rename_refusal(error)) return error;
    std::this_thread::sleep_for(pause);
    pause = std::min(pause * 2, std::chrono::milliseconds(250));
  }
}

std::string temp_suffix() {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::size_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::format("{:x}-{:x}", ticks, thread);
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

std::expected<void, Refusal> write_atomic(const std::filesystem::path& path,
                                          std::span<const std::byte> bytes) {
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
  if (const std::error_code synced = sync_file(temp)) {
    std::filesystem::remove(temp, error);
    if (synced == std::errc::no_space_on_device)
      return std::unexpected(Refusal{RefusalCode::InsufficientSpace,
                                     std::format("disque plein : {} ({})", to_utf8(path), synced.message())});
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", to_utf8(temp), synced.message())});
  }
  if (const std::error_code renamed = rename_retrying(temp, path)) {
    std::filesystem::remove(temp, error);
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", to_utf8(path), renamed.message())});
  }
  sync_directory(path.has_parent_path() ? path.parent_path() : std::filesystem::path("."));
  return {};
}

std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text) {
  return write_atomic(path, std::as_bytes(std::span(text)));
}

}  // namespace ajp::core
