#include "core/file_sync.h"

#include <chrono>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
#include <string>
#include <thread>

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

std::string temp_suffix() {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::size_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::format("{:x}-{:x}", ticks, thread);
}

}  // namespace

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
      return std::unexpected(Refusal{RefusalCode::FileUnwritable, temp.string()});
    }
  }
  // Sur le disque avant le renommage : une coupure juste après ne laisse pas un fichier vide.
  if (const std::error_code synced = sync_file(temp)) {
    std::filesystem::remove(temp, error);
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", temp.string(), synced.message())});
  }
  std::filesystem::rename(temp, path, error);
  if (error) {
    const std::string why = error.message();
    std::filesystem::remove(temp, error);
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", path.string(), why)});
  }
  sync_directory(path.has_parent_path() ? path.parent_path() : std::filesystem::path("."));
  return {};
}

std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text) {
  return write_atomic(path, std::as_bytes(std::span(text)));
}

}  // namespace ajp::core
