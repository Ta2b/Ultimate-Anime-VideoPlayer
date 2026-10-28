#include "core/file_sync.h"

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

}  // namespace ajp::core
