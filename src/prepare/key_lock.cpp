// Repris de animejanai-inference-rtx@118c272 src/cache_key_lock.h:21-76 :
// flock sous POSIX, LockFileEx sous Windows, attente par tranches de 20 ms.
#include "prepare/key_lock.h"

#include <chrono>
#include <thread>
#include <utility>

#include "prepare/cache_io.h"
#include "prepare/utf8.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace ajp::prepare {
namespace {

constexpr std::intptr_t kInvalid = -1;  // INVALID_HANDLE_VALUE comme descripteur absent
constexpr auto kPollInterval = std::chrono::milliseconds(20);

enum class Attempt : std::uint8_t { Locked, Busy, Failed };

#ifdef _WIN32
HANDLE as_handle(std::intptr_t handle) {
  return reinterpret_cast<HANDLE>(handle);
}

std::intptr_t open_lock(const std::filesystem::path& path, std::error_code& error) {
  HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) error = {static_cast<int>(GetLastError()), std::system_category()};
  return reinterpret_cast<std::intptr_t>(handle);
}

Attempt try_lock(std::intptr_t handle, std::error_code& error) {
  OVERLAPPED offset{};
  if (LockFileEx(as_handle(handle), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &offset)) {
    return Attempt::Locked;
  }
  const DWORD code = GetLastError();
  if (code == ERROR_LOCK_VIOLATION) return Attempt::Busy;
  error = {static_cast<int>(code), std::system_category()};
  return Attempt::Failed;
}

void close_lock(std::intptr_t handle) {
  CloseHandle(as_handle(handle));
}
#else
std::intptr_t open_lock(const std::filesystem::path& path, std::error_code& error) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) error = {errno, std::generic_category()};
  return fd;
}

Attempt try_lock(std::intptr_t handle, std::error_code& error) {
  if (::flock(static_cast<int>(handle), LOCK_EX | LOCK_NB) == 0) return Attempt::Locked;
  if (errno == EWOULDBLOCK || errno == EAGAIN || errno == EINTR) return Attempt::Busy;
  error = {errno, std::generic_category()};
  return Attempt::Failed;
}

void close_lock(std::intptr_t handle) {
  ::close(static_cast<int>(handle));
}
#endif

}  // namespace

std::expected<KeyLock, Refusal> KeyLock::acquire(const std::filesystem::path& lock_file,
                                                 const std::stop_token& stop, LockWait wait) {
  std::error_code error;
  KeyLock lock(open_lock(lock_file, error));
  if (lock.handle_ == kInvalid) return std::unexpected(io_refusal("ouverture du verrou", lock_file, error));
  for (;;) {
    if (stop.stop_requested()) {
      return std::unexpected(
          Refusal{RefusalCode::Cancelled, "annulé en attendant le verrou " + utf8_path(lock_file)});
    }
    switch (try_lock(lock.handle_, error)) {
      case Attempt::Locked:
        return lock;
      case Attempt::Failed:
        return std::unexpected(io_refusal("prise du verrou", lock_file, error));
      case Attempt::Busy:
        break;
    }
    if (wait == LockWait::TryOnly) {
      return std::unexpected(Refusal{RefusalCode::CacheBusy, "verrou tenu : " + utf8_path(lock_file)});
    }
    std::this_thread::sleep_for(kPollInterval);
  }
}

KeyLock::KeyLock(KeyLock&& other) noexcept : handle_(std::exchange(other.handle_, kInvalid)) {}

KeyLock& KeyLock::operator=(KeyLock&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, kInvalid);
  }
  return *this;
}

KeyLock::~KeyLock() {
  close();
}

void KeyLock::close() noexcept {
  if (handle_ != kInvalid) close_lock(handle_);
  handle_ = kInvalid;
}

}  // namespace ajp::prepare
