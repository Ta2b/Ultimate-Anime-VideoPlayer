#include "app/play/crash_dump.h"

#include <algorithm>
#include <iterator>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
// dbghelp.h après windows.h, qui le précède forcément.
#include <array>
#include <csignal>
#include <cwchar>
#include <dbghelp.h>
#endif

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

#ifdef _WIN32
// Les plus anciens vidages au-delà de keep - 1 : la place du prochain.
void prune(const fs::path& dir, int keep) {
  std::error_code error;
  std::vector<std::pair<fs::file_time_type, fs::path>> dumps;
  for (fs::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (it->path().extension() == ".dmp") dumps.emplace_back(it->last_write_time(error), it->path());
  }
  if (std::ssize(dumps) < keep) return;
  std::ranges::sort(dumps);
  for (std::size_t i = 0; i + static_cast<std::size_t>(keep) <= dumps.size(); ++i)
    fs::remove(dumps[i].second, error);
}

// Le dossier, fixé à l'installation : le gestionnaire n'alloue rien.
std::array<wchar_t, 1024> g_dir{};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

bool dump(const wchar_t* file, EXCEPTION_POINTERS* info) {
  const HANDLE out =
      CreateFileW(file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (out == INVALID_HANDLE_VALUE) return false;
  MINIDUMP_EXCEPTION_INFORMATION exception{};
  exception.ThreadId = GetCurrentThreadId();
  exception.ExceptionPointers = info;
  exception.ClientPointers = FALSE;
  const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpWithIndirectlyReferencedMemory |
                                               MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules);
  const BOOL written = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), out, type,
                                         info != nullptr ? &exception : nullptr, nullptr, nullptr);
  CloseHandle(out);
  return written != FALSE;
}

void dump_to_dir(EXCEPTION_POINTERS* info) {
  if (g_dir[0] == L'\0') return;
  SYSTEMTIME now{};
  GetLocalTime(&now);
  std::array<wchar_t, 1100> file{};
  const int length = std::swprintf(
      file.data(), file.size(), L"%ls\\ajp-%04u%02u%02u-%02u%02u%02u-%lu.dmp", g_dir.data(),
      static_cast<unsigned>(now.wYear), static_cast<unsigned>(now.wMonth), static_cast<unsigned>(now.wDay),
      static_cast<unsigned>(now.wHour), static_cast<unsigned>(now.wMinute),
      static_cast<unsigned>(now.wSecond), static_cast<unsigned long>(GetCurrentProcessId()));
  if (length > 0) dump(file.data(), info);
}

LONG WINAPI on_unhandled(EXCEPTION_POINTERS* info) {
  dump_to_dir(info);
  return EXCEPTION_CONTINUE_SEARCH;  // le rapport d'erreur de Windows suit
}

void on_abort(int /*signal*/) {
  dump_to_dir(nullptr);
}
#endif

}  // namespace

void install_crash_dumps([[maybe_unused]] const fs::path& dir) {
#ifdef _WIN32
  std::error_code error;
  fs::create_directories(dir, error);
  const std::wstring wide = dir.wstring();
  if (error || wide.size() >= g_dir.size()) return;
  prune(dir, kKeptCrashDumps);
  std::ranges::copy(wide, g_dir.begin());
  g_dir[wide.size()] = L'\0';
  SetUnhandledExceptionFilter(on_unhandled);
  std::signal(SIGABRT, on_abort);
#endif
}

bool write_crash_dump([[maybe_unused]] const fs::path& file) {
#ifdef _WIN32
  return dump(file.c_str(), nullptr);
#else
  return false;
#endif
}

}  // namespace ajp::app::play
