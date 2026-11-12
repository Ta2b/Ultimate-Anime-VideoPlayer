#include "app/play/crash_dump.h"

#include <algorithm>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "core/thread_role.h"
#include "core/utf8_path.h"

#ifdef _WIN32
#include <windows.h>
// dbghelp.h après windows.h, qui le précède forcément.
#include <array>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <dbghelp.h>
#else
#include <array>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#if __has_include(<execinfo.h>)
#include <cstdint>
#include <dlfcn.h>
#include <execinfo.h>
#define AJP_HAS_BACKTRACE 1
#endif
#endif

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

// L'étape en cours, lue par le gestionnaire de plantage sans verrou ni allocation : un octet nul final
// toujours là, une écriture concurrente ne peut que mêler deux étapes, jamais déborder.
std::array<char, 201> g_stage{};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex g_stage_writers;       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

std::size_t stage_length() {
  std::size_t length = 0;
  while (length + 1 < g_stage.size() && g_stage[length] != '\0') ++length;
  return length;
}

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

// Le dossier et le journal, fixés à l'installation : le gestionnaire n'alloue rien.
std::array<wchar_t, 1024> g_dir{};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::array<wchar_t, 1100> g_log{};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Une ligne ajoutée à ajp.log : la séance s'arrête, le journal doit le dire (A39-04). Le fichier est ouvert
// ici, sans attendre la file du journal, qui ne vide plus.
void note_crash(unsigned long code, const char* dump_name) {
  if (g_log[0] == L'\0') return;
  const HANDLE out = CreateFileW(g_log.data(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (out == INVALID_HANDLE_VALUE) return;
  std::array<char, 200> line{};
  const int length = std::snprintf(line.data(), line.size(),
                                   "[critical] plantage : code 0x%08lX, minidump %s\n", code, dump_name);
  DWORD written = 0;
  if (length > 0) WriteFile(out, line.data(), static_cast<DWORD>(length), &written, nullptr);
  if (const std::size_t stage = stage_length(); stage > 0) {
    constexpr std::string_view kStage = "[critical] dernière étape : ";  // i18n-ok : journal
    WriteFile(out, kStage.data(), static_cast<DWORD>(kStage.size()), &written, nullptr);
    WriteFile(out, g_stage.data(), static_cast<DWORD>(stage), &written, nullptr);
    WriteFile(out, "\n", 1, &written, nullptr);
  }
  CloseHandle(out);
}

bool dump(const wchar_t* file, EXCEPTION_POINTERS* info) {
  const HANDLE out =
      CreateFileW(file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (out == INVALID_HANDLE_VALUE) return false;
  MINIDUMP_EXCEPTION_INFORMATION exception{};
  exception.ThreadId = GetCurrentThreadId();
  exception.ExceptionPointers = info;
  exception.ClientPointers = FALSE;
  // Sans MiniDumpWithIndirectlyReferencedMemory : cette option emporte la mémoire pointée par les piles
  // (chemins complets, texte des sous-titres, nom d'utilisateur), et le vidage se joint à une issue publique
  // (A39-04).
  // MiniDumpFilterModulePaths retire le nom d'utilisateur des chemins de modules (C39-13).
  const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpNormal | MiniDumpWithThreadInfo |
                                               MiniDumpWithUnloadedModules | MiniDumpFilterModulePaths);
  const BOOL written = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), out, type,
                                         info != nullptr ? &exception : nullptr, nullptr, nullptr);
  CloseHandle(out);
  return written != FALSE;
}

void dump_to_dir(EXCEPTION_POINTERS* info) {
  if (g_dir[0] == L'\0') return;
  const unsigned long code = info != nullptr ? info->ExceptionRecord->ExceptionCode : 0;
  SYSTEMTIME now{};
  GetLocalTime(&now);
  std::array<wchar_t, 1100> file{};
  const int length = std::swprintf(
      file.data(), file.size(), L"%ls\\ajp-%04u%02u%02u-%02u%02u%02u-%lu.dmp", g_dir.data(),
      static_cast<unsigned>(now.wYear), static_cast<unsigned>(now.wMonth), static_cast<unsigned>(now.wDay),
      static_cast<unsigned>(now.wHour), static_cast<unsigned>(now.wMinute),
      static_cast<unsigned>(now.wSecond), static_cast<unsigned long>(GetCurrentProcessId()));
  if (length <= 0 || !dump(file.data(), info)) return;
  std::array<char, 64> name{};
  std::snprintf(name.data(), name.size(), "ajp-%04u%02u%02u-%02u%02u%02u-%lu.dmp",
                static_cast<unsigned>(now.wYear), static_cast<unsigned>(now.wMonth),
                static_cast<unsigned>(now.wDay), static_cast<unsigned>(now.wHour),
                static_cast<unsigned>(now.wMinute), static_cast<unsigned>(now.wSecond),
                static_cast<unsigned long>(GetCurrentProcessId()));
  note_crash(code, name.data());
}

// Un seul vidage par plantage : un paramètre invalide suivi de son abort n'en écrit pas deux.
std::atomic<bool> g_dumped{false};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void dump_once(EXCEPTION_POINTERS* info) {
  if (!g_dumped.exchange(true)) dump_to_dir(info);
}

LONG WINAPI on_unhandled(EXCEPTION_POINTERS* info) {
  dump_once(info);
  return EXCEPTION_CONTINUE_SEARCH;  // le rapport d'erreur de Windows suit
}

void on_abort(int /*signal*/) {
  dump_once(nullptr);
}

// C34-05 : un paramètre invalide passé au CRT (printf, strcpy_s…) et un appel virtuel pur finissaient par
// __fastfail, que le filtre des exceptions ne voit jamais : ni vidage ni ligne au journal. Ils en laissent
// un, puis le processus s'arrête comme avant.
void on_invalid_parameter(const wchar_t* /*expression*/, const wchar_t* /*function*/, const wchar_t* /*file*/,
                          unsigned int /*line*/, std::uintptr_t /*reserved*/) {
  dump_once(nullptr);
  std::abort();
}

void on_pure_call() {
  dump_once(nullptr);
  std::abort();
}
#else
// Un signal fatal : une ligne (et la pile, quand la libc en donne une) dans ajp.log, puis la fin normale du
// signal, pour que le système garde son core (systemd-coredump sous CachyOS). Rien n'est alloué dans le
// gestionnaire : le chemin, le tampon et la pile de secours sont fixés à l'installation, et write(2) est de
// ceux qu'un gestionnaire peut appeler (A39-04).
std::array<char, 1100> g_log_path{};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
constexpr std::array<int, 5> kFatalSignals{SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT};

void put(int fd, const char* text) {
  const ssize_t ignored = write(fd, text, std::strlen(text));
  static_cast<void>(ignored);
}

void put_number(int fd, long value) {
  std::array<char, 24> digits{};
  std::size_t at = digits.size();
  const bool negative = value < 0;
  unsigned long magnitude =
      negative ? 0UL - static_cast<unsigned long>(value) : static_cast<unsigned long>(value);
  do {
    digits[--at] = static_cast<char>('0' + magnitude % 10);
    magnitude /= 10;
  } while (magnitude != 0 && at > 1);
  if (negative) digits[--at] = '-';
  const ssize_t ignored = write(fd, digits.data() + at, digits.size() - at);
  static_cast<void>(ignored);
}

const char* signal_name(int signo) {
  switch (signo) {
    case SIGSEGV:
      return "SIGSEGV";
    case SIGBUS:
      return "SIGBUS";
    case SIGFPE:
      return "SIGFPE";
    case SIGILL:
      return "SIGILL";
    case SIGABRT:
      return "SIGABRT";
    default:
      return "?";
  }
}

#ifdef AJP_HAS_BACKTRACE
// La pile, une ligne par cadre : « nom-du-module(+0xdécalage) [adresse] », sans le dossier du module, que le
// journal ne doit pas dire (B39-05) ; backtrace_symbols_fd écrivait le chemin complet. dladdr est ce que
// backtrace_symbols_fd appelle lui-même.
void put_hex(int fd, std::uintptr_t value) {
  std::array<char, 2 + 2 * sizeof(value)> digits{};
  std::size_t at = digits.size();
  do {
    digits[--at] = "0123456789abcdef"[value & 0xFU];
    value >>= 4U;
  } while (value != 0 && at > 2);
  digits[--at] = 'x';
  digits[--at] = '0';
  const ssize_t ignored = write(fd, digits.data() + at, digits.size() - at);
  static_cast<void>(ignored);
}

void put_frames(int fd, void* const* frames, int count) {
  for (int i = 0; i < count; ++i) {
    const auto address = reinterpret_cast<std::uintptr_t>(frames[i]);  // NOLINT(*-reinterpret-cast)
    Dl_info info{};
    if (dladdr(frames[i], &info) != 0 && info.dli_fname != nullptr) {
      const char* name = info.dli_fname;
      for (const char* c = info.dli_fname; *c != '\0'; ++c)
        if (*c == '/') name = c + 1;
      put(fd, name);
      put(fd, "(+");
      put_hex(fd, address - reinterpret_cast<std::uintptr_t>(info.dli_fbase));  // NOLINT(*-reinterpret-cast)
      put(fd, ") ");
    }
    put(fd, "[");
    put_hex(fd, address);
    put(fd, "]\n");
  }
}
#endif

void on_fatal_signal(int signo) {
  const int fd = open(g_log_path.data(), O_WRONLY | O_APPEND | O_CLOEXEC);
  if (fd >= 0) {
    put(fd, "[critical] plantage : signal ");
    put_number(fd, signo);
    put(fd, " (");
    put(fd, signal_name(signo));
    put(fd, ")\n");
    if (const std::size_t stage = stage_length(); stage > 0) {
      put(fd, "[critical] dernière étape : ");  // i18n-ok : journal
      const ssize_t ignored = write(fd, g_stage.data(), stage);
      static_cast<void>(ignored);
      put(fd, "\n");
    }
#ifdef AJP_HAS_BACKTRACE
    std::array<void*, 48> frames{};
    const int count = backtrace(frames.data(), static_cast<int>(frames.size()));
    put_frames(fd, frames.data(), count);
#endif
    close(fd);
  }
  // SA_RESETHAND a remis l'action par défaut : le même signal, et le système fait son travail.
  raise(signo);
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
  const std::wstring log = (dir.parent_path() / "ajp.log").wstring();
  if (log.size() < g_log.size()) {
    std::ranges::copy(log, g_log.begin());
    g_log[log.size()] = L'\0';
  }
  SetUnhandledExceptionFilter(on_unhandled);
  std::signal(SIGABRT, on_abort);
  _set_invalid_parameter_handler(on_invalid_parameter);
  _set_purecall_handler(on_pure_call);
  // Un débordement de pile du thread principal laisse la place au filtre ; les threads de module la
  // réservent au démarrage (thread_start, core/thread_role.h). Un __fastfail (/GS, std::terminate de
  // certaines bibliothèques) ne passe par aucun gestionnaire du processus : seul le rapport d'erreur de
  // Windows le voit, et la séance restée ouverte le dit au lancement suivant (problem_report.h).
  ajp::reserve_crash_stack();
#else
  const std::string log = core::to_utf8(dir.parent_path() / "ajp.log");
  if (log.size() >= g_log_path.size()) return;
  std::ranges::copy(log, g_log_path.begin());
  g_log_path[log.size()] = '\0';
  // Une pile de secours : un débordement de pile ne laisse pas de place au gestionnaire sur la sienne.
  static std::array<char, std::size_t{64} * 1024> spare{};
  stack_t stack{};
  stack.ss_sp = spare.data();
  stack.ss_size = spare.size();
  sigaltstack(&stack, nullptr);
#ifdef AJP_HAS_BACKTRACE
  std::array<void*, 1> warm{};
  backtrace(warm.data(), 1);  // le premier appel charge la libc de pile : pas dans le gestionnaire
#endif
  for (const int signo : kFatalSignals) {
    struct sigaction action{};
    action.sa_handler = on_fatal_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = static_cast<int>(SA_RESETHAND | SA_ONSTACK);
    sigaction(signo, &action, nullptr);
  }
#endif
}

void note_crash_stage(std::string_view stage) {
  const std::scoped_lock lock(g_stage_writers);
  const std::size_t length = std::min(stage.size(), g_stage.size() - 1);
  std::ranges::copy(stage.substr(0, length), g_stage.begin());
  g_stage[length] = '\0';
}

std::string crash_stage() {
  const std::scoped_lock lock(g_stage_writers);
  return {g_stage.data(), stage_length()};
}

bool write_crash_dump([[maybe_unused]] const fs::path& file) {
#ifdef _WIN32
  return dump(file.c_str(), nullptr);
#else
  return false;
#endif
}

}  // namespace ajp::app::play
