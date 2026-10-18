// Lanceur Windows : CreateProcessW dans un job qui tue l'enfant avec le lecteur.
//
// Seuls le tube de sortie et le périphérique nul sont hérités
// (PROC_THREAD_ATTRIBUTE_HANDLE_LIST). La ligne de commande passe en UTF-16 :
// CreateProcessA relirait les chemins UTF-8 dans la page de code ANSI, comme le
// corrige aji_trt.cpp:709-714 du fork. Non compilé hors de Windows.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <array>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>
#include <windows.h>

#include "prepare/process.h"

namespace ajp::prepare {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr DWORD kPollSlice = 20;

class Handle {
 public:
  Handle() = default;
  explicit Handle(HANDLE handle) : handle_(handle) {}
  Handle(Handle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Handle& operator=(Handle&&) = delete;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() { reset(); }
  HANDLE get() const { return handle_; }
  HANDLE* out() { return &handle_; }
  bool valid() const { return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE; }
  void reset() {
    if (valid()) CloseHandle(handle_);
    handle_ = nullptr;
  }

 private:
  HANDLE handle_ = nullptr;
};

std::unexpected<Refusal> failure(const std::filesystem::path& program, std::string_view step) {
  const auto error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
  return std::unexpected(Refusal{
      RefusalCode::BuilderFailed,
      "lancement impossible de " + utf8_path(program) + " (" + std::string(step) + ") : " + error.message()});
}

std::wstring widen(std::string_view utf8) {
  if (utf8.empty()) return {};
  const int size = static_cast<int>(utf8.size());
  const int count = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), size, nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(count), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), size, wide.data(), count);
  return wide;
}

// Lit tout ce qui est disponible ; true quand tous les écrivains ont fermé.
bool drain(HANDLE pipe, LineSplitter& lines) {
  std::array<char, 4096> buffer{};
  for (;;) {
    DWORD available = 0;
    if (!PeekNamedPipe(pipe, nullptr, 0, nullptr, &available, nullptr)) return true;  // ERROR_BROKEN_PIPE
    if (available == 0) return false;
    DWORD got = 0;
    const DWORD wanted = std::min<DWORD>(available, static_cast<DWORD>(buffer.size()));
    if (!ReadFile(pipe, buffer.data(), wanted, &got, nullptr)) return true;
    lines.feed(std::string_view(buffer.data(), got));
  }
}

}  // namespace

std::expected<ProcessExit, Refusal> run_process(const ProcessSpec& spec, const LineSink& on_line,
                                                const std::stop_token& stop) {
  if (spec.timeout <= milliseconds::zero()) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "délai du constructeur nul ou négatif"});
  }
  SECURITY_ATTRIBUTES inherit{
      .nLength = sizeof(SECURITY_ATTRIBUTES), .lpSecurityDescriptor = nullptr, .bInheritHandle = TRUE};
  Handle read;
  Handle write;
  if (!CreatePipe(read.out(), write.out(), &inherit, 0)) return failure(spec.program, "CreatePipe");
  SetHandleInformation(read.get(), HANDLE_FLAG_INHERIT, 0);
  const Handle nul(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
  if (!nul.valid()) return failure(spec.program, "NUL");

  // Le job tue l'enfant si le lecteur meurt : sa dernière poignée se ferme avec lui.
  const Handle job(CreateJobObjectW(nullptr, nullptr));
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!job.valid() ||
      !SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
    return failure(spec.program, "job");
  }

  SIZE_T list_size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &list_size);
  std::vector<std::byte> list_storage(list_size);
  auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_storage.data());
  if (!InitializeProcThreadAttributeList(list, 1, 0, &list_size)) return failure(spec.program, "attributs");
  std::array<HANDLE, 2> inherited{write.get(), nul.get()};
  UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
                            sizeof(HANDLE) * inherited.size(), nullptr, nullptr);

  STARTUPINFOEXW info{};
  info.StartupInfo.cb = sizeof(info);
  info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  info.StartupInfo.hStdInput = nul.get();
  info.StartupInfo.hStdOutput = write.get();
  info.StartupInfo.hStdError = write.get();
  info.lpAttributeList = list;

  std::vector<std::string> argv{utf8_path(spec.program)};
  argv.insert(argv.end(), spec.args.begin(), spec.args.end());
  std::wstring command = widen(windows_command_line(argv));  // CreateProcessW peut la modifier
  const std::wstring application = spec.program.wstring();
  PROCESS_INFORMATION started{};
  const auto start = Clock::now();
  const BOOL ok = CreateProcessW(
      application.c_str(), command.data(), nullptr, nullptr, TRUE,
      CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
      nullptr, nullptr, &info.StartupInfo, &started);
  DeleteProcThreadAttributeList(list);
  if (!ok) return failure(spec.program, "CreateProcessW");
  const Handle process(started.hProcess);
  const Handle thread(started.hThread);
  if (!AssignProcessToJobObject(job.get(), process.get())) {
    TerminateProcess(process.get(), 1);
    return failure(spec.program, "AssignProcessToJobObject");
  }
  ResumeThread(thread.get());
  write.reset();  // seul l'enfant écrit : sa fin ferme le tube

  LineSplitter lines(on_line);
  std::optional<ExitKind> killed;
  bool eof = false;
  for (;;) {
    const auto now = Clock::now();
    if (!killed && (stop.stop_requested() || now >= start + spec.timeout)) {
      killed = stop.stop_requested() ? ExitKind::Cancelled : ExitKind::TimedOut;
      TerminateJobObject(job.get(), 1);
    }
    if (!eof) eof = drain(read.get(), lines);
    const auto left = std::chrono::duration_cast<milliseconds>(start + spec.timeout - now).count();
    const DWORD slice = killed ? INFINITE : static_cast<DWORD>(std::clamp<long long>(left, 0, kPollSlice));
    if (WaitForSingleObject(process.get(), slice) == WAIT_OBJECT_0) break;
  }
  if (!eof) drain(read.get(), lines);
  lines.finish();

  DWORD code = 0;
  GetExitCodeProcess(process.get(), &code);
  ProcessExit exit{.code = code, .elapsed = std::chrono::duration_cast<milliseconds>(Clock::now() - start)};
  if (killed) {
    exit.kind = *killed;
  } else if ((code & 0xF0000000U) == 0xC0000000U) {
    exit.kind = ExitKind::Crashed;  // NTSTATUS d'erreur : violation d'accès, pile épuisée…
  }
  return exit;
}

}  // namespace ajp::prepare
