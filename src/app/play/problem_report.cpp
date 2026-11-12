#include "app/play/problem_report.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>

#include "core/log_redact.h"
#include "core/utf8_path.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <cstdint>
#include <cstring>
#include <elf.h>
#include <link.h>
#endif

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

struct State {
  std::mutex mutex;
  std::vector<std::pair<std::string, std::string>> facts;
  fs::path session_dir;
  fs::path log_file;
  bool previous_crashed = false;
  bool owns_mark = false;           // session.open porte le PID de ce processus : seul lui la retire
  std::string previous_relaunched;  // la raison de la relance voulue qui a lancé cette séance ; vide : aucune
};

#ifdef __linux__
// Le premier objet que dl_iterate_phdr donne est l'exécutable : ses notes ELF portent le build-id.
int find_build_id(dl_phdr_info* info, std::size_t /*size*/, void* data) {
  std::string& hex = *static_cast<std::string*>(data);
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) & header = info->dlpi_phdr[i];
    if (header.p_type != PT_NOTE) continue;
    // L'adresse du segment est un entier (base de chargement + décalage) : le cast est voulu.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const auto* at = reinterpret_cast<const std::uint8_t*>(info->dlpi_addr + header.p_vaddr);
    const std::uint8_t* end = at + header.p_memsz;
    while (static_cast<std::size_t>(end - at) >= sizeof(ElfW(Nhdr))) {
      ElfW(Nhdr) note;
      std::memcpy(&note, at, sizeof note);
      const std::size_t desc_at = sizeof note + ((note.n_namesz + 3U) & ~3U);
      const std::size_t next = desc_at + ((note.n_descsz + 3U) & ~3U);
      if (next > static_cast<std::size_t>(end - at)) break;
      if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
          std::memcmp(at + sizeof note, "GNU", 4) == 0) {
        constexpr std::string_view kDigits = "0123456789abcdef";
        for (std::uint32_t b = 0; b < note.n_descsz; ++b) {
          hex += kDigits[at[desc_at + b] >> 4U];
          hex += kDigits[at[desc_at + b] & 15U];
        }
        return 1;
      }
      at += next;
    }
  }
  return 1;  // seulement le premier objet
}
#endif

State& state() {
  static State s;
  return s;
}

}  // namespace

void note_report_fact(std::string_view name, std::string_view value) {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  for (auto& [key, old] : s.facts) {
    if (key == name) {
      old = std::string(value);
      return;
    }
  }
  s.facts.emplace_back(std::string(name), std::string(value));
}

void forget_report_fact(std::string_view name) {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  std::erase_if(s.facts, [&](const auto& fact) { return fact.first == name; });
}

void note_engine_facts(std::string_view engine, std::string_view passed_over) {
  note_report_fact("Moteur", engine);
  // Un refus d'un moteur précédent ne reste pas affiché quand le moteur suivant a réussi sans refus (C39-06).
  if (!passed_over.empty())
    note_report_fact("Moteurs refusés", passed_over);
  else
    forget_report_fact("Moteurs refusés");
}

void note_build_failure(std::string_view code, std::string_view detail) {
  constexpr std::string_view kMarker = " ; journal : ";
  constexpr std::size_t kLogBytes = std::size_t{6} * 1024;
  constexpr std::size_t kLogLines = 40;
  std::string text(code);
  const std::size_t at = detail.rfind(kMarker);
  // La cause d'abord, sans le chemin du journal (il sert à le lire, il ne dit rien de plus).
  const std::string_view cause = at == std::string_view::npos ? detail : detail.substr(0, at);
  if (!cause.empty()) text.append(" : ").append(cause);
  if (at != std::string_view::npos) {
    std::string log = read_tail(core::path_from_utf8(detail.substr(at + kMarker.size())), kLogBytes);
    while (!log.empty() && (log.back() == '\n' || log.back() == '\r')) log.pop_back();
    // Les dernières lignes seulement : la fin de trtexec dit l'erreur, le début la commande et les tactiques.
    std::size_t lines = 0;
    for (std::size_t i = log.size(); i-- > 0;) {
      if (log[i] != '\n' || ++lines <= kLogLines) continue;
      log.erase(0, i + 1);
      break;
    }
    if (!log.empty()) text.append("\n").append(log);
  }
  note_report_fact("Construction", text);
}

std::vector<std::pair<std::string, std::string>> report_facts() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  return s.facts;
}

void clear_report_facts() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  s.facts.clear();
}

std::string build_id_description() {
#ifdef __linux__
  std::string hex;
  dl_iterate_phdr(find_build_id, &hex);
  return hex;
#else
  return {};
#endif
}

std::string os_description() {
#ifdef _WIN32
  using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
  OSVERSIONINFOW info{};
  info.dwOSVersionInfoSize = sizeof(info);
  if (const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"); ntdll != nullptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto get =
        reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")));
    if (get != nullptr && get(&info) == 0)
      return "Windows " + std::to_string(info.dwMajorVersion) + "." + std::to_string(info.dwMinorVersion) +
             "." + std::to_string(info.dwBuildNumber);
  }
  return "Windows";
#else
  std::string name;
#ifdef __linux__
  std::ifstream release("/etc/os-release");
  for (std::string line; std::getline(release, line);) {
    constexpr std::string_view kKey = "PRETTY_NAME=";
    if (line.starts_with(kKey)) {
      name = line.substr(kKey.size());
      std::erase(name, '"');
      break;
    }
  }
#endif
  utsname uts{};
  if (uname(&uts) != 0) return name.empty() ? "inconnu" : name;
  const std::string kernel = std::string(uts.sysname) + " " + uts.release;
  return name.empty() ? kernel : name + " · " + kernel;
#endif
}

std::string sanitize_report_paths(std::string_view text, std::string_view home) {
  std::string out = reduce_log_paths(text);
  if (!home.empty()) {
    // Le dossier personnel qui aurait échappé au journal (autre casse, autre séparateur).
    for (std::size_t from = out.find(home); from != std::string::npos; from = out.find(home, from)) {
      out.replace(from, home.size(), "~");
      ++from;
    }
  }
  return out;
}

// Dit au lecteur du rapport que le début du journal manque (A23-14).
constexpr std::string_view kLogCut = "[… début du journal omis]\n";

std::string read_tail(const fs::path& file, std::size_t max_bytes) {
  std::ifstream in(file, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const auto size = static_cast<std::streamoff>(in.tellg());
  const std::streamoff start = std::max<std::streamoff>(0, size - static_cast<std::streamoff>(max_bytes));
  in.seekg(start);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (start > 0) {
    // Coupé au milieu d'une ligne : la première, partielle, saute.
    const std::size_t newline = text.find('\n');
    text.erase(0, newline == std::string::npos ? text.size() : newline + 1);
    text.insert(0, kLogCut);
  }
  return text;
}

std::string latest_dump_name(const fs::path& dir) {
  std::error_code error;
  std::string best;
  fs::file_time_type best_time{};
  for (fs::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (it->path().extension() != ".dmp") continue;
    const auto time = it->last_write_time(error);
    if (best.empty() || time > best_time) {
      best = core::to_utf8(it->path().filename());
      best_time = time;
    }
  }
  return best;
}

std::string compose_report(const ReportParts& parts, std::size_t max_bytes) {
  std::string head;
  for (const auto& [name, value] : parts.facts) {
    head.append(name).append(" : ").append(value).append("\n");
  }
  if (!parts.dump_name.empty())
    head.append("Minidump : ")
        .append(parts.dump_name)
        .append(" (à joindre en privé si on le demande, il peut contenir des chemins)\n");
  head = sanitize_report_paths(head, parts.home);
  head += "\n";
  const std::string previous = sanitize_report_paths(parts.previous_log, parts.home);
  const std::string current = sanitize_report_paths(parts.current_log, parts.home);
  const std::string kPrevious = parts.previous_title.empty()
                                    ? std::string("--- séance précédente (arrêtée brutalement) ---\n")
                                    : std::format("--- séance précédente ({}) ---\n", parts.previous_title);
  constexpr std::string_view kCurrent = "--- séance en cours ---\n";
  const bool both = !previous.empty();
  std::size_t room = max_bytes > head.size() ? max_bytes - head.size() : 0;
  if (both) room = room > kPrevious.size() + kCurrent.size() ? room - kPrevious.size() - kCurrent.size() : 0;
  // La séance d'avant garde la moitié de la place : c'est elle qui porte le plantage.
  const std::size_t previous_room = both ? room / 2 : 0;
  const std::size_t current_room = room - previous_room;
  const auto tail = [](const std::string& text, std::size_t room_bytes) {
    if (text.size() <= room_bytes) return text;
    if (room_bytes <= kLogCut.size()) return std::string();
    std::string cut = text.substr(text.size() - (room_bytes - kLogCut.size()));
    const std::size_t newline = cut.find('\n');
    cut.erase(0, newline == std::string::npos ? cut.size() : newline + 1);
    return std::string(kLogCut) + cut;
  };
  std::string report = head;
  if (both) {
    report += kPrevious;
    report += tail(previous, previous_room);
    if (!report.empty() && report.back() != '\n') report += '\n';
    report += kCurrent;
  }
  report += tail(current, current_room);
  return report;
}

namespace {

std::uint64_t own_pid() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

// Le processus de ce PID tourne-t-il ? Une marque d'un lecteur encore là n'est pas celle d'un plantage.
bool process_alive(std::uint64_t pid) {
  if (pid == 0) return false;
#ifdef _WIN32
  const HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
  if (process == nullptr) return GetLastError() == ERROR_ACCESS_DENIED;
  DWORD code = 0;
  const bool alive = GetExitCodeProcess(process, &code) != 0 && code == STILL_ACTIVE;
  CloseHandle(process);
  return alive;
#else
  return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
}

}  // namespace

bool begin_session(const fs::path& dir) {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  s.session_dir = dir;
  std::error_code error;
  const fs::path mark = dir / "session.open";
  const bool existed = fs::exists(mark, error);
  // Le PID du lecteur qui l'a posée (l'ancien format n'avait qu'un saut de ligne : inconnu). Un autre lecteur
  // encore vivant sur ce dossier (un lancement qui ne passe pas par l'instance unique) n'a pas planté : sa
  // marque reste la sienne, ce lancement n'y touche pas et ne la retirera pas à sa sortie.
  std::uint64_t other = 0;
  if (existed) {
    std::ifstream in(mark, std::ios::binary);
    in >> other;
  }
  if (existed && other != 0 && other != own_pid() && process_alive(other)) {
    s.owns_mark = false;
    s.previous_crashed = false;
    return false;
  }
  std::ofstream(mark, std::ios::binary) << own_pid() << '\n';
  s.owns_mark = true;
  s.previous_crashed = existed;
  return existed;
}

void end_session() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  if (s.session_dir.empty()) return;
  std::error_code error;
  if (s.owns_mark) fs::remove(s.session_dir / "session.open", error);
  s.owns_mark = false;
  s.session_dir.clear();
}

void note_previous_session_relaunched(std::string_view why) {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  s.previous_relaunched = std::string(why);
}

std::string previous_session_relaunched() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  return s.previous_relaunched;
}

bool previous_session_crashed() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  return s.previous_crashed;
}

void set_report_log_file(const fs::path& file) {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  s.log_file = file;
}

fs::path report_log_file() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  return s.log_file;
}

std::string current_report(std::string_view home) {
  constexpr std::size_t kBody = std::size_t{60} * 1024;  // sous les 65 536 caractères d'un corps d'issue
  constexpr std::size_t kTail = std::size_t{48} * 1024;
  const fs::path log = report_log_file();
  ReportParts parts;
  parts.facts = report_facts();
  parts.home = std::string(home);
  if (!log.empty()) {
    parts.current_log = read_tail(log, kTail);
    const std::string relaunched = previous_session_relaunched();
    if (previous_session_crashed() || !relaunched.empty()) {
      fs::path previous = log;
      previous += ".1";
      parts.previous_log = read_tail(previous, kTail);
      if (!previous_session_crashed()) parts.previous_title = "relancée : " + relaunched;
    }
    // Le minidump n'est nommé que si la séance d'avant a planté : un vieux .dmp sans lien ferait demander un
    // fichier pour rien (C39-07).
    if (previous_session_crashed()) parts.dump_name = latest_dump_name(log.parent_path() / "crashes");
  }
  if (parts.current_log.empty()) parts.current_log = "(journal introuvable)\n";
  return compose_report(parts, kBody);
}

}  // namespace ajp::app::play
