#include "app/play/problem_report.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <mutex>
#include <system_error>

#include "core/utf8_path.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/utsname.h>
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
};

State& state() {
  static State s;
  return s;
}

bool is_separator(char c) {
  return c == '/' || c == '\\';
}

// Ce qui peut précéder un chemin : le début, un blanc, une ponctuation d'ouverture ou de liaison, ou un
// caractère hors ASCII (flèche, espace insécable : ses octets valent 0x80 et plus). Liste large exprès : un
// chemin non reconnu part en clair, avec le nom d'utilisateur, dans une issue publique (B19-05, B39-07).
bool opens_path(char before) {
  if (static_cast<unsigned char>(before) >= 0x80) return true;
  switch (before) {
    case ' ':
    case '\t':
    case '\n':
    case '"':
    case '\'':
    case '(':
    case '[':
    case '{':
    case '=':
    case ':':
    case ',':
    case ';':
    case '<':
    case '>':
    case '|':
    case '@':
    case '`':
    case '*':
      return true;
    default:
      return false;
  }
}

// Le dossier personnel nu (« /home/nom », « C:\\Users\\nom », sans fichier après) : réduit à « …/nom » il
// laisserait le nom d'utilisateur ; il devient « ~ ».
bool is_home_folder(std::string_view path) {
  std::string_view rest = path;
  if (rest.size() > 2 && rest[1] == ':') rest.remove_prefix(2);
  for (const std::string_view prefix : {"/home/", "/Users/", "/users/", "\\Users\\"}) {
    if (rest.starts_with(prefix)) {
      const std::string_view name = rest.substr(prefix.size());
      return !name.empty() && name.find_first_of("/\\") == std::string_view::npos;
    }
  }
  return false;
}

// Le début d'un chemin absolu en text[at] : « X:\ », « X:/ », « \\ » (réseau), « / » suivi d'un nom, « ~/ ».
bool path_starts(std::string_view text, std::size_t at) {
  const char c = text[at];
  const char next = at + 1 < text.size() ? text[at + 1] : '\0';
  if (c == '/') return next != '\0' && next != ' ' && next != '/' && next != '\n';
  if (c == '\\') return next == '\\';
  if (c == '~') return is_separator(next);
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
    return next == ':' && at + 2 < text.size() && is_separator(text[at + 2]);
  return false;
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

void note_engine_facts(std::string_view engine, std::string_view passed_over) {
  note_report_fact("Moteur", engine);
  if (!passed_over.empty()) note_report_fact("Moteurs refusés", passed_over);
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
  std::string out;
  out.reserve(text.size());
  std::size_t at = 0;
  while (at < text.size()) {
    const bool line_start = at == 0 || text[at - 1] == '\n';
    // « file:///home/nom/a.mkv » : le chemin commence après « file:// ».
    const bool after_scheme = at >= 7 && text.substr(at - 7, 7) == "file://";
    if (!(line_start || after_scheme || opens_path(text[at - 1])) || !path_starts(text, at)) {
      out.push_back(text[at++]);
      continue;
    }
    // L'étendue : un nom de dossier peut porter des espaces (« Données SSD »), elle court donc jusqu'à une
    // fin de ligne, un guillemet, un séparateur du journal (« , », « : », « · », « ; »), la parenthèse qui
    // ferme celle d'où le chemin sort, ou un autre chemin.
    const char opener = at == 0 ? '\0' : text[at - 1];
    std::size_t end = at;
    while (end < text.size()) {
      const char c = text[end];
      if (c == '\n' || c == '"') break;
      const std::string_view rest = text.substr(end);
      if (rest.starts_with(", ") || rest.starts_with(" : ") || rest.starts_with(" · ") ||
          rest.starts_with("; "))
        break;
      if (opener == '(' && c == ')') break;
      if (c == ' ' && end + 1 < text.size() && path_starts(text, end + 1)) break;
      ++end;
    }
    std::size_t last = std::string_view::npos;
    std::size_t separators = 0;
    for (std::size_t i = at; i < end; ++i) {
      if (is_separator(text[i])) {
        last = i;
        ++separators;
      }
    }
    // « /s », « /x » : un seul séparateur, pas un dossier.
    if (text[at] == '/' && separators < 2) {
      out.push_back(text[at++]);
      continue;
    }
    if (is_home_folder(text.substr(at, end - at))) {
      out += '~';
      at = end;
      continue;
    }
    out += "…/";
    out.append(text.substr(last + 1, end - last - 1));
    at = end;
  }
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
  constexpr std::string_view kPrevious = "--- séance précédente (arrêtée brutalement) ---\n";
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

bool begin_session(const fs::path& dir) {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  s.session_dir = dir;
  std::error_code error;
  const fs::path mark = dir / "session.open";
  const bool existed = fs::exists(mark, error);
  std::ofstream(mark, std::ios::binary).put('\n');
  s.previous_crashed = existed;
  return existed;
}

void end_session() {
  State& s = state();
  const std::scoped_lock lock(s.mutex);
  if (s.session_dir.empty()) return;
  std::error_code error;
  fs::remove(s.session_dir / "session.open", error);
  s.session_dir.clear();
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
    if (previous_session_crashed()) {
      fs::path previous = log;
      previous += ".1";
      parts.previous_log = read_tail(previous, kTail);
    }
    parts.dump_name = latest_dump_name(log.parent_path() / "crashes");
  }
  if (parts.current_log.empty()) parts.current_log = "(journal introuvable)\n";
  return compose_report(parts, kBody);
}

}  // namespace ajp::app::play
