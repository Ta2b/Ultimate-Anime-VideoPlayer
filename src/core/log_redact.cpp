#include "core/log_redact.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/sink.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "core/utf8_path.h"
#endif

namespace ajp {
namespace {

struct Home {
  std::mutex mutex;
  std::vector<std::string> forms;   // sous mutex : le dossier sous chaque séparateur, et sa forme courte
  std::vector<std::string> mounts;  // sous mutex : « /run/media/<nom>/ », « /media/<nom>/ »
};

Home& home() {
  static Home state;
  return state;
}

// Sous Windows, les chemins ne distinguent pas la casse : « c:\users\jean » est le même dossier.
constexpr bool kFoldCase =
#ifdef _WIN32
    true;
#else
    false;
#endif

char fold(char c) {
  return kFoldCase && c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

bool same_at(std::string_view text, std::size_t at, std::string_view form) {
  if (at + form.size() > text.size()) return false;
  for (std::size_t i = 0; i < form.size(); ++i)
    if (fold(text[at + i]) != fold(form[i])) return false;
  return true;
}

// Un nom de dossier continue après la forme (« /home/al » dans « /home/alice ») : pas une frontière.
bool name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
         c == '-' || c == '~' || static_cast<unsigned char>(c) >= 0x80;
}

// La forme finit à une frontière de composant : séparateur, fin, espace, ponctuation. Un point n'en est une
// que s'il n'est pas suivi d'un caractère de nom (« /home/jean. » finit la phrase, « /home/jean.x » est un
// autre dossier) (B20-02).
bool ends_component(std::string_view text, std::size_t after) {
  if (after == text.size()) return true;
  if (text[after] == '.') return after + 1 == text.size() || !name_char(text[after + 1]);
  return !name_char(text[after]);
}

// Remplace form par to là où il finit à une frontière de composant.
void replace_all(std::string& text, std::string_view form, std::string_view to) {
  std::size_t at = 0;
  while (at + form.size() <= text.size()) {
    const std::size_t after = at + form.size();
    if (same_at(text, at, form) && ends_component(text, after)) {
      text.replace(at, form.size(), to);
      at += to.size();
    } else {
      ++at;
    }
  }
}

std::string user_of(std::string_view dir) {
  const std::size_t cut = dir.find_last_of("/\\");
  return std::string(cut == std::string_view::npos ? dir : dir.substr(cut + 1));
}

#ifdef _WIN32
// La forme courte 8.3 du dossier (« C:\Users\JEAN-P~1 »), que GetTempPathW et d'autres rendent.
std::string short_form(const std::string& dir) {
  const std::filesystem::path path = core::path_from_utf8(dir);
  const DWORD size = GetShortPathNameW(path.c_str(), nullptr, 0);
  if (size == 0) return {};
  std::wstring out(size, L'\0');
  const DWORD written = GetShortPathNameW(path.c_str(), out.data(), size);
  if (written == 0 || written >= size) return {};
  out.resize(written);
  return core::to_utf8(std::filesystem::path(out));
}
#endif

bool is_separator(char c) {
  return c == '/' || c == '\\';
}

bool is_digit(char c) {
  return c >= '0' && c <= '9';
}

// Ce qui peut précéder un chemin : le début, un blanc, une ponctuation d'ouverture ou de liaison, ou un
// caractère hors ASCII (flèche, espace insécable : ses octets valent 0x80 et plus). Liste large exprès : un
// chemin non reconnu part en clair, avec le nom d'utilisateur, dans une issue publique (B19-05, B39-07).
bool opens_path(char before) {
  if (static_cast<unsigned char>(before) >= 0x80) return true;
  constexpr std::string_view kOpeners = " \t\n\"'([{=:,;<>|@`*";
  return kOpeners.find(before) != std::string_view::npos;
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

// La fin du chemin qui commence en text[at] : un nom de dossier peut porter des espaces (« Données SSD »),
// l'étendue court donc jusqu'à une fin de ligne, un guillemet, un séparateur du journal (« , », « : »,
// « · », « ; »), la parenthèse qui ferme celle d'où le chemin sort, ou un autre chemin.
std::size_t path_end(std::string_view text, std::size_t at) {
  const char opener = at == 0 ? '\0' : text[at - 1];
  std::size_t end = at;
  while (end < text.size()) {
    const char c = text[end];
    if (c == '\n' || c == '"') break;
    const std::string_view rest = text.substr(end);
    if (rest.starts_with(", ") || rest.starts_with(" : ") || rest.starts_with(" · ") ||
        rest.starts_with("; "))
      break;
    if ((opener == '(' && c == ')') || (opener == '\'' && c == '\'')) break;
    // « ep.mkv (24000/1001 fps) » : la parenthèse ouverte après un blanc commente le chemin, sauf si elle
    // ferme sur un séparateur (« Anime (2020)/ep.mkv », un nom de dossier) (C20-01).
    if (rest.starts_with(" (")) {
      const std::size_t close = rest.find(')');
      if (close == std::string_view::npos || close + 1 >= rest.size() || !is_separator(rest[close + 1]))
        break;
    }
    if (c == ' ' && end + 1 < text.size() && path_starts(text, end + 1)) break;
    ++end;
  }
  return end;
}

class RedactingSink final : public spdlog::sinks::sink {
 public:
  explicit RedactingSink(spdlog::sink_ptr inner) : inner_(std::move(inner)) {}

  void log(const spdlog::details::log_msg& msg) override {
    // Aussi propre que le rapport de « Signaler un problème » (B39-05) : l'utilisateur joint ajp.log quand
    // une boîte d'erreur le lui dit, sans passer par le bouton.
    const std::string text =
        reduce_log_paths(redact_log_text(std::string_view(msg.payload.data(), msg.payload.size())));
    spdlog::details::log_msg copy = msg;
    copy.payload = text;
    inner_->log(copy);
  }
  void flush() override { inner_->flush(); }
  void set_pattern(const std::string& pattern) override { inner_->set_pattern(pattern); }
  void set_formatter(std::unique_ptr<spdlog::formatter> formatter) override {
    inner_->set_formatter(std::move(formatter));
  }

 private:
  spdlog::sink_ptr inner_;
};

}  // namespace

void set_log_home(std::string_view path) {
  std::string dir(path);
  while (dir.size() > 1 && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
  std::vector<std::string> forms;
  std::vector<std::string> mounts;
  // « / » seul ou un dossier d'un caractère : tout chemin y serait remplacé.
  if (dir.size() > 3) {
    std::vector<std::string> bases{dir};
#ifdef _WIN32
    if (std::string brief = short_form(dir); !brief.empty() && brief != dir)
      bases.push_back(std::move(brief));
#endif
    for (const std::string& base : bases) {
      std::string slashes = base;
      std::ranges::replace(slashes, '\\', '/');
      std::string backslashes = base;
      std::ranges::replace(backslashes, '/', '\\');
      // Les barres inverses doublées d'un chemin échappé (« C:\\Users\\jean ») (B20-02).
      std::string doubled;
      for (const char c : backslashes) doubled.append(c == '\\' ? 2 : 1, c);
      forms.push_back(std::move(slashes));
      if (doubled != backslashes) forms.push_back(std::move(doubled));
      forms.push_back(std::move(backslashes));
    }
    // Les disques montés de l'utilisateur gardent son nom (A20-03).
    if (const std::string user = user_of(dir); user.size() > 1)
      for (const std::string_view root : {"/run/media/", "/media/"})
        mounts.push_back(std::string(root) + user);
  }
  const std::scoped_lock lock(home().mutex);
  home().forms = std::move(forms);
  home().mounts = std::move(mounts);
}

std::string redact_log_text(std::string_view text) {
  std::string out(text);
  const std::scoped_lock lock(home().mutex);
  for (const std::string& form : home().forms) replace_all(out, form, "~");
  for (const std::string& mount : home().mounts)
    replace_all(out, mount, mount.substr(0, mount.find_last_of('/') + 1) + "<user>");
  return out;
}

std::string reduce_log_paths(std::string_view text) {
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
    const std::size_t end = path_end(text, at);
    std::size_t last = std::string_view::npos;
    std::size_t separators = 0;
    for (std::size_t i = at; i < end; ++i) {
      // « 3/4 », « 16/9 » : une fraction après le nom n'est pas son dossier (C20-01).
      const bool fraction = i > at && i + 1 < end && is_digit(text[i - 1]) && is_digit(text[i + 1]);
      if (is_separator(text[i]) && !fraction) {
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
  return out;
}

spdlog::sink_ptr redacting_sink(spdlog::sink_ptr inner) {
  return std::make_shared<RedactingSink>(std::move(inner));
}

}  // namespace ajp
