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

// Remplace form par to là où il finit à une frontière de composant (séparateur, fin, espace, ponctuation).
void replace_all(std::string& text, std::string_view form, std::string_view to) {
  std::size_t at = 0;
  while (at + form.size() <= text.size()) {
    const std::size_t after = at + form.size();
    if (same_at(text, at, form) && (after == text.size() || !name_char(text[after]))) {
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

class RedactingSink final : public spdlog::sinks::sink {
 public:
  explicit RedactingSink(spdlog::sink_ptr inner) : inner_(std::move(inner)) {}

  void log(const spdlog::details::log_msg& msg) override {
    const std::string text = redact_log_text(std::string_view(msg.payload.data(), msg.payload.size()));
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
      forms.push_back(std::move(slashes));
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

spdlog::sink_ptr redacting_sink(spdlog::sink_ptr inner) {
  return std::make_shared<RedactingSink>(std::move(inner));
}

}  // namespace ajp
