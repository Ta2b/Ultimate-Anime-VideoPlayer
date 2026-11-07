#include "core/log_redact.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/sink.h>

namespace ajp {
namespace {

struct Home {
  std::mutex mutex;
  std::vector<std::string> forms;  // sous mutex : le dossier sous chacun des deux séparateurs
};

Home& home() {
  static Home state;
  return state;
}

void replace_all(std::string& text, std::string_view from, std::string_view to) {
  for (std::size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
    text.replace(at, from.size(), to);
  }
}

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
  // « / » seul ou un dossier d'un caractère : tout chemin y serait remplacé.
  if (dir.size() > 3) {
    std::string slashes = dir;
    std::ranges::replace(slashes, '\\', '/');
    std::string backslashes = dir;
    std::ranges::replace(backslashes, '/', '\\');
    forms = {slashes, backslashes};
  }
  const std::scoped_lock lock(home().mutex);
  home().forms = std::move(forms);
}

std::string redact_log_text(std::string_view text) {
  std::string out(text);
  const std::scoped_lock lock(home().mutex);
  for (const std::string& form : home().forms) replace_all(out, form, "~");
  return out;
}

spdlog::sink_ptr redacting_sink(spdlog::sink_ptr inner) {
  return std::make_shared<RedactingSink>(std::move(inner));
}

}  // namespace ajp
