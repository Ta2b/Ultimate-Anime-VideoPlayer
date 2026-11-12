#include "core/log_file.h"

#include <chrono>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <system_error>
#include <utility>

#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/base_sink.h>

namespace ajp {
namespace {

namespace fs = std::filesystem;

class PathFileSink final : public spdlog::sinks::base_sink<std::mutex> {
 public:
  PathFileSink(fs::path file, std::size_t max_bytes, std::size_t parts)
      : file_(std::move(file)), max_bytes_(max_bytes), parts_(parts) {}

  bool open() {
    out_.open(file_, std::ios::binary | std::ios::app);
    if (!out_) return false;
    std::error_code error;
    const auto size = fs::file_size(file_, error);
    written_ = error ? 0 : static_cast<std::size_t>(size);
    return true;
  }

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    // Une part pleine passe à la suivante ; jamais avant d'avoir écrit quelque chose (disque plein).
    if (written_ > 0 && written_ + formatted.size() > max_bytes_) rotate();
    // C20-02 : la part refusée à l'ouverture (un antivirus la tenait pendant la rotation) se rouvre, au plus
    // une fois par seconde, au lieu de perdre en silence toute la suite de la séance.
    if (!out_.is_open()) reopen();
    if (!out_.is_open()) return;
    out_.write(formatted.data(), static_cast<std::streamsize>(formatted.size()));
    written_ += formatted.size();
  }
  void flush_() override { out_.flush(); }

 private:
  void rotate() {
    out_.close();
    std::error_code error;
    for (std::size_t i = parts_; i > 0; --i) {
      const fs::path from = log_part(file_, i - 1);
      if (!fs::exists(from, error)) continue;
      const fs::path to = log_part(file_, i);
      fs::remove(to, error);
      fs::rename(from, to, error);
    }
    // Le renommage refusé (un antivirus tient l'ancienne part) : la part en cours recommence quand même,
    // le fichier ne grossit pas sans fin.
    out_.open(file_, std::ios::binary | std::ios::trunc);
    out_.clear();
    written_ = 0;
    tried_at_ = std::chrono::steady_clock::now();
  }

  void reopen() {
    const auto now = std::chrono::steady_clock::now();
    if (now - tried_at_ < std::chrono::seconds(1)) return;
    tried_at_ = now;
    out_.clear();
    static_cast<void>(open());
  }

  fs::path file_;
  std::size_t max_bytes_;
  std::size_t parts_;
  std::ofstream out_;
  std::size_t written_ = 0;
  std::chrono::steady_clock::time_point tried_at_{};
};

}  // namespace

fs::path log_part(const fs::path& file, std::size_t index) {
  if (index == 0) return file;
  fs::path part = file;
  part.replace_filename(fs::path(file.stem()) += fs::path(std::format(".{}", index)) += file.extension());
  return part;
}

void rotate_session_logs(const fs::path& file, int kept, std::size_t parts) {
  std::error_code error;
  for (std::size_t part = 0; part <= parts; ++part) {
    const fs::path base = log_part(file, part);
    for (int i = kept - 1; i >= 0; --i) {
      fs::path from = base;
      if (i > 0) from += std::format(".{}", i);
      fs::path to = base;
      to += std::format(".{}", i + 1);
      if (fs::exists(from, error)) {
        fs::rename(from, to, error);
      } else if (i == 0) {
        fs::remove(to, error);  // ajp.1.log.1 d'une séance plus ancienne que ajp.log.1
      }
    }
  }
}

spdlog::sink_ptr log_file_sink(const fs::path& file, std::size_t max_bytes, std::size_t parts) {
  auto sink = std::make_shared<PathFileSink>(file, max_bytes, parts);
  if (!sink->open()) return nullptr;
  return sink;
}

}  // namespace ajp
