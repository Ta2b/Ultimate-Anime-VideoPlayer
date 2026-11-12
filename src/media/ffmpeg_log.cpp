#include "media/ffmpeg_log.h"

#include <array>
#include <cstdarg>
#include <format>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/log.h>
}

#include "media/log.h"

namespace ajp::media {
namespace {

struct Folded {
  std::mutex mutex;
  ErrorFolder folder;
};

Folded& folded() {
  static Folded state;
  return state;
}

// La ligne sans ses nombres : « Invalid NAL unit size (123 > 45) » et « (124 > 50) » sont la même.
std::string without_numbers(const std::string& line) {
  std::string key;
  key.reserve(line.size());
  for (const char c : line) {
    if (c >= '0' && c <= '9') {
      if (key.empty() || key.back() != '#') key.push_back('#');
    } else {
      key.push_back(c);
    }
  }
  return key;
}

void on_ffmpeg_log(void* context, int level, const char* format, std::va_list args) {
  if (level > AV_LOG_ERROR) return;
  std::array<char, 1024> buffer{};
  // B30-09 : l'état que FFmpeg garde d'un appel à l'autre pour recoller une ligne écrite en plusieurs
  // morceaux ; un par thread de décodage, sinon deux threads se le disputent et chaque appel repart de 1.
  thread_local int prefix = 1;
  av_log_format_line2(context, level, format, args, buffer.data(), static_cast<int>(buffer.size()), &prefix);
  std::string line = buffer.data();
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
  if (line.empty()) return;
  static const auto log = make_logger();
  Folded& state = folded();
  std::vector<std::string> out;
  {
    const std::scoped_lock lock(state.mutex);
    out = state.folder.push(line, std::chrono::steady_clock::now());
  }
  for (const std::string& text : out) log->error("{}", text);
}

}  // namespace

std::vector<std::string> ErrorFolder::push(const std::string& line,
                                           std::chrono::steady_clock::time_point now) {
  std::vector<std::string> out;
  const std::string key = without_numbers(line);
  if (key == last_) {
    ++(last_shown_ ? repeats_ : omitted_);
    return out;
  }
  if (now - window_start_ >= std::chrono::seconds(1)) {
    window_start_ = now;
    window_lines_ = 0;
  }
  if (repeats_ > 0) out.push_back(std::format("ffmpeg : la ligne précédente, {} fois de plus", repeats_));
  repeats_ = 0;
  last_ = key;
  last_shown_ = window_lines_ < kMaxLinesPerSecond;
  if (!last_shown_) {
    ++omitted_;
    return out;
  }
  ++window_lines_;
  if (omitted_ > 0)
    out.push_back(std::format("ffmpeg : {} lignes d'erreur omises (plus de {} par seconde)", omitted_,
                              kMaxLinesPerSecond));
  omitted_ = 0;
  out.push_back("ffmpeg : " + line);
  return out;
}

void route_ffmpeg_log_to_journal() {
  av_log_set_level(AV_LOG_ERROR);
  av_log_set_callback(on_ffmpeg_log);
}

}  // namespace ajp::media
