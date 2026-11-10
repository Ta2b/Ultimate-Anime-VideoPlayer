#include "media/ffmpeg_log.h"

#include <array>
#include <cstdarg>
#include <mutex>
#include <string>

extern "C" {
#include <libavutil/log.h>
}

#include "media/log.h"

namespace ajp::media {
namespace {

struct Repeats {
  std::mutex mutex;
  std::string last;
  int count = 0;  // fois que last s'est répétée après la première
};

Repeats& repeats() {
  static Repeats state;
  return state;
}

void on_ffmpeg_log(void* context, int level, const char* format, std::va_list args) {
  if (level > AV_LOG_ERROR) return;
  std::array<char, 1024> buffer{};
  int prefix = 1;
  av_log_format_line2(context, level, format, args, buffer.data(), static_cast<int>(buffer.size()), &prefix);
  std::string line = buffer.data();
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
  if (line.empty()) return;
  static const auto log = make_logger();
  Repeats& state = repeats();
  const std::scoped_lock lock(state.mutex);
  if (line == state.last) {
    ++state.count;
    return;
  }
  if (state.count > 0) log->error("ffmpeg : la ligne précédente, {} fois de plus", state.count);
  state.last = line;
  state.count = 0;
  log->error("ffmpeg : {}", line);
}

}  // namespace

void route_ffmpeg_log_to_journal() {
  av_log_set_level(AV_LOG_ERROR);
  av_log_set_callback(on_ffmpeg_log);
}

}  // namespace ajp::media
