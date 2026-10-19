#include "export/av_util.h"

#include <array>
#include <format>
#include <utility>

extern "C" {
#include <libavutil/error.h>
}

namespace ajp::exporter {

std::string av_error_text(int error) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  if (av_strerror(error, text.data(), text.size()) < 0) return std::format("erreur FFmpeg {}", error);
  return text.data();
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::unexpected<Refusal> refuse_av(RefusalCode code, std::string_view where, int error) {
  return refuse(code, std::format("{} : {}", where, av_error_text(error)));
}

std::expected<InputPtr, Refusal> open_input(const std::string& path) {
  AVFormatContext* raw = nullptr;
  if (const int error = avformat_open_input(&raw, path.c_str(), nullptr, nullptr); error < 0) {
    return refuse_av(RefusalCode::MediaUnreadable, std::format("avformat_open_input {}", path), error);
  }
  InputPtr input(raw);
  if (const int error = avformat_find_stream_info(input.get(), nullptr); error < 0) {
    return refuse_av(RefusalCode::MediaUnreadable, std::format("avformat_find_stream_info {}", path), error);
  }
  return input;
}

}  // namespace ajp::exporter
