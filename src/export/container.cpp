#include "export/container.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <string>

namespace ajp::exporter {

std::optional<Container> container_of(const std::filesystem::path& output) {
  std::string extension = output.extension().string();
  std::ranges::transform(extension, extension.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (extension == ".mkv") return Container::Matroska;
  if (extension == ".mp4") return Container::Mp4;
  if (extension == ".mov") return Container::Mov;
  return std::nullopt;
}

std::string_view label(Container container) {
  switch (container) {
    case Container::Mp4:
      return "MP4";
    case Container::Mov:
      return "MOV";
    default:
      return "MKV";
  }
}

std::string_view format_name(Container container) {
  switch (container) {
    case Container::Mp4:
      return "mp4";
    case Container::Mov:
      return "mov";
    default:
      return "matroska";
  }
}

bool carries(Container container, Codec codec) {
  return codec != Codec::Lossless || container != Container::Mov;
}

bool carries_attachments(Container container) {
  return container == Container::Matroska;
}

int video_timescale(int fps_num, int fps_den) {
  if (fps_num <= 0 || fps_den <= 0) return 0;
  // Un instant juste à la microseconde près reste à moins d'un demi-pas tant qu'un pas dépasse 2 µs.
  constexpr int kMin = 10'000;
  constexpr int kMax = 500'000;
  int timescale = fps_num / std::gcd(fps_num, fps_den);
  while (timescale < kMin) timescale *= 2;
  return timescale <= kMax ? timescale : 0;
}

}  // namespace ajp::exporter
