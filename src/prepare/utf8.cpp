#include "prepare/utf8.h"

#include <cstddef>
#include <cstdint>

namespace ajp::prepare {

std::string utf8_path(const std::filesystem::path& path) {
  const std::u8string text = path.u8string();
  return {text.begin(), text.end()};
}

std::filesystem::path path_from_utf8(std::string_view text) {
  return {std::u8string(text.begin(), text.end())};
}

bool is_utf8(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<std::uint8_t>(text[i]);
    std::size_t length = 1;
    std::uint32_t point = lead;
    if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4;
      point = lead & 0x07U;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3;
      point = lead & 0x0FU;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2;
      point = lead & 0x1FU;
    } else if (lead >= 0x80) {
      return false;  // octet de suite isolé, C0, C1 ou F5 à FF
    }
    if (text.size() - i < length) return false;
    for (std::size_t k = 1; k < length; ++k) {
      const auto next = static_cast<std::uint8_t>(text[i + k]);
      if ((next & 0xC0U) != 0x80U) return false;
      point = (point << 6U) | (next & 0x3FU);
    }
    const bool overlong = (length == 3 && point < 0x800) || (length == 4 && point < 0x10000);
    const bool surrogate = point >= 0xD800 && point <= 0xDFFF;
    if (overlong || surrogate || point > 0x10FFFF) return false;
    i += length;
  }
  return true;
}

}  // namespace ajp::prepare
