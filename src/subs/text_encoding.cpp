#include "subs/text_encoding.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ajp::subs {
namespace {

void append(std::string& out, char32_t code) {
  if (code < 0x80) {
    out.push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code >> 6)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

bool valid_utf8(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<unsigned char>(text[i]);
    std::size_t extra = 0;
    if (lead < 0x80) {
      extra = 0;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
      extra = 1;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      extra = 2;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      extra = 3;
    } else {
      return false;
    }
    for (std::size_t k = 1; k <= extra; ++k) {
      if (i + k >= text.size() || (static_cast<unsigned char>(text[i + k]) & 0xC0) != 0x80) return false;
    }
    i += extra + 1;
  }
  return true;
}

// Windows-1252 de 0x80 à 0x9F ; le reste est Latin-1. Les cinq cases non définies donnent U+FFFD.
constexpr std::array<char32_t, 32> kCp1252High = {
    0x20AC, 0xFFFD, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
    0x2039, 0x0152, 0xFFFD, 0x017D, 0xFFFD, 0xFFFD, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
    0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0xFFFD, 0x017E, 0x0178};

std::string from_cp1252(std::string_view bytes) {
  std::string out;
  out.reserve(bytes.size() + bytes.size() / 8);
  for (const char c : bytes) {
    const auto byte = static_cast<unsigned char>(c);
    append(out, byte >= 0x80 && byte <= 0x9F ? kCp1252High[byte - 0x80] : static_cast<char32_t>(byte));
  }
  return out;
}

std::string from_utf16(std::string_view bytes, bool big_endian) {
  std::string out;
  out.reserve(bytes.size());
  const auto unit = [&](std::size_t at) {
    const auto a = static_cast<unsigned char>(bytes[at]);
    const auto b = static_cast<unsigned char>(bytes[at + 1]);
    return static_cast<char16_t>(big_endian ? (a << 8) | b : (b << 8) | a);
  };
  for (std::size_t i = 0; i + 1 < bytes.size(); i += 2) {
    char32_t code = unit(i);
    if (code >= 0xD800 && code <= 0xDBFF && i + 3 < bytes.size()) {
      const char32_t low = unit(i + 2);
      if (low >= 0xDC00 && low <= 0xDFFF) {
        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        i += 2;
      }
    }
    append(out, code);
  }
  return out;
}

}  // namespace

std::string to_utf8(std::string_view bytes) {
  if (bytes.size() >= 2 && static_cast<unsigned char>(bytes[0]) == 0xFF &&
      static_cast<unsigned char>(bytes[1]) == 0xFE) {
    return from_utf16(bytes.substr(2), false);
  }
  if (bytes.size() >= 2 && static_cast<unsigned char>(bytes[0]) == 0xFE &&
      static_cast<unsigned char>(bytes[1]) == 0xFF) {
    return from_utf16(bytes.substr(2), true);
  }
  if (bytes.size() >= 3 && bytes.substr(0, 3) == "\xEF\xBB\xBF") bytes.remove_prefix(3);
  if (valid_utf8(bytes)) return std::string(bytes);
  return from_cp1252(bytes);
}

}  // namespace ajp::subs
