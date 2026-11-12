#include "media/tag_text.h"

#include <algorithm>
#include <cstdint>

namespace ajp::media {
namespace {

// Longueur de la séquence UTF-8 conforme (RFC 3629) qui commence à text[i], 0 si elle est invalide.
std::size_t sequence_length(std::string_view text, std::size_t i) {
  const auto lead = static_cast<unsigned char>(text[i]);
  std::size_t extra = 0;
  unsigned char lo = 0x80;
  unsigned char hi = 0xBF;
  if (lead < 0x80) {
    return 1;
  } else if (lead >= 0xC2 && lead <= 0xDF) {
    extra = 1;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    extra = 2;
    if (lead == 0xE0) lo = 0xA0;
    if (lead == 0xED) hi = 0x9F;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    extra = 3;
    if (lead == 0xF0) lo = 0x90;
    if (lead == 0xF4) hi = 0x8F;
  } else {
    return 0;
  }
  for (std::size_t k = 1; k <= extra; ++k) {
    if (i + k >= text.size()) return 0;
    const auto next = static_cast<unsigned char>(text[i + k]);
    if (k == 1 ? next < lo || next > hi : (next & 0xC0) != 0x80) return 0;
  }
  return extra + 1;
}

char32_t decode(std::string_view sequence) {
  const auto b0 = static_cast<unsigned char>(sequence[0]);
  switch (sequence.size()) {
    case 1:
      return b0;
    case 2:
      return (char32_t{b0 & 0x1FU} << 6) | (static_cast<unsigned char>(sequence[1]) & 0x3FU);
    case 3:
      return (char32_t{b0 & 0x0FU} << 12) | (char32_t{static_cast<unsigned char>(sequence[1]) & 0x3FU} << 6) |
             (static_cast<unsigned char>(sequence[2]) & 0x3FU);
    default:
      return (char32_t{b0 & 0x07U} << 18) | (char32_t{static_cast<unsigned char>(sequence[1]) & 0x3FU} << 12) |
             (char32_t{static_cast<unsigned char>(sequence[2]) & 0x3FU} << 6) |
             (static_cast<unsigned char>(sequence[3]) & 0x3FU);
  }
}

bool is_bidi_control(char32_t c) {
  return c == 0x200E || c == 0x200F || c == 0x061C || (c >= 0x202A && c <= 0x202E) || (c >= 0x2066 && c <= 0x2069);
}

}  // namespace

std::string clean_tag_text(std::string_view raw) {
  std::string out;
  out.reserve(std::min(raw.size(), kMaxTagBytes) + 3);
  for (std::size_t i = 0; i < raw.size();) {
    const std::size_t length = sequence_length(raw, i);
    std::string_view piece = length == 0 ? std::string_view("\xEF\xBF\xBD") : raw.substr(i, length);
    i += length == 0 ? 1 : length;
    const char32_t code = length == 0 ? 0xFFFD : decode(piece);
    if (is_bidi_control(code)) continue;
    if (code < 0x20 || code == 0x7F) piece = " ";
    if (out.size() + piece.size() > kMaxTagBytes) break;
    out.append(piece);
  }
  return out;
}

}  // namespace ajp::media
