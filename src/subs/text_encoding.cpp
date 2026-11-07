#include "subs/text_encoding.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#ifdef _WIN32
#include <windows.h>
#else
#include <iconv.h>
#endif

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

// UTF-8 conforme à la RFC 3629 (P3.74) : ni forme trop longue, ni moitié de paire UTF-16, ni code au-delà
// de U+10FFFF. Le deuxième octet borne ces cas selon l'octet de tête.
bool valid_utf8(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<unsigned char>(text[i]);
    std::size_t extra = 0;
    unsigned char lo = 0x80;
    unsigned char hi = 0xBF;
    if (lead < 0x80) {
      extra = 0;
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
      return false;
    }
    for (std::size_t k = 1; k <= extra; ++k) {
      if (i + k >= text.size()) return false;
      const auto next = static_cast<unsigned char>(text[i + k]);
      if (k == 1 ? next < lo || next > hi : (next & 0xC0) != 0x80) return false;
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

// Un encodage hérité à double octet ou cyrillique, décodé par le système (MultiByteToWideChar, iconv).
struct Legacy {
  unsigned code_page;  // Windows
  const char* name;    // iconv
};
constexpr Legacy kShiftJis{932, "CP932"};
constexpr Legacy kGbk{936, "GBK"};
constexpr Legacy kBig5{950, "BIG5"};
constexpr Legacy kCp1251{1251, "CP1251"};

std::optional<std::string> decode_legacy(std::string_view bytes, Legacy legacy) {
  if (bytes.empty()) return std::string{};
#ifdef _WIN32
  const int size = static_cast<int>(bytes.size());
  const int wide =
      MultiByteToWideChar(legacy.code_page, MB_ERR_INVALID_CHARS, bytes.data(), size, nullptr, 0);
  if (wide <= 0) return std::nullopt;
  std::wstring w(static_cast<std::size_t>(wide), L'\0');
  MultiByteToWideChar(legacy.code_page, MB_ERR_INVALID_CHARS, bytes.data(), size, w.data(), wide);
  const int narrow = WideCharToMultiByte(CP_UTF8, 0, w.data(), wide, nullptr, 0, nullptr, nullptr);
  std::string out(static_cast<std::size_t>(narrow), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), wide, out.data(), narrow, nullptr, nullptr);
  return out;
#else
  iconv_t cd = iconv_open("UTF-8", legacy.name);
  if (cd == reinterpret_cast<iconv_t>(-1)) return std::nullopt;
  std::string in(bytes);
  std::string out(bytes.size() * 3 + 4, '\0');
  char* src = in.data();
  char* dst = out.data();
  std::size_t left = in.size();
  std::size_t room = out.size();
  const std::size_t done = iconv(cd, &src, &left, &dst, &room);
  iconv_close(cd);
  if (done == static_cast<std::size_t>(-1) || left != 0) return std::nullopt;
  out.resize(out.size() - room);
  return out;
#endif
}

// L'encodage dit par le nom du fichier : « ep01.ja.srt », « ep01.chs.ass », « ep01.cp1251.srt ».
std::optional<Legacy> hinted(std::string_view hint) {
  std::string h;
  for (const char c : hint) h.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c);
  if (h == "ja" || h == "jp" || h == "jpn" || h == "sjis" || h == "shift-jis" || h == "cp932")
    return kShiftJis;
  if (h == "zh-hant" || h == "zh-tw" || h == "zh-hk" || h == "cht" || h == "tc" || h == "big5" ||
      h == "cp950")
    return kBig5;
  if (h == "zh" || h == "zh-hans" || h == "zh-cn" || h == "chi" || h == "zho" || h == "chs" || h == "sc" ||
      h == "gbk" || h == "gb2312" || h == "cp936")
    return kGbk;
  if (h == "ru" || h == "rus" || h == "uk" || h == "ukr" || h == "bg" || h == "bul" || h == "sr" ||
      h == "be" || h == "cp1251" || h == "windows-1251")
    return kCp1251;
  return std::nullopt;
}

// Sans indice, la forme des octets hauts : le kana de Shift-JIS (têtes 0x82 et 0x83), le cyrillique de
// CP1251 (mots de lettres hautes, minuscules de 0xE0 à 0xFF), le chinois en lignes entières de paires. Un
// texte d'Europe de l'Ouest (lettres hautes isolées) reste en Windows-1252.
std::optional<Legacy> guessed(std::string_view bytes) {
  std::size_t high = 0, paired = 0, lower_cyr = 0, kana = 0, big5_trail = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto b = static_cast<unsigned char>(bytes[i]);
    if (b < 0x80) continue;
    ++high;
    if (b >= 0xE0) ++lower_cyr;
    const bool prev_high = i > 0 && static_cast<unsigned char>(bytes[i - 1]) >= 0x80;
    const bool next_high = i + 1 < bytes.size() && static_cast<unsigned char>(bytes[i + 1]) >= 0x80;
    // Une tête sous 0xE0 suivie d'un ASCII de 0x40 à 0x7E : une paire Big5 ou Shift-JIS. Les accents
    // français (0xE0 à 0xFC) suivis d'une lettre n'en sont pas.
    const bool ascii_trail = b < 0xE0 && i + 1 < bytes.size() &&
                             static_cast<unsigned char>(bytes[i + 1]) >= 0x40 &&
                             static_cast<unsigned char>(bytes[i + 1]) <= 0x7E;  // Shift-JIS, Big5, GBK
    if (prev_high || next_high || ascii_trail) ++paired;
    if (i + 1 < bytes.size()) {
      const auto t = static_cast<unsigned char>(bytes[i + 1]);
      if ((b == 0x82 && t >= 0x9F && t <= 0xF1) || (b == 0x83 && t >= 0x40 && t <= 0x96)) ++kana;
      if (b >= 0xA1 && t >= 0x40 && t <= 0x7E) ++big5_trail;
    }
  }
  if (high < 4 || paired * 2 < high) return std::nullopt;  // lettres isolées : Europe de l'Ouest
  if (kana * 10 >= high && decode_legacy(bytes, kShiftJis)) return kShiftJis;
  if (lower_cyr * 10 >= high * 6 && decode_legacy(bytes, kCp1251)) return kCp1251;
  if (big5_trail > 0 && decode_legacy(bytes, kBig5)) return kBig5;
  if (decode_legacy(bytes, kGbk)) return kGbk;
  if (decode_legacy(bytes, kBig5)) return kBig5;
  return std::nullopt;
}

}  // namespace

std::string to_utf8(std::string_view bytes, std::string_view hint) {
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
  const std::optional<Legacy> legacy = hinted(hint) ? hinted(hint) : guessed(bytes);
  if (legacy) {
    if (auto text = decode_legacy(bytes, *legacy)) return *std::move(text);
  }
  return from_cp1252(bytes);
}

}  // namespace ajp::subs
