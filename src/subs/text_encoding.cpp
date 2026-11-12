#include "subs/text_encoding.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
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
    if (code >= 0xD800 && code <= 0xDFFF) {
      // Une demi-paire seule (fichier coupé, éditeur fautif) n'a pas d'UTF-8 : U+FFFD (A17-04).
      const char32_t low = i + 3 < bytes.size() ? unit(i + 2) : 0;
      if (code <= 0xDBFF && low >= 0xDC00 && low <= 0xDFFF) {
        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        i += 2;
      } else {
        code = 0xFFFD;
      }
    }
    append(out, code);
  }
  return out;
}

// UTF-16 sans BOM (A17-04), reconnu à ses zéros alternés sur les 512 premiers octets : un texte de
// sous-titres a toujours ses horodatages et ses retours à la ligne en ASCII, donc un octet nul sur deux.
// Rend l'ordre des octets, big endian si vrai.
std::optional<bool> utf16_without_bom(std::string_view bytes) {
  const std::size_t units = std::min<std::size_t>(bytes.size(), 512) / 2;
  std::size_t zero_even = 0, zero_odd = 0;
  for (std::size_t u = 0; u < units; ++u) {
    zero_even += bytes[2 * u] == '\0' ? 1U : 0U;
    zero_odd += bytes[2 * u + 1] == '\0' ? 1U : 0U;
  }
  if (units < 4) return std::nullopt;
  if (zero_odd * 10 >= units * 3 && zero_even * 20 < units) return false;
  if (zero_even * 10 >= units * 3 && zero_odd * 20 < units) return true;
  return std::nullopt;
}

// Un encodage hérité à double octet ou cyrillique, décodé par le système (MultiByteToWideChar, iconv).
struct Legacy {
  unsigned code_page;              // Windows
  const char* name;                // iconv
  const char* fallback = nullptr;  // iconv, quand le premier nom manque ou refuse le texte
  // Une page à un octet (C34-07) : une case non définie (CP1251 0x98) devient U+FFFD sous iconv, comme
  // MultiByteToWideChar la rend en un caractère de contrôle sous Windows, au lieu de refuser tout le texte.
  bool single_byte = false;
};
constexpr Legacy kShiftJis{932, "CP932"};
constexpr Legacy kGbk{936, "GBK"};
// CP950 d'abord : la table de Windows, avec les extensions ETEN, pour décoder comme lui (A34-11) ; BIG5 quand
// l'iconv du système ne connaît pas CP950.
constexpr Legacy kBig5{950, "CP950", "BIG5"};
constexpr Legacy kCp1251{1251, "CP1251", nullptr, true};
constexpr Legacy kEucKr{949, "CP949"};  // CP949 contient l'EUC-KR
constexpr Legacy kEucJp{20932, "EUC-JP"};
// B17-05 : les pages d'une seule langue, jamais devinées (sans indice le repli reste Windows-1252), prises
// quand le nom du fichier nomme la langue : « ep01.pl.srt », « ep01.tr.srt ».
constexpr Legacy kCp1250{1250, "CP1250", nullptr,
                         true};  // europe centrale : polonais, tchèque, hongrois, roumain…
constexpr Legacy kCp1253{1253, "CP1253", nullptr, true};  // grec
constexpr Legacy kCp1254{1254, "CP1254", nullptr, true};  // turc
constexpr Legacy kCp1255{1255, "CP1255", nullptr, true};  // hébreu
constexpr Legacy kCp1256{1256, "CP1256", nullptr, true};  // arabe, persan
constexpr Legacy kCp1257{1257, "CP1257", nullptr, true};  // baltes
constexpr Legacy kCp1258{1258, "CP1258", nullptr, true};  // vietnamien
constexpr Legacy kCp874{874, "CP874", nullptr, true};     // thaï

bool same(Legacy a, Legacy b) {
  return a.code_page == b.code_page;
}

#ifndef _WIN32
// Le texte décodé par l'iconv du système depuis l'encodage name, ou rien : nom inconnu, octets refusés.
// single_byte : une page à un octet, dont un octet sans caractère devient U+FFFD (trois octets UTF-8, pas
// plus qu'un caractère de la page) au lieu de faire refuser tout le texte (C34-07). Une page à deux octets
// reste stricte : son refus sert à deviner l'encodage.
std::optional<std::string> decode_iconv(std::string_view bytes, const char* name, bool single_byte) {
  iconv_t cd = iconv_open("UTF-8", name);
  // L'échec d'iconv_open : (iconv_t)-1.
  if (cd == reinterpret_cast<iconv_t>(-1)) return std::nullopt;  // NOLINT(performance-no-int-to-ptr)
  constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
  std::string in(bytes);
  std::string out(bytes.size() * 3 + 4, '\0');
  char* src = in.data();
  char* dst = out.data();
  std::size_t left = in.size();
  std::size_t room = out.size();
  bool refused = false;
  while (left > 0) {
    if (iconv(cd, &src, &left, &dst, &room) != static_cast<std::size_t>(-1)) break;
    if (!single_byte || errno != EILSEQ || room < kReplacement.size()) {
      refused = true;
      break;
    }
    dst = std::copy(kReplacement.begin(), kReplacement.end(), dst);
    room -= kReplacement.size();
    ++src;
    --left;
  }
  iconv_close(cd);
  if (refused || left != 0) return std::nullopt;
  out.resize(out.size() - room);
  return out;
}
#endif

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
  for (const char* name : {legacy.name, legacy.fallback}) {
    if (name == nullptr) continue;
    if (auto text = decode_iconv(bytes, name, legacy.single_byte)) return text;
  }
  return std::nullopt;
#endif
}

// Ce que dit un mot du nom de fichier : « ja », « chs », « zh-tw », « cp1251 ». Une langue donne son
// encodage habituel et, s'il y en a un autre courant, celui-là ; un nom d'encodage n'en donne qu'un.
struct Hint {
  Legacy first;
  std::optional<Legacy> second;
};

std::optional<Hint> hint_word(std::string_view word) {
  std::string h;
  for (const char c : word)
    h.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c == '_' ? '-' : c);
  const auto any = [&h](std::initializer_list<std::string_view> names) {
    return std::ranges::find(names, std::string_view(h)) != names.end();
  };
  if (any({"ja", "jp", "jpn"})) return Hint{kShiftJis, kEucJp};
  if (any({"sjis", "shift-jis", "cp932"})) return Hint{kShiftJis, {}};
  if (any({"euc-jp", "eucjp"})) return Hint{kEucJp, {}};
  if (any({"zh-hant", "zh-tw", "zh-hk", "cht", "tc"})) return Hint{kBig5, kGbk};
  if (any({"big5", "cp950"})) return Hint{kBig5, {}};
  if (any({"zh", "zh-hans", "zh-cn", "chi", "zho", "chs", "sc"})) return Hint{kGbk, kBig5};
  if (any({"gbk", "gb2312", "cp936"})) return Hint{kGbk, {}};
  if (any({"ko", "kor", "kr", "euc-kr", "cp949", "uhc"})) return Hint{kEucKr, {}};
  if (any({"ru", "rus", "uk", "ukr", "bg", "bul", "sr", "be", "cp1251", "windows-1251"}))
    return Hint{kCp1251, {}};
  if (any({"pl", "pol", "cs", "ces", "cze", "hu", "hun", "ro", "ron", "rum", "sk", "slk", "hr", "hrv", "sl",
           "slv", "cp1250", "windows-1250"}))
    return Hint{kCp1250, {}};
  if (any({"el", "ell", "gre", "cp1253", "windows-1253"})) return Hint{kCp1253, {}};
  if (any({"tr", "tur", "cp1254", "windows-1254"})) return Hint{kCp1254, {}};
  if (any({"he", "heb", "iw", "cp1255", "windows-1255"})) return Hint{kCp1255, {}};
  if (any({"ar", "ara", "fa", "fas", "per", "cp1256", "windows-1256"})) return Hint{kCp1256, {}};
  if (any({"et", "est", "lv", "lav", "lt", "lit", "cp1257", "windows-1257"})) return Hint{kCp1257, {}};
  if (any({"vi", "vie", "cp1258", "windows-1258"})) return Hint{kCp1258, {}};
  if (any({"th", "tha", "cp874", "windows-874"})) return Hint{kCp874, {}};
  return std::nullopt;
}

// Les suffixes du nom, du dernier au premier (« ep01.chs&jpn.forced ») : le premier mot reconnu.
std::optional<Hint> hinted(std::string_view hint) {
  while (!hint.empty()) {
    const std::size_t dot = hint.rfind('.');
    std::string_view suffix = dot == std::string_view::npos ? hint : hint.substr(dot + 1);
    hint = dot == std::string_view::npos ? std::string_view{} : hint.substr(0, dot);
    while (!suffix.empty()) {
      const std::size_t cut = suffix.find_first_of("&+,");
      if (auto found = hint_word(suffix.substr(0, cut))) return found;
      suffix = cut == std::string_view::npos ? std::string_view{} : suffix.substr(cut + 1);
    }
  }
  return std::nullopt;
}

// Les encodages EUC (A17-03), en paires d'octets de 0xA1 à 0xFE : l'EUC-JP à ses kana (lignes 0xA4 et
// 0xA5, nombreuses en japonais, absentes du chinois), l'EUC-KR à ses syllabes (têtes de 0xB0 à 0xC8
// seulement) séparées par des espaces, que le chinois n'a pas.
std::optional<Legacy> guessed_euc(std::string_view bytes) {
  std::size_t pairs = 0, kana = 0, hangul = 0, spaced = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto b = static_cast<unsigned char>(bytes[i]);
    if (b < 0x80) {
      if (b == ' ' && i > 0 && static_cast<unsigned char>(bytes[i - 1]) >= 0xA1) ++spaced;
      continue;
    }
    if (b < 0xA1 || b == 0xFF || i + 1 >= bytes.size()) return std::nullopt;  // pas de l'EUC
    const auto t = static_cast<unsigned char>(bytes[++i]);
    if (t < 0xA1 || t == 0xFF) return std::nullopt;
    ++pairs;
    if ((b == 0xA4 || b == 0xA5) && t <= 0xF6) ++kana;
    if (b >= 0xB0 && b <= 0xC8) ++hangul;
  }
  if (pairs < 4) return std::nullopt;
  if (kana * 10 >= pairs * 3 && decode_legacy(bytes, kEucJp)) return kEucJp;
  if (pairs >= 8 && hangul * 20 >= pairs * 19 && spaced * 10 >= pairs && decode_legacy(bytes, kEucKr))
    return kEucKr;
  return std::nullopt;
}

// Sans indice, la forme des octets hauts : le kana de Shift-JIS (têtes 0x82 et 0x83), le cyrillique de
// CP1251 (mots de lettres hautes, minuscules de 0xE0 à 0xFF), le chinois en lignes entières de paires. Un
// texte d'Europe de l'Ouest (lettres hautes isolées) reste en Windows-1252.
// Les octets hauts d'un texte, comptés selon leur forme.
struct HighBytes {
  std::size_t high = 0, paired = 0, lower_cyr = 0, kana = 0, big5_trail = 0;
};

HighBytes count_high(std::string_view bytes) {
  HighBytes n;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto b = static_cast<unsigned char>(bytes[i]);
    if (b < 0x80) continue;
    ++n.high;
    if (b >= 0xE0) ++n.lower_cyr;
    const bool prev_high = i > 0 && static_cast<unsigned char>(bytes[i - 1]) >= 0x80;
    const bool next_high = i + 1 < bytes.size() && static_cast<unsigned char>(bytes[i + 1]) >= 0x80;
    // Une tête sous 0xE0 suivie d'un ASCII de 0x40 à 0x7E : une paire Big5 ou Shift-JIS. Les accents
    // français (0xE0 à 0xFC) suivis d'une lettre n'en sont pas.
    const bool ascii_trail = b < 0xE0 && i + 1 < bytes.size() &&
                             static_cast<unsigned char>(bytes[i + 1]) >= 0x40 &&
                             static_cast<unsigned char>(bytes[i + 1]) <= 0x7E;  // Shift-JIS, Big5, GBK
    if (prev_high || next_high || ascii_trail) ++n.paired;
    if (i + 1 < bytes.size()) {
      const auto t = static_cast<unsigned char>(bytes[i + 1]);
      if ((b == 0x82 && t >= 0x9F && t <= 0xF1) || (b == 0x83 && t >= 0x40 && t <= 0x96)) ++n.kana;
      if (b >= 0xA1 && t >= 0x40 && t <= 0x7E) ++n.big5_trail;
    }
  }
  return n;
}

std::optional<Legacy> guessed(std::string_view bytes) {
  const HighBytes n = count_high(bytes);
  if (n.high < 4 || n.paired * 2 < n.high) return std::nullopt;  // lettres isolées : Europe de l'Ouest
  if (n.kana * 10 >= n.high && decode_legacy(bytes, kShiftJis)) return kShiftJis;
  if (auto euc = guessed_euc(bytes)) return euc;
  if (n.lower_cyr * 10 >= n.high * 6 && decode_legacy(bytes, kCp1251)) return kCp1251;
  if (n.big5_trail > 0 && decode_legacy(bytes, kBig5)) return kBig5;
  if (decode_legacy(bytes, kGbk)) return kGbk;
  if (decode_legacy(bytes, kBig5)) return kBig5;
  return std::nullopt;
}

// L'indice choisit entre ses encodages celui que la forme des octets suggère ; un indice qui ne décode pas
// laisse la forme des octets décider (A17-03), pas le Windows-1252.
std::optional<std::string> decode_guessed(std::string_view bytes, std::string_view hint) {
  const std::optional<Legacy> guess = guessed(bytes);
  if (const std::optional<Hint> h = hinted(hint)) {
    const bool agrees = guess && (same(*guess, h->first) || (h->second && same(*guess, *h->second)));
    if (!agrees) {
      if (auto text = decode_legacy(bytes, h->first)) return text;
      if (h->second) {
        if (auto text = decode_legacy(bytes, *h->second)) return text;
      }
    }
  }
  if (guess) return decode_legacy(bytes, *guess);
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
  if (const std::optional<bool> big_endian = utf16_without_bom(bytes)) return from_utf16(bytes, *big_endian);
  if (bytes.size() >= 3 && bytes.substr(0, 3) == "\xEF\xBB\xBF") bytes.remove_prefix(3);
  if (valid_utf8(bytes)) return std::string(bytes);
  if (auto text = decode_guessed(bytes, hint)) return *std::move(text);
  return from_cp1252(bytes);
}

}  // namespace ajp::subs
