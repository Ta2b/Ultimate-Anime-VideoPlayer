// Balises HTML légères des sous-titres texte, traduites en balises ASS.
#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/text.h"
#include "subs/srt.h"

namespace ajp::subs {
namespace {

bool is_hex(char c) {
  c = core::ascii_lower(c);
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
}

struct NamedColor {
  std::string_view name;
  std::string_view rrggbb;
};

constexpr std::array kNamedColors{
    NamedColor{"white", "FFFFFF"},   NamedColor{"black", "000000"},   NamedColor{"red", "FF0000"},
    NamedColor{"lime", "00FF00"},    NamedColor{"green", "008000"},   NamedColor{"blue", "0000FF"},
    NamedColor{"yellow", "FFFF00"},  NamedColor{"cyan", "00FFFF"},    NamedColor{"aqua", "00FFFF"},
    NamedColor{"magenta", "FF00FF"}, NamedColor{"fuchsia", "FF00FF"}, NamedColor{"gray", "808080"},
    NamedColor{"grey", "808080"},    NamedColor{"silver", "C0C0C0"},  NamedColor{"orange", "FFA500"},
    NamedColor{"purple", "800080"},
};

// « #RRGGBB », « RRGGBB » ou un nom HTML de base, en couleur ASS « &HBBGGRR& ».
std::optional<std::string> ass_color(std::string_view value) {
  std::string rgb = core::ascii_lower(value);
  for (const NamedColor& named : kNamedColors) {
    if (rgb == named.name) rgb = named.rrggbb;
  }
  std::string_view hex = rgb;
  if (hex.starts_with('#')) hex.remove_prefix(1);
  if (hex.size() != 6) return std::nullopt;
  if (!std::ranges::all_of(hex, is_hex)) return std::nullopt;
  std::string out = "&H";
  for (const std::size_t pair : {4U, 2U, 0U}) {  // RRGGBB devient BBGGRR
    out += core::ascii_upper(hex[pair]);
    out += core::ascii_upper(hex[pair + 1]);
  }
  out += '&';
  return out;
}

// Attributs posés par une balise <font>, déjà au format ASS.
struct FontTag {
  std::optional<std::string> color;  // &HBBGGRR&
  std::optional<std::string> face;
  std::optional<std::string> size;
};

struct Attribute {
  std::string key;
  std::string value;
};

// key=value, key="value" ou key='value', séparés par des espaces.
std::vector<Attribute> parse_attributes(std::string_view s) {
  std::vector<Attribute> attributes;
  while (true) {
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    const std::size_t eq = s.find('=');
    if (s.empty() || eq == std::string_view::npos) return attributes;
    Attribute attribute{.key = core::ascii_lower(s.substr(0, eq)), .value = {}};
    while (!attribute.key.empty() && attribute.key.back() == ' ') attribute.key.pop_back();
    s.remove_prefix(eq + 1);
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    const char quote = (!s.empty() && (s.front() == '"' || s.front() == '\'')) ? s.front() : ' ';
    if (quote != ' ') s.remove_prefix(1);
    const std::size_t end = s.find(quote);
    attribute.value = std::string(s.substr(0, end));
    s.remove_prefix(end == std::string_view::npos ? s.size() : end + 1);
    attributes.push_back(std::move(attribute));
  }
}

// B17-04 : un nom de police qui sort du bloc de balises ASS (« } », « \ ») ou qui porte un caractère de
// contrôle injecterait des balises de l'auteur du fichier : il est refusé en entier.
bool safe_face(std::string_view face) {
  return std::ranges::none_of(
      face, [](char c) { return c == '{' || c == '}' || c == '\\' || static_cast<unsigned char>(c) < 0x20; });
}

// B17-03, B17-04 : la taille est un entier. De 1 à 7 sans unité, c'est l'échelle HTML (3 : la taille du
// style, 16) ; au-delà, des points ASS, bornés à 200. Autre chose (« +1 », « big », vide, 0) : ignoré.
std::optional<std::string> font_size(std::string_view value) {
  constexpr std::array<int, 7> kHtmlSizes{10, 13, 16, 18,
                                          24, 32, 48};  // 0,63 ; 0,82 ; 1 ; 1,12 ; 1,5 ; 2 ; 3
  constexpr int kMaxSize = 200;
  const bool unit =
      value.ends_with("px") || value.ends_with("pt");  // une unité : des points ASS, pas l'échelle
  if (unit) value.remove_suffix(2);
  if (value.empty() || value.size() > 4 ||
      !std::ranges::all_of(value, [](char c) { return c >= '0' && c <= '9'; }))
    return std::nullopt;
  int size = 0;
  for (const char c : value) size = size * 10 + (c - '0');
  if (size < 1) return std::nullopt;
  if (size <= 7 && !unit) size = kHtmlSizes[static_cast<std::size_t>(size - 1)];
  return std::to_string(std::min(size, kMaxSize));
}

FontTag parse_font(std::string_view attributes) {
  FontTag tag;
  for (const Attribute& attribute : parse_attributes(attributes)) {
    if (attribute.key == "color") tag.color = ass_color(attribute.value);
    if (attribute.key == "face" && !attribute.value.empty() && safe_face(attribute.value))
      tag.face = attribute.value;
    if (attribute.key == "size") tag.size = font_size(attribute.value);
  }
  return tag;
}

void emit_font(const FontTag& tag, std::string& out) {
  if (!tag.color && !tag.face && !tag.size) return;
  out += '{';
  if (tag.color) out += "\\c" + *tag.color;
  if (tag.face) out += "\\fn" + *tag.face;
  if (tag.size) out += "\\fs" + *tag.size;
  out += '}';
}

// Valeur active d'un attribut après fermeture : celle de la balise englobante la
// plus proche, sinon vide, ce qui rend à libass la valeur du style.
std::string enclosing(const std::vector<FontTag>& stack, std::optional<std::string> FontTag::* field) {
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    if (const std::optional<std::string>& value = (*it).*field) return *value;
  }
  return {};
}

void close_font(std::vector<FontTag>& stack, std::string& out) {
  if (stack.empty()) return;
  const FontTag closed = std::move(stack.back());
  stack.pop_back();
  FontTag restore;
  if (closed.color) restore.color = enclosing(stack, &FontTag::color);
  if (closed.face) restore.face = enclosing(stack, &FontTag::face);
  if (closed.size) restore.size = enclosing(stack, &FontTag::size);
  emit_font(restore, out);
}

// Traduit une balise (sans < ni >) ; false si elle est inconnue.
bool convert_tag(std::string_view tag, std::vector<FontTag>& fonts, std::string& out) {
  const bool closing = tag.starts_with('/');
  if (closing) tag.remove_prefix(1);
  const std::size_t name_end = tag.find(' ');
  const std::string name = core::ascii_lower(tag.substr(0, name_end));
  const std::string_view rest = name_end == std::string_view::npos ? "" : tag.substr(name_end);

  if ((name == "i" || name == "b" || name == "u" || name == "s") &&
      rest.find_first_not_of(' ') == std::string_view::npos) {
    out += "{\\" + name + (closing ? "0}" : "1}");
    return true;
  }
  if (name == "font") {
    if (closing) {
      close_font(fonts, out);
    } else {
      fonts.push_back(parse_font(rest));
      emit_font(fonts.back(), out);
    }
    return true;
  }
  return false;
}

}  // namespace

std::string srt_text_to_ass(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  std::vector<FontTag> fonts;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\r') continue;
    if (c == '\n') {
      out += "\\N";
      continue;
    }
    if (c == '<') {
      const std::size_t close = text.find('>', i);
      if (close != std::string_view::npos && convert_tag(text.substr(i + 1, close - i - 1), fonts, out)) {
        i = close;
        continue;
      }
    }
    // Un bloc {…} sans « \ » est un commentaire pour libass : le texte disparaîtrait (C17-02). Seul un bloc
    // de balises ASS (« {\an8} ») passe tel quel, le reste s'écrit avec une accolade échappée.
    if (c == '{' && (i + 1 >= text.size() || text[i + 1] != '\\')) {
      out += "\\{";
      continue;
    }
    out += c;
  }
  return out;
}

}  // namespace ajp::subs
