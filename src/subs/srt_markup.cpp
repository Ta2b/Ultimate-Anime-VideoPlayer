// Balises HTML légères des sous-titres texte, traduites en balises ASS.
#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "subs/srt.h"

namespace ajp::subs {
namespace {

char lower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

std::string lowercase(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = lower(c);
  return out;
}

bool is_hex(char c) {
  c = lower(c);
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
}

char upper(char c) {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
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
  std::string rgb = lowercase(value);
  for (const NamedColor& named : kNamedColors) {
    if (rgb == named.name) rgb = named.rrggbb;
  }
  std::string_view hex = rgb;
  if (hex.starts_with('#')) hex.remove_prefix(1);
  if (hex.size() != 6) return std::nullopt;
  if (!std::ranges::all_of(hex, is_hex)) return std::nullopt;
  std::string out = "&H";
  for (const std::size_t pair : {4U, 2U, 0U}) {  // RRGGBB devient BBGGRR
    out += upper(hex[pair]);
    out += upper(hex[pair + 1]);
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
    Attribute attribute{.key = lowercase(s.substr(0, eq)), .value = {}};
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

FontTag parse_font(std::string_view attributes) {
  FontTag tag;
  for (const Attribute& attribute : parse_attributes(attributes)) {
    if (attribute.key == "color") tag.color = ass_color(attribute.value);
    if (attribute.key == "face" && !attribute.value.empty()) tag.face = attribute.value;
    if (attribute.key == "size" && !attribute.value.empty()) tag.size = attribute.value;
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
  const std::string name = lowercase(tag.substr(0, name_end));
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
    out += c;
  }
  return out;
}

}  // namespace ajp::subs
