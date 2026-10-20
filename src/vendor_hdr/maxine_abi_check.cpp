#include "vendor_hdr/maxine_abi_check.h"

#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#include "vendor_hdr/maxine_abi.h"

namespace ajp::vendor_hdr {
namespace {

constexpr int kMaxAliasDepth = 4;
constexpr std::uintmax_t kMaxHeaderBytes = std::uintmax_t{4} * 1024 * 1024;

std::string_view trim(std::string_view text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

bool is_identifier_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

bool is_identifier(std::string_view text) {
  if (text.empty() || (text.front() >= '0' && text.front() <= '9')) return false;
  for (const char c : text) {
    if (!is_identifier_char(c)) return false;
  }
  return true;
}

// Les chaînes de ces en-têtes ne contiennent ni « // » ni « /* ». Une ligne finie par « \ » se
// poursuit sur la suivante, comme pour le préprocesseur : nvVFXTrueHDR.h:21-22 définit
// NVVFXTRUEHDR_DEBANDING_OFF ainsi.
std::string strip_comments(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    if (text.substr(i, 2) == "\\\n" || text.substr(i, 3) == "\\\r\n") {
      i += text[i + 1] == '\n' ? std::size_t{2} : std::size_t{3};
    } else if (text.substr(i, 2) == "//") {
      i = text.find('\n', i);
      if (i == std::string_view::npos) break;
    } else if (text.substr(i, 2) == "/*") {
      const std::size_t end = text.find("*/", i + 2);
      if (end == std::string_view::npos) break;
      out += ' ';
      i = end + 2;
    } else {
      out += text[i++];
    }
  }
  return out;
}

// « 13 », « -2 », « 0x0D », « (1) ».
std::optional<long long> parse_integer(std::string_view text) {
  text = trim(text);
  while (text.size() >= 2 && text.front() == '(' && text.back() == ')')
    text = trim(text.substr(1, text.size() - 2));
  const bool negative = text.starts_with('-');
  if (negative) text = trim(text.substr(1));
  int base = 10;
  if (text.starts_with("0x") || text.starts_with("0X")) {
    base = 16;
    text.remove_prefix(2);
  }
  long long value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return negative ? -value : value;
}

std::optional<std::string> find_in(std::span<const std::string> headers, std::string_view name, int depth);

// Un alias (#define A B) est suivi jusqu'à kMaxAliasDepth niveaux.
std::optional<std::string> resolve(std::span<const std::string> headers, std::string_view value, int depth) {
  if (!is_identifier(value)) return std::string(value);
  if (depth == 0) return std::nullopt;
  return find_in(headers, value, depth - 1);
}

std::optional<std::string> define_value(std::string_view text, std::string_view name) {
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = trim(text.substr(0, end));
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    if (!line.starts_with('#')) continue;
    line = trim(line.substr(1));
    if (!line.starts_with("define") || line.size() == 6 || is_identifier_char(line[6])) continue;
    line = trim(line.substr(6));
    const std::size_t space = line.find_first_of(" \t");
    if (space == std::string_view::npos || line.substr(0, space) != name) continue;
    if (const std::string_view value = trim(line.substr(space)); !value.empty()) return std::string(value);
  }
  return std::nullopt;
}

// Valeur de name dans le corps d'un enum qui le contient : explicite, ou précédente plus un.
std::optional<std::string> enumerator_value(std::span<const std::string> headers, std::string_view text,
                                            std::string_view name, int depth) {
  for (std::size_t at = text.find(name); at != std::string_view::npos; at = text.find(name, at + 1)) {
    const bool word = (at == 0 || !is_identifier_char(text[at - 1])) &&
                      (at + name.size() == text.size() || !is_identifier_char(text[at + name.size()]));
    const std::size_t open = text.rfind('{', at);
    const std::size_t closed = text.rfind('}', at);
    if (!word || open == std::string_view::npos || (closed != std::string_view::npos && closed > open))
      continue;
    const std::size_t statement = text.find_last_of(";}", open);
    const std::size_t head = statement == std::string_view::npos ? 0 : statement + 1;
    if (text.substr(head, open - head).find("enum") == std::string_view::npos) continue;
    std::string_view body = text.substr(open + 1, text.find('}', open) - open - 1);
    std::optional<long long> next = 0;
    while (!body.empty()) {
      const std::size_t comma = body.find(',');
      const std::string_view entry = trim(body.substr(0, comma));
      body = comma == std::string_view::npos ? std::string_view{} : body.substr(comma + 1);
      if (entry.empty()) continue;
      const std::size_t equals = entry.find('=');
      std::optional<long long> number = next;
      if (equals != std::string_view::npos) {
        const std::optional<std::string> value = resolve(headers, trim(entry.substr(equals + 1)), depth);
        number.reset();  // pas d'optional temporaire : GCC -O3 y voit une lecture non initialisée
        if (value) number = parse_integer(*value);
      }
      if (trim(entry.substr(0, equals)) == name) {
        if (!number) return std::nullopt;
        return std::to_string(*number);
      }
      next = number ? std::optional<long long>(*number + 1) : std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<std::string> find_in(std::span<const std::string> headers, std::string_view name, int depth) {
  for (const std::string& text : headers) {
    if (auto value = define_value(text, name)) return resolve(headers, *value, depth);
  }
  for (const std::string& text : headers) {
    if (auto value = enumerator_value(headers, text, name, depth)) return value;
  }
  return std::nullopt;
}

std::vector<std::string> stripped(std::span<const std::string> headers) {
  std::vector<std::string> out;
  out.reserve(headers.size());
  for (const std::string& text : headers) out.push_back(strip_comments(text));
  return out;
}

Done compare(const maxine::AbiConstant& constant, const std::optional<std::string>& found) {
  if (!found) {
    return refuse(RefusalCode::SdkRejected,
                  std::format("{} absent des en-têtes du SDK : SDK d'une autre version ?", constant.name));
  }
  const bool same = std::visit(
      [&](const auto& expected) {
        if constexpr (std::is_same_v<std::decay_t<decltype(expected)>, long long>) {
          return parse_integer(*found) == expected;
        } else {
          return found->size() >= 2 && found->front() == '"' && found->back() == '"' &&
                 std::string_view(*found).substr(1, found->size() - 2) == expected;
        }
      },
      constant.value);
  if (same) return {};
  const std::string expected = std::visit(
      [](const auto& value) {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, long long>) {
          return std::to_string(value);
        } else {
          return std::format("\"{}\"", value);
        }
      },
      constant.value);
  return refuse(RefusalCode::SdkRejected,
                std::format("{} vaut {} dans les en-têtes du SDK, ce lecteur attend {} (maxine_abi.h)",
                            constant.name, *found, expected));
}

void read_headers(const std::filesystem::path& directory, std::vector<std::string>& headers) {
  std::error_code error;
  for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end;
       it.increment(error)) {
    if (it->path().extension() != ".h") continue;
    std::error_code size_error;
    const std::uintmax_t size = std::filesystem::file_size(it->path(), size_error);
    if (size_error || size > kMaxHeaderBytes) continue;
    std::ifstream in(it->path(), std::ios::binary);
    headers.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
}

}  // namespace

std::optional<std::string> find_definition(std::span<const std::string> headers, std::string_view name) {
  const std::vector<std::string> texts = stripped(headers);
  return find_in(texts, name, kMaxAliasDepth);
}

Done check_maxine_abi(std::span<const std::string> headers) {
  const std::vector<std::string> texts = stripped(headers);
  for (const maxine::AbiConstant& constant : maxine::kAbi) {
    if (Done same = compare(constant, find_in(texts, constant.name, kMaxAliasDepth)); !same) return same;
  }
  return {};
}

Done check_maxine_abi(const std::filesystem::path& root) {
  std::vector<std::string> headers;
  read_headers(root / "include", headers);
  read_headers(root / "features" / "nvvfxtruehdr" / "include", headers);
  if (headers.empty()) {
    return refuse(RefusalCode::SdkRejected, "en-têtes du SDK absents : include/ ne contient aucun .h");
  }
  return check_maxine_abi(headers);
}

}  // namespace ajp::vendor_hdr
