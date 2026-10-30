#include "render/cube_lut.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <format>
#include <limits>

namespace ajp::render {
namespace {

Refusal refused(std::string detail) {
  return Refusal{RefusalCode::FileUnreadable, std::move(detail)};
}

std::string_view trimmed(std::string_view line) {
  while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.remove_prefix(1);
  while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
    line.remove_suffix(1);
  return line;
}

// Un flottant décimal (signe, chiffres, point, exposant), sans dépendre de la locale ni de
// std::from_chars à virgule flottante, que libc++ de macOS ne fournit pas.
bool read_float(std::string_view& text, float& out) {
  while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
  std::size_t at = 0;
  bool negative = false;
  if (at < text.size() && (text[at] == '+' || text[at] == '-')) negative = text[at++] == '-';
  double mantissa = 0.0;
  int digits = 0;
  while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
    mantissa = mantissa * 10.0 + (text[at++] - '0');
    ++digits;
  }
  int exponent = 0;
  if (at < text.size() && text[at] == '.') {
    ++at;
    while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
      mantissa = mantissa * 10.0 + (text[at++] - '0');
      --exponent;
      ++digits;
    }
  }
  if (digits == 0) return false;
  if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
    std::size_t probe = at + 1;
    bool exponent_negative = false;
    if (probe < text.size() && (text[probe] == '+' || text[probe] == '-'))
      exponent_negative = text[probe++] == '-';
    int value = 0;
    int exponent_digits = 0;
    while (probe < text.size() && text[probe] >= '0' && text[probe] <= '9') {
      value = std::min(value * 10 + (text[probe++] - '0'), 400);
      ++exponent_digits;
    }
    if (exponent_digits == 0) return false;
    exponent += exponent_negative ? -value : value;
    at = probe;
  }
  if (at < text.size() && text[at] != ' ' && text[at] != '\t') return false;
  const double result = (negative ? -mantissa : mantissa) * std::pow(10.0, exponent);
  out = static_cast<float>(result);
  if (!std::isfinite(out)) return false;
  text.remove_prefix(at);
  return true;
}

bool read_triple(std::string_view text, float (&out)[3]) {
  for (float& value : out) {
    if (!read_float(text, value)) return false;
  }
  return trimmed(text).empty();
}

}  // namespace

std::expected<CubeLut, Refusal> parse_cube(std::string_view text) {
  if (text.size() > kLutMaxBytes) return std::unexpected(refused("LUT trop grosse"));
  CubeLut lut;
  float domain_min[3] = {0.0F, 0.0F, 0.0F};
  float domain_max[3] = {1.0F, 1.0F, 1.0F};
  std::size_t expected = 0;
  bool in_data = false;
  std::size_t line_number = 0;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = trimmed(text.substr(0, end));
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    ++line_number;
    if (line.empty() || line.front() == '#') continue;
    if (!in_data) {
      if (line.starts_with("TITLE")) {
        std::string_view title = trimmed(line.substr(5));
        if (title.size() >= 2 && title.front() == '"' && title.back() == '"')
          title = title.substr(1, title.size() - 2);
        lut.title = std::string(title);
        continue;
      }
      if (line.starts_with("LUT_1D_SIZE"))
        return std::unexpected(refused("LUT 1D : seules les LUT 3D sont lues"));
      if (line.starts_with("LUT_3D_SIZE")) {
        int size = 0;
        const std::string_view value = trimmed(line.substr(11));
        const auto [ptr, error] = std::from_chars(value.data(), value.data() + value.size(), size);
        if (error != std::errc{} || ptr != value.data() + value.size())
          return std::unexpected(refused(std::format("ligne {} : LUT_3D_SIZE illisible", line_number)));
        if (size < kLutMinSize || size > kLutMaxSize)
          return std::unexpected(
              refused(std::format("LUT_3D_SIZE = {} (de {} à {} points)", size, kLutMinSize, kLutMaxSize)));
        lut.size = size;
        expected =
            static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
        lut.rgb.reserve(expected * 3);
        continue;
      }
      if (line.starts_with("DOMAIN_MIN") || line.starts_with("DOMAIN_MAX")) {
        auto& target = line.starts_with("DOMAIN_MIN") ? domain_min : domain_max;
        if (!read_triple(line.substr(10), target))
          return std::unexpected(refused(std::format("ligne {} : DOMAIN illisible", line_number)));
        continue;
      }
      if (line.starts_with("LUT_3D_INPUT_RANGE")) {  // « min max » pour les trois canaux
        std::string_view range = line.substr(18);
        float lo = 0.0F;
        float hi = 0.0F;
        if (!read_float(range, lo) || !read_float(range, hi) || !trimmed(range).empty() || !(hi > lo))
          return std::unexpected(
              refused(std::format("ligne {} : LUT_3D_INPUT_RANGE illisible", line_number)));
        std::fill(std::begin(domain_min), std::end(domain_min), lo);
        std::fill(std::begin(domain_max), std::end(domain_max), hi);
        continue;
      }
      if (line.starts_with("LUT_") || line.starts_with("DOMAIN_"))
        return std::unexpected(refused(std::format("ligne {} : mot-clé inconnu", line_number)));
      in_data = true;  // la première ligne qui n'est ni un mot-clé ni un commentaire ouvre les données
    }
    if (lut.size == 0) return std::unexpected(refused("LUT_3D_SIZE absent avant les données"));
    float rgb[3];
    if (!read_triple(line, rgb))
      return std::unexpected(refused(std::format("ligne {} : trois nombres attendus", line_number)));
    if (lut.rgb.size() >= expected * 3)
      return std::unexpected(refused(std::format("plus de {} lignes de données", expected)));
    for (int c = 0; c < 3; ++c) {
      const float span = domain_max[c] - domain_min[c];
      if (!(span > 0.0F)) return std::unexpected(refused("DOMAIN_MAX doit dépasser DOMAIN_MIN"));
      lut.rgb.push_back(std::clamp((rgb[c] - domain_min[c]) / span, 0.0F, 1.0F));
    }
  }
  if (lut.size == 0) return std::unexpected(refused("LUT_3D_SIZE absent"));
  if (lut.rgb.size() != expected * 3)
    return std::unexpected(
        refused(std::format("{} lignes de données au lieu de {}", lut.rgb.size() / 3, expected)));
  return lut;
}

std::uint16_t to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent >= 31)
    return static_cast<std::uint16_t>(sign | 0x7c00U);  // trop grand : infini (jamais pour [0, 1])
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);  // trop petit : zéro
    mantissa |= 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t half = mantissa >> shift;
    if ((mantissa >> (shift - 1U)) & 1U) ++half;  // arrondi au plus proche
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = (static_cast<std::uint32_t>(exponent) << 10U) | (mantissa >> 13U);
  if (mantissa & 0x1000U) ++half;  // arrondi au plus proche (la retenue peut monter l'exposant : voulu)
  return static_cast<std::uint16_t>(sign | half);
}

std::size_t fill_unreachable(CubeLut& lut) {
  const auto n = static_cast<std::size_t>(lut.size);
  if (n < 2 || lut.rgb.size() != n * n * n * 3) return 0;
  const auto index = [&](std::size_t r, std::size_t g, std::size_t b) { return (b * n * n + g * n + r); };
  // Invalide : sortie exactement noire alors que l'entrée n'est pas proche du noir (somme des indices > 3).
  std::vector<char> valid(n * n * n, 1);
  std::size_t invalid = 0;
  for (std::size_t b = 0; b < n; ++b)
    for (std::size_t g = 0; g < n; ++g)
      for (std::size_t r = 0; r < n; ++r) {
        const std::size_t i = index(r, g, b);
        if (r + g + b > 3 && lut.rgb[3 * i] == 0.0F && lut.rgb[3 * i + 1] == 0.0F &&
            lut.rgb[3 * i + 2] == 0.0F) {
          valid[i] = 0;
          ++invalid;
        }
      }
  const std::size_t total = invalid;
  while (invalid > 0) {  // une couche de voisins par tour, vers l'intérieur de la zone noire
    std::vector<std::size_t> fill;
    std::vector<std::array<float, 3>> values;
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t g = 0; g < n; ++g)
        for (std::size_t r = 0; r < n; ++r) {
          if (valid[index(r, g, b)]) continue;
          std::array<float, 3> sum{0.0F, 0.0F, 0.0F};
          int count = 0;
          const std::array<std::array<int, 3>, 6> steps{
              {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}};
          for (const auto& step : steps) {
            const auto rr = static_cast<long>(r) + step[0];
            const auto gg = static_cast<long>(g) + step[1];
            const auto bb = static_cast<long>(b) + step[2];
            if (rr < 0 || gg < 0 || bb < 0 || rr >= static_cast<long>(n) || gg >= static_cast<long>(n) ||
                bb >= static_cast<long>(n))
              continue;
            const std::size_t j = index(static_cast<std::size_t>(rr), static_cast<std::size_t>(gg),
                                        static_cast<std::size_t>(bb));
            if (!valid[j]) continue;
            for (std::size_t c = 0; c < 3; ++c) sum[c] += lut.rgb[3 * j + c];
            ++count;
          }
          if (count == 0) continue;
          fill.push_back(index(r, g, b));
          values.push_back({sum[0] / static_cast<float>(count), sum[1] / static_cast<float>(count),
                            sum[2] / static_cast<float>(count)});
        }
    if (fill.empty()) break;  // plus aucun voisin valide : le reste est laissé tel quel
    for (std::size_t k = 0; k < fill.size(); ++k) {
      for (std::size_t c = 0; c < 3; ++c) lut.rgb[3 * fill[k] + c] = values[k][c];
      valid[fill[k]] = 1;
    }
    invalid -= fill.size();
  }
  return total - invalid;
}

CubeLut identity_lut(int size) {
  CubeLut lut;
  lut.size = size;
  const auto n = static_cast<std::size_t>(size);
  lut.rgb.reserve(n * n * n * 3);
  for (int b = 0; b < size; ++b)
    for (int g = 0; g < size; ++g)
      for (int r = 0; r < size; ++r) {
        lut.rgb.push_back(static_cast<float>(r) / static_cast<float>(size - 1));
        lut.rgb.push_back(static_cast<float>(g) / static_cast<float>(size - 1));
        lut.rgb.push_back(static_cast<float>(b) / static_cast<float>(size - 1));
      }
  return lut;
}

std::string lut_texture_block(const CubeLut& lut, std::string_view name) {
  std::string out =
      std::format("//!TEXTURE {}\n//!SIZE {} {} {}\n//!FORMAT rgba16hf\n//!FILTER LINEAR\n//!BORDER CLAMP\n",
                  name, lut.size, lut.size, lut.size);
  static constexpr char kHex[] = "0123456789abcdef";
  const std::size_t texels = lut.rgb.size() / 3;
  out.reserve(out.size() + texels * 16 + texels / 4 + 8);
  const std::uint16_t one = to_half(1.0F);
  const auto put = [&](std::uint16_t half) {  // petit-boutiste, comme la texture en mémoire
    const unsigned char bytes[2] = {static_cast<unsigned char>(half & 0xffU),
                                    static_cast<unsigned char>(half >> 8U)};
    for (const unsigned char byte : bytes) {
      out.push_back(kHex[byte >> 4U]);
      out.push_back(kHex[byte & 0xfU]);
    }
  };
  for (std::size_t i = 0; i < texels; ++i) {
    put(to_half(lut.rgb[3 * i]));
    put(to_half(lut.rgb[3 * i + 1]));
    put(to_half(lut.rgb[3 * i + 2]));
    put(one);
    if (i % 4 == 3) out.push_back('\n');
  }
  out.push_back('\n');
  return out;
}

}  // namespace ajp::render
