#include "app/play/display_edid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <vector>

namespace ajp::app::play {
namespace {

constexpr std::size_t kBlock = 128;
constexpr std::uint8_t kCtaTag = 0x02;
constexpr std::uint8_t kDisplayIdTag = 0x70;
constexpr std::uint8_t kCtaExtended = 7;           // étiquette « étendue » d'un bloc de données CTA
constexpr std::uint8_t kCtaHdrStatic = 0x06;       // métadonnées HDR statiques
constexpr std::uint8_t kDisplayParameters = 0x21;  // DisplayID 2.0
constexpr std::size_t kTechnologyByte = 0x1E;      // dans le bloc 0x21, en-tête compris
constexpr std::uint8_t kAmoled = 0b010;
constexpr double kOledBlackNits = 0.001;
constexpr int kOledBlackCode = 2;
constexpr double kOledFullScreenRatio = 0.5;

// Demi-flottant IEEE 754 (luminances de DisplayID 2.0, en nits).
double half_to_double(std::uint16_t h) {
  const int exponent = (h >> 10U) & 0x1F;
  const int mantissa = h & 0x3FF;
  const double sign = (h & 0x8000U) != 0 ? -1.0 : 1.0;
  if (exponent == 0) return sign * std::ldexp(mantissa, -24);
  if (exponent == 31) return sign * INFINITY;
  return sign * std::ldexp(mantissa + 1024, exponent - 25);
}

std::string monitor_name(std::span<const std::uint8_t> base) {
  for (std::size_t at = 54; at + 18 <= 126; at += 18) {
    if (base[at] != 0 || base[at + 1] != 0 || base[at + 3] != 0xFC) continue;
    std::string name;
    for (std::size_t i = at + 5; i < at + 18 && base[i] != 0x0A; ++i) name += static_cast<char>(base[i]);
    while (!name.empty() && name.back() == ' ') name.pop_back();
    return name;
  }
  return {};
}

void read_cta(std::span<const std::uint8_t> block, DisplayFacts& out) {
  const std::size_t end = std::min<std::size_t>(block[2], kBlock - 1);
  for (std::size_t at = 4; at < end;) {
    const std::uint8_t tag = block[at] >> 5U;
    const std::size_t length = block[at] & 0x1FU;
    if (at + 1 + length > end) break;
    if (tag == kCtaExtended && length >= 3 && block[at + 1] == kCtaHdrStatic) {
      const std::span<const std::uint8_t> hdr = block.subspan(at + 2, length - 1);
      out.pq = (hdr[0] & 0x04U) != 0;
      if (hdr.size() >= 3 && hdr[2] != 0) out.peak_nits = 50.0 * std::exp2(hdr[2] / 32.0);
      if (hdr.size() >= 4 && hdr[3] != 0) out.frame_average_nits = 50.0 * std::exp2(hdr[3] / 32.0);
      if (hdr.size() >= 5) {
        out.min_code = hdr[4];
        if (out.peak_nits) out.min_nits = *out.peak_nits * std::pow(hdr[4] / 255.0, 2.0) / 100.0;
      }
    }
    at += 1 + length;
  }
}

void read_displayid(std::span<const std::uint8_t> block, DisplayFacts& out) {
  const std::size_t end = std::min<std::size_t>(5 + block[2], kBlock - 1);
  for (std::size_t at = 5; at + 3 <= end;) {
    const std::uint8_t tag = block[at];
    const std::size_t length = block[at + 2];
    if (at + 3 + length > end) break;
    if (tag == kDisplayParameters && 3 + length > kTechnologyByte) {
      const std::span<const std::uint8_t> params = block.subspan(at, 3 + length);
      out.displayid_oled = ((params[kTechnologyByte] >> 4U) & 0x7U) == kAmoled;
      const auto nits = [&](std::size_t lo) {
        return half_to_double(static_cast<std::uint16_t>(params[lo] | (params[lo + 1] << 8U)));
      };
      if (const double peak = nits(0x1A); peak > 0.0 && !out.peak_nits) out.peak_nits = peak;
      if (const double black = nits(0x1C); black > 0.0) out.min_nits = black;
    }
    at += 3 + length;
  }
}

// Modèles dont l'EDID ne dit pas la technologie (pas de bloc 0x21).
struct KnownPanel {
  std::string_view manufacturer;
  std::uint16_t product;
  std::string_view name;
  std::string_view panel;
  bool oled;
};
constexpr std::array kKnownPanels{
    // MSI MAG 272U E16 (EDID CND 0x8CD7) : QD-OLED « penta tandem », selon son propriétaire. Il
    // annonce 1037 nits, 265 en plein écran et un noir code 2 (0,00064 nit).
    KnownPanel{
        .manufacturer = "CND", .product = 0x8CD7, .name = "MAG 272U E16", .panel = "QD-OLED", .oled = true},
};

std::string pnp_id(std::uint8_t high, std::uint8_t low) {
  const unsigned code = (static_cast<unsigned>(high) << 8U) | low;
  std::string id;
  for (const unsigned shift : {10U, 5U, 0U}) id += static_cast<char>('@' + ((code >> shift) & 0x1FU));
  return id;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string trimmed(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::string text;
  std::getline(file, text);
  return text;
}

}  // namespace

DisplayFacts parse_edid(std::span<const std::uint8_t> edid) {
  DisplayFacts out;
  if (edid.size() < kBlock) return out;
  out.name = monitor_name(edid.first(kBlock));
  out.manufacturer = pnp_id(edid[8], edid[9]);
  out.product = static_cast<std::uint16_t>(edid[10] | (edid[11] << 8U));
  for (std::size_t at = kBlock; at + kBlock <= edid.size(); at += kBlock) {
    const std::span<const std::uint8_t> block = edid.subspan(at, kBlock);
    if (block[0] == kCtaTag) read_cta(block, out);
    if (block[0] == kDisplayIdTag) read_displayid(block, out);
  }
  if (out.displayid_oled) {
    out.oled = *out.displayid_oled;
    out.source = OledSource::DisplayId;
    return out;
  }
  for (const KnownPanel& known : kKnownPanels) {
    if ((known.manufacturer == out.manufacturer && known.product == out.product) || known.name == out.name) {
      out.panel = known.panel;
      out.oled = known.oled;
      out.source = OledSource::KnownModel;
      return out;
    }
  }
  const bool black =
      (out.min_code && *out.min_code <= kOledBlackCode) || (out.min_nits && *out.min_nits <= kOledBlackNits);
  const bool limiter = !out.peak_nits || !out.frame_average_nits ||
                       *out.frame_average_nits <= kOledFullScreenRatio * *out.peak_nits;
  if (out.min_code || out.min_nits) {
    out.oled = black && limiter;
    out.source = OledSource::Heuristic;
  }
  return out;
}

std::string describe(const DisplayFacts& d) {
  std::string text = d.name.empty() ? std::string("écran sans nom") : d.name;
  if (d.pq) text += " · HDR10";
  if (d.peak_nits) text += std::format(", pic {:.0f} nits", *d.peak_nits);
  if (d.frame_average_nits) text += std::format(", moyenne {:.0f}", *d.frame_average_nits);
  if (d.min_nits) text += std::format(", noir {:.5f} nit", *d.min_nits);
  switch (d.source) {
    case OledSource::DisplayId:
      text += d.oled ? " · OLED (DisplayID)" : " · pas OLED (DisplayID)";
      break;
    case OledSource::KnownModel:
      text += std::format(" · {} (modèle connu)", d.panel);
      break;
    case OledSource::Heuristic:
      text += d.oled ? " · OLED probable (EDID)" : " · pas OLED (EDID)";
      break;
    case OledSource::Unknown:
      text += " · OLED : inconnu";
      break;
  }
  return text;
}

std::optional<DisplayFacts> read_display(std::string_view wanted) {
  std::vector<DisplayFacts> found;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", error)) {
    const std::filesystem::path& dir = entry.path();
    if (trimmed(dir / "status") != "connected") continue;
    const std::vector<std::uint8_t> edid = read_file(dir / "edid");
    if (edid.size() < kBlock) continue;
    DisplayFacts facts = parse_edid(edid);
    facts.connector = dir.filename().string();
    found.push_back(std::move(facts));
  }
  if (found.empty()) return std::nullopt;
  if (found.size() == 1) return found.front();
  for (const DisplayFacts& facts : found) {
    if (!facts.name.empty() && wanted.find(facts.name) != std::string_view::npos) return facts;
  }
  return found.front();
}

}  // namespace ajp::app::play
