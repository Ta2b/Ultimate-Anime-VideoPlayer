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

#include "app/play/i18n.h"
#include "core/dir_entries.h"

#if defined(__linux__)
#include <bit>
#include <cstdint>
#include <cstring>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// SetupAPI après windows.h.
#include <setupapi.h>
#endif

namespace ajp::app::play {
namespace {

constexpr std::size_t kBlock = 128;
constexpr std::uint8_t kCtaTag = 0x02;
constexpr std::uint8_t kDisplayIdTag = 0x70;
constexpr std::uint8_t kCtaVendor = 3;                 // bloc propre au fabricant (tag 3)
constexpr std::uint8_t kDisplayIdAdaptiveSync = 0x2B;  // DisplayID 2.0
constexpr int kMinVrrSpanHz = 10;                      // une plage plus étroite n'est pas du VRR
constexpr std::uint8_t kCtaExtended = 7;               // étiquette « étendue » d'un bloc de données CTA
constexpr std::uint8_t kCtaHdrStatic = 0x06;           // métadonnées HDR statiques
constexpr std::uint8_t kDisplayParameters = 0x21;      // DisplayID 2.0
constexpr std::size_t kTechnologyByte = 0x1E;          // dans le bloc 0x21, en-tête compris
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

// Une luminance qu'un écran peut annoncer (A11-05) : finie et positive, 10 000 nits au plus (le plafond du
// PQ). Un demi-flottant 0x7C00 (infini) ou un NaN d'un EDID abîmé n'est pas retenu.
bool plausible_nits(double nits) {
  return std::isfinite(nits) && nits > 0.0 && nits <= 10'000.0;
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

// Les plages VRR trouvées, par source ; parse_edid garde la première qui se tient, dans l'ordre des champs.
struct VrrSources {
  std::optional<DisplayFacts::VrrRange> displayid;
  std::optional<DisplayFacts::VrrRange> freesync;
  std::optional<DisplayFacts::VrrRange> hdmi;
  std::optional<DisplayFacts::VrrRange> descriptor;  // 0xFD, si l'EDID se dit à fréquence continue
};

// Une plage VRR qui se tient (max − min > kMinVrrSpanHz) ; la première trouvée d'une source reste.
void offer_range(std::optional<DisplayFacts::VrrRange>& slot, int min_hz, int max_hz) {
  if (slot || min_hz <= 0 || max_hz - min_hz <= kMinVrrSpanHz) return;
  slot = DisplayFacts::VrrRange{.min_hz = min_hz, .max_hz = max_hz};
}

bool has_oui(std::span<const std::uint8_t> payload, std::uint8_t a, std::uint8_t b, std::uint8_t c) {
  return payload.size() >= 3 && payload[0] == a && payload[1] == b && payload[2] == c;
}

// Les blocs propres au fabricant de CTA-861 qui disent une plage VRR : AMD FreeSync (OUI 00-00-1A, octets
// de la charge 5 et 6) et HDMI Forum (OUI C4-5D-D8, HF-VSDB : VRRmin octet 8 bits 5:0, VRRmax octets 8
// bits 7:6 et 9).
void read_vendor_vrr(std::span<const std::uint8_t> payload, VrrSources& vrr) {
  if (has_oui(payload, 0x1A, 0x00, 0x00) && payload.size() >= 7)
    offer_range(vrr.freesync, payload[5], payload[6]);
  if (has_oui(payload, 0xD8, 0x5D, 0xC4) && payload.size() >= 10) {
    const int max_hz = static_cast<int>(((payload[8] & 0xC0U) << 2U) | payload[9]);
    offer_range(vrr.hdmi, static_cast<int>(payload[8] & 0x3FU), max_hz);
  }
}

void read_cta(std::span<const std::uint8_t> block, DisplayFacts& out, VrrSources& vrr) {
  const std::size_t end = std::min<std::size_t>(block[2], kBlock - 1);
  for (std::size_t at = 4; at < end;) {
    const std::uint8_t tag = block[at] >> 5U;
    const std::size_t length = block[at] & 0x1FU;
    if (at + 1 + length > end) break;
    if (tag == kCtaVendor) read_vendor_vrr(block.subspan(at + 1, length), vrr);
    if (tag == kCtaExtended && length >= 3 && block[at + 1] == kCtaHdrStatic) {
      const std::span<const std::uint8_t> hdr = block.subspan(at + 2, length - 1);
      out.pq = (hdr[0] & 0x04U) != 0;
      if (hdr.size() >= 3 && hdr[2] != 0) out.peak_nits = out.cta_peak_nits = 50.0 * std::exp2(hdr[2] / 32.0);
      if (hdr.size() >= 4 && hdr[3] != 0) out.frame_average_nits = 50.0 * std::exp2(hdr[3] / 32.0);
      if (hdr.size() >= 5) {
        out.min_code = hdr[4];
        if (out.cta_peak_nits)
          out.min_nits = out.cta_min_nits = *out.cta_peak_nits * std::pow(hdr[4] / 255.0, 2.0) / 100.0;
      }
    }
    at += 1 + length;
  }
}

void read_displayid(std::span<const std::uint8_t> block, DisplayFacts& out, VrrSources& vrr) {
  const std::size_t end = std::min<std::size_t>(5 + block[2], kBlock - 1);
  for (std::size_t at = 5; at + 3 <= end;) {
    const std::uint8_t tag = block[at];
    const std::size_t length = block[at + 2];
    if (at + 3 + length > end) break;
    if (tag == kDisplayIdAdaptiveSync) {
      // Des descripteurs de 6 octets plus ceux que dit la révision du bloc (bits 6:4, A11-11) : minimum
      // (octet 2), maximum − 1 sur 10 bits (octets 3 et 4, bits 1:0).
      const std::size_t step = 6 + ((block[at + 1] >> 4U) & 0x7U);
      for (std::size_t d = at + 3; d + step <= at + 3 + length; d += step)
        offer_range(vrr.displayid, block[d + 2],
                    static_cast<int>(((block[d + 4] & 0x03U) << 8U) | block[d + 3]) + 1);
    }
    if (tag == kDisplayParameters && 3 + length > kTechnologyByte) {
      const std::span<const std::uint8_t> params = block.subspan(at, 3 + length);
      out.displayid_oled = ((params[kTechnologyByte] >> 4U) & 0x7U) == kAmoled;
      const auto nits = [&](std::size_t lo) {
        return half_to_double(static_cast<std::uint16_t>(params[lo] | (params[lo + 1] << 8U)));
      };
      if (const double peak = nits(0x1A); plausible_nits(peak)) {
        out.displayid_peak_nits = peak;
        if (!out.peak_nits) out.peak_nits = peak;
      }
      if (const double black = nits(0x1C); plausible_nits(black))
        out.min_nits = out.displayid_min_nits = black;
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

#ifndef _WIN32
std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string first_line(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::string text;
  std::getline(file, text);
  return text;
}
#endif

}  // namespace

namespace {

std::optional<render::Chromaticities> edid_primaries(std::span<const std::uint8_t> base) {
  const std::uint8_t low_rg = base[0x19];
  const std::uint8_t low_bw = base[0x1A];
  const auto at = [&](std::size_t high, std::uint8_t low, unsigned shift) {
    return static_cast<double>((static_cast<unsigned>(base[high]) << 2U) | ((low >> shift) & 0x03U)) / 1024.0;
  };
  const render::Chromaticities c{.rx = at(0x1B, low_rg, 6),
                                 .ry = at(0x1C, low_rg, 4),
                                 .gx = at(0x1D, low_rg, 2),
                                 .gy = at(0x1E, low_rg, 0),
                                 .bx = at(0x1F, low_bw, 6),
                                 .by = at(0x20, low_bw, 4),
                                 .wx = at(0x21, low_bw, 2),
                                 .wy = at(0x22, low_bw, 0)};
  if (std::abs(c.wx - 0.3127) > 0.01 || std::abs(c.wy - 0.3290) > 0.01) return std::nullopt;
  return c;
}

}  // namespace

DisplayFacts parse_edid(std::span<const std::uint8_t> edid) {
  DisplayFacts out;
  if (edid.size() < kBlock) return out;
  out.name = monitor_name(edid.first(kBlock));
  out.manufacturer = pnp_id(edid[8], edid[9]);
  out.product = static_cast<std::uint16_t>(edid[10] | (edid[11] << 8U));
  out.primaries = edid_primaries(edid.first(kBlock));
  // Quatre descripteurs de 18 octets dans le bloc de base, de l'octet 54 : 00 00 00 FD est la plage de
  // fréquences.
  // Le descripteur 0xFD ne dit une fréquence variable que sur un EDID 1.4 à fréquence continue : un écran
  // 60 Hz fixe y annonce souvent 56 à 76 Hz.
  const bool continuous = edid[0x12] == 1 && edid[0x13] >= 4 && (edid[0x18] & 0x01U) != 0;
  VrrSources vrr;
  for (std::size_t at = 54; at + 18 <= kBlock; at += 18) {
    if (edid[at] == 0 && edid[at + 1] == 0 && edid[at + 2] == 0 && edid[at + 3] == 0xFD) {
      // Octet 4, bits 1:0 : 2 = le maximum vertical se lit +255, 3 = le minimum aussi.
      const int min_offset = (edid[at + 4] & 0x03U) == 0x03U ? 255 : 0;
      const int max_offset = (edid[at + 4] & 0x02U) != 0 ? 255 : 0;
      if (edid[at + 5] != 0) {
        out.vrr_min_hz = edid[at + 5] + min_offset;
        if (continuous) offer_range(vrr.descriptor, *out.vrr_min_hz, edid[at + 6] + max_offset);
      }
    }
  }
  for (std::size_t at = kBlock; at + kBlock <= edid.size(); at += kBlock) {
    const std::span<const std::uint8_t> block = edid.subspan(at, kBlock);
    if (block[0] == kCtaTag) read_cta(block, out, vrr);
    if (block[0] == kDisplayIdTag) read_displayid(block, out, vrr);
  }
  out.vrr_range = vrr.displayid  ? vrr.displayid
                  : vrr.freesync ? vrr.freesync
                  : vrr.hdmi     ? vrr.hdmi
                                 : vrr.descriptor;
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

std::string luminance_sources(const DisplayFacts& d) {
  const bool cta = d.cta_peak_nits || d.cta_min_nits;
  const bool displayid = d.displayid_peak_nits || d.displayid_min_nits;
  if (!cta || !displayid) return {};
  const auto nits = [](const std::optional<double>& value, int digits) {
    return value ? std::format("{:.{}f}", *value, digits) : std::string("rien");
  };
  return std::format(
      " ; EDID, deux sources : pic CTA {} et DisplayID {}, noir CTA {} et DisplayID {} ; "
      "retenus pic {} et noir {} ; DisplayID d'abord donnerait pic {} et noir {}",
      nits(d.cta_peak_nits, 0), nits(d.displayid_peak_nits, 0), nits(d.cta_min_nits, 5),
      nits(d.displayid_min_nits, 5), nits(d.peak_nits, 0), nits(d.min_nits, 5),
      nits(d.displayid_peak_nits ? d.displayid_peak_nits : d.cta_peak_nits, 0),
      nits(d.displayid_min_nits ? d.displayid_min_nits : d.cta_min_nits, 5));
}

std::string describe_for_log(const DisplayFacts& d) {
  return describe(d) + luminance_sources(d);
}

std::string describe(const DisplayFacts& d) {
  std::string text = d.name.empty() ? std::string(i18n::tr("unnamed display")) : d.name;
  if (d.pq) text += i18n::tr(" · HDR10");
  if (d.peak_nits) text += i18n::tr_format(", peak {:.0f} nits", *d.peak_nits);
  if (d.frame_average_nits) text += i18n::tr_format(", average {:.0f}", *d.frame_average_nits);
  if (d.min_nits) text += i18n::tr_format(", black {:.5f} nit", *d.min_nits);
  switch (d.source) {
    case OledSource::DisplayId:
      text += d.oled ? i18n::tr(" · OLED (DisplayID)") : i18n::tr(" · not OLED (DisplayID)");
      break;
    case OledSource::KnownModel:
      text += i18n::tr_format(" · {} (known model)", d.panel);
      break;
    case OledSource::Heuristic:
      text += d.oled ? i18n::tr(" · probably OLED (EDID)") : i18n::tr(" · not OLED (EDID)");
      break;
    case OledSource::Unknown:
      text += i18n::tr(" · OLED: unknown");
      break;
  }
  if (d.vrr) text += *d.vrr ? i18n::tr(" · VRR on") : i18n::tr(" · VRR off");
  return text;
}

namespace {

#ifdef _WIN32
// Les écrans présents (classe de périphériques Monitor) et l'EDID que Windows garde de chacun dans
// le registre (Enum\DISPLAY\…\Device Parameters\EDID), lisible sans droits.
std::vector<DisplayFacts> connected_displays() {
  constexpr GUID kMonitorClass{0x4d36e96e, 0xe325, 0x11ce, {0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18}};
  constexpr std::size_t kMaxEdid = 256 * kBlock;
  std::vector<DisplayFacts> found;
  HDEVINFO set = SetupDiGetClassDevsW(&kMonitorClass, nullptr, nullptr, DIGCF_PRESENT);
  if (set == INVALID_HANDLE_VALUE) return found;
  SP_DEVINFO_DATA device{};
  device.cbSize = sizeof(device);
  for (DWORD i = 0; SetupDiEnumDeviceInfo(set, i, &device) != FALSE; ++i) {
    HKEY key = SetupDiOpenDevRegKey(set, &device, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ);
    if (key == INVALID_HANDLE_VALUE) continue;
    std::vector<std::uint8_t> edid(kMaxEdid);
    DWORD size = static_cast<DWORD>(edid.size());
    DWORD type = 0;
    const LSTATUS read = RegQueryValueExW(key, L"EDID", nullptr, &type, edid.data(), &size);
    RegCloseKey(key);
    if (read != ERROR_SUCCESS || type != REG_BINARY || size < kBlock) continue;
    edid.resize(size);
    DisplayFacts facts = parse_edid(edid);
    facts.connector = std::format("monitor{}", i);
    // La capacité, jamais le tearing (12.2) ; armé ou non : NVAPI (window_display) ou le retour de
    // présentation.
    facts.vrr = facts.vrr_range.has_value();
    found.push_back(std::move(facts));
  }
  SetupDiDestroyDeviceInfoList(set);
  return found;
}
#else
#if defined(__linux__)
// La valeur d'une propriété DRM d'un objet (connecteur, CRTC), par son nom ; rien si elle manque.
std::optional<std::uint64_t> drm_property(int fd, std::uint32_t object, std::uint32_t type,
                                          std::string_view name) {
  drm_mode_obj_get_properties get{};
  get.obj_id = object;
  get.obj_type = type;
  if (ioctl(fd, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &get) < 0 || get.count_props == 0) return std::nullopt;
  std::vector<std::uint32_t> ids(get.count_props);
  std::vector<std::uint64_t> values(get.count_props);
  get.props_ptr = std::bit_cast<std::uintptr_t>(ids.data());
  get.prop_values_ptr = std::bit_cast<std::uintptr_t>(values.data());
  if (ioctl(fd, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &get) < 0) return std::nullopt;
  for (std::size_t i = 0; i < std::min<std::size_t>(get.count_props, ids.size()); ++i) {
    drm_mode_get_property prop{};
    prop.prop_id = ids[i];
    if (ioctl(fd, DRM_IOCTL_MODE_GETPROPERTY, &prop) < 0) continue;
    if (name == std::string_view(static_cast<const char*>(prop.name), strnlen(prop.name, sizeof(prop.name))))
      return values[i];
  }
  return std::nullopt;
}

// Le VRR d'une sortie (« card1-DP-1 ») : vrr_capable du connecteur, lu en lecture seule sur le nœud DRM.
std::optional<bool> drm_vrr(const std::filesystem::path& dir) {
  const std::string name = dir.filename().string();
  const std::size_t dash = name.find('-');
  if (dash == std::string::npos) return std::nullopt;
  const std::string connector_text = first_line(dir / "connector_id");
  if (connector_text.empty()) return std::nullopt;
  const std::uint32_t connector_id =
      static_cast<std::uint32_t>(std::strtoul(connector_text.c_str(), nullptr, 10));
  const int fd = open(("/dev/dri/" + name.substr(0, dash)).c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return std::nullopt;
  // Capable suffit : KDE (adaptive sync « Automatique », le défaut) n'arme VRR_ENABLED qu'en plein
  // écran, jamais au lancement en fenêtre. Seul un VRR coupé à la main passerait à côté.
  std::optional<bool> result;
  const auto capable = drm_property(fd, connector_id, DRM_MODE_OBJECT_CONNECTOR, "vrr_capable");
  if (capable) result = *capable != 0;
  close(fd);
  return result;
}
// VRR_ENABLED du CRTC que l'encodeur de la sortie alimente : ce que le compositeur a armé en ce moment.
std::optional<bool> drm_vrr_armed(const std::filesystem::path& dir) {
  const std::string name = dir.filename().string();
  const std::size_t dash = name.find('-');
  if (dash == std::string::npos) return std::nullopt;
  const std::string connector_text = first_line(dir / "connector_id");
  if (connector_text.empty()) return std::nullopt;
  const std::uint32_t connector_id =
      static_cast<std::uint32_t>(std::strtoul(connector_text.c_str(), nullptr, 10));
  const int fd = open(("/dev/dri/" + name.substr(0, dash)).c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return std::nullopt;
  std::optional<bool> result;
  drm_mode_get_connector connector{};
  connector.connector_id = connector_id;
  if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &connector) >= 0) {
    result = false;
    if (connector.encoder_id != 0) {
      drm_mode_get_encoder encoder{};
      encoder.encoder_id = connector.encoder_id;
      if (ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &encoder) >= 0 && encoder.crtc_id != 0)
        result = drm_property(fd, encoder.crtc_id, DRM_MODE_OBJECT_CRTC, "VRR_ENABLED").value_or(0) != 0;
    }
  }
  close(fd);
  return result;
}
#endif

// Les sorties branchées de DRM et leur EDID (/sys/class/drm/<sortie>/edid). Sur macOS, rien :
// l'OLED y reste inconnu et le pic vient de SDL (host_facts.cpp).
std::vector<DisplayFacts> connected_displays() {
  std::vector<DisplayFacts> found;
  std::error_code error;
  for (const auto& entry : core::dir_entries("/sys/class/drm", error)) {
    const std::filesystem::path& dir = entry.path();
    if (first_line(dir / "status") != "connected") continue;
    const std::vector<std::uint8_t> edid = read_file(dir / "edid");
    if (edid.size() < kBlock) continue;
    DisplayFacts facts = parse_edid(edid);
    facts.connector = dir.filename().string();
#if defined(__linux__)
    facts.vrr = drm_vrr(dir);
    facts.vrr_armed = drm_vrr_armed(dir);
#endif
    found.push_back(std::move(facts));
  }
  return found;
}
#endif

}  // namespace

std::optional<DisplayFacts> pick_display(const std::vector<DisplayFacts>& found, std::string_view wanted) {
  if (found.empty()) return std::nullopt;
  if (found.size() == 1) return found.front();
  // « card1-DP-2 » : le nom de la sortie est ce qui suit le premier tiret.
  for (const DisplayFacts& facts : found) {
    const std::size_t dash = facts.connector.find('-');
    if (dash == std::string::npos || dash + 1 >= facts.connector.size()) continue;
    const std::string_view output = std::string_view(facts.connector).substr(dash + 1);
    const std::size_t at = wanted.find(output);
    if (at == std::string_view::npos) continue;
    // « DP-1 » ne doit pas se reconnaître dans « DP-10 ».
    const std::size_t end = at + output.size();
    if (end < wanted.size() && wanted[end] >= '0' && wanted[end] <= '9') continue;
    return facts;
  }
  std::optional<DisplayFacts> named;
  for (const DisplayFacts& facts : found) {
    if (facts.name.empty() || wanted.find(facts.name) == std::string_view::npos) continue;
    if (!named) {
      named = facts;
      continue;
    }
    named->connector.clear();  // un second écran du même nom : quelle sortie porte la fenêtre, inconnu
    named->vrr_armed.reset();
  }
  return named ? named : found.front();
}

std::optional<DisplayFacts> read_display(std::string_view wanted) {
  return pick_display(connected_displays(), wanted);
}

std::optional<bool> vrr_armed_now([[maybe_unused]] std::string_view connector) {
#if defined(__linux__)
  if (connector.empty()) return std::nullopt;
  return drm_vrr_armed(std::filesystem::path("/sys/class/drm") / std::string(connector));
#else
  return std::nullopt;
#endif
}

}  // namespace ajp::app::play
