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
#include <dxgi1_5.h>
#include <windows.h>
// SetupAPI après windows.h.
#include <setupapi.h>
#endif

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

#ifndef _WIN32
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
#endif

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
  if (d.vrr) text += *d.vrr ? " · VRR actif" : " · VRR inactif";
  return text;
}

namespace {

#ifdef _WIN32
// Windows laisse présenter sans attendre le vsync (DXGI_FEATURE_PRESENT_ALLOW_TEARING) : l'écran à
// fréquence variable est là et le système l'utilise. Rien si DXGI 1.5 manque.
std::optional<bool> windows_vrr() {
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) ||
      factory == nullptr)
    return std::nullopt;
  std::optional<bool> result;
  IDXGIFactory5* newer = nullptr;
  if (SUCCEEDED(factory->QueryInterface(__uuidof(IDXGIFactory5), reinterpret_cast<void**>(&newer))) &&
      newer) {
    BOOL tearing = FALSE;
    if (SUCCEEDED(newer->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
      result = tearing != FALSE;
    newer->Release();
  }
  factory->Release();
  return result;
}

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
    facts.vrr = windows_vrr();
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

// Le VRR d'une sortie (« card1-DP-1 ») : vrr_capable du connecteur ET VRR_ENABLED du CRTC que son
// encodeur alimente, lus en lecture seule sur le nœud DRM (le même état que celui posé par le
// compositeur).
std::optional<bool> drm_vrr(const std::filesystem::path& dir) {
  const std::string name = dir.filename().string();
  const std::size_t dash = name.find('-');
  if (dash == std::string::npos) return std::nullopt;
  const std::string connector_text = trimmed(dir / "connector_id");
  if (connector_text.empty()) return std::nullopt;
  const std::uint32_t connector_id =
      static_cast<std::uint32_t>(std::strtoul(connector_text.c_str(), nullptr, 10));
  const int fd = open(("/dev/dri/" + name.substr(0, dash)).c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) return std::nullopt;
  std::optional<bool> result;
  const auto capable = drm_property(fd, connector_id, DRM_MODE_OBJECT_CONNECTOR, "vrr_capable");
  if (capable) {
    result = false;
    drm_mode_get_connector connector{};
    connector.connector_id = connector_id;
    drm_mode_get_encoder encoder{};
    if (*capable != 0 && ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &connector) >= 0 &&
        connector.encoder_id != 0) {
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
  for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", error)) {
    const std::filesystem::path& dir = entry.path();
    if (trimmed(dir / "status") != "connected") continue;
    const std::vector<std::uint8_t> edid = read_file(dir / "edid");
    if (edid.size() < kBlock) continue;
    DisplayFacts facts = parse_edid(edid);
    facts.connector = dir.filename().string();
#if defined(__linux__)
    facts.vrr = drm_vrr(dir);
#endif
    found.push_back(std::move(facts));
  }
  return found;
}
#endif

}  // namespace

std::optional<DisplayFacts> read_display(std::string_view wanted) {
  const std::vector<DisplayFacts> found = connected_displays();
  if (found.empty()) return std::nullopt;
  if (found.size() == 1) return found.front();
  for (const DisplayFacts& facts : found) {
    if (!facts.name.empty() && wanted.find(facts.name) != std::string_view::npos) return facts;
  }
  return found.front();
}

}  // namespace ajp::app::play
