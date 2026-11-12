#include "app/play/system_peak.h"

#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>

#include <SDL3/SDL_filesystem.h>

#include "core/charconv.h"
#include "core/log.h"
#include "render/display_model.h"

#if defined(_WIN32)
#include <dxgi1_6.h>
#include <windows.h>
#endif

namespace ajp::app::play {
namespace {

// Le nombre qui suit « "key": » dans text ; rien s'il n'y est pas (ou s'il vaut null, 0).
std::optional<double> number_after(std::string_view text, std::string_view key) {
  const std::string quoted = std::string("\"") + std::string(key) + "\"";
  const std::size_t at = text.find(quoted);
  if (at == std::string_view::npos) return std::nullopt;
  std::size_t i = text.find(':', at + quoted.size());
  if (i == std::string_view::npos) return std::nullopt;
  ++i;
  while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
  double value = 0.0;
  const auto [end, ec] = core::from_chars(text.data() + i, text.data() + text.size(), value);
  if (ec != std::errc{} || end == text.data() + i || !(value > 0.0)) return std::nullopt;
  return value;
}

// L'objet JSON qui contient la position at : de l'accolade ouvrante à sa fermante (les chaînes comprises).
std::string_view enclosing_object(std::string_view json, std::size_t at) {
  int depth = 0;
  std::size_t open = std::string_view::npos;
  for (std::size_t i = at; i-- > 0;) {
    if (json[i] == '}') ++depth;
    if (json[i] == '{') {
      if (depth == 0) {
        open = i;
        break;
      }
      --depth;
    }
  }
  if (open == std::string_view::npos) return {};
  depth = 0;
  bool in_string = false;
  for (std::size_t i = open; i < json.size(); ++i) {
    const char c = json[i];
    if (in_string) {
      if (c == '\\')
        ++i;
      else if (c == '"')
        in_string = false;
      continue;
    }
    if (c == '"') in_string = true;
    if (c == '{') ++depth;
    if (c == '}' && --depth == 0) return json.substr(open, i - open + 1);
  }
  return {};
}

// « "highDynamicRange": true » dans l'objet.
bool hdr_on(std::string_view object) {
  const std::size_t at = object.find("\"highDynamicRange\"");
  if (at == std::string_view::npos) return false;
  const std::size_t colon = object.find(':', at);
  if (colon == std::string_view::npos) return false;
  const std::size_t value = object.find_first_not_of(" \t\r\n", colon + 1);
  return value != std::string_view::npos && object.substr(value).starts_with("true");
}

// « "edidIdentifier": "MSI 15066 …" » de l'objet commence par la clé de l'écran branché.
bool same_screen(std::string_view object, std::string_view edid) {
  if (edid.empty()) return false;
  const std::size_t at = object.find("\"edidIdentifier\"");
  const std::size_t colon = at == std::string_view::npos ? at : object.find(':', at);
  const std::size_t quote = colon == std::string_view::npos ? colon : object.find('"', colon);
  if (quote == std::string_view::npos) return false;
  const std::string_view value = object.substr(quote + 1);
  return value.starts_with(edid) && value.size() > edid.size() && value[edid.size()] == ' ';
}

}  // namespace

std::string kwin_edid_key(std::string_view manufacturer, std::uint16_t product) {
  return manufacturer.empty() ? std::string() : std::format("{} {}", manufacturer, product);
}

std::optional<KwinOutput> parse_kwin_output(std::string_view json, std::string_view connector,
                                            std::string_view edid) {
  if (connector.empty()) return std::nullopt;
  // KWin garde une entrée par écran déjà branché : plusieurs peuvent porter le même connecteur. Celle de
  // l'écran branché l'emporte (A11-04) ; sans elle, celle où le HDR est actif, sinon la première.
  std::optional<KwinOutput> first;
  std::optional<KwinOutput> hdr;
  for (std::size_t from = 0;;) {
    const std::size_t at = json.find("\"connectorName\"", from);
    if (at == std::string_view::npos) break;
    from = at + 1;
    const std::size_t colon = json.find(':', at);
    const std::size_t q1 = colon == std::string_view::npos ? colon : json.find('"', colon);
    const std::size_t q2 = q1 == std::string_view::npos ? q1 : json.find('"', q1 + 1);
    if (q2 == std::string_view::npos) break;
    if (json.substr(q1 + 1, q2 - q1 - 1) != connector) continue;
    const std::string_view object = enclosing_object(json, at);
    if (object.empty()) continue;
    KwinOutput out;
    out.peak_override = number_after(object, "maxPeakBrightnessOverride");
    out.reference = number_after(object, "sdrBrightness");
    if (!out.reference) out.reference = number_after(object, "referenceLuminance");
    if (same_screen(object, edid)) return out;
    if (!hdr && hdr_on(object)) hdr = out;
    if (!first) first = out;
  }
  return hdr ? hdr : first;
}

namespace {

// Le pic se relit toutes les deux secondes (l'écran peut changer) : la ligne n'entre au journal que quand
// elle change, plus 1 800 fois par heure (P2.43).
[[maybe_unused]] void tell_peak(const std::string& line) {
  static std::mutex mutex;
  static std::string last;
  const std::scoped_lock lock(mutex);
  if (line == last) return;
  last = line;
  const auto log = log_channel("app");
  log->info("{}", line);
}

#if defined(_WIN32)
// MaxLuminance (et le plein champ) de la sortie DXGI de cet écran, ou de la plus lumineuse en HDR sans lui.
std::optional<double> dxgi_peak(HMONITOR monitor, double* full_frame_nits) {
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
    return std::nullopt;
  std::optional<double> best;
  double full_frame = 0.0;
  // Toute erreur arrête l'énumération (C11-02) : seul S_OK rend un objet. Pendant une réinitialisation du
  // pilote, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE ou E_FAIL laissaient entrer dans la boucle avec un pointeur
  // nul (ou celui, déjà libéré, du tour d'avant), et le relevé plantait le lecteur.
  IDXGIAdapter1* adapter = nullptr;
  for (UINT a = 0; SUCCEEDED(factory->EnumAdapters1(a, &adapter)) && adapter != nullptr; ++a) {
    IDXGIOutput* output = nullptr;
    for (UINT o = 0; SUCCEEDED(adapter->EnumOutputs(o, &output)) && output != nullptr; ++o) {
      IDXGIOutput6* output6 = nullptr;
      if (SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&output6)))) {
        DXGI_OUTPUT_DESC1 desc{};
        if (SUCCEEDED(output6->GetDesc1(&desc)) &&
            desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 && desc.MaxLuminance > 0.0F &&
            (monitor == nullptr || desc.Monitor == monitor) && (!best || desc.MaxLuminance > *best)) {
          best = static_cast<double>(desc.MaxLuminance);
          full_frame = static_cast<double>(desc.MaxFullFrameLuminance);
        }
        output6->Release();
      }
      output->Release();
      output = nullptr;
    }
    adapter->Release();
    adapter = nullptr;
  }
  factory->Release();
  if (best && full_frame_nits != nullptr && full_frame > 0.0) *full_frame_nits = full_frame;
  if (best)
    tell_peak(std::format("pic du système (DXGI MaxLuminance) : {:.0f} nits, plein champ {:.0f} nits", *best,
                          full_frame));
  return best;
}
#endif

}  // namespace

std::optional<double> system_peak_nits([[maybe_unused]] std::string_view connector,
                                       [[maybe_unused]] double headroom, [[maybe_unused]] const void* monitor,
                                       [[maybe_unused]] std::string_view edid,
                                       [[maybe_unused]] double* full_frame_nits) {
#if defined(_WIN32)
  return dxgi_peak(static_cast<HMONITOR>(const_cast<void*>(monitor)), full_frame_nits);
#elif defined(__linux__)
  std::string_view port = connector;
  if (port.starts_with("card")) {
    if (const std::size_t dash = port.find('-'); dash != std::string_view::npos) port.remove_prefix(dash + 1);
  }
  const std::optional<KwinOutput> output = read_kwin_output(connector, edid);
  if (!output) {
    // B37-10 : seul KDE garde un pic calibré lisible ; ailleurs, la marge de SDL suppose un blanc de 203
    // nits.
    tell_peak(
        std::format("pic du système : aucune sortie {} dans ~/.config/kwinoutputconfig.json (KDE "
                    "seulement) : pic = marge de SDL × 203 nits",
                    port));
    return std::nullopt;
  }
  // H1 : KWin affiche le contenu × blanc de référence / 203 (render/display_model.h) ; le pic se vise en
  // unités du contenu.
  const double reference = output->reference.value_or(render::kContentReferenceNits);
  if (output->peak_override) {
    const render::DisplayModel model{.panel_peak_nits = *output->peak_override, .reference_nits = reference};
    tell_peak(
        std::format("pic du système (KDE, {}) : {:.0f} nits affichés, blanc {:.0f} : {:.0f} nits du contenu",
                    port, *output->peak_override, reference, model.content_peak_nits()));
    return model.content_peak_nits();
  }
  if (output->reference && headroom > 1.0) {
    const render::DisplayModel model{.panel_peak_nits = headroom * reference, .reference_nits = reference};
    tell_peak(
        std::format("pic du système (KDE, {}) : marge {:.2f} × blanc {:.0f} = {:.0f} nits affichés, {:.0f} "
                    "nits du contenu",
                    port, headroom, reference, model.panel_peak_nits, model.content_peak_nits()));
    return model.content_peak_nits();
  }
  return std::nullopt;
#else
  return std::nullopt;
#endif
}

namespace {

[[maybe_unused]] std::optional<std::filesystem::path> kwin_config_path() {  // Linux seulement
  const char* home = SDL_GetUserFolder(SDL_FOLDER_HOME);
  if (home == nullptr) return std::nullopt;
  return std::filesystem::path(home) / ".config" / "kwinoutputconfig.json";
}

}  // namespace

std::optional<KwinOutput> read_kwin_output(std::string_view connector,
                                           [[maybe_unused]] std::string_view edid) {
#ifdef __linux__
  std::string_view port = connector;
  if (port.starts_with("card")) {
    if (const std::size_t dash = port.find('-'); dash != std::string_view::npos) port.remove_prefix(dash + 1);
  }
  const auto path = kwin_config_path();
  if (!path || port.empty()) return std::nullopt;
  std::ifstream in(*path, std::ios::binary);
  if (!in) return std::nullopt;
  const std::string json{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  return parse_kwin_output(json, port, edid);
#else
  (void)connector;
  return std::nullopt;
#endif
}

std::optional<std::int64_t> kwin_config_stamp() {
#ifdef __linux__
  const auto path = kwin_config_path();
  if (!path) return std::nullopt;
  std::error_code error;
  const auto stamp = std::filesystem::last_write_time(*path, error);
  if (error) return std::nullopt;
  return static_cast<std::int64_t>(stamp.time_since_epoch().count());
#else
  return std::nullopt;
#endif
}

}  // namespace ajp::app::play
