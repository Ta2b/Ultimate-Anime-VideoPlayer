#include "vendor_hdr/sdk_version.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "vendor_hdr/binary_probe.h"
#include "vendor_hdr/maxine_abi_check.h"
#include "vendor_hdr/path_text.h"
#include "vendor_hdr/refusal.h"
#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {
namespace {

// Un version.h tient en quelques Kio ; au-delà, ce n'est pas celui d'un SDK.
constexpr std::size_t kMaxHeaderBytes = std::size_t{1024} * 1024;
constexpr std::array<std::string_view, 4> kVersionSuffixes{"_MAJOR", "_MINOR", "_RELEASE", "_BUILD"};

std::string_view trim(std::string_view text) {
  const std::size_t first = text.find_first_not_of(" \t\r");
  if (first == std::string_view::npos) return {};
  const std::size_t last = text.find_last_not_of(" \t\r");
  return text.substr(first, last - first + 1);
}

// Le premier mot de text, et le reste après lui.
std::pair<std::string_view, std::string_view> split_word(std::string_view text) {
  text = trim(text);
  const std::size_t end = text.find_first_of(" \t");
  if (end == std::string_view::npos) return {text, {}};
  return {text.substr(0, end), trim(text.substr(end))};
}

// « #define <prefix>_MINOR 3 » : range la valeur à sa place dans parts.
void collect_define(std::string_view line, std::string_view prefix,
                    std::array<std::optional<std::uint32_t>, 4>& parts) {
  line = trim(line);
  if (!line.starts_with('#')) return;
  auto [directive, rest] = split_word(line.substr(1));
  if (directive != "define") return;
  auto [identifier, value] = split_word(rest);
  if (!identifier.starts_with(prefix)) return;
  identifier.remove_prefix(prefix.size());
  value = split_word(value).first;
  for (std::size_t part = 0; part < kVersionSuffixes.size(); ++part) {
    if (identifier != kVersionSuffixes[part]) continue;
    std::uint32_t number = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error == std::errc{} && end == value.data() + value.size()) parts[part] = number;
    return;
  }
}

std::optional<std::string> read_small_text(const std::filesystem::path& file) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  if (error || size > kMaxHeaderBytes) return std::nullopt;
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::nullopt;
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Version de la fonction TrueHDR : #define NVVFXTRUEHDR_VERSION "1.3.0.0" dans son en-tête.
std::optional<SdkVersion> feature_version(const std::filesystem::path& root) {
  const std::optional<std::string> text = read_small_text(root / kMaxineFeatureHeader);
  if (!text) return std::nullopt;
  const std::vector<std::string> headers{*text};
  const std::optional<std::string> quoted = find_definition(headers, kMaxineFeatureVersion);
  if (!quoted || quoted->size() < 2 || quoted->front() != '"' || quoted->back() != '"') return std::nullopt;
  return parse_version(std::string_view(*quoted).substr(1, quoted->size() - 2));
}

// Version portée par le nom réel de lib/libVideoFX.so, et combien de nombres ce nom en donne :
// libVideoFX.so.1.3.0 n'en porte que trois, sans le build de l'en-tête.
struct NamedVersion {
  SdkVersion version;
  std::size_t parts = 0;
};

std::optional<NamedVersion> library_version(const std::filesystem::path& root) {
  constexpr std::string_view kLibrary = "libVideoFX.so";
  std::error_code error;
  const std::filesystem::path real = std::filesystem::canonical(root / "lib" / kLibrary, error);
  if (error) return std::nullopt;
  const std::string name = utf8(real.filename());
  const std::optional<SdkVersion> version = parse_versioned_name(name, kLibrary);
  if (!version) return std::nullopt;
  const auto dots = std::ranges::count(std::string_view(name).substr(kLibrary.size() + 1), '.');
  return NamedVersion{.version = *version, .parts = static_cast<std::size_t>(dots) + 1};
}

// L'en-tête et le nom réel s'accordent sur les nombres que le nom porte.
bool agrees(const SdkVersion& header, const NamedVersion& library) {
  return std::ranges::equal(std::span(header.parts).first(library.parts),
                            std::span(library.version.parts).first(library.parts));
}

std::expected<SdkVersion, Refusal> maxine_version(const std::filesystem::path& root) {
  std::optional<SdkVersion> version;
  if (const auto text = read_small_text(root / kMaxineVersionHeader)) {
    version = parse_version_header(*text, kMaxineVersionPrefix);
  }
  const std::optional<NamedVersion> library = library_version(root);
  if (version && library && !agrees(*version, *library)) {
    return refuse(RefusalCode::SdkRejected,
                  std::format("{} annonce {}, mais lib/libVideoFX.so est en {} : SDK mélangé",
                              kMaxineVersionHeader, version->text(), library->version.text()));
  }
  if (!version && library) version = library->version;
  if (!version) {
    return refuse(
        RefusalCode::SdkRejected,
        std::format("version du SDK introuvable : ni {} ni lib/libVideoFX.so.<version> dans le dossier",
                    kMaxineVersionHeader));
  }
  const std::optional<SdkVersion> feature = feature_version(root);
  if (!feature) {
    return refuse(RefusalCode::SdkRejected,
                  std::format("version de la fonction TrueHDR introuvable : {} absent de {}",
                              kMaxineFeatureVersion, kMaxineFeatureHeader));
  }
  if (feature->major() != version->major() || feature->minor() != version->minor()) {
    return refuse(RefusalCode::SdkRejected,
                  std::format("la fonction TrueHDR {} ne correspond pas au SDK {} : réinstaller nvvfxtruehdr "
                              "avec install_feature.sh de ce SDK",
                              feature->text(), version->text()));
  }
  return *version;
}

std::expected<SdkVersion, Refusal> rtx_version(const std::filesystem::path& root) {
  // Le pont n'a pas de FileVersion : la version est celle de la fonction NGX du SDK.
  constexpr std::string_view dll = "nvngx_truehdr.dll";
  if (auto version = pe_file_version(root / dll)) return *version;
  return refuse(RefusalCode::SdkRejected,
                std::format("version du SDK introuvable : {} n'a pas de FileVersion", dll));
}

}  // namespace

std::string SdkVersion::text() const {
  return std::format("{}.{}.{}.{}", parts[0], parts[1], parts[2], parts[3]);
}

std::optional<SdkVersion> parse_version(std::string_view text) {
  SdkVersion version;
  std::size_t count = 0;
  while (true) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), version.parts[count]);
    if (error != std::errc{}) return std::nullopt;
    ++count;
    text.remove_prefix(static_cast<std::size_t>(end - text.data()));
    if (text.empty()) break;
    if (text.front() != '.' || count == version.parts.size()) return std::nullopt;
    text.remove_prefix(1);
  }
  if (count < 2) return std::nullopt;
  return version;
}

std::optional<SdkVersion> parse_version_header(std::string_view text, std::string_view prefix) {
  std::array<std::optional<std::uint32_t>, 4> parts;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    collect_define(text.substr(0, end), prefix, parts);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
  }
  const std::optional<std::uint32_t> major = parts[0];
  const std::optional<std::uint32_t> minor = parts[1];
  if (!major || !minor) return std::nullopt;
  return SdkVersion{.parts = {*major, *minor, parts[2].value_or(0), parts[3].value_or(0)}};
}

std::optional<SdkVersion> parse_versioned_name(std::string_view file_name, std::string_view library) {
  if (!file_name.starts_with(library) || file_name.size() <= library.size() + 1) return std::nullopt;
  if (file_name[library.size()] != '.') return std::nullopt;
  return parse_version(file_name.substr(library.size() + 1));
}

std::expected<SdkVersion, Refusal> read_sdk_version(SdkKind kind, const std::filesystem::path& root) {
  return kind == SdkKind::RtxVideo ? rtx_version(root) : maxine_version(root);
}

}  // namespace ajp::vendor_hdr
