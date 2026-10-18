#include "vendor_hdr/sdk_version.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <vector>

#include "vendor_hdr/binary_probe.h"
#include "vendor_hdr/path_text.h"
#include "vendor_hdr/refusal.h"
#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {
namespace {

// Un version.h tient en quelques Kio ; au-delà, ce n'est pas celui d'un SDK.
constexpr std::size_t kMaxHeaderBytes = std::size_t{1024} * 1024;
constexpr std::array<std::string_view, 4> kVersionSuffixes{"_MAJOR_VERSION", "_MINOR_VERSION",
                                                           "_PATCH_VERSION", "_BUILD_VERSION"};

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

struct PrefixVersion {
  std::string prefix;
  std::array<std::optional<std::uint32_t>, 4> parts;
};

// « #define NVVFX_MINOR_VERSION 3 » : range la valeur sous le préfixe NVVFX, dans l'ordre vu.
void collect_define(std::string_view line, std::vector<PrefixVersion>& found) {
  line = trim(line);
  if (!line.starts_with('#')) return;
  auto [directive, rest] = split_word(line.substr(1));
  if (directive != "define") return;
  auto [identifier, value] = split_word(rest);
  value = split_word(value).first;
  for (std::size_t part = 0; part < kVersionSuffixes.size(); ++part) {
    if (!identifier.ends_with(kVersionSuffixes[part])) continue;
    std::uint32_t number = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size()) return;
    const std::string prefix(identifier.substr(0, identifier.size() - kVersionSuffixes[part].size()));
    auto entry = std::ranges::find(found, prefix, &PrefixVersion::prefix);
    if (entry == found.end()) entry = found.insert(found.end(), PrefixVersion{.prefix = prefix});
    entry->parts[part] = number;
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

std::expected<SdkVersion, Refusal> maxine_version(const std::filesystem::path& root) {
  for (const char* candidate : {"version.h", "include/version.h"}) {
    if (auto text = read_small_text(root / candidate)) {
      if (auto version = parse_version_header(*text)) return *version;
    }
  }
  constexpr std::string_view kLibrary = "libVideoFX.so";
  std::error_code error;
  const std::filesystem::path real = std::filesystem::canonical(root / "lib" / kLibrary, error);
  if (!error) {
    if (auto version = parse_versioned_name(utf8(real.filename()), kLibrary)) return *version;
  }
  for (std::filesystem::directory_iterator it(root / "lib", error), end; !error && it != end;
       it.increment(error)) {
    if (auto version = parse_versioned_name(utf8(it->path().filename()), kLibrary)) return *version;
  }
  return refuse(RefusalCode::SdkRejected,
                "version du SDK introuvable : ni version.h ni lib/libVideoFX.so.<version> dans le dossier");
}

std::expected<SdkVersion, Refusal> rtx_version(const std::filesystem::path& root) {
  const std::string_view dll = layout(SdkKind::RtxVideo).main_library;
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

std::optional<SdkVersion> parse_version_header(std::string_view text) {
  std::vector<PrefixVersion> found;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    collect_define(text.substr(0, end), found);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
  }
  for (const PrefixVersion& entry : found) {
    const std::optional<std::uint32_t> major = entry.parts[0];
    const std::optional<std::uint32_t> minor = entry.parts[1];
    if (!major || !minor) continue;
    return SdkVersion{.parts = {*major, *minor, entry.parts[2].value_or(0), entry.parts[3].value_or(0)}};
  }
  return std::nullopt;
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
