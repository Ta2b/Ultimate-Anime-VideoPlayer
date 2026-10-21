#include "vendor_hdr/sdk_manifest.h"

#include <filesystem>
#include <format>
#include <sstream>
#include <utility>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>

#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {
namespace {

std::unexpected<ManifestProblem> problem(std::string reason) {
  return std::unexpected(ManifestProblem{.newer_format = false, .reason = std::move(reason)});
}

// Chemin relatif qui ne remonte jamais : ni absolu, ni « .. », ni vide.
bool contained(std::string_view text) {
  const std::filesystem::path path(text);
  if (text.empty() || path.is_absolute() || path.has_root_name()) return false;
  for (const std::filesystem::path& part : path) {
    if (part == "..") return false;
  }
  return true;
}

std::expected<std::vector<ManifestFile>, ManifestProblem> parse_files(const toml::table& table) {
  std::vector<ManifestFile> files;
  const toml::array* entries = table["file"].as_array();
  if (entries == nullptr) return files;
  for (const toml::node& node : *entries) {
    const toml::table* entry = node.as_table();
    if (entry == nullptr) return problem("manifeste : [[file]] attendu");
    const std::optional<std::string> path = (*entry)["path"].value<std::string>();
    const std::optional<std::int64_t> bytes = (*entry)["bytes"].value<std::int64_t>();
    if (!path || !contained(*path) || !bytes || *bytes < 0) return problem("manifeste : fichier mal décrit");
    files.push_back(ManifestFile{.path = *path, .bytes = static_cast<std::uintmax_t>(*bytes)});
  }
  return files;
}

}  // namespace

std::string format_manifest(const Manifest& manifest) {
  toml::array files;
  for (const ManifestFile& file : manifest.files) {
    files.push_back(toml::table{{"path", file.path}, {"bytes", static_cast<std::int64_t>(file.bytes)}});
  }
  const toml::table root{{"format", kManifestFormat},
                         {"kind", std::string(layout(manifest.kind).directory)},
                         {"version", manifest.version.text()},
                         {"copy", manifest.copy},
                         {"file", std::move(files)}};
  std::ostringstream out;
  out << "# Écrit par le lecteur à chaque import ; ne pas modifier.\n" << root << '\n';
  return out.str();
}

std::expected<Manifest, ManifestProblem> parse_manifest(std::string_view text, SdkKind expected) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed) return problem(std::format("manifeste illisible : {}", parsed.error().description()));
  const toml::table& table = parsed.table();

  const std::optional<std::int64_t> format = table["format"].value<std::int64_t>();
  if (format && *format > kManifestFormat) {
    return std::unexpected(ManifestProblem{
        .newer_format = true,
        .reason = std::format(
            "manifeste au format {}, écrit par un lecteur plus récent : mettre à jour le lecteur", *format)});
  }
  if (format != kManifestFormat) return problem("manifeste sans format connu");
  if (table["kind"].value<std::string>() != layout(expected).directory) {
    return problem("manifeste d'un autre SDK");
  }
  const std::optional<SdkVersion> version = parse_version(table["version"].value_or(std::string()));
  if (!version) return problem("manifeste sans version lisible");
  const std::optional<std::string> copy = table["copy"].value<std::string>();
  if (!copy || !contained(*copy) || !copy->starts_with("versions/"))
    return problem("manifeste : copie hors du magasin");

  auto files = parse_files(table);
  if (!files) return std::unexpected(files.error());
  return Manifest{.kind = expected, .version = *version, .copy = *copy, .files = std::move(*files)};
}

}  // namespace ajp::vendor_hdr
