#include "core/update_manifest.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <optional>
#include <vector>

#include "core/toml_fields.h"

namespace ajp::core {
namespace {

using fields::refuse;

struct Version {
  std::array<std::uint64_t, 3> parts{};
  std::string pre;  // « preview.1 » ; vide : version finale
};

std::optional<Version> parse_version(std::string_view text) {
  Version out;
  const std::size_t dash = text.find('-');
  if (dash != std::string_view::npos) {
    out.pre = std::string(text.substr(dash + 1));
    if (out.pre.empty()) return std::nullopt;
    text = text.substr(0, dash);
  }
  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t dot = text.find('.');
    const std::string_view part = text.substr(0, dot);
    if (part.empty()) return std::nullopt;
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), value);
    if (error != std::errc{} || end != part.data() + part.size()) return std::nullopt;
    out.parts[i] = value;
    if (dot == std::string_view::npos) {
      if (i != 2) return std::nullopt;
      return out;
    }
    text = text.substr(dot + 1);
  }
  return std::nullopt;  // un quatrième nombre
}

bool is_sha256(std::string_view text) {
  return text.size() == 64 &&
         std::ranges::all_of(text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// « 2027-02-27 » : un jour du calendrier ; rien si le texte ou la date est faux.
std::optional<std::chrono::sys_days> parse_day(std::string_view text) {
  int parts[3] = {};
  if (text.size() != 10 || text[4] != '-' || text[7] != '-') return std::nullopt;
  const std::string_view fields[3] = {text.substr(0, 4), text.substr(5, 2), text.substr(8, 2)};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto [end, error] =
        std::from_chars(fields[i].data(), fields[i].data() + fields[i].size(), parts[i]);
    if (error != std::errc{} || end != fields[i].data() + fields[i].size()) return std::nullopt;
  }
  const std::chrono::year_month_day day{std::chrono::year{parts[0]},
                                        std::chrono::month{static_cast<unsigned>(parts[1])},
                                        std::chrono::day{static_cast<unsigned>(parts[2])}};
  if (!day.ok()) return std::nullopt;
  return std::chrono::sys_days{day};
}

std::expected<std::optional<UpdateArtifact>, Refusal> artifact(const toml::table& root,
                                                               std::string_view name) {
  const toml::table* table = root[name].as_table();
  if (table == nullptr) return std::optional<UpdateArtifact>{};
  UpdateArtifact out;
  out.url = (*table)["url"].value_or(std::string());
  out.sha256 = (*table)["sha256"].value_or(std::string());
  const std::int64_t size = (*table)["size"].value_or(std::int64_t{0});
  if (!out.url.starts_with("https://"))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("manifeste [{}] : l'adresse doit être en https", name));
  if (!is_sha256(out.sha256))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("manifeste [{}] : sha256 attendu (64 chiffres hexadécimaux)", name));
  if (size <= 0 || size > kMaxUpdateBytes)
    return refuse(
        RefusalCode::FieldOutOfRange,
        std::format("manifeste [{}] : taille {} hors de 1 à {} octets", name, size, kMaxUpdateBytes));
  out.size = static_cast<std::uint64_t>(size);
  return std::optional<UpdateArtifact>(std::move(out));
}

}  // namespace

std::expected<UpdateManifest, Refusal> parse_update_manifest(std::string_view toml_text) {
  auto root = fields::parse(toml_text, RefusalCode::FileUnreadable, "manifeste");
  if (!root) return std::unexpected(root.error());
  UpdateManifest out;
  out.version = (*root)["version"].value_or(std::string());
  out.channel = (*root)["channel"].value_or(std::string("stable"));
  out.date = (*root)["date"].value_or(std::string());
  out.notes_url = (*root)["notes_url"].value_or(std::string());
  const auto expires = parse_day((*root)["expires"].value_or(std::string()));
  if (!expires)
    return refuse(RefusalCode::FieldMissing, "manifeste : expires (AAAA-MM-JJ) absent ou illisible");
  out.expires = *expires;
  if (const toml::node* revoked = root->get("revoked_keys")) {
    const toml::array* keys = revoked->as_array();
    if (keys == nullptr)
      return refuse(RefusalCode::FieldOutOfRange, "manifeste : revoked_keys doit être une liste");
    for (const toml::node& key : *keys) {
      std::string hex = key.value_or(std::string());
      std::ranges::transform(hex, hex.begin(),
                             [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (!is_sha256(hex))  // même forme : 64 chiffres hexadécimaux
        return refuse(RefusalCode::FieldOutOfRange,
                      "manifeste : revoked_keys attend des clés de 64 chiffres");
      out.revoked_keys.push_back(std::move(hex));
    }
  }
  if (!parse_version(out.version))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("manifeste : version « {} » illisible", out.version));
  if (out.channel != "stable" && out.channel != "preview")
    return refuse(RefusalCode::FieldOutOfRange, std::format("manifeste : canal « {} » inconnu", out.channel));
  if (!out.notes_url.empty() && !out.notes_url.starts_with("https://"))
    return refuse(RefusalCode::FieldOutOfRange, "manifeste : notes_url doit être en https");
  struct Slot {
    std::string_view name;
    std::optional<UpdateArtifact>* into;
  };
  for (const Slot slot : {Slot{"linux-x64", &out.linux_x64}, Slot{"windows-x64", &out.windows_x64},
                          Slot{"macos-arm64", &out.macos_arm64}}) {
    auto found = artifact(*root, slot.name);
    if (!found) return std::unexpected(found.error());
    *slot.into = std::move(*found);
  }
  return out;
}

const UpdateArtifact* artifact_for(const UpdateManifest& manifest, Os os) {
  const std::optional<UpdateArtifact>& found = os == Os::Linux     ? manifest.linux_x64
                                               : os == Os::Windows ? manifest.windows_x64
                                                                   : manifest.macos_arm64;
  return found ? &*found : nullptr;
}

std::optional<int> compare_versions(std::string_view a, std::string_view b) {
  const auto left = parse_version(a);
  const auto right = parse_version(b);
  if (!left || !right) return std::nullopt;
  if (left->parts != right->parts) return left->parts < right->parts ? -1 : 1;
  if (left->pre.empty() != right->pre.empty())
    return left->pre.empty() ? 1 : -1;  // la finale suit ses préversions
  if (left->pre == right->pre) return 0;
  // « preview.2 » contre « preview.10 » : par nombre quand les deux finissent par un nombre.
  const auto number = [](const std::string& pre) -> std::optional<std::uint64_t> {
    const std::size_t dot = pre.rfind('.');
    if (dot == std::string::npos) return std::nullopt;
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(pre.data() + dot + 1, pre.data() + pre.size(), value);
    if (error != std::errc{} || end != pre.data() + pre.size()) return std::nullopt;
    return value;
  };
  const auto x = number(left->pre);
  const auto y = number(right->pre);
  if (x && y && left->pre.substr(0, left->pre.rfind('.')) == right->pre.substr(0, right->pre.rfind('.')))
    return *x < *y ? -1 : (*x > *y ? 1 : 0);
  return left->pre < right->pre ? -1 : 1;
}

}  // namespace ajp::core
