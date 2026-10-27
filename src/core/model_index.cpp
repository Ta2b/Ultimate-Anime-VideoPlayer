#include "core/model_index.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <utility>

#include "core/toml_fields.h"

namespace ajp::core {
namespace {

using fields::refuse;

// --- SHA-256 : constantes de FIPS 180-4, section 4.2.2. ---

constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

// --- Index. ---

constexpr std::array<std::pair<std::string_view, EngineId>, 8> kEngineNames{{
    {"vulkan", EngineId::Vulkan},
    {"tensorrt", EngineId::TensorRt},
    {"rtx_video_sdk", EngineId::RtxVideoSdk},
    {"maxine_vfx", EngineId::MaxineVfx},
    {"migraphx", EngineId::MiGraphX},
    {"openvino", EngineId::OpenVino},
    {"directml", EngineId::DirectMl},
    {"metal", EngineId::Metal},
}};
constexpr std::array<std::pair<std::string_view, Os>, 3> kOsNames{
    {{"windows", Os::Windows}, {"linux", Os::Linux}, {"macos", Os::MacOs}}};

// Un identifiant devient un nom de fichier ou de dossier : ni séparateur ni « .. », ni point
// final, que Windows retire en silence (« 11.3. » y deviendrait le dossier « 11.3 »).
bool safe_name(std::string_view text) {
  const auto allowed = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
           c == '_' || c == '-';
  };
  return !text.empty() && text.size() <= 128 && text.front() != '.' && text.back() != '.' &&
         std::ranges::all_of(text, allowed);
}

bool hex_digest(std::string_view text) {
  const auto hex = [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); };
  return text.size() == 64 && std::ranges::all_of(text, hex);
}

// Champs communs d'un fichier téléchargeable : url https, sha256, taille.
struct Download {
  std::string url;
  std::string sha256;
  std::uint64_t size = 0;
};

std::expected<Download, Refusal> parse_download(const toml::table& table, const std::string& where) {
  auto url = fields::required<std::string>(table, "url", where);
  if (!url) return std::unexpected(url.error());
  if (!url->starts_with("https://"))
    return refuse(RefusalCode::FieldOutOfRange, where + ".url (https seulement)");
  auto sha = fields::required<std::string>(table, "sha256", where);
  if (!sha) return std::unexpected(sha.error());
  if (!hex_digest(*sha)) return refuse(RefusalCode::FieldType, where + ".sha256");
  auto size = fields::required<std::int64_t>(table, "size", where);
  if (!size) return std::unexpected(size.error());
  if (auto r = fields::in_range(*size, limits::kDownloadSize, where + ".size"); !r)
    return std::unexpected(r.error());
  return Download{.url = *url, .sha256 = *sha, .size = static_cast<std::uint64_t>(*size)};
}

// Parcourt chaque table d'un tableau [[name]], avec son chemin pour les refus. skipped nul : la
// première entrée refusée refuse tout le fichier ; sinon elle y est rangée et les autres restent.
template <class Entry, class ParseOne>
std::expected<std::vector<Entry>, Refusal> parse_entries(std::string_view text, std::string_view file,
                                                         std::string_view name, ParseOne parse_one,
                                                         std::vector<Refusal>* skipped = nullptr) {
  auto root = fields::parse(text, RefusalCode::FileUnreadable, file);
  if (!root) return std::unexpected(root.error());
  if (auto version = fields::check_version(*root, kIndexVersion, file); !version) {
    return std::unexpected(version.error());
  }
  auto tables = fields::array(*root, name, file);
  if (!tables) return std::unexpected(tables.error());
  std::vector<Entry> entries;
  for (std::size_t i = 0; i < (*tables)->size(); ++i) {
    const std::string where = std::format("{}[{}]", name, i);
    const toml::table* table = (*tables)->get(i)->as_table();
    std::expected<Entry, Refusal> entry = std::unexpected(Refusal{RefusalCode::FieldType, where});
    if (table != nullptr) entry = parse_one(*table, where, entries);
    if (entry) {
      entries.push_back(std::move(*entry));
    } else if (skipped != nullptr) {
      skipped->push_back(std::move(entry.error()));
    } else {
      return std::unexpected(std::move(entry.error()));
    }
  }
  return entries;
}

std::expected<ModelEntry, Refusal> parse_model(const toml::table& table, const std::string& where,
                                               const std::vector<ModelEntry>& previous) {
  auto id = fields::required<std::string>(table, "id", where);
  if (!id) return std::unexpected(id.error());
  if (!safe_name(*id)) return refuse(RefusalCode::FieldType, std::format("{}.id = « {} »", where, *id));
  if (find_model(previous, *id) != nullptr)
    return refuse(RefusalCode::FieldOutOfRange, where + ".id (déjà défini)");
  auto download = parse_download(table, where);
  if (!download) return std::unexpected(download.error());
  auto license = fields::required<std::string>(table, "license", where);
  if (!license) return std::unexpected(license.error());
  if (license->empty()) return refuse(RefusalCode::FieldMissing, where + ".license");

  ModelEntry model{.id = *id,
                   .url = download->url,
                   .sha256 = download->sha256,
                   .size = download->size,
                   .license = *license};
  auto profiles = fields::array(table, "profiles", where);
  if (!profiles) return std::unexpected(profiles.error());
  for (std::size_t i = 0; i < (*profiles)->size(); ++i) {
    const std::string slot_where = std::format("{}.profiles[{}]", where, i);
    const std::optional<std::int64_t> slot = (*profiles)->get(i)->value_exact<std::int64_t>();
    if (!slot) return refuse(RefusalCode::FieldType, slot_where);
    if (auto r = fields::in_range(*slot, limits::kStartProfile, slot_where); !r)
      return std::unexpected(r.error());
    if (std::ranges::contains(model.profiles, *slot)) return refuse(RefusalCode::FieldOutOfRange, slot_where);
    model.profiles.push_back(static_cast<int>(*slot));
  }
  return model;
}

// Vulkan et Metal sont livrés avec le lecteur, jamais téléchargés. Les SDK de RTX HDR ont un
// paquet quand la release en publie un (sous Linux, les bibliothèques TrueHDR de Maxine), et
// s'importent sinon depuis le SDK de l'utilisateur.
bool downloadable(EngineId engine) {
  switch (engine) {
    case EngineId::TensorRt:
    case EngineId::MiGraphX:
    case EngineId::OpenVino:
    case EngineId::DirectMl:
    case EngineId::RtxVideoSdk:
    case EngineId::MaxineVfx:
      return true;
    case EngineId::Vulkan:
    case EngineId::Metal:
      return false;
  }
  return false;
}

std::expected<EnginePackage, Refusal> parse_package(const toml::table& table, const std::string& where,
                                                    const std::vector<EnginePackage>& previous) {
  auto id = fields::required<std::string>(table, "id", where);
  if (!id) return std::unexpected(id.error());
  const std::optional<EngineId> engine = parse_engine(*id);
  if (!engine) return refuse(RefusalCode::FieldOutOfRange, std::format("{}.id = « {} »", where, *id));
  if (!downloadable(*engine)) {
    return refuse(
        RefusalCode::FieldOutOfRange,
        std::format("{}.id = « {} » : livré avec le lecteur ou importé, jamais téléchargé", where, *id));
  }
  auto platform_name = fields::required<std::string>(table, "platform", where);
  if (!platform_name) return std::unexpected(platform_name.error());
  const std::optional<Os> platform = parse_os(*platform_name);
  if (!platform)
    return refuse(RefusalCode::FieldOutOfRange, std::format("{}.platform = « {} »", where, *platform_name));
  if (find_package(previous, *engine, *platform) != nullptr) {
    return refuse(RefusalCode::FieldOutOfRange, where + " (moteur et plateforme déjà définis)");
  }
  auto download = parse_download(table, where);
  if (!download) return std::unexpected(download.error());
  auto version = fields::required<std::string>(table, "version", where);
  if (!version) return std::unexpected(version.error());
  if (!safe_name(*version))
    return refuse(RefusalCode::FieldType, std::format("{}.version = « {} »", where, *version));
  auto license = fields::optional<std::string>(table, "license", where);
  if (!license) return std::unexpected(license.error());
  if (*license && !(*license)->starts_with("https://"))
    return refuse(RefusalCode::FieldOutOfRange, std::format("{}.license : https seulement", where));
  return EnginePackage{.engine = *engine,
                       .platform = *platform,
                       .url = download->url,
                       .sha256 = download->sha256,
                       .size = download->size,
                       .version = *version,
                       .license = license->value_or("")};
}

template <class E, std::size_t N>
std::optional<E> find_value(std::string_view text,
                            const std::array<std::pair<std::string_view, E>, N>& names) {
  for (const auto& [name, value] : names) {
    if (name == text) return value;
  }
  return std::nullopt;
}

template <class E, std::size_t N>
std::string_view find_name(E value, const std::array<std::pair<std::string_view, E>, N>& names) {
  for (const auto& [name, candidate] : names) {
    if (candidate == value) return name;
  }
  return "";
}

}  // namespace

// --- SHA-256. ---

void Sha256::update(std::span<const std::byte> data) {
  total_bytes_ += data.size();
  for (const std::byte byte : data) {
    block_[block_size_++] = std::to_integer<std::uint8_t>(byte);
    if (block_size_ == block_.size()) {
      compress(block_);
      block_size_ = 0;
    }
  }
}

void Sha256::update(std::string_view text) {
  update(std::as_bytes(std::span(text.data(), text.size())));
}

Sha256::Digest Sha256::finish() {
  const std::uint64_t bits = total_bytes_ * 8;
  constexpr std::array<std::byte, 1> kOne{std::byte{0x80}};
  constexpr std::array<std::byte, 1> kZero{std::byte{0}};
  update(kOne);
  while (block_size_ != 56) update(kZero);
  std::array<std::byte, 8> length{};
  for (std::size_t i = 0; i < 8; ++i) length[i] = static_cast<std::byte>(bits >> (56 - 8 * i));
  update(length);

  Digest digest{};
  for (std::size_t i = 0; i < 32; ++i) {
    digest[i] = static_cast<std::uint8_t>(state_[i / 4] >> (24 - 8 * (i % 4)));
  }
  *this = Sha256{};
  return digest;
}

void Sha256::compress(std::span<const std::uint8_t, 64> block) {
  std::array<std::uint32_t, 64> w{};
  for (std::size_t i = 0; i < 16; ++i) {
    w[i] = (std::uint32_t{block[4 * i]} << 24) | (std::uint32_t{block[4 * i + 1]} << 16) |
           (std::uint32_t{block[4 * i + 2]} << 8) | std::uint32_t{block[4 * i + 3]};
  }
  for (std::size_t i = 16; i < 64; ++i) {
    const std::uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  auto [a, b, c, d, e, f, g, h] = state_;
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint32_t t1 =
        h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) + ((e & f) ^ (~e & g)) + kRound[i] + w[i];
    const std::uint32_t t2 =
        (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  const std::array<std::uint32_t, 8> mixed{a, b, c, d, e, f, g, h};
  for (std::size_t i = 0; i < 8; ++i) state_[i] += mixed[i];
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view kDigits = "0123456789abcdef";
  std::string text;
  text.reserve(bytes.size() * 2);
  for (const std::uint8_t byte : bytes) {
    text += kDigits[byte >> 4];
    text += kDigits[byte & 0x0F];
  }
  return text;
}

std::string sha256_hex(std::string_view data) {
  Sha256 hash;
  hash.update(data);
  return to_hex(hash.finish());
}

std::expected<std::string, Refusal> sha256_file(const std::filesystem::path& path, std::stop_token stop) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, path.string());
  Sha256 hash;
  std::vector<char> buffer(std::size_t{1} << 16);
  while (in) {
    if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, path.string());
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = static_cast<std::size_t>(in.gcount());
    hash.update(std::as_bytes(std::span(buffer.data(), count)));
  }
  if (in.bad()) return refuse(RefusalCode::FileUnreadable, path.string());
  return to_hex(hash.finish());
}

// --- Noms. ---

std::string_view engine_name(EngineId engine) {
  return find_name(engine, kEngineNames);
}
std::optional<EngineId> parse_engine(std::string_view name) {
  return find_value(name, kEngineNames);
}
std::string_view os_name(Os os) {
  return find_name(os, kOsNames);
}
std::optional<Os> parse_os(std::string_view name) {
  return find_value(name, kOsNames);
}

// --- Index. ---

std::expected<std::vector<ModelEntry>, Refusal> parse_models_lock(std::string_view toml_text) {
  return parse_entries<ModelEntry>(toml_text, "models.lock", "model", parse_model);
}

std::expected<std::vector<EnginePackage>, Refusal> parse_engine_index(std::string_view toml_text) {
  return parse_entries<EnginePackage>(toml_text, "engines.lock", "engine", parse_package);
}

std::expected<EngineIndex, Refusal> read_engine_index(std::string_view toml_text) {
  EngineIndex index;
  auto packages =
      parse_entries<EnginePackage>(toml_text, "engines.lock", "engine", parse_package, &index.skipped);
  if (!packages) return std::unexpected(std::move(packages.error()));
  index.packages = std::move(*packages);
  return index;
}

std::expected<std::optional<NgcSdkSource>, Refusal> parse_maxine_source(std::string_view toml_text) {
  auto root = fields::parse(toml_text, RefusalCode::FileUnreadable, "engines.lock");
  if (!root) return std::unexpected(root.error());
  const toml::table* table = (*root)["maxine_vfx"].as_table();
  if (table == nullptr) return std::optional<NgcSdkSource>{};
  NgcSdkSource source;
  const std::array<std::pair<std::string_view, std::string*>, 6> names{{{"org", &source.org},
                                                                        {"team", &source.team},
                                                                        {"resource", &source.resource},
                                                                        {"version", &source.version},
                                                                        {"file", &source.file},
                                                                        {"feature", &source.feature}}};
  for (const auto& [key, out] : names) {
    auto value = fields::required<std::string>(*table, key, "maxine_vfx");
    if (!value) return std::unexpected(value.error());
    if (!safe_name(*value))
      return refuse(RefusalCode::FieldType, std::format("maxine_vfx.{} = « {} »", key, *value));
    *out = std::move(*value);
  }
  auto license = fields::required<std::string>(*table, "license", "maxine_vfx");
  if (!license) return std::unexpected(license.error());
  if (!license->starts_with("https://"))
    return refuse(RefusalCode::FieldOutOfRange, "maxine_vfx.license : https seulement");
  source.license = std::move(*license);
  return source;
}

const ModelEntry* find_model(std::span<const ModelEntry> lock, std::string_view id) {
  const auto found = std::ranges::find(lock, id, &ModelEntry::id);
  return found == lock.end() ? nullptr : &*found;
}

const EnginePackage* find_package(std::span<const EnginePackage> index, EngineId engine, Os platform) {
  const auto found = std::ranges::find_if(
      index, [&](const EnginePackage& p) { return p.engine == engine && p.platform == platform; });
  return found == index.end() ? nullptr : &*found;
}

std::vector<ModelEntry> models_for_profile(std::span<const ModelEntry> lock, int slot) {
  std::vector<ModelEntry> models;
  for (const ModelEntry& model : lock) {
    if (std::ranges::contains(model.profiles, slot)) models.push_back(model);
  }
  return models;
}

std::expected<void, Refusal> verify_file(const std::filesystem::path& path, std::uint64_t size,
                                         std::string_view sha256, std::stop_token stop) {
  std::error_code error;
  const std::uintmax_t actual_size = std::filesystem::file_size(path, error);
  if (error) return refuse(RefusalCode::FileUnreadable, path.string());
  if (actual_size != size) {
    return refuse(RefusalCode::HashMismatch,
                  std::format("{} : {} octets, {} attendus", path.filename().string(), actual_size, size));
  }
  auto actual = sha256_file(path, stop);
  if (!actual) return std::unexpected(actual.error());
  if (*actual != sha256) {
    return refuse(RefusalCode::HashMismatch,
                  std::format("{} : SHA-256 {}, {} attendu", path.filename().string(), *actual, sha256));
  }
  return {};
}

// --- Installation. ---

InstallLayout install_layout(Os os, const std::filesystem::path& user_data_dir) {
  return InstallLayout{.os = os,
                       .models = user_data_dir / "models",
                       .engines = user_data_dir / "backends",
                       .downloads = user_data_dir / "downloads"};
}

std::filesystem::path model_path(const InstallLayout& layout, const ModelEntry& model) {
  return layout.models / (model.id + ".onnx");
}

std::filesystem::path engine_dir(const InstallLayout& layout, const EnginePackage& package) {
  return layout.engines / std::string(engine_name(package.engine)) / package.version;
}

std::filesystem::path backend_library(const InstallLayout& layout, const EnginePackage& package) {
  // Modules CMake : .so sous macOS aussi.
  const std::string_view suffix = layout.os == Os::Windows ? ".dll" : ".so";
  return engine_dir(layout, package) / std::format("ajp_backend_{}{}", engine_name(package.engine), suffix);
}

std::filesystem::path download_path(const InstallLayout& layout, std::string_view sha256) {
  return layout.downloads / std::format("{}.part", sha256);
}

}  // namespace ajp::core
