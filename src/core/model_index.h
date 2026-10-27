// Index haché des modèles (models.lock) et des paquets de moteurs (engines.lock).
//
// Aucun binaire n'entre dans le dépôt : chaque modèle et chaque moteur se
// télécharge depuis l'URL de son index, puis se vérifie par sa taille et son
// SHA-256 avant d'être installé sous le dossier de données de l'utilisateur.
// Ce dossier est passé par app ; rien ici ne lit l'environnement.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/engines.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

// --- SHA-256 (FIPS 180-4), en C++ pur. ---

class Sha256 {
 public:
  using Digest = std::array<std::uint8_t, 32>;

  void update(std::span<const std::byte> data);
  void update(std::string_view text);
  // Termine le calcul ; l'objet repart ensuite de zéro.
  Digest finish();

 private:
  void compress(std::span<const std::uint8_t, 64> block);

  std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t total_bytes_ = 0;
};

std::string to_hex(std::span<const std::uint8_t> bytes);  // minuscules
std::string sha256_hex(std::string_view data);
// Refus Cancelled si `stop` est demandé pendant la lecture (plusieurs Go).
std::expected<std::string, Refusal> sha256_file(const std::filesystem::path& path, std::stop_token stop = {});

// --- Noms stables, écrits dans les index, settings.toml et la ligne de commande. ---

std::string_view engine_name(EngineId engine);  // « vulkan », « tensorrt », « rtx_video_sdk »…
std::optional<EngineId> parse_engine(std::string_view name);
std::string_view os_name(Os os);  // « windows », « linux », « macos »
std::optional<Os> parse_os(std::string_view name);

// --- Index. ---

struct ModelEntry {
  ModelId id;          // nom de fichier sans extension : lettres, chiffres, « . », « _ », « - »
  std::string url;     // https seulement
  std::string sha256;  // 64 chiffres hexadécimaux minuscules
  std::uint64_t size = 0;
  std::string license;
  std::vector<int> profiles;  // slots 1 à 9 qui l'utilisent
  friend bool operator==(const ModelEntry&, const ModelEntry&) = default;
};

struct EnginePackage {
  EngineId engine = EngineId::Vulkan;
  Os platform = Os::Windows;
  std::string url;
  std::string sha256;
  std::uint64_t size = 0;
  std::string version;  // « 11.3.0 » : lettres, chiffres, « . », « _ », « - »
  // Licence à accepter avant le téléchargement (https), vide si aucune : TensorRT (NVIDIA).
  std::string license;
  friend bool operator==(const EnginePackage&, const EnginePackage&) = default;
};

inline constexpr int kIndexVersion = 1;

namespace limits {
inline constexpr Range<std::int64_t> kDownloadSize{1, 64LL << 30};  // 64 Gio au plus
}  // namespace limits

// [[model]] id, url, sha256, size, license, profiles ; un id au plus une fois.
std::expected<std::vector<ModelEntry>, Refusal> parse_models_lock(std::string_view toml_text);
// [[engine]] id, platform, url, sha256, size, version, license (facultative) ; un couple (id, platform)
// au plus une fois.
std::expected<std::vector<EnginePackage>, Refusal> parse_engine_index(std::string_view toml_text);

// L'index tel que le lecteur le lit : une entrée refusée (un moteur qu'il ne connaît pas encore,
// publié par une release plus récente) est écartée seule, et les autres restent téléchargeables.
// Refus : TOML illisible, version d'index inconnue, [[engine]] qui n'est pas un tableau.
struct EngineIndex {
  std::vector<EnginePackage> packages;
  std::vector<Refusal> skipped;  // une par entrée écartée
};
std::expected<EngineIndex, Refusal> read_engine_index(std::string_view toml_text);

// [maxine_vfx] : le SDK Video Effects de NVIDIA (TrueHDR sous Linux), qui ne se redistribue pas
// et se télécharge depuis NGC avec la clé de l'utilisateur, sa licence acceptée à l'écran.
//   org, team, resource, version, file : la ressource du cœur du SDK (une archive .tgz) ;
//   feature : la fonction installée par son install_feature.sh ; license : https.
struct NgcSdkSource {
  std::string org;
  std::string team;
  std::string resource;
  std::string version;
  std::string file;
  std::string feature;
  std::string license;
  friend bool operator==(const NgcSdkSource&, const NgcSdkSource&) = default;
};
// Rien si l'index n'a pas de table [maxine_vfx].
std::expected<std::optional<NgcSdkSource>, Refusal> parse_maxine_source(std::string_view toml_text);

const ModelEntry* find_model(std::span<const ModelEntry> lock, std::string_view id);
const EnginePackage* find_package(std::span<const EnginePackage> index, EngineId engine, Os platform);
// Modèles à télécharger pour un profil, dans l'ordre de l'index.
std::vector<ModelEntry> models_for_profile(std::span<const ModelEntry> lock, int slot);

// Refus HashMismatch si la taille ou le SHA-256 du fichier diffèrent de l'attendu.
std::expected<void, Refusal> verify_file(const std::filesystem::path& path, std::uint64_t size,
                                         std::string_view sha256, std::stop_token stop = {});

// --- Installation, sous le dossier de données de l'utilisateur. ---

struct InstallLayout {
  Os os = Os::Windows;
  std::filesystem::path models;     // <données>/models/<id>.onnx
  std::filesystem::path engines;    // <données>/backends/<moteur>/<version>/ (engines/ : le cache)
  std::filesystem::path downloads;  // <données>/downloads/<sha256>.part, jamais lu comme installé
};

InstallLayout install_layout(Os os, const std::filesystem::path& user_data_dir);
std::filesystem::path model_path(const InstallLayout& layout, const ModelEntry& model);
std::filesystem::path engine_dir(const InstallLayout& layout, const EnginePackage& package);
// Bibliothèque du backend dans engine_dir : ajp_backend_<nom>.dll sous Windows, .so ailleurs.
std::filesystem::path backend_library(const InstallLayout& layout, const EnginePackage& package);
std::filesystem::path download_path(const InstallLayout& layout, std::string_view sha256);

}  // namespace ajp::core
