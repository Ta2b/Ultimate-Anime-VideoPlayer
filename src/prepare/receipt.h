// Reçu de qualification : la preuve, hachée, qu'un Plan a passé ses goldens et
// son banc sur un pilote et des moteurs donnés. Il marque l'état « qualifié »
// d'une capacité sans conditionner son chargement.
//
// Document TOML. Le texte canonique (canonical_text) est écrit par ajp, champ
// par champ, dans un ordre fixe : son SHA-256, rangé dans la table [seal],
// ne dépend donc d'aucune version de toml++. La relecture recompose ce texte
// depuis les valeurs lues et refuse le reçu si le sceau diffère.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::prepare {

inline constexpr int kReceiptVersion = 1;

struct DriverRecord {
  std::string backend;  // BackendCaps::name : "tensorrt", "vulkan"…
  std::string device;
  std::string version;
  bool needs_host_sync = false;  // BackendCaps::needs_host_sync
};

// SHA-256 encadré du backend, de l'appareil et de la version du pilote.
std::string driver_hash(const DriverRecord& driver);

struct EngineRecord {
  std::string model;      // ModelSpec::id
  std::string cache_key;  // Artifact::cache_key
  std::string sha256;     // du fichier moteur
};

struct GoldenRecord {
  std::string name;  // image de référence, par exemple « span-1080p-0042 »
  double flip = 0.0;
  double flip_max = 0.0;
  double psnr_db = 0.0;
  double psnr_min_db = 0.0;
  bool passed() const { return flip <= flip_max && psnr_db >= psnr_min_db; }
};

struct BenchRecord {
  std::uint64_t pairs = 0;  // paires d'images mesurées
  double pair_ms_mean = 0.0;
  double pair_ms_p50 = 0.0;
  double pair_ms_p99 = 0.0;
  double pair_ms_max = 0.0;
  std::uint64_t host_waits = 0;     // attentes hôte hors appel constructeur
  std::uint64_t truehdr_waits = 0;  // attentes du SDK TrueHDR, comptées à part
};

struct Receipt {
  std::uint64_t plan_hash = 0;  // core::Plan::hash
  std::string created_utc;      // « 2026-10-18T15:29:26Z », UTC à la seconde
  DriverRecord driver;
  std::vector<EngineRecord> engines;
  std::vector<GoldenRecord> goldens;
  BenchRecord bench;
  std::string seal;  // SHA-256 du texte canonique ; vide avant seal_receipt
};

// Texte TOML de tout le reçu sauf son sceau.
std::string canonical_text(const Receipt& receipt);

// Calcule le sceau ; refuse une valeur hors plage (temps négatif ou non fini,
// horodatage mal formé, texte qui n'est pas de l'UTF-8…) : tout reçu scellé se relit.
std::expected<Receipt, Refusal> seal_receipt(Receipt receipt);

// Texte complet : canonique, puis la table [seal].
std::string receipt_text(const Receipt& receipt);

// Relit un reçu et vérifie son sceau et son hachage de pilote.
std::expected<Receipt, Refusal> parse_receipt(std::string_view text);

// Qualifié : au moins un golden, tous réussis, un banc mesuré, et aucune attente
// hôte pour un backend qui attend côté GPU (le reçu NVIDIA échoue sinon).
bool qualified(const Receipt& receipt);

// Écriture atomique et relecture vérifiée.
std::expected<void, Refusal> write_receipt(const std::filesystem::path& file, const Receipt& receipt);
std::expected<Receipt, Refusal> read_receipt(const std::filesystem::path& file);

// Enregistrement d'un moteur construit : son fichier est haché ici.
std::expected<EngineRecord, Refusal> record_engine(std::string model, std::string cache_key,
                                                   const std::filesystem::path& engine);

}  // namespace ajp::prepare
