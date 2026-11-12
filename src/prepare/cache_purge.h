// Purge du cache des moteurs TensorRT (A33-04). Le nom d'un moteur est le condensé de sa clé (taille de la
// source, runtime, pilote) : une mise à jour du pilote ou une vidéo d'une autre taille en ajoute un jeu, et
// rien ne retirait les anciens. Un moteur repris rafraîchit la date de son empreinte (produce_once) : la
// purge retire ceux qui n'ont pas servi depuis longtemps, puis les moins récents au-delà d'une taille.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>

namespace ajp::prepare {

struct CachePurgeLimits {
  std::chrono::hours idle{24 * 45};                  // sans usage depuis : retiré
  std::uint64_t max_bytes = std::uint64_t{8} << 30;  // au-delà : les moins récents d'abord
  std::chrono::hours protected_recent{24};           // jamais retiré pour la taille : servi aujourd'hui
  // Un dossier de travail « .aji-build-* » plus vieux que cela est le reste d'une construction tuée (la plus
  // longue dure moins de deux heures) : retiré, quelle que soit la clé qui l'a ouvert (B05-01, B33-02).
  std::chrono::hours stale_build{24};
};

struct CachePurgeResult {
  int removed = 0;  // moteurs retirés (avec leur empreinte, journal et rapport des couches)
  std::uint64_t removed_bytes = 0;
  int kept = 0;
  std::uint64_t kept_bytes = 0;
  int removed_stages = 0;  // dossiers de travail orphelins retirés
  int removed_locks = 0;   // verrous « .key-lock » d'un moteur disparu retirés
};

// Ne touche que les fichiers du cache que prepare écrit dans dir : « aji-v3-*.engine » et leurs voisins
// (.sha256, .build.log, .layers.json), et les caches de temps « aji-timing-*.cache ». Ne retire
// les verrous (.key-lock) que d'un moteur disparu et sans usage depuis limits.idle, et les dossiers de
// travail « .aji-build-* » (marqueur du module) plus vieux que limits.stale_build. Aucune exception : un
// fichier qui résiste reste.
CachePurgeResult purge_engine_cache(
    const std::filesystem::path& dir, const CachePurgeLimits& limits = {},
    std::filesystem::file_time_type now = std::filesystem::file_time_type::clock::now());

}  // namespace ajp::prepare
