// Production d'une entrée du cache une seule fois, entre threads et processus.
//
// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:757-806
// (run_build_spec, rejected_digest compris) et src/model_identity.cpp:157-197
// (begin_engine, recover_engine, finish_engine) : verrou de la clé, nettoyage
// des dossiers laissés par un constructeur mort, production dans un dossier de
// travail marqué, puis publication par renommage atomique.
#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>
#include <string_view>

#include "core/types.h"
#include "prepare/key_lock.h"

namespace ajp::prepare {

enum class Production : std::uint8_t {
  Reused,    // l'entrée existait déjà, ou un autre processus l'a produite pendant l'attente
  Produced,  // produite et publiée par cet appel
};

struct CacheEntry {
  std::filesystem::path destination;  // fichier final, publié par renommage
  std::string kind = "engine";        // propriétaire des dossiers : « <kind>:<nom du fichier> »
  // Attente maximale du verrou de la clé : au moins tout ce que le détenteur peut y passer (attente du GPU
  // puis construction), sans quoi un second lecteur abandonne alors que rien n'est figé (A05-02).
  std::chrono::milliseconds max_wait = kMaxLockWait;
  // Appelé une fois quand le verrou de la clé est tenu par un autre : l'attente qui suit peut durer toute la
  // construction de l'autre (C05-05). Vide : on attend sans rien dire.
  std::function<void()> on_wait;

  std::filesystem::path lock_file() const;  // « <destination>.key-lock », jamais supprimé
  // « <destination>.sha256 » : l'empreinte écrite à la publication, revérifiée à chaque reprise (P2.31).
  std::filesystem::path digest_file() const;
  std::string owner() const;
};

// Préfixe des dossiers de travail, celui du fork.
inline constexpr std::string_view kStagePrefix = ".aji-build-";

// Écrit l'entrée dans output, un fichier neuf du dossier de travail.
using Producer = std::function<std::expected<void, Refusal>(const std::filesystem::path& output)>;

// Une entrée publiée est toujours complète : elle se réutilise sans verrou si son empreinte (digest_file)
// est la sienne ; une entrée d'avant l'empreinte la reçoit à sa première reprise. Sinon
// produce tourne sous le verrou de la clé ; un échec ne laisse aucun reste, et
// une entrée refusée reste en place jusqu'à son remplacement réussi.
std::expected<Production, Refusal> produce_once(const CacheEntry& entry, const Producer& produce,
                                                const std::stop_token& stop = {});

// Met l'entrée publiée de côté (« <destination>.bad »), avec son empreinte retirée, sous le verrou de la
// clé : le runtime l'a refusée au chargement, la reprise suivante la reconstruit au lieu de la rendre
// telle quelle à chaque ouverture (C05-01). Une entrée absente n'est pas une erreur.
std::expected<void, Refusal> set_aside_entry(const CacheEntry& entry, const std::stop_token& stop = {});

// Fichier régulier non vide.
bool is_published(const std::filesystem::path& file);

}  // namespace ajp::prepare
