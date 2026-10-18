// Production d'une entrée du cache une seule fois, entre threads et processus.
//
// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:757-806
// (run_build_spec) et src/model_identity.cpp:157-197 (begin_engine,
// recover_engine, finish_engine) : verrou de la clé, nettoyage des dossiers
// laissés par un constructeur mort, production dans un dossier de travail
// marqué, puis publication par renommage atomique.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::prepare {

enum class Production : std::uint8_t {
  Reused,    // l'entrée existait déjà, ou un autre processus l'a produite pendant l'attente
  Produced,  // produite et publiée par cet appel
};

struct CacheEntry {
  std::filesystem::path destination;  // fichier final, publié par renommage
  std::string kind = "engine";        // propriétaire des dossiers : « <kind>:<nom du fichier> »

  std::filesystem::path lock_file() const;  // « <destination>.key-lock », jamais supprimé
  std::string owner() const;
};

// Préfixe des dossiers de travail, celui du fork.
inline constexpr std::string_view kStagePrefix = ".aji-build-";

// Écrit l'entrée dans output, un fichier neuf du dossier de travail.
using Producer = std::function<std::expected<void, Refusal>(const std::filesystem::path& output)>;

// Une entrée publiée est toujours complète : elle se réutilise sans verrou. Sinon
// produce tourne sous le verrou de la clé ; un échec ne laisse aucun reste.
std::expected<Production, Refusal> produce_once(const CacheEntry& entry, const Producer& produce,
                                                const std::stop_token& stop = {});

// Fichier régulier non vide.
bool is_published(const std::filesystem::path& file);

}  // namespace ajp::prepare
