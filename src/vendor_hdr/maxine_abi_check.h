// Vérifie, avant tout chargement, que les constantes déclarées dans maxine_abi.h sont celles des
// en-têtes du SDK importé. Les en-têtes sont lus comme du texte : rien n'est compilé ni chargé.
#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "vendor_hdr/refusal.h"

namespace ajp::vendor_hdr {

// Valeur d'une constante dans des en-têtes : « #define NOM valeur », ou énumérateur, explicite
// ou implicite (le précédent plus un). Les alias (#define A B) sont suivis ; une chaîne garde
// ses guillemets. Rien si le nom n'est défini nulle part.
std::optional<std::string> find_definition(std::span<const std::string> headers, std::string_view name);

// Refuse au premier écart entre maxine::kAbi et ces en-têtes, en nommant la constante.
Done check_maxine_abi(std::span<const std::string> headers);

// Lit les .h de include/ et de features/nvvfxtruehdr/include/ sous root, puis les compare.
Done check_maxine_abi(const std::filesystem::path& root);

}  // namespace ajp::vendor_hdr
