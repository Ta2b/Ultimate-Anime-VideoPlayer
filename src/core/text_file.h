// Fichiers texte entiers : play.toml, settings.toml, user.toml, calibration.toml et les autres. L'écriture
// passe par file_sync.h (write_text_atomically).
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>

#include "core/file_sync.h"  // IWYU pragma: export (write_text_atomically)
#include "core/types.h"

namespace ajp::core {

// Lecture entière d'un fichier ordinaire ; refus FileUnreadable s'il manque ou ne se lit pas.
std::expected<std::string, Refusal> read_text_file(const std::filesystem::path& path);
// Fichier facultatif : std::nullopt s'il n'existe pas ; refus FileUnreadable s'il
// existe sans se lire, ou si son existence ne peut pas être établie (droits).
std::expected<std::optional<std::string>, Refusal> read_text_file_if_present(
    const std::filesystem::path& path);

}  // namespace ajp::core
