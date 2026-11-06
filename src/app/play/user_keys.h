// Les raccourcis du clavier de l'utilisateur : la section [keys] de user.toml, dans le dossier de données du
// lecteur (core/keys.h). Le reste de user.toml (les anciens profils, retirés) n'est plus lu.
#pragma once

#include <filesystem>
#include <string_view>

#include "core/keys.h"

namespace ajp::app::play {

// Les raccourcis de [keys], pour les écrans et la lecture. Tant que apply_user_keys n'a pas été appelée, ou
// si la section est invalide (avertissement au journal), ce sont les raccourcis par défaut.
void apply_user_keys(std::string_view user_toml);
// Lit user.toml s'il existe et en applique [keys] ; absent ou illisible : les raccourcis par défaut.
void load_user_keys(const std::filesystem::path& user_toml);
const core::KeyMap& configured_keys();

}  // namespace ajp::app::play
