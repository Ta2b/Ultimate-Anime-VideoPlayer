// Les raccourcis du clavier de l'utilisateur : la section [keys] de user.toml, dans le dossier de données du
// lecteur (core/keys.h). Le reste de user.toml (les anciens profils, retirés) n'est plus lu.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/keys.h"

namespace ajp::app::play {

class Controls;

// Les raccourcis de [keys], pour les écrans et la lecture. Tant que apply_user_keys n'a pas été appelée, ce
// sont les raccourcis par défaut. Une entrée fautive est ignorée seule (journal, puis note_ignored_keys) :
// les autres redéfinitions tiennent (A24-07).
void apply_user_keys(std::string_view user_toml);
// Lit user.toml s'il existe et en applique [keys] ; absent ou illisible : les raccourcis par défaut.
void load_user_keys(const std::filesystem::path& user_toml);
const core::KeyMap& configured_keys();
// Les entrées ignorées par la dernière lecture, une seule fois.
std::vector<std::string> take_ignored_keys();
// Les dit à l'écran, au premier message de l'interface : un raccourci qui ne répond pas a une raison visible.
void note_ignored_keys(Controls& controls);

}  // namespace ajp::app::play
