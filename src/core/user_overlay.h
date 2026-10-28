// user.toml, superposé au profil de base.
//
// Le profil de base est livré et jamais réécrit. L'interface n'écrit dans
// user.toml que les clés qui diffèrent de la base ; revenir à la valeur de la
// base retire la clé. Lecture : la base, puis user.toml clé par clé, puis les
// plages et refus de profile.cpp et de keys.cpp sur le résultat.
//
// Superposition : les tables se fusionnent clé par clé ; les [[profile]]
// s'apparient par slot ; toute autre valeur, règles comprises, remplace celle
// de la base. toml++ réécrit user.toml sans ses commentaires, mais garde ses
// clés inconnues.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/keys.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

struct LayeredProfile {
  ProfileFile profile;
  KeyMap keys;
  friend bool operator==(const LayeredProfile&, const LayeredProfile&) = default;
};

// Texte TOML de la base superposée de user ; user vide = base seule.
std::expected<std::string, Refusal> merge_layers(std::string_view base_toml, std::string_view user_toml);
std::expected<LayeredProfile, Refusal> parse_layered(std::string_view base_toml, std::string_view user_toml);
// user.toml absent : la base seule.
std::expected<LayeredProfile, Refusal> load_layered(const std::filesystem::path& base,
                                                    const std::filesystem::path& user);

// Une modification faite dans l'interface.
namespace edit {
struct StartProfile {
  int slot = 1;
};
struct ProfileName {
  int slot = 1;
  std::string name;
};
struct ProfileHdr {
  int slot = 1;
  HdrMode mode = HdrMode::Off;
};
struct ProfileRules {
  int slot = 1;
  std::vector<Rule> rules;
};
struct Shortcut {
  Action action = Action::TogglePause;
  std::optional<Chord> chord;  // std::nullopt : sans raccourci
};
// Retire de user.toml tout ce qui touche ce profil : il redevient celui de la base.
struct ResetProfile {
  int slot = 1;
};
}  // namespace edit

using UserEdit = std::variant<edit::StartProfile, edit::ProfileName, edit::ProfileHdr, edit::ProfileRules,
                              edit::Shortcut, edit::ResetProfile>;

// Nouveau texte de user.toml : user_toml modifié par edits. Chaque clé modifiée
// s'écrit si elle diffère de la base, et se retire si elle lui redevient égale ;
// les clés que edits ne touchent pas restent telles quelles. Refus si une
// modification, ou le résultat superposé, l'est.
std::expected<std::string, Refusal> apply_edits(std::string_view base_toml, std::string_view user_toml,
                                                std::span<const UserEdit> edits);

// Relit base et user.toml, applique edits, puis remplace user.toml atomiquement.
// En cas de refus, user.toml reste tel quel ; la base n'est jamais écrite, et un
// user.toml qui désignerait le fichier de base est refusé (FileUnwritable).
std::expected<LayeredProfile, Refusal> save_edits(const std::filesystem::path& base,
                                                  const std::filesystem::path& user,
                                                  std::span<const UserEdit> edits);

// Fichiers texte, partagés par user.toml et settings.toml.
// Lecture entière d'un fichier ordinaire ; refus FileUnreadable s'il manque ou ne se lit pas.
std::expected<std::string, Refusal> read_text_file(const std::filesystem::path& path);
// Fichier facultatif : std::nullopt s'il n'existe pas ; refus FileUnreadable s'il
// existe sans se lire, ou si son existence ne peut pas être établie (droits).
std::expected<std::optional<std::string>, Refusal> read_text_file_if_present(
    const std::filesystem::path& path);
// Écrit dans un fichier temporaire du même dossier, l'écrit sur le disque, puis le renomme sur
// path : un lecteur voit l'ancien fichier entier ou le nouveau entier, jamais un mélange, et une
// coupure de courant ne laisse pas un fichier vide (core/file_sync.h).
std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text);

}  // namespace ajp::core
