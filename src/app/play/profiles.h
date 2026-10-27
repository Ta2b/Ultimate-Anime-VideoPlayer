// Les 9 profils de « ajp play » : une base livrée (TOML v5, core/profile.h) et user.toml par-dessus
// (core/user_overlay.h), dans le dossier de données du lecteur. Un profil fixe la chaîne
// (réduction, RIFE, SPAN, RTX HDR) ; play.toml garde le reste (réglages de RTX HDR, son,
// sous-titres, statistiques). Ctrl+1 à Ctrl+9 et l'onglet Profils du menu en changent.
//
// La chaîne de la tranche verticale est « réduction ? RIFE ? SPAN ×2 ? » : une interpolation
// écrite après l'agrandissement est faite avant, une étape qu'elle ne sait pas faire est dite.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/settings.h"
#include "core/keys.h"
#include "core/profile.h"
#include "core/resolve.h"
#include "core/types.h"
#include "core/user_overlay.h"

namespace ajp::app::play {

// La base livrée : 9 profils, le 1 au démarrage.
std::string_view base_profiles_toml();

// Les noms de modèles qu'un profil donne : le nom du fichier sans « .onnx ».
inline constexpr std::string_view kRifeModelId = "rife_v4.25";
inline constexpr std::string_view kSpanModelId =
    "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16";

// La chaîne du profil pour cette source (première règle qui correspond) appliquée à settings : les
// réglages hors chaîne restent. notes reçoit ce qui n'a pas pu être suivi tel quel.
PlaySettings apply_profile(const core::Profile& profile, const core::SourceInfo& source,
                           PlaySettings settings, std::vector<std::string>& notes);

// Le profil qui refait la chaîne de settings pour toutes les sources (une règle).
std::vector<core::Rule> rules_of(const PlaySettings& settings, std::string_view default_model);
core::HdrMode hdr_of(const PlaySettings& settings);

// « 720p Area → RIFE ×3 → SPAN ×2 · RTX HDR », pour le menu ; « source telle quelle » sans étape.
std::string profile_summary(const core::Profile& profile, const core::SourceInfo& source);

// La base et user.toml lus ensemble ; user.toml absent : la base seule. Refus : user.toml illisible
// ou invalide (la lecture continue alors avec la base, dite au journal).
struct Profiles {
  core::ProfileFile file;
  std::string user_text;  // user.toml tel qu'il est sur le disque, vide s'il manque
};
// Les raccourcis de la section [keys] de user.toml, pour les écrans et la lecture. Tant que
// apply_user_keys n'a pas été appelée, ou si la section est invalide (avertissement au journal,
// le reste du fichier reste utilisé), ce sont les raccourcis par défaut.
void apply_user_keys(std::string_view user_toml);
const core::KeyMap& configured_keys();

std::expected<Profiles, Refusal> load_profiles(const std::filesystem::path& user_toml);
// Applique edits à user.toml (seulement ce qui diffère de la base) et l'écrit d'un coup.
std::expected<Profiles, Refusal> edit_profiles(const std::filesystem::path& user_toml,
                                               const Profiles& current,
                                               std::span<const core::UserEdit> edits);

}  // namespace ajp::app::play
