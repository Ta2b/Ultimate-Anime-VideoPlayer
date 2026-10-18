// Importeur à sens unique des fichiers animejanai.conf, formats 1 à 4, vers le
// profil TOML v5. Rien n'est jamais réécrit en INI.
//
// Formats lus, d'après :
//   animejanai-inference-rtx@118c272 src/aji_conf.cpp:85-205 : INI, [slot_N],
//     chain_K_* (bornes, model_M_*, rife_*) ;
//   mpv-animejanai-hdr AnimeJaNaiPlatform/PlatformServices.cs:345-452 : étapes
//     ordonnées chain_K_stage_J_* (pipeline_version = 1) et migration des clés
//     historiques ; 562-795 : validation des étapes ;
//   manager-v2 ViewModels/MainWindowViewModel.cs:923-960 : config_version,
//     default_slot, rife_engine, rife_drawings.
//
// Ce qui change ou disparaît est rendu dans ImportResult::notes, avec sa ligne,
// pour être montré à l'utilisateur ; ce qui n'a pas d'équivalent en v5 est un
// refus nommé par sa ligne.
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

inline constexpr int kIniNewestVersion = 4;

struct ImportNote {
  int line = 0;  // 0 : le fichier entier
  std::string text;
};

struct ImportResult {
  ProfileFile profile;
  std::string toml;        // texte v5 équivalent, que parse_profile relit en profile
  int source_version = 1;  // config_version du fichier lu (absent : 1)
  std::vector<ImportNote> notes;
};

std::expected<ImportResult, Refusal> import_ini(std::string_view ini_text);
std::expected<ImportResult, Refusal> import_ini_file(const std::filesystem::path& path);

}  // namespace ajp::core
