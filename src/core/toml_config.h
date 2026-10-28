// Configuration de toml++, lue par chaque inclusion (TOML_CONFIG_HEADER, posé par
// cmake/Dependencies.cmake sur ajp::dep::toml) au milieu de son préprocesseur : après sa
// définition de TOML_ASSUME, avant celles de TOML_MAX_NESTED_VALUES et de TOML_ASSERT_ASSUME.
#pragma once

// La copie corrigée par cmake/TomlFixes.cmake : <toml++-ajp/toml.hpp>, jamais <toml++/toml.hpp>,
// dont l'analyse a des comportements indéfinis sur un fichier mal formé.
#ifndef AJP_TOML_FIXED
#error "inclure <toml++-ajp/toml.hpp>, la copie de toml++ corrigée par cmake/TomlFixes.cmake"
#endif

// Une version de toml++ qui lirait ce fichier avant sa propre définition de TOML_ASSUME
// l'écraserait sans avertissement (en-tête système) : l'erreur le dit.
#ifndef TOML_ASSUME
#error "toml++ lit TOML_CONFIG_HEADER avant TOML_ASSUME : revoir src/core/toml_config.h"
#endif
// En Release, TOML_ASSERT_ASSUME devient __assume (MSVC) ou __builtin_assume (Clang) : une
// précondition de l'analyse que toml++ croit acquise et qu'une entrée mal formée viole (comme
// celle que corrige TomlFixes.cmake) laisserait le compilateur supprimer la branche qui rapporte
// l'erreur. Rien n'est supposé : la vérification reste en Debug (TOML_ASSERT), sans effet ailleurs.
#undef TOML_ASSUME
#define TOML_ASSUME(expr) static_cast<void>(0)

// 128 valeurs imbriquées (tableaux, tables en ligne) au plus, au lieu de 256 : chaque niveau prend
// environ 3 Ko de pile sous ASan, et la branche master de toml++ a baissé la limite pour ça.
#define TOML_MAX_NESTED_VALUES 128
