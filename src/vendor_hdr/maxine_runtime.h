// Bibliothèques du SDK Maxine VFX ouvertes par leur chemin absolu, à la place de LD_LIBRARY_PATH.
//
// libVideoFX et sa fonction TrueHDR ouvrent leurs dépendances par leur nom (NEEDED, puis dlopen
// de libVideoFXLocal.so et de la fonction) ; le lecteur actuel mettait donc lib/, features/*/lib,
// external/cuda/lib et external/tensorrt/lib dans LD_LIBRARY_PATH (ta2b/mpv-animejanai-hdr
// portable_config/hdr-tools/launch-linux.sh:29-33). Une bibliothèque déjà chargée satisfait ce
// nom : on ouvre donc d'avance, par leur chemin et dans l'ordre de leurs dépendances, exactement
// celles que TrueHDR charge (maxine_runtime() de sdk_layout.h, relevé sur le vrai SDK). Toutes
// sont exigées. CUDA, NPP et le cœur s'ouvrent en RTLD_LOCAL, la fonction en RTLD_GLOBAL
// (animejanai-inference-rtx@118c272 src/pipeline_hdr.c:58).
#pragma once

#include <expected>
#include <filesystem>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "vendor_hdr/dynamic_library.h"

namespace ajp::vendor_hdr {

struct LoadedLibrary {
  std::filesystem::path relative;  // relatif à la racine du SDK, tel que sdk_layout le nomme
  DynamicLibrary library;
};

// Dans l'ordre d'ouverture ; rien ne reste ouvert après un refus (LibraryUnavailable).
std::expected<std::vector<LoadedLibrary>, Refusal> load_maxine_runtime(const std::filesystem::path& root);

// Bibliothèque ouverte sous ce chemin relatif ; nullptr si elle n'y est pas.
const DynamicLibrary* find_library(const std::vector<LoadedLibrary>& loaded, std::string_view relative);

}  // namespace ajp::vendor_hdr
