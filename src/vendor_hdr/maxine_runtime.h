// Bibliothèques du SDK Maxine VFX ouvertes par leur chemin absolu, à la place de LD_LIBRARY_PATH.
//
// libVideoFX et sa fonction TrueHDR ouvrent CUDA, NPP, TensorRT et la fonction par leur nom ;
// le lecteur actuel mettait donc lib/, features/*/lib, external/cuda/lib et
// external/tensorrt/lib dans LD_LIBRARY_PATH (ta2b/mpv-animejanai-hdr
// portable_config/hdr-tools/launch-linux.sh:29-33). Une bibliothèque déjà chargée satisfait ce
// nom : on les ouvre toutes d'avance, par leur chemin (loadable_libraries de sdk_layout.h).
//   - les bibliothèques exigées par sdk_layout, et toutes celles de features/nvvfxtruehdr/lib
//     comme dans animejanai-inference-rtx@118c272 src/pipeline_hdr.c:55-61, doivent s'ouvrir ;
//   - les autres de lib/ et d'external/, que LD_LIBRARY_PATH n'aurait chargées qu'à la demande,
//     s'ouvrent si elles le peuvent : un échec est noté au journal, sans refus.
// Les dépendances d'abord (external/), puis le cœur (lib/) en RTLD_LOCAL, puis la fonction en
// RTLD_GLOBAL (pipeline_hdr.c:58). L'ordre exact entre elles n'étant pas connu, on refait des
// passes tant qu'une passe en ouvre au moins une.
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
