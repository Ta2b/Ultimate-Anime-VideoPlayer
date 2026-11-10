// « ajp play » : la tranche verticale de bout en bout sur ce poste.
//
//   fichier ─▶ media (NVDEC) ─▶ gpu (SPAN ×2, TensorRT) ─▶ ring ─▶ écran (libplacebo, Vulkan)
//                 ├─▶ audio (SDL3, horloge maîtresse)
//                 └─▶ subs (libass) ─▶ calque des sous-titres
//
// La boucle du thread main attend les événements SDL et la présentation (FIFO) ; entre deux
// vsyncs, elle vide sa boîte et fait avancer la session.
#pragma once

#include <expected>
#include <filesystem>
#include <vector>

#include "app/play/default_player.h"
#include "app/play/options.h"
#include "core/types.h"

namespace ajp::app::play {

struct PlayerPaths {
  std::filesystem::path home;  // dossier de l'utilisateur (SDL_GetUserFolder)
  // Le dossier du lecteur (SDL_GetPrefPath) : update.toml, updates/, plans/, shaders.cache. L'inscription
  // comme lecteur par défaut part de son grand-parent (place_of).
  std::filesystem::path data;
  std::filesystem::path executable;  // le lecteur lui-même, que l'inscription comme lecteur par défaut nomme
  std::filesystem::path engines;     // moteurs construits (SDL_GetPrefPath)
  std::filesystem::path
      settings;  // play.toml, user.toml, bench.toml… ; vide : le parent de engines (--data-dir)
  std::vector<std::filesystem::path>
      backends;                        // ajp_backend_tensorrt.so, ajp_backend_vulkan.so…, par préférence
  std::filesystem::path engine_index;  // engines.lock livré avec le lecteur (engine_install.h)
  std::filesystem::path models;  // modèles livrés : models/onnx (SPAN), models/rife ; cherchés en dernier
  // Écrit par play() : vrai quand un moteur vient d'être mis à jour et que le lecteur doit se relancer
  // (play() rend alors sans erreur, fenêtre fermée). nullptr : la relance n'est pas demandée.
  bool* relaunch = nullptr;
};

// Où vit le lecteur, pour l'inscription et la page des mises à jour.
inline PlayerPlace place_of(const PlayerPaths& paths) {
  return {.executable = paths.executable,
          .home = paths.home,
          .data = user_data(paths.data),
          .app_data = paths.data};
}

// Lit le fichier jusqu'à la fin, à « q » ou à la fermeture de la fenêtre. SDL doit être
// initialisé (vidéo et audio) et les puits du journal installés.
std::expected<void, Refusal> play(const PlayOptions& launch_options, const PlayerPaths& paths);

// --bake-looks : la pile de LUT de `settings_file` (play.toml), intensités et ordre compris, composée en une
// seule LUT de 65 points écrite dans `out` ; sans fenêtre.
std::expected<void, Refusal> bake_looks(const std::filesystem::path& settings_file,
                                        const std::filesystem::path& luts_dir,
                                        const std::filesystem::path& out);

}  // namespace ajp::app::play
