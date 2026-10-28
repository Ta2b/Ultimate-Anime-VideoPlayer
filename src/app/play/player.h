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

#include "app/play/options.h"
#include "core/types.h"

namespace ajp::app::play {

struct PlayerPaths {
  std::filesystem::path home;        // dossier de l'utilisateur (SDL_GetUserFolder)
  std::filesystem::path data;        // ses données (default_player.h : PlayerPlace::data)
  std::filesystem::path executable;  // le lecteur lui-même, que l'inscription comme lecteur par défaut nomme
  std::filesystem::path engines;     // moteurs construits (SDL_GetPrefPath)
  std::filesystem::path settings;    // play.toml, user.toml, bench.toml… ; vide : le parent de engines (--data-dir)
  std::vector<std::filesystem::path>
      backends;                        // ajp_backend_tensorrt.so, ajp_backend_vulkan.so…, par préférence
  std::filesystem::path engine_index;  // engines.lock livré avec le lecteur (engine_install.h)
  std::filesystem::path models;  // modèles livrés : models/onnx (SPAN), models/rife ; cherchés en dernier
};

// Lit le fichier jusqu'à la fin, à « q » ou à la fermeture de la fenêtre. SDL doit être
// initialisé (vidéo et audio) et les puits du journal installés.
std::expected<void, Refusal> play(const PlayOptions& options, const PlayerPaths& paths);

}  // namespace ajp::app::play
