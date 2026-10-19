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
  std::filesystem::path home;                   // dossier de l'utilisateur (SDL_GetUserFolder)
  std::filesystem::path engines;                // moteurs construits (SDL_GetPrefPath)
  std::vector<std::filesystem::path> backends;  // ajp_backend_tensorrt.so, par ordre de préférence
};

// Lit le fichier jusqu'à la fin, à « q » ou à la fermeture de la fenêtre. SDL doit être
// initialisé (vidéo et audio) et les puits du journal installés.
std::expected<void, Refusal> play(const PlayOptions& options, const PlayerPaths& paths);

}  // namespace ajp::app::play
