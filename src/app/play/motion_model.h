// La variante « mouvement » d'un modèle RIFE : le même réseau, dont le graphe sort le flot (1 × 4)
// et le masque (1 × 1) au lieu de l'image (prepare/onnx_outputs.h), pour le warp de
// kernels/spec/warp.h. Dérivée une fois dans le dossier de données du lecteur, jamais à côté du
// modèle ni dans le dépôt ; son nom porte l'empreinte du modèle source.
#pragma once

#include <expected>
#include <filesystem>

#include "core/types.h"

namespace ajp::app::play {

// Le fichier dérivé (créé s'il manque) pour rife_onnx, dans dir. Refus : FieldMissing pour un
// export RIFE dont le graphe n'a pas les tenseurs connus (la lecture garde alors RIFE image),
// ModelUnreadable, FileUnwritable.
std::expected<std::filesystem::path, Refusal> motion_model(const std::filesystem::path& rife_onnx,
                                                           const std::filesystem::path& dir);

// Les plans d'entrée du modèle RIFE : 11 (exports v1 de vs-mlrt) ou 7 (v2, grille dans le graphe).
// Refus : ModelUnreadable, InputMismatch pour un autre nombre de plans.
std::expected<int, Refusal> rife_input_planes(const std::filesystem::path& rife_onnx);

}  // namespace ajp::app::play
