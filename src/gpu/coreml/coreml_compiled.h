// Le modèle compilé par Core ML qu'ONNX Runtime laisse dans son cache, tel que Core ML direct
// (coreml_direct.h) le reprend : sa place, et les noms de ses entrées et sorties. Sans Core ML ni
// ONNX Runtime, pour que les tests les vérifient sous tous les systèmes.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ajp::gpu::coreml_rt {

// Le nom qu'ONNX Runtime donne à un tenseur dans un programme ML (GetSafeName de model_builder.cc) :
// « _ » devant un chiffre initial, « _ » à la place de tout caractère hors de [A-Za-z0-9_].
std::string ml_program_name(std::string_view name);

// Le seul modèle compilé (dossier .mlmodelc) sous dir ; rien s'il n'y en a aucun, ou plusieurs :
// ONNX Runtime a coupé le réseau en morceaux, avec des nœuds au CPU entre eux.
std::optional<std::filesystem::path> single_compiled(const std::filesystem::path& dir);

}  // namespace ajp::gpu::coreml_rt
