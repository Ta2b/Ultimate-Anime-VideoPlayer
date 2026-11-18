// Les modèles de la stabilisation temporelle (temporal_fix.h) : les ONNX de pifroggi/vs_temporalfix (Apache
// 2.0), livrés dans models/temporalfix, puis leur variante à la taille de travail de la lecture
// (derived_model.h) : dimensions fixées, et l'entrée des sept images empilées (1 × 21 × H × W) coupée en sept
// entrées 1 × 3 × H × W (prepare/onnx_frames.h), que l'étage gpu fait tourner sans copie.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "app/play/temporal_fix.h"
#include "core/types.h"

namespace ajp::app::play {

// Le nom de l'entrée des modèles livrés, et le dossier où ils sont rangés sous les modèles du lecteur.
inline constexpr std::string_view kTemporalFixInput = "input";
inline constexpr std::string_view kTemporalFixDir = "temporalfix";

// Le modèle de cette force dans le premier de ces dossiers qui l'a ; rien éteint, ou absent partout.
std::optional<std::filesystem::path> find_temporal_fix(TemporalFix fix,
                                                       std::span<const std::filesystem::path> dirs);

// Le modèle à construire pour des images de travail width × height (paires) : dimensions fixées et entrée
// coupée en kTemporalFixFrames, dans dir. Refus : FieldOutOfRange (taille impaire ou nulle), ceux de
// derived_model et des transformations.
std::expected<std::filesystem::path, Refusal> static_temporal_fix(const std::filesystem::path& model,
                                                                  int width, int height,
                                                                  const std::filesystem::path& dir);

}  // namespace ajp::app::play
