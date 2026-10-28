// Le filtre qui étire la sortie de la chaîne vers la fenêtre, et le débandage de la composition
// finale : les deux réglages de libplacebo que l'écran applique à chaque image.
#pragma once

#include <cstdint>
#include <string_view>

namespace ajp::app::play {

enum class FinalScaler : std::uint8_t { Default, EwaLanczos, EwaLanczosSharp };
inline constexpr int kFinalScalerCount = 3;
std::string_view name(FinalScaler scaler);   // « ewa_lanczos », la clé de play.toml
std::string_view label(FinalScaler scaler);  // « EWA Lanczos », pour le menu

}  // namespace ajp::app::play
