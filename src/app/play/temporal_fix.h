// La stabilisation temporelle (TemporalFix de pifroggi, docs/temporalfix.md) : un réseau qui lit sept images
// de travail (trois avant, l'image, trois après) et rend l'image sans le fourmillement d'une image à l'autre.
// Elle passe avant SPAN et l'interpolation, à la taille de travail : SPAN reçoit une image stable et la rend
// stable, RIFE estime son mouvement sur des images qui ne scintillent plus. Ses trois forces sont trois
// modèles (s1, s2, s3) ; l'étage gpu tient trois images d'avance (upscale_temporal.cpp).
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace ajp::app::play {

enum class TemporalFix : std::uint8_t { Off, Light, Normal, Strong };
inline constexpr int kTemporalFixCount = 4;

// Les images que le réseau lit : kTemporalFixRadius avant et après l'image.
inline constexpr int kTemporalFixRadius = 3;
inline constexpr int kTemporalFixFrames = 2 * kTemporalFixRadius + 1;

// Les images en vol que l'étage gpu tient en plus (RingStage, extra_frames) : la fenêtre de la plus lointaine
// image lue en avance (look_ahead en lit une sans que frames_ahead la compte).
constexpr std::size_t temporal_extra_frames(bool on) {
  return on ? static_cast<std::size_t>(kTemporalFixRadius) + 1 : 0;
}

struct TemporalFixLevel {
  std::string_view key;    // play.toml
  std::string_view label;  // menu, traduit par i18n::tr
  std::string_view model;  // fichier ONNX de pifroggi/vs_temporalfix ; vide : éteint
};

inline constexpr std::array<TemporalFixLevel, kTemporalFixCount> kTemporalFixLevels{{
    {"off", "Off", ""},
    {"light", "Light", "temporalfix_s1_v1.1_op18_fp16.onnx"},
    {"normal", "Normal", "temporalfix_s2_v1_op18_fp16.onnx"},
    {"strong", "Strong", "temporalfix_s3_v1_op18_fp16.onnx"},
}};

constexpr const TemporalFixLevel& level_of(TemporalFix fix) {
  return kTemporalFixLevels[static_cast<std::size_t>(fix)];
}
constexpr std::string_view name(TemporalFix fix) {
  return level_of(fix).key;
}
constexpr std::string_view label(TemporalFix fix) {
  return level_of(fix).label;
}

}  // namespace ajp::app::play
