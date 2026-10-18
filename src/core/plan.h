// Plan : ce que gpu, render et export exécutent. Immuable une fois résolu,
// haché pour que les reçus et les exports se refassent à l'identique.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/profile.h"

namespace ajp::core {

enum class Schedule : std::uint8_t {
  BeforeUpscale,  // S1 : interpole à la source, puis agrandit chaque image
  AfterUpscale,   // S2 : agrandit, puis interpole à la sortie
  OutputRes,      // S3 : mouvement à la source, warp et mélange à la sortie
};

struct InterpLevel {
  Ratio factor;
  ModelId model;
  Policy policy = Policy::Quality;
  Pacing pacing = Pacing::Source;
  Schedule schedule = Schedule::BeforeUpscale;
};

struct DownscaleStep {
  DownscaleFilter filter = DownscaleFilter::Area;
  int height = 0;
};

struct ModelPass {
  ModelId model;
  int factor = 1;
};

// Un seul agrandissement par source, éventuellement en plusieurs passes de modèle.
struct UpscaleStep {
  std::vector<ModelPass> passes;
  int factor() const;
};

struct HdrRoute {
  HdrMode kind = HdrMode::Off;  // paramètres du grade et de TrueHDR à M5
};

enum class RingFormat : std::uint8_t { Rgba16Unorm, Yuv444p16 };

// Grille de présentation et facteur calé sur l'écran : core/pacing.h les calcule.
enum class Grid : std::uint8_t {
  Timestamps,  // chaque image à son horodatage : VRR, ou écran fixe sans calage possible
  Vsync,       // écran fixe : chaque image tient vsyncs_per_frame vsyncs
};

struct CadenceChoice {
  Ratio factor{1, 1};  // facteur d'interpolation total ; une fraction calée est réduite
  Grid grid = Grid::Timestamps;
  int vsyncs_per_frame = 0;  // Vsync seulement
  double speed = 1.0;        // vitesse de lecture qui cale la sortie sur les vsyncs
  double output_fps = 0.0;   // fps source × facteur, à la vitesse 1
};

struct Plan {
  std::optional<DownscaleStep> downscale;
  std::optional<UpscaleStep> upscale;
  std::vector<InterpLevel> levels;  // vide : S0, agrandissement seul
  HdrRoute hdr;
  RingFormat ring = RingFormat::Rgba16Unorm;
  // L'échelle de modèles (ladder) arrive à M6.
  int output_width = 0;
  int output_height = 0;
  double output_fps = 0.0;
  // Calage sur l'écran ; absent sans règle ou quand l'écran est inconnu.
  std::optional<CadenceChoice> cadence;
  std::uint64_t hash = 0;
};

// Hachage FNV-1a 64 bits de tout le Plan sauf le champ hash lui-même.
std::uint64_t compute_hash(const Plan& plan);

}  // namespace ajp::core
