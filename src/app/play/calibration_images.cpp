#include "app/play/calibration_images.h"

#include <algorithm>
#include <array>
#include <cstddef>

#include <SDL3/SDL_filesystem.h>

namespace ajp::app::play {

namespace {

struct Scene {
  std::string_view title;
  std::string_view guidance;
};

// Les plans de la vidéo, dans leur ordre : d'abord les cinq obligatoires (nuit avec un éclat, pénombre avec
// un visage, jour avec un visage, ciel et soleil, scène claire et ses lampes), puis les six bonus.
const std::array<Scene, kCalibrationImages> kScenes{{
    {"Akudama Drive · the cell and its shaft of light",
     "The shaft of light through the door stays bright with a soft pink glow; the bars and the "
     "walls of the cell stay a dark, readable grey blue, and the floor at the bottom stays "
     "almost black. Bars lost in black: lower Deep blacks. Shaft spreading into a white flare: "
     "lower Middle gray. Greyish, washed-out cell: raise Deep blacks."},
    {"Weathering with You · Hina in the half-light",
     "Hina's face, lit from the side, keeps a natural, warm skin; her hair stays a dark brown "
     "with visible strands, and the room behind her stays a deep blue grey without turning "
     "black. Face too dark to read: raise Middle gray. Pale, glowing face: lower it. Greyish "
     "background: raise Deep blacks."},
    {"Weathering with You · Hina in the rain",
     "Hina's skin stays a natural pinkish beige, neither pale nor orange; her white top keeps "
     "its folds, and the wet paving stones keep their reflections. Pale, washed-out skin or a "
     "glaring top: lower Middle gray. Orange or reddened skin: lower Colors."},
    {"Violet Evergarden · the town under the sun",
     "The sun shines but stays a small disc; the sky stays a deep blue that lightens toward the "
     "horizon, the clouds stay white with soft shading, and the roofs of the town keep their "
     "colors. Sky burnt into white around the sun: lower Middle gray. Sea and roofs crushed into"
     " black: lower Deep blacks. Electric blue sky: lower Colors."},
    {"Akudama Drive · the hall of lamps",
     "The lamps on the pillars and the ceiling glow, but the gold of the pillars and the walls "
     "between them stay readable, and the small figures on the floor keep their colors. Lamps "
     "bleeding into the walls, or a dazzling ceiling: lower Middle gray. Gold turning orange: "
     "lower Colors."},
    {"Akudama Drive · spotlights in the fog",
     "The spotlights stay small, sharp points; the fog stays a smooth pink and violet, without "
     "banding, and the dark shapes in front keep their edges. Fog turning a flat fuchsia: lower "
     "Colors. Spotlights spreading into large white blobs: lower Middle gray."},
    {"Frieren · the heroes in the forest",
     "The foliage stays a natural green, with patches of light and shade; the white cloak keeps "
     "its folds, and the faces keep a natural skin. Fluorescent green leaves: lower Colors. "
     "Cloak turned into a flat white: lower Middle gray. Shade under the trees lost in black: "
     "lower Deep blacks."},
    {"Violet Evergarden · Violet in white",
     "The sheets stay white with grey folds, Violet's skin stays pale but warm, and her blue "
     "eyes stay the most vivid color. Sheets turning into a glaring white: lower Middle gray. "
     "Grey, dull sheets: raise it. Greyish skin: raise Colors."},
    {"Weathering with You · the fireworks",
     "The sparks stay sharp gold and green streaks, the brightest ones white at their core, over"
     " a dark city. Sparks merging into a glowing haze: lower Middle gray. Neon green or orange "
     "sparks: lower Colors. City lost in black: lower Deep blacks."},
    {"Frieren · the old priest in the shadow",
     "The old man's hair stays a pale beige, his cloak a warm brown with visible folds, and the "
     "brick wall behind him stays readable in the dark. Wall lost in black: lower Deep blacks. "
     "Orange, glowing hair: lower Colors. Face too dark to read: raise Middle gray."},
    {"Your Name · twilight above the clouds",
     "The beam of light and the low sun shine without burning the sea of clouds, which keeps its"
     " gold and pink relief; Taki's face stays warm without turning orange, and his dark jacket "
     "keeps its folds. Clouds burnt into white: lower Middle gray. Orange skin or red sky: lower"
     " Colors. Jacket lost in black: lower Deep blacks."},
}};

}  // namespace

CalibrationImage calibration_shot(int id) {
  const int at = std::clamp(id, 0, kCalibrationImages - 1);
  const Scene& scene = kScenes[static_cast<std::size_t>(at)];
  return {.id = at,
          .title = scene.title,
          .guidance = scene.guidance,
          .at_us = at * kCalibrationSegmentUs + kCalibrationSegmentUs / 2};
}

std::filesystem::path calibration_clip() {
  const char* base = SDL_GetBasePath();
  return base == nullptr ? std::filesystem::path()
                         : std::filesystem::path(base) / "calibration" / "calibration.mkv";
}

}  // namespace ajp::app::play
