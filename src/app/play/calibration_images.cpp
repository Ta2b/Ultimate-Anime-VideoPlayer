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
// un visage, jour avec un visage, ciel et soleil, scène claire et ses lampes), puis les treize bonus.
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
    {"Re:Zero · the blue dragon up close",
     "The dragon's scales stay a deep, even blue with visible shading, its yellow eye glows "
     "without blooming, and its white teeth stay sharp. Blue banding or a neon, flat blue: "
     "lower Colors. Scales lost in black: lower Deep blacks. Teeth glaring white: lower Middle "
     "gray."},
    {"Re:Zero · Ram on the dark stairs",
     "The stone stairs and the walls stay readable in a dark green grey, Ram's pink hair and "
     "her dress keep their color, and the old man at the back stays visible. Stairs lost in "
     "black: lower Deep blacks. Whole scene too dark: raise Middle gray. Greyish, foggy stone: "
     "raise Deep blacks."},
    {"Re:Zero · Rem and her glowing horn",
     "The horn glows pink and white with a soft halo, Rem's blue hair stays smooth without "
     "bands or blotches, and her face stays pale. Bands or blotches in the blue hair or around "
     "the halo: lower Colors. Halo spreading into a white flare: lower Middle gray. Hood lost "
     "in black: lower Deep blacks."},
    {"Slime · the pink sky",
     "The pink and violet sky keeps its soft clouds and light rays, the two faces stay "
     "natural, and the dark jacket keeps its folds. Neon pink or violet sky: lower Colors. "
     "Clouds burnt into white: lower Middle gray. Jacket lost in black: lower Deep blacks."},
    {"Slime · the valley in the light",
     "The river stays a bright silver ribbon without burning, the grass a warm green and the "
     "tree a deep, readable green. Glaring white river: lower Middle gray. Neon green grass: "
     "lower Colors. Tree lost in black: lower Deep blacks."},
    {"Black Clover · the wall of fire",
     "The wall of fire stays a deep orange with visible flames and windows, the blue lightning "
     "on the left stays a thin, sharp line, and the stone floor stays readable in the dark. "
     "Fire burning into yellow or white: lower Middle gray. Neon blue lightning or flat "
     "orange: lower Colors. Floor lost in black: lower Deep blacks."},
    {"Black Clover · the chain of light",
     "The chain and the magic circle glow a pale blue without bands, the red glove stays a "
     "vivid red, and the ruins behind stay a dark, readable blue green. Bands around the chain "
     "or a neon blue glow: lower Colors. Chain spreading into a white flare: lower Middle "
     "gray. Ruins lost in black: lower Deep blacks."},
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
