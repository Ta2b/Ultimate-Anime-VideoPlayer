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

// Les étapes : sombre (gris moyen, noirs profonds, couleurs), sombre avec éclat (les mêmes, à part), puis
// claire (gris moyen, noirs profonds et réduction des couleurs d'une scène claire).
const std::array<Scene, kCalibrationImages> kScenes{{
    {"Re:Zero · the staircase in the dark",
     "The stone blocks of the walls and every step of the staircase stay visible, and the area under "
     "the stairs on the left stays almost black. Ram's black tights stand out from her boots, and the "
     "small figure sitting at the top left can be made out. Steps or wall joints lost in black: lower "
     "Deep blacks or raise Mid gray. Grey, washed-out walls: raise Deep blacks."},
    {"Re:Zero · Ram in the shadow",
     "Ram's hair stays a soft, dusty pink, neither bright nor flat; her red eye, its white highlight "
     "and her eyelashes are sharp. The dark background on the right stays almost black, without "
     "turning grey. Face too dark to read: raise Mid gray. Greyish background or glowing hair: lower "
     "it. Hair turning fuchsia: lower Colors."},
    {"Re:Zero · the green magic circle",
     "The swirl stays a bright green with lighter edges, and its glow on the steps on the right is a "
     "soft, dark green. Ram's blouse stays white with a slight green tint, her skirt a pale turquoise."
     " The bricks of the wall on the left can still be seen in the dark. Neon or flat green: lower "
     "Colors, glow scene. Bricks lost in black: lower Deep blacks, glow scene."},
    {"Re:Zero · Rem's glowing horn",
     "The horn glows white at its core, with a violet halo that stays around the head. Rem's blue hair"
     " keeps its strands, her collar stays a soft pink, and the cloak she lies on stays a dark reddish"
     " brown with visible folds. The wall behind stays a dark grey blue. Halo spreading as a flat "
     "magenta, or a glaring pink collar: lower Colors, glow scene. Folds of the cloak lost in black: "
     "lower Deep blacks, glow scene, or raise Middle gray, glow scene. Glaring white core: lower "
     "Middle gray, glow scene."},
    {"Re:Zero · the dragon under the blue sky",
     "The sky stays a deep blue with light, wispy clouds you can tell apart; the white pillar keeps "
     "its grooves and the grey shading on its right side; the dragon's scales and the dark gaps of its"
     " wing stay visible. Pillar turned into a flat white bar, or a dazzling sky: lower Middle gray, "
     "bright scene. Dark parts of the dragon lost in black: lower Deep blacks, bright scene."},
    {"Slime · the valley and the river",
     "The river shines but keeps its banks and its small ripples; the leaves of the tree stay readable"
     " against the light, and the trunk and its shadow on the grass stay a dark brown, not black. The "
     "sky over the mountains stays a pastel pink. River burnt into a white strip: lower Middle gray, "
     "bright scene. Trunk and foliage crushed into black: lower Deep blacks, bright scene."},
    {"Slime · the pink sky",
     "The clouds stay a pastel pink against a lavender sky, the faces keep a natural, warm skin, and "
     "the man's tie is the only real red. Clouds turning red or orange, or reddened faces: raise "
     "Colors cut, bright scene. The man's dark collar and the shadows on his face keep their detail: "
     "if they go black, lower Deep blacks, bright scene."},
    {"Re:Zero · Emilia in daylight",
     "Emilia's hair stays white with lilac shading, the blush on her cheeks stays a light pink, and "
     "the paving stones keep their joints. The violet collar and the gold brooch stay rich without "
     "glowing. Hair turned flat white or dazzling stones: lower Middle gray, bright scene. Cheeks "
     "turning red: raise Colors cut, bright scene."},
}};

}  // namespace

// Le premier plan de chaque étape, et celui d'après la dernière.
constexpr std::array<int, kCalibrationSteps + 1> kFirst{0, 2, 4, kCalibrationImages};

int calibration_count(CalibrationStep step) {
  const auto i = static_cast<std::size_t>(step);
  return kFirst[i + 1] - kFirst[i];
}

CalibrationImage calibration_image(CalibrationStep step, int index) {
  const int id = kFirst[static_cast<std::size_t>(step)] + std::clamp(index, 0, calibration_count(step) - 1);
  const Scene& scene = kScenes[static_cast<std::size_t>(id)];
  return {.id = id,
          .title = scene.title,
          .guidance = scene.guidance,
          .at_us = id * kCalibrationSegmentUs + kCalibrationSegmentUs / 2};
}

std::filesystem::path calibration_clip() {
  const char* base = SDL_GetBasePath();
  return base == nullptr ? std::filesystem::path()
                         : std::filesystem::path(base) / "calibration" / "calibration.mkv";
}

}  // namespace ajp::app::play
