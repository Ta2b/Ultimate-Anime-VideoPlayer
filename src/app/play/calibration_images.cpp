#include "app/play/calibration_images.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "app/play/calibration_data.h"
#include "core/log.h"
#include "media/picture.h"

namespace ajp::app::play {

namespace {

struct Scene {
  const std::string_view* bytes;
  std::string_view title;
  std::string_view guidance;
};

// Les étapes : sombre (gris moyen, noirs profonds, couleurs), puis claire (gris moyen, noirs profonds et
// réduction des couleurs d'une scène claire).
const std::array<Scene, 2 * kCalibrationPerStep> kScenes{{
    {&kCalibrationDark1, "Re:Zero · the staircase in the dark",
     "The stone blocks of the walls and every step of the staircase stay visible, and the area under "
     "the stairs on the left stays almost black. Ram's black tights stand out from her boots, and the "
     "small figure sitting at the top left can be made out. Steps or wall joints lost in black: lower "
     "Deep blacks or raise Mid gray. Grey, washed-out walls: raise Deep blacks."},
    {&kCalibrationDark2, "Re:Zero · Ram in the shadow",
     "Ram's hair stays a soft, dusty pink, neither bright nor flat; her red eye, its white highlight "
     "and her eyelashes are sharp. The dark background on the right stays almost black, without "
     "turning grey. Face too dark to read: raise Mid gray. Greyish background or glowing hair: lower "
     "it. Hair turning fuchsia: lower Colors."},
    {&kCalibrationDark3, "Re:Zero · the green magic circle",
     "The swirl stays a bright green with lighter edges, and its glow on the steps on the right is a "
     "soft, dark green. Ram's blouse stays white with a slight green tint, her skirt a pale turquoise."
     " The bricks of the wall on the left can still be seen in the dark. Neon or flat green: lower "
     "Colors. Bricks lost in black: lower Deep blacks."},
    {&kCalibrationDark4, "Re:Zero · Rem's glowing horn",
     "The horn glows white at its core, with a violet halo that stays around the head. Rem's blue hair"
     " keeps its strands, her collar stays a soft pink, and the cloak she lies on stays a dark reddish"
     " brown with visible folds. The wall behind stays a dark grey blue. Halo spreading as a flat "
     "magenta, or a glaring pink collar: lower Colors. Folds of the cloak lost in black: lower Deep "
     "blacks or raise Mid gray."},
    {&kCalibrationBright1, "Re:Zero · the dragon under the blue sky",
     "The sky stays a deep blue with light, wispy clouds you can tell apart; the white pillar keeps "
     "its grooves and the grey shading on its right side; the dragon's scales and the dark gaps of its"
     " wing stay visible. Pillar turned into a flat white bar, or a dazzling sky: lower Middle gray, "
     "bright scene. Dark parts of the dragon lost in black: lower Deep blacks, bright scene."},
    {&kCalibrationBright2, "Slime · the valley and the river",
     "The river shines but keeps its banks and its small ripples; the leaves of the tree stay readable"
     " against the light, and the trunk and its shadow on the grass stay a dark brown, not black. The "
     "sky over the mountains stays a pastel pink. River burnt into a white strip: lower Middle gray, "
     "bright scene. Trunk and foliage crushed into black: lower Deep blacks, bright scene."},
    {&kCalibrationBright3, "Slime · the pink sky",
     "The clouds stay a pastel pink against a lavender sky, the faces keep a natural, warm skin, and "
     "the man's tie is the only real red. Clouds turning red or orange, or reddened faces: raise "
     "Colors cut, bright scene. The man's dark collar and the shadows on his face keep their detail: "
     "if they go black, lower Deep blacks, bright scene."},
    {&kCalibrationBright4, "Re:Zero · Emilia in daylight",
     "Emilia's hair stays white with lilac shading, the blush on her cheeks stays a light pink, and "
     "the paving stones keep their joints. The violet collar and the gold brooch stay rich without "
     "glowing. Hair turned flat white or dazzling stones: lower Middle gray, bright scene. Cheeks "
     "turning red: raise Colors cut, bright scene."},
}};

}  // namespace

CalibrationImage calibration_image(bool dark, int index) {
  static int decoded_id = -1;
  static media::Picture decoded;
  const int id = (dark ? 0 : kCalibrationPerStep) + std::clamp(index, 0, kCalibrationPerStep - 1);
  const Scene& scene = kScenes[static_cast<std::size_t>(id)];
  if (decoded_id != id) {
    decoded_id = id;
    auto picture = media::decode_picture(*scene.bytes);
    if (!picture) log_channel("app")->warn("image de calibration {} : {}", id, picture.error().detail);
    decoded = picture ? std::move(*picture) : media::Picture{};
  }
  return {.id = id,
          .title = scene.title,
          .guidance = scene.guidance,
          .width = decoded.width,
          .height = decoded.height,
          .rgba = decoded.rgba};
}

}  // namespace ajp::app::play
