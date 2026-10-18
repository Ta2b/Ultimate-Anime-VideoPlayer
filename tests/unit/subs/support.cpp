#include "support.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <iterator>
#include <random>

#include <catch2/catch_test_macros.hpp>

namespace ajp::subs::test {
namespace {

constexpr std::string_view kHeader =
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 640\n"
    "PlayResY: 360\n"
    "WrapStyle: 0\n"
    "ScaledBorderAndShadow: yes\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
    "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
    "MarginL, MarginR, MarginV, Encoding\n"
    "Style: Default,DejaVu Sans,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,"
    "20,20,20,1\n"
    "Style: Top,DejaVu Sans,30,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,"
    "20,20,20,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

std::size_t index(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

}  // namespace

std::string_view reference_script() {
  static const std::string script = std::string(kHeader) +
                                    "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Bas de l'écran\n"
                                    "Dialogue: 0,0:00:01.00,0:00:02.00,Top,,0,0,0,,En haut\n"
                                    "Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,"
                                    "{\\an7\\pos(40,60)\\bord0\\1c&H0000FF&}Rouge\n"
                                    "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,"
                                    "{\\an7\\pos(40,60)\\bord0\\alpha&H80&}Transparent\n"
                                    "Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,"
                                    "{\\an7\\move(40,60,440,60)\\bord0}Glisse\n";
  return script;
}

std::string numbered_script(int lines) {
  std::string script(kHeader);
  for (int i = 0; i < lines; ++i) {
    script += std::format("Dialogue: 0,0:00:{:02}.00,0:00:{:02}.00,Default,,0,0,0,,Ligne {}\n", i, i + 1, i);
  }
  return script;
}

float Canvas::opacity_at(int x, int y) const {
  return opacity[index(x, y, width)];
}

const std::array<float, 3>& Canvas::rgb_at(int x, int y) const {
  return rgb[index(x, y, width)];
}

Box Canvas::covered(float threshold) const {
  Box box{.x0 = width, .y0 = height, .x1 = 0, .y1 = 0};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (opacity_at(x, y) <= threshold) continue;
      box = {std::min(box.x0, x), std::min(box.y0, y), std::max(box.x1, x + 1), std::max(box.y1, y + 1)};
    }
  }
  return box;
}

Canvas draw(const SubtitleFrame& frame) {
  Canvas canvas{.width = frame.width, .height = frame.height, .opacity = {}, .rgb = {}};
  const std::size_t pixels = index(0, frame.height, frame.width);
  canvas.opacity.assign(pixels, 0.0F);
  canvas.rgb.assign(pixels, {0.0F, 0.0F, 0.0F});
  if (frame.atlas == nullptr) return canvas;
  const SubtitleAtlas& atlas = *frame.atlas;
  for (const SubtitleRect& rect : frame.rects) {
    const std::array<float, 3> color{rect.color.r / 255.0F, rect.color.g / 255.0F, rect.color.b / 255.0F};
    for (int y = 0; y < rect.src.height; ++y) {
      for (int x = 0; x < rect.src.width; ++x) {
        const int px = rect.dst_x + x;
        const int py = rect.dst_y + y;
        if (px < 0 || py < 0 || px >= frame.width || py >= frame.height) continue;
        const float coverage = atlas.pixels[index(rect.src.x + x, rect.src.y + y, atlas.width)] / 255.0F;
        const float o = coverage * (rect.color.a / 255.0F);
        float& opacity = canvas.opacity[index(px, py, frame.width)];
        opacity += o * (1.0F - opacity);
        for (std::size_t c = 0; c < 3; ++c) {
          float& channel = canvas.rgb[index(px, py, frame.width)][c];
          channel = channel * (1.0F - o) + color[c] * o;
        }
      }
    }
  }
  return canvas;
}

void check_atlas(const SubtitleFrame& frame) {
  REQUIRE(frame.atlas != nullptr);
  const SubtitleAtlas& atlas = *frame.atlas;
  REQUIRE(atlas.width >= 1);
  REQUIRE(atlas.height >= 1);
  REQUIRE(atlas.width <= kMaxAtlasSide);
  REQUIRE(atlas.height <= kMaxAtlasSide);
  REQUIRE(atlas.pixels.size() == index(0, atlas.height, atlas.width));
  std::vector<bool> inside(atlas.pixels.size(), false);
  bool any_coverage = false;
  for (const SubtitleRect& rect : frame.rects) {
    REQUIRE(rect.src.x >= 0);
    REQUIRE(rect.src.y >= 0);
    REQUIRE(rect.src.x + rect.src.width <= atlas.width);
    REQUIRE(rect.src.y + rect.src.height <= atlas.height);
    for (int y = rect.src.y; y < rect.src.y + rect.src.height; ++y) {
      for (int x = rect.src.x; x < rect.src.x + rect.src.width; ++x) {
        inside[index(x, y, atlas.width)] = true;
        any_coverage = any_coverage || atlas.pixels[index(x, y, atlas.width)] != 0;
      }
    }
  }
  CHECK(any_coverage);
  for (std::size_t i = 0; i < atlas.pixels.size(); ++i) {
    if (!inside[i] && atlas.pixels[i] != 0) FAIL(std::format("pixel {} non nul hors des rectangles", i));
  }
}

std::vector<std::byte> system_font_bytes() {
  for (const char* path :
       {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"}) {
    std::ifstream file(path, std::ios::binary);
    if (!file) continue;
    const std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::vector<std::byte> font(bytes.size());
    std::ranges::transform(bytes, font.begin(), [](char c) { return static_cast<std::byte>(c); });
    return font;
  }
  return {};
}

bool RecordingSink::try_push(SubtitleFrame&& frame) {
  {
    const std::scoped_lock lock(mutex_);
    frames_.push_back(std::move(frame));
  }
  arrived_.notify_all();
  return true;
}

void RecordingSink::refused(const Refusal& refusal) {
  const std::scoped_lock lock(mutex_);
  refusals_.push_back(refusal);
}

bool RecordingSink::wait_for_frames(std::size_t count, std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  return arrived_.wait_for(lock, timeout, [&] { return frames_.size() >= count; });
}

std::size_t RecordingSink::frame_count() const {
  const std::scoped_lock lock(mutex_);
  return frames_.size();
}

std::vector<SubtitleFrame> RecordingSink::take_frames() {
  const std::scoped_lock lock(mutex_);
  return std::exchange(frames_, {});
}

std::vector<Refusal> RecordingSink::refusals() const {
  const std::scoped_lock lock(mutex_);
  return refusals_;
}

TempFile::TempFile(std::string_view name, std::string_view contents)
    : path_(std::filesystem::temp_directory_path() /
            std::format("ajp-subs-{}-{}", std::random_device{}(), name)) {
  std::ofstream file(path_, std::ios::binary);
  file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

TempFile::~TempFile() {
  std::error_code ignored;
  std::filesystem::remove(path_, ignored);
}

}  // namespace ajp::subs::test
