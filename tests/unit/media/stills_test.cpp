// Les images fixes de la dérivation du SPAN FP8 : conversion de couleur, choix des images, extraction d'une
// vidéo réelle (la séquence de l'évaluation, écrite à l'exécution).
#include "media/stills.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clips.h"
#include "media/bench_clip.h"

using namespace ajp::media;
using ajp::ColorInfo;
using ajp::ColorMatrix;
using ajp::ColorRange;
using ajp::DecodedFrame;
using ajp::FrameData;
using ajp::FrameMemory;
using ajp::PixelFormat;
using ajp::RefusalCode;
using ajp::Transfer;

namespace {

// Une image en mémoire hôte, planes données.
class HostFrame final : public FrameData {
 public:
  HostFrame(std::vector<std::vector<std::byte>> planes, std::vector<std::size_t> strides)
      : planes_(std::move(planes)), strides_(std::move(strides)) {}
  FrameMemory memory() const override { return FrameMemory::Host; }
  int plane_count() const override { return static_cast<int>(planes_.size()); }
  std::span<const std::byte> host_plane(int plane) const override {
    return planes_[static_cast<std::size_t>(plane)];
  }
  std::size_t host_stride(int plane) const override { return strides_[static_cast<std::size_t>(plane)]; }
  std::uintptr_t native_plane(int) const override { return 0; }
  std::size_t native_pitch(int) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }

 private:
  std::vector<std::vector<std::byte>> planes_;
  std::vector<std::size_t> strides_;
};

// Une image Yuv420p de 256x8 : le tiers gauche blanc (Y 235), le milieu noir (Y 16), le reste rouge BT.709.
DecodedFrame yuv420p(ColorInfo color, int width = 256, int height = 8) {
  const auto w = static_cast<std::size_t>(width), h = static_cast<std::size_t>(height);
  std::vector<std::byte> y(w * h), u(w * h / 4, static_cast<std::byte>(128)),
      v(w * h / 4, static_cast<std::byte>(128));
  for (std::size_t row = 0; row < h; ++row)
    for (std::size_t col = 0; col < w; ++col) {
      const std::size_t zone = col * 3 / w;
      y[row * w + col] = static_cast<std::byte>(zone == 0 ? 235 : zone == 1 ? 16 : 63);
      if (zone == 2) {
        u[(row / 2) * (w / 2) + col / 2] = static_cast<std::byte>(102);
        v[(row / 2) * (w / 2) + col / 2] = static_cast<std::byte>(240);
      }
    }
  DecodedFrame frame;
  frame.width = width;
  frame.height = height;
  frame.format = PixelFormat::Yuv420p;
  frame.color = color;
  frame.pts_us = 1234;
  frame.data = std::make_shared<HostFrame>(std::vector<std::vector<std::byte>>{y, u, v},
                                           std::vector<std::size_t>{w, w / 2, w / 2});
  return frame;
}

StillsSpec narrow() {
  return StillsSpec{.max_width = 256, .max_height = 8, .align_width = 128, .align_height = 4};
}

int at(const Still& still, int channel, int x, int y) {
  return still.rgb[(static_cast<std::size_t>(channel) * static_cast<std::size_t>(still.height) +
                    static_cast<std::size_t>(y)) *
                       static_cast<std::size_t>(still.width) +
                   static_cast<std::size_t>(x)];
}

}  // namespace

TEST_CASE("une image décodée devient du RGB 8 bits planaire", "[media][stills]") {
  const auto still = still_from_frame(yuv420p(ColorInfo{}), narrow());
  REQUIRE(still);
  CHECK(still->width == 256);
  CHECK(still->height == 8);
  CHECK(still->pts_us == 1234);
  CHECK(still->rgb.size() == std::size_t{3} * 256 * 8);
  CHECK(at(*still, 0, 10, 3) == 255);  // blanc en plage limitée : Y 235 -> 1
  CHECK(at(*still, 1, 10, 3) == 255);
  CHECK(at(*still, 2, 10, 3) == 255);
  CHECK(at(*still, 0, 128, 3) == 0);  // noir : Y 16 -> 0
  CHECK(at(*still, 1, 128, 3) == 0);
  const int red = at(*still, 0, 250, 3), green = at(*still, 1, 250, 3), blue = at(*still, 2, 250, 3);
  CHECK(red > 200);  // un rouge franc
  CHECK(green < 40);
  CHECK(blue < 40);
  CHECK(still->mean > 0.2);
  CHECK(still->spread > 0.2);
}

TEST_CASE("la découpe est centrée et alignée sur ce que les noyaux prennent", "[media][stills]") {
  StillsSpec spec = narrow();
  spec.max_width = 128;
  const auto cropped = still_from_frame(yuv420p(ColorInfo{}), spec);
  REQUIRE(cropped);
  CHECK(cropped->width == 128);
  CHECK(at(*cropped, 0, 0, 0) == 255);  // colonne 64 de l'image : encore dans la zone blanche (0 à 84)
  CHECK(at(*cropped, 0, 64, 0) == 0);   // colonne 128 : le milieu de l'image, la zone noire (85 à 169)

  const auto small = still_from_frame(yuv420p(ColorInfo{}, 64, 8), narrow());  // 64 < 128
  REQUIRE_FALSE(small);
  CHECK(small.error().code == RefusalCode::FormatUnsupported);
}

TEST_CASE("une vidéo HDR ou d'un format non pris est refusée avec une raison", "[media][stills]") {
  ColorInfo pq;
  pq.transfer = Transfer::Pq;
  const auto hdr = still_from_frame(yuv420p(pq), narrow());
  REQUIRE_FALSE(hdr);
  CHECK(hdr.error().detail.starts_with("vidéo HDR"));

  DecodedFrame gray = yuv420p(ColorInfo{});
  gray.format = PixelFormat::Rgba16Unorm;
  CHECK_FALSE(still_from_frame(gray, narrow()));

  ColorInfo rgb;
  rgb.matrix = ColorMatrix::Rgb;
  CHECK_FALSE(still_from_frame(yuv420p(rgb), narrow()));
}

TEST_CASE("le choix écarte les noires et les aplats, garde les plus contrastées dans l'ordre du temps",
          "[media][stills]") {
  const auto make = [](std::int64_t pts, double mean, double spread) {
    Still still;
    still.pts_us = pts;
    still.mean = mean;
    still.spread = spread;
    return still;
  };
  StillsSpec spec;
  spec.count = 3;
  spec.minimum = 3;
  std::vector<Still> all{make(10, 0.4, 0.10), make(20, 0.01, 0.30),  // noire
                         make(30, 0.4, 0.01),                        // aplat
                         make(40, 0.5, 0.20), make(50, 0.5, 0.05),  make(60, 0.4, 0.30), make(70, 0.5, 0.15)};
  const std::vector<std::size_t> picked = pick_stills(all, spec);
  REQUIRE(picked.size() == 3);
  CHECK(picked[0] == 3);  // pts 40, spread 0,20
  CHECK(picked[1] == 5);  // pts 60, spread 0,30
  CHECK(picked[2] == 6);  // pts 70, spread 0,15
  spec.minimum = 6;       // 5 utilisables : pas assez
  CHECK(pick_stills(all, spec).empty());
}

TEST_CASE("douze images sont tirées d'une vidéo, et rien n'est gardé", "[media][stills]") {
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp-stills-test.mkv";
  std::filesystem::remove(file);
  const auto written =
      write_bench_clip(file, BenchClipSpec{.width = 640, .height = 360, .fps = 24, .seconds = 12});
  INFO((written ? std::string() : written.error().detail));
  REQUIRE(written);
  float last = -1.0F;
  const auto stills = extract_stills(file, StillsSpec{}, std::stop_token{}, [&](float f) { last = f; });
  INFO((stills ? std::string() : stills.error().detail));
  REQUIRE(stills);
  CHECK(stills->size() >= 8);
  CHECK(stills->size() <= 12);
  CHECK(last == 1.0F);
  for (std::size_t i = 0; i < stills->size(); ++i) {
    const Still& still = (*stills)[i];
    CHECK(still.width == 640);
    CHECK(still.height == 360);
    CHECK(still.mean >= 0.06);
    CHECK(still.spread >= 0.03);
    if (i > 0) CHECK(still.pts_us >= (*stills)[i - 1].pts_us);
    CHECK(still.pts_us > 800'000);     // pas le début
    CHECK(still.pts_us < 11'200'000);  // pas la fin
  }

  std::stop_source stop;
  stop.request_stop();
  const auto cancelled = extract_stills(file, StillsSpec{}, stop.get_token());
  REQUIRE_FALSE(cancelled);
  CHECK(cancelled.error().code == RefusalCode::Cancelled);
  std::filesystem::remove(file);
}

TEST_CASE("une vidéo HDR est refusée clairement", "[media][stills]") {
  const auto stills = extract_stills(test::require_clip(test::hdr_clip()), StillsSpec{}, std::stop_token{});
  REQUIRE_FALSE(stills);
  CHECK(stills.error().detail.find("HDR") != std::string::npos);
}
