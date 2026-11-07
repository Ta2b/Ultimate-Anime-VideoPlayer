// P3.25 : une image entrelacée (deux champs décalés, des peignes une ligne sur deux) sort de bwdif sans
// peigne, en 1080i comme en 480i, avec son horodatage.
#include "media/deinterlace.h"

#include <cstdint>
#include <cstdlib>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

using namespace ajp::media;

namespace {

// Un carré clair qui avance de 16 pixels par image : le champ du bas le montre 8 pixels plus loin.
FramePtr combed(int width, int height, int index) {
  FramePtr f(av_frame_alloc());
  f->width = width;
  f->height = height;
  f->format = AV_PIX_FMT_YUV420P;
  REQUIRE(av_frame_get_buffer(f.get(), 0) == 0);
  for (int y = 0; y < height; ++y) {
    const int left = (index * 16 + (y % 2) * 8) % (width / 2);
    for (int x = 0; x < width; ++x)
      f->data[0][y * f->linesize[0] + x] = x >= left && x < left + 64 ? 235 : 16;
  }
  for (int plane = 1; plane < 3; ++plane)
    for (int y = 0; y < height / 2; ++y)
      for (int x = 0; x < width / 2; ++x) f->data[plane][y * f->linesize[plane] + x] = 128;
  f->flags |= AV_FRAME_FLAG_INTERLACED | AV_FRAME_FLAG_TOP_FIELD_FIRST;
  f->pts = index * 40'000;
  return f;
}

// La pire différence entre deux lignes voisines, rapportée au carré : les peignes la portent au-delà de 20.
double comb(const AVFrame& f) {
  double worst = 0.0;
  for (int y = 1; y + 1 < f.height; y += 2) {
    long sum = 0;
    for (int x = 0; x < f.width; ++x)
      sum += std::abs(f.data[0][y * f.linesize[0] + x] - f.data[0][(y - 1) * f.linesize[0] + x]);
    worst = std::max(worst, static_cast<double>(sum) / 64.0);  // rapporté à la largeur du carré
  }
  return worst;
}

}  // namespace

TEST_CASE("bwdif retire les peignes d'une image entrelacée, 1080i et 480i", "[media][deinterlace]") {
  for (const auto& [width, height] : {std::pair{1920, 1080}, std::pair{720, 480}}) {
    CAPTURE(width, height);
    REQUIRE(comb(*combed(width, height, 3)) > 20.0);
    Deinterlacer deinterlacer;
    int out = 0;
    for (int i = 0; i < 6; ++i) {
      REQUIRE(deinterlacer.push(combed(width, height, i)));
      auto frame = deinterlacer.pull();
      REQUIRE(frame);
      if (!*frame) continue;
      CHECK((*frame)->pts == out * 40'000);
      CHECK(comb(**frame) < 20.0);
      ++out;
    }
    CHECK(deinterlacer.active());
    REQUIRE(deinterlacer.flush());
    while (true) {
      auto frame = deinterlacer.pull();
      REQUIRE(frame);
      if (!*frame) break;
      ++out;
    }
    CHECK(out == 6);
  }
}
