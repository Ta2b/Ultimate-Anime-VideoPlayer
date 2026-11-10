// P3.25 : une image entrelacée (deux champs décalés, des peignes une ligne sur deux) sort de bwdif sans
// peigne, en 1080i comme en 480i, avec son horodatage.
#include "media/deinterlace.h"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
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

namespace {

// Une image progressive : le carré à la position de l'image `index`, sur toutes ses lignes.
void paint(AVFrame& f, int top_index, int bottom_index) {
  for (int y = 0; y < f.height; ++y) {
    const int index = y % 2 == 0 ? top_index : bottom_index;
    const int left = (index * 16) % (f.width / 2);
    for (int x = 0; x < f.width; ++x)
      f.data[0][y * f.linesize[0] + x] = x >= left && x < left + 64 ? 235 : 16;
  }
  for (int plane = 1; plane < 3; ++plane)
    for (int y = 0; y < f.height / 2; ++y)
      for (int x = 0; x < f.width / 2; ++x) f.data[plane][y * f.linesize[plane] + x] = 128;
}

FramePtr woven(int top_index, int bottom_index, int n) {
  FramePtr f(av_frame_alloc());
  f->width = 720;
  f->height = 480;
  f->format = AV_PIX_FMT_YUV420P;
  REQUIRE(av_frame_get_buffer(f.get(), 0) == 0);
  paint(*f, top_index, bottom_index);
  f->flags |= AV_FRAME_FLAG_INTERLACED | AV_FRAME_FLAG_TOP_FIELD_FIRST;
  f->pts = n * 1'001'000LL / 30;  // 29,97 i/s
  f->duration = 1'001'000LL / 30;
  return f;
}

// Le télécinéma 3:2 de `sources` images : champs A A, B B B, C C, D D D… cousus deux à deux (haut, bas).
std::vector<FramePtr> telecined(int sources) {
  std::vector<int> fields;  // l'image source de chaque champ, en alternance haut, bas
  for (int i = 0; i < sources; ++i) fields.insert(fields.end(), i % 2 == 0 ? 2 : 3, i);
  std::vector<FramePtr> frames;
  for (std::size_t f = 0; f + 1 < fields.size(); f += 2)
    frames.push_back(woven(fields[f], fields[f + 1], static_cast<int>(frames.size())));
  return frames;
}

int square_index(const AVFrame& f, int y) {
  for (int x = 0; x < f.width; ++x)
    if (f.data[0][y * f.linesize[0] + x] > 128) return x / 16;
  return -1;
}

std::vector<FramePtr> run(Deinterlacer& deinterlacer, std::vector<FramePtr> frames) {
  std::vector<FramePtr> out;
  for (FramePtr& frame : frames) {
    REQUIRE(deinterlacer.push(std::move(frame)));
    while (true) {
      auto next = deinterlacer.pull();
      REQUIRE(next);
      if (!*next) break;
      out.push_back(std::move(*next));
    }
  }
  REQUIRE(deinterlacer.flush());
  while (true) {
    auto next = deinterlacer.pull();
    REQUIRE(next);
    if (!*next) break;
    out.push_back(std::move(*next));
  }
  return out;
}

}  // namespace

TEST_CASE("Télécinéma 3:2 à 29,97 : les images d'origine, entières et dans l'ordre (A15-02)",
          "[media][deinterlace]") {
  const std::vector<FramePtr> input = telecined(20);  // 50 champs, 25 images entrelacées
  REQUIRE(input.size() == 25);
  Deinterlacer deinterlacer(FieldMode::InverseTelecine);
  std::vector<FramePtr> frames;
  for (const FramePtr& f : input) frames.emplace_back(av_frame_clone(f.get()));
  const std::vector<FramePtr> out = run(deinterlacer, std::move(frames));
  // decimate garde 4 images sur 5 : 20 images, chacune une image source entière, aucune en double.
  REQUIRE(out.size() == 20);
  int previous = -1;
  for (std::size_t i = 0; i < out.size(); ++i) {
    CAPTURE(i);
    const int top = square_index(*out[i], 100);
    CHECK(top == square_index(*out[i], 101));  // les deux champs viennent de la même image
    CHECK(comb(*out[i]) < 20.0);
    CHECK(top > previous);
    previous = top;
    if (i > 0) CHECK(out[i]->pts - out[i - 1]->pts == Catch::Approx(41'708).margin(2));
    CHECK(out[i]->duration == 41'708);
  }
}

TEST_CASE("bwdif rend l'image retenue quand la taille change (A15-04)", "[media][deinterlace]") {
  std::vector<FramePtr> frames;
  for (int i = 0; i < 5; ++i) frames.push_back(combed(720, 480, i));
  for (int i = 5; i < 10; ++i) frames.push_back(combed(1920, 1080, i));
  Deinterlacer deinterlacer;
  const std::vector<FramePtr> out = run(deinterlacer, std::move(frames));
  REQUIRE(out.size() == 10);
  for (std::size_t i = 0; i < out.size(); ++i) CHECK(out[i]->pts == static_cast<std::int64_t>(i) * 40'000);
  CHECK(out[4]->width == 720);
  CHECK(out[5]->width == 1920);
}

TEST_CASE("bwdif garde l'horodatage deviné d'une image sans pts (A15-05)", "[media][deinterlace]") {
  std::vector<FramePtr> frames;
  for (int i = 0; i < 4; ++i) {
    FramePtr f = combed(720, 480, i);
    f->best_effort_timestamp = f->pts;
    f->pts = AV_NOPTS_VALUE;
    frames.push_back(std::move(f));
  }
  Deinterlacer deinterlacer;
  const std::vector<FramePtr> out = run(deinterlacer, std::move(frames));
  REQUIRE(out.size() == 4);
  for (std::size_t i = 0; i < out.size(); ++i) {
    CHECK(out[i]->pts == static_cast<std::int64_t>(i) * 40'000);
    CHECK(out[i]->best_effort_timestamp == out[i]->pts);
  }
}
