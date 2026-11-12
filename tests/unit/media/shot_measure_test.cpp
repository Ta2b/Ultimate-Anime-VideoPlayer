// La pré-analyse des plans au-delà de la découpe (audit 42) : la mesure d'une image décodée (formats,
// matrices, plages, transferts), les décisions de saut et de rythme de l'analyse, la chronologie qui garde la
// part non recouverte d'un voisin, le limiteur de flashs sur une source PQ, et le cache par variante.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#include "clips.h"
#include "kernels/ref/transfer.h"
#include "media/frame_data.h"
#include "media/shot_cache.h"
#include "media/shot_cutter.h"
#include "media/shot_probe.h"
#include "media/shot_probe_steps.h"
#include "media/shot_timeline.h"

using namespace ajp::media;
using Catch::Approx;

namespace {

// Les codes Y'CbCr d'une couleur R'G'B' (0 à 1) dans une matrice et une plage, à `bits` bits.
struct Codes {
  int y = 0, cb = 0, cr = 0;
};
Codes codes_of(std::array<double, 3> rgb, double kr, double kb, bool full, int bits) {
  const double y = (kr * rgb[0]) + ((1.0 - kr - kb) * rgb[1]) + (kb * rgb[2]);
  const double cb = (rgb[2] - y) / (2.0 * (1.0 - kb));
  const double cr = (rgb[0] - y) / (2.0 * (1.0 - kr));
  const double scale = std::ldexp(1.0, bits - 8);
  const double y_off = full ? 0.0 : 16.0;
  const double y_range = full ? 255.0 : 219.0;
  const double c_range = full ? 255.0 : 224.0;
  return {.y = static_cast<int>(std::lround((y_off + (y * y_range)) * scale)),
          .cb = static_cast<int>(std::lround((128.0 + (cb * c_range)) * scale)),
          .cr = static_cast<int>(std::lround((128.0 + (cr * c_range)) * scale))};
}

// Une image hôte uniforme de 256 × 144 au format planaire `format` (8 à 16 bits), ou NV12.
FramePtr uniform(AVPixelFormat format, int bits, Codes c) {
  FramePtr f(av_frame_alloc());
  f->width = 256;
  f->height = 144;
  f->format = format;
  REQUIRE(av_frame_get_buffer(f.get(), 0) == 0);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  const bool nv12 = format == AV_PIX_FMT_NV12;
  for (int plane = 0; plane < (nv12 ? 2 : 3); ++plane) {
    const int w = plane == 0 ? 256 : AV_CEIL_RSHIFT(256, desc->log2_chroma_w);
    const int h = plane == 0 ? 144 : AV_CEIL_RSHIFT(144, desc->log2_chroma_h);
    const int value = plane == 0 ? c.y : plane == 1 ? c.cb : c.cr;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        std::uint8_t* row = f->data[plane] + (static_cast<std::ptrdiff_t>(y) * f->linesize[plane]);
        if (nv12 && plane == 1) {
          row[2 * x] = static_cast<std::uint8_t>(c.cb);
          row[(2 * x) + 1] = static_cast<std::uint8_t>(c.cr);
        } else if (bits > 8) {
          reinterpret_cast<std::uint16_t*>(row)[x] = static_cast<std::uint16_t>(value);
        } else {
          row[x] = static_cast<std::uint8_t>(value);
        }
      }
  }
  return f;
}

ShotFrame measured(AVPixelFormat format, int bits, Codes c, ajp::ColorInfo color, bool full_host = false) {
  auto decoded = to_decoded_frame(uniform(format, bits, c), 0, 40'000, {}, full_host);
  REQUIRE(decoded);
  decoded->color = color;
  auto frame = measure_frame(*decoded);
  REQUIRE(frame);
  return *frame;
}

}  // namespace

// C42-11 : la mesure de l'hôte et le noyau GPU partagent un contrat ; chaque format que l'analyse lit (NV12,
// 4:2:0 8 et 10 bits, P016 d'un 12 bits, 4:4:4 16 bits) rend le même gris.
TEST_CASE("plans : chaque format décodé mesure le même gris", "[media][shots]") {
  struct Case {
    AVPixelFormat format;
    int bits;
    bool full_host;
    ajp::PixelFormat expected;
  };
  const Case c = GENERATE(Case{AV_PIX_FMT_NV12, 8, false, ajp::PixelFormat::Nv12},
                          Case{AV_PIX_FMT_YUV420P, 8, false, ajp::PixelFormat::Yuv420p},
                          Case{AV_PIX_FMT_YUV420P10LE, 10, false, ajp::PixelFormat::Yuv420p10},
                          Case{AV_PIX_FMT_YUV420P12LE, 12, true, ajp::PixelFormat::P016},
                          Case{AV_PIX_FMT_YUV444P16LE, 16, true, ajp::PixelFormat::Yuv444p16});
  CAPTURE(av_get_pix_fmt_name(c.format));
  auto decoded =
      to_decoded_frame(uniform(c.format, c.bits, codes_of({0.5, 0.5, 0.5}, 0.2126, 0.0722, false, c.bits)), 0,
                       40'000, {}, c.full_host);
  REQUIRE(decoded);
  CHECK(decoded->format == c.expected);
  const ShotFrame f =
      measured(c.format, c.bits, codes_of({0.5, 0.5, 0.5}, 0.2126, 0.0722, false, c.bits), {}, c.full_host);
  CHECK(f.mean == Approx(0.5).margin(0.01));
  CHECK(f.chroma < 0.01F);
  // 0,5 en gamma 2,4 au blanc de 203 cd/m².
  CHECK(f.light.apl_nits == Approx(203.0 * std::pow(0.5, 2.4)).epsilon(0.03));
}

// C42-11 : la matrice et la plage de la source sont suivies : un rouge pur reste un rouge pur.
TEST_CASE("plans : la mesure suit la matrice et la plage de la source", "[media][shots]") {
  struct Case {
    ajp::ColorMatrix matrix;
    double kr, kb;
    bool full;
  };
  const Case c = GENERATE(Case{ajp::ColorMatrix::Bt601, 0.299, 0.114, false},
                          Case{ajp::ColorMatrix::Bt709, 0.2126, 0.0722, false},
                          Case{ajp::ColorMatrix::Bt2020Ncl, 0.2627, 0.0593, false},
                          Case{ajp::ColorMatrix::Bt709, 0.2126, 0.0722, true});
  CAPTURE(static_cast<int>(c.matrix), c.full);
  const ajp::ColorInfo color{.matrix = c.matrix,
                             .range = c.full ? ajp::ColorRange::Full : ajp::ColorRange::Limited};
  const ShotFrame red =
      measured(AV_PIX_FMT_YUV420P10LE, 10, codes_of({1.0, 0.0, 0.0}, c.kr, c.kb, c.full, 10), color);
  CHECK(red.chroma == Approx(1.0).margin(0.02));  // max − min des canaux : 1 pour un rouge pur
  CHECK(red.blue < 0.02F);
  const ShotFrame gray =
      measured(AV_PIX_FMT_YUV420P10LE, 10, codes_of({0.5, 0.5, 0.5}, c.kr, c.kb, c.full, 10), color);
  CHECK(gray.mean == Approx(0.5).margin(0.01));
}

// C42-11 : une source PQ se mesure en ST 2084, pas en gamma.
TEST_CASE("plans : une source PQ se mesure en ST 2084", "[media][shots]") {
  const ajp::ColorInfo pq{.matrix = ajp::ColorMatrix::Bt2020Ncl, .transfer = ajp::Transfer::Pq};
  const ShotFrame f =
      measured(AV_PIX_FMT_YUV420P10LE, 10, codes_of({0.5, 0.5, 0.5}, 0.2627, 0.0593, false, 10), pq);
  CHECK(f.transfer == ajp::kernels::FrameTransfer::Pq);
  CHECK(f.light.apl_nits == Approx(ajp::kernels::ref::pq_eotf(0.5F)).epsilon(0.03));
}

// C42-04 : le limiteur de flashs lit les zones d'une source PQ par ST 2084. En gamma 2,4, une zone claire de
// PQ 0,58 contre un reste à 0,38 (201,7 et 26,0 cd/m² vrais) donnait 134 et 48,6.
TEST_CASE("plans : les zones d'une source PQ gardent leur écart réel (C42-04)", "[media][shots]") {
  ShotFrame frame;
  frame.transfer = ajp::kernels::FrameTransfer::Pq;
  frame.grid.fill(0.38F);
  frame.grid.at(0) = frame.grid.at(1) = frame.grid.at(4) = frame.grid.at(5) = 0.58F;  // un quart clair
  const float bright = ajp::kernels::ref::pq_eotf(0.58F);
  const float dark = ajp::kernels::ref::pq_eotf(0.38F);
  frame.light.apl_nits = (bright + (3.0F * dark)) / 4.0F;
  const auto zones = zone_nits_of(frame);
  CHECK(zones.at(0) == Approx(bright).epsilon(0.01));
  CHECK(zones.at(15) == Approx(dark).epsilon(0.01));
  // Une source SDR garde le gamma 2,4.
  frame.transfer = ajp::kernels::FrameTransfer::Sdr;
  const auto sdr = zone_nits_of(frame);
  CHECK(sdr.at(0) / sdr.at(15) == Approx(std::pow(0.58 / 0.38, 2.4)).epsilon(0.01));
}

// C42-03 : un plan entier qui ne recouvre qu'une part d'un voisin connu lui laisse le reste.
TEST_CASE("plans : un plan entier ne fait pas de trou chez son voisin (C42-03)", "[media][shots]") {
  ShotTimeline timeline;
  timeline.add(Shot{.start_us = 0, .end_us = 90'000'000});
  timeline.add(Shot{.start_us = 100'000'000, .end_us = 200'000'000, .cut_before = false});
  timeline.add(Shot{.start_us = 90'000'000, .end_us = 150'000'000});
  REQUIRE(timeline.at(170'000'000));
  CHECK(timeline.at(170'000'000)->start_us == 150'000'000);
  CHECK_FALSE(timeline.at(170'000'000)->cut_before);
  CHECK(timeline.end_us() == 200'000'000);
  CHECK(timeline.first_gap_us() == 200'000'000);
  // Un plan entier au milieu d'un voisin plus long : les deux bouts restent.
  ShotTimeline middle;
  middle.add(Shot{.start_us = 0, .end_us = 100'000'000});
  middle.add(Shot{.start_us = 40'000'000, .end_us = 60'000'000});
  CHECK(middle.size() == 3);
  CHECK(middle.at(20'000'000)->end_us == 40'000'000);
  CHECK(middle.at(80'000'000)->start_us == 60'000'000);
  CHECK(middle.at(80'000'000)->serial != middle.at(20'000'000)->serial);
  CHECK(middle.first_gap_us() == 100'000'000);
}

// C42-09 : une séquence de flashs qui s'allonge ne change pas la version des plans, et les plans sont
// partagés tant qu'ils ne changent pas.
TEST_CASE("plans : les flashs ne font pas recopier les plans (C42-09)", "[media][shots]") {
  ShotTimeline timeline;
  timeline.add(Shot{.start_us = 0, .end_us = 1'000'000});
  const auto shots = timeline.shots_version();
  const auto all = timeline.version();
  const auto snapshot = timeline.shots_snapshot();
  timeline.add_flash({.from_us = 100'000, .until_us = 200'000});
  timeline.add_flash({.from_us = 100'000, .until_us = 300'000});
  CHECK(timeline.version() != all);
  CHECK(timeline.shots_version() == shots);
  CHECK(timeline.shots_snapshot() == snapshot);
  timeline.add(Shot{.start_us = 1'000'000, .end_us = 2'000'000});
  CHECK(timeline.shots_version() != shots);
  REQUIRE(timeline.shots_snapshot()->size() == 2);
  CHECK(snapshot->size() == 1);  // l'ancien instantané ne bouge pas sous son lecteur
}

// C42-11 : où l'analyse repart après un saut de la lecture.
TEST_CASE("plans : le départ d'une analyse après un saut", "[media][shots]") {
  constexpr std::int64_t kSkip = 5'000'000;
  ShotTimeline timeline;
  timeline.add(Shot{.start_us = 0, .end_us = 60'000'000});
  timeline.add(Shot{.start_us = 60'000'000, .end_us = 120'000'000});
  // Une cible inconnue : l'analyse part d'elle.
  CHECK(restart_for_seek(timeline, 200'000'000, 120'000'000, 600'000'000, kSkip) == 200'000'000);
  // Une cible connue devant l'analyse : elle part du bout de la zone connue.
  CHECK(restart_for_seek(timeline, 30'000'000, 10'000'000, 600'000'000, kSkip) == 120'000'000);
  // Une cible connue derrière l'analyse : elle continue.
  CHECK_FALSE(restart_for_seek(timeline, 30'000'000, 125'000'000, 600'000'000, kSkip));
  // Connue jusqu'à la fin du fichier : rien.
  CHECK_FALSE(restart_for_seek(timeline, 30'000'000, 10'000'000, 120'200'000, kSkip));
  // À moins de skip_known_us du bout connu : l'analyse part de la cible.
  CHECK(restart_for_seek(timeline, 117'000'000, 10'000'000, 600'000'000, kSkip) == 117'000'000);
}

// C42-01 : à pleine vitesse jusqu'à l'avance libre, puis au rythme demandé ; de même derrière la lecture.
TEST_CASE("plans : le rythme de l'analyse au-delà de son avance libre (C42-01)", "[media][shots]") {
  ProbePace pace(4.0, 20'000'000);
  // Avance courte : jamais d'attente, même 100 s de vidéo décodées en une seconde.
  CHECK(pace.delay_us(0, 0, 0) == 0);
  CHECK(pace.delay_us(10'000'000, 10'000'000, 100'000) == 0);
  // La dernière image sous 20 s d'avance pose l'ancre ; au-delà, quatre fois le temps réel.
  CHECK(pace.delay_us(25'000'000, 19'999'999, 1'000'000) == 0);
  CHECK(pace.delay_us(29'000'000, 29'000'000, 1'000'000) == 1'000'000);  // 4 s de vidéo : 1 s
  CHECK(pace.delay_us(29'000'000, 29'000'000, 1'600'000) == 400'000);
  CHECK(pace.delay_us(29'000'000, 29'000'000, 2'500'000) == 0);  // en retard sur le rythme : continuer
  // Un saut en arrière : le rythme repart de la nouvelle image.
  CHECK(pace.delay_us(5'000'000, 25'000'000, 3'000'000) == 0);
  // Un trou comblé loin derrière la lecture : au rythme aussi.
  ProbePace behind(4.0, 20'000'000);
  CHECK(behind.delay_us(0, -600'000'000, 0) == 0);
  CHECK(behind.delay_us(8'000'000, -592'000'000, 0) == 2'000'000);
  // Sans rythme (0) : jamais d'attente.
  ProbePace free(0.0, 20'000'000);
  CHECK(free.delay_us(0, 100'000'000, 0) == 0);
  CHECK(free.delay_us(100'000'000, 100'000'000, 0) == 0);
}

// C42-05 : chaque variante (écran d'un autre pic, visages) a son fichier : l'une n'efface pas l'autre.
TEST_CASE("plans : le cache garde une variante par fichier (C42-05)", "[media][shots]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-shot-variant-test";
  std::filesystem::remove_all(dir);
  const std::string key = "0123456789abcdef";
  const std::array<Shot, 1> first{Shot{.start_us = 0, .end_us = 1'000'000}};
  const std::array<Shot, 1> second{Shot{.start_us = 0, .end_us = 2'000'000}};
  REQUIRE(save_shot_cache(dir, key, first, true, kShotCacheKeep, {}, 680));
  REQUIRE(save_shot_cache(dir, key, second, true, kShotCacheKeep, {}, 1360));
  const auto a = load_shot_cache(dir, key, 680);
  const auto b = load_shot_cache(dir, key, 1360);
  REQUIRE(a);
  REQUIRE(b);
  CHECK(a->shots.at(0).end_us == 1'000'000);
  CHECK(b->shots.at(0).end_us == 2'000'000);
  CHECK_FALSE(load_shot_cache(dir, key, 0));
  std::filesystem::remove_all(dir);
}

// C42-06 : la destruction n'attend que la fin de l'analyse ; l'écriture du cache (on_stop) finit sur son
// thread, confié à abandon, sans retenir la fenêtre.
TEST_CASE("plans : la fermeture n'attend pas l'écriture du cache (C42-06)", "[media][shots]") {
  auto writing = std::make_shared<std::atomic<bool>>(false);
  std::jthread abandoned;
  std::function<bool()> abandoned_stopped;
  ShotProbeOptions options;
  options.on_stop = [writing](const ShotTimeline& /*timeline*/, bool /*complete*/) {
    writing->store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  };
  options.abandon = [&](std::jthread thread, std::function<bool()> stopped) {
    abandoned = std::move(thread);
    abandoned_stopped = std::move(stopped);
  };
  options.stop_limit = std::chrono::milliseconds(5000);
  auto probe = ShotProbe::start("inexistant.mkv", std::make_shared<ShotTimeline>(), 0, options);
  for (int i = 0; i < 400 && !writing->load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  REQUIRE(writing->load());
  const auto before = std::chrono::steady_clock::now();
  probe.reset();
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::milliseconds(700));
  REQUIRE(abandoned.joinable());
  abandoned.join();
  CHECK(abandoned_stopped());
}

// C42-02 : une zone connue sautée une première fois l'est encore quand l'analyse y revient en comblant un
// trou plus bas. Avant, le garde du premier saut la faisait décoder en entier (ses plans prenaient de
// nouveaux numéros).
TEST_CASE("plans : une zone connue reste sautée au second passage (C42-02)", "[media][shots]") {
  const std::filesystem::path clip = ajp::media::test::require_clip(ajp::media::test::shots_clip());
  const auto whole = std::make_shared<ShotTimeline>();
  {
    const auto probe = ShotProbe::start(clip, whole, 0, ShotProbeOptions{.pace = 0.0});
    for (int i = 0; i < 3000 && !probe->finished(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(probe->finished());
    REQUIRE_FALSE(probe->failure());
  }
  // Connue : [0 ; 6 s) et [18 s ; fin). L'analyse part de 18 s (une reprise), saute à la fin, puis comble le
  // trou de 6 à 18 s et retrouve la zone connue.
  const auto known = std::make_shared<ShotTimeline>();
  for (const Shot& shot : whole->shots())
    if (shot.start_us < 5'000'000 || shot.start_us > 17'000'000) known->add(shot);
  REQUIRE(known->first_gap_us() < 7'000'000);
  const std::uint64_t serial = known->at(50'000'000)->serial;
  const std::uint64_t first = known->at(20'000'000)->serial;
  const auto probe = ShotProbe::start(clip, known, 18'000'000, ShotProbeOptions{.pace = 0.0});
  for (int i = 0; i < 3000 && !probe->finished(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(probe->finished());
  REQUIRE_FALSE(probe->failure());
  CHECK(known->first_gap_us() == known->end_us());
  CHECK(known->at(50'000'000)->serial == serial);
  CHECK(known->at(20'000'000)->serial == first);
}
