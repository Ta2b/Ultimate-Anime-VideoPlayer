// Les horodatages de av_util.h aux bornes (P2.29) : ni soustraction indéfinie ni faux « sans horodatage ».
#include "media/av_util.h"

#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "core/frame.h"

TEST_CASE("av_util : un horodatage qui déborde reste sans horodatage, le décalage sature", "[media]") {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  // 1 s par tic : av_rescale_q déborde et rend INT64_MIN.
  CHECK(kirari::media::to_us(kMax / 2, AVRational{1, 1}, 0) == kirari::kNoPts);
  CHECK(kirari::media::to_us(-kMax / 2, AVRational{1, 1}, 0) == kirari::kNoPts);
  CHECK(kirari::media::to_us(-1'000, AVRational{1, 1'000'000}, kMax) == kirari::kNoPts + 1);
  CHECK(kirari::media::to_us(2, AVRational{1, 1}, 500'000) == 1'500'000);
  CHECK(kirari::media::from_us(kMax, AVRational{1, 1'000'000}, 10) == kMax);
}

TEST_CASE("av_util : le seek compare les distances même entre les extrêmes", "[media]") {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = kirari::kNoPts + 1;
  CHECK(kirari::media::time_distance(kMin, kMax) == std::numeric_limits<std::uint64_t>::max() - 1);
  CHECK(kirari::media::time_distance(kMax, kMin) == kirari::media::time_distance(kMin, kMax));
  CHECK(kirari::media::add_saturated(kMax - 1, 10) == kMax);
  CHECK(kirari::media::duration_to_us(kMax / 2, AVRational{1, 1}) == 0);
}

// A15-01 : NVDEC n'est pas essayé pour une chroma au-delà du 4:2:0, que l'ingestion CUDA ne prend pas.
TEST_CASE("av_util : chroma au-delà du 4:2:0, par le format ou, inconnu, par le profil", "[media]") {
  using kirari::media::chroma_beyond_420;
  AVCodecParameters av{};
  av.codec_id = AV_CODEC_ID_HEVC;
  av.format = AV_PIX_FMT_YUV420P10LE;
  CHECK_FALSE(chroma_beyond_420(av));
  av.format = AV_PIX_FMT_YUV444P10LE;
  CHECK(chroma_beyond_420(av));
  av.format = AV_PIX_FMT_YUV422P10LE;  // HEVC 4:2:2, que NVDEC décode sur RTX 50
  CHECK(chroma_beyond_420(av));
  av.format = AV_PIX_FMT_GRAY8;  // un seul plan : rien à sous-échantillonner
  CHECK_FALSE(chroma_beyond_420(av));
  av.format = AV_PIX_FMT_NONE;
  av.profile = AV_PROFILE_HEVC_REXT;
  CHECK(chroma_beyond_420(av));
  av.profile = AV_PROFILE_HEVC_MAIN_10;
  CHECK_FALSE(chroma_beyond_420(av));
  av.codec_id = AV_CODEC_ID_VP9;
  av.profile = AV_PROFILE_VP9_1;
  CHECK(chroma_beyond_420(av));
  av.codec_id = AV_CODEC_ID_AV1;
  av.profile = AV_PROFILE_AV1_MAIN;
  CHECK_FALSE(chroma_beyond_420(av));
}
