// Les horodatages de av_util.h aux bornes (P2.29) : ni soustraction indéfinie ni faux « sans horodatage ».
#include "media/av_util.h"

#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "core/frame.h"

TEST_CASE("av_util : un horodatage qui déborde reste sans horodatage, le décalage sature", "[media]") {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  // 1 s par tic : av_rescale_q déborde et rend INT64_MIN.
  CHECK(ajp::media::to_us(kMax / 2, AVRational{1, 1}, 0) == ajp::kNoPts);
  CHECK(ajp::media::to_us(-kMax / 2, AVRational{1, 1}, 0) == ajp::kNoPts);
  CHECK(ajp::media::to_us(-1'000, AVRational{1, 1'000'000}, kMax) == ajp::kNoPts + 1);
  CHECK(ajp::media::to_us(2, AVRational{1, 1}, 500'000) == 1'500'000);
  CHECK(ajp::media::from_us(kMax, AVRational{1, 1'000'000}, 10) == kMax);
}

// A15-01 : NVDEC n'est pas essayé pour une chroma au-delà du 4:2:0, que l'ingestion CUDA ne prend pas.
TEST_CASE("av_util : chroma au-delà du 4:2:0, par le format ou, inconnu, par le profil", "[media]") {
  using ajp::media::chroma_beyond_420;
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
