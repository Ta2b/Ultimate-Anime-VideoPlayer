// Décodage matériel réel sur un GPU NVIDIA : NVDEC dans le contexte primaire de
// l'appareil choisi par UUID, et Vulkan Video. Les clips viennent des encodeurs
// NVENC de FFmpeg, écrits par le test ; sans GPU NVIDIA (la CI, lavapipe), chaque
// test SKIP avec sa raison. Sur ce PC : ajp-gpu ctest -R 'NVDEC|Vulkan Video'.
//
// Sanitizers sur la RTX :
// - ASan réserve l'espace d'adresses que le pilote CUDA veut pour son UVM : cuInit rend
//   alors l'erreur 2 et ces tests SKIP. Il faut ASAN_OPTIONS=protect_shadow_gap=0.
// - Des bibliothèques NVIDIA sont déchargées avant la sortie : l'ICD Vulkan (libnvidia-
//   glcore…) par le chargeur Vulkan à la dernière VkInstance détruite, libnvidia-encode et
//   libnvcuvid par FFmpeg à la fermeture de l'encodeur ou du décodeur. Leurs fuites internes
//   deviennent « <unknown module> », qu'aucune ligne de tools/ci/lsan.supp n'atteint (vu le
//   19 octobre : 3231 octets en 69 allocations sous av_hwdevice_ctx_create Vulkan et sous
//   avcodec_open2 de NVENC). Sur la RTX, ces tests et les tests [vulkan] (leur nom dit
//   NVDEC ou Vulkan) tournent donc avec detect_leaks=0, le reste de media avec les fuites :
//     ajp-gpu env ASAN_OPTIONS=protect_shadow_gap=0 ctest --preset linux-prefix-asan
//       -R '^media: ' -E 'NVDEC|Vulkan'
//     ajp-gpu env ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0 ctest --preset
//       linux-prefix-asan -R '^media: .*(NVDEC|Vulkan)'
//   Les fuites des tests [vulkan] se vérifient sur lavapipe (with-lavapipe, la CI).
#include <bit>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// hwcontext_cuda.h n'a besoin que de CUcontext et CUstream, comme hw_device.cpp.
#define CUDA_VERSION 7050
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include "clips.h"
#include "media/av_util.h"
#include "media/cuda_driver.h"
#include "media/demuxer.h"
#include "media/frame_data.h"
#include "media/hw_device.h"
#include "media/video_decoder.h"

using namespace ajp::media;
using ajp::DecodedFrame;
using ajp::FrameMemory;
using ajp::PixelFormat;

namespace {

constexpr int kFrames = 36;  // flat_luma(35) = 242 : la luma uniforme tient sur 8 bits

struct Cuda {
  std::shared_ptr<CudaDriver> driver;
  CudaUuid uuid{};  // appareil d'ordinal 0
};

// Pilote et premier appareil CUDA, ou SKIP.
Cuda require_cuda() {
  auto driver = CudaDriver::load();
  if (!driver) SKIP("pas de pilote CUDA : " << driver.error().detail);
  auto uuids = (*driver)->uuids();
  REQUIRE(uuids.has_value());
  if (uuids->empty()) SKIP("aucun appareil CUDA");
  return Cuda{.driver = std::move(*driver), .uuid = uuids->front()};
}

struct Decoded {
  std::vector<DecodedFrame> frames;
  HwApi api = HwApi::Software;
  std::vector<Attempt> rejected;
  int surfaces = 0;
};

// Décode tout le clip ; le décodeur est détruit avant le retour, les images restent.
Decoded decode(const std::filesystem::path& clip, const DecoderOptions& options) {
  auto demuxer = Demuxer::open(clip);
  REQUIRE(demuxer.has_value());
  auto decoder = VideoDecoder::open(demuxer->tracks()[0], options);
  INFO((decoder ? std::string() : decoder.error().detail));
  REQUIRE(decoder.has_value());
  Decoded out;
  while (!decoder->drained()) {
    auto frame = decoder->receive();
    INFO((frame ? std::string() : frame.error().detail));
    REQUIRE(frame.has_value());
    if (*frame) {
      out.frames.push_back(std::move(**frame));
      continue;
    }
    if (decoder->drained()) break;
    auto packet = demuxer->read();
    REQUIRE(packet.has_value());
    if (!*packet) {
      REQUIRE(decoder->send_end());
    } else if ((*packet)->track == 0) {
      REQUIRE(decoder->send(**packet));
    }
  }
  out.api = decoder->api();
  out.rejected.assign(decoder->rejected().begin(), decoder->rejected().end());
  out.surfaces = decoder->hw_surfaces();
  return out;
}

// Luma au centre, relue en mémoire hôte par FFmpeg (copie CUDA ou Vulkan).
int center_luma(const DecodedFrame& frame) {
  const AVFrame* gpu = av_frame_of(*frame.data);
  REQUIRE(gpu != nullptr);
  FramePtr host(av_frame_alloc());
  REQUIRE(host);
  REQUIRE(av_hwframe_transfer_data(host.get(), gpu, 0) >= 0);
  const int x = frame.width / 2;
  const int y = frame.height / 2;
  return host->data[0][static_cast<std::ptrdiff_t>(y) * host->linesize[0] + x];
}

// Chaque image a sa luma uniforme et son pts, dans l'ordre.
void check_content(const std::vector<DecodedFrame>& frames) {
  REQUIRE(frames.size() == static_cast<std::size_t>(kFrames));
  for (std::size_t n = 0; n < frames.size(); ++n) {
    INFO("image " << n);
    CHECK(frames[n].pts_us == static_cast<std::int64_t>(n) * test::kFrameUs);
    CHECK(test::frame_of_flat_luma(center_luma(frames[n])) == static_cast<int>(n));
  }
}

}  // namespace

TEST_CASE("NVDEC : contexte primaire de l'appareil choisi par UUID, flux de copie non bloquant",
          "[media][hw][nvidia]") {
  const Cuda cuda = require_cuda();
  auto hw = open_hw_device(AV_CODEC_ID_H264, HwApi::Cuda, {.render_device = cuda.uuid});
  INFO((hw ? std::string() : hw.error().detail));
  REQUIRE(hw.has_value());
  REQUIRE(hw->device);
  REQUIRE(hw->stream);
  const auto& context = *reinterpret_cast<const AVHWDeviceContext*>(hw->device->data);
  REQUIRE(context.type == AV_HWDEVICE_TYPE_CUDA);
  const auto& device = *static_cast<const AVCUDADeviceContext*>(context.hwctx);
  auto primary = cuda.driver->primary_context(0);
  REQUIRE(primary.has_value());
  CHECK(std::bit_cast<std::uintptr_t>(device.cuda_ctx) == *primary);  // AV_CUDA_USE_PRIMARY_CONTEXT
  CHECK(device.stream != nullptr);                                    // le nôtre, pas le flux 0

  CudaUuid other = cuda.uuid;
  other[15] = static_cast<std::uint8_t>(other[15] ^ 0xFFU);
  auto mismatch = open_hw_device(AV_CODEC_ID_H264, HwApi::Cuda, {.render_device = other});
  REQUIRE_FALSE(mismatch);
  CHECK(mismatch.error().detail.find("UUID") != std::string::npos);
}

TEST_CASE("NVDEC : H.264, HEVC et AV1 en mémoire CUDA, pas de ligne et plans cohérents",
          "[media][decoder][nvidia]") {
  const std::string encoder = GENERATE("h264_nvenc", "hevc_nvenc", "av1_nvenc");
  INFO(encoder);
  const Cuda cuda = require_cuda();
  const auto clip = test::require_gpu_clip(test::gpu_clip(encoder));
  const Decoded decoded = decode(clip, {.attempts = {HwApi::Cuda}, .render_device = cuda.uuid});
  CHECK(decoded.api == HwApi::Cuda);
  CHECK(decoded.rejected.empty());
  REQUIRE(decoded.frames.size() == static_cast<std::size_t>(kFrames));
  for (const DecodedFrame& frame : decoded.frames) {
    const AVFrame* av = av_frame_of(*frame.data);
    REQUIRE(av != nullptr);
    CHECK(frame.data->memory() == FrameMemory::Cuda);
    CHECK(frame.format == PixelFormat::Nv12);
    CHECK(frame.width == 256);
    CHECK(frame.height == 144);  // recadrée : les macroblocs couvrent 160 lignes
    REQUIRE(frame.data->plane_count() == 2);
    CHECK(frame.data->host_plane(0).empty());
    CHECK(frame.data->native_plane(0) == reinterpret_cast<std::uintptr_t>(av->data[0]));
    CHECK(frame.data->native_plane(1) == reinterpret_cast<std::uintptr_t>(av->data[1]));
    CHECK(frame.data->native_pitch(0) == static_cast<std::size_t>(av->linesize[0]));
    CHECK(frame.data->native_pitch(0) >= 256);
    CHECK(frame.data->native_pitch(1) == frame.data->native_pitch(0));  // NV12 : chroma entrelacée
    CHECK(frame.data->native_pitch(2) == 0);
    // La chroma suit la luma dans la même allocation, après au moins les lignes visibles.
    CHECK(frame.data->native_plane(1) >= frame.data->native_plane(0) + frame.data->native_pitch(0) * 144);
    CHECK(frame.data->ready_event() != 0);  // CUevent de fin de copie
    CHECK(frame.data->ready_value() == 0);
  }
  check_content(decoded.frames);
}

TEST_CASE("NVDEC : extra_hw_frames agrandit le pool, les images tenues survivent au décodeur",
          "[media][decoder][nvidia]") {
  const Cuda cuda = require_cuda();
  const auto clip = test::require_gpu_clip(test::gpu_clip("h264_nvenc"));
  const Decoded none =
      decode(clip, {.attempts = {HwApi::Cuda}, .render_device = cuda.uuid, .extra_hw_frames = 0});
  const Decoded more =
      decode(clip, {.attempts = {HwApi::Cuda}, .render_device = cuda.uuid, .extra_hw_frames = 8});
  CHECK(none.surfaces > 0);
  CHECK(more.surfaces == none.surfaces + 8);
  // Copie de sécurité (pas AV_HWACCEL_FLAG_UNSAFE_OUTPUT) : chaque image sort de la surface
  // de NVDEC par une copie dans une image du pool CUDA, alloué à la demande (nvdec.c). Les
  // 36 images, tenues ensemble, dépassent les surfaces de décodage (32 au plus) ; aucune
  // n'est réécrite, même une fois le décodeur détruit.
  REQUIRE(more.frames.size() > 32);
  REQUIRE(more.frames.size() > static_cast<std::size_t>(more.surfaces));
  check_content(more.frames);
}

TEST_CASE("Vulkan Video : H.264, HEVC et AV1 décodés sur la file vidéo du GPU", "[media][decoder][nvidia]") {
  const std::string encoder = GENERATE("h264_nvenc", "hevc_nvenc", "av1_nvenc");
  INFO(encoder);
  const auto clip = test::require_gpu_clip(test::gpu_clip(encoder));
  const Decoded decoded = decode(clip, {.attempts = {HwApi::Vulkan, HwApi::Software}});
  if (decoded.api != HwApi::Vulkan) {
    REQUIRE_FALSE(decoded.rejected.empty());
    SKIP("pas de décodage Vulkan Video ici : " << decoded.rejected.front().why.detail);
  }
  CHECK(decoded.rejected.empty());
  REQUIRE(decoded.frames.size() == static_cast<std::size_t>(kFrames));
  for (const DecodedFrame& frame : decoded.frames) {
    CHECK(frame.data->memory() == FrameMemory::Vulkan);
    CHECK(frame.format == PixelFormat::Nv12);
    CHECK(frame.height == 144);
    REQUIRE(frame.data->plane_count() == 1);  // une VkImage multiplan
    CHECK(frame.data->native_plane(0) != 0);
    CHECK(frame.data->native_pitch(0) == 0);  // disposition opaque
    CHECK(frame.data->ready_event() != 0);    // sémaphore timeline de l'AVVkFrame
    CHECK(frame.data->ready_value() > 0);
    // Protocole de lecture de l'AVVkFrame : même sémaphore, valeur au moins celle de la
    // livraison. Sans soumission, l'image garde sa valeur (le verrou est rendu avant la
    // relecture de check_content, qui le prend aussi).
    auto access = frame.data->begin_vulkan_access();
    REQUIRE(access.has_value());
    CHECK(access->semaphore == frame.data->ready_event());
    CHECK(access->wait_value >= frame.data->ready_value());
    CHECK(access->signal_value == access->wait_value + 1);
    frame.data->end_vulkan_access(access->state, false);
    auto again = frame.data->begin_vulkan_access();
    REQUIRE(again.has_value());
    CHECK(again->wait_value == access->wait_value);
    frame.data->end_vulkan_access(again->state, false);
  }
  check_content(decoded.frames);
}
