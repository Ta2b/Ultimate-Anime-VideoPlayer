// Accès Vulkan à une image décodée : chaque soumission suit l'état noté dans l'AVVkFrame
// (valeur du sémaphore, disposition, file), sous son verrou, comme hwcontext_vulkan.h le
// demande. Une image FFmpeg sur lavapipe, sans décodage : le protocole ne dépend pas du codec.
#include <chrono>
#include <cstdint>
#include <future>
#include <optional>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixdesc.h>
}

#include "media/frame_data.h"
#include "media/hw_device.h"

using namespace ajp::media;
using ajp::FrameMemory;
using ajp::RefusalCode;
using ajp::VulkanImageState;
using namespace std::chrono_literals;

namespace {

// Une image du pool Vulkan de FFmpeg, d'une seule VkImage, ou rien si l'appareil la refuse.
struct VulkanFrame {
  BufferPtr frames;
  FramePtr frame;
};

std::optional<VulkanFrame> vulkan_frame(AVBufferRef& device, AVPixelFormat sw_format) {
  BufferPtr frames(av_hwframe_ctx_alloc(&device));
  if (!frames) return std::nullopt;
  auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
  context->format = AV_PIX_FMT_VULKAN;
  context->sw_format = sw_format;
  context->width = 64;
  context->height = 64;
  if (av_hwframe_ctx_init(frames.get()) < 0) return std::nullopt;
  FramePtr frame(av_frame_alloc());
  if (!frame || av_hwframe_get_buffer(frames.get(), frame.get(), 0) < 0) return std::nullopt;
  if (reinterpret_cast<const AVVkFrame*>(frame->data[0])->img[1] != VK_NULL_HANDLE) return std::nullopt;
  return VulkanFrame{std::move(frames), std::move(frame)};
}

// Ce que ferait la soumission de la lecture : signaler la valeur promise, ici depuis l'hôte.
// Sans ce signal, FFmpeg attendrait la valeur pour toujours en libérant son pool.
bool signal_from_host(const AVHWFramesContext& frames, VkSemaphore semaphore, std::uint64_t value) {
  const auto& device = *static_cast<const AVVulkanDeviceContext*>(frames.device_ctx->hwctx);
  const auto device_proc =
      reinterpret_cast<PFN_vkGetDeviceProcAddr>(device.get_proc_addr(device.inst, "vkGetDeviceProcAddr"));
  if (device_proc == nullptr) return false;
  const auto signal =
      reinterpret_cast<PFN_vkSignalSemaphore>(device_proc(device.act_dev, "vkSignalSemaphore"));
  if (signal == nullptr) return false;
  const VkSemaphoreSignalInfo info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = semaphore, .value = value};
  return signal(device.act_dev, &info) == VK_SUCCESS;
}

}  // namespace

TEST_CASE("frame_data : une image hôte n'ouvre aucun accès Vulkan", "[media][frame_data]") {
  FramePtr frame(av_frame_alloc());
  REQUIRE(frame);
  frame->format = AV_PIX_FMT_NV12;
  frame->width = 64;
  frame->height = 64;
  REQUIRE(av_frame_get_buffer(frame.get(), 0) == 0);
  auto decoded = to_decoded_frame(std::move(frame), 0, 40'000);
  REQUIRE(decoded);
  REQUIRE(decoded->data->memory() == FrameMemory::Host);
  auto access = decoded->data->begin_vulkan_access();
  REQUIRE_FALSE(access);
  CHECK(access.error().code == RefusalCode::FormatUnsupported);
  decoded->data->end_vulkan_access({}, true);  // sans accès ouvert : rien à rendre
}

TEST_CASE("frame_data : l'accès Vulkan suit l'état de l'AVVkFrame sous son verrou",
          "[media][frame_data][vulkan]") {
  auto device = open_hw_device(AV_CODEC_ID_HEVC, HwApi::Vulkan, DecoderOptions{});
  if (!device) SKIP("aucun appareil Vulkan : " << device.error().detail);
  std::optional<VulkanFrame> pooled = vulkan_frame(*device->device, AV_PIX_FMT_NV12);
  if (!pooled) pooled = vulkan_frame(*device->device, AV_PIX_FMT_RGBA);
  if (!pooled) SKIP("aucune image Vulkan d'une seule VkImage sur cet appareil");

  auto* frames = reinterpret_cast<AVHWFramesContext*>(pooled->frames->data);
  auto* context = static_cast<AVVulkanFramesContext*>(frames->hwctx);
  AVVkFrame* image = reinterpret_cast<AVVkFrame*>(pooled->frame->data[0]);
  const std::uint64_t before = image->sem_value[0];
  // Ce que le décodeur aurait laissé : une disposition de décodage, notée dans l'image.
  image->layout[0] = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

  auto decoded = to_decoded_frame(std::move(pooled->frame), 0, 40'000);
  REQUIRE(decoded);
  const ajp::FrameData& data = *decoded->data;
  REQUIRE(data.memory() == FrameMemory::Vulkan);

  auto access = data.begin_vulkan_access();
  REQUIRE(access);
  CHECK(access->semaphore == reinterpret_cast<std::uintptr_t>(image->sem[0]));
  CHECK(access->wait_value == before);
  CHECK(access->signal_value == before + 1);
  CHECK(access->state.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  CHECK(access->state.queue_family == image->queue_family[0]);

  // Le verrou de l'image reste pris jusqu'à end_vulkan_access : le décodeur attend.
  auto decoder = std::async(std::launch::async, [&] {
    context->lock_frame(frames, image);
    const std::uint64_t seen = image->sem_value[0];
    context->unlock_frame(frames, image);
    return seen;
  });
  CHECK(decoder.wait_for(50ms) == std::future_status::timeout);

  data.end_vulkan_access({.layout = VK_IMAGE_LAYOUT_GENERAL, .queue_family = access->state.queue_family},
                         true);
  const bool signalled = signal_from_host(*frames, image->sem[0], access->signal_value);
  if (!signalled) image->sem_value[0] = before;  // sinon la libération du pool attendrait sans fin
  REQUIRE(signalled);
  CHECK(decoder.get() == before + 1);  // le décodeur voit la valeur que la lecture signale
  CHECK(image->layout[0] == VK_IMAGE_LAYOUT_GENERAL);

  // Une soumission abandonnée ne change rien : aucune valeur promise jamais signalée.
  auto abandoned = data.begin_vulkan_access();
  REQUIRE(abandoned);
  CHECK(abandoned->wait_value == before + 1);
  CHECK(abandoned->state.layout == VK_IMAGE_LAYOUT_GENERAL);
  data.end_vulkan_access(VulkanImageState{.layout = VK_IMAGE_LAYOUT_UNDEFINED}, false);
  CHECK(image->sem_value[0] == before + 1);
  CHECK(image->layout[0] == VK_IMAGE_LAYOUT_GENERAL);
}

namespace {

// Une image hôte de 64x32 : luma en dégradé, Cb en bandes de deux colonnes (0 puis plein), Cr constant.
FramePtr host_frame(AVPixelFormat format, int bits) {
  FramePtr f(av_frame_alloc());
  f->width = 64;
  f->height = 32;
  f->format = format;
  REQUIRE(av_frame_get_buffer(f.get(), 0) == 0);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  const int top = (1 << bits) - 1;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane == 0 ? 64 : AV_CEIL_RSHIFT(64, desc->log2_chroma_w);
    const int h = plane == 0 ? 32 : AV_CEIL_RSHIFT(32, desc->log2_chroma_h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int value = plane == 0 ? (x * top) / 63 : plane == 1 ? ((x / 2) % 2 == 0 ? 0 : top) : top / 2;
        if (bits > 8) {
          reinterpret_cast<std::uint16_t*>(f->data[plane] + y * f->linesize[plane])[x] =
              static_cast<std::uint16_t>(value);
        } else {
          f->data[plane][y * f->linesize[plane] + x] = static_cast<std::uint8_t>(value);
        }
      }
  }
  return f;
}

}  // namespace

TEST_CASE("frame_data : 4:4:4, 4:2:2 et 12 bits deviennent du 4:2:0 que la chaîne lit (A15-13)",
          "[media][frame_data]") {
  struct Case {
    AVPixelFormat from;
    int bits;
    ajp::PixelFormat to;
  };
  const Case c = GENERATE(Case{AV_PIX_FMT_YUV444P, 8, ajp::PixelFormat::Yuv420p},
                          Case{AV_PIX_FMT_YUV444P10LE, 10, ajp::PixelFormat::Yuv420p10},
                          Case{AV_PIX_FMT_YUV422P10LE, 10, ajp::PixelFormat::Yuv420p10},
                          Case{AV_PIX_FMT_YUV420P12LE, 12, ajp::PixelFormat::Yuv420p10});
  CAPTURE(av_get_pix_fmt_name(c.from));
  for (int round = 0; round < 3; ++round) {  // le pool resert d'une image à l'autre
    FramePtr frame = host_frame(c.from, c.bits);
    frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
    auto decoded = to_decoded_frame(std::move(frame), 1000, 40'000, {});
    REQUIRE(decoded);
    CHECK(decoded->format == c.to);
    CHECK(decoded->pts_us == 1000);
    const AVFrame* out = av_frame_of(*decoded->data);
    REQUIRE(out != nullptr);
    CHECK(out->width == 64);
    CHECK(out->height == 32);
    // 12 bits en 4:2:0 : la chroma n'est pas rééchantillonnée, sa position reste (A15-07).
    CHECK(out->chroma_location ==
          (c.from == AV_PIX_FMT_YUV420P12LE ? AVCHROMA_LOC_TOPLEFT : AVCHROMA_LOC_LEFT));
    const int out_bits = c.to == ajp::PixelFormat::Yuv420p ? 8 : 10;
    const auto luma = [&](int x) {
      return out_bits > 8 ? reinterpret_cast<const std::uint16_t*>(out->data[0] + 5 * out->linesize[0])[x]
                          : out->data[0][5 * out->linesize[0] + x];
    };
    CHECK(luma(0) == 0);
    CHECK(luma(63) == (1 << out_bits) - 1);
  }
}

TEST_CASE("frame_data : la chaîne qui les lit garde le 4:4:4 entier et les 12 bits (A15-07, A49-13)",
          "[media][frame_data]") {
  struct Case {
    AVPixelFormat from;
    int bits;
    ajp::PixelFormat to;
  };
  const Case c = GENERATE(Case{AV_PIX_FMT_YUV444P, 8, ajp::PixelFormat::Yuv444p16},
                          Case{AV_PIX_FMT_YUV444P10LE, 10, ajp::PixelFormat::Yuv444p16},
                          Case{AV_PIX_FMT_YUV444P12LE, 12, ajp::PixelFormat::Yuv444p16},
                          Case{AV_PIX_FMT_YUV420P12LE, 12, ajp::PixelFormat::P016},
                          Case{AV_PIX_FMT_YUV422P10LE, 10, ajp::PixelFormat::Yuv420p10});
  CAPTURE(av_get_pix_fmt_name(c.from));
  FramePtr frame = host_frame(c.from, c.bits);
  frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
  auto decoded = to_decoded_frame(std::move(frame), 1000, 40'000, {}, true);
  REQUIRE(decoded);
  CHECK(decoded->format == c.to);
  const AVFrame* out = av_frame_of(*decoded->data);
  REQUIRE(out != nullptr);
  CHECK(out->width == 64);
  CHECK(out->height == 32);
  // 12 bits en 4:2:0 : la chroma n'est pas rééchantillonnée, sa position reste.
  if (c.to == ajp::PixelFormat::P016) CHECK(out->chroma_location == AVCHROMA_LOC_TOPLEFT);
  if (c.to == ajp::PixelFormat::Yuv420p10) return;
  // Le noir et le blanc du code source se retrouvent aux deux bouts des 16 bits, à un code source près.
  const auto luma = [&](int x) {
    return reinterpret_cast<const std::uint16_t*>(out->data[0] + 5 * out->linesize[0])[x];
  };
  const int step = 1 << (16 - c.bits);
  CHECK(luma(0) == 0);
  CHECK(luma(63) >= 65536 - step);
  // 4:4:4 : le plan de chroma a la taille de l'image.
  if (c.to == ajp::PixelFormat::Yuv444p16) CHECK(out->linesize[1] >= 2 * 64);
}

TEST_CASE("frame_data : sous TensorRT, le 4:2:2 garde sa chroma et sa position en 16 bits (B01-03)",
          "[media][frame_data]") {
  const int bits = GENERATE(8, 10, 12);
  const AVPixelFormat from = bits == 8    ? AV_PIX_FMT_YUV422P
                             : bits == 10 ? AV_PIX_FMT_YUV422P10LE
                                          : AV_PIX_FMT_YUV422P12LE;
  CAPTURE(bits);
  FramePtr frame = host_frame(from, bits);
  frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
  auto decoded = to_decoded_frame(std::move(frame), 0, 40'000, {}, true, true);
  REQUIRE(decoded);
  CHECK(decoded->format == ajp::PixelFormat::Yuv422p16);
  const AVFrame* out = av_frame_of(*decoded->data);
  REQUIRE(out != nullptr);
  CHECK(out->chroma_location == AVCHROMA_LOC_TOPLEFT);  // rien n'est rééchantillonné
  // Chroma à demi-largeur, pleine hauteur : les bandes de Cb restent, ligne après ligne.
  const int step = 1 << (16 - bits);
  for (const int y : {0, 31}) {
    const auto* cb = reinterpret_cast<const std::uint16_t*>(out->data[1] + y * out->linesize[1]);
    CHECK(cb[1] == 0);
    CHECK(cb[2] >= 65536 - step);
    CHECK(cb[3] >= 65536 - step);
    CHECK(cb[4] == 0);
  }
  // Sans host_422 (Vulkan), la même source est réduite en 4:2:0 comme avant.
  auto reduced = to_decoded_frame(host_frame(from, bits), 0, 40'000, {}, true, false);
  REQUIRE(reduced);
  CHECK(reduced->format == (bits > 8 ? ajp::PixelFormat::Yuv420p10 : ajp::PixelFormat::Yuv420p));
}

TEST_CASE("frame_data : un format que la chaîne ne lit pas sans full est converti (B01-03)",
          "[media][frame_data]") {
  // Du yuv444p16le natif sans full_host_formats : ramené au 4:2:0 10 bits, plus jamais rendu tel quel.
  auto decoded = to_decoded_frame(host_frame(AV_PIX_FMT_YUV444P16LE, 16), 0, 40'000, {}, false);
  REQUIRE(decoded);
  CHECK(decoded->format == ajp::PixelFormat::Yuv420p10);
  auto kept = to_decoded_frame(host_frame(AV_PIX_FMT_YUV444P16LE, 16), 0, 40'000, {}, true);
  REQUIRE(kept);
  CHECK(kept->format == ajp::PixelFormat::Yuv444p16);
}

// Une rampe de chroma en largeur : tout filtre normalisé la rend telle quelle, à la position où il pose ses
// points. À gauche, le point j de la chroma réduite vaut celui de la colonne 2j ; au centre, il vaudrait la
// moyenne des colonnes 2j et 2j+1, une demi-marche plus haut.
TEST_CASE("frame_data : la chroma réduite se pose à gauche, comme l'image le dit (A15-07)",
          "[media][frame_data]") {
  const AVPixelFormat from = GENERATE(AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV422P10LE);
  CAPTURE(av_get_pix_fmt_name(from));
  FramePtr frame = host_frame(from, 10);
  const bool wide = from == AV_PIX_FMT_YUV444P10LE;
  constexpr int kStep = 8;  // codes par colonne de luma
  for (int y = 0; y < 32; ++y) {
    auto* row = reinterpret_cast<std::uint16_t*>(frame->data[1] + y * frame->linesize[1]);
    for (int x = 0; x < (wide ? 64 : 32); ++x)
      row[x] = static_cast<std::uint16_t>(200 + kStep * (wide ? x : 2 * x));
  }
  auto decoded = to_decoded_frame(std::move(frame), 0, 40'000, {});
  REQUIRE(decoded);
  REQUIRE(decoded->format == ajp::PixelFormat::Yuv420p10);
  CHECK(decoded->color.siting == ajp::ChromaSiting::Left);
  const AVFrame* out = av_frame_of(*decoded->data);
  const auto* row = reinterpret_cast<const std::uint16_t*>(out->data[1] + 8 * out->linesize[1]);
  for (const int j : {8, 16, 24}) {
    CAPTURE(j);
    CHECK(std::abs(static_cast<int>(row[j]) - (200 + kStep * 2 * j)) <= 1);
  }
}

TEST_CASE("frame_data : yuvj444p (plage pleine) sort en plage limitée", "[media][frame_data]") {
  FramePtr frame = host_frame(AV_PIX_FMT_YUVJ444P, 8);
  auto decoded = to_decoded_frame(std::move(frame), 0, 40'000, {});
  REQUIRE(decoded);
  CHECK(decoded->format == ajp::PixelFormat::Yuv420p);
  CHECK(decoded->color.range == ajp::ColorRange::Limited);
  const AVFrame* out = av_frame_of(*decoded->data);
  CHECK(out->data[0][5 * out->linesize[0]] == 16);
  CHECK(out->data[0][5 * out->linesize[0] + 63] == 235);
}
