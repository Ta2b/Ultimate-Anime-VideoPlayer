// Les options du décodeur par backend : TensorRT lit les surfaces NVDEC en place, sans copie de
// sécurité ni copie en mémoire hôte ; les autres backends gardent la copie de FFmpeg.
#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain_setup.h"
#include "core/gpu_device.h"
#include "media/hw_api.h"

namespace ajp::app::play {
namespace {

TEST_CASE("decoder_options : surfaces directes sous TensorRT seulement", "[app][decoder]") {
  GpuDevice render;
  render.uuid[0] = 7;
  const media::DecoderOptions trt = decoder_options("tensorrt", render, nullptr);
  // Le logiciel prend ce que NVDEC refuse : le backend envoie ses images (host_upload.h).
  CHECK(trt.attempts == std::vector<media::HwApi>{media::HwApi::Cuda, media::HwApi::Software});
  CHECK(trt.render_device == render.uuid);
  CHECK_FALSE(trt.download);
  CHECK(trt.direct_surfaces == kHeldDecodedFrames);
  CHECK(trt.direct_surfaces <= media::kMaxExtraHwFrames);

  const media::DecoderOptions vulkan = decoder_options("vulkan", render, nullptr);
  CHECK(vulkan.download);
  CHECK(vulkan.direct_surfaces == 0);
  CHECK(vulkan.attempts.back() == media::HwApi::Software);
#if defined(__linux__)
  CHECK(std::ranges::find(vulkan.attempts, media::HwApi::Vaapi) != vulkan.attempts.end());
  GpuDevice nvidia;
  nvidia.pci_vendor = pci::kNvidia;
  const media::DecoderOptions on_nvidia = decoder_options("vulkan", nvidia, nullptr);
  CHECK(std::ranges::find(on_nvidia.attempts, media::HwApi::Vaapi) == on_nvidia.attempts.end());
#endif
#if defined(_WIN32)
  CHECK(std::ranges::find(vulkan.attempts, media::HwApi::D3d11va) != vulkan.attempts.end());
#endif
#if defined(__APPLE__)
  CHECK(vulkan.attempts.front() == media::HwApi::VideoToolbox);
  CHECK(decoder_options("coreml", render, nullptr).attempts.front() == media::HwApi::VideoToolbox);
#else
  CHECK(std::ranges::find(vulkan.attempts, media::HwApi::VideoToolbox) == vulkan.attempts.end());
#endif
}

}  // namespace
}  // namespace ajp::app::play
