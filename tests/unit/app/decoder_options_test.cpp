// Les options du décodeur par backend : TensorRT lit les surfaces NVDEC en place, sans copie de
// sécurité ni copie en mémoire hôte ; les autres backends gardent la copie de FFmpeg.
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
}

TEST_CASE("decoder_attempts : le décodeur matériel de chaque système avant le logiciel", "[app][decoder]") {
  using media::HwApi;
#if defined(__APPLE__)
  // VideoToolbox d'abord : MoltenVK n'a pas de file vidéo.
  CHECK(decoder_attempts("coreml", true) ==
        std::vector<HwApi>{HwApi::VideoToolbox, HwApi::Vulkan, HwApi::Software});
  CHECK(decoder_attempts("vulkan", false) == std::vector<HwApi>{HwApi::VideoToolbox, HwApi::Software});
#elif defined(__linux__)
  CHECK(decoder_attempts("vulkan", true) == std::vector<HwApi>{HwApi::Vulkan, HwApi::Vaapi, HwApi::Software});
  CHECK(decoder_attempts("vulkan", false) == std::vector<HwApi>{HwApi::Vaapi, HwApi::Software});
#else
  CHECK(decoder_attempts("vulkan", true) == std::vector<HwApi>{HwApi::Vulkan, HwApi::Software});
#endif
  CHECK(decoder_attempts("tensorrt", true) == std::vector<HwApi>{HwApi::Cuda, HwApi::Software});
}

}  // namespace
}  // namespace ajp::app::play
