#include "vendor_hdr/rtx_video.h"

#include <format>
#include <utility>

#include "vendor_hdr/dynamic_library.h"
#include "vendor_hdr/log.h"
#include "vendor_hdr/rtx_video_abi.h"
#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {
namespace {

namespace fs = std::filesystem;

// Qualité de Super Resolution transmise par politesse : VSR est désactivé à la création.
constexpr int kUnusedVsrQuality = 2;

class RtxVideoEffect final : public VendorEffect {
 public:
  RtxVideoEffect(DynamicLibrary library, rtx_video::CudaEvaluateFn* evaluate,
                 rtx_video::CudaShutdownFn* shutdown)
      : library_(std::move(library)), evaluate_(evaluate), shutdown_(shutdown) {}

  RtxVideoEffect(const RtxVideoEffect&) = delete;
  RtxVideoEffect& operator=(const RtxVideoEffect&) = delete;

  // Le SDK s'arrête avant que la DLL ne se ferme (library_ est détruit en dernier).
  ~RtxVideoEffect() override { shutdown_(); }

  TrueHdrPixel input_pixel() const override { return TrueHdrPixel::Rgb10A2; }

  // Les réglages accompagnent chaque évaluation ; rien à transmettre ici. ThdrSetting n'a pas
  // de débandage à couper : false est refusé plutôt qu'ignoré.
  Done apply(const TrueHdrParams& params) override {
    if (!params.debanding) {
      return refuse(RefusalCode::FieldOutOfRange,
                    "truehdr.debanding = false : TrueHDR RTX Video ne sait pas couper son débandage");
    }
    thdr_ = rtx_video::ThdrSetting{.contrast = params.contrast,
                                   .saturation = params.saturation,
                                   .middle_gray = params.middle_gray,
                                   .max_luminance = params.peak_nits};
    return {};
  }

  Done run_on_legacy(const TrueHdrImage& in, const TrueHdrImage& out) override {
    // L'API ne prend pas de pas de ligne : les tampons doivent être serrés.
    for (const TrueHdrImage* image : {&in, &out}) {
      if (image->pitch != static_cast<std::size_t>(image->width) * 4) {
        return refuse(RefusalCode::InputMismatch, "RTX Video attend des lignes serrées (pas = 4 × largeur)");
      }
    }
    const rtx_video::ApiRect rect{.left = 0, .top = 0, .right = in.width, .bottom = in.height};
    if (!evaluate_(static_cast<rtx_video::CUdeviceptr>(in.device),
                   static_cast<rtx_video::CUdeviceptr>(out.device), rect, rect, &vsr_, &thdr_)) {
      return refuse(RefusalCode::VendorCallFailed, "TrueHDR RTX Video : évaluation refusée par le SDK");
    }
    return {};
  }

 private:
  DynamicLibrary library_;
  rtx_video::CudaEvaluateFn* evaluate_;
  rtx_video::CudaShutdownFn* shutdown_;
  rtx_video::VsrSetting vsr_{.quality_level = kUnusedVsrQuality};
  rtx_video::ThdrSetting thdr_{};
};

}  // namespace

namespace detail {

std::expected<std::unique_ptr<TrueHdr>, Refusal> open_rtx_video_at(const fs::path& library,
                                                                   const TrueHdrConfig& config) {
  if (Done valid = check(config); !valid) return std::unexpected(valid.error());
  // Sans contexte, le SDK créerait le sien : son flux 0 ne serait pas celui que les événements
  // de GpuStream ordonnent, et les tampons de gpu lui seraient étrangers.
  if (config.cuda_context == 0) {
    return refuse(RefusalCode::InputMismatch, "TrueHDR RTX Video : contexte CUDA du thread gpu absent");
  }
  auto opened = DynamicLibrary::open(library);
  if (!opened) return std::unexpected(opened.error());
  auto create = opened->symbol<rtx_video::CudaCreateFn>(rtx_video::kCreate);
  auto evaluate = opened->symbol<rtx_video::CudaEvaluateFn>(rtx_video::kEvaluate);
  auto shutdown = opened->symbol<rtx_video::CudaShutdownFn>(rtx_video::kShutdown);
  if (!create) return std::unexpected(create.error());
  if (!evaluate) return std::unexpected(evaluate.error());
  if (!shutdown) return std::unexpected(shutdown.error());

  // Flux 0 hérité : process() encadre chaque appel d'événements (aji_trt.cpp:2336-2337).
  auto context =
      reinterpret_cast<rtx_video::CUcontext>(config.cuda_context);  // NOLINT(performance-no-int-to-ptr)
  if (!(*create)(&context, nullptr, true, false)) {
    return refuse(RefusalCode::VendorCallFailed,
                  "TrueHDR RTX Video : création refusée (GPU, pilote ou SDK incompatible)");
  }
  auto effect = std::make_unique<RtxVideoEffect>(std::move(*opened), *evaluate, *shutdown);
  auto made = make_true_hdr(config, std::move(effect));
  if (made) make_logger()->info("TrueHDR RTX Video prêt en {}x{}", config.width, config.height);
  return made;
}

}  // namespace detail

std::expected<std::unique_ptr<TrueHdr>, Refusal> open_rtx_video(const InstalledSdk& sdk,
                                                                const TrueHdrConfig& config) {
  if (sdk.kind() != SdkKind::RtxVideo) {
    return refuse(RefusalCode::LibraryUnavailable, "le SDK fourni n'est pas le RTX Video SDK");
  }
  return detail::open_rtx_video_at(sdk.root() / layout(SdkKind::RtxVideo).main_library, config);
}

}  // namespace ajp::vendor_hdr
