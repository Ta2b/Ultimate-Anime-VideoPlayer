#include "vendor_hdr/open_true_hdr.h"

#include "vendor_hdr/maxine_vfx.h"
#include "vendor_hdr/rtx_video.h"

namespace ajp::vendor_hdr {

std::expected<std::unique_ptr<TrueHdr>, Refusal> open_true_hdr(const InstalledSdk& sdk,
                                                               const TrueHdrConfig& config) {
  switch (sdk.kind()) {
    case SdkKind::RtxVideo:
      return open_rtx_video(sdk, config);
    case SdkKind::MaxineVfx:
      return open_maxine(sdk, config);
  }
  return refuse(RefusalCode::LibraryUnavailable, "SDK TrueHDR inconnu");
}

}  // namespace ajp::vendor_hdr
