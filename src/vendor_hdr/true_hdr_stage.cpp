#include "vendor_hdr/true_hdr_stage.h"

#include <utility>

#include "vendor_hdr/log.h"

namespace ajp::vendor_hdr {

TrueHdrStage::TrueHdrStage(std::expected<std::unique_ptr<TrueHdr>, Refusal> opened) {
  if (opened && *opened) {
    sdk_ = std::move(*opened);
  } else {
    fall_back(opened ? Refusal{RefusalCode::LibraryUnavailable, "aucun TrueHDR ouvert"} : opened.error());
  }
}

HdrPath TrueHdrStage::process(const TrueHdrInput& in, const TrueHdrImage& out, GpuStream& stream) {
  if (!sdk_) return HdrPath::Native;
  if (Done processed = sdk_->process(in, out, stream); !processed) {
    fall_back(processed.error());
    return HdrPath::Native;
  }
  return HdrPath::TrueHdr;
}

std::expected<HdrPath, Refusal> TrueHdrStage::set_params(const TrueHdrParams& params) {
  if (auto valid = validate(params); !valid) return std::unexpected(valid.error());
  if (!sdk_) return HdrPath::Native;  // l'émulation de la route native lit les réglages ailleurs
  if (Done applied = sdk_->set_params(params); !applied) {
    fall_back(applied.error());
    return HdrPath::Native;
  }
  return HdrPath::TrueHdr;
}

std::optional<TrueHdrFallback> TrueHdrStage::take_fallback() {
  return std::exchange(pending_, std::nullopt);
}

void TrueHdrStage::fall_back(Refusal cause) {
  log().warn("TrueHDR abandonné, route native : {} ({})", cause.detail, name(cause.code));
  sdk_.reset();
  pending_ = TrueHdrFallback{.cause = std::move(cause), .notice = kFallbackNotice};
}

}  // namespace ajp::vendor_hdr
