// AVFrame vers ajp::DecodedFrame : FrameData tient une référence AVFrame, la
// dernière copie relâchée rend le tampon à son pool. Interne au module.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "core/frame.h"
#include "core/types.h"
#include "media/av_util.h"

namespace ajp::media {

// Événement GPU à attendre avant de lire l'image (CUevent du flux de décodage),
// avec ce qui le garde en vie. handle 0 : rien à attendre.
struct ReadyEvent {
  std::uintptr_t handle = 0;
  std::shared_ptr<const void> owner;
};

// Image décodée, sans identité ni époque : le thread source les pose.
// pts_us et duration_us viennent du décodeur, qui comble les trous.
std::expected<DecodedFrame, Refusal> to_decoded_frame(FramePtr frame, std::int64_t pts_us,
                                                      std::int64_t duration_us, ReadyEvent ready = {});

}  // namespace ajp::media
